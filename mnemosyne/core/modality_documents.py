"""Document understanding, done locally.

A document already *is* text, so the right way to make it recallable is to
extract that text and store it with its location, not to ask a vision model to
look at it. This backend does that with no network and no model:

- ``.txt`` / ``.md``: decoded as UTF-8, split on blank lines, char spans.
- ``.docx``: paragraphs from ``word/document.xml`` (stdlib zip + XML), char spans.
- ``.pptx``: one unit per slide, page spans numbered by slide.
- ``.epub``: one unit per spine document, page spans numbered by chapter.
- ``.pdf``: one unit per page via ``pypdfium2`` (the ``media`` extra), page
  spans. A page with no text layer, i.e. a scan, is rendered and handed to
  whatever *image* backend is registered, so it gets described instead of
  silently dropped.

Units are packed in reading order into moments of about
``TARGET_MOMENT_CHARS`` so a recalled row is a readable passage, and packing
stops at ``max_moments``. When it stops early the result says how far it got;
it never pretends to have covered the whole file.

Still behind ``modality_enabled``: with the gate off, no bytes are read.
"""

from __future__ import annotations

import io
import logging
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple
from xml.etree import ElementTree

from mnemosyne.core.modality_backends import DescribedMoment, DescribeRequest, DescribeResult

logger = logging.getLogger(__name__)

NAME = "local_document"
#: A moment is a readable passage, not a whole book and not a sentence.
TARGET_MOMENT_CHARS = 1500
#: Rendering scale for scanned PDF pages sent to the image backend (72 dpi * 2).
SCAN_RENDER_SCALE = 2.0
#: Scanned pages sent to the image backend per document, to bound cost.
MAX_SCANNED_PAGES = 8


@dataclass
class _Unit:
    """One addressable piece of a document, in reading order."""

    text: str
    page: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_MIME_FORMAT = {
    "text/plain": "text",
    "text/markdown": "text",
    "text/x-markdown": "text",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/epub+zip": "epub",
}
_EXT_FORMAT = {
    ".txt": "text", ".md": "text", ".markdown": "text",
    ".pdf": "pdf", ".docx": "docx", ".pptx": "pptx", ".epub": "epub",
}


def document_format(uri: Optional[str], mime: Optional[str], raw: bytes) -> Optional[str]:
    """Explicit mime, then extension, then magic bytes."""
    if mime and str(mime).split(";", 1)[0].strip().lower() in _MIME_FORMAT:
        return _MIME_FORMAT[str(mime).split(";", 1)[0].strip().lower()]
    tail = str(uri or "").rsplit("/", 1)[-1].split("?", 1)[0].lower()
    for ext, fmt in _EXT_FORMAT.items():
        if tail.endswith(ext):
            return fmt
    if raw.startswith(b"%PDF-"):
        return "pdf"
    if raw.startswith(b"PK\x03\x04"):
        try:
            names = set(zipfile.ZipFile(io.BytesIO(raw)).namelist())
        except zipfile.BadZipFile:
            return None
        if "word/document.xml" in names:
            return "docx"
        if any(n.startswith("ppt/slides/") for n in names):
            return "pptx"
        if "META-INF/container.xml" in names:
            return "epub"
    return None


# ---------------------------------------------------------------------------
# Extractors. Each returns units in reading order, or raises on a corrupt file.
# ---------------------------------------------------------------------------

def _text_units(raw: bytes) -> List[_Unit]:
    text = raw.decode("utf-8", errors="replace")
    units = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\s*\Z)", text, re.S):
        units.append(_Unit(text=match.group(0), char_start=match.start(), char_end=match.end()))
    return units


def _xml_text(data: bytes, tag_suffix: str, para_suffix: Optional[str] = None) -> List[str]:
    """Text of every element whose tag ends with ``tag_suffix``, joined per
    paragraph element when ``para_suffix`` is given."""
    root = ElementTree.fromstring(data)
    if para_suffix is None:
        return ["".join(el.text or "" for el in root.iter() if el.tag.endswith(tag_suffix))]
    out = []
    for para in root.iter():
        if para.tag.endswith(para_suffix):
            out.append("".join(el.text or "" for el in para.iter() if el.tag.endswith(tag_suffix)))
    return out


def _docx_units(raw: bytes) -> List[_Unit]:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        paragraphs = _xml_text(zf.read("word/document.xml"), "}t", "}p")
    units, offset = [], 0
    for para in paragraphs:
        para = para.strip()
        if para:
            units.append(_Unit(text=para, char_start=offset, char_end=offset + len(para)))
            offset += len(para) + 1
    return units


def _pptx_units(raw: bytes) -> List[_Unit]:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        slides = sorted(
            (n for n in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[-1]).group(1)),
        )
        units = []
        for number, name in enumerate(slides, start=1):
            lines = [t.strip() for t in _xml_text(zf.read(name), "}t", "}p") if t.strip()]
            if lines:
                units.append(_Unit(text="\n".join(lines), page=number))
    return units


def _html_to_text(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>|</(p|div|h[1-6]|li|tr)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    import html as _html

    text = _html.unescape(text)
    return re.sub(r"[ \t\r\f\v]+", " ", re.sub(r"\n\s*\n+", "\n\n", text)).strip()


def _epub_units(raw: bytes) -> List[_Unit]:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        container = ElementTree.fromstring(zf.read("META-INF/container.xml"))
        opf_path = next(el.get("full-path") for el in container.iter() if el.tag.endswith("rootfile"))
        opf = ElementTree.fromstring(zf.read(opf_path))
        base = posixpath.dirname(opf_path)
        manifest = {el.get("id"): el.get("href") for el in opf.iter() if el.tag.endswith("}item")}
        spine = [el.get("idref") for el in opf.iter() if el.tag.endswith("}itemref")]
        units = []
        for number, idref in enumerate(spine, start=1):
            href = manifest.get(idref)
            if not href:
                continue
            path = posixpath.normpath(posixpath.join(base, href)) if base else href
            try:
                text = _html_to_text(zf.read(path))
            except KeyError:
                continue
            if text:
                units.append(_Unit(text=text, page=number))
    return units


def _pdf_units(raw: bytes, request: DescribeRequest, warnings: List[str]) -> Optional[List[_Unit]]:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        warnings.append("PDF text extraction needs the media extra: pip install 'mnemosyne-memory[media]'")
        return None

    units: List[_Unit] = []
    scanned: List[int] = []
    pdf = pdfium.PdfDocument(raw)
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            try:
                textpage = page.get_textpage()
                try:
                    text = (textpage.get_text_range() or "").replace("\r\n", "\n").strip()
                finally:
                    textpage.close()
                if text:
                    units.append(_Unit(text=text, page=index + 1))
                elif len(scanned) < MAX_SCANNED_PAGES:
                    described = _describe_scanned_page(page, index + 1, request)
                    if described:
                        units.append(_Unit(text=described, page=index + 1))
                    scanned.append(index + 1)
            finally:
                page.close()
    finally:
        pdf.close()

    if scanned and not any(u.page in scanned for u in units):
        warnings.append(
            f"{len(scanned)} page(s) have no text layer and no image backend described them"
        )
    return units


def _describe_scanned_page(page, number: int, request: DescribeRequest) -> Optional[str]:
    """Render a text-less page and ask the registered *image* backend about it."""
    from mnemosyne.core.modality_backends import call_modality_describe

    try:
        image = page.render(scale=SCAN_RENDER_SCALE).to_pil()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        png = buffer.getvalue()
    except Exception:
        logger.info("could not render scanned PDF page %d", number, exc_info=True)
        return None
    result = call_modality_describe(DescribeRequest(
        modality="image",
        uri=f"{request.uri}#page={number}.png",
        mime="image/png",
        hint="This is one scanned page of a document. Transcribe its text faithfully, then describe any figures.",
        max_moments=1,
        timeout=request.timeout,
        fetch=lambda: png,
    ))
    if result is None or result.refused:
        return None
    parts = [m.text for m in result.moments if m.text] or ([result.summary] if result.summary else [])
    return "\n".join(parts).strip() or None


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def pack_units(units: List[_Unit], max_moments: int) -> Tuple[List[DescribedMoment], int]:
    """Pack units into passages. Returns ``(moments, units_covered)``."""
    cap = max(1, int(max_moments or 1))
    moments: List[DescribedMoment] = []
    covered = 0
    batch: List[_Unit] = []

    def flush():
        if not batch:
            return
        first, last = batch[0], batch[-1]
        moment = DescribedMoment(kind="page", text="\n\n".join(u.text for u in batch))
        if first.page is not None:
            moment.page_start, moment.page_end = first.page, last.page
        elif first.char_start is not None:
            moment.char_start, moment.char_end = first.char_start, last.char_end
        moments.append(moment)
        batch.clear()

    for unit in units:
        size = sum(len(u.text) for u in batch)
        if batch and size + len(unit.text) > TARGET_MOMENT_CHARS:
            flush()
            if len(moments) >= cap:
                return moments, covered
        batch.append(unit)
        covered += 1
    flush()
    return moments[:cap], covered


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@dataclass
class LocalDocumentBackend:
    """Extract, locate and pack document text. No network, no model."""

    name: str = NAME
    modalities: FrozenSet[str] = field(default_factory=lambda: frozenset({"document"}))

    def describe(self, request: DescribeRequest) -> Optional[DescribeResult]:
        if request.fetch is None:
            return None
        try:
            raw = request.fetch()
        except Exception:
            logger.info("document fetch failed", exc_info=True)
            return None
        if not raw:
            return None

        fmt = document_format(request.uri, request.mime, raw)
        warnings: List[str] = []
        try:
            if fmt == "text":
                units = _text_units(raw)
            elif fmt == "docx":
                units = _docx_units(raw)
            elif fmt == "pptx":
                units = _pptx_units(raw)
            elif fmt == "epub":
                units = _epub_units(raw)
            elif fmt == "pdf":
                units = _pdf_units(raw, request, warnings)
            else:
                return None
        except Exception:
            logger.info("document extraction failed (%s)", fmt, exc_info=True)
            return None
        if not units:
            return DescribeResult(provider=self.name, warnings=warnings) if warnings else None

        moments, covered = pack_units(units, request.max_moments)
        if covered < len(units):
            where = (f"through page {units[covered - 1].page}" if units[covered - 1].page is not None
                     else f"through character {units[covered - 1].char_end}")
            warnings.append(
                f"document truncated {where} of {len(units)} section(s) by the "
                f"{request.max_moments}-moment cap; raise MNEMOSYNE_MODALITY_MAX_MOMENTS to cover more"
            )
        return DescribeResult(moments=moments, provider=self.name, model=fmt, warnings=warnings)
