"""Local document understanding: text is extracted and located, not described.

Fixtures are built in the test (stdlib zip/XML, a handwritten PDF), so no
binary files live in the repo.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from mnemosyne.core.modality_backends import (
    CallableModalityBackend,
    DescribedMoment,
    DescribeRequest,
    DescribeResult,
    set_modality_backend,
)
from mnemosyne.core.modality_documents import (
    TARGET_MOMENT_CHARS,
    LocalDocumentBackend,
    _Unit,
    document_format,
    pack_units,
)


def _req(raw: bytes, uri: str, max_moments: int = 12, mime=None):
    return DescribeRequest(modality="document", uri=uri, mime=mime,
                           max_moments=max_moments, timeout=10, fetch=lambda: raw)


def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
A = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
     'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')


def _docx(paragraphs):
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    return _zip({"word/document.xml": f"<w:document {W}><w:body>{body}</w:body></w:document>"})


def _pptx(slides):
    files = {}
    for i, lines in enumerate(slides, start=1):
        paras = "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in lines)
        files[f"ppt/slides/slide{i}.xml"] = f"<p:sld {A}><p:cSld><p:spTree><p:sp><p:txBody>{paras}</p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
    return _zip(files)


def _epub(chapters):
    manifest = "".join(f'<item id="c{i}" href="text/c{i}.xhtml" media-type="application/xhtml+xml"/>'
                       for i in range(len(chapters)))
    spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
    files = {
        "META-INF/container.xml": '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                                  '<rootfile full-path="OEBPS/content.opf"/></rootfiles></container>',
        "OEBPS/content.opf": f'<package xmlns="http://www.idpf.org/2007/opf"><manifest>{manifest}</manifest>'
                             f'<spine>{spine}</spine></package>',
    }
    for i, body in enumerate(chapters):
        files[f"OEBPS/text/c{i}.xhtml"] = f"<html><body><h1>Chapter {i + 1}</h1><p>{body}</p></body></html>"
    return _zip(files)


def _pdf(pages):
    """A minimal valid PDF. ``None`` in ``pages`` is a page with no text layer."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = (f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET" if text else "").encode()
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream")
        content_ref = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_ref} 0 R >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{obj}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_format_detection_prefers_mime_then_extension_then_magic():
    assert document_format("a.bin", "text/markdown", b"") == "text"
    assert document_format("notes.MD", None, b"") == "text"
    assert document_format("blob://sha256/x", None, b"%PDF-1.4") == "pdf"
    assert document_format("blob://sha256/x", None, _docx(["x"])) == "docx"
    assert document_format("blob://sha256/x", None, _pptx([["x"]])) == "pptx"
    assert document_format("blob://sha256/x", None, _epub(["x"])) == "epub"
    assert document_format("a.xyz", None, b"\x00\x01") is None


def test_markdown_paragraphs_become_char_located_passages():
    raw = b"# Title\n\nFirst paragraph about recall.\n\nSecond paragraph about sleep.\n"
    result = LocalDocumentBackend().describe(_req(raw, "notes.md"))
    assert result.provider == "local_document" and result.model == "text"
    [moment] = result.moments
    assert moment.kind == "page"
    assert raw.decode()[moment.char_start:moment.char_end].endswith("about sleep.")
    assert "First paragraph about recall." in moment.text


def test_docx_paragraphs_are_extracted_in_order():
    result = LocalDocumentBackend().describe(_req(_docx(["Alpha plan", "Beta risks"]), "plan.docx"))
    assert result.moments[0].text == "Alpha plan\n\nBeta risks"
    assert result.moments[0].char_start == 0


def test_pptx_slides_are_located_by_slide_number():
    big = "x" * (TARGET_MOMENT_CHARS + 10)
    result = LocalDocumentBackend().describe(_req(_pptx([["Intro"], [big], ["Close"]]), "deck.pptx"))
    spans = [(m.page_start, m.page_end) for m in result.moments]
    assert spans == [(1, 1), (2, 2), (3, 3)]
    assert result.moments[2].text == "Close"


def test_epub_chapters_follow_the_spine():
    result = LocalDocumentBackend().describe(_req(_epub(["Once upon a time.", "The end."]), "book.epub"))
    text = result.moments[0].text
    assert text.index("Once upon a time.") < text.index("The end.")
    assert (result.moments[0].page_start, result.moments[0].page_end) == (1, 2)


def test_pdf_text_pages_are_located_by_page():
    pytest.importorskip("pypdfium2")
    result = LocalDocumentBackend().describe(_req(_pdf(["Quarterly revenue grew", "Churn fell"]), "q3.pdf"))
    assert result.model == "pdf"
    assert "Quarterly revenue grew" in result.moments[0].text and "Churn fell" in result.moments[0].text
    assert (result.moments[0].page_start, result.moments[0].page_end) == (1, 2)


def test_scanned_pdf_page_goes_to_the_image_backend(monkeypatch, tmp_path):
    pytest.importorskip("pypdfium2")
    pytest.importorskip("PIL")
    from mnemosyne.core.config import MnemosyneConfig

    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("MNEMOSYNE_MODALITY_ENABLED", "1")
    MnemosyneConfig.reset_instance()
    seen = []

    def describe(request):
        seen.append((request.modality, request.mime, request.fetch()[:8]))
        return DescribeResult(moments=[DescribedMoment(kind="ocr", text="INVOICE 42 total 99 EUR")],
                              provider="img")

    set_modality_backend(CallableModalityBackend(name="img", func=describe,
                                                 modalities=frozenset({"image"})))
    try:
        result = LocalDocumentBackend().describe(_req(_pdf(["Cover letter", None]), "scan.pdf"))
    finally:
        MnemosyneConfig.reset_instance()
    assert seen == [("image", "image/png", b"\x89PNG\r\n\x1a\n")]
    assert "INVOICE 42" in result.moments[0].text
    assert result.moments[0].page_end == 2


def test_scanned_pdf_without_an_image_backend_says_so(monkeypatch, tmp_path):
    pytest.importorskip("pypdfium2")
    result = LocalDocumentBackend().describe(_req(_pdf([None]), "scan.pdf"))
    assert result.moments == []
    assert "no text layer" in result.warnings[0]


def test_pdf_without_the_media_extra_degrades_with_an_install_hint(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "pypdfium2":
            raise ImportError("no")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    result = LocalDocumentBackend().describe(_req(b"%PDF-1.4\n", "a.pdf"))
    assert result.moments == [] and "mnemosyne-memory[media]" in result.warnings[0]


def test_packing_respects_the_cap_and_reports_truncation():
    units = [_Unit(text="p" * (TARGET_MOMENT_CHARS - 1), page=i) for i in range(1, 6)]
    moments, covered = pack_units(units, 2)
    assert len(moments) == 2 and covered == 2
    result = LocalDocumentBackend().describe(_req(_pptx([["s" * TARGET_MOMENT_CHARS]] * 5), "d.pptx", max_moments=2))
    assert len(result.moments) == 2
    assert "truncated through page 2 of 5" in result.warnings[0]


def test_corrupt_file_is_unavailable_not_an_exception():
    assert LocalDocumentBackend().describe(_req(b"PK\x03\x04garbage", "x.docx")) is None


def test_remember_media_on_a_markdown_file_needs_no_endpoint(tmp_path, monkeypatch):
    """The gate is the only requirement: no base URL, no key, no model."""
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.config import MnemosyneConfig
    from mnemosyne.core.media import MediaStore

    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("MNEMOSYNE_MODALITY_ENABLED", "1")
    MnemosyneConfig.reset_instance()
    try:
        doc = tmp_path / "runbook.md"
        doc.write_text("# Runbook\n\nRestart the sync relay with systemctl restart mnemosyne-sync.\n")
        beam = BeamMemory(session_id="docs", db_path=tmp_path / "m.db")
        result = beam.remember_media(str(doc))
        assert result.status == "ok", result.warnings
        moments = MediaStore(conn=beam.conn).get_moments(result.asset_id)
        assert [(m["kind"], m["span_kind"]) for m in moments] == [("page", "char")]
        hits = beam.recall("how do I restart the sync relay", top_k=5)
        assert any("systemctl restart mnemosyne-sync" in (h.get("content") or "") for h in hits)
    finally:
        MnemosyneConfig.reset_instance()


def test_documents_are_not_read_with_the_gate_off(tmp_path, monkeypatch):
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.config import MnemosyneConfig

    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("MNEMOSYNE_MODALITY_ENABLED", "0")
    MnemosyneConfig.reset_instance()
    try:
        doc = tmp_path / "secret.md"
        doc.write_text("do not read")
        result = BeamMemory(session_id="off", db_path=tmp_path / "m.db").remember_media(str(doc))
        assert result.status == "unavailable" and result.moment_ids == []
    finally:
        MnemosyneConfig.reset_instance()
