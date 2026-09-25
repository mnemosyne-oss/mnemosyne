"""The ``mnemosyne_remember_media`` tool, shared by MCP and both Hermes providers.

``BeamMemory.remember_media()`` trusts its caller: it is SDK code, run by the
person whose files it reads. A *tool* call is different. Over MCP the caller
may be a remote client (SSE / Streamable HTTP on a non-loopback bind), and in
Hermes it is a model that can be steered by whatever text it last read. So the
tool surface adds three guards the SDK does not have:

- **Local files are opt-in.** A file reference is accepted only when it
  resolves, after symlinks, inside a directory listed in
  ``media_allowed_paths`` (``MNEMOSYNE_MEDIA_ALLOWED_PATHS``). With nothing
  listed, tool calls cannot name local files at all. Otherwise a single call
  could pull ``~/.ssh`` or a ``.env`` into memory, where recall would hand it
  back to anyone who can query.
- **No internal URLs.** ``http(s)`` references whose host resolves to a
  loopback, private, link-local or reserved address are refused, because the
  audio and video paths fetch them from this machine. ``media_allow_private_urls``
  turns that off for a trusted LAN media server.
- **Bounded inline payloads.** A ``data:`` URI is decoded into the blob store,
  so its size is capped at ``MAX_INLINE_BYTES``.

``blob://`` and bare sha256 references name content already in the store and
pass unchanged.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import unquote, urlparse

#: Largest decoded ``data:`` payload a tool call may inline.
MAX_INLINE_BYTES = 25 * 1024 * 1024

_MODALITIES = ("image", "video", "audio", "document")


class MediaToolError(ValueError):
    """A tool argument was refused. The message is safe to show the caller."""


def allowed_roots() -> List[Path]:
    """Directories a tool call may read local media from, resolved."""
    from mnemosyne.core import config

    raw = str(config.get_str("media_allowed_paths", "") or "")
    roots: List[Path] = []
    for part in raw.replace(",", os.pathsep).split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        try:
            roots.append(Path(part).expanduser().resolve(strict=False))
        except (OSError, RuntimeError):
            continue
    return roots


def _allow_private_urls() -> bool:
    from mnemosyne.core import config

    return bool(config.get_bool("media_allow_private_urls", False))


def _check_file(ref: str) -> str:
    """Return the resolved absolute path, or raise."""
    raw = ref[len("file://"):] if ref.startswith("file://") else ref
    raw = unquote(raw) if ref.startswith("file://") else raw
    roots = allowed_roots()
    if not roots:
        raise MediaToolError(
            "local file paths are disabled for tool calls. Set "
            "MNEMOSYNE_MEDIA_ALLOWED_PATHS (or media_allowed_paths in config.yaml) "
            "to the directories media may be read from, or pass an https:// URL "
            "or a data: URI instead."
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise MediaToolError("local file paths must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise MediaToolError("file not found") from None
    if not resolved.is_file():
        raise MediaToolError("not a regular file")
    for root in roots:
        try:
            resolved.relative_to(root)
            return str(resolved)
        except ValueError:
            continue
    raise MediaToolError("file is outside MNEMOSYNE_MEDIA_ALLOWED_PATHS")


def _is_internal(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return (ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved
            or ip.is_multicast or ip.is_unspecified)


def _check_url(ref: str) -> None:
    host = urlparse(ref).hostname
    if not host:
        raise MediaToolError("URL has no host")
    if _allow_private_urls():
        return
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError:
        raise MediaToolError("URL host does not resolve") from None
    if any(_is_internal(info[4][0]) for info in infos):
        raise MediaToolError(
            "URL resolves to a loopback, private or link-local address; set "
            "MNEMOSYNE_MEDIA_ALLOW_PRIVATE_URLS=1 to allow a trusted local media server"
        )


def _check_inline(ref: str) -> None:
    comma = ref.find(",")
    if comma < 0:
        raise MediaToolError("malformed data: URI")
    header, body = ref[:comma], ref[comma + 1:]
    size = (len(body) * 3) // 4 if ";base64" in header else len(unquote(body))
    if size > MAX_INLINE_BYTES:
        raise MediaToolError(f"inline payload is over the {MAX_INLINE_BYTES // (1024 * 1024)} MB limit")


def check_tool_ref(ref: Any) -> str:
    """Validate a tool-supplied reference. Returns the ref to ingest."""
    if not isinstance(ref, str) or not ref.strip():
        raise MediaToolError("ref is required")
    ref = ref.strip()
    lowered = ref.lower()
    if lowered.startswith("data:"):
        _check_inline(ref)
        return ref
    if lowered.startswith("blob://") or (len(ref) == 64 and all(c in "0123456789abcdefABCDEF" for c in ref)):
        return ref
    if lowered.startswith(("http://", "https://")):
        _check_url(ref)
        return ref
    if "://" in ref and not lowered.startswith("file://"):
        raise MediaToolError("unsupported reference scheme")
    return _check_file(ref)


def remember_media_tool(beam, args: Dict[str, Any], *, default_scope: str = "session") -> Dict[str, Any]:
    """Run one ``mnemosyne_remember_media`` call. Never raises."""
    try:
        ref = check_tool_ref(args.get("ref"))
        modality = args.get("modality")
        if modality is not None and str(modality).strip().lower() not in _MODALITIES:
            raise MediaToolError(f"modality must be one of {', '.join(_MODALITIES)}")
        scope = str(args.get("scope") or default_scope).strip().lower()
        if scope not in ("session", "global"):
            raise MediaToolError("scope must be 'session' or 'global'")
        max_moments = args.get("max_moments")
        if max_moments is not None:
            max_moments = int(max_moments)
            if not 1 <= max_moments <= 100:
                raise MediaToolError("max_moments must be between 1 and 100")
        importance = float(args.get("importance", 0.5))
        if not 0.0 <= importance <= 1.0:
            raise MediaToolError("importance must be between 0 and 1")
    except MediaToolError as exc:
        return {"status": "error", "error": str(exc)}
    except (TypeError, ValueError):
        return {"status": "error", "error": "invalid argument types"}

    try:
        result = beam.remember_media(
            ref,
            modality=str(modality).strip().lower() if modality else None,
            mime=args.get("mime") or None,
            title=args.get("title") or None,
            hint=args.get("hint") or None,
            max_moments=max_moments,
            importance=importance,
            scope=scope,
        )
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}

    payload = asdict(result)
    payload["described"] = payload.get("status") in ("ok", "partial")
    return payload

