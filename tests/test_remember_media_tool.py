"""``mnemosyne_remember_media`` on every tool surface: MCP, both Hermes providers, CLI.

The tool adds guards the SDK does not have, because a tool caller may be a
remote MCP client or a steerable model: local files only inside
MNEMOSYNE_MEDIA_ALLOWED_PATHS, no internal URLs, bounded inline payloads.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mnemosyne.core import media_tool
from mnemosyne.core.media_tool import MediaToolError, check_tool_ref

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    """Fresh config seeded from the env, the path a new install takes."""
    from mnemosyne.core.config import MnemosyneConfig

    def _set(**env):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "cfg"))
        monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(tmp_path / "blobs"))
        monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        MnemosyneConfig.reset_instance()

    yield _set
    MnemosyneConfig.reset_instance()


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_local_files_are_refused_until_a_root_is_configured(cfg, tmp_path):
    cfg()
    image = tmp_path / "shot.png"
    image.write_bytes(PNG)
    with pytest.raises(MediaToolError, match="MNEMOSYNE_MEDIA_ALLOWED_PATHS"):
        check_tool_ref(str(image))


def test_local_file_inside_an_allowed_root_is_accepted_and_resolved(cfg, tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    image = media / "shot.png"
    image.write_bytes(PNG)
    cfg(MNEMOSYNE_MEDIA_ALLOWED_PATHS=str(media))
    assert check_tool_ref(str(image)) == str(image.resolve())
    assert check_tool_ref("file://" + str(image)) == str(image.resolve())


def test_paths_outside_the_root_and_symlink_escapes_are_refused(cfg, tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    secret = tmp_path / "id_rsa"
    secret.write_text("-----BEGIN OPENSSH PRIVATE KEY-----")
    link = media / "innocent.png"
    link.symlink_to(secret)
    cfg(MNEMOSYNE_MEDIA_ALLOWED_PATHS=str(media))
    with pytest.raises(MediaToolError, match="outside"):
        check_tool_ref(str(secret))
    with pytest.raises(MediaToolError, match="outside"):
        check_tool_ref(str(link))
    with pytest.raises(MediaToolError, match="outside"):
        check_tool_ref(str(media / ".." / "id_rsa"))


def test_relative_paths_and_directories_are_refused(cfg, tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    cfg(MNEMOSYNE_MEDIA_ALLOWED_PATHS=str(media))
    with pytest.raises(MediaToolError, match="absolute"):
        check_tool_ref("shot.png")
    with pytest.raises(MediaToolError, match="regular file"):
        check_tool_ref(str(media))


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8080/a.mp4",
    "http://localhost/a.mp3",
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/a.png",
    "http://[::1]/a.png",
])
def test_internal_urls_are_refused(cfg, url):
    cfg()
    with pytest.raises(MediaToolError, match="loopback, private or link-local"):
        check_tool_ref(url)


def test_internal_urls_can_be_allowed_for_a_trusted_lan(cfg):
    cfg(MNEMOSYNE_MEDIA_ALLOW_PRIVATE_URLS="1")
    assert check_tool_ref("http://127.0.0.1:8080/a.mp4") == "http://127.0.0.1:8080/a.mp4"


def test_public_url_passes_when_it_resolves_publicly(cfg, monkeypatch):
    cfg()
    monkeypatch.setattr(media_tool.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    assert check_tool_ref("https://example.com/a.png") == "https://example.com/a.png"


def test_hostnames_that_resolve_internally_are_refused(cfg, monkeypatch):
    cfg()
    monkeypatch.setattr(media_tool.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("192.168.1.10", 0))])
    with pytest.raises(MediaToolError):
        check_tool_ref("https://nas.example.com/a.png")


def test_inline_payload_size_is_bounded(cfg, monkeypatch):
    cfg()
    monkeypatch.setattr(media_tool, "MAX_INLINE_BYTES", 16)
    small = "data:image/png;base64," + base64.b64encode(b"x" * 8).decode()
    big = "data:image/png;base64," + base64.b64encode(b"x" * 64).decode()
    assert check_tool_ref(small) == small
    with pytest.raises(MediaToolError, match="limit"):
        check_tool_ref(big)


def test_unknown_schemes_are_refused(cfg):
    cfg()
    with pytest.raises(MediaToolError, match="scheme"):
        check_tool_ref("ftp://example.com/a.png")


def test_argument_validation_returns_errors_not_exceptions(cfg):
    cfg()
    assert media_tool.remember_media_tool(None, {})["error"] == "ref is required"
    bad = media_tool.remember_media_tool(None, {"ref": "data:image/png;base64,AA==", "modality": "smell"})
    assert "modality" in bad["error"]
    bad = media_tool.remember_media_tool(None, {"ref": "data:image/png;base64,AA==", "max_moments": 0})
    assert "max_moments" in bad["error"]


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------

def _mcp_env(cfg, tmp_path, **extra):
    cfg(MNEMOSYNE_HOME=str(tmp_path / "home"), **extra)


def test_mcp_advertises_the_tool_with_a_tenant_bank(cfg, tmp_path):
    from mnemosyne import mcp_tools

    _mcp_env(cfg, tmp_path)
    tool = next(t for t in mcp_tools.get_tool_definitions() if t["name"] == "mnemosyne_remember_media")
    props = tool["input_schema"]["properties"]
    assert set(props) >= {"ref", "modality", "hint", "max_moments", "bank"}


def test_mcp_registers_an_inline_image_into_the_requested_bank(cfg, tmp_path):
    from mnemosyne import mcp_tools

    _mcp_env(cfg, tmp_path)
    ref = "data:image/png;base64," + base64.b64encode(PNG).decode()
    out = mcp_tools.handle_tool_call("mnemosyne_remember_media", {"ref": ref, "title": "logo", "bank": "tenant-a"})
    payload = json.loads(out) if isinstance(out, str) else out
    if "content" in payload and isinstance(payload["content"], list):
        payload = json.loads(payload["content"][0]["text"])
    assert payload["status"] == "unavailable" and payload["bank"] == "tenant-a"
    assert payload["asset_id"] and payload["anchor_memory_id"]
    assert payload["described"] is False


def test_mcp_refuses_a_local_path_without_allowed_roots(cfg, tmp_path):
    from mnemosyne import mcp_tools

    _mcp_env(cfg, tmp_path)
    (tmp_path / "secrets.txt").write_text("token=abc")
    payload = mcp_tools._handle_remember_media({"ref": str(tmp_path / "secrets.txt")})
    assert payload["status"] == "error" and "MNEMOSYNE_MEDIA_ALLOWED_PATHS" in payload["error"]


# ---------------------------------------------------------------------------
# Hermes providers
# ---------------------------------------------------------------------------

def _import_providers():
    import importlib

    out = {}
    for name, root in (("hermes_memory_provider", PROJECT_ROOT),
                       ("mnemosyne_hermes", PROJECT_ROOT / "integrations" / "hermes" / "src")):
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            sys.modules.pop(key)
        sys.path.insert(0, str(root))
        try:
            out[name] = importlib.import_module(name)
        finally:
            sys.path.remove(str(root))
    return out


@pytest.mark.parametrize("which", ["hermes_memory_provider", "mnemosyne_hermes"])
def test_hermes_provider_describes_a_document_inside_the_allowed_root(cfg, tmp_path, which):
    media = tmp_path / "media"
    media.mkdir()
    doc = media / "runbook.md"
    doc.write_text("# Runbook\n\nRotate the relay key with sync-generate-key.\n")
    cfg(MNEMOSYNE_MEDIA_ALLOWED_PATHS=str(media), MNEMOSYNE_MODALITY_ENABLED="1")
    module = _import_providers()[which]
    provider = module.MnemosyneMemoryProvider()
    provider.initialize(f"media-{which}", hermes_home=str(tmp_path / which),
                        profile_isolation=False, agent_context="primary")
    assert "mnemosyne_remember_media" in {s["name"] for s in provider.get_tool_schemas()}

    ok = json.loads(provider.handle_tool_call("mnemosyne_remember_media", {"ref": str(doc)}))
    assert ok["status"] == "ok" and ok["described"] is True, ok
    assert len(ok["memory_ids"]) == 1

    refused = json.loads(provider.handle_tool_call("mnemosyne_remember_media",
                                                   {"ref": str(tmp_path / "elsewhere.md")}))
    assert refused["status"] == "error"
    assert "not found" in refused["error"] or "outside" in refused["error"]

    hits = json.loads(provider.handle_tool_call("mnemosyne_recall", {"query": "rotate the relay key"}))
    blob = json.dumps(hits)
    assert "sync-generate-key" in blob


def test_provider_schema_copies_match_canonical_without_the_tenant_bank(cfg):
    from mnemosyne.tool_schemas import REMEMBER_MEDIA_SCHEMA as canonical

    cfg()
    modules = _import_providers()
    legacy = modules["hermes_memory_provider"].REMEMBER_MEDIA_SCHEMA
    sys.path.insert(0, str(PROJECT_ROOT / "integrations" / "hermes" / "src"))
    try:
        from mnemosyne_hermes.tools import REMEMBER_MEDIA_SCHEMA as packaged
    finally:
        sys.path.pop(0)
    expected = json.loads(json.dumps(canonical))
    expected["parameters"]["properties"].pop("bank", None)
    assert legacy == expected
    assert packaged == expected


def test_catalog_declares_the_tool():
    text = (PROJECT_ROOT / "integrations" / "hermes-catalog" / "plugin.yaml").read_text()
    assert "  - mnemosyne_remember_media\n" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_media_registers_a_file_and_prints_json(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("Alpha release notes.\n")
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "MNEMOSYNE_DATA_DIR": str(tmp_path / "data"),
        "MNEMOSYNE_BLOB_DIR": str(tmp_path / "blobs"),
        "MNEMOSYNE_NO_EMBEDDINGS": "1",
        "MNEMOSYNE_MODALITY_ENABLED": "1",
        "PYTHONPATH": str(PROJECT_ROOT),
    }
    out = subprocess.run([sys.executable, "-m", "mnemosyne.cli", "media", str(doc), "--json"],
                         capture_output=True, text=True, env=env, timeout=180)
    if out.returncode != 0 and "No module named mnemosyne.cli.__main__" in out.stderr:
        out = subprocess.run([sys.executable, "-c", "import sys; from mnemosyne.cli import run_cli; "
                              f"sys.argv=['mnemosyne','media',{str(doc)!r},'--json']; run_cli()"],
                             capture_output=True, text=True, env=env, timeout=180)
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok" and len(payload["memory_ids"]) == 1
