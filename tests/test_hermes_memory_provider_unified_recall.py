from __future__ import annotations

import json
from pathlib import Path

from hermes_memory_provider import MnemosyneMemoryProvider


def _provider(tmp_path: Path, monkeypatch, *, shared_surface_read: bool = False):
    data_dir = tmp_path / "mnemosyne-data"
    hermes_home = tmp_path / "profiles" / "Mob"
    hermes_home.mkdir(parents=True)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir / "private"))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    provider = MnemosyneMemoryProvider()
    provider.initialize(
        session_id="mob-session",
        hermes_home=str(hermes_home),
        agent_identity="Mob",
        shared_surface_path=str(data_dir / "shared" / "mnemosyne.db"),
        shared_surface_read=shared_surface_read,
    )
    assert provider._beam is not None
    return provider, data_dir


def _call(provider: MnemosyneMemoryProvider, name: str, args: dict) -> dict:
    return json.loads(provider.handle_tool_call(name, args))


def _seed_private(provider: MnemosyneMemoryProvider, content: str, importance: float = 0.6) -> str:
    res = _call(provider, "mnemosyne_remember", {
        "content": content,
        "importance": importance,
        "source": "fact",
        "scope": "global",
    })
    assert res["status"] == "stored", res
    return res["memory_id"]


def _seed_surface(provider: MnemosyneMemoryProvider, content: str, kind: str = "preference") -> str:
    res = _call(provider, "mnemosyne_shared_remember", {
        "content": content,
        "kind": kind,
        "importance": 0.8,
    })
    assert res["status"] == "stored_shared", res
    return res["memory_id"]


# --- Default behavior: shared_surface_read=False ---------------------------

def test_recall_default_returns_private_only(tmp_path, monkeypatch):
    provider, _ = _provider(tmp_path, monkeypatch)

    _seed_private(provider, "User keeps notes in /home/user/notes.md")
    _seed_surface(provider, "User prefers Tailscale over OpenVPN")

    res = _call(provider, "mnemosyne_recall", {"query": "Tailscale", "limit": 10})

    assert res["shared_surface_read"] is False
    contents = [r["content"] for r in res["results"]]
    assert all("Tailscale" not in c for c in contents)


def test_recall_default_tags_results_as_private(tmp_path, monkeypatch):
    provider, _ = _provider(tmp_path, monkeypatch)
    _seed_private(provider, "Project root lives at /tmp/project")

    res = _call(provider, "mnemosyne_recall", {"query": "project root", "limit": 5})

    assert len(res["results"]) >= 1
    for r in res["results"]:
        assert r.get("bank") == "private"
        assert not r.get("shared_surface")


# --- Opt-in behavior: shared_surface_read=True -----------------------------

def test_recall_with_surface_read_includes_shared_results(tmp_path, monkeypatch):
    provider, _ = _provider(tmp_path, monkeypatch, shared_surface_read=True)
    _seed_private(provider, "User stores notes in /tmp/notes.md")
    _seed_surface(provider, "User prefers Tailscale over OpenVPN for VPN")

    res = _call(provider, "mnemosyne_recall", {"query": "Tailscale VPN", "limit": 10})

    assert res["shared_surface_read"] is True
    banks = {r.get("bank") for r in res["results"]}
    assert "surface" in banks


def test_recall_tags_surface_results_with_bank_surface(tmp_path, monkeypatch):
    provider, _ = _provider(tmp_path, monkeypatch, shared_surface_read=True)
    _seed_surface(provider, "User uses cargo nextest for Rust testing")

    res = _call(provider, "mnemosyne_recall", {"query": "cargo nextest", "limit": 5})

    surface_rows = [r for r in res["results"] if r.get("bank") == "surface"]
    assert surface_rows, res
    for r in surface_rows:
        assert r["shared_surface"] is True


def test_recall_merges_results_from_both_banks(tmp_path, monkeypatch):
    provider, _ = _provider(tmp_path, monkeypatch, shared_surface_read=True)
    _seed_private(provider, "User project Acme uses Postgres on port 5432")
    _seed_surface(provider, "User prefers Postgres for production databases")

    res = _call(provider, "mnemosyne_recall", {"query": "Postgres", "limit": 10})

    banks = {r.get("bank") for r in res["results"]}
    assert "private" in banks
    assert "surface" in banks


def test_recall_truncates_to_top_k_after_merge(tmp_path, monkeypatch):
    provider, _ = _provider(tmp_path, monkeypatch, shared_surface_read=True)
    for i in range(5):
        _seed_private(provider, f"User runs script number {i} for migration tasks")
    for i in range(5):
        _seed_surface(provider, f"User prefers tool variant {i} for migration tasks")

    res = _call(provider, "mnemosyne_recall", {"query": "migration tasks", "limit": 4})

    assert res["count"] <= 4
    assert len(res["results"]) <= 4


def test_recall_surface_init_failure_falls_back_to_private(tmp_path, monkeypatch):
    provider, _ = _provider(tmp_path, monkeypatch, shared_surface_read=True)
    _seed_private(provider, "User has a homelab Proxmox setup")
    # Force surface beam to fail by clearing it and breaking the path
    provider._surface_beam = None
    provider._shared_surface_path = Path("/nonexistent/dir/that/cannot/be/created/file.db")

    # Bypass mkdir guard by also patching _ensure_surface_beam to raise
    def _raise(*_a, **_k):
        raise RuntimeError("surface init forced failure")
    monkeypatch.setattr(provider, "_ensure_surface_beam", _raise, raising=True)

    res = _call(provider, "mnemosyne_recall", {"query": "Proxmox homelab", "limit": 5})

    assert res["shared_surface_read"] is True
    # Private results should still come through
    assert len(res["results"]) >= 1
    for r in res["results"]:
        assert r.get("bank") == "private"


# --- Config wiring ----------------------------------------------------------

def test_shared_surface_read_in_config_schema(tmp_path, monkeypatch):
    provider, _ = _provider(tmp_path, monkeypatch)
    schema = provider.get_config_schema()
    keys = [entry["key"] for entry in schema]
    assert "shared_surface_read" in keys
    entry = next(e for e in schema if e["key"] == "shared_surface_read")
    assert entry["default"] is False


def test_shared_surface_read_reads_from_config_yaml(tmp_path, monkeypatch):
    data_dir = tmp_path / "mnemosyne-data"
    hermes_home = tmp_path / "profiles" / "Mob"
    hermes_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "memory:\n  mnemosyne:\n    shared_surface_read: true\n"
    )
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir / "private"))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")

    provider = MnemosyneMemoryProvider()
    provider.initialize(
        session_id="mob-session",
        hermes_home=str(hermes_home),
        agent_identity="Mob",
        shared_surface_path=str(data_dir / "shared" / "mnemosyne.db"),
    )

    assert provider._shared_surface_read is True


def test_shared_surface_read_kwarg_overrides_config(tmp_path, monkeypatch):
    data_dir = tmp_path / "mnemosyne-data"
    hermes_home = tmp_path / "profiles" / "Mob"
    hermes_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "memory:\n  mnemosyne:\n    shared_surface_read: true\n"
    )
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir / "private"))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")

    provider = MnemosyneMemoryProvider()
    provider.initialize(
        session_id="mob-session",
        hermes_home=str(hermes_home),
        agent_identity="Mob",
        shared_surface_path=str(data_dir / "shared" / "mnemosyne.db"),
        shared_surface_read=False,
    )

    assert provider._shared_surface_read is False
