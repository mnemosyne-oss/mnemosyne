"""Regression tests for the optional memory browser helpers."""

import sqlite3
from pathlib import Path

import pytest

from mnemosyne.core.banks import BankManager
from mnemosyne.integrations import memory_browser


def test_build_html_renders_css_template_without_format_key_error():
    html = memory_browser._build_html([])

    assert "<style>" in html
    assert ":root {" in html
    assert "No memories found." in html


def test_resolve_db_path_matches_bank_manager_for_default_and_named_banks(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    manager = BankManager(data_dir=tmp_path)

    assert Path(memory_browser._resolve_db_path("default")) == manager.get_bank_db_path("default")
    assert Path(memory_browser._resolve_db_path("work")) == manager.get_bank_db_path("work")


@pytest.mark.parametrize("bank", ["", "../escape", "work/name"])
def test_resolve_db_path_rejects_invalid_bank_names(bank):
    with pytest.raises(ValueError):
        memory_browser._resolve_db_path(bank)


@pytest.mark.parametrize("statement", ["INSERT INTO items VALUES (1)", "CREATE TABLE extra (id INTEGER)"])
def test_existing_database_connection_rejects_writes(tmp_path, statement):
    db_path = tmp_path / "existing.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE items (id INTEGER)")

    conn = memory_browser._get_connection(str(db_path))
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(statement)
    finally:
        conn.close()


def test_missing_database_stats_does_not_create_a_database_file(tmp_path):
    missing_db = tmp_path / "missing.db"

    stats = memory_browser.get_memory_stats(str(missing_db))

    assert "error" in stats
    assert not missing_db.exists()


XSS = '"><img src=x onerror=alert(1)><script>alert(2)</script>'


def test_build_html_escapes_reflected_query():
    html = memory_browser._build_html([], query=XSS)

    assert "<img src=x onerror" not in html
    assert "<script>alert(2)" not in html
    assert 'value="&quot;&gt;&lt;img src=x onerror=alert(1)&gt;' in html


def test_build_html_escapes_stored_memory_fields():
    memory = {
        "id": 'a"b/<c>',
        "content": XSS,
        "source": XSS,
        "timestamp": XSS,
        "importance": XSS,
        "tier": XSS,
    }
    html = memory_browser._build_html([memory], source=XSS)

    assert "<img src=x onerror" not in html
    assert "<script>alert(2)" not in html
    assert 'href="/detail/a%22b%2F%3Cc%3E"' in html


def test_build_html_sets_script_nonce():
    html = memory_browser._build_html([], nonce="abc123")

    assert '<script nonce="abc123">' in html
    assert "onclick=" not in html


@pytest.mark.parametrize(
    "header, expected",
    [
        ("127.0.0.1:8081", "127.0.0.1"),
        ("LOCALHOST:8081", "localhost"),
        ("[::1]:8081", "[::1]"),
        ("evil.example", "evil.example"),
        ("evil.example:80", "evil.example"),
    ],
)
def test_host_name_parsing(header, expected):
    assert memory_browser._host_name(header) == expected


def _client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    db_path = Path(memory_browser._resolve_db_path("default"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE working_memory (id TEXT, content TEXT, source TEXT, timestamp TEXT, importance REAL)"
        )
        conn.execute(
            "INSERT INTO working_memory VALUES (?, ?, ?, ?, ?)",
            ("m1", XSS, "web", "2026-11-14T00:00:00", 0.5),
        )
    app = memory_browser.create_app(banks=["default"], default_bank="default")
    return TestClient(app, base_url="http://127.0.0.1:8081")


def test_index_and_detail_escape_and_send_csp(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    index = client.get("/", params={"q": XSS})
    assert index.status_code == 200
    assert "<img src=x onerror" not in index.text
    csp = index.headers["content-security-policy"]
    assert "script-src 'nonce-" in csp
    nonce = csp.split("'nonce-", 1)[1].split("'", 1)[0]
    assert f'<script nonce="{nonce}">' in index.text

    detail = client.get("/detail/m1")
    assert detail.status_code == 200
    assert "<img src=x onerror" not in detail.text

    reflected = client.get("/detail/%3Cimg%20src%3Dx%20onerror%3Dalert(1)%3E")
    assert "<img src=x onerror" not in reflected.text


def test_rejects_non_local_host_header(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    response = client.get("/api/search", headers={"host": "attacker.example:8081"})
    assert response.status_code == 403
