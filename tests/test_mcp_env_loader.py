"""
Test auto .env loader functionality in mnemosyne/mcp_server.py.
"""

import os
import pytest
from mnemosyne import mcp_server


@pytest.fixture(autouse=True)
def restore_env():
    """Ensure os.environ state is restored after each test."""
    old_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(old_env)


def test_load_dotenv_explicit_path(tmp_path, monkeypatch):
    """Test loading an explicit .env file via --env-file / env_file_path."""
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        "export TEST_MCP_VAR_1=hello\n"
        "TEST_MCP_VAR_2='world'\n"
        "# comment line\n"
        "TEST_MCP_VAR_3=\"quotes\"\n",
        encoding="utf-8"
    )
    monkeypatch.delenv("TEST_MCP_VAR_1", raising=False)
    monkeypatch.delenv("TEST_MCP_VAR_2", raising=False)
    monkeypatch.delenv("TEST_MCP_VAR_3", raising=False)

    loaded = mcp_server._load_dotenv(str(env_file))
    assert loaded == str(env_file)
    assert os.environ.get("TEST_MCP_VAR_1") == "hello"
    assert os.environ.get("TEST_MCP_VAR_2") == "world"
    assert os.environ.get("TEST_MCP_VAR_3") == "quotes"


def test_load_dotenv_hermes_home(tmp_path, monkeypatch):
    """Test auto-discovering .env from HERMES_HOME."""
    hermes_dir = tmp_path / "hermes_test"
    hermes_dir.mkdir()
    env_file = hermes_dir / ".env"
    env_file.write_text("HERMES_MCP_AUTO_TEST=active\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(hermes_dir))
    monkeypatch.delenv("HERMES_MCP_AUTO_TEST", raising=False)

    loaded = mcp_server._load_dotenv()
    assert loaded == str(env_file)
    assert os.environ.get("HERMES_MCP_AUTO_TEST") == "active"


def test_load_dotenv_does_not_override_existing_env(tmp_path, monkeypatch):
    """Test that existing os.environ variables take precedence over .env file."""
    env_file = tmp_path / "override.env"
    env_file.write_text("EXISTING_VAR=from_file\n", encoding="utf-8")

    monkeypatch.setenv("EXISTING_VAR", "from_process")

    mcp_server._load_dotenv(str(env_file))
    assert os.environ.get("EXISTING_VAR") == "from_process"
