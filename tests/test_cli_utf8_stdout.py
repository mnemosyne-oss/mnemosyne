"""Regression test: the CLI must not crash when stdout/stderr are cp1252 pipes.

Agent tooling (and CI shells) frequently spawn the CLI with piped output. On
Windows, Python then defaults sys.stdout/sys.stderr to the cp1252 codec, and
printing a memory whose content contains a character outside cp1252 (e.g.
'\\u20b1') raised UnicodeEncodeError and killed the whole command. run_cli()
now reconfigures both streams to UTF-8 with errors='replace' at startup.
"""

import io
import sys

import pytest

from mnemosyne import cli


class _FakeMemory:
    """Minimal recall backend returning content with non-cp1252 characters."""

    def recall(self, query, top_k=5, explain=False):
        return [
            {
                "id": "abc123",
                "content": "salary \u20b135,000 per month \u2014 with \u2192 arrow",
                "score": 0.9,
            }
        ]


def _cp1252_stream():
    """Return a TextIOWrapper simulating a Windows cp1252 pipe."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", write_through=True)


def _install_argv(monkeypatch, query="peso salary"):
    """Point sys.argv at a plain 'mnemosyne recall <query>' invocation."""
    monkeypatch.setattr(sys, "argv", ["mnemosyne", "recall", query])


def test_run_cli_recall_survives_cp1252_stdout(monkeypatch):
    monkeypatch.setattr(cli, "_get_memory", lambda: _FakeMemory())
    stdout = _cp1252_stream()
    monkeypatch.setattr(sys, "stdout", stdout)
    _install_argv(monkeypatch)

    cli.run_cli()  # must not raise UnicodeEncodeError

    output = stdout.buffer.getvalue().decode("utf-8")
    assert "abc123" in output
    # The non-cp1252 characters must survive verbatim, not degrade to '?'.
    assert "\u20b1" in output
    assert "\u2014" in output
    assert "\u2192" in output


def test_run_cli_stderr_reconfigured_for_utf8(monkeypatch):
    monkeypatch.setattr(cli, "_get_memory", lambda: _FakeMemory())
    stderr = _cp1252_stream()
    monkeypatch.setattr(sys, "stderr", stderr)
    _install_argv(monkeypatch)

    cli.run_cli()

    # run_cli() must reconfigure stderr to UTF-8, not leave it as cp1252.
    assert stderr.encoding.lower().replace("-", "") == "utf8"


def test_run_cli_stderr_replacement_instead_of_crash(monkeypatch):
    """Unknown command echoes non-ASCII text through stderr instead of raising."""
    stderr = _cp1252_stream()
    monkeypatch.setattr(sys, "stderr", stderr)
    # Unknown command: run_cli prints the error (with the non-ASCII command
    # name echoed back) to stderr and exits 2.
    monkeypatch.setattr(sys, "argv", ["mnemosyne", "b\u00e5dcommand"])

    with pytest.raises(SystemExit) as excinfo:
        cli.run_cli()

    assert excinfo.value.code == 2
    assert stderr.encoding.lower().replace("-", "") == "utf8"
    output = stderr.buffer.getvalue().decode("utf-8")
    assert "Unknown command" in output
    assert "b\u00e5dcommand" in output
