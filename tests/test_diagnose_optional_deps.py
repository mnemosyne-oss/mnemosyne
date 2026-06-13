import mnemosyne

from mnemosyne import diagnose


def _entry(summary, check):
    return next(item for item in summary["entries"] if item["check"] == check)


def test_diagnose_version_falls_back_to_distribution_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delattr(mnemosyne, "__version__", raising=False)
    monkeypatch.setattr(diagnose.importlib.metadata, "version", lambda name: "9.9.9" if name == "mnemosyne-memory" else "0")

    summary = diagnose.run_diagnostics()

    assert _entry(summary, "mnemosyne_version")["status"] == "9.9.9"


def test_diagnose_treats_ctransformers_as_optional(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))

    summary = diagnose.run_diagnostics()
    ctransformers = _entry(summary, "ctransformers")

    assert ctransformers["status"] in {"OK", "OPTIONAL"}
    if ctransformers["status"] == "OPTIONAL":
        assert "local-GGUF fallback" in ctransformers["detail"]
    assert ctransformers["status"] not in {"MISSING", "ERROR"}
