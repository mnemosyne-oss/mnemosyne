from mnemosyne.diagnose import run_diagnostics


def test_diagnose_treats_ctransformers_as_optional(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    result = run_diagnostics()
    ctransformers = [
        entry for entry in result["entries"]
        if entry["category"] == "deps" and entry["check"] == "ctransformers"
    ]

    assert ctransformers, "diagnostics should report local LLM backend status"
    assert ctransformers[0]["status"] in {"OK", "OPTIONAL_MISSING", "OPTIONAL_ERROR"}
    assert ctransformers[0]["status"] not in {"MISSING", "ERROR"}

    if ctransformers[0]["status"] != "OK":
        assert result["checks_optional_missing"] >= 1
