"""Embedding settings must honor config.yaml, not only os.environ (#1061).

The embedding module froze model, key, URL and dimension from the environment
at import time, so a config-only deployment silently fell back to the local
default model with no key. These tests run a fresh interpreter so the
import-time constants are exercised, with the parent's env stripped.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run(tmp_path, config_text: str, env_extra: dict | None = None) -> dict:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text(config_text, encoding="utf-8")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MNEMOSYNE_")
    }
    env.pop("HERMES_HOME", None)
    env.pop("OPENAI_API_KEY", None)
    env["MNEMOSYNE_DATA_DIR"] = str(data_dir)
    env["HOME"] = str(tmp_path / "home")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    if env_extra:
        env.update(env_extra)
    script = (
        "import json\n"
        "from mnemosyne.core import embeddings as E\n"
        "print(json.dumps({\n"
        "  'model': E._DEFAULT_MODEL,\n"
        "  'dim': E.EMBEDDING_DIM,\n"
        "  'disabled': E._is_disabled(),\n"
        "  'api_model': E._is_api_model(E._DEFAULT_MODEL),\n"
        "  'available': E.available(),\n"
        "  'available_api': E.available_api(),\n"
        "  'base': E._effective_base_url(),\n"
        "}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_config_only_embeddings_are_honored(tmp_path):
    result = _run(
        tmp_path,
        "\n".join(
            [
                "embedding_model: jina-embeddings-v4",
                "embedding_dim: 2048",
                "embedding_api_key: test-key",
                "embedding_api_url: https://embeddings.example.test/v1",
                "embeddings_via_api: true",
                "embeddings_off: false",
                "",
            ]
        ),
    )

    assert result == {
        "model": "jina-embeddings-v4",
        "dim": 2048,
        "disabled": False,
        "api_model": True,
        "available": True,
        "available_api": True,
        "base": "https://embeddings.example.test/v1",
    }


def test_known_model_dimension_is_not_overridden_by_the_seeded_default(tmp_path):
    result = _run(
        tmp_path,
        "embedding_model: BAAI/bge-base-en-v1.5\nembedding_dim: 384\n",
    )

    assert result["model"] == "BAAI/bge-base-en-v1.5"
    assert result["dim"] == 768


def test_env_values_apply_when_config_omits_the_key(tmp_path):
    result = _run(
        tmp_path,
        "embedding_model: BAAI/bge-small-en-v1.5\n",
        env_extra={
            "MNEMOSYNE_EMBEDDING_API_URL": "https://env.example.test/v1",
            "MNEMOSYNE_EMBEDDINGS_VIA_API": "1",
        },
    )

    assert result["base"] == "https://env.example.test/v1"
    assert result["api_model"] is True
