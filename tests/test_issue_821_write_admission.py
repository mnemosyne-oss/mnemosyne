"""Fresh-process regressions for issue #821 write admission."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERMES_SRC = ROOT / "integrations" / "hermes" / "src"


def _run(script: str, *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(env)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(HERMES_SRC), environment.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        env=environment,
        check=True,
    )


@pytest.mark.parametrize(
    ("yaml", "env_mode", "env_patterns", "expected"),
    [
        ("write_classifier: off\nignore_patterns: ''\n", "strict", "ISSUE821", True),
        ("write_classifier: warn\nignore_patterns: ISSUE821\n", "strict", "NO_MATCH", True),
        ("write_classifier: strict\nignore_patterns: ISSUE821\n", "off", "NO_MATCH", False),
    ],
)
def test_facade_write_path_honors_yaml_allow_warn_and_strict(
    tmp_path: Path,
    yaml: str,
    env_mode: str,
    env_patterns: str,
    expected: bool,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text(yaml)
    result = _run(
        """
import os
from mnemosyne.core.memory import Mnemosyne

memory = Mnemosyne(session_id="issue-821", db_path=os.environ["TEST_DB"])
try:
    memory_id = memory.remember("ISSUE821 private facade sentinel", source="user")
    count = memory.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content = ?",
        ("ISSUE821 private facade sentinel",),
    ).fetchone()[0]
    print(bool(memory_id) and count == 1)
finally:
    memory.conn.close()
""",
        env={
            "MNEMOSYNE_DATA_DIR": str(data_dir),
            "MNEMOSYNE_WRITE_CLASSIFIER": env_mode,
            "MNEMOSYNE_IGNORE_PATTERNS": env_patterns,
            "MNEMOSYNE_NO_EMBEDDINGS": "1",
            "TEST_DB": str(tmp_path / "facade.db"),
        },
    )
    assert result.stdout.strip() == str(expected)


_GATEWAY_SCRIPT = r"""
import importlib
import json
import os
import sys
import types
from pathlib import Path
from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation

hermes_constants = types.ModuleType("hermes_constants")
hermes_constants.get_hermes_home = lambda: Path(os.environ["HERMES_HOME"])
sys.modules.setdefault("hermes_constants", hermes_constants)

module = importlib.import_module(os.environ["PROVIDER_MODULE"])
Provider = module.MnemosyneMemoryProvider
provider = Provider()
policy_env = ("MNEMOSYNE_IGNORE_PATTERNS", "MNEMOSYNE_WRITE_CLASSIFIER")
env_before = {key: os.environ.get(key) for key in policy_env}
init_kwargs = {
    "hermes_home": os.environ["HERMES_HOME"],
    "shared_surface_path": os.environ["SHARED_DB"],
}
if os.environ["POLICY_SOURCE"] == "initialize":
    init_kwargs.update(ignore_patterns=[r"^ISSUE821"], write_classifier="strict")
elif os.environ["POLICY_SOURCE"] == "initialize_empty_list":
    init_kwargs.update(ignore_patterns=[], write_classifier="off")
elif os.environ["POLICY_SOURCE"] == "initialize_empty_string":
    init_kwargs.update(ignore_patterns="", write_classifier="off")
provider.initialize("issue-821", **init_kwargs)
assert provider._beam is not None
marker = os.environ.get("CONTENT") or (
    "ISSUE821 I feel like a private gateway sentinel"
    if os.environ["GATEWAY"] == "sync_identity"
    else "ISSUE821 private gateway sentinel"
)
response = None
original = None
validate_compatibility = None
update_not_found = None
if os.environ["GATEWAY"] == "remember":
    response = provider.handle_tool_call("mnemosyne_remember", {"content": marker})
elif os.environ["GATEWAY"] == "pending_apply":
    pending_id = module._stage_pending_write({"tool": "mnemosyne_remember", "content": marker})
    response = provider.handle_tool_call("mnemosyne_apply_pending", {"pending_ids": [pending_id]})
elif os.environ["GATEWAY"] == "shared_remember":
    response = provider.handle_tool_call(
        "mnemosyne_shared_remember", {"content": marker, "kind": "meta"}
    )
elif os.environ["GATEWAY"] == "batch":
    response = provider.handle_tool_call(
        "mnemosyne_batch",
        {"operations": [{"action": "remember", "content": marker}]},
    )
elif os.environ["GATEWAY"] in {"update", "validate_update"}:
    memory_id = provider._beam.remember(
        "allowed original", _write_policy=WritePolicySnapshot((), "off")
    )
    if os.environ["GATEWAY"] == "update":
        response = provider.handle_tool_call(
            "mnemosyne_update", {"memory_id": memory_id, "content": marker}
        )
        update_not_found = json.loads(provider.handle_tool_call(
            "mnemosyne_update", {"memory_id": "missing-id", "content": "allowed update"}
        ))
    else:
        response = provider.handle_tool_call(
            "mnemosyne_validate",
            {"memory_id": memory_id, "action": "update", "new_content": marker},
        )
        validation_rows_after_reject = provider._beam.conn.execute(
            "SELECT COUNT(*) FROM memory_validations WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0]
        no_content = json.loads(provider.handle_tool_call(
            "mnemosyne_validate", {"memory_id": memory_id, "action": "update"}
        ))
        attest = json.loads(provider.handle_tool_call(
            "mnemosyne_validate", {"memory_id": memory_id, "action": "attest"}
        ))
        validate_compatibility = {
            "validation_rows_after_reject": validation_rows_after_reject,
            "no_content_error": no_content.get("error"),
            "attest_status": attest.get("status"),
        }
    original = provider._beam.get(memory_id)["content"]
elif os.environ["GATEWAY"] == "sync_identity":
    provider.sync_turn(marker, "", session_id="issue-821")
    response = json.dumps(provider._sync_turn_diagnostics())
elif os.environ["GATEWAY"] in {"on_memory_add", "on_memory_replace"}:
    provider.on_memory_write(os.environ["GATEWAY"].removeprefix("on_memory_"), "user", marker)
elif os.environ["GATEWAY"] in {"canonical_create", "canonical_update"}:
    if os.environ["GATEWAY"] == "canonical_update":
        with write_policy_operation(WritePolicySnapshot((), "off")):
            provider._beam.canonical.remember("default", "identity", "slot", "allowed original")
    response = provider.handle_tool_call(
        "mnemosyne_remember_canonical",
        {"category": "identity", "name": "slot", "body": marker},
    )
    current = provider._beam.canonical.recall("default", "identity", "slot")
    original = current["body"] if current else None
elif os.environ["GATEWAY"] == "task_progress":
    response = provider.handle_tool_call(
        "mnemosyne_task_progress", {"action": "set", "task": "gate", "state": marker}
    )
elif os.environ["GATEWAY"] == "scratchpad":
    response = provider.handle_tool_call("mnemosyne_scratchpad_write", {"content": marker})
elif os.environ["GATEWAY"] == "triple_add":
    response = provider.handle_tool_call(
        "mnemosyne_triple_add",
        {"subject": "user", "predicate": "prefers", "object": marker},
    )
else:
    raise AssertionError(os.environ["GATEWAY"])
beams = [provider._beam]
if provider._surface_beam is not None:
    beams.append(provider._surface_beam)
count = 0
for beam in beams:
    for table_row in beam.conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        table = table_row[0]
        quoted_table = '"' + table.replace('"', '""') + '"'
        for column in beam.conn.execute(f"PRAGMA table_info({quoted_table})"):
            if "TEXT" not in str(column[2]).upper():
                continue
            quoted_column = '"' + column[1].replace('"', '""') + '"'
            count += beam.conn.execute(
                f"SELECT COUNT(*) FROM {quoted_table} WHERE {quoted_column} LIKE ?",
                ("%ISSUE821%",),
            ).fetchone()[0]
print(json.dumps({
    "count": count,
    "original": original,
    "response": json.loads(response) if response else None,
    "validate_compatibility": validate_compatibility,
    "update_not_found": update_not_found,
    "mode": provider._write_policy.classifier_mode,
    "patterns": provider._write_policy.ignore_patterns,
    "same_env": env_before == {key: os.environ.get(key) for key in policy_env},
}))
"""


@pytest.mark.parametrize("provider_module", ["hermes_memory_provider", "mnemosyne_hermes"])
@pytest.mark.parametrize(
    "gateway",
    [
        "remember", "pending_apply", "shared_remember", "batch", "update",
        "validate_update", "sync_identity", "on_memory_add", "on_memory_replace",
        "canonical_create", "canonical_update", "task_progress", "scratchpad",
        "triple_add",
    ],
)
@pytest.mark.parametrize("policy_source", ["initialize", "hermes"])
def test_every_provider_gateway_honors_provider_policy_over_conflicting_env(
    tmp_path: Path, gateway: str, provider_module: str, policy_source: str
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    if policy_source == "hermes":
        (hermes_home / "config.yaml").write_text(
            "memory:\n  mnemosyne:\n    write_classifier: strict\n"
            "    ignore_patterns: ['^ISSUE821']\n"
        )
    result = _run(
        _GATEWAY_SCRIPT,
        env={
            "GATEWAY": gateway,
            "POLICY_SOURCE": policy_source,
            "PROVIDER_MODULE": provider_module,
            "HERMES_HOME": str(hermes_home),
            "MNEMOSYNE_DATA_DIR": str(data_dir),
            "MNEMOSYNE_WRITE_CLASSIFIER": "off",
            "MNEMOSYNE_IGNORE_PATTERNS": "NO_MATCH",
            "MNEMOSYNE_NO_EMBEDDINGS": "1",
            "MNEMOSYNE_HOST_LLM_ENABLED": "0",
            "SHARED_DB": str(tmp_path / "shared.db"),
            "SECRET": "ISSUE821 private gateway sentinel",
        },
    )
    payload = json.loads(result.stdout)
    assert payload["count"] == 0
    assert payload["mode"] == "strict"
    assert payload["patterns"] == ["^ISSUE821"]
    assert payload["same_env"] is True
    if gateway in {"update", "validate_update", "canonical_update"}:
        assert payload["original"] == "allowed original"
    if gateway == "update":
        assert payload["response"]["status"] == "filtered"
        assert payload["update_not_found"] == {
            "status": "not_found", "memory_id": "missing-id"
        }
    if gateway in {"canonical_create", "canonical_update", "task_progress"}:
        assert payload["response"] == {"status": "filtered", "store": "canonical"}
    if gateway == "scratchpad":
        assert payload["response"] == {"status": "filtered", "store": "scratchpad"}
    if gateway == "triple_add":
        assert payload["response"] == {"status": "filtered"}
    if gateway == "validate_update":
        assert payload["response"]["status"] == "filtered"
        assert set(payload["response"]) <= {"status", "memory_id", "store", "bank"}
        assert payload["validate_compatibility"] == {
            "validation_rows_after_reject": 0,
            "no_content_error": "new_content is required for action='update'",
            "attest_status": "validation_attest",
        }
    assert "ISSUE821 private gateway sentinel" not in result.stderr
    assert "ISSUE821 private gateway sentinel" not in result.stdout
    if gateway == "sync_identity":
        identity_marker = "ISSUE821 I feel like a private gateway sentinel"
        assert identity_marker not in result.stderr
        assert identity_marker not in result.stdout


@pytest.mark.parametrize("provider_module", ["hermes_memory_provider", "mnemosyne_hermes"])
@pytest.mark.parametrize(
    "policy_source", ["initialize_empty_list", "initialize_empty_string", "hermes_empty"]
)
def test_empty_provider_patterns_override_conflicting_core_env(
    tmp_path: Path, provider_module: str, policy_source: str
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    if policy_source == "hermes_empty":
        (hermes_home / "config.yaml").write_text(
            "memory:\n  mnemosyne:\n    write_classifier: off\n    ignore_patterns: []\n"
        )
    result = _run(
        _GATEWAY_SCRIPT,
        env={
            "GATEWAY": "remember",
            "POLICY_SOURCE": policy_source,
            "PROVIDER_MODULE": provider_module,
            "HERMES_HOME": str(hermes_home),
            "MNEMOSYNE_DATA_DIR": str(data_dir),
            "MNEMOSYNE_WRITE_CLASSIFIER": "off",
            "MNEMOSYNE_IGNORE_PATTERNS": "^ISSUE821",
            "MNEMOSYNE_NO_EMBEDDINGS": "1",
            "MNEMOSYNE_HOST_LLM_ENABLED": "0",
            "SHARED_DB": str(tmp_path / "shared.db"),
        },
    )
    payload = json.loads(result.stdout)
    assert payload["count"] >= 1
    assert payload["patterns"] == []
    assert payload["response"]["status"] == "stored"


@pytest.mark.parametrize("provider_module", ["hermes_memory_provider", "mnemosyne_hermes"])
@pytest.mark.parametrize("write_classifier", [None, "off"])
def test_provider_patterns_do_not_enable_write_classifier(
    tmp_path: Path, provider_module: str, write_classifier: str | None
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    mode_config = (
        "    write_classifier: off\n" if write_classifier is not None else ""
    )
    (hermes_home / "config.yaml").write_text(
        "memory:\n"
        "  mnemosyne:\n"
        "    ignore_patterns: ['^DO_NOT_MATCH']\n"
        f"{mode_config}"
    )

    result = _run(
        _GATEWAY_SCRIPT,
        env={
            "GATEWAY": "remember",
            "POLICY_SOURCE": "hermes",
            "PROVIDER_MODULE": provider_module,
            "HERMES_HOME": str(hermes_home),
            "MNEMOSYNE_DATA_DIR": str(data_dir),
            "MNEMOSYNE_WRITE_CLASSIFIER": "off",
            "MNEMOSYNE_IGNORE_PATTERNS": "",
            "MNEMOSYNE_NO_EMBEDDINGS": "1",
            "MNEMOSYNE_HOST_LLM_ENABLED": "0",
            "SHARED_DB": str(tmp_path / "shared.db"),
            "CONTENT": "$ pip install ISSUE821 --quiet",
        },
    )

    payload = json.loads(result.stdout)
    assert payload["mode"] == "off"
    assert payload["patterns"] == ["^DO_NOT_MATCH"]
    assert payload["response"]["status"] == "stored"
    assert payload["count"] >= 1


def test_direct_core_and_mcp_canonical_and_scratchpad_rejections_are_atomic(
    tmp_path: Path, monkeypatch
):
    from mnemosyne import mcp_tools
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.canonical import CanonicalStore
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation
    from mnemosyne.core.memory import Mnemosyne

    marker = "ISSUE821 private direct sentinel"
    strict = WritePolicySnapshot(("^ISSUE821",), "strict")

    beam = BeamMemory(session_id="core", db_path=tmp_path / "core.db")
    store = CanonicalStore(db_path=beam.db_path, conn=beam.conn)
    store.remember("owner", "identity", "existing", "allowed original")
    with write_policy_operation(strict):
        assert store.remember("owner", "identity", "new", marker) is None
        assert store.remember("owner", "identity", "existing", marker) is None
        assert beam.scratchpad_write(marker) is None
    assert store.recall("owner", "identity", "new") is None
    assert store.recall("owner", "identity", "existing")["body"] == "allowed original"
    assert beam.scratchpad_read() == []

    memory = Mnemosyne(session_id="mcp", db_path=tmp_path / "mcp.db")
    memory.beam.canonical.remember("default", "identity", "existing", "allowed original")
    monkeypatch.setattr(mcp_tools, "_create_instance", lambda **_kwargs: memory)
    with write_policy_operation(strict):
        responses = [
            mcp_tools.handle_tool_call("mnemosyne_remember_canonical", {
                "category": "identity", "name": "new", "body": marker,
            }),
            mcp_tools.handle_tool_call("mnemosyne_remember_canonical", {
                "category": "identity", "name": "existing", "body": marker,
            }),
            mcp_tools.handle_tool_call("mnemosyne_scratchpad_write", {"content": marker}),
        ]
    assert responses == [
        {"status": "filtered", "store": "canonical"},
        {"status": "filtered", "store": "canonical"},
        {"status": "filtered", "store": "scratchpad"},
    ]
    assert marker not in json.dumps(responses)
    assert memory.beam.canonical.recall("default", "identity", "new") is None
    assert memory.beam.canonical.recall("default", "identity", "existing")["body"] == "allowed original"
    assert memory.scratchpad_read() == []


def test_only_internal_write_capabilities_are_exempt():
    from mnemosyne.core.filters import (
        _RESTORE_WRITE_CAPABILITY,
        _SYSTEM_DERIVED_WRITE_CAPABILITY,
        WritePolicySnapshot,
        admit_memory_write,
    )

    strict = WritePolicySnapshot(("ISSUE821",), "strict")
    assert admit_memory_write("ISSUE821", policy=strict)[0] is False
    assert admit_memory_write("ISSUE821", write_kind="batch", policy=strict)[0] is False
    assert admit_memory_write("ISSUE821", write_kind="restore", policy=strict)[0] is False
    assert admit_memory_write(
        "ISSUE821", write_kind="system_derived", policy=strict
    )[0] is False
    assert admit_memory_write(
        "ISSUE821", write_kind=_RESTORE_WRITE_CAPABILITY, policy=strict
    )[0] is True
    assert admit_memory_write(
        "ISSUE821", write_kind=_SYSTEM_DERIVED_WRITE_CAPABILITY, policy=strict
    )[0] is True


def test_direct_mcp_batch_updates_use_one_immutable_policy_snapshot(
    tmp_path: Path, monkeypatch
):
    from mnemosyne import mcp_tools
    from mnemosyne.core.memory import Mnemosyne

    class ChangingConfig:
        calls = 0

        def get_many(self, defaults):
            self.calls += 1
            if self.calls > 1:
                return {"ignore_patterns": "", "write_classifier": "off"}
            return {"ignore_patterns": "ISSUE821", "write_classifier": "strict"}

    config = ChangingConfig()
    memory = Mnemosyne(session_id="mcp_default", db_path=tmp_path / "batch.db")
    try:
        first = memory.remember("allowed first")
        second = memory.remember("allowed second")
        assert first is not None and second is not None
        monkeypatch.setattr("mnemosyne.core.filters.get_config", lambda: config)
        monkeypatch.setattr(mcp_tools, "_create_instance", lambda **_kwargs: memory)
        result = mcp_tools._handle_batch({"operations": [
            {"action": "update", "memory_id": first, "content": "ISSUE821 first"},
            {"action": "update", "memory_id": second, "content": "ISSUE821 second"},
        ]})
        assert [item["status"] for item in result["results"]] == ["filtered", "filtered"]
        assert config.calls == 1
        assert memory.beam.get(first)["content"] == "allowed first"
        assert memory.beam.get(second)["content"] == "allowed second"
    finally:
        memory.conn.close()
