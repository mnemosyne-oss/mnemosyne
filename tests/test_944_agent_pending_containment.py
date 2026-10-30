"""Real-provider subprocess checks for the #944 agent approval boundary."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("approval_containment_worker.py")


def run_worker(tmp_path, package, mode, **values):
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "integrations/hermes/src")]),
        HERMES_HOME=str(tmp_path / "a"),
        MNEMOSYNE_DATA_DIR=str(tmp_path / "a/data"),
        MNEMOSYNE_NO_EMBEDDINGS="1",
        MNEMOSYNE_HOST_LLM_ENABLED="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    result = subprocess.run(
        [sys.executable, str(WORKER), str(tmp_path), package, mode, json.dumps(values)],
        env=env, capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(next(line[7:] for line in result.stdout.splitlines() if line.startswith("RESULT ")))


@pytest.mark.parametrize("package", ["hermes_memory_provider", "mnemosyne_hermes"])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("comma_ids", [False, True])
def test_agent_apply_preserves_both_homes_and_restart(tmp_path, package, direct, comma_ids):
    """Removing the handler/dispatch guard would commit and consume proposals."""
    result = run_worker(tmp_path, package, "deny", direct=direct, comma_ids=comma_ids)
    assert result["response"]["status"] == "approval_required", result
    assert result["response"]["error"] == "agent_pending_replay_disabled", result
    assert result["pending_unchanged"] is True
    assert result["a_contents"] == ["original seed remains unchanged"]
    assert result["b_contents"] == []
    restarted = run_worker(tmp_path, package, "restart")
    assert restarted == {"a_found": False, "b_found": False, "pending_unchanged": True}


@pytest.mark.parametrize("package", ["hermes_memory_provider", "mnemosyne_hermes"])
@pytest.mark.parametrize("enabled", [True, "on", "true", "yes", "1", "approve", "enabled"])
def test_discovery_respects_dynamic_approval_and_explicit_whitelist(tmp_path, package, enabled):
    """An explicit tools whitelist must not restore an approval-blocked tool."""
    result = run_worker(tmp_path, package, "discovery", enabled=enabled)
    assert result == {"before": True, "after": False, "other_tool": True}


@pytest.mark.parametrize("package", ["hermes_memory_provider", "mnemosyne_hermes"])
def test_approval_off_replay_still_commits_existing_record(tmp_path, package):
    """The containment must not remove explicitly approval-OFF replay."""
    result = run_worker(tmp_path, package, "off")
    assert result == {"applied": 1, "pending_removed": True, "found": True}


@pytest.mark.parametrize("package", ["hermes_memory_provider", "mnemosyne_hermes"])
def test_direct_denial_precedes_pending_argument_and_home_resolution(tmp_path, package):
    """Malformed arguments must not bypass denial or enter file processing."""
    result = run_worker(tmp_path, package, "early")
    assert result["status"] == "approval_required", result
    assert result["error"] == "agent_pending_replay_disabled", result


@pytest.mark.parametrize("package", ["hermes_memory_provider", "mnemosyne_hermes"])
def test_staging_does_not_instruct_the_agent_to_self_apply(tmp_path, package):
    """Remember and batch responses must not advertise the forbidden replay."""
    result = run_worker(tmp_path, package, "messages")
    assert result["staged"] == ["staged", "staged"]
    assert all("mnemosyne_apply_pending" not in message for message in result["messages"])
