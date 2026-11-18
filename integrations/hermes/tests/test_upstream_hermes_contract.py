"""Contract test against an exact upstream Hermes checkout.

Set HERMES_AGENT_SOURCE to a checkout at HERMES_AGENT_REVISION. The pin makes
contract changes explicit instead of silently testing whichever Hermes happens
to be installed on a contributor's machine.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


HERMES_AGENT_REVISION = "c661785f872b5647fbac7c138d965180783bd9af"
PROJECT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = PROJECT / "integrations" / "hermes" / "src" / "mnemosyne_hermes"


def _hermes_source() -> Path:
    configured = os.environ.get("HERMES_AGENT_SOURCE")
    if not configured:
        pytest.skip(
            "set HERMES_AGENT_SOURCE to a Hermes checkout at "
            f"{HERMES_AGENT_REVISION} to run the pinned upstream contract"
        )
    assert configured is not None
    return Path(configured).resolve()


def test_pinned_upstream_setup_schema_and_status_contract(tmp_path):
    hermes_source = _hermes_source()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=hermes_source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == HERMES_AGENT_REVISION

    script = r'''
import contextlib
import io
import os
import sys
from pathlib import Path

hermes_source = Path(os.environ["CONTRACT_HERMES_SOURCE"])
package_dir = Path(os.environ["CONTRACT_PACKAGE_DIR"])
hermes_home = Path(os.environ["CONTRACT_HERMES_HOME"])
sys.path[:0] = [str(hermes_source), str(package_dir.parent)]

import hermes_cli.config as hermes_config
import hermes_cli.memory_setup as memory_setup
import hermes_cli.tools_config as tools_config
import plugins.memory as memory_plugins
import plugins.memory.config_schema as declared_schema
import tools.memory_tool as memory_tool
from mnemosyne_hermes import MnemosyneMemoryProvider

# Hermes' declared schema has no config.yaml-backed storage. If Mnemosyne
# declared fields, current Hermes would write $HERMES_HOME/mnemosyne/config.json.
router_source = (
    hermes_source / "hermes_cli" / "web_routers" / "memory_providers.py"
).read_text(encoding="utf-8")
assert 'return get_hermes_home() / provider.name / "config.json"' in router_source
assert "STORAGE_FLAT_JSON" in declared_schema.__dict__
assert "STORAGE_HONCHO_HOST_BLOCK" in declared_schema.__dict__
memory_plugins.find_provider_dir = lambda _name: package_dir
declared_schema._SCHEMA_CACHE.clear()
assert declared_schema.get_provider_config_schema("mnemosyne") is None

provider = MnemosyneMemoryProvider()
schema = memory_setup._schema_of(provider)
assert len(schema) == 12
assert {field["key"] for field in schema} >= {"default_scope", "profile_isolation", "tools"}

config_path = hermes_home / "config.yaml"
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(
    "memory:\n  provider: mnemosyne\n  mnemosyne:\n    existing: keep\n",
    encoding="utf-8",
)
provider.save_config({"default_scope": "global"}, str(hermes_home))
text = config_path.read_text(encoding="utf-8")
assert "provider: mnemosyne" in text
assert "existing: keep" in text
assert "default_scope: global" in text
assert not (hermes_home / "mnemosyne" / "config.json").exists()

malicious = {
    "memory": {
        "provider": "mnemosyne",
        "memory_enabled": True,
        "user_profile_enabled": True,
        "mnemosyne": {
            "default_scope": "global",
            "shared_surface_path": "safe\x1b[2Jspoofed",
            "tools": [
                "mnemosyne_recall",
                "mnemosyne_not_real",
                "mnemosyne_forged\x1b[31m" + "x" * 20000,
            ] + ["mnemosyne_remember"] * 1000,
        },
    }
}
hermes_config.load_config = lambda: malicious
tools_config._get_platform_tools = lambda *_args, **_kwargs: ["memory"]
memory_tool.check_memory_requirements = lambda: True
provider.is_available = lambda: True
memory_setup._get_available_providers = lambda: [("mnemosyne", "test", provider)]

stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    memory_setup.cmd_status(None)
output = stdout.getvalue()
assert "mnemosyne_recall" in output
assert "mnemosyne_remember" in output
assert "mnemosyne_not_real" not in output
assert "mnemosyne_forged" not in output
assert "spoofed" not in output
assert "\x1b" not in output
assert len(output) < 4096
assert not (hermes_home / "mnemosyne" / "config.json").exists()
'''
    env = os.environ.copy()
    env.update(
        {
            "CONTRACT_HERMES_SOURCE": str(hermes_source),
            "CONTRACT_PACKAGE_DIR": str(PACKAGE_DIR),
            "CONTRACT_HERMES_HOME": str(tmp_path),
            "HERMES_HOME": str(tmp_path),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
