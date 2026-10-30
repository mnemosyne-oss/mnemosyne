"""Subprocess harness; no model, host UI or production memory is accessed."""

import importlib
import json
import os
from pathlib import Path
import sys
import types


root, package, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
options = json.loads(sys.argv[4])
approval = {"value": True}
host = types.ModuleType("hermes_constants")
host.get_hermes_home = lambda: Path(os.environ["HERMES_HOME"])
host.hermes_home_key = lambda home=None: str(Path(home or host.get_hermes_home()).resolve())
cli = types.ModuleType("hermes_cli")
cli.__path__ = []
config = types.ModuleType("hermes_cli.config")
config.load_config = lambda: {"memory": {"write_approval": approval["value"]}}


def cfg_get(value, *keys, default=None):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


config.cfg_get = cfg_get
sys.modules.update({"hermes_constants": host, "hermes_cli": cli, "hermes_cli.config": config})
module = importlib.import_module(package)


def provider(name):
    home = root / name
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HERMES_HOME"] = str(home)
    os.environ["MNEMOSYNE_DATA_DIR"] = str(home / "data")
    p = module.MnemosyneMemoryProvider()
    p.initialize("approval-probe", hermes_home=str(home), profile_isolation=False)
    assert p._beam is not None, str(p._init_error)
    return p


def stage(p, content):
    result = json.loads(p.handle_tool_call("mnemosyne_remember", {"content": content, "scope": "global"}))
    assert result["status"] == "staged", result
    return result["pending_id"]


def pending_snapshot():
    return {str(p.relative_to(root)): p.read_bytes().hex() for p in root.glob("*/pending/memory/*") if p.is_file()}


def contents(p, name):
    """Read through the matching home binding, never an ambient fallback."""
    home = root / name
    os.environ["HERMES_HOME"] = str(home)
    os.environ["MNEMOSYNE_DATA_DIR"] = str(home / "data")
    beam = p._beam
    assert beam is not None
    assert Path(beam.db_path).resolve().is_relative_to(home.resolve())
    return [r[0] for r in beam.conn.execute("SELECT content FROM working_memory ORDER BY content")]


def found(p, term):
    result = json.loads(p.handle_tool_call("mnemosyne_recall", {"query": term, "limit": 5}))
    return any(term in row.get("content", "") for row in result.get("results", []))


if mode == "deny":
    a = provider("a")
    seed = a._beam.remember("original seed remains unchanged", source="fact", scope="global")
    aid = stage(a, "quasar cobalt proposal only")
    batch = json.loads(a.handle_tool_call("mnemosyne_batch", {"operations": [
        {"action": "remember", "content": "nebula bronze proposal only", "scope": "global"},
        {"action": "update", "memory_id": seed, "content": "seed changed without permission"},
    ]}))
    assert batch["status"] == "staged", batch
    b = provider("b")
    bid = stage(b, "saffron atlas proposal only")
    snapshot = pending_snapshot()
    (root / "pending-snapshot.json").write_text(json.dumps(snapshot))
    os.environ["HERMES_HOME"] = str(root / "a")
    os.environ["MNEMOSYNE_DATA_DIR"] = str(root / "a/data")
    ids = [aid, *batch["pending_ids"], bid, "../invalid"]
    args = {"pending_ids": ",".join(ids) if options["comma_ids"] else ids}
    raw = a._handle_apply_pending(args) if options["direct"] else a.handle_tool_call("mnemosyne_apply_pending", args)
    output = {"response": json.loads(raw), "pending_unchanged": pending_snapshot() == snapshot,
              "a_contents": contents(a, "a"), "b_contents": contents(b, "b")}
elif mode == "restart":
    a = provider("a")
    a_found = found(a, "quasar cobalt")
    b = provider("b")
    output = {"a_found": a_found, "b_found": found(b, "saffron atlas"),
              "pending_unchanged": pending_snapshot() == json.loads((root / "pending-snapshot.json").read_text())}
elif mode == "discovery":
    home = root / "a"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("memory:\n  mnemosyne:\n    tools: [mnemosyne_apply_pending, mnemosyne_remember]\n")
    approval["value"] = False
    p = provider("a")
    before = p.has_tool("mnemosyne_apply_pending")
    approval["value"] = options["enabled"]
    output = {"before": before, "after": p.has_tool("mnemosyne_apply_pending"), "other_tool": p.has_tool("mnemosyne_remember")}
elif mode == "off":
    p = provider("a")
    pid = stage(p, "quasar cobalt legitimate replay")
    approval["value"] = False
    result = json.loads(p.handle_tool_call("mnemosyne_apply_pending", {"pending_ids": [pid]}))
    output = {"applied": len(result.get("applied", [])), "pending_removed": not (root / "a/pending/memory" / f"{pid}.json").exists(), "found": found(p, "quasar cobalt")}
elif mode == "early":
    p = module.MnemosyneMemoryProvider()
    def unavailable_home():
        raise AssertionError("blocked replay attempted to resolve a pending home")
    host.get_hermes_home = unavailable_home
    output = json.loads(p._handle_apply_pending(None))
elif mode == "messages":
    p = provider("a")
    remember = json.loads(p.handle_tool_call("mnemosyne_remember", {"content": "quasar cobalt staging"}))
    batch = json.loads(p.handle_tool_call("mnemosyne_batch", {"operations": [{"action": "remember", "content": "nebula bronze staging"}]}))
    output = {"staged": [remember["status"], batch["status"]], "messages": [remember["message"], batch["message"]]}
else:
    raise AssertionError(mode)
print("RESULT " + json.dumps(output))
