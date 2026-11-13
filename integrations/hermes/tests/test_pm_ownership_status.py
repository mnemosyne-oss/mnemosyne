"""Owned synthetic metadata only; layout from pinned Hermes PM workspace/facts writers."""
import base64
from contextlib import contextmanager
import csv
import hashlib
import importlib.util

import json
import os
from pathlib import Path

import subprocess
import sys

import pytest
import yaml
import _mnemosyne_hermes_cli as cli

requires_tomllib = pytest.mark.skipif(
    importlib.util.find_spec("tomllib") is None,
    reason="PM TOML metadata comparisons require the documented stdlib tomllib reader",
)

def module():
    if "_mnemosyne_hermes_installer" not in sys.modules:
        with pytest.raises(SystemExit) as exc:
            cli.main(["--help"])
        assert exc.value.code == 0
    return sys.modules["_mnemosyne_hermes_installer"]


def distribution(site, name, package, version, extras=()):
    info = site / (name.replace("-", "_") + "-" + version + ".dist-info")
    info.mkdir(parents=True)
    (site / package).mkdir(exist_ok=True)
    files = {info / "METADATA": f"Name: {name}\nVersion: {version}\n" + "".join(f"Provides-Extra: {e}\n" for e in extras), info / "WHEEL": "Wheel-Version: 1.0\n", site / package / "__init__.py": "raise AssertionError('package must not execute')\n"}
    rows = []
    for path, text in files.items():
        path.write_text(text)
        data = text.encode()
        rows.append([path.relative_to(site).as_posix(), "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("="), str(len(data))])
    with (info / "RECORD").open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)


@pytest.fixture
def recorded(tmp_path):
    home = tmp_path / "home"
    candidate = home / ".mnemosyne-pm/mnemosyne"
    candidate.mkdir(parents=True)
    req = ["mnemosyne-memory[embeddings]==3.15.1", "mnemosyne-hermes==0.5.0"]
    (candidate / "__init__.py").write_text("# register_memory_provider\nfrom mnemosyne_hermes import *\n")
    (candidate / "plugin.yaml").write_text(yaml.safe_dump({"name": "hermes-mnemosyne", "version": "0.5.0", "python_dependencies": req}))
    target = home / "plugins/mnemosyne"
    target.parent.mkdir()
    target.symlink_to(candidate)
    project = home / "hermes-agent"
    project.mkdir()
    state = home / "installs" / hashlib.sha256(str(project).encode()).hexdigest()[:16]
    generation = state / "environments" / ("a" * 32)
    env = generation / "venv"
    env.mkdir(parents=True)
    (env / "pyvenv.cfg").write_text("home = /synthetic\n")
    site = env / "lib/python3.14/site-packages"
    site.mkdir(parents=True)
    distribution(site, "mnemosyne-memory", "mnemosyne", "3.15.1", ["embeddings"])
    distribution(site, "mnemosyne-hermes", "mnemosyne_hermes", "0.5.0")
    workspace = generation / "workspace"
    key = "mnemosyne-" + hashlib.sha256(str(candidate).encode()).hexdigest()[:16]
    member = workspace / "plugin-deps" / key
    member.mkdir(parents=True)
    (member / "pyproject.toml").write_text(f'[project]\nname = "hermes-plugin-{key}"\nversion = "0.0.0"\nrequires-python = ">=3.11"\ndependencies = {json.dumps(req)}\n[tool.uv]\npackage = false\n')
    (workspace / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ' + json.dumps(["plugin-deps/" + key]))
    # Standard uv lock package array; synthetic selected versions, no resolver run.
    (workspace / "uv.lock").write_text('version = 1\n' + ''.join(
        f'\n[[package]]\nname = "{name}"\nversion = "{version}"\nsource = {{ registry = "https://pypi.org/simple" }}\n'
        for name, version in [("mnemosyne-memory", "3.15.1"), ("mnemosyne-hermes", "0.5.0")]))
    facts = state / "facts.json"
    facts.write_text(json.dumps({"schema": 1, "packages": {"venv": {"stamp": "b" * 64, "extras": ["cron"], "environment": str(env), "resolved_lock": str(workspace / "uv.lock")}}}))
    return home, facts, site, candidate, target


def snapshot(root):
    return {str(p.relative_to(root)): (p.lstat().st_mode, os.readlink(p) if p.is_symlink() else p.read_bytes() if p.is_file() else None) for p in [root, *root.rglob("*")]}


@pytest.mark.parametrize("parser_mode", ["actual", "blocked"])
def test_complete_repeated_corefree_readonly(recorded, parser_mode):
    home, _, site, _, _ = recorded
    # No site activation or imports of these executable traps are permitted.
    (site / "trap.pth").write_text("raise AssertionError('pth activated')\n")
    (home / "mnemosyne").mkdir()
    (home / "mnemosyne/only-data.txt").write_text("not a Core installation")
    before = snapshot(home)
    # Collection may import the provider in the parent. Never weaken the census:
    # audit the public loader in a fresh process with an explicit owned env.
    env_root = home.parent / "child-env"
    env = {"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    for name in ("HOME", "HERMES_HOME", "MNEMOSYNE_HOME", "MNEMOSYNE_DATA_DIR",
                 "MNEMOSYNE_FLEET_ROOT", "TMPDIR", "XDG_CACHE_HOME", "UV_CACHE_DIR"):
        directory = env_root / name.lower()
        directory.mkdir(parents=True)
        env[name] = str(directory)
    code = r'''
import builtins, contextlib, importlib.abc, importlib.util, io, os, socket, sqlite3, subprocess, sys
from pathlib import Path
from unittest.mock import patch
source, home = Path(sys.argv[1]), Path(sys.argv[2])
parser_mode = sys.argv[3]
parser_available = importlib.util.find_spec("tomllib") is not None
sys.path.insert(0, str(source))
import _mnemosyne_hermes_cli as cli
with contextlib.redirect_stdout(io.StringIO()):
    try:
        cli.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
mod = sys.modules["_mnemosyne_hermes_installer"]
assert Path(cli.__file__).resolve() == source / "_mnemosyne_hermes_cli.py"
assert Path(mod.__file__).resolve() == source / "mnemosyne_hermes/install.py"
def census():
    assert not any(n.split(".")[0] in {"mnemosyne", "mnemosyne_hermes", "pm"} for n in sys.modules)
def snapshot():
    return {str(p.relative_to(home)): (p.lstat().st_mode, os.readlink(p) if p.is_symlink() else p.read_bytes() if p.is_file() else None) for p in [home, *home.rglob("*")]}
def forbidden(*args, **kwargs):
    raise AssertionError("forbidden status side effect")
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == "tomllib" and parser_mode == "blocked":
            raise ModuleNotFoundError("tomllib intentionally unavailable in test")
        if fullname.split(".")[0] in {"mnemosyne", "mnemosyne_hermes", "pm"}:
            forbidden()
census()
before = snapshot()
if parser_mode == "blocked":
    sys.modules.pop("tomllib", None)
sys.meta_path.insert(0, Block())
with contextlib.ExitStack() as stack:
    for obj, name in [(subprocess, "Popen"), (subprocess, "run"), (socket, "socket"), (socket, "create_connection"), (sqlite3, "connect"), (os, "mkdir"), (os, "rename"), (os, "replace"), (os, "unlink"), (os, "remove"), (os, "chmod"), (os, "utime"), (os, "system"), (mod, "check_mnemosyne_core"), (mod, "_find_hermes_python"), (mod, "_sync_adoption_pm"), (mod, "plugin_state")]:
        stack.enter_context(patch.object(obj, name, forbidden))
    original_os_open = os.open
    def ro_os_open(path, flags, *args, **kwargs):
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        return original_os_open(path, flags, *args, **kwargs)
    stack.enter_context(patch.object(os, "open", ro_os_open))
    for obj in (builtins, io):
        original = obj.open
        def ro_open(file, mode="r", *args, _original=original, **kwargs):
            assert not any(c in mode for c in "wax+")
            return _original(file, mode, *args, **kwargs)
        stack.enter_context(patch.object(obj, "open", ro_open))
    outputs = []
    for _ in range(2):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            assert cli.main(["--hermes-home", str(home), "status"]) == 0
        outputs.append(output.getvalue())
    assert outputs[0] == outputs[1]
    if parser_available and parser_mode == "actual":
        assert "adoption complete (METADATA ONLY)" in outputs[0]
    else:
        assert "Ownership: unknown" in outputs[0]
        assert "Package metadata consistency: unknown" in outputs[0]
        assert "tomllib" in outputs[0]
        assert "adoption complete" not in outputs[0]
    assert "Runtime/provider function/hooks/autosave: NOT CHECKED" in outputs[0]
    absent = home.parent / "absent"
    assert not absent.exists()
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli.main(["--hermes-home", str(absent), "status"]) == 1
    assert not absent.exists()
assert snapshot() == before
census()
print("fresh-process Core-free status and read-only guards passed")
'''
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code,
         str(Path(cli.__file__).resolve().parent), str(home), parser_mode],
        cwd=home, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fresh-process Core-free status and read-only guards passed" in result.stdout
    assert snapshot(home) == before


@pytest.mark.parametrize("change,ownership,reason", [
    ("no_facts", "incomplete", "alone"),
    pytest.param("old_binding", "incomplete", "binding mismatch", marks=requires_tomllib),
    pytest.param("missing_dist", "unknown", "distribution metadata", marks=requires_tomllib),
    pytest.param("mismatch", "incomplete", "metadata mismatch", marks=requires_tomllib),
    ("schema", "unknown", "schema"), ("partial", "unknown", "environment"),
    pytest.param("broken", "unknown", "No such file", marks=requires_tomllib),
    ("unsafe", "unknown", "symlink"),
    pytest.param("candidate_changed", "incomplete", "requirements mismatch", marks=requires_tomllib),
])
def test_bad_metadata_never_green(recorded, change, ownership, reason):
    home, facts, site, candidate, target = recorded
    mod = module()
    if change == "no_facts": facts.unlink()
    elif change == "old_binding":
        prior = home / "prior/mnemosyne_hermes"
        prior.mkdir(parents=True)
        (prior / "__init__.py").write_text("# register_memory_provider\n")
        target.unlink(); target.symlink_to(prior)
    elif change == "missing_dist": (site / "mnemosyne_memory-3.15.1.dist-info/METADATA").unlink()
    elif change == "mismatch":
        import shutil
        shutil.rmtree(site / "mnemosyne_memory-3.15.1.dist-info")
        distribution(site, "mnemosyne-memory", "mnemosyne", "3.15.2", ["embeddings"])
    elif change == "schema":
        data = json.loads(facts.read_text()); data["schema"] = 9; facts.write_text(json.dumps(data))
    elif change == "partial": facts.write_text('{"schema":1,"packages":{"venv":{}}}')
    elif change == "broken": (site.parents[2] / "pyvenv.cfg").unlink()
    elif change == "unsafe":
        data = facts.read_text(); facts.unlink(); other = home / "other.json"; other.write_text(data); facts.symlink_to(other)
    elif change == "candidate_changed":
        data = yaml.safe_load((candidate / "plugin.yaml").read_text()); data["python_dependencies"] = []; (candidate / "plugin.yaml").write_text(yaml.safe_dump(data))
    result = mod._metadata_status(home)
    assert result["ownership"] == ownership, result
    assert reason in result["reason"] or reason in result["consistency"], result


def test_missing_home_stays_absent(tmp_path):
    home = tmp_path / "absent"
    assert module()._metadata_status(home)["plugin"] == "missing"
    assert not home.exists()


def test_external_metadata_is_legitimate(tmp_path):
    target = tmp_path / "plugins/mnemosyne"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("# register_memory_provider\n")
    (target / "mnemosyne-wrapper.json").write_text(json.dumps({"schema_version": 1, "package": "mnemosyne_hermes", "python": "/explicit/external/bin/python", "site_packages": "/explicit/external/lib/site-packages"}))
    result = module()._metadata_status(tmp_path)
    assert result["ownership"] == "external"
    assert "not checked" in result["reason"]


def test_verified_legacy(recorded):
    home, facts, site, _, target = recorded
    facts.unlink(); target.unlink(); target.symlink_to(site / "mnemosyne_hermes")
    assert module()._metadata_status(home)["ownership"] == "legacy"


def test_external_wrapper_with_pm_facts(recorded):
    home, facts, _, _, target = recorded
    target.unlink(); target.mkdir()
    (target / "__init__.py").write_text("# register_memory_provider\n")
    (target / "mnemosyne-wrapper.json").write_text(json.dumps({"schema_version": 1, "package": "mnemosyne_hermes", "python": "/external/bin/python", "site_packages": "/external/lib/site-packages"}))
    assert facts.exists()
    result = module()._metadata_status(home)
    assert result["ownership"] == "external", result
    assert "not active for this binding" in result["reason"]


@requires_tomllib
@pytest.mark.parametrize("layout", ["posix", "windows"])
def test_literal_directory_layout(recorded, layout):
    home, _, site, _, _ = recorded
    if layout == "windows":
        # Remove lowercase tree first: this also works on case-insensitive hosts.
        import shutil
        saved = home / "saved-site"
        shutil.move(str(site), saved)
        shutil.rmtree(site.parents[1])
        destination = site.parents[2] / "Lib/site-packages"
        destination.parent.mkdir()
        shutil.move(saved, destination)
    assert module()._metadata_status(home)["ownership"] == "pm_managed"


@requires_tomllib
@pytest.mark.parametrize("bad", ["empty", "malformed", "conflict", "duplicate", "missing"])
def test_lock_must_record_exact_pair(recorded, bad):
    home, facts, _, _, _ = recorded
    lock = Path(json.loads(facts.read_text())["packages"]["venv"]["resolved_lock"])
    if bad == "empty": lock.write_text("version = 1\n")
    elif bad == "malformed": lock.write_text("[[package\n")
    elif bad == "conflict": lock.write_text(lock.read_text().replace("3.15.1", "3.15.2"))
    elif bad == "duplicate": lock.write_text(lock.read_text() + '\n[[package]]\nname = "mnemosyne-memory"\nversion = "3.15.1"\n')
    elif bad == "missing": lock.write_text('version = 1\npackage = []\n')
    result = module()._metadata_status(home)
    assert result["ownership"] == "unknown", result
    assert "complete" not in result["consistency"]


@requires_tomllib
@pytest.mark.parametrize("text", ["[invalid", "name: a\nname: b\n", "- not-a-mapping", "!!python/object:evil {}"])
def test_malformed_yaml_is_unknown(recorded, text):
    home, _, _, candidate, _ = recorded
    (candidate / "plugin.yaml").write_text(text)
    assert module()._metadata_status(home)["ownership"] == "unknown"


@requires_tomllib
@pytest.mark.parametrize("dependencies", [["mnemosyne-hermes==0.5.0"] * 2, {"mnemosyne-hermes": "0.5.0"}, [42]])
def test_invalid_dependencies(recorded, dependencies):
    home, facts, _, _, _ = recorded
    workspace = Path(json.loads(facts.read_text())["packages"]["venv"]["resolved_lock"]).parent
    member = next((workspace / "plugin-deps").iterdir()) / "pyproject.toml"
    text = member.read_text().split("dependencies = ")[0]
    member.write_text(text + "dependencies = " + json.dumps(dependencies))
    assert module()._metadata_status(home)["ownership"] == "unknown"


@requires_tomllib
def test_duplicate_distribution_metadata(recorded):
    home, _, site, _, _ = recorded
    metadata = site / "mnemosyne_memory-3.15.1.dist-info/METADATA"
    metadata.write_text(metadata.read_text() + "Version: 3.15.1\n")
    assert module()._metadata_status(home)["ownership"] == "unknown"


@requires_tomllib
def test_atime_only_mock_keeps_actual_status_complete(recorded, monkeypatch, capsys):
    """Controlled mock, not evidence of an observed native relatime update."""
    home, facts, *_ = recorded
    mod = module()
    assert mod._metadata_status(home)["ownership"] == "pm_managed"
    original = Path.stat
    original_open = Path.open
    sampled = False
    changed_after_read = False
    def atime_only(path, *args, **kwargs):
        nonlocal changed_after_read
        value = original(path, *args, **kwargs)
        if path == facts:
            row = list(value)
            if sampled:
                row[7] += 1
                changed_after_read = True
            return os.stat_result(row)
        return value
    @contextmanager
    def opened(path, mode="r", *args, **kwargs):
        nonlocal sampled
        with original_open(path, mode, *args, **kwargs) as stream:
            yield stream
        if path == facts and mode == "rb":
            sampled = True
    monkeypatch.setattr(Path, "stat", atime_only)
    monkeypatch.setattr(Path, "open", opened)
    capsys.readouterr()
    assert cli.main(["--hermes-home", str(home), "status"]) == 0
    assert "adoption complete (METADATA ONLY)" in capsys.readouterr().out
    assert sampled and changed_after_read


@requires_tomllib
@pytest.mark.parametrize("field", [0, 1, 2, 4, 5, 6, 8, 9])
def test_substantive_stat_change_mock_still_refused(recorded, monkeypatch, field):
    """Controlled stat change AFTER the first facts read, not incidental stat calls."""
    home, facts, *_ = recorded
    mod = module()
    assert mod._metadata_status(home)["ownership"] == "pm_managed"
    original = Path.stat
    original_open = Path.open
    sampled = False
    changed_after_read = False
    def changed(path, *args, **kwargs):
        nonlocal changed_after_read
        value = original(path, *args, **kwargs)
        if path == facts:
            row = list(value)
            if sampled:
                row[field] += 1
                changed_after_read = True
            return os.stat_result(row)
        return value
    @contextmanager
    def opened(path, mode="r", *args, **kwargs):
        nonlocal sampled
        with original_open(path, mode, *args, **kwargs) as stream:
            yield stream
        if path == facts and mode == "rb":
            sampled = True
    monkeypatch.setattr(Path, "stat", changed)
    monkeypatch.setattr(Path, "open", opened)
    result = mod._metadata_status(home)
    assert sampled and changed_after_read
    assert result["ownership"] == "unknown", result
    assert "metadata changed" in result["reason"]


@requires_tomllib
def test_real_content_change_between_reads_still_refused(recorded, monkeypatch):
    """Fixture injects an actual byte change after the first facts read."""
    from contextlib import contextmanager
    home, facts, *_ = recorded
    mod = module()
    assert mod._metadata_status(home)["ownership"] == "pm_managed"
    original = Path.open
    before = facts.read_bytes()
    after = before.replace(b'"bbbb', b'"cbbb', 1)
    assert after != before and len(after) == len(before)
    changed = False
    @contextmanager
    def opened(path, mode="r", *args, **kwargs):
        nonlocal changed
        with original(path, mode, *args, **kwargs) as stream:
            yield stream
        if path == facts and mode == "rb" and not changed:
            changed = True
            with original(facts, "wb") as stream:
                stream.write(after)
    monkeypatch.setattr(Path, "open", opened)
    result = mod._metadata_status(home)
    assert changed
    assert result["ownership"] == "unknown", result
    assert "metadata changed" in result["reason"]


@requires_tomllib
@pytest.mark.parametrize("alias_kind", ["home", "ancestor"])
def test_selected_home_alias_preserves_canonical_pm_key(recorded, alias_kind):
    home, facts, *_ = recorded
    before = snapshot(home)
    alias = home.parent / "selected-alias"
    alias.symlink_to(home if alias_kind == "home" else home.parent)
    selected = alias if alias_kind == "home" else alias / home.name
    result = module()._metadata_status(selected)
    assert result["ownership"] == "pm_managed", result
    assert result["path"] == str(home / "plugins/mnemosyne")
    assert facts.parent.name == hashlib.sha256(str(home / "hermes-agent").encode()).hexdigest()[:16]
    assert snapshot(home) == before


@pytest.mark.parametrize("component", ["directory", "file"])
def test_internal_metadata_links_still_refused(recorded, component):
    home, facts, *_ = recorded
    if component == "file":
        saved = home / "saved-facts.json"
        facts.rename(saved); facts.symlink_to(saved)
    else:
        installs = home / "installs"
        saved = home / "saved-installs"
        installs.rename(saved); installs.symlink_to(saved)
    result = module()._metadata_status(home)
    assert result["ownership"] == "unknown", result
    assert "symlink" in result["reason"]


@pytest.mark.parametrize("dangling", [False, True])
def test_outside_binding_present_without_destination_inspection(tmp_path, monkeypatch, capsys, dangling):
    home = tmp_path / "home"
    target = home / "plugins/mnemosyne"
    target.parent.mkdir(parents=True)
    foreign = tmp_path / "foreign"
    if not dangling:
        foreign.mkdir()
        (foreign / "__init__.py").write_text("raise AssertionError('must not import')")
    target.symlink_to(foreign)
    mod = module()
    def forbidden(*args, **kwargs):
        raise AssertionError("status probe forbidden")
    for name in ("_find_hermes_python", "check_mnemosyne_core", "_check_wrapper_import", "_sync_adoption_pm"):
        monkeypatch.setattr(mod, name, forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    for name in ("exists", "stat", "lstat", "open", "read_bytes", "is_symlink", "resolve"):
        original = getattr(Path, name)
        def guarded(path, *args, _original=original, **kwargs):
            assert not path.is_relative_to(foreign), "foreign destination inspected"
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(Path, name, guarded)
    capsys.readouterr()
    assert cli.main(["--hermes-home", str(home), "status"]) == 0
    out = capsys.readouterr()
    assert not out.err
    assert "binding present (destination not inspected)" in out.out
    assert "Ownership: unknown" in out.out
    assert "NOT CHECKED" in out.out
    assert "Ownership: legacy" not in out.out and "pm_managed" not in out.out


@pytest.mark.parametrize("broken", [False, True])
def test_missing_and_in_home_broken_binding_are_distinct(tmp_path, broken):
    home = tmp_path / "home"
    target = home / "plugins/mnemosyne"
    target.parent.mkdir(parents=True)
    if broken:
        target.symlink_to(home / "missing-package")
    assert module()._metadata_status(home)["plugin"] == ("broken_symlink" if broken else "missing")


@requires_tomllib
@pytest.mark.parametrize("fault", ["unreadable", "oversize", "directory", "file_link", "parent_link", "recheck_failure"])
def test_skill_fault_is_independent_of_verified_plugin(recorded, monkeypatch, capsys, fault):
    home, *_ = recorded
    mod = module()
    skill = mod.skill_target_file(home)
    skill.parent.mkdir(parents=True)
    if fault == "directory":
        skill.mkdir()
    elif fault == "file_link":
        skill.symlink_to(home / "absent-skill")
    elif fault == "parent_link":
        skill.parent.rmdir()
        skill.parent.symlink_to(home / "absent-skills")
    else:
        skill.write_text("x" * (1024 * 1024 + 1) if fault == "oversize" else "synthetic skill")
    if fault in {"unreadable", "recheck_failure"}:
        original = Path.open
        calls = 0
        def failing(path, mode="r", *args, **kwargs):
            nonlocal calls
            if path == skill:
                calls += 1
                if fault == "unreadable" or calls > 1:
                    raise PermissionError("synthetic skill denial\x1b[2J")
            return original(path, mode, *args, **kwargs)
        monkeypatch.setattr(Path, "open", failing)
    result = mod._metadata_status(home)
    assert result["ownership"] == "pm_managed", result
    assert result["plugin"] == "present (metadata only)"
    assert result["skill"] == "unknown"
    assert result["skill_reason"]
    capsys.readouterr()
    assert cli.main(["--hermes-home", str(home), "status"]) == 0
    output = capsys.readouterr().out
    assert "Skill reason:" in output
    assert "\x1b" not in output


def test_selected_home_resolve_failure_is_unknown(tmp_path, monkeypatch):
    mod = module()
    def failed(*args, **kwargs):
        raise OSError("synthetic selected-home resolve failure")
    monkeypatch.setattr(Path, "resolve", failed)
    result = mod._metadata_status(tmp_path)
    assert result["ownership"] == "unknown"
    assert "resolve failure" in result["reason"]
