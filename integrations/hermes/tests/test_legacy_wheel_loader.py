"""Real Hermes discovery across the legacy per-file plugin layout (#1095).

Run with HERMES_AGENT_SOURCE set to a Hermes checkout and
MNEMOSYNE_LEGACY_WHEELS set to a directory containing published 0.5.0 and
0.7.0 provider wheels plus the compatible 3.15.1 core wheel. The ordinary
unit suite skips this network-free, opt-in wheel integration test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX legacy symlink layout")
def test_dangling_legacy_plugin_link_refuses_without_install(tmp_path):
    guide = (Path(__file__).resolve().parents[3] / "docs" / "hermes-integration.md").read_text()
    section = guide.split("#### Recover a legacy per-file plugin directory", 1)[1]
    script = section.split("```bash\n", 1)[1].split("\n```", 1)[0]

    package = tmp_path / "site" / "mnemosyne_hermes"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    home = tmp_path / "selected-home"
    target = home / "plugins" / "mnemosyne"
    target.mkdir(parents=True)
    initializer = target / "__init__.py"
    initializer.symlink_to(package / "__init__.py")
    custom = target / "custom.txt"
    custom.write_text("custom plugin data")
    dangling = home / "plugins" / "hermes-mnemosyne"
    missing = tmp_path / "custom-plugin" / "missing"
    dangling.symlink_to(missing)
    backup = home / "mnemosyne-legacy-plugin-backup"

    installer_marker = tmp_path / "installer-reached"
    python_wrapper = tmp_path / "gateway-python"
    python_wrapper.write_text(
        '#!/bin/sh\n'
        'if [ "$1" = "-m" ] && [ "$2" = "mnemosyne_hermes.install" ]; then\n'
        '  : > "$INSTALL_MARKER"\n'
        '  exit 97\n'
        'fi\n'
        'exec "$REAL_PYTHON" "$@"\n'
    )
    python_wrapper.chmod(0o755)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "HERMES_HOME": str(home),
        "HERMES_PYTHON": str(python_wrapper),
        "MNEMOSYNE_DATA_DIR": str(tmp_path / "data"),
        "MNEMOSYNE_NO_EMBEDDINGS": "1",
        "PYTHONPATH": str(package.parent),
        "REAL_PYTHON": sys.executable,
        "INSTALL_MARKER": str(installer_marker),
    }
    refused = subprocess.run(
        ["bash", "-c", script], env=env, cwd=tmp_path,
        text=True, capture_output=True, timeout=90,
    )
    assert refused.returncode != 0, (refused.stdout, refused.stderr)
    assert "Old plugin path also exists; inspect it first" in refused.stderr
    assert not installer_marker.exists()
    assert target.is_dir() and not target.is_symlink()
    assert initializer.is_symlink() and initializer.resolve() == package / "__init__.py"
    assert custom.read_text() == "custom plugin data"
    assert dangling.is_symlink() and os.readlink(dangling) == str(missing)
    assert not backup.exists() and not backup.is_symlink()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX legacy symlink layout")
def test_legacy_wheel_upgrade_and_owned_layout_recovery(tmp_path):
    source = os.environ.get("HERMES_AGENT_SOURCE")
    wheel_dir = os.environ.get("MNEMOSYNE_LEGACY_WHEELS")
    if not source or not wheel_dir:
        pytest.skip("set HERMES_AGENT_SOURCE and MNEMOSYNE_LEGACY_WHEELS")
    source = Path(source).resolve()
    wheels = Path(wheel_dir).resolve()
    assert (source / "plugins" / "memory" / "__init__.py").is_file()

    def wheel(project, version):
        candidates = list(wheels.glob(f"{project}-{version}-*.whl"))
        assert len(candidates) == 1, (project, version, candidates)
        return candidates[0]

    site = tmp_path / "site"
    home = tmp_path / "selected-home"
    other_home = tmp_path / "other-home"
    target = home / "plugins" / "mnemosyne"
    target.mkdir(parents=True)
    (other_home / "plugins" / "mnemosyne").mkdir(parents=True)
    (other_home / "plugins" / "mnemosyne" / "custom.txt").write_text("other profile")
    (home / "config.yaml").write_text("memory:\n  provider: mnemosyne\n")

    def install_wheel(path, *, upgrade=False):
        args = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--disable-pip-version-check",
            "--target",
            str(site),
        ]
        if upgrade:
            args.append("--upgrade")
        subprocess.run([*args, str(path)], check=True, capture_output=True, text=True)

    install_wheel(wheel("mnemosyne_memory", "3.15.1"))
    install_wheel(wheel("mnemosyne_hermes", "0.5.0"))
    package = site / "mnemosyne_hermes"
    for entry in package.iterdir():
        (target / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    (target / "custom.txt").write_text("keep custom data")
    assert not (target / "_verbatim_compat.py").exists()

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join([str(site), str(source)]),
            "HOME": str(tmp_path),
            "HERMES_HOME": str(home),
            "MNEMOSYNE_DATA_DIR": str(tmp_path / "database"),
            "MNEMOSYNE_NO_EMBEDDINGS": "1",
            "HERMES_ENABLE_PROJECT_PLUGINS": "0",
        }
    )
    probe = """
import json
import os
from plugins.memory import load_memory_provider
provider = load_memory_provider("mnemosyne", register_skills=False)
assert provider is not None, "real Hermes loader did not construct the provider"
provider.initialize("wheel-upgrade-test", hermes_home=os.environ["HERMES_HOME"])
assert provider._beam is not None
label = os.environ["LEGACY_PROBE_LABEL"]
capture = f"orchid {label} capture from user"
memory = f"orchid {label} ledger entry"
provider.sync_turn(capture, f"I will retain the {label} orchid capture in memory.",
                   session_id="wheel-upgrade-test")
assert any(f"[USER] {capture}" in row[0]
           for row in provider._beam.conn.execute("SELECT content FROM working_memory"))
written = json.loads(provider.handle_tool_call("mnemosyne_remember", {"content": memory}))
assert written.get("status") == "stored" and written.get("memory_id"), written
assert any(row[0] == memory for row in provider._beam.conn.execute(
    "SELECT content FROM working_memory")), written
recalled = json.loads(provider.handle_tool_call("mnemosyne_recall", {"query": f"orchid {label} ledger", "limit": 5}))
assert recalled.get("count", 0) > 0, recalled
assert any(memory in str(row) for row in recalled["results"]), recalled
print("provider constructed, captured, remembered and recalled")
"""

    def load(expect_success, label):
        probe_env = {**env, "LEGACY_PROBE_LABEL": label}
        result = subprocess.run(
            [sys.executable, "-c", probe],
            env=probe_env,
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=90,
        )
        if expect_success:
            assert result.returncode == 0, result.stderr
            assert "provider constructed, captured, remembered and recalled" in result.stdout
        else:
            assert result.returncode != 0, (result.stdout, result.stderr)
            assert "_verbatim_compat" in result.stderr

    load(True, "before-upgrade")
    install_wheel(wheel("mnemosyne_hermes", "0.7.0"), upgrade=True)
    assert (package / "_verbatim_compat.py").is_file()
    assert not (target / "_verbatim_compat.py").exists()
    load(False, "stale-layout")

    # Execute the documented ownership preflight against the upgraded wheel,
    # including its refusal paths, rather than duplicating its safety logic.
    guide = (Path(__file__).resolve().parents[3] / "docs" / "hermes-integration.md").read_text()
    section = guide.split("#### Recover a legacy per-file plugin directory", 1)[1]
    script = section.split("```bash\n", 1)[1].split("\n```", 1)[0]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    hermes_stub = bin_dir / "hermes"
    hermes_stub.write_text('#!/bin/sh\n[ "$1" = memory ] && [ "$2" = status ]\n')
    hermes_stub.chmod(0o755)
    env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
    env["HERMES_PYTHON"] = sys.executable
    backup = home / "mnemosyne-legacy-plugin-backup"
    foreign = target / "foreign-link.txt"
    foreign.symlink_to(other_home / "plugins" / "mnemosyne" / "custom.txt")
    refused = subprocess.run(
        ["bash", "-c", script],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert refused.returncode != 0 and "Foreign per-file link" in refused.stderr
    assert target.is_dir() and not backup.exists()
    foreign.unlink()
    profile = home / "profiles" / "custom" / "plugins" / "mnemosyne"
    profile.mkdir(parents=True)
    (profile / "owned.txt").write_text("profile artifact")
    refused = subprocess.run(
        ["bash", "-c", script],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert refused.returncode != 0 and "Profile plugin targets" in refused.stderr
    assert (profile / "owned.txt").read_text() == "profile artifact"
    profile.rename(other_home / "profile-artifact")

    recovered = subprocess.run(
        ["bash", "-c", script],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert recovered.returncode == 0, recovered.stderr
    assert "Installed. Symlink" in recovered.stdout
    assert backup.is_dir() and backup.parent != target.parent
    assert (backup / "custom.txt").read_text() == "keep custom data"
    discovery = subprocess.run(
        [
            sys.executable,
            "-c",
            "from plugins.memory import list_memory_provider_names; print(list_memory_provider_names())",
        ],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert discovery.returncode == 0, discovery.stderr
    assert "mnemosyne-legacy-plugin-backup" not in discovery.stdout
    assert "mnemosyne" in discovery.stdout
    assert target.is_symlink() and target.resolve() == package
    assert (backup / "custom.txt").read_text() == "keep custom data"
    assert (other_home / "plugins" / "mnemosyne" / "custom.txt").read_text() == "other profile"
    load(True, "after-recovery")
