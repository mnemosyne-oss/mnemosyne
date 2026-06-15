"""Upgrade command for mnemosyne-hermes.

Detects the installation method (pipx / uv-tool / pip) and runs the
correct upgrade command, then re-registers the plugin with Hermes.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path
from typing import Optional


def detect_install_method() -> str:
    """Detect how mnemosyne-hermes was installed.

    Returns one of: "pipx", "uv-tool", "pip", "unknown".
    """
    try:
        dist = importlib.metadata.distribution("mnemosyne-hermes")
        location = Path(dist.locate_file("")).resolve()
        loc_str = str(location)

        if "pipx" in loc_str:
            return "pipx"
        if "uv" in loc_str and ("tool" in loc_str or "tools" in loc_str):
            return "uv-tool"
        if ".local/share/uv" in loc_str:
            return "uv-tool"
        return "pip"
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def get_current_version() -> str:
    """Return the currently installed mnemosyne-hermes version."""
    try:
        return importlib.metadata.version("mnemosyne-hermes")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def get_current_core_version() -> str:
    """Return the currently installed mnemosyne-memory version."""
    try:
        return importlib.metadata.version("mnemosyne-memory")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def check_available_version(method: str) -> Optional[str]:
    """Check what version is available via the given install method.

    Returns version string or None if we can't determine it.
    """
    try:
        if method == "pipx":
            result = subprocess.run(
                ["pipx", "run", "--", "pip", "index", "versions", "mnemosyne-hermes"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                # Parse "Available versions: 0.2.0, 0.1.9, ..."
                for line in result.stdout.splitlines():
                    if "Available versions" in line:
                        parts = line.split(":")
                        if len(parts) > 1:
                            versions = [v.strip() for v in parts[1].split(",")]
                            return versions[0] if versions else None
        elif method == "pip" or method == "uv-tool":
            result = subprocess.run(
                [sys.executable, "-m", "pip", "index", "versions", "mnemosyne-hermes"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "Available versions" in line:
                        parts = line.split(":")
                        if len(parts) > 1:
                            versions = [v.strip() for v in parts[1].split(",")]
                            return versions[0] if versions else None
    except Exception:
        pass
    return None


def run_upgrade_command(method: str, capture: bool = True) -> tuple[int, str]:
    """Run the upgrade command for the detected install method.

    Returns (exit_code, stdout+stderr text).
    """
    if method == "pipx":
        cmd = ["pipx", "upgrade", "mnemosyne-hermes"]
    elif method == "uv-tool":
        cmd = ["uv", "tool", "upgrade", "mnemosyne-hermes"]
    elif method == "pip":
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "mnemosyne-hermes[all]"]
    else:
        return (1, "Unknown install method")

    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return (result.returncode, (result.stdout + result.stderr).strip())
    else:
        print(f"→ Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, timeout=120)
        return (result.returncode, "")


def upgrade_command(args=None) -> int:
    """Main upgrade logic — detects method, shows versions, upgrades, re-registers.

    Can be called from CLI or programmatically.
    """
    dry_run = getattr(args, "dry_run", False) if args else False
    hermes_home_path = getattr(args, "hermes_home", None) if args else None

    method = detect_install_method()
    current_ver = get_current_version()
    core_ver = get_current_core_version()

    print(f"mnemosyne-hermes: {current_ver}")
    print(f"  Core (mnemosyne-memory): {core_ver}")
    print(f"  Install method: {method}")

    if method == "unknown":
        print("\n⚠ Could not detect installation method.")
        print("  Please upgrade manually:")
        print("    pipx upgrade mnemosyne-hermes")
        print("    uv tool upgrade mnemosyne-hermes")
        print("    pip install --upgrade mnemosyne-hermes[all]")
        print("\n  After upgrading, re-register the plugin:")
        print("    mnemosyne-hermes install --force")
        return 1

    # Check available version (best-effort)
    try:
        available = check_available_version(method)
        if available:
            print(f"  Available: {available}")
            if available == current_ver:
                print("\n✓ Already up to date.")
                return 0
    except Exception:
        pass

    if dry_run:
        method_name = {"pipx": "pipx upgrade", "uv-tool": "uv tool upgrade", "pip": "pip install --upgrade"}.get(method, method)
        print(f"\n  Would run: {method_name} mnemosyne-hermes")
        print("  Would run: mnemosyne-hermes install --force")
        print(f"  Plugin symlink target: {Path(hermes_home_path or '~/.hermes').expanduser() / 'plugins' / 'mnemosyne'}")
        return 0

    # Run the upgrade
    print(f"\n→ Upgrading via {method}...")
    code, output = run_upgrade_command(method, capture=True)
    if code != 0:
        if "not installed" in output.lower():
            print("  ⚠ Not installed via this method. Install it first:")
            install_help = {"pipx": "pipx install mnemosyne-hermes", "uv-tool": "uv tool install mnemosyne-hermes", "pip": "pip install mnemosyne-hermes[all]"}
            print(f"     {install_help.get(method, 'pip install mnemosyne-hermes')}")
        else:
            print(f"  ⚠ Upgrade failed:\n    {output[:500]}")
        return 1

    # Show result
    new_ver = get_current_version()
    print(f"  {current_ver} → {new_ver}")
    print("  ✓ Package upgraded.")

    # Re-register the plugin
    print("\n→ Re-registering plugin with Hermes...")
    try:
        from mnemosyne_hermes.install import run_install  # noqa: F811
        result_code = run_install(force=True, hermes_home_path=hermes_home_path)
        if result_code != 0:
            print("  ⚠ Re-registration had issues (see output above).")
            print("  Run manually: mnemosyne-hermes install --force")
            return 1
    except Exception as e:
        print(f"  ⚠ Could not auto-run install step: {e}")
        print("  Run manually: mnemosyne-hermes install --force")
        return 1

    print("\n✓ Upgrade complete!")
    print("\nNext steps:")
    print("  systemctl --user restart hermes-gateway")
    print("  hermes memory status")
    return 0
