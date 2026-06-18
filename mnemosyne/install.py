"""
Mnemosyne Hermes Installer
==========================

One-command setup for Mnemosyne as a Hermes MemoryProvider.

Usage:
    python -m mnemosyne.install
    # or after pip install:
    mnemosyne-install
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _get_mnemosyne_root() -> Path:
    """Return the absolute path to the Mnemosyne repo root."""
    # This file is at mnemosyne/install.py, so parent.parent is repo root
    return Path(__file__).resolve().parent.parent


def _get_hermes_home() -> Path:
    """Return the Hermes home directory, or None if not found."""
    # Check env var first
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    # Default location
    default = Path.home() / ".hermes"
    if default.exists():
        return default
    return None


def _get_hermes_agent_path() -> Path | None:
    """Try to find the hermes-agent installation."""
    # Check common locations
    candidates = [
        Path.home() / ".hermes" / "hermes-agent",
        Path.home() / "hermes-agent",
        Path("/opt/hermes/hermes-agent"),
    ]
    for c in candidates:
        if (c / "run_agent.py").exists():
            return c
    return None


def _is_windows() -> bool:
    return sys.platform.startswith("win32")


def _remove_link(link_path: Path) -> None:
    """Remove a symlink or junction. Works cross-platform."""
    if _is_windows():
        # Windows: junctions aren't detected by is_symlink(), and rmdir /
        # shutil.rmtree may follow the reparse point. Use rmdir which
        # removes the junction itself on Windows (like a directory symlink).
        import subprocess
        try:
            subprocess.run(
                ["cmd", "/c", "rmdir", str(link_path)],
                check=True, capture_output=True, text=True,
            )
            return
        except subprocess.CalledProcessError:
            pass  # fall through to fallback

    # Fallback: normal removal
    if link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        import shutil
        shutil.rmtree(link_path)


def _make_link(target: Path, source: Path) -> tuple[bool, str]:
    """Create a symlink (POSIX) or junction (Windows) from target to source.

    Returns (ok, message). The caller must remove any existing path at
    ``target`` first.
    """
    if _is_windows():
        import subprocess
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(source)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return False, f"Failed to create junction: {result.stderr.strip()}"
        return True, f"Junction: {target} -> {source}"
    try:
        target.symlink_to(source, target_is_directory=True)
    except OSError as e:
        return False, f"Failed to create symlink: {e}"
    return True, f"Symlinked: {target} -> {source}"


def _ensure_link() -> bool:
    """Create the plugin link from ~/.hermes/plugins/mnemosyne -> hermes_memory_provider.

    Uses symlinks on POSIX, directory junctions on Windows (no admin required).
    Automatically migrates from old plugin directory name (hermes-mnemosyne).
    """
    hermes_home = _get_hermes_home()
    if not hermes_home:
        print("❌ Hermes not found. Is Hermes installed?")
        print("   Expected: ~/.hermes/ or $HERMES_HOME set")
        return False

    plugins_dir = hermes_home / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)

    target = plugins_dir / "mnemosyne"
    source = _get_mnemosyne_root() / "hermes_memory_provider"

    if not source.exists():
        print(f"❌ Mnemosyne MemoryProvider not found at {source}")
        return False

    # Migrate from old plugin directory name (hermes-mnemosyne -> mnemosyne)
    old_target = plugins_dir / "hermes-mnemosyne"
    if old_target.is_symlink() or old_target.exists():
        _remove_link(old_target)
        print(f"🔄 Removed old plugin directory: {old_target}")

    # Migrate config from old provider name
    config_path = hermes_home / "config.yaml"
    if config_path.is_file():
        try:
            config_text = config_path.read_text(encoding="utf-8")
            if "provider: hermes-mnemosyne" in config_text:
                new_text = config_text.replace(
                    "provider: hermes-mnemosyne", "provider: mnemosyne"
                )
                config_path.write_text(new_text, encoding="utf-8")
                print("🔄 Updated config: memory.provider hermes-mnemosyne -> mnemosyne")
        except Exception:
            pass

    # Remove existing link or directory
    if target.is_symlink() or target.exists():
        _remove_link(target)
        print(f"🔄 Removed existing {target}")

    ok, msg = _make_link(target, source)
    if not ok:
        print(f"❌ {msg}")
        return False
    print(f"✅ {msg}")
    return True


def _config_selects_mnemosyne(text: str) -> bool:
    """Return True when a profile config selects ``memory.provider: mnemosyne``.

    Prefers a real YAML parse, which ignores comments and tolerates arbitrary
    whitespace. The line-anchored regex is used **only** when PyYAML is genuinely
    unavailable (``ImportError``), so the core package keeps working without a
    hard YAML dependency. Malformed YAML is treated as "not opted in" rather than
    falling through to the looser regex.
    """
    try:
        import yaml
    except ImportError:
        import re
        return re.search(
            r"^\s*provider\s*:\s*mnemosyne\s*(#.*)?$", text, re.MULTILINE
        ) is not None
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    if isinstance(cfg, dict):
        memory = cfg.get("memory")
        if isinstance(memory, dict):
            return memory.get("provider") == "mnemosyne"
    return False


def _iter_mnemosyne_profiles() -> list[Path]:
    """Return profile dirs under <hermes_home>/profiles/* that opt into Mnemosyne.

    A profile opts in when its ``config.yaml`` parses to
    ``memory.provider == "mnemosyne"`` (see ``_config_selects_mnemosyne``).
    Symlinked profile entries are skipped (the installer must not follow a
    profile symlink and write under its target). Profiles without a
    ``config.yaml`` are skipped. Returns an empty list when no ``profiles/``
    directory exists (the default, no-profile install).
    """
    hermes_home = _get_hermes_home()
    if not hermes_home:
        return []
    profiles_dir = hermes_home / "profiles"
    if not profiles_dir.is_dir():
        return []
    selected: list[Path] = []
    for child in sorted(profiles_dir.iterdir()):
        if child.is_symlink():
            continue
        if not child.is_dir():
            continue
        config_path = child / "config.yaml"
        if not config_path.is_file():
            continue
        try:
            text = config_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if _config_selects_mnemosyne(text):
            selected.append(child)
    return selected


def _link_profile(profile_home: Path, source: Path, *, force: bool = False) -> bool:
    """Link ``profile_home/plugins/mnemosyne`` to source. Idempotent.

    A link already pointing at ``source`` is left untouched. A stale or broken
    symlink is replaced. A real (non-symlink) path is left in place and reported
    unless ``force`` is set — the installer never silently deletes user data.
    Returns True on success.
    """
    plugins_dir = profile_home / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    target = plugins_dir / "mnemosyne"

    if target.is_symlink():
        try:
            if target.resolve() == source.resolve():
                print(f"✅ Profile {profile_home.name}: already linked")
                return True
        except OSError:
            pass
        _remove_link(target)  # stale/broken symlink — safe to replace
    elif target.exists():
        if not force:
            print(f"⏭️  Profile {profile_home.name}: {target} exists (not a link), skipped")
            return False
        _remove_link(target)

    ok, msg = _make_link(target, source)
    print(f"{'✅' if ok else '❌'} Profile {profile_home.name}: {msg}")
    return ok


def _link_all_profiles() -> None:
    """Link Mnemosyne into every opted-in profile. No-op without profiles.

    A failure on one profile is reported and does not abort the remaining
    profiles.
    """
    profiles = _iter_mnemosyne_profiles()
    if not profiles:
        return
    source = _get_mnemosyne_root() / "hermes_memory_provider"
    print()
    print(f"🔗 Linking {len(profiles)} profile(s)...")
    for profile_home in profiles:
        try:
            _link_profile(profile_home, source)
        except OSError as e:
            print(f"❌ Profile {profile_home.name}: {e}")


def _unlink_all_profiles() -> None:
    """Remove the per-profile plugin links created by ``_link_all_profiles``.

    Scans every profile directory by *link*, not by config opt-in: a profile's
    ``plugins/mnemosyne`` is removed when it is a symlink resolving to the
    provider source, regardless of what (or whether) the profile's
    ``config.yaml`` currently selects. This still never touches a real directory
    or a link pointing elsewhere. Symlinked profile entries are skipped.
    """
    hermes_home = _get_hermes_home()
    if not hermes_home:
        return
    profiles_dir = hermes_home / "profiles"
    if not profiles_dir.is_dir():
        return
    source = _get_mnemosyne_root() / "hermes_memory_provider"
    for child in sorted(profiles_dir.iterdir()):
        if child.is_symlink():
            continue
        target = child / "plugins" / "mnemosyne"
        if not target.is_symlink():
            continue
        try:
            if target.resolve() == source.resolve():
                _remove_link(target)
                print(f"Removed profile link: {target}")
        except OSError:
            continue


def _verify_links() -> bool:
    """Print PASS/FAIL for each home that should have a resolvable plugin link.

    Checks the default home plus every opted-in profile. Returns True only
    when every checked link resolves to the provider source.
    """
    source = _get_mnemosyne_root() / "hermes_memory_provider"
    hermes_home = _get_hermes_home()
    homes: list[Path] = []
    if hermes_home:
        homes.append(hermes_home)
    homes.extend(_iter_mnemosyne_profiles())

    all_ok = True
    print()
    print("🔍 Verifying plugin links...")
    for home in homes:
        target = home / "plugins" / "mnemosyne"
        ok = target.is_symlink() or target.exists()
        if ok:
            try:
                ok = target.resolve() == source.resolve()
            except OSError:
                ok = False
        all_ok = all_ok and ok
        print(f"  {'PASS' if ok else 'FAIL'}  {home.name or home}: {target}")
    return all_ok


def _configure_hermes() -> bool:
    """Set memory.provider = mnemosyne in Hermes config."""
    hermes_home = _get_hermes_home()
    if not hermes_home:
        return False

    config_path = hermes_home / "config.yaml"

    # Read existing config
    config_text = ""
    if config_path.exists():
        config_text = config_path.read_text(encoding="utf-8")

    # Check if already configured
    if "provider: mnemosyne" in config_text:
        print("✅ Hermes config already has memory.provider = mnemosyne")
        return True

    # Simple append approach (YAML-compatible)
    if "memory:" in config_text:
        # Replace existing memory block
        import re
        # Find memory: block and replace provider
        new_config = re.sub(
            r'(memory:\s*)\n(\s*provider:\s*\S+)?',
            r'\1\n  provider: mnemosyne\n',
            config_text,
            count=1,
        )
        if new_config == config_text:
            # No provider line found, insert one
            new_config = config_text.replace(
                "memory:",
                "memory:\n  provider: mnemosyne"
            )
        config_path.write_text(new_config, encoding="utf-8")
    else:
        # Append memory block
        with open(config_path, "a", encoding="utf-8") as f:
            f.write("\nmemory:\n  provider: mnemosyne\n")

    print(f"✅ Updated {config_path}: memory.provider = mnemosyne")
    return True


def _verify() -> bool:
    """Try to import and verify the provider works."""
    hermes_home = _get_hermes_home()
    if not hermes_home:
        return False

    # Add Hermes to path for verification
    agent_path = _get_hermes_agent_path()
    if agent_path and str(agent_path) not in sys.path:
        sys.path.insert(0, str(agent_path))

    try:
        from plugins.memory import load_memory_provider
        provider = load_memory_provider("mnemosyne")
        if provider and provider.is_available():
            print(f"✅ Provider verified: {provider.name} is_available=True")
            return True
        else:
            print("⚠️  Provider loaded but not available (Mnemosyne core not importable)")
            return False
    except Exception as e:
        print(f"⚠️  Verification skipped: {e}")
        return False


def install():
    """Run the full Mnemosyne Hermes installation."""
    print("🌀 Mnemosyne Hermes Installer")
    print("=" * 40)
    print()

    # Step 1: Link default home (symlink on POSIX, junction on Windows)
    if not _ensure_link():
        print()
        print("❌ Install failed at symlink step.")
        sys.exit(1)

    # Step 1b: Link any named profiles that opt into Mnemosyne
    _link_all_profiles()

    # Step 2: Configure (default home only — profile configs stay the user's)
    _configure_hermes()

    # Step 3: Verify
    print()
    print("🔍 Verifying...")
    _verify()
    _verify_links()

    print()
    print("✅ Mnemosyne is ready!")
    print()
    print("Next steps:")
    print("  • Restart Hermes (if running)")
    print("  • Run: hermes memory status")
    print("  • Run: hermes mnemosyne stats")
    print()


def uninstall():
    """Remove Mnemosyne from Hermes."""
    hermes_home = _get_hermes_home()
    if not hermes_home:
        print("❌ Hermes not found.")
        return

    target = hermes_home / "plugins" / "mnemosyne"
    old_target = hermes_home / "plugins" / "hermes-mnemosyne"

    found = False
    for t in [target, old_target]:
        if t.is_symlink() or t.exists():
            _remove_link(t)
            found = True
    if found:
        print(f"Removed {target}")
    else:
        print("ℹ️  Mnemosyne plugin not found in Hermes.")

    # Remove any per-profile links created at install time
    _unlink_all_profiles()

    # Reset config
    config_path = hermes_home / "config.yaml"
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        if "provider: mnemosyne" in text:
            new_text = text.replace("provider: mnemosyne", "provider: null")
            config_path.write_text(new_text, encoding="utf-8")
            print("✅ Reset memory.provider to null")

    print("\n✅ Mnemosyne uninstalled. Hermes will use built-in memory.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mnemosyne Hermes Installer")
    parser.add_argument("--uninstall", action="store_true", help="Remove Mnemosyne from Hermes")
    args = parser.parse_args()

    if args.uninstall:
        uninstall()
    else:
        install()
