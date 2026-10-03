"""
Mnemosyne Hermes Installer
==========================

Installs Mnemosyne as a Hermes MemoryProvider through the standalone
``mnemosyne-hermes`` provider package — the supported route.

This module is the compatibility entry point behind the ``mnemosyne-install``
and ``mnemosyne-uninstall`` console scripts. It used to create the legacy
source-checkout symlink (``~/.hermes/plugins/mnemosyne`` pointing at
``hermes_memory_provider/``). That route is obsolete (#651): the standalone
``mnemosyne-hermes`` package owns the plugin directory, the bundled skill, the
per-profile links and the wrapper mode, and its ``mnemosyne-hermes install``
command is the supported installer.

What this entry point does instead:

* delegates install, uninstall and status to the standalone provider;
* detects a legacy ``hermes_memory_provider`` plugin link, warns about it, and
  removes it during install (``--migrate`` does only that);
* fails clearly, naming the install command, when the standalone provider is
  not available.

The legacy module ``hermes_memory_provider/`` is still shipped for the
providers that import it; only the installer stops wiring it into Hermes.

Usage:
    mnemosyne-install                 # delegate an install to mnemosyne-hermes
    mnemosyne-install --status        # verify the standalone provider
    mnemosyne-install --migrate       # remove legacy links only
    mnemosyne-uninstall               # delegate an uninstall
    python -m mnemosyne.install --status
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

PLUGIN_DIRNAME = "mnemosyne"
LEGACY_PROVIDER_DIRNAME = "hermes_memory_provider"
LEGACY_PLUGIN_DIRNAME = "hermes-mnemosyne"
STANDALONE_MODULE = "mnemosyne_hermes.install"
STANDALONE_DISTRIBUTION = "mnemosyne-hermes"
STANDALONE_CONSOLE_SCRIPT = "mnemosyne-hermes"
STANDALONE_INSTALL_HINT = (
    "pipx install mnemosyne-hermes",
    "or, into Hermes' own venv:",
    'uv pip install --python <hermes-python> -U "mnemosyne-hermes[all]"',
)


def _print_install_hint(stream=sys.stderr) -> None:
    """Print the standalone-provider install commands, one per line."""
    for line in STANDALONE_INSTALL_HINT:
        print(f"     {line}", file=stream)

_PATH_SCRUB = (
    "import os, sys\n"
    "try:\n"
    "    _cwd = os.getcwd()\n"
    "except OSError:\n"
    "    _cwd = None\n"
    "sys.path[:] = [p for p in sys.path if p not in ('', '.', _cwd)]\n"
)
"""Drop the implicit cwd entry before importing anything.

``python -c`` puts the working directory at ``sys.path[0]``. A directory that
happens to contain a ``mnemosyne`` or ``mnemosyne_hermes`` entry — a source
checkout, or a workspace umbrella holding one — then shadows the installed
packages and makes a delegated probe or status report a false negative.
"""

_PROVIDER_PROBE = _PATH_SCRUB + (
    "import importlib.util as u\n"
    "sys.exit(0 if u.find_spec('mnemosyne_hermes') else 1)\n"
)

_DELEGATE_TO_STANDALONE = _PATH_SCRUB + (
    "from mnemosyne_hermes.install import main\n"
    "sys.exit(main())\n"
)


def _get_mnemosyne_root() -> Path:
    """Return the absolute path to the Mnemosyne repo root."""
    # This file is at mnemosyne/install.py, so parent.parent is repo root
    return Path(__file__).resolve().parent.parent


def _get_hermes_home() -> Path | None:
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


def _resolve_hermes_home(hermes_home_path: str | Path | None = None) -> Path | None:
    """Return the explicit Hermes home, else the discovered default."""
    if hermes_home_path is not None:
        return Path(hermes_home_path).expanduser()
    return _get_hermes_home()


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
        shutil.rmtree(link_path)


# ---------------------------------------------------------------------------
# Legacy install detection and migration (#651)
# ---------------------------------------------------------------------------


def _legacy_provider_dir() -> Path:
    """Return the obsolete source-checkout provider directory.

    The pre-#651 installer linked ``~/.hermes/plugins/mnemosyne`` at this
    directory, whether it came from a source checkout or from the copy shipped
    inside the installed ``mnemosyne-memory`` distribution.
    """
    return _get_mnemosyne_root() / LEGACY_PROVIDER_DIRNAME


def is_legacy_plugin_path(path: Path) -> bool:
    """Return whether ``path`` is a legacy ``hermes_memory_provider`` link.

    Detection is by the resolved target's directory name, so a link into a
    source checkout, a link into an installed distribution's copy, and a broken
    legacy link are all recognized. A link into the standalone
    ``mnemosyne_hermes`` package is deliberately **not** legacy: that target is
    the supported route, even when a user linked it by hand.
    """
    if not (path.is_symlink() or path.exists()):
        return False
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved.name == LEGACY_PROVIDER_DIRNAME


def _plugin_paths(hermes_home: Path | None) -> list[Path]:
    """Return the plugin link paths the installer is responsible for."""
    if hermes_home is None:
        return []
    paths = [hermes_home / "plugins" / PLUGIN_DIRNAME]
    paths.extend(
        profile / "plugins" / PLUGIN_DIRNAME
        for profile in _iter_mnemosyne_profiles(hermes_home)
    )
    return paths


def detect_legacy_installs(hermes_home_path: str | Path | None = None) -> list[Path]:
    """Return every legacy install path present in this Hermes home.

    Covers the default home and the opted-in profiles, plus the pre-rename
    ``hermes-mnemosyne`` plugin directory in the default home.
    """
    hermes_home = _resolve_hermes_home(hermes_home_path)
    found = [path for path in _plugin_paths(hermes_home) if is_legacy_plugin_path(path)]
    if hermes_home is not None:
        renamed = hermes_home / "plugins" / LEGACY_PLUGIN_DIRNAME
        if renamed.is_symlink() or renamed.exists():
            found.append(renamed)
    return found


def migrate_legacy_install(
    *,
    dry_run: bool = False,
    hermes_home_path: str | Path | None = None,
) -> list[Path]:
    """Remove legacy plugin links, returning the paths removed (or planned).

    Only the links the installer itself created are touched: a real directory
    is reported and left alone, so user data is never deleted silently.
    """
    planned = detect_legacy_installs(hermes_home_path)
    removed: list[Path] = []
    for path in planned:
        if not path.is_symlink() and path.exists():
            # Never report a path as removable when the real run would keep it.
            print(f"⏭️  Kept {path} (not a link) — remove it manually if it is obsolete")
            continue
        if not dry_run:
            _remove_link(path)
        removed.append(path)
    return removed


# ---------------------------------------------------------------------------
# Configuration (kept: the one-command setup writes provider selection)
# ---------------------------------------------------------------------------


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


def _iter_mnemosyne_profiles(hermes_home_path: str | Path | None = None) -> list[Path]:
    """Return profile dirs under <hermes_home>/profiles/* that opt into Mnemosyne.

    A profile opts in when its ``config.yaml`` parses to
    ``memory.provider == "mnemosyne"`` (see ``_config_selects_mnemosyne``).
    Symlinked profile entries are skipped (the installer must not follow a
    profile symlink and write under its target). Profiles without a
    ``config.yaml`` are skipped. Returns an empty list when no ``profiles/``
    directory exists (the default, no-profile install).
    """
    hermes_home = _resolve_hermes_home(hermes_home_path)
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


def _configure_hermes(hermes_home_path: str | Path | None = None) -> bool:
    """Set memory.provider = mnemosyne in Hermes config."""
    hermes_home = _resolve_hermes_home(hermes_home_path)
    if not hermes_home:
        return False

    config_path = hermes_home / "config.yaml"

    # Read existing config
    config_text = ""
    if config_path.exists():
        config_text = config_path.read_text(encoding="utf-8")

    # Check if already configured
    if _config_selects_mnemosyne(config_text):
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


# ---------------------------------------------------------------------------
# Standalone provider delegation
# ---------------------------------------------------------------------------


def _load_standalone_installer():
    """Return the standalone provider's install module, or None when absent.

    ``find_spec`` on the distribution name is a cheap presence check that does
    not execute the package. The module import is deferred until the provider is
    actually needed, so ``--status`` stays useful when the provider is missing.
    """
    try:
        if importlib.util.find_spec("mnemosyne_hermes") is None:
            return None
    except (ImportError, ValueError):
        return None
    try:
        return importlib.import_module(STANDALONE_MODULE)
    except Exception:
        return None


def _hermes_venv_python(hermes_home: Path | None) -> Path | None:
    """Return a Hermes venv Python, or None when it cannot be located.

    An explicit HERMES_HOME scopes discovery to that deployment; only when no
    home is known does this fall back to the common install locations.
    """
    candidates: list[Path] = []
    if hermes_home is not None:
        candidates.append(hermes_home / "hermes-agent" / "venv" / "bin" / "python")
    else:
        agent_path = _get_hermes_agent_path()
        if agent_path is not None:
            candidates.append(agent_path / "venv" / "bin" / "python")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _python_has_standalone(python: Path) -> bool:
    """Return whether ``python`` can import the standalone provider."""
    try:
        result = subprocess.run(
            [str(python), "-c", _PROVIDER_PROBE],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _standalone_runner(hermes_home: Path | None):
    """Return a callable running standalone-installer argv, or None.

    The provider is looked for, in order: in this Python, as the published
    ``mnemosyne-hermes`` console script, and finally in Hermes' own venv, which
    is where a Hermes-managed install lands.
    """
    module = _load_standalone_installer()
    if module is not None:
        return lambda argv: int(module.main(list(argv)))

    console_script = shutil.which(STANDALONE_CONSOLE_SCRIPT)
    if console_script:
        return lambda argv: subprocess.call([console_script, *argv])

    hermes_python = _hermes_venv_python(hermes_home)
    if hermes_python is not None and _python_has_standalone(hermes_python):
        command = [str(hermes_python), "-c", _DELEGATE_TO_STANDALONE]
        return lambda argv: subprocess.call([*command, *argv])

    return None


def _print_standalone_missing(legacy_paths: list[Path]) -> None:
    """Explain what is missing and how to fix it."""
    sys.stdout.flush()  # keep the banner above this stderr block in a terminal
    print("❌ The standalone Mnemosyne provider is not available.", file=sys.stderr)
    print(file=sys.stderr)
    if legacy_paths:
        print("   A legacy hermes_memory_provider install was found:", file=sys.stderr)
        for path in legacy_paths:
            print(f"     {path}", file=sys.stderr)
        print(
            "   That route is obsolete and is no longer created or repaired (#651).",
            file=sys.stderr,
        )
    print(f"   Install the supported provider ({STANDALONE_DISTRIBUTION}):", file=sys.stderr)
    _print_install_hint()
    print(file=sys.stderr)
    print("   Then re-run: mnemosyne-install", file=sys.stderr)
    print("   Verify with: mnemosyne-install --status", file=sys.stderr)


# ---------------------------------------------------------------------------
# Status / install / uninstall
# ---------------------------------------------------------------------------


def status(hermes_home_path: str | Path | None = None) -> bool:
    """Report whether the standalone provider is installed and active.

    Returns True only when no legacy link remains, the standalone provider is
    available, its plugin directory is discoverable by Hermes, and the Hermes
    config selects ``memory.provider: mnemosyne``.
    """
    hermes_home = _resolve_hermes_home(hermes_home_path)
    ok = True

    print("🔍 Mnemosyne Hermes provider status")
    print()

    legacy = detect_legacy_installs(hermes_home_path)
    if legacy:
        ok = False
        print("❌ Legacy hermes_memory_provider install detected (obsolete, #651):")
        for path in legacy:
            print(f"     {path}")
        print("   Fix: mnemosyne-install --migrate")
    else:
        print("✅ No legacy hermes_memory_provider plugin link")

    module = _load_standalone_installer()
    if module is not None:
        try:
            state = module.plugin_state(hermes_home_path=hermes_home)
        except Exception as exc:  # provider API drift must not crash the check
            ok = False
            print(f"❌ Could not read plugin state: {exc}")
        else:
            if state.installed:
                print(f"✅ Provider installed ({state.mode}): {state.target}")
                if state.link_target is not None:
                    print(f"     -> {state.link_target}")
            else:
                ok = False
                print(f"❌ Provider not installed ({state.status}): {state.message}")
    else:
        # The provider lives in another interpreter (typically Hermes' own venv),
        # so its own status command is the authority on plugin state.
        runner = _standalone_runner(hermes_home)
        if runner is None:
            ok = False
            print(f"❌ Standalone provider ({STANDALONE_DISTRIBUTION}) is not available here")
            _print_install_hint(sys.stdout)
        else:
            argv = ["status"]
            if hermes_home_path is not None:
                argv += ["--hermes-home", str(hermes_home_path)]
            print("ℹ️  Provider found outside this Python; delegating the plugin check")
            sys.stdout.flush()  # the delegated status writes to the same terminal
            if runner(argv) == 0:
                print("✅ Standalone provider reports the plugin installed and discoverable")
            else:
                ok = False
                print("❌ Standalone provider reports the plugin NOT installed")

    if hermes_home is None:
        ok = False
        print("❌ Hermes home not found (set HERMES_HOME or pass --hermes-home)")
    else:
        config_path = hermes_home / "config.yaml"
        if not config_path.is_file():
            ok = False
            print(f"❌ No Hermes config at {config_path}")
        else:
            try:
                text = config_path.read_text(encoding="utf-8")
            except OSError as exc:
                ok = False
                print(f"❌ Could not read {config_path}: {exc}")
            else:
                if _config_selects_mnemosyne(text):
                    print(f"✅ {config_path} selects memory.provider: mnemosyne")
                else:
                    ok = False
                    print(f"❌ {config_path} does not select memory.provider: mnemosyne")

    print()
    print("✅ Standalone provider is installed and active"
          if ok else "❌ Standalone provider is NOT fully installed")
    return ok


def install(
    *,
    force: bool = False,
    dry_run: bool = False,
    migrate_only: bool = False,
    hermes_home_path: str | Path | None = None,
) -> None:
    """Delegate an install to the standalone provider, migrating legacy links."""
    hermes_home = _resolve_hermes_home(hermes_home_path)

    print("🌀 Mnemosyne Hermes Installer")
    print("=" * 40)
    print()

    legacy = detect_legacy_installs(hermes_home_path)

    if migrate_only:
        if not legacy:
            print("✅ No legacy hermes_memory_provider install to migrate")
            return
        print(f"🔄 Migrating {len(legacy)} legacy path(s)...")
        for path in migrate_legacy_install(dry_run=dry_run, hermes_home_path=hermes_home_path):
            print(f"{'Would remove' if dry_run else 'Removed'}: {path}")
        print()
        print("✅ Legacy migration complete. Re-run mnemosyne-install to install the "
              "standalone provider.")
        return

    runner = _standalone_runner(hermes_home)
    if runner is None:
        _print_standalone_missing(legacy)
        sys.exit(1)

    if legacy:
        print("⚠️  Legacy hermes_memory_provider install detected (obsolete, #651):")
        for path in legacy:
            print(f"     {path}")
        print("   The supported route is the standalone "
              f"{STANDALONE_DISTRIBUTION} provider; migrating.")
        print()
        for path in migrate_legacy_install(dry_run=dry_run, hermes_home_path=hermes_home):
            print(f"{'Would remove' if dry_run else '🔄 Removed'}: {path}")
        print()

    argv = ["install"]
    if force:
        argv.append("--force")
    if dry_run:
        argv.append("--dry-run")
    if hermes_home_path is not None:
        argv += ["--hermes-home", str(hermes_home_path)]

    sys.stdout.flush()  # the delegated installer writes to the same terminal
    returncode = runner(argv)
    if returncode != 0:
        print()
        print("❌ Standalone provider install failed.", file=sys.stderr)
        print("   Run it directly for the full report: "
              f"{STANDALONE_CONSOLE_SCRIPT} install", file=sys.stderr)
        sys.exit(returncode)

    if dry_run:
        return

    _configure_hermes(hermes_home_path)

    print()
    if not status(hermes_home_path):
        sys.exit(1)


def uninstall(hermes_home_path: str | Path | None = None) -> None:
    """Delegate an uninstall, and remove any legacy links this route left."""
    hermes_home = _resolve_hermes_home(hermes_home_path)

    runner = _standalone_runner(hermes_home)
    if runner is None:
        print("⚠️  The standalone Mnemosyne provider is not available here; "
              "removing legacy links only.", file=sys.stderr)
    else:
        argv = ["uninstall"]
        if hermes_home_path is not None:
            argv += ["--hermes-home", str(hermes_home_path)]
        runner(argv)

    for path in migrate_legacy_install(hermes_home_path=hermes_home_path):
        print(f"Removed legacy link: {path}")

    if hermes_home is not None:
        config_path = hermes_home / "config.yaml"
        if config_path.exists():
            text = config_path.read_text(encoding="utf-8")
            if _config_selects_mnemosyne(text):
                new_text = text.replace("provider: mnemosyne", "provider: null")
                config_path.write_text(new_text, encoding="utf-8")
                print("✅ Reset memory.provider to null")

    print("\n✅ Mnemosyne uninstalled. Hermes will use built-in memory.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mnemosyne-install",
        description=(
            "Install the Mnemosyne Hermes memory provider. Delegates to the "
            "standalone mnemosyne-hermes provider, migrating any legacy "
            "hermes_memory_provider install."
        ),
    )
    parser.add_argument("--hermes-home", help="Hermes home. Defaults to HERMES_HOME or ~/.hermes.")
    parser.add_argument("--status", action="store_true",
                        help="Verify the standalone provider and exit.")
    parser.add_argument("--migrate", action="store_true",
                        help="Remove legacy hermes_memory_provider links and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Replace an existing provider plugin directory.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes.")
    parser.add_argument("--uninstall", action="store_true", help="Remove Mnemosyne from Hermes.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.status:
        return 0 if status(args.hermes_home) else 1
    if args.uninstall:
        uninstall(args.hermes_home)
        return 0
    install(
        force=args.force,
        dry_run=args.dry_run,
        migrate_only=args.migrate,
        hermes_home_path=args.hermes_home,
    )
    return 0


def _uninstall_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mnemosyne-uninstall",
        description=(
            "Remove Mnemosyne from Hermes: delegates to the standalone provider's "
            "uninstall, removes legacy links, and resets memory.provider to null. "
            "Memory databases are not deleted."
        ),
    )
    parser.add_argument("--hermes-home", help="Hermes home. Defaults to HERMES_HOME or ~/.hermes.")
    return parser


def uninstall_main(argv: list[str] | None = None) -> int:
    """Console entry point for ``mnemosyne-uninstall``.

    The entry point used to call :func:`uninstall` directly, which never reads
    its arguments, so ``mnemosyne-uninstall --help`` ran a real uninstall and
    rewrote the Hermes config. Parsing first means ``--help`` prints usage and
    any unknown argument is rejected before anything is touched.
    """
    args = _uninstall_parser().parse_args(argv)
    uninstall(args.hermes_home)
    return 0


if __name__ == "__main__":
    sys.exit(main())
