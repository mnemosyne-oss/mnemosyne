"""Installer CLI for the Mnemosyne Hermes memory provider."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

PLUGIN_NAME = "mnemosyne"


def hermes_home() -> Path:
    """Return the Hermes home directory used for user-installed plugins."""
    return Path(os.environ.get("HERMES_HOME") or "~/.hermes").expanduser()


def _resolve_package_dir() -> Path:
    """Return the installed mnemosyne_hermes package directory."""
    import mnemosyne_hermes
    return Path(mnemosyne_hermes.__file__).resolve().parent


def plugin_target_dir(hermes_home_path: str | Path | None = None) -> Path:
    """Return the Hermes memory plugin destination for Mnemosyne.

    Directory name matches the provider name used in
    ``memory.provider: mnemosyne`` config. Hermes discovers memory
    providers by scanning ``$HERMES_HOME/plugins/<name>/`` for
    directories whose ``__init__.py`` contains ``register_memory_provider``
    or ``MemoryProvider``.
    """
    base = Path(hermes_home_path).expanduser() if hermes_home_path else hermes_home()
    return base / "plugins" / PLUGIN_NAME


def _find_hermes_python() -> Optional[Path]:
    """Try to find Hermes' python executable for dep validation.

    Returns None when we can't find it (user runs manually).
    """
    hermes_home_path = hermes_home()

    # 1. Resolve the `hermes` launcher on PATH back to its venv Python.
    #    This is the most reliable probe: a pip/pipx-installed Hermes puts its
    #    console script next to the interpreter that runs it, so the Python is
    #    always a sibling of the resolved binary. Covers the common
    #    /usr/local/lib/hermes-agent/venv layout that the hardcoded roots below
    #    miss entirely (the silent-no-op that left provider deps out of Hermes'
    #    actual venv and produced "loaded but no provider instance found").
    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        # NOTE: resolve the *launcher* symlink (hermes -> venv/bin/hermes) to
        # find the venv bin dir, but do NOT resolve the python symlink itself.
        # A venv's bin/python is a symlink to the base interpreter; running the
        # venv path activates the venv site-packages, running the resolved base
        # path does NOT. Returning the resolved base interpreter would silently
        # drop the provider deps again.
        bin_dir = Path(hermes_bin).resolve().parent
        for py_name in ("python", "python3"):
            candidate = bin_dir / py_name
            if candidate.is_file():
                return candidate

    # 2. Check known hermes-agent checkout / install roots with a venv.
    for root in [
        hermes_home_path / "hermes-agent",
        Path.home() / "hermes-agent",
        Path("/opt/hermes/hermes-agent"),
        Path("/usr/local/lib/hermes-agent"),
        Path("/usr/lib/hermes-agent"),
    ]:
        for venv_name in ("venv", ".venv"):
            candidate = root / venv_name / "bin" / "python"
            if candidate.is_file():
                return candidate.resolve()

    # 3. Check if we're running inside Hermes' venv ourselves
    if sys.prefix != sys.base_prefix:
        venv_python = Path(sys.prefix) / "bin" / "python"
        if venv_python.is_file():
            return venv_python.resolve()

    # 4. Check VIRTUAL_ENV env var (uv-managed or explicit)
    ve = os.environ.get("VIRTUAL_ENV")
    if ve:
        candidate = Path(ve) / "bin" / "python"
        if candidate.is_file():
            return candidate.resolve()

    return None


def _bootstrap_hermes_venv(hermes_python: Path) -> bool:
    """Install mnemosyne-hermes into Hermes' Python venv."""
    from . import __version__
    pkg_name = f"mnemosyne-hermes[all]=={__version__}"
    cmd = [str(hermes_python), "-m", "pip", "install", "--upgrade", pkg_name]
    print(f"  Installing {pkg_name} into {hermes_python.parent.parent.name}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            stderr = result.stderr.strip()[:500]
            print(f"  ⚠ Bootstrap failed: {stderr}", file=sys.stderr)
            return False
        print(f"  ✓ mnemosyne-hermes installed into Hermes' venv")
        return True
    except Exception as exc:
        print(f"  ⚠ Bootstrap failed: {exc}", file=sys.stderr)
        return False


def check_mnemosyne_core() -> bool:
    """Verify mnemosyne-memory core library is installed."""
    try:
        importlib.import_module("mnemosyne.core.beam")
        import mnemosyne
        print(f"  mnemosyne-memory {mnemosyne.__version__} installed")
        return True
    except ImportError:
        return False


def check_mnemosyne_core_for_hermes_python(hermes_python: Path) -> Optional[str]:
    """Check if Hermes' Python can import mnemosyne core.

    Returns the version string if importable, None otherwise.
    """
    try:
        result = subprocess.run(
            [str(hermes_python), "-c",
             "import mnemosyne; print(mnemosyne.__version__); "
             "import sqlite_vec"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
        return None
    except Exception:
        return None


def install_plugin(
    *,
    hermes_home_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Install the Mnemosyne provider into Hermes' user plugin directory.

    Creates a symlink from ``$HERMES_HOME/plugins/mnemosyne/`` to the
    installed ``mnemosyne_hermes`` package directory. Hermes discovers
    memory providers by scanning ``$HERMES_HOME/plugins/<name>/`` for
    directories whose ``__init__.py`` contains ``register_memory_provider``
    or ``MemoryProvider``.

    The symlink approach means all relative imports (cli, tools, audit)
    resolve correctly through the real package, and ``hermes update`` /
    ``pipx upgrade mnemosyne-hermes`` automatically refreshes the target.
    """
    source = _resolve_package_dir()
    if not source.is_dir():
        raise FileNotFoundError(
            f"mnemosyne_hermes package not found at {source}"
        )

    base = Path(hermes_home_path).expanduser() if hermes_home_path else hermes_home()
    target = plugin_target_dir(hermes_home_path)

    # Migrate from old hermes-mnemosyne directory (deploy script era)
    old_plugin_dir = base / "plugins" / "hermes-mnemosyne"
    if old_plugin_dir.is_symlink() or old_plugin_dir.exists():
        if old_plugin_dir.is_symlink() or os.path.islink(str(old_plugin_dir)):
            old_plugin_dir.unlink()
        else:
            shutil.rmtree(old_plugin_dir)
        logger = print
        logger(f"  Removed old plugin directory: {old_plugin_dir}")

    # Also migrate config from old provider name
    config_path = base / "config.yaml"
    if config_path.is_file():
        try:
            config_text = config_path.read_text(encoding="utf-8")
            if "provider: hermes-mnemosyne" in config_text:
                new_text = config_text.replace("provider: hermes-mnemosyne", "provider: mnemosyne")
                config_path.write_text(new_text, encoding="utf-8")
                print("  Updated config: memory.provider hermes-mnemosyne -> mnemosyne")
        except Exception:
            pass

    if target.is_symlink() or target.exists():
        if not force:
            raise FileExistsError(
                f"{target} already exists. Re-run with --force to replace it."
            )
        # Remove existing link or directory cleanly
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(source), str(target))
    return target


def uninstall_plugin(*, hermes_home_path: str | Path | None = None) -> Path:
    """Remove the Mnemosyne provider symlink from Hermes' user plugin directory."""
    target = plugin_target_dir(hermes_home_path)
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    return target


def cleanup_plugin(
    *,
    hermes_home_path: str | Path | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Remove all traces of mnemosyne from Hermes' plugin directory.

    Safe to run -- never touches the database or memory files.

    Returns a list of actions taken (or would be taken with dry_run=True).
    """
    base = Path(hermes_home_path).expanduser() if hermes_home_path else hermes_home()
    actions: list[str] = []

    # 1. Current plugin symlink/dir
    target = plugin_target_dir(hermes_home_path)
    if target.is_symlink() or target.exists():
        if dry_run:
            actions.append(f"Would remove: {target}")
        else:
            if target.is_symlink():
                target.unlink()
            else:
                shutil.rmtree(target)
            actions.append(f"Removed: {target}")

    # 2. Old hermes-mnemosyne directory (deploy script era)
    old_dir = base / "plugins" / "hermes-mnemosyne"
    if old_dir.is_symlink() or old_dir.exists():
        if dry_run:
            actions.append(f"Would remove: {old_dir}")
        else:
            if old_dir.is_symlink() or os.path.islink(str(old_dir)):
                old_dir.unlink()
            else:
                shutil.rmtree(old_dir)
            actions.append(f"Removed: {old_dir}")

    # 3. Reset config if it points to mnemosyne
    config_path = base / "config.yaml"
    if config_path.is_file():
        try:
            config_text = config_path.read_text(encoding="utf-8")
            if "memory.provider: mnemosyne" in config_text or "memory:\n  provider: mnemosyne" in config_text:
                if dry_run:
                    actions.append("Would reset config: memory.provider from 'mnemosyne' to unset")
                else:
                    # Simple line-based replacement to remove the provider setting
                    import re as _re
                    new_text = _re.sub(
                        r"^memory:\n\s+provider: mnemosyne",
                        "memory:\n  # provider: mnemosyne (unset by cleanup)",
                        config_text,
                        flags=_re.MULTILINE,
                    )
                    # Also handle inline form
                    new_text = new_text.replace("memory.provider: mnemosyne", "# memory.provider: mnemosyne (unset by cleanup)")
                    if new_text != config_text:
                        config_path.write_text(new_text, encoding="utf-8")
                        actions.append("Reset config: memory.provider from 'mnemosyne' to unset")
        except Exception:
            pass

    return actions


def _do_upgrade(*, force: bool = True, hermes_home_path: str | Path | None = None) -> bool:
    """Run pipx upgrade mnemosyne-hermes then install --force."""
    import subprocess as _sp

    print("  Upgrading mnemosyne-hermes via pipx...")
    try:
        result = _sp.run(
            ["pipx", "upgrade", "mnemosyne-hermes"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[:300]
            if "not installed" in stderr:
                print("  ⚠ mnemosyne-hermes not installed via pipx. Install it first:")
                print("     pipx install mnemosyne-hermes")
                return False
            print(f"  ⚠ pipx upgrade failed: {stderr}")
            # Continue anyway -- maybe the user installed via pip directly
            print("  Continuing with re-install...")
        else:
            out = result.stdout.strip()[:200]
            if out:
                print(f"  {out}")
    except FileNotFoundError:
        print("  ⚠ pipx not found. Install it: pip install pipx")
        return False

    # Now re-install the plugin symlink
    print("  Re-installing plugin symlink...")
    try:
        target = install_plugin(hermes_home_path=hermes_home_path, force=force)
        print(f"  Installed. Symlink at {target}")
        print(f"    -> {os.readlink(str(target))}")
        return True
    except Exception as exc:
        print(f"  ⚠ Re-install failed: {exc}")
        return False


def is_installed(*, hermes_home_path: str | Path | None = None) -> bool:
    """Return whether the Mnemosyne provider symlink exists for Hermes discovery.

    Checks that the target is a symlink (or directory) with a valid
    ``__init__.py`` containing the expected symbols.
    """
    target = plugin_target_dir(hermes_home_path)
    if not target.exists():
        return False
    init_file = target / "__init__.py"
    if not init_file.is_file():
        return False
    try:
        source = init_file.read_text(errors="replace")
        return "register_memory_provider" in source or "MnemosyneMemoryProvider" in source
    except Exception:
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mnemosyne-hermes",
        description="Install the Mnemosyne memory provider for Hermes Agent.",
    )
    parser.add_argument(
        "--hermes-home",
        help="Hermes home directory. Defaults to HERMES_HOME or ~/.hermes.",
    )

    subparsers = parser.add_subparsers(dest="command")

    install = subparsers.add_parser(
        "install",
        help="Install Mnemosyne into Hermes' memory provider plugin directory.",
    )
    install.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing Mnemosyne plugin directory.",
    )
    install.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes.",
    )
    install.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip auto-installing mnemosyne-hermes into Hermes' venv.",
    )
    subparsers = subparsers
    subparsers.add_parser(
        "uninstall",
        help="Remove Mnemosyne from Hermes' memory provider plugin directory.",
    )
    subparsers.add_parser(
        "status",
        help="Show whether Mnemosyne is installed for Hermes memory discovery.",
    )
    cleanup = subparsers.add_parser(
        "cleanup",
        help="Remove all traces of Mnemosyne from Hermes plugin directory (safe, never touches database).",
    )
    cleanup.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without removing anything.",
    )
    upgrade = subparsers.add_parser(
        "upgrade",
        help="Upgrade mnemosyne-hermes via pipx and re-install the plugin symlink.",
    )
    upgrade.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the mnemosyne-hermes installer CLI."""
    parser = _parser()
    args = parser.parse_args(argv)
    command = args.command or "install"

    try:
        if command == "install":
            # Check core library first (installer's own Python)
            core_ok = check_mnemosyne_core()
            if not core_ok:
                print(
                    "  mnemosyne-memory NOT found in this Python. Install it first:\n"
                    "    pip install mnemosyne-hermes[all]",
                    file=sys.stderr,
                )
                return 1

            # Dry-run: just show what would happen
            hermes_python = _find_hermes_python()
            target = plugin_target_dir(args.hermes_home)
            if getattr(args, "dry_run", False):
                print(f"  Plugin target dir: {target}")
                print(f"  Hermes Python: {hermes_python or 'not found'}")
                print(f"  Currently installed: {'yes' if is_installed(hermes_home_path=args.hermes_home) else 'no'}")
                print(f"  Will force: {bool(getattr(args, 'force', False))}")
                if hermes_python:
                    print(f"  Will bootstrap: {not getattr(args, 'no_bootstrap', False)}")
                return 0

            # Find Hermes' Python and validate deps there too
            hermes_python = _find_hermes_python()
            if hermes_python and hermes_python.resolve() != Path(sys.executable).resolve():
                hermes_core = check_mnemosyne_core_for_hermes_python(hermes_python)
                if hermes_core is None:
                    print(f"\n  ⚠ Hermes' Python at {hermes_python} can't import mnemosyne core.")
                    print(f"     mnemosyne-hermes is installed in YOUR Python ({sys.executable}),")
                    print(f"     but Hermes runs from a different venv.\n")
                    if not getattr(args, "no_bootstrap", False):
                        print("  → Attempting auto-bootstrap...")
                        if _bootstrap_hermes_venv(hermes_python):
                            print("     ✓ Hermes venv now has mnemosyne-hermes installed.\n")
                        else:
                            print("\n  Install it manually:\n"
                                  f"    uv pip install --python {hermes_python} -U 'mnemosyne-hermes[all]'\n"
                                  "  Then re-run: mnemosyne-hermes install")
                            return 1
                    else:
                        print("  → Skipping auto-bootstrap (--no-bootstrap).\n"
                              "    Install manually:\n"
                              f"      uv pip install --python {hermes_python} -U 'mnemosyne-hermes[all]'\n"
                              "    Then re-run: mnemosyne-hermes install")
                        return 1
                else:
                    print(f"  Hermes' Python: mnemosyne-memory {hermes_core} OK")

            target = install_plugin(
                hermes_home_path=args.hermes_home,
                force=getattr(args, "force", False),
            )
            print(f"Installed. Symlink at {target}")
            print(f"  -> {os.readlink(str(target))}")
            print("Done. Next steps:")
            print("  hermes config set memory.provider mnemosyne")
            print("  hermes memory status")
            return 0

        if command == "uninstall":
            target = uninstall_plugin(hermes_home_path=args.hermes_home)
            print(f"Removed. Symlink at {target} deleted.")
            return 0

        if command == "status":
            target = plugin_target_dir(args.hermes_home)
            installed = is_installed(hermes_home_path=args.hermes_home)
            hermes_python = _find_hermes_python()
            print(f"Status for mnemosyne-hermes plugin")
            print(f"  Plugin symlink: {target}")
            if installed:
                try:
                    link = os.readlink(str(target))
                    print(f"  Target: {link}")
                except Exception:
                    print(f"  Type: directory (not symlink)")
            else:
                print(f"  NOT installed (no symlink at {target})")
            print(f"  Core library: {'OK' if check_mnemosyne_core() else 'MISSING'}")
            print(f"  This Python: {sys.executable} ({sys.version.split()[0]})")
            if hermes_python:
                try:
                    import subprocess as _sp
                    _r = _sp.run([str(hermes_python), "--version"], capture_output=True, text=True, timeout=5)
                    _ver = _r.stdout.strip() or _r.stderr.strip()
                    print(f"  Hermes' Python: {hermes_python} ({_ver})")
                    if hermes_python.resolve() != Path(sys.executable).resolve():
                        print(f"  ⚠ Python version MISMATCH! Install and Hermes use different Python versions.")
                        print(f"  → Run: {_ver.split()[1]}" if " " in _ver else "")
                except Exception:
                    print(f"  Hermes' Python: {hermes_python} (unable to check version)")
            else:
                print(f"  Hermes' Python: not found")
            if installed and hermes_python and hermes_python.resolve() != Path(sys.executable).resolve():
                print(f"  → Her Python vs install Python mismatch means the symlink exists but Hermes")
                print(f"     may not be able to import mnemosyne core. Run with --dry-run to diagnose.")
            return 0 if installed else 1

        if command == "cleanup":
            dry_run = getattr(args, "dry_run", False)
            mode = " (dry-run)" if dry_run else ""
            print(f"Cleaning up mnemosyne-hermes plugin{mode}...")
            actions = cleanup_plugin(
                hermes_home_path=args.hermes_home,
                dry_run=dry_run,
            )
            if not actions:
                print("  Nothing to clean up.")
            for a in actions:
                print(f"  {a}")
            return 0

        if command == "upgrade":
            dry_run = getattr(args, "dry_run", False)
            if dry_run:
                print("  Would run: pipx upgrade mnemosyne-hermes")
                print("  Would run: mnemosyne-hermes install --force")
                t = plugin_target_dir(args.hermes_home)
                print(f"  Plugin symlink target: {t}")
                return 0
            print("Upgrading mnemosyne-hermes...")
            ok = _do_upgrade(hermes_home_path=args.hermes_home)
            if ok:
                print("Done. Next steps:")
                print("  hermes config set memory.provider mnemosyne")
                print("  hermes memory status")
            return 0 if ok else 1

    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
