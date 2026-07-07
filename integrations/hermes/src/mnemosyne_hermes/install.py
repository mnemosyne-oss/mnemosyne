"""Installer CLI for the Mnemosyne Hermes memory provider."""

from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import resources
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional



PLUGIN_NAME = "mnemosyne"
SKILL_NAME = "mnemosyne-memory-override"
SKILL_CATEGORY = "memory"
BUNDLED_SKILL_RESOURCE = ("skills", SKILL_NAME, "SKILL.md")


@dataclass(frozen=True)
class PluginState:
    """Detailed installation state for the Hermes plugin directory."""

    status: str
    installed: bool
    target: Path
    message: str
    link_target: Path | None = None
    mode: str = "missing"
    wrapper_python: Path | None = None
    wrapper_site_packages: Path | None = None
    wrapper_import_ok: bool | None = None
    wrapper_import_error: str | None = None


@dataclass(frozen=True)
class SkillState:
    """Detailed installation state for the bundled Hermes skill."""

    status: str
    installed: bool
    target: Path
    message: str


@dataclass(frozen=True)
class SkillInstallResult:
    """Result of installing, skipping, or planning the bundled skill."""

    action: str
    changed: bool
    target: Path
    message: str


def hermes_home() -> Path:
    """Return the Hermes home directory used for user-installed plugins."""
    return Path(os.environ.get("HERMES_HOME") or "~/.hermes").expanduser()


def _resolve_package_dir() -> Path:
    """Return the installed mnemosyne_hermes package directory.

    Avoid importing the package here: console-script loading first imports the
    package ``__init__`` before this module, and install/status commands should
    remain useful even when the Mnemosyne core dependency is unavailable or
    broken.
    """
    return Path(__file__).resolve().parent


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


def skill_target_file(hermes_home_path: str | Path | None = None) -> Path:
    """Return the deterministic install target for the bundled Hermes skill.

    Hermes supports categorized skill directories under ``skills/<category>/<name>/SKILL.md``;
    keep this memory guardrail in the memory category rather than the package's historical
    flat source-tree ``skills/*.md`` location.
    """
    base = Path(hermes_home_path).expanduser() if hermes_home_path else hermes_home()
    return base / "skills" / SKILL_CATEGORY / SKILL_NAME / "SKILL.md"


def bundled_skill_resource():
    """Return the importlib resource for the bundled memory override skill."""
    resource = resources.files("mnemosyne_hermes")
    for part in BUNDLED_SKILL_RESOURCE:
        resource = resource.joinpath(part)
    return resource


def bundled_skill_text() -> str:
    """Read the bundled memory override skill from package data."""
    source = bundled_skill_resource()
    if not source.is_file():
        raise FileNotFoundError(
            "Bundled Mnemosyne memory override skill is missing from package data: "
            f"{'/'.join(BUNDLED_SKILL_RESOURCE)}"
        )
    return source.read_text(encoding="utf-8")


def skill_state(*, hermes_home_path: str | Path | None = None) -> SkillState:
    """Return state for the bundled Hermes skill install target."""
    target = skill_target_file(hermes_home_path)
    if target.is_file():
        return SkillState(
            status="installed",
            installed=True,
            target=target,
            message="Bundled memory override skill is installed.",
        )
    if target.exists():
        return SkillState(
            status="invalid_target",
            installed=False,
            target=target,
            message=f"Skill target exists but is not a file: {target}",
        )
    return SkillState(
        status="missing",
        installed=False,
        target=target,
        message=f"No bundled memory override skill at {target}.",
    )


def _skill_backup_file(target: Path) -> Path:
    """Return the backup path used before overwriting a user-editable skill."""
    return target.with_name(f"{target.name}.bak")


def _skill_hash_file(target: Path) -> Path:
    """Return the sidecar path used to track installer-managed skill content."""
    return target.with_name(f"{target.name}.sha256")


def _sha256_text(content: str) -> str:
    """Return a stable digest for UTF-8 skill content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _is_managed_skill_copy(target: Path) -> bool:
    """Return whether target still matches the installer-managed sidecar hash."""
    if not target.is_file():
        return False
    hash_file = _skill_hash_file(target)
    if not hash_file.is_file():
        return False
    try:
        expected = hash_file.read_text(encoding="utf-8").strip()
        return expected == _sha256_text(target.read_text(encoding="utf-8"))
    except OSError:
        return False


def _write_skill_hash(target: Path, content: str) -> None:
    """Record the digest for installer-managed skill content."""
    _skill_hash_file(target).write_text(_sha256_text(content) + "\n", encoding="utf-8")


def install_bundled_skill(
    *,
    hermes_home_path: str | Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> SkillInstallResult:
    """Install the bundled memory override skill into Hermes' skills directory."""
    target = skill_target_file(hermes_home_path)
    exists = target.exists()
    backup = _skill_backup_file(target)
    content = bundled_skill_text()
    managed_copy = _is_managed_skill_copy(target)
    up_to_date = target.is_file() and target.read_text(encoding="utf-8") == content

    if exists and not force and not managed_copy:
        if up_to_date:
            _write_skill_hash(target, content)
            return SkillInstallResult(
                action="skip",
                changed=False,
                target=target,
                message=f"Skill already exists at {target}; already up to date.",
            )
        return SkillInstallResult(
            action="skip",
            changed=False,
            target=target,
            message=f"Skill already exists at {target}; skipped (use --force to overwrite).",
        )

    action = "refresh" if exists and managed_copy else ("overwrite" if exists else "install")
    if dry_run:
        backup_note = f" Existing file would be backed up to {backup}." if target.is_file() and force else ""
        return SkillInstallResult(
            action=action,
            changed=False,
            target=target,
            message=f"Would {action} bundled skill at {target}.{backup_note}",
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    backup_note = ""
    if exists and target.is_file() and force:
        shutil.copy2(target, backup)
        backup_note = f" Backup written to {backup}."
    elif exists and not target.is_file():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.write_text(content, encoding="utf-8")
    _write_skill_hash(target, content)
    verb = {"install": "Installed", "overwrite": "Overwrote", "refresh": "Refreshed"}[action]
    return SkillInstallResult(
        action=action,
        changed=True,
        target=target,
        message=f"{verb} bundled skill at {target}.{backup_note}",
    )


def _provider_init_is_valid(init_file: Path) -> bool:
    """Return whether an __init__.py looks like a Mnemosyne Hermes provider."""
    try:
        source = init_file.read_text(errors="replace")
        return "register_memory_provider" in source or "MnemosyneMemoryProvider" in source
    except Exception:
        return False


def _extract_wrapper_metadata(init_file: Path) -> tuple[Path | None, Path | None]:
    """Return (python, site-packages) metadata from a generated wrapper shim."""
    try:
        source = init_file.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None, None

    def _match(name: str) -> Path | None:
        match = re.search(rf"^{name}\s*=\s*(['\"])(.*?)\1", source, flags=re.MULTILINE)
        if not match:
            return None
        value = match.group(2).strip()
        return Path(value).expanduser() if value else None

    return _match("_PYTHON"), _match("_SITE")


def _site_packages_for_python(python: Path) -> Path:
    """Ask an interpreter for its purelib/site-packages path."""
    result = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Could not resolve site-packages for {python}: {stderr}")
    site = Path(result.stdout.strip()).expanduser()
    if not site:
        raise RuntimeError(f"Could not resolve site-packages for {python}")
    return site


def _check_wrapper_import(site_packages: Path, python: Path | None = None) -> tuple[bool, str | None]:
    """Return whether mnemosyne_hermes imports from the wrapper target."""
    if not site_packages.exists():
        return False, f"site-packages target missing: {site_packages}"
    if python is not None and not python.is_file():
        return False, f"wrapper Python missing: {python}"
    runner = python or Path(sys.executable)
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(site_packages)!r}); "
        "import mnemosyne_hermes; "
        "print(getattr(mnemosyne_hermes, '__version__', 'unknown'))"
    )
    result = subprocess.run(
        [str(runner), "-c", code],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode == 0:
        return True, None
    return False, (result.stderr.strip() or result.stdout.strip() or "import failed")[:500]


def _copy_plugin_yaml(target: Path) -> None:
    source_yaml = _resolve_package_dir() / "plugin.yaml"
    if source_yaml.is_file():
        shutil.copy2(source_yaml, target / "plugin.yaml")


def plugin_state(*, hermes_home_path: str | Path | None = None) -> PluginState:
    """Return detailed state for Hermes' Mnemosyne plugin discovery path."""
    target = plugin_target_dir(hermes_home_path)

    if target.is_symlink():
        raw_link = os.readlink(str(target))
        link_target = Path(raw_link)
        if not link_target.is_absolute():
            link_target = target.parent / link_target
        link_target = link_target.expanduser()
        if not link_target.exists():
            return PluginState(
                status="broken_symlink",
                installed=False,
                target=target,
                link_target=link_target,
                mode="symlink",
                message=(
                    "Plugin symlink exists but target is missing "
                    "(likely after a Hermes venv rebuild, Docker image update, "
                    "or package reinstall)."
                ),
            )

    if not target.exists():
        return PluginState(
            status="missing",
            installed=False,
            target=target,
            message=f"No plugin directory or symlink at {target}.",
        )

    init_file = target / "__init__.py"
    if not init_file.is_file():
        return PluginState(
            status="missing_init",
            installed=False,
            target=target,
            mode="symlink" if target.is_symlink() else "directory",
            message=f"Plugin path exists but has no __init__.py: {target}",
        )

    if not _provider_init_is_valid(init_file):
        return PluginState(
            status="invalid_provider",
            installed=False,
            target=target,
            mode="symlink" if target.is_symlink() else "directory",
            message=(
                "Plugin path exists but does not look like a Mnemosyne provider "
                "(__init__.py lacks provider markers)."
            ),
        )

    link_target = None
    mode = "wrapper"
    wrapper_python = None
    wrapper_site = None
    wrapper_import_ok = None
    wrapper_import_error = None

    if target.is_symlink():
        mode = "symlink"
        raw_link = os.readlink(str(target))
        link_target = Path(raw_link)
        if not link_target.is_absolute():
            link_target = target.parent / link_target
        link_target = link_target.expanduser()
    else:
        wrapper_python, wrapper_site = _extract_wrapper_metadata(init_file)
        if wrapper_site is None:
            mode = "directory"
        else:
            wrapper_import_ok, wrapper_import_error = _check_wrapper_import(
                wrapper_site,
                wrapper_python,
            )
            if not wrapper_import_ok:
                return PluginState(
                    status="stale_wrapper",
                    installed=False,
                    target=target,
                    mode="wrapper",
                    wrapper_python=wrapper_python,
                    wrapper_site_packages=wrapper_site,
                    wrapper_import_ok=wrapper_import_ok,
                    wrapper_import_error=wrapper_import_error,
                    message="Wrapper plugin exists but its target package cannot be imported.",
                )

    return PluginState(
        status="installed",
        installed=True,
        target=target,
        link_target=link_target,
        mode=mode,
        wrapper_python=wrapper_python,
        wrapper_site_packages=wrapper_site,
        wrapper_import_ok=wrapper_import_ok,
        wrapper_import_error=wrapper_import_error,
        message="Plugin is installed and discoverable.",
    )

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


def _config_selects_mnemosyne(text: str) -> bool:
    """Return True when a profile config selects ``memory.provider: mnemosyne``.

    Prefers a real YAML parse, which ignores comments and tolerates arbitrary
    whitespace. The line-anchored regex is used **only** when PyYAML is genuinely
    unavailable (``ImportError``). Malformed YAML is treated as "not opted in"
    rather than falling through to the looser regex.
    """
    try:
        import yaml
    except ImportError:
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
    base = Path(hermes_home_path).expanduser() if hermes_home_path else hermes_home()
    profiles_dir = base / "profiles"
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


def _link_profile(profile_home: Path, source: Path, *, force: bool = False) -> Optional[Path]:
    """Symlink ``profile_home/plugins/mnemosyne`` to source. Idempotent.

    A link already pointing at ``source`` is left untouched. A stale or broken
    link is replaced only when ``force`` is set; otherwise it is left in place
    and reported. Returns the link path on success, else None.
    """
    target = profile_home / "plugins" / PLUGIN_NAME
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_symlink() or target.exists():
        try:
            already = target.resolve() == source.resolve()
        except OSError:
            already = False
        if already:
            print(f"  Profile {profile_home.name}: already linked")
            return target
        if not force:
            print(f"  Profile {profile_home.name}: exists, skipped (use --force to replace)")
            return None
        if target.is_symlink():
            print(f"  Profile {profile_home.name}: replacing existing link -> {target.readlink()}")
            target.unlink()
        elif target.is_dir():
            print(f"  Profile {profile_home.name}: replacing existing directory {target}")
            shutil.rmtree(target)
        else:
            print(f"  Profile {profile_home.name}: replacing existing file {target}")
            target.unlink()

    try:
        os.symlink(str(source), str(target))
    except OSError as e:
        print(f"  Profile {profile_home.name}: failed to link: {e}")
        return None
    print(f"  Profile {profile_home.name}: linked {target}")
    return target


def _link_all_profiles(
    source: Path,
    *,
    hermes_home_path: str | Path | None = None,
    force: bool = False,
) -> list[Path]:
    """Link Mnemosyne into every opted-in profile. No-op without profiles.

    A failure on one profile is reported and does not abort the remaining
    profiles.
    """
    linked: list[Path] = []
    for profile_home in _iter_mnemosyne_profiles(hermes_home_path):
        try:
            result = _link_profile(profile_home, source, force=force)
        except OSError as e:
            print(f"  Profile {profile_home.name}: failed: {e}")
            continue
        if result is not None:
            linked.append(result)
    return linked


def _verify_links(*, hermes_home_path: str | Path | None = None) -> bool:
    """Print PASS/FAIL for each home that should have a resolvable plugin link.

    Checks the default home plus every opted-in profile. Returns True only
    when every checked link resolves to the provider source.
    """
    source = _resolve_package_dir()
    base = Path(hermes_home_path).expanduser() if hermes_home_path else hermes_home()
    homes: list[Path] = [base]
    homes.extend(_iter_mnemosyne_profiles(hermes_home_path))

    all_ok = True
    print("Verifying plugin links...")
    for home in homes:
        target = home / "plugins" / PLUGIN_NAME
        ok = target.is_symlink() or target.exists()
        if ok:
            try:
                ok = target.resolve() == source.resolve()
            except OSError:
                ok = False
        all_ok = all_ok and ok
        print(f"  {'PASS' if ok else 'FAIL'}  {home.name or home}: {target}")
    return all_ok


def _unlink_all_profiles(*, hermes_home_path: str | Path | None = None) -> None:
    """Remove the per-profile plugin links created by ``_link_all_profiles``.

    Scans every profile directory by *link*, not by config opt-in: a profile's
    ``plugins/mnemosyne`` is removed when it is a symlink resolving to the
    provider source, regardless of what (or whether) the profile's
    ``config.yaml`` currently selects. This still never touches a real directory
    or a link pointing elsewhere. Symlinked profile entries are skipped.
    """
    base = Path(hermes_home_path).expanduser() if hermes_home_path else hermes_home()
    profiles_dir = base / "profiles"
    if not profiles_dir.is_dir():
        return
    source = _resolve_package_dir()
    for child in sorted(profiles_dir.iterdir()):
        if child.is_symlink():
            continue
        target = child / "plugins" / PLUGIN_NAME
        if not target.is_symlink():
            continue
        try:
            if target.resolve() == source.resolve():
                target.unlink()
                print(f"  Removed profile link: {target}")
        except OSError:
            continue


def _prepare_plugin_target(base: Path, target: Path, *, force: bool) -> None:
    """Migrate legacy plugin names and remove an existing target when forced."""
    old_plugin_dir = base / "plugins" / "hermes-mnemosyne"
    if old_plugin_dir.is_symlink() or old_plugin_dir.exists():
        if old_plugin_dir.is_symlink() or os.path.islink(str(old_plugin_dir)):
            old_plugin_dir.unlink()
        else:
            shutil.rmtree(old_plugin_dir)
        print(f"  Removed old plugin directory: {old_plugin_dir}")

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
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)


def _write_wrapper_plugin(target: Path, *, python: Path, site_packages: Path) -> None:
    """Create a persistent Hermes plugin shim that imports from a selected env."""
    target.mkdir(parents=True, exist_ok=False)
    init_source = f"""\"\"\"Persistent Mnemosyne Hermes plugin wrapper.

Generated by ``mnemosyne-hermes install --mode wrapper``. The wrapper keeps the
Hermes discovery directory stable while importing the real ``mnemosyne_hermes``
package from the selected Python environment.
\"\"\"
from __future__ import annotations

import sys as _sys

# Metadata used by ``mnemosyne-hermes status``.
_PYTHON = {str(python)!r}
_SITE = {str(site_packages)!r}

if _SITE not in _sys.path:
    _sys.path.insert(0, _SITE)

# Hermes discovery marker: register_memory_provider / MnemosyneMemoryProvider
from mnemosyne_hermes import *  # noqa: F401,F403,E402
"""
    (target / "__init__.py").write_text(init_source, encoding="utf-8")
    _copy_plugin_yaml(target)


def install_plugin(
    *,
    hermes_home_path: str | Path | None = None,
    force: bool = False,
    mode: str = "symlink",
    python: str | Path | None = None,
) -> Path:
    """Install the Mnemosyne provider into Hermes' user plugin directory.

    ``mode='symlink'`` keeps the historical behavior. ``mode='wrapper'``
    creates a real persistent plugin directory containing a tiny shim that adds
    the selected interpreter's site-packages path to ``sys.path`` and imports
    ``mnemosyne_hermes`` from there.
    """
    if mode not in {"symlink", "wrapper"}:
        raise ValueError("mode must be 'symlink' or 'wrapper'")

    source = _resolve_package_dir()
    if not source.is_dir():
        raise FileNotFoundError(f"mnemosyne_hermes package not found at {source}")

    base = Path(hermes_home_path).expanduser() if hermes_home_path else hermes_home()
    target = plugin_target_dir(hermes_home_path)
    _prepare_plugin_target(base, target, force=force)

    if mode == "symlink":
        os.symlink(str(source), str(target))
        _link_all_profiles(source, hermes_home_path=hermes_home_path, force=force)
        return target

    wrapper_python = Path(python).expanduser() if python else Path(sys.executable)
    if not wrapper_python.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {wrapper_python}")
    site_packages = _site_packages_for_python(wrapper_python)
    import_ok, import_error = _check_wrapper_import(site_packages, wrapper_python)
    if not import_ok:
        raise RuntimeError(
            "Selected Python environment cannot import mnemosyne_hermes: "
            f"{import_error}"
        )
    _write_wrapper_plugin(target, python=wrapper_python, site_packages=site_packages)
    _link_all_profiles(source, hermes_home_path=hermes_home_path, force=force)
    return target

def uninstall_plugin(*, hermes_home_path: str | Path | None = None) -> Path:
    """Remove the Mnemosyne provider symlink from Hermes' user plugin directory."""
    _unlink_all_profiles(hermes_home_path=hermes_home_path)
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
    """Return whether the Mnemosyne provider is installed for Hermes discovery."""
    return plugin_state(hermes_home_path=hermes_home_path).installed



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
        help=(
            "Replace an existing Mnemosyne plugin directory. Also overwrites the "
            "bundled memory override skill after writing a SKILL.md.bak backup."
        ),
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
    install.add_argument(
        "--mode",
        choices=("symlink", "wrapper"),
        default="symlink",
        help="Install mode: symlink (default) or persistent wrapper shim.",
    )
    install.add_argument(
        "--python",
        dest="python",
        help="Python interpreter whose site-packages the wrapper should import from.",
    )
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


def run_install(
    *,
    force: bool = False,
    hermes_home_path: str | Path | None = None,
    no_bootstrap: bool = False,
    mode: str = "symlink",
    python: str | Path | None = None,
) -> int:
    """Core install logic — check deps, bootstrap Hermes venv if needed, create symlink.

    Returns 0 on success, 1 on failure.
    Can be called from the CLI ``install`` subcommand or programmatically
    (e.g., from ``upgrade.py`` after upgrading the pip package).
    """
    # Check core library first (installer's own Python)
    core_ok = check_mnemosyne_core()
    if not core_ok:
        print(
            "  mnemosyne-memory NOT found in this Python. Install it first:\n"
            "    pip install mnemosyne-hermes[all]",
            file=sys.stderr,
        )
        return 1

    # Symlink installs need Hermes' own Python to contain the package. Wrapper
    # installs validate the explicitly selected interpreter in install_plugin().
    hermes_python = _find_hermes_python() if mode == "symlink" else None
    if hermes_python and hermes_python.resolve() != Path(sys.executable).resolve():
        hermes_core = check_mnemosyne_core_for_hermes_python(hermes_python)
        if hermes_core is None:
            print(f"\n  ⚠ Hermes' Python at {hermes_python} can't import mnemosyne core.")
            print(f"     mnemosyne-hermes is installed in YOUR Python ({sys.executable}),")
            print(f"     but Hermes runs from a different venv.\n")
            if not no_bootstrap:
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
        hermes_home_path=hermes_home_path,
        force=force,
        mode=mode,
        python=python,
    )
    skill_result = install_bundled_skill(
        hermes_home_path=hermes_home_path,
        force=force,
    )
    if mode == "wrapper":
        state = plugin_state(hermes_home_path=hermes_home_path)
        print(f"Installed. Wrapper directory at {target}")
        if state.wrapper_python:
            print(f"  Python: {state.wrapper_python}")
        if state.wrapper_site_packages:
            print(f"  Site-packages: {state.wrapper_site_packages}")
    else:
        print(f"Installed. Symlink at {target}")
        print(f"  -> {os.readlink(str(target))}")
    print(f"  Skill: {skill_result.message}")
    print("Done. Next steps:")
    print("  hermes config set memory.provider mnemosyne")
    print("  hermes memory status")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the mnemosyne-hermes installer CLI."""
    parser = _parser()
    args = parser.parse_args(argv)
    command = args.command or "install"

    try:
        if command == "install":
            # Dry-run: just show what would happen
            hermes_python = _find_hermes_python()
            target = plugin_target_dir(args.hermes_home)
            if getattr(args, "dry_run", False):
                skill = skill_state(hermes_home_path=args.hermes_home)
                skill_plan = install_bundled_skill(
                    hermes_home_path=args.hermes_home,
                    force=getattr(args, "force", False),
                    dry_run=True,
                )
                print(f"  Plugin target dir: {target}")
                print(f"  Hermes Python: {hermes_python or 'not found'}")
                print(f"  Currently installed: {'yes' if is_installed(hermes_home_path=args.hermes_home) else 'no'}")
                print(f"  Install mode: {getattr(args, 'mode', 'symlink')}")
                print(f"  Skill target file: {skill.target}")
                print(f"  Skill state: {skill.status}")
                print(f"  Skill action: {skill_plan.message}")
                if getattr(args, "mode", "symlink") == "wrapper":
                    wrapper_python = Path(getattr(args, "python", None) or sys.executable).expanduser()
                    print(f"  Wrapper Python: {wrapper_python}")
                    if wrapper_python.is_file():
                        print(f"  Wrapper site-packages: {_site_packages_for_python(wrapper_python)}")
                print(f"  Will force: {bool(getattr(args, 'force', False))}")
                if hermes_python:
                    print(f"  Will bootstrap: {not getattr(args, 'no_bootstrap', False)}")
                return 0

            return run_install(
                force=getattr(args, "force", False),
                hermes_home_path=args.hermes_home,
                no_bootstrap=getattr(args, "no_bootstrap", False),
                mode=getattr(args, "mode", "symlink"),
                python=getattr(args, "python", None),
            )

        if command == "uninstall":
            target = uninstall_plugin(hermes_home_path=args.hermes_home)
            print(f"Removed. Symlink at {target} deleted.")
            return 0

        if command == "status":
            state = plugin_state(hermes_home_path=args.hermes_home)
            target = state.target
            installed = state.installed
            hermes_python = _find_hermes_python()
            print(f"Status for mnemosyne-hermes plugin")
            print(f"  Plugin path: {target}")
            print(f"  State: {state.status}")
            print(f"  Mode: {state.mode}")
            if installed:
                if state.mode == "symlink" and state.link_target is not None:
                    print(f"  Target: {state.link_target}")
                elif state.mode == "wrapper":
                    print(f"  Wrapper Python: {state.wrapper_python}")
                    print(f"  Wrapper site-packages: {state.wrapper_site_packages}")
                    print(f"  Wrapper import: {'OK' if state.wrapper_import_ok else 'not checked'}")
                else:
                    print(f"  Type: directory (not symlink)")
                print(f"  Plugin:    installed ✓")
            elif state.status == "broken_symlink":
                print(f"  Plugin:    broken symlink (target missing) ✗")
                print(f"  Broken target: {state.link_target}")
                print(f"  → Run: mnemosyne-hermes install --force")
            elif state.status == "stale_wrapper":
                print(f"  Plugin:    stale wrapper target ✗")
                print(f"  Wrapper Python: {state.wrapper_python}")
                print(f"  Wrapper site-packages: {state.wrapper_site_packages}")
                print(f"  Import error: {state.wrapper_import_error}")
                print(f"  → Re-run: mnemosyne-hermes install --mode wrapper --force --python <venv>/bin/python")
            else:
                print(f"  NOT installed: {state.message}")
                if state.link_target is not None:
                    print(f"  Broken target: {state.link_target}")
            skill = skill_state(hermes_home_path=args.hermes_home)
            print(f"  Skill path: {skill.target}")
            if skill.installed:
                print("  Skill:     installed ✓")
            else:
                print(f"  Skill:     {skill.status} ✗ ({skill.message})")
            print(f"  Core library: {'OK' if check_mnemosyne_core() else 'MISSING'}")
            print(f"  This Python: {sys.executable} ({sys.version.split()[0]})")
            if hermes_python:
                try:
                    import subprocess as _sp
                    _r = _sp.run([str(hermes_python), "--version"], capture_output=True, text=True, timeout=5)
                    _ver = _r.stdout.strip() or _r.stderr.strip()
                    print(f"  Hermes' Python: {hermes_python} ({_ver})")
                    if state.mode == "symlink" and hermes_python.resolve() != Path(sys.executable).resolve():
                        print(f"  ⚠ Python version MISMATCH! Install and Hermes use different Python versions.")
                        print(f"  → Run: {_ver.split()[1]}" if " " in _ver else "")
                except Exception:
                    print(f"  Hermes' Python: {hermes_python} (unable to check version)")
            else:
                print(f"  Hermes' Python: not found")
            if installed and state.mode == "symlink" and hermes_python and hermes_python.resolve() != Path(sys.executable).resolve():
                print(f"  → Hermes Python vs install Python mismatch means the symlink exists but Hermes")
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
            from mnemosyne_hermes.upgrade import upgrade_command
            return upgrade_command(args)

    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
