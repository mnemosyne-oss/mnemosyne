"""Installer CLI for the Mnemosyne Hermes memory provider."""

from __future__ import annotations

import argparse
import importlib.metadata
import hashlib
import importlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional

PLUGIN_NAME = "mnemosyne"
DEFAULT_WRAPPER_IMPORT_TIMEOUT = 60.0
SKILL_NAME = "mnemosyne-memory-override"
SKILL_CATEGORY = "memory"
BUNDLED_SKILL_RESOURCE = ("skills", SKILL_NAME, "SKILL.md")

_MAX_HERMES_BIN_DEPTH = 10
LOGGER = logging.getLogger(__name__)


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
class _WrapperMetadata:
    """Validated wrapper manifest metadata or a manifest validation error."""

    python: Path | None = None
    site_packages: Path | None = None
    error: str | None = None


@dataclass(frozen=True)
class _ProfileLinkSnapshot:
    """Identity of a recognized profile symlink at discovery time."""

    path: Path
    link_target: str
    device: int
    inode: int


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


class _WindowsSymlinkPrivilegeError(RuntimeError):
    """A Windows symlink failure for which the CLI has a safe retry."""


def _is_windows_symlink_privilege_error(error: OSError) -> bool:
    """Return whether ``error`` is Windows ERROR_PRIVILEGE_NOT_HELD (1314)."""
    return getattr(error, "winerror", None) == 1314


def _windows_wrapper_retry_command(hermes_python: Path) -> str:
    """Format a Windows-command-line-safe wrapper retry for a selected Python."""
    return subprocess.list2cmdline(
        ["mnemosyne-hermes", "install", "--mode", "wrapper", "--python", str(hermes_python)]
    )


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


def _provider_init_is_mnemosyne(init_file: Path) -> bool:
    """Return whether a plugin has installer-owned Mnemosyne identity markers."""
    try:
        plugin_yaml = init_file.with_name("plugin.yaml").read_text(errors="replace")
        if not re.search(
            r"^\s*name:\s*['\"]?hermes-mnemosyne['\"]?\s*$",
            plugin_yaml,
            flags=re.MULTILINE,
        ):
            return False

        wrapper_manifest = init_file.with_name("mnemosyne-wrapper.json")
        if wrapper_manifest.is_file():
            metadata = _wrapper_metadata(init_file.parent, init_file)
            return (
                metadata.error is None
                and metadata.python is not None
                and metadata.site_packages is not None
            )

        source = init_file.read_text(errors="replace")
        legacy_python, legacy_site = _extract_wrapper_metadata(init_file)
        if (
            legacy_python is not None
            and legacy_python.is_absolute()
            and legacy_site is not None
            and legacy_site.is_absolute()
            and re.search(
                r"^from\s+mnemosyne_hermes\s+import\s+\*(?:\s+#.*)?$",
                source,
                flags=re.MULTILINE,
            )
        ):
            return True
        return False
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


def _wrapper_metadata(target: Path, init_file: Path) -> _WrapperMetadata:
    """Read wrapper metadata, using legacy assignments only when no manifest exists."""
    manifest = target / "mnemosyne-wrapper.json"
    try:
        if not manifest.exists():
            python, site_packages = _extract_wrapper_metadata(init_file)
            return _WrapperMetadata(python=python, site_packages=site_packages)
        if not manifest.is_file():
            return _WrapperMetadata(error="Invalid Mnemosyne wrapper manifest: not a file")
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        return _WrapperMetadata(error=f"Invalid Mnemosyne wrapper manifest: {exc}")

    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or data.get("package") != "mnemosyne_hermes"
    ):
        return _WrapperMetadata(error="Invalid Mnemosyne wrapper manifest schema")
    if not isinstance(data.get("python"), str) or not isinstance(data.get("site_packages"), str):
        return _WrapperMetadata(error="Invalid Mnemosyne wrapper manifest paths")

    python = Path(data["python"]).expanduser()
    site_packages = Path(data["site_packages"]).expanduser()
    if not python.is_absolute() or not site_packages.is_absolute():
        return _WrapperMetadata(error="Invalid Mnemosyne wrapper manifest paths")
    return _WrapperMetadata(python=python, site_packages=site_packages)


def _validated_import_timeout(value: float | str) -> float:
    """Return a safe wrapper-validation timeout in seconds."""
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("import timeout must be a positive finite number of seconds") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("import timeout must be a positive finite number of seconds")
    return timeout


def _parse_import_timeout(value: str) -> float:
    """Argparse adapter that keeps invalid timeout diagnostics actionable."""
    try:
        return _validated_import_timeout(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _timeout_diagnostic(python: Path, timeout: float, *, retry_hint: bool = True) -> str:
    """Return an actionable timeout diagnostic for selected wrapper Python."""
    seconds = f"{timeout:g}"
    message = f"wrapper validation timed out after {seconds} seconds for interpreter {python}."
    if not retry_hint:
        return message + (
            " Status validates wrapper imports with its fixed default "
            f"{DEFAULT_WRAPPER_IMPORT_TIMEOUT:g}-second policy. Inspect the selected "
            "interpreter and its import performance; --import-timeout only affects "
            "installer validation."
        )
    retry_seconds = f"{max(timeout * 2, timeout + 1):g}"
    retry_args = [
        "mnemosyne-hermes",
        "install",
        "--mode",
        "wrapper",
        "--python",
        str(python),
        "--import-timeout",
        retry_seconds,
    ]
    retry_command = (
        subprocess.list2cmdline(retry_args) if os.name == "nt" else shlex.join(retry_args)
    )
    return message + f" Retry with: {retry_command}"


def _site_packages_for_python(
    python: Path, *, import_timeout: float = DEFAULT_WRAPPER_IMPORT_TIMEOUT
) -> Path:
    """Ask an interpreter for its purelib/site-packages path."""
    import_timeout = _validated_import_timeout(import_timeout)
    try:
        result = subprocess.run(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            capture_output=True,
            text=True,
            timeout=import_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(_timeout_diagnostic(python, import_timeout)) from exc
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Could not resolve site-packages for {python}: {stderr}")
    site = Path(result.stdout.strip()).expanduser()
    if not site:
        raise RuntimeError(f"Could not resolve site-packages for {python}")
    return site


def _check_wrapper_import(
    site_packages: Path,
    python: Path | None = None,
    *,
    import_timeout: float = DEFAULT_WRAPPER_IMPORT_TIMEOUT,
    retry_hint: bool = True,
) -> tuple[bool, str | None, bool]:
    """Return import success, error text, and whether the runtime is invalid."""
    if not site_packages.is_dir():
        return False, f"site-packages target missing: {site_packages}", False
    runner = python or Path(sys.executable)
    if not runner.is_file():
        return False, f"wrapper Python missing: {runner}", True
    if not os.access(runner, os.X_OK):
        return False, f"wrapper Python is not executable: {runner}", True
    import_timeout = _validated_import_timeout(import_timeout)
    code = (
        "import site\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"selected_site = Path({str(site_packages)!r}).resolve()\n"
        "site.addsitedir(str(selected_site))\n"
        "import mnemosyne_hermes\n"
        # mnemosyne_hermes deliberately degrades when its optional core imports
        # are unavailable. A wrapper needs the actual runtime, not merely that
        # importable fallback package, so prove the same core entry point as the
        # installer's CLI preflight in the selected interpreter.
        "import mnemosyne.core.beam\n"
        "origin = getattr(mnemosyne_hermes, '__file__', None)\n"
        "if not origin:\n"
        "    raise SystemExit('mnemosyne_hermes package has no file origin')\n"
        "actual = Path(origin).resolve()\n"
        "if not actual.is_file():\n"
        "    raise SystemExit(f'mnemosyne_hermes package origin is not a file: {actual}')\n"
        + "print(getattr(mnemosyne_hermes, '__version__', 'unknown'))\n"
    )
    try:
        # Python puts the subprocess working directory on sys.path.  Use a
        # fresh empty directory so validation cannot borrow a checkout or the
        # installer's current directory to satisfy the selected runtime.
        with tempfile.TemporaryDirectory(prefix="mnemosyne-wrapper-import-") as probe_cwd:
            result = subprocess.run(
                [str(runner), "-S", "-c", code],
                capture_output=True,
                text=True,
                timeout=import_timeout,
                # -S and a selected site directory isolate imports from the runner's
                # ambient site/user directories. Filtering PYTHONPATH prevents a
                # caller-controlled package shadowing that contract; filtering
                # PYTHONOPTIMIZE keeps assertion elision from changing probe behavior.
                env={
                    key: value
                    for key, value in os.environ.items()
                    if key not in {"PYTHONPATH", "PYTHONOPTIMIZE"}
                },
                cwd=probe_cwd,
            )
    except OSError as exc:
        return False, f"could not run wrapper Python {runner}: {exc}", True
    except subprocess.TimeoutExpired:
        return False, _timeout_diagnostic(runner, import_timeout, retry_hint=retry_hint), False
    if result.returncode == 0:
        return True, None, False
    return False, (result.stderr.strip() or result.stdout.strip() or "import failed")[:500], False


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

    wrapper_metadata = _wrapper_metadata(target, init_file) if not target.is_symlink() else None
    if wrapper_metadata is not None and wrapper_metadata.error is not None:
        return PluginState(
            status="invalid_wrapper",
            installed=False,
            target=target,
            mode="wrapper",
            wrapper_import_ok=False,
            wrapper_import_error=wrapper_metadata.error,
            message=wrapper_metadata.error,
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
        assert wrapper_metadata is not None
        wrapper_python = wrapper_metadata.python
        wrapper_site = wrapper_metadata.site_packages
        if wrapper_site is None:
            mode = "directory"
        else:
            wrapper_import_ok, wrapper_import_error, invalid_runtime = _check_wrapper_import(
                wrapper_site,
                wrapper_python,
                import_timeout=DEFAULT_WRAPPER_IMPORT_TIMEOUT,
                retry_hint=False,
            )
            if not wrapper_import_ok:
                status = "invalid_wrapper" if invalid_runtime else "stale_wrapper"
                return PluginState(
                    status=status,
                    installed=False,
                    target=target,
                    mode="wrapper",
                    wrapper_python=wrapper_python,
                    wrapper_site_packages=wrapper_site,
                    wrapper_import_ok=wrapper_import_ok,
                    wrapper_import_error=wrapper_import_error,
                    message=(
                        "Wrapper plugin has invalid runtime metadata: " + (wrapper_import_error or "unknown error")
                        if status == "invalid_wrapper"
                        else "Wrapper plugin exists but its target package cannot be imported."
                    ),
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

_MAX_WRAPPER_READ_BYTES = 4096


def _is_env_assignment(token: str) -> bool:
    """True if the token is a plain ``NAME=value`` shell assignment."""
    name, sep, value = token.partition("=")
    if not sep or not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    if not all(char.isalnum() or char == "_" for char in name):
        return False
    # Reject command/parameter substitution we cannot evaluate.
    return not value.startswith(("$", "`"))


def _strip_env_prefix(tokens: list[str]) -> list[str] | None:
    """Consume ``env`` assignments and no-arg flags, returning the command.

    Returns None for any option that takes an argument (``-u NAME``, ``-C DIR``,
    ``-S ...``) or is otherwise unrecognized, so an unfamiliar layout is rejected
    rather than misread.
    """
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            break
        if token in ("-", "-i", "--ignore-environment"):
            index += 1
            continue
        if token.startswith("-"):
            return None
        if _is_env_assignment(token):
            index += 1
            continue
        break
    return tokens[index:]


def _mentions_exec(tokens: list[str]) -> bool:
    """True if any token is an ``exec`` we did not parse as the leading form.

    Matches a bare ``exec`` token (``if true; then exec /x; fi``) and one nested
    inside a quoted argument (``sh -c "exec /x"``), which arrives as a single
    token. Deliberately exact on the first word rather than a substring search:
    a Python console script containing ``os.execv(...)`` or ``exec(code)`` is a
    direct launcher, and treating it as an unresolvable wrapper would break the
    pipx layout the launcher branch exists to serve.
    """
    return any(token.split(maxsplit=1)[:1] == ["exec"] for token in tokens if token)


def _wrapper_exec_target(path: Path) -> tuple[bool, str | None]:
    """Inspect a launcher for an ``exec`` handoff to another program.

    Returns ``(is_wrapper, target)``:
      * ``(False, None)`` - read the launcher and it has no ``exec`` handoff;
                             treat ``path`` as the binary.
      * ``(True, target)`` - wrapper execs ``target`` (a path or command name).
      * ``(True, None)``   - the handoff is unresolvable *or* undeterminable
                             (unreadable, larger than the read bound, or
                             unparseable); callers must not fall back to the
                             wrapper itself.

    Supported forms (trailing ``"$@"`` and arguments ignored)::

        exec /path/to/hermes "$@"
        exec "./hermes" "$@"
        VAR=val exec /path/to/hermes "$@"
        exec env [VAR=val ...] /path/to/hermes "$@"
    """
    # Every uncertain answer below is (True, None): "this may hand off, and we
    # cannot say where". (False, None) is a positive finding -- we read the
    # launcher and it is a binary -- because it licenses the caller to trust the
    # interpreter sitting beside it. Confusing the two is how an unrelated
    # sibling python gets bootstrapped (#618).
    try:
        with path.open("rb") as fh:
            raw = fh.read(_MAX_WRAPPER_READ_BYTES + 1)
    except OSError:
        return True, None

    if not raw.startswith(b"#!"):
        # A compiled console script has no exec line to read, and reading it is
        # how we know that. Binary mode keeps this decision independent of the
        # bytes decoding cleanly.
        return False, None

    if len(raw) > _MAX_WRAPPER_READ_BYTES:
        # A shebang script larger than the bound. Its handoff may sit past the
        # part we read, so calling it a direct executable would trust whatever
        # python happens to sit beside it.
        return True, None

    source = raw.decode("utf-8", errors="replace")

    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            # Unbalanced quoting. If the line could be the handoff, skipping it
            # would let the file reach the "not a wrapper" return below.
            if "exec" in line:
                return True, None
            continue

        index = 0
        while index < len(tokens) and _is_env_assignment(tokens[index]):
            index += 1
        tokens = tokens[index:]
        if not tokens or tokens[0] != "exec":
            # A handoff we cannot read as the supported leading form, such as
            # `if true; then exec /opt/hermes/bin/hermes "$@"; fi`. Skipping the
            # line lets the file reach the "no handoff" return and licenses the
            # caller to trust the launcher's sibling interpreter.
            if _mentions_exec(tokens):
                return True, None
            continue

        # This line hands off with exec, so the wrapper is never the target.
        rest = tokens[1:]
        if not rest or rest[0].startswith("-"):
            return True, None
        if rest[0] == "env":
            rest = _strip_env_prefix(rest[1:])
            if not rest:
                return True, None
        return True, rest[0]

    return False, None


def _resolve_exec_target(raw_target: str, wrapper: Path) -> Path | None:
    """Turn an ``exec`` target into a concrete executable path.

    Absolute paths are used as-is, explicit relative paths (``./hermes``,
    ``../bin/hermes``, ``bin/hermes``) resolve against the wrapper's own
    directory, and bare command names go through PATH. The raw target is
    classified before ``Path`` drops a leading ``./``, so a relative launcher is
    never shadowed by a same-named file in the process working directory.
    """
    expanded = os.path.expanduser(raw_target)

    if os.path.isabs(expanded):
        candidate = Path(expanded)
    elif os.sep in expanded or (os.altsep and os.altsep in expanded):
        candidate = wrapper.parent / expanded
    else:
        found = shutil.which(expanded)
        if not found:
            return None
        candidate = Path(found)

    return candidate if candidate.is_file() else None


def _resolve_hermes_bin(hermes_bin: str) -> Path | None:
    """Resolve the real Hermes executable from a launcher on PATH.

    Follows symlinks and shell wrappers that `exec` another binary (common for
    PATH shims), tracking visited paths to avoid symlink loops. A direct
    executable that is not a wrapper is returned as well, preserving discovery
    for pipx / package entry points.

    Returns None when resolution fails or the target is not executable.
    """
    path = Path(hermes_bin)
    seen: set[Path] = set()

    for _ in range(_MAX_HERMES_BIN_DEPTH):
        try:
            canonical = path.resolve()
        except (OSError, RuntimeError) as exc:
            LOGGER.debug(
                "Failed to resolve Hermes launcher %r: %s", hermes_bin, exc
            )
            return None

        if canonical in seen:
            LOGGER.debug("Hermes launcher %r has a symlink loop", hermes_bin)
            return None
        seen.add(canonical)

        if not canonical.is_file() or not os.access(canonical, os.X_OK):
            LOGGER.debug(
                "Hermes launcher %r resolves to non-executable %r",
                hermes_bin,
                canonical,
            )
            return None

        is_wrapper, exec_target = _wrapper_exec_target(canonical)
        if not is_wrapper:
            return canonical
        if exec_target is None:
            LOGGER.debug(
                "Hermes wrapper %r uses an unsupported exec form", hermes_bin
            )
            return None

        next_path = _resolve_exec_target(exec_target, canonical)
        if next_path is None:
            LOGGER.debug(
                "Hermes wrapper %r execs an invalid target %r",
                hermes_bin,
                exec_target,
            )
            return None

        path = next_path

    LOGGER.debug(
        "Hermes launcher %r exceeded maximum resolution depth (%d)",
        hermes_bin,
        _MAX_HERMES_BIN_DEPTH,
    )
    return None


def _is_windows_platform() -> bool:
    """Return whether the installer is running on native Windows."""
    return os.name == "nt"


DEFAULT_INSTALL_MODE_POSIX = "symlink"
DEFAULT_INSTALL_MODE_WINDOWS = "wrapper"


def default_install_mode() -> str:
    """Return the install mode to use when the caller did not choose one.

    Symlink mode is the historical default and stays the default everywhere it
    can actually succeed. It cannot succeed on native Windows: creating a
    symbolic link there requires ``SeCreateSymbolicLinkPrivilege``, which in
    practice means Developer Mode is enabled or the shell is elevated. Without
    one of those, ``os.symlink`` raises ``WinError 1314`` and the install fails.

    That is why native Windows installs have been reported as working for some
    people and not others: the outcome depends on a privilege nobody thinks to
    check. Wrapper mode writes a real plugin directory and needs no privilege,
    so it is the default there. An explicit ``--mode symlink`` still works for
    anyone who does hold the privilege. See issue #857.
    """
    return DEFAULT_INSTALL_MODE_WINDOWS if _is_windows_platform() else DEFAULT_INSTALL_MODE_POSIX


def _venv_python_candidates(venv_root: Path) -> tuple[Path, ...]:
    """Return supported interpreter paths in platform-appropriate order.

    Native Windows virtual environments keep their interpreter in
    ``Scripts/python.exe``. POSIX layouts remain ``bin/python`` followed by
    ``bin/python3``. The predicate that consumes these candidates still requires
    the adjacent ``pyvenv.cfg`` marker and executability.
    """
    posix = (venv_root / "bin" / "python", venv_root / "bin" / "python3")
    if _is_windows_platform():
        return (venv_root / "Scripts" / "python.exe", *posix)
    return posix


def _is_venv_bin_dir(bin_dir: Path) -> bool:
    """Return whether ``bin_dir`` is an interpreter directory of a real venv.

    ``pyvenv.cfg`` is what separates a venv from a directory that merely holds
    executables. It is the only cheap signal that discriminates the #618 case:
    ``~/.local/bin`` holds both a ``hermes`` launcher and an unrelated
    ``python``, so "the launcher sits next to a python" proves nothing on its
    own. The same marker applies to native Windows ``Scripts/`` layouts.
    """
    return (bin_dir.parent / "pyvenv.cfg").is_file()


def _is_validated_venv_python(candidate: Path) -> bool:
    """Return whether ``candidate`` is usable as Hermes' runtime.

    The single predicate every *implicitly discovered* candidate must satisfy,
    so the launcher, the known install roots, ``sys.prefix`` and ``VIRTUAL_ENV``
    cannot drift apart. Only ``--python`` bypasses it, deliberately: an
    explicitly named interpreter is reported against by the caller rather than
    silently swapped for another.

    The candidate must exist as a file, be executable (everything downstream
    runs it as ``<python> -m pip install ...``, so a file that cannot be
    executed is not a runtime), and live in a real virtualenv.
    """
    return (
        candidate.is_file()
        and os.access(candidate, os.X_OK)
        and _is_venv_bin_dir(candidate.parent)
    )


def _validate_explicit_python(explicit_python: str | Path | None) -> None:
    """Reject a supplied --python value that names no interpreter."""
    if explicit_python is not None and not str(explicit_python).strip():
        raise ValueError(
            "--python was given an empty value. Pass the path to Hermes' "
            "interpreter, or omit --python to let the installer find it."
        )


def _find_hermes_python(
    explicit_python: str | Path | None = None,
    hermes_home_path: str | Path | None = None,
) -> Optional[Path]:
    """Try to find Hermes' Python executable for dependency validation.

    ``hermes_home_path`` scopes known-root discovery to an explicit CLI home;
    otherwise the configured default Hermes home is used.

    Returns None when no *validated* Hermes runtime is found. A candidate is
    never returned on the strength of sitting next to the launcher alone: the
    caller bootstraps into whatever this returns, and an unvalidated sibling is
    typically the user's Homebrew or system interpreter (#618). The caller is
    expected to stop and point at ``--python`` rather than guess.

    NOTE: none of the branches below resolve the python symlink they return. A
    venv's bin/python is a symlink to the base interpreter; running the venv
    path activates the venv site-packages, running the resolved base path does
    NOT. Returning the resolved base interpreter silently drops the provider
    deps into the wrong environment.
    """
    # 0. An explicitly selected interpreter is authoritative. Return it as
    #    given, including when it looks wrong: the caller validates it and
    #    reports against the interpreter the user actually named, which beats
    #    silently probing for a different one.
    #
    #    None is the only "not supplied" signal. An empty or blank --python is a
    #    supplied value that names nothing, and falling through to discovery
    #    would answer with a different interpreter than the one the user asked
    #    for -- exactly the silent substitution this branch exists to prevent.
    if explicit_python is not None:
        _validate_explicit_python(explicit_python)
        selected = str(explicit_python)
        # Strip only to decide whether anything was named. A POSIX path may
        # legitimately begin or end with whitespace, so stripping the value we
        # return would select a different interpreter than the one requested,
        # or fail to find it at all.
        return Path(selected).expanduser()

    scoped_home = hermes_home_path is not None
    hermes_home_path = (
        Path(hermes_home_path).expanduser()
        if scoped_home
        else hermes_home()
    )

    # An explicit --hermes-home is a discovery boundary. It must never bind the
    # selected plugin home to a launcher, active venv, or global install that
    # belongs to another Hermes deployment.
    if not scoped_home:
        # 1. Resolve the `hermes` launcher on PATH back to its venv Python.
        # A pip/pipx-installed Hermes puts its console script next to the
        # interpreter that runs it, so the Python is a sibling of the resolved
        # binary. Covers the common /usr/local/lib/hermes-agent/venv layout that
        # the hardcoded roots below miss entirely.
        hermes_bin = shutil.which("hermes")
        if hermes_bin:
            resolved = _resolve_hermes_bin(hermes_bin)
            if resolved:
                for candidate in _venv_python_candidates(resolved.parent.parent):
                    if _is_validated_venv_python(candidate):
                        return candidate

    # Check the selected Hermes home first. With an explicit home it is the
    # only permitted root; otherwise retain the established global fallbacks.
    roots = [hermes_home_path / "hermes-agent"]
    if not scoped_home:
        roots.extend([
            Path.home() / "hermes-agent",
            Path("/opt/hermes/hermes-agent"),
            Path("/usr/local/lib/hermes-agent"),
            Path("/usr/lib/hermes-agent"),
        ])
    for root in roots:
        for venv_name in ("venv", ".venv"):
            for candidate in _venv_python_candidates(root / venv_name):
                if _is_validated_venv_python(candidate):
                    return candidate

    if scoped_home:
        return None

    # 3. Check if we're running inside Hermes' venv ourselves.
    #    `sys.prefix != sys.base_prefix` says the *running* interpreter is in a
    #    venv; it says nothing about the bin/python being asked for here, which
    #    can be absent or non-executable in a partially built environment.
    if sys.prefix != sys.base_prefix:
        for candidate in _venv_python_candidates(Path(sys.prefix)):
            if _is_validated_venv_python(candidate):
                return candidate

    # 4. Check VIRTUAL_ENV env var (uv-managed or explicit).
    #    This is an ordinary environment variable, not an assertion that a venv
    #    is live: a stale or hand-set `VIRTUAL_ENV=/usr` names `/usr/bin/python`
    #    and would hand bootstrap the system interpreter, which is the outcome
    #    this function exists to prevent.
    ve = os.environ.get("VIRTUAL_ENV")
    if ve:
        for candidate in _venv_python_candidates(Path(ve)):
            if _is_validated_venv_python(candidate):
                return candidate

    # Nothing validated. Better to stop and let the caller ask for --python
    # than to bootstrap into an interpreter that only looked plausible.
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
        print("  ✓ mnemosyne-hermes installed into Hermes' venv")
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
    """Check if Hermes' Python can import Mnemosyne's operational dependencies.

    Returns the version string if importable, None otherwise.
    """
    try:
        result = subprocess.run(
            [str(hermes_python), "-c",
             "import mnemosyne; print(mnemosyne.__version__); "
             "import mnemosyne.core.beam; import sqlite_vec"],
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


def _hermes_python_mismatch(hermes_python: Path) -> bool:
    """Return whether Hermes runs in a different environment from the installer.

    Compares the environment roots (``sys.prefix`` vs the Hermes interpreter's
    ``parent.parent``) instead of resolved interpreter paths: separate venvs
    created over one base interpreter resolve to the same binary while keeping
    different site-packages, so a resolved-path comparison misses the mismatch.

    Both sides are normalised with ``os.path.normpath`` first. It collapses
    ``.`` and ``..`` lexically and leaves symlinks alone, so a path spelled
    ``<venv>/bin/../bin/python`` still yields ``<venv>`` rather than
    ``<venv>/bin/..``, which names the same directory but compares unequal and
    would report one environment as two. Normalising lexically is what keeps
    venv identity intact, since following the symlink is the original defect.
    """
    hermes_root = Path(os.path.normpath(hermes_python)).parent.parent
    return hermes_root != Path(os.path.normpath(sys.prefix))


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


def _profile_links_preference_path(hermes_home_path: str | Path | None = None) -> Path:
    """Return the installer-managed profile-link preference for a Hermes home."""
    base = Path(hermes_home_path).expanduser() if hermes_home_path else hermes_home()
    return base / "plugins" / ".mnemosyne-profile-links.json"


def _atomic_write_profile_links_preference(path: Path, payload: bytes) -> None:
    """Replace the profile-link preference without truncating an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.staging-", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_profile_links_preference(
    enabled: bool,
    *,
    hermes_home_path: str | Path | None = None,
) -> None:
    """Persist the selected profile-link behavior for later upgrades."""
    path = _profile_links_preference_path(hermes_home_path)
    payload = (json.dumps({"link_profiles": enabled}) + "\n").encode("utf-8")
    _atomic_write_profile_links_preference(path, payload)


def _restore_profile_links_preference(path: Path, previous: bytes | None) -> None:
    """Restore a preference snapshot after a failed plugin replacement."""
    if previous is None:
        path.unlink(missing_ok=True)
        return
    _atomic_write_profile_links_preference(path, previous)


def profile_links_enabled(*, hermes_home_path: str | Path | None = None) -> bool:
    """Return the selected profile-link behavior for a Hermes home.

    New installs persist the explicit selection, which lets upgrades distinguish
    the default enabled behavior from an explicit root-only installation even
    when no opted-in child profile exists yet. Existing installs without the
    preference file retain the legacy observed-link fallback.
    """
    try:
        preference = json.loads(
            _profile_links_preference_path(hermes_home_path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        preference = None
    if isinstance(preference, dict) and isinstance(preference.get("link_profiles"), bool):
        return preference["link_profiles"]

    target = plugin_target_dir(hermes_home_path)
    if not (target.is_symlink() or target.exists()):
        return False
    try:
        expected = target.resolve()
    except OSError:
        return False
    for profile_home in _iter_mnemosyne_profiles(hermes_home_path):
        profile_target = profile_home / "plugins" / PLUGIN_NAME
        if not profile_target.is_symlink():
            continue
        try:
            if profile_target.resolve() == expected:
                return True
        except OSError:
            continue
    return False


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


def _snapshot_profile_link(path: Path) -> _ProfileLinkSnapshot | None:
    """Capture one stable symlink identity, or None if it changes while read."""
    try:
        before = path.lstat()
        if not stat.S_ISLNK(before.st_mode):
            return None
        link_target = os.readlink(path)
        after = path.lstat()
    except OSError:
        return None
    if (
        not stat.S_ISLNK(after.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
    ):
        return None
    return _ProfileLinkSnapshot(
        path=path,
        link_target=link_target,
        device=after.st_dev,
        inode=after.st_ino,
    )


def _same_profile_link_identity(
    left: _ProfileLinkSnapshot, right: _ProfileLinkSnapshot
) -> bool:
    """Return whether two snapshots identify the same symlink object."""
    return (
        left.link_target == right.link_target
        and left.device == right.device
        and left.inode == right.inode
    )


def _recognized_profile_links(
    *,
    hermes_home_path: str | Path | None = None,
    recognized_targets: tuple[Path, ...] = (),
) -> list[_ProfileLinkSnapshot]:
    """Return profile links that currently resolve to a recognized target.

    Profiles are considered even if they no longer opt in, but unrelated links
    and real directories are ignored.
    """
    base = Path(hermes_home_path).expanduser() if hermes_home_path else hermes_home()
    profiles_dir = base / "profiles"
    if not profiles_dir.is_dir():
        return []
    recognized = {_resolve_package_dir().resolve()}
    for target in recognized_targets:
        try:
            recognized.add(target.resolve())
        except OSError:
            continue
    links: list[_ProfileLinkSnapshot] = []
    for child in sorted(profiles_dir.iterdir()):
        if child.is_symlink():
            continue
        target = child / "plugins" / PLUGIN_NAME
        before = _snapshot_profile_link(target)
        if before is None:
            continue
        try:
            resolved = target.resolve()
        except OSError:
            continue
        after = _snapshot_profile_link(target)
        if (
            after is not None
            and _same_profile_link_identity(before, after)
            and resolved in recognized
        ):
            links.append(after)
    return links


def _unlink_profile_links(links: list[_ProfileLinkSnapshot]) -> None:
    """Quarantine each candidate, then remove only the snapshotted symlink."""
    for snapshot in links:
        target = snapshot.path
        if _snapshot_profile_link(target) is None:
            continue
        quarantine_parent: Path | None = None
        quarantined: Path | None = None
        moved = False
        try:
            quarantine_parent = Path(
                tempfile.mkdtemp(prefix=f".{target.name}.unlink-", dir=target.parent)
            )
            quarantined = quarantine_parent / target.name
            target.replace(quarantined)
            moved = True
            quarantined_snapshot = _snapshot_profile_link(quarantined)
            if quarantined_snapshot is not None and _same_profile_link_identity(
                snapshot, quarantined_snapshot
            ):
                quarantined.unlink()
                moved = False
                print(f"  Removed profile link: {target}")
            elif not target.is_symlink() and not target.exists():
                quarantined.replace(target)
                moved = False
        except OSError:
            raise
        finally:
            if moved and quarantined is not None:
                try:
                    if not target.is_symlink() and not target.exists():
                        quarantined.replace(target)
                        moved = False
                except OSError:
                    pass
            if moved and quarantined is not None:
                print(
                    f"  Profile entry changed during cleanup; preserved at: {quarantined}",
                    file=sys.stderr,
                )
            if quarantine_parent is not None:
                try:
                    quarantine_parent.rmdir()
                except OSError:
                    pass


def _unlink_all_profiles(
    *,
    hermes_home_path: str | Path | None = None,
    recognized_targets: tuple[Path, ...] = (),
) -> None:
    """Remove profile links that resolve to a recognized Mnemosyne target."""
    _unlink_profile_links(
        _recognized_profile_links(
            hermes_home_path=hermes_home_path,
            recognized_targets=recognized_targets,
        )
    )


def _unlink_profile_links_or_restore_preference(
    links: list[_ProfileLinkSnapshot],
    preference_path: Path,
    previous_preference: bytes | None,
) -> None:
    """Fail root-only cleanup and restore its preference when link removal fails."""
    try:
        _unlink_profile_links(links)
    except OSError:
        try:
            _restore_profile_links_preference(preference_path, previous_preference)
        except OSError:
            print(
                f"⚠ Profile-link preference rollback failed; inspect: {preference_path}",
                file=sys.stderr,
            )
        raise


def _prepare_plugin_target(
    base: Path, target: Path, *, force: bool, remove_target: bool = True
) -> None:
    """Migrate legacy names and, optionally, remove an existing plugin target."""
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
        if remove_target:
            if target.is_symlink():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)


def _is_wrapper_plugin_target(target: Path) -> bool:
    """Return whether ``target`` is a generated Mnemosyne wrapper directory."""
    if target.is_symlink() or not target.is_dir():
        return False
    if (target / "mnemosyne-wrapper.json").exists():
        return True
    python, site_packages = _extract_wrapper_metadata(target / "__init__.py")
    return python is not None or site_packages is not None


def _validated_wrapper_environment(
    python: str | Path | None,
    *,
    import_timeout: float = DEFAULT_WRAPPER_IMPORT_TIMEOUT,
) -> tuple[Path, Path]:
    """Validate the selected wrapper runtime before touching an installed plugin."""
    if python is None:
        wrapper_python = Path(sys.executable)
    else:
        selected = str(python)
        if not selected.strip():
            raise ValueError(
                "--python was given an empty value. Pass the path to Hermes' "
                "interpreter, or omit --python to let the installer find it."
            )
        # The import probe runs from an isolated temporary working directory.
        # Make an explicit relative path independent of that cwd before either
        # validation or the probe uses it.  ``absolute()`` is lexical here:
        # ``resolve()`` would follow a venv's bin/python symlink to its base
        # interpreter and lose the selected virtual environment.
        wrapper_python = Path(selected).expanduser().absolute()
    if not wrapper_python.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {wrapper_python}")
    import_timeout = _validated_import_timeout(import_timeout)
    site_packages = _site_packages_for_python(wrapper_python, import_timeout=import_timeout)
    import_ok, import_error, _invalid_runtime = _check_wrapper_import(
        site_packages, wrapper_python, import_timeout=import_timeout
    )
    if not import_ok:
        raise RuntimeError(
            "Selected Python environment cannot import required mnemosyne core "
            f"(mnemosyne.core.beam): {import_error}"
        )
    return wrapper_python, site_packages


def _replace_plugin_target_with_staged(target: Path, staged: Path) -> None:
    """Swap a fully written wrapper into place, restoring the old target on error."""
    previous: Path | None = None
    previous_parent: Path | None = None
    if target.is_symlink() or target.exists():
        previous_parent = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.previous-", dir=target.parent)
        )
        previous = previous_parent / target.name
        target.replace(previous)
    try:
        staged.replace(target)
    except Exception:
        restored_previous = False
        if previous is not None and not target.exists() and not target.is_symlink():
            try:
                previous.replace(target)
            except OSError:
                # Preserve the failed swap as the primary exception. Keeping the
                # backup is safer than masking it with a rollback cleanup error.
                print(
                    f"⚠ Wrapper rollback failed; previous plugin retained at: {previous}",
                    file=sys.stderr,
                )
            else:
                restored_previous = True
        if restored_previous and previous_parent is not None:
            try:
                previous_parent.rmdir()
            except OSError:
                pass
        raise
    else:
        if previous is not None:
            try:
                if previous.is_symlink():
                    previous.unlink()
                elif previous.is_dir():
                    shutil.rmtree(previous)
                else:
                    previous.unlink()
            except OSError:
                pass
        if previous_parent is not None:
            try:
                previous_parent.rmdir()
            except OSError:
                pass


def _write_wrapper_plugin(target: Path, *, python: Path, site_packages: Path) -> None:
    """Create a persistent Hermes plugin shim that imports from a selected env."""
    target.mkdir(parents=True, exist_ok=False)
    # Keep a virtualenv interpreter symlink intact: resolving it would select
    # the base interpreter while retaining the virtualenv's site-packages.
    python = python.expanduser().absolute()
    site_packages = site_packages.resolve()
    manifest = {
        "schema_version": 1,
        "python": str(python),
        "site_packages": str(site_packages),
        "package": "mnemosyne_hermes",
    }
    bootstrap_source = """\"\"\"Bootstrap a generated Mnemosyne Hermes wrapper using only the stdlib.\"\"\"
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import re
import site as site_module
import sys


def _guard_selected_site_packages_python_compatibility(selected_site_packages: Path) -> None:
    \"\"\"Reject a selected virtualenv that targets another Python minor version.\"\"\"
    selected_site_packages = selected_site_packages.resolve()
    # Standard POSIX layouts put pyvenv.cfg three levels above site-packages
    # (/venv/lib/pythonX/site-packages); Windows needs only two. Do not let
    # this early bootstrap probe inspect arbitrary filesystem ancestors.
    for candidate in (selected_site_packages, *selected_site_packages.parents[:3]):
        config_path = candidate / "pyvenv.cfg"
        if not config_path.exists():
            continue
        selected_version = None
        try:
            config_text = config_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            config_text = ""
        for line in config_text.splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip().lower() in {"version_info", "version"}:
                match = re.search(r"(?<!\\d)(\\d+)\\.(\\d+)(?!\\d)", value)
                if match:
                    selected_version = (int(match.group(1)), int(match.group(2)))
                    break
        runtime_version = sys.version_info[:2]
        if selected_version is None or selected_version != runtime_version:
            selected_text = (
                f"{selected_version[0]}.{selected_version[1]}"
                if selected_version is not None
                else "unknown"
            )
            raise RuntimeError(
                "Mnemosyne runtime Python compatibility error: "
                f"runtime Python {runtime_version[0]}.{runtime_version[1]}; "
                f"selected Mnemosyne environment Python {selected_text}. "
                "Recreate the Mnemosyne environment using Hermes' Python, "
                "then reinstall mnemosyne-hermes."
            )
        return


def activate() -> dict[str, object]:
    \"\"\"Validate the wrapper and import its selected package identity.\"\"\"
    manifest_path = Path(__file__).with_name("mnemosyne-wrapper.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid Mnemosyne wrapper manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("Invalid Mnemosyne wrapper manifest schema")
    if manifest.get("package") != "mnemosyne_hermes":
        raise RuntimeError("Invalid Mnemosyne wrapper package")
    python = manifest.get("python")
    site_packages = manifest.get("site_packages")
    if not isinstance(python, str) or not isinstance(site_packages, str):
        raise RuntimeError("Invalid Mnemosyne wrapper manifest paths")
    python_path = Path(python)
    site_path = Path(site_packages)
    if (
        not python_path.is_absolute()
        or not python_path.is_file()
        or not os.access(python_path, os.X_OK)
    ):
        raise RuntimeError("Invalid Mnemosyne wrapper Python executable")
    if not site_path.is_absolute() or not site_path.is_dir():
        raise RuntimeError("Invalid Mnemosyne wrapper site-packages target")
    _guard_selected_site_packages_python_compatibility(site_path)
    package_root = site_path / "mnemosyne_hermes"
    package_init = package_root / "__init__.py"
    expected_root = package_root.resolve() if package_init.is_file() else None
    site = str(site_path)
    while site in sys.path:
        sys.path.remove(site)
    # Process .pth files from the selected runtime as well as direct packages.
    # Setuptools' PEP 660 editable installs use a .pth-installed finder, so a
    # bare sys.path insertion would make a valid selected environment fail.
    site_module.addsitedir(site)
    while site in sys.path:
        sys.path.remove(site)
    sys.path.insert(0, site)
    def from_selected_package(module: object) -> bool:
        if expected_root is None:
            return False
        source = getattr(module, "__file__", None)
        if not source:
            return False
        try:
            Path(source).resolve().relative_to(expected_root)
        except (OSError, ValueError):
            return False
        return True

    package_name = "mnemosyne_hermes"
    cached_package = sys.modules.get(package_name)
    if cached_package is not None and not from_selected_package(cached_package):
        names_to_remove = [
            name for name in sys.modules
            if name == package_name or name.startswith(package_name + ".")
        ]
    else:
        names_to_remove = [
            name for name, module in sys.modules.items()
            if name.startswith(package_name + ".") and not from_selected_package(module)
        ]
    for name in names_to_remove:
        sys.modules.pop(name, None)

    selected_package = importlib.import_module(package_name)
    if expected_root is not None and not from_selected_package(selected_package):
        raise RuntimeError("Mnemosyne wrapper imported package from an unexpected origin")
    return manifest
"""
    init_source = """\"\"\"Persistent Mnemosyne Hermes plugin wrapper.\"\"\"
from ._mnemosyne_bootstrap import activate as _activate

_activate()

# Hermes discovery marker: register_memory_provider / MnemosyneMemoryProvider
from mnemosyne_hermes import *  # noqa: F401,F403,E402
"""
    cli_source = """\"\"\"Hermes CLI wrapper for the selected Mnemosyne package.\"\"\"
import importlib.util
from pathlib import Path

_bootstrap_path = Path(__file__).resolve().with_name("_mnemosyne_bootstrap.py")
_bootstrap_spec = importlib.util.spec_from_file_location(
    f"{__name__}._mnemosyne_bootstrap", _bootstrap_path
)
if _bootstrap_spec is None or _bootstrap_spec.loader is None:
    raise ImportError("Cannot load Mnemosyne wrapper bootstrap")
_bootstrap_module = importlib.util.module_from_spec(_bootstrap_spec)
_bootstrap_spec.loader.exec_module(_bootstrap_module)
_activate = _bootstrap_module.activate

_activate()

from mnemosyne_hermes.cli import *  # noqa: F401,F403,E402
"""
    (target / "mnemosyne-wrapper.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target / "_mnemosyne_bootstrap.py").write_text(bootstrap_source, encoding="utf-8")
    (target / "__init__.py").write_text(init_source, encoding="utf-8")
    (target / "cli.py").write_text(cli_source, encoding="utf-8")
    _copy_plugin_yaml(target)


def install_plugin(
    *,
    hermes_home_path: str | Path | None = None,
    force: bool = False,
    mode: str | None = None,
    python: str | Path | None = None,
    import_timeout: float = DEFAULT_WRAPPER_IMPORT_TIMEOUT,
    migrate_wrapper_to_symlink: bool = False,
    link_profiles: bool = True,
) -> Path:
    """Install the Mnemosyne provider into Hermes' user plugin directory.

    ``mode='symlink'`` keeps the historical behavior. ``mode='wrapper'``
    creates a real persistent plugin directory containing a tiny shim that
    activates the selected interpreter's site-packages (including editable
    install ``.pth`` files) and imports ``mnemosyne_hermes`` from there.
    ``link_profiles`` preserves the historical
    opted-in profile fan-out by default; set it False to install only at the
    selected Hermes home.
    """
    if mode is None:
        mode = default_install_mode()
    if mode not in {"symlink", "wrapper"}:
        raise ValueError("mode must be 'symlink' or 'wrapper'")
    if migrate_wrapper_to_symlink and (mode != "symlink" or not force):
        raise ValueError("migrate_wrapper_to_symlink requires mode='symlink' and force=True")

    source = _resolve_package_dir()
    if not source.is_dir():
        raise FileNotFoundError(f"mnemosyne_hermes package not found at {source}")

    base = Path(hermes_home_path).expanduser() if hermes_home_path else hermes_home()
    target = plugin_target_dir(hermes_home_path)
    if (
        mode == "symlink"
        and force
        and _is_wrapper_plugin_target(target)
        and not migrate_wrapper_to_symlink
    ):
        raise RuntimeError(
            "Refusing to replace an existing wrapper with a symlink. "
            "Re-run with --migrate-wrapper-to-symlink --force to migrate intentionally."
        )

    recognized_profile_targets = [source]
    # Preserve a validated previous provider target before --force replaces it.
    if _provider_init_is_mnemosyne(target / "__init__.py"):
        try:
            recognized_profile_targets.append(target.resolve())
        except OSError:
            pass
        wrapper_metadata = _wrapper_metadata(target, target / "__init__.py")
        if wrapper_metadata.error is None and wrapper_metadata.site_packages is not None:
            recognized_profile_targets.append(
                wrapper_metadata.site_packages / "mnemosyne_hermes"
            )
    profile_links_to_unlink = (
        _recognized_profile_links(
            hermes_home_path=hermes_home_path,
            recognized_targets=tuple(recognized_profile_targets),
        )
        if not link_profiles
        else []
    )
    preference_path = _profile_links_preference_path(hermes_home_path)
    try:
        previous_preference = preference_path.read_bytes()
    except FileNotFoundError:
        previous_preference = None
        if profile_links_enabled(hermes_home_path=hermes_home_path):
            # Preserve an effective legacy opt-in even if replacing the root
            # target changes the no-file fallback before cleanup can fail.
            previous_preference = b'{"link_profiles": true}\n'

    if mode == "symlink":
        if migrate_wrapper_to_symlink and _is_wrapper_plugin_target(target):
            print(
                "  ⚠ Migrating existing Mnemosyne wrapper to a symlink; "
                "the wrapper's selected Python will no longer be used."
            )
        _write_profile_links_preference(link_profiles, hermes_home_path=hermes_home_path)
        try:
            _prepare_plugin_target(base, target, force=force)
            try:
                os.symlink(str(source), str(target))
            except OSError as error:
                if _is_windows_symlink_privilege_error(error):
                    raise _WindowsSymlinkPrivilegeError from error
                raise
        except Exception:
            try:
                _restore_profile_links_preference(preference_path, previous_preference)
            except OSError:
                print(
                    f"⚠ Profile-link preference rollback failed; inspect: {preference_path}",
                    file=sys.stderr,
                )
            raise
        if link_profiles:
            _link_all_profiles(source, hermes_home_path=hermes_home_path, force=force)
        else:
            _unlink_profile_links_or_restore_preference(
                profile_links_to_unlink,
                preference_path,
                previous_preference,
            )
        return target

    # Validate and fully write the replacement before removing a working wrapper.
    # In particular, a bad --python or a selected environment missing this package
    # must leave the existing wrapper and every profile link untouched.
    wrapper_python, site_packages = _validated_wrapper_environment(
        python, import_timeout=import_timeout
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    staged = staging_parent / target.name
    preference_written = False
    try:
        _write_wrapper_plugin(staged, python=wrapper_python, site_packages=site_packages)
        _write_profile_links_preference(link_profiles, hermes_home_path=hermes_home_path)
        preference_written = True
        _prepare_plugin_target(base, target, force=force, remove_target=False)
        _replace_plugin_target_with_staged(target, staged)
    except Exception:
        if preference_written:
            try:
                _restore_profile_links_preference(preference_path, previous_preference)
            except OSError:
                print(
                    f"⚠ Profile-link preference rollback failed; inspect: {preference_path}",
                    file=sys.stderr,
                )
        raise
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)
    if link_profiles:
        _link_all_profiles(target, hermes_home_path=hermes_home_path, force=force)
    else:
        _unlink_profile_links_or_restore_preference(
            profile_links_to_unlink,
            preference_path,
            previous_preference,
        )
    return target


def uninstall_plugin(*, hermes_home_path: str | Path | None = None) -> Path:
    """Remove the Mnemosyne provider symlink from Hermes' user plugin directory."""
    target = plugin_target_dir(hermes_home_path)
    _unlink_all_profiles(
        hermes_home_path=hermes_home_path,
        recognized_targets=(_resolve_package_dir(), target),
    )
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    _profile_links_preference_path(hermes_home_path).unlink(missing_ok=True)
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

    preference_path = _profile_links_preference_path(hermes_home_path)
    if preference_path.exists():
        if dry_run:
            actions.append(f"Would remove: {preference_path}")
        else:
            preference_path.unlink()
            actions.append(f"Removed: {preference_path}")

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


def _distribution_version(distribution: str) -> str:
    """Return an installed distribution version without importing package globals."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _runtime_python_json_error(message: str) -> int:
    """Print a runtime Python error using the command's JSON contract."""
    print(json.dumps({"ok": False, "error": message}))
    return 1


def _runtime_python_json(
    explicit_python: str | Path | None = None,
    hermes_home_path: str | Path | None = None,
) -> int:
    """Print the selected Hermes interpreter and version as JSON."""
    try:
        python = _find_hermes_python(
            explicit_python=explicit_python,
            hermes_home_path=hermes_home_path,
        )
    except ValueError as exc:
        return _runtime_python_json_error(str(exc))

    if python is None:
        return _runtime_python_json_error(
            "Could not identify Hermes' Python. Pass --python "
            "/path/to/hermes/venv/bin/python."
        )
    if not python.is_file() or not os.access(python, os.X_OK):
        return _runtime_python_json_error(
            f"Selected Python is not an executable file: {python}"
        )

    try:
        result = subprocess.run(
            [
                str(python),
                "-I",
                "-S",
                "-B",
                "-c",
                (
                    "import json, platform; "
                    "print(json.dumps({'runtime': 'python', "
                    "'version': platform.python_version()}))"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _runtime_python_json_error(f"Could not run selected Python at {python}: {exc}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = None
    version = payload.get("version") if isinstance(payload, dict) else None
    if (
        result.returncode != 0
        or not isinstance(payload, dict)
        or payload.get("runtime") != "python"
        or not isinstance(version, str)
        or not version
    ):
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        if result.returncode == 0 and not result.stderr.strip():
            detail = "unexpected runtime probe response"
        return _runtime_python_json_error(
            f"Could not read Python version from {python}: {detail}"
        )

    print(json.dumps({"ok": True, "python": str(python), "version": version}))
    return 0



def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mnemosyne-hermes",
        description="Install the Mnemosyne memory provider for Hermes Agent.",
    )
    parser.add_argument(
        "--hermes-home",
        help="Hermes home directory. Defaults to HERMES_HOME or ~/.hermes.",
    )
    parser.add_argument("--version", action="store_true", help="Show installed package versions and exit.")

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
        default=None,
        help=(
            "Install mode. Defaults to symlink, except on native Windows where "
            "it defaults to the persistent wrapper shim because symbolic links "
            "there require Developer Mode or an elevated shell."
        ),
    )
    install.add_argument(
        "--python",
        dest="python",
        help=(
            "Hermes' Python interpreter. Authoritative when given: skips launcher "
            "and install-root discovery. Also selects the site-packages a wrapper "
            "install imports from."
        ),
    )
    install.add_argument(
        "--import-timeout",
        type=_parse_import_timeout,
        default=DEFAULT_WRAPPER_IMPORT_TIMEOUT,
        metavar="SECONDS",
        help=(
            "Wrapper validation timeout in seconds "
            f"(default: {DEFAULT_WRAPPER_IMPORT_TIMEOUT:g})."
        ),
    )
    install.add_argument(
        "--migrate-wrapper-to-symlink",
        action="store_true",
        help=(
            "With --mode symlink and --force, intentionally replace an existing "
            "wrapper with the default symlink install."
        ),
    )
    install.add_argument(
        "--no-profile-links",
        dest="link_profiles",
        action="store_false",
        default=True,
        help="Install only at the selected Hermes home; do not link opted-in child profiles.",
    )
    subparsers.add_parser(
        "uninstall",
        help="Remove Mnemosyne from Hermes' memory provider plugin directory.",
    )
    subparsers.add_parser(
        "status",
        help="Show whether Mnemosyne is installed for Hermes memory discovery.",
    )
    subparsers.add_parser("version", help="Show installed package versions.")
    runtime_python = subparsers.add_parser(
        "runtime-python",
        help="Report the Python interpreter selected for Hermes.",
    )
    runtime_python.add_argument(
        "--json",
        action="store_true",
        required=True,
        help="Emit the selected interpreter and version as JSON.",
    )
    runtime_python.add_argument(
        "--python",
        help="Explicit Hermes Python interpreter; bypasses automatic discovery.",
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
    mode: str | None = None,
    python: str | Path | None = None,
    import_timeout: float = DEFAULT_WRAPPER_IMPORT_TIMEOUT,
    migrate_wrapper_to_symlink: bool = False,
    link_profiles: bool = True,
) -> int:
    """Core install logic — check deps, bootstrap Hermes venv if needed, create symlink.

    Returns 0 on success, 1 on failure.
    Can be called from the CLI ``install`` subcommand or programmatically
    (e.g., from ``upgrade.py`` after upgrading the pip package).
    """
    if mode is None:
        mode = default_install_mode()

    # Validate supplied input before the symlink-mode installer preflight. A
    # blank value is explicit, not an omitted --python request, and must never
    # be hidden by an earlier preflight return.
    _validate_explicit_python(python)

    # Wrapper validation belongs to its selected environment, including the
    # interpreter discovered when --python is omitted. Only symlink installs
    # retain the installer-Python core preflight and bootstrap path.
    if mode == "symlink" and not check_mnemosyne_core():
        print(
            "  mnemosyne-memory NOT found in this Python. Install it first:\n"
            "    pip install mnemosyne-hermes[all]",
            file=sys.stderr,
        )
        return 1

    # Both install modes need a Hermes interpreter. Symlink installs bootstrap
    # it when needed; wrapper installs validate it and record its metadata. An
    # explicit --python remains authoritative through _find_hermes_python().
    hermes_python = _find_hermes_python(
        explicit_python=python,
        hermes_home_path=hermes_home_path,
    )
    if mode == "wrapper" and hermes_python is None:
        print(
            "\n  ⚠ Could not identify Hermes' Python for wrapper mode.\n"
            "     Pass --python /path/to/hermes/venv/bin/python to select the "
            "environment the wrapper must import from.",
            file=sys.stderr,
        )
        return 1
    if mode == "symlink" and hermes_python is None:
        # Discovery found no validated Hermes runtime, so there is nothing safe
        # to bootstrap into. Before #618 this path guessed at the launcher's
        # sibling, which is typically the user's Homebrew or system interpreter.
        print(
            "\n  ⚠ Could not identify Hermes' Python.\n"
            "     No `hermes` launcher on PATH resolved into a virtual environment,\n"
            "     and no Hermes install root contains one.\n\n"
            "  Point the installer at it directly:\n"
            "    mnemosyne-hermes install --python /path/to/hermes/venv/bin/python\n\n"
            "  `mnemosyne-hermes install --dry-run` shows what discovery found.",
            file=sys.stderr,
        )
        # --no-bootstrap already means "do not touch Hermes' venv", so there is
        # no wrong-interpreter install to prevent and the run continues without
        # dependency validation, as it did before. Failing here instead would
        # also preempt the guard that refuses to replace an existing wrapper
        # install, turning a data-safety message into a discovery message.
        if not no_bootstrap:
            return 1
        print("     Continuing without dependency validation (--no-bootstrap).", file=sys.stderr)
    # Compare the paths as selected, not resolved. A venv's bin/python resolves
    # to its base interpreter, so resolving both sides reports a venv and the
    # base install as the same runtime and skips the check that bootstraps
    # Hermes' venv (#618).
    if mode == "symlink" and hermes_python and hermes_python != Path(sys.executable):
        hermes_core = check_mnemosyne_core_for_hermes_python(hermes_python)
        if hermes_core is None:
            print(f"\n  ⚠ Hermes' Python at {hermes_python} can't import mnemosyne core.")
            print(f"     mnemosyne-hermes is installed in YOUR Python ({sys.executable}),")
            print("     but Hermes runs from a different venv.\n")
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

    try:
        target = install_plugin(
            hermes_home_path=hermes_home_path,
            force=force,
            mode=mode,
            python=hermes_python if mode == "wrapper" else python,
            import_timeout=import_timeout,
            migrate_wrapper_to_symlink=migrate_wrapper_to_symlink,
            link_profiles=link_profiles,
        )
    except _WindowsSymlinkPrivilegeError:
        print(
            "\n  ⚠ Windows symbolic-link privilege is unavailable (WinError 1314).\n"
            "     Symlink mode was requested explicitly, so the installer did not\n"
            "     switch modes on your behalf. Either enable Developer Mode or run\n"
            "     with an account granted the symbolic-link privilege, or install in\n"
            "     persistent wrapper mode, which needs no privilege and is the\n"
            "     default on Windows when no mode is given.\n",
            file=sys.stderr,
        )
        if hermes_python is not None:
            print(
                "  Persistent no-symlink-privilege alternative:\n"
                f"    {_windows_wrapper_retry_command(hermes_python)}\n",
                file=sys.stderr,
            )
        else:
            print(
                "  A safe wrapper retry cannot be generated because Hermes' Python was "
                "not resolved.\n"
                "  Locate the Hermes interpreter, then re-run with --python "
                "<path-to-hermes-python>.\n",
                file=sys.stderr,
            )
        return 1
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
    if args.version or args.command == "version":
        print(f"Mnemosyne {_distribution_version('mnemosyne-memory')}")
        print(f"Mnemosyne Hermes {_distribution_version('mnemosyne-hermes')}")
        return 0
    command = args.command or "install"

    try:
        if command == "install":
            # argparse leaves --mode as None when the user did not choose one, so
            # resolve it here, before anything reads it. Every downstream path
            # (dry-run reporting included) then sees a concrete mode.
            if getattr(args, "mode", None) is None:
                args.mode = default_install_mode()
                if args.mode == DEFAULT_INSTALL_MODE_WINDOWS and _is_windows_platform():
                    print(
                        "  Native Windows detected: installing in persistent wrapper mode.\n"
                        "  Symbolic links need Developer Mode or an elevated shell here, so\n"
                        "  wrapper mode is the default. Pass --mode symlink to override."
                    )

            # Dry-run: just show what would happen
            hermes_python = _find_hermes_python(
                explicit_python=getattr(args, "python", None),
                hermes_home_path=args.hermes_home,
            )
            if args.mode == "wrapper" and hermes_python is None:
                print(
                    "\n  ⚠ Could not identify Hermes' Python for wrapper mode.\n"
                    "     Pass --python /path/to/hermes/venv/bin/python to select "
                    "the environment the wrapper must import from.",
                    file=sys.stderr,
                )
                return 1
            target = plugin_target_dir(args.hermes_home)
            if getattr(args, "dry_run", False):
                invalid_wrapper_migration_args = (
                    getattr(args, "migrate_wrapper_to_symlink", False)
                    and (
                        getattr(args, "mode", "symlink") != "symlink"
                        or not getattr(args, "force", False)
                    )
                )
                refuses_wrapper_migration = (
                    getattr(args, "mode", "symlink") == "symlink"
                    and getattr(args, "force", False)
                    and _is_wrapper_plugin_target(target)
                    and not getattr(args, "migrate_wrapper_to_symlink", False)
                )
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
                print(f"  Will link opted-in profiles: {bool(getattr(args, 'link_profiles', True))}")
                print(f"  Skill target file: {skill.target}")
                print(f"  Skill state: {skill.status}")
                print(f"  Skill action: {skill_plan.message}")
                if getattr(args, "mode", "symlink") == "wrapper":
                    # Match run_install(): wrapper metadata and validation use
                    # the discovered Hermes runtime unless --python selected one.
                    # Preserve a venv's python symlink; absolute() is lexical.
                    assert hermes_python is not None
                    wrapper_python = hermes_python.absolute()
                    print(f"  Wrapper Python: {wrapper_python}")
                    if wrapper_python.is_file():
                        print(
                            "  Wrapper site-packages: "
                            f"{_site_packages_for_python(wrapper_python, import_timeout=args.import_timeout)}"
                        )
                    print(f"  Wrapper import timeout: {args.import_timeout:g} seconds")
                print(f"  Will force: {bool(getattr(args, 'force', False))}")
                if getattr(args, "migrate_wrapper_to_symlink", False):
                    print("  Will allow wrapper-to-symlink migration: yes")
                if invalid_wrapper_migration_args:
                    print(
                        "  Will refuse --migrate-wrapper-to-symlink unless "
                        "--mode symlink and --force are both set."
                    )
                if refuses_wrapper_migration:
                    print(
                        "  Will refuse to replace the existing wrapper without "
                        "--migrate-wrapper-to-symlink."
                    )
                # Only a symlink install bootstraps Hermes' environment;
                # run_install() does not even look for an interpreter in wrapper
                # mode. Reporting a bootstrap here described something that
                # would never run, and `--python` made it print for wrapper
                # installs specifically.
                if hermes_python and getattr(args, "mode", "symlink") == "symlink":
                    print(f"  Will bootstrap: {not getattr(args, 'no_bootstrap', False)}")
                return 1 if invalid_wrapper_migration_args or refuses_wrapper_migration else 0

            return run_install(
                force=getattr(args, "force", False),
                hermes_home_path=args.hermes_home,
                no_bootstrap=getattr(args, "no_bootstrap", False),
                mode=getattr(args, "mode", "symlink"),
                python=getattr(args, "python", None),
                import_timeout=getattr(args, "import_timeout", DEFAULT_WRAPPER_IMPORT_TIMEOUT),
                migrate_wrapper_to_symlink=getattr(args, "migrate_wrapper_to_symlink", False),
                link_profiles=getattr(args, "link_profiles", True),
            )

        if command == "uninstall":
            target = uninstall_plugin(hermes_home_path=args.hermes_home)
            print(f"Removed. Symlink at {target} deleted.")
            return 0

        if command == "status":
            state = plugin_state(hermes_home_path=args.hermes_home)
            target = state.target
            installed = state.installed
            hermes_python = _find_hermes_python(hermes_home_path=args.hermes_home)
            print("Status for mnemosyne-hermes plugin")
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
                    print("  Type: directory (not symlink)")
                print("  Plugin:    installed ✓")
            elif state.status == "broken_symlink":
                print("  Plugin:    broken symlink (target missing) ✗")
                print(f"  Broken target: {state.link_target}")
                print("  → Run: mnemosyne-hermes install --force")
            elif state.status == "stale_wrapper":
                print("  Plugin:    stale wrapper target ✗")
                print(f"  Wrapper Python: {state.wrapper_python}")
                print(f"  Wrapper site-packages: {state.wrapper_site_packages}")
                print(f"  Import error: {state.wrapper_import_error}")
                print("  → Re-run: mnemosyne-hermes install --mode wrapper --force --python <venv>/bin/python")
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
                    if state.mode == "symlink" and _hermes_python_mismatch(hermes_python):
                        print("  ⚠ Different Python interpreters! Install and Hermes are not the same environment.")
                        print(f"  → Run: {shlex.quote(str(hermes_python))} -m pip install -U 'mnemosyne-hermes[all]'")
                except Exception:
                    print(f"  Hermes' Python: {hermes_python} (unable to check version)")
            else:
                print("  Hermes' Python: not found")
            if installed and state.mode == "symlink" and hermes_python and _hermes_python_mismatch(hermes_python):
                print("  → Hermes Python vs install Python mismatch means the symlink exists but Hermes")
                print("     may not be able to import mnemosyne core. Run with --dry-run to diagnose.")
            return 0 if installed else 1

        if command == "runtime-python":
            return _runtime_python_json(
                explicit_python=args.python,
                hermes_home_path=args.hermes_home,
            )

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
