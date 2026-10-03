"""Regression coverage for optional dependency constraints.

onnxruntime has two upper bounds, chosen per platform (#1108):

- everywhere else, ``<1.29`` — 1.29's import invokes an unavailable ``blkid``
- macOS on x86_64, ``<1.24`` — Microsoft stopped shipping ``macosx_*_x86_64``
  wheels after 1.23.2, so a plain ``<1.29`` resolves to a version with no
  wheel at all and ``uv sync`` exits 2

Both forms have to be declared together: one of the two markers is false on any
given host, so a single requirement can only ever apply to half the platforms.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# The unconditional cap, and the narrower one macOS x86_64 gets.
ONNXRUNTIME_DEFAULT = "onnxruntime>=1.21.0,<1.24; sys_platform != 'darwin' or platform_machine != 'x86_64'"
ONNXRUNTIME_MACOS_INTEL = "onnxruntime>=1.21.0,<1.29; sys_platform == 'darwin' and platform_machine == 'x86_64'"
ONNXRUNTIME_REQUIREMENTS = (ONNXRUNTIME_DEFAULT, ONNXRUNTIME_MACOS_INTEL)
EXPECTED_EXTRAS = {"embeddings", "all"}


def _pyproject_optional_dependencies() -> dict[str, list[str]]:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    optional_dependencies = pyproject.split("[project.optional-dependencies]", 1)[1].split(
        "\n[", 1
    )[0]
    # The extras are now multi-line lists, so read the whole block and literal_eval each.
    assignments = re.findall(
        r"^(embeddings|all)\s*=\s*(\[[^\]]*\])", optional_dependencies, re.MULTILINE | re.S
    )
    dependencies = {
        extra: ast.literal_eval(requirements) for extra, requirements in assignments
    }
    assert set(dependencies) == EXPECTED_EXTRAS
    for requirement in ONNXRUNTIME_REQUIREMENTS:
        assert optional_dependencies.count(requirement) == len(EXPECTED_EXTRAS)
    return dependencies


def _setup_py_optional_dependencies() -> dict[str, list[str]]:
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    setup_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ((isinstance(node.func, ast.Name) and node.func.id == "setup")
             or (isinstance(node.func, ast.Attribute) and node.func.attr == "setup"))
    )
    extras_keyword = next(
        keyword for keyword in setup_call.keywords if keyword.arg == "extras_require"
    )
    return ast.literal_eval(extras_keyword.value)


def test_embedding_extras_bound_onnxruntime_below_1_29():
    """Both platform pins are declared, and neither extra drops either one."""
    for optional_dependencies in (
        _pyproject_optional_dependencies(),
        _setup_py_optional_dependencies(),
    ):
        for extra in EXPECTED_EXTRAS:
            declared = optional_dependencies[extra]
            for requirement in ONNXRUNTIME_REQUIREMENTS:
                assert declared.count(requirement) == 1, (extra, requirement, declared)

        extras_with_requirement = {
            extra
            for extra, requirements in optional_dependencies.items()
            if any(requirement in requirements for requirement in ONNXRUNTIME_REQUIREMENTS)
        }
        assert extras_with_requirement == EXPECTED_EXTRAS


def test_the_two_markers_partition_every_platform():
    """One marker is always true and the other always false — that is what makes it work.

    A single unconditional requirement cannot express this: on macOS x86_64 the narrower
    ceiling is the only one that resolves, and everywhere else it is the wider one.
    """
    from packaging.markers import Marker

    universal = Marker(ONNXRUNTIME_DEFAULT.split(";", 1)[1])
    macos_intel = Marker(ONNXRUNTIME_MACOS_INTEL.split(";", 1)[1])

    platforms = [
        {"sys_platform": "darwin", "platform_machine": "x86_64"},
        {"sys_platform": "darwin", "platform_machine": "arm64"},
        {"sys_platform": "linux", "platform_machine": "x86_64"},
        {"sys_platform": "linux", "platform_machine": "aarch64"},
        {"sys_platform": "win32", "platform_machine": "AMD64"},
    ]
    for environment in platforms:
        assert universal.evaluate(environment) != macos_intel.evaluate(environment), environment

    # The narrower bound is only ever active on the platform that needs it.
    assert macos_intel.evaluate(
        {"sys_platform": "darwin", "platform_machine": "x86_64"}
    )
    for environment in platforms[1:]:
        assert not macos_intel.evaluate(environment), environment


def test_only_macos_intel_gets_the_wider_bound():
    """Nothing the x86_64-macOS branch admits may sit above the last wheel build.

    This is the property the bug is about, so it is what the test asserts. Asserting that a
    marker is present, or that a ceiling holds in isolation, also passes on the old
    unconditional ``<1.29`` — the shape of the requirement is not what was broken, the
    range it admitted on a platform that cannot use it.

    Microsoft stopped shipping ``macosx_*_x86_64`` wheels after 1.23.2. With ``<1.29`` a
    resolver may pick 1.24 through 1.28 there and the install exits 2 with no candidate; with
    ``<1.24`` every version it admits exists for that platform.
    """
    from packaging.requirements import Requirement
    from packaging.version import Version

    macos_intel_requirement = Requirement(ONNXRUNTIME_MACOS_INTEL)
    narrow = Requirement(ONNXRUNTIME_DEFAULT)

    # 1.23.2 is the last macOS x86_64 release with a wheel, and both bounds admit it.
    assert macos_intel_requirement.specifier.contains(Version("1.23.2"))
    assert narrow.specifier.contains(Version("1.23.2"))

    # Every version above the last wheel build reaches only the wide bound, and only on
    # macOS x86_64 — the one platform where it cannot be installed.
    for version in ("1.24.0", "1.28.0"):
        assert macos_intel_requirement.specifier.contains(Version(version))
        assert not narrow.specifier.contains(Version(version))

    # 1.29 stays excluded everywhere: its import invokes an unavailable `blkid`.
    for version in ("1.29.0", "1.30.0"):
        assert not macos_intel_requirement.specifier.contains(Version(version))
        assert not narrow.specifier.contains(Version(version))


def test_the_unconditional_pin_would_still_break_an_intel_mac():
    """Pin the defect itself, so a re-flattening of the two branches fails here.

    With the single ``onnxruntime>=1.21.0,<1.29`` the resolver on macOS x86_64 has five
    versions to choose from that have no wheel for it. The narrow branch leaves none. This
    is the arithmetic #1108 reports, asserted rather than described.
    """
    from packaging.requirements import Requirement
    from packaging.version import Version

    versions_without_an_x86_wheel = ("1.24.0", "1.25.0", "1.26.0", "1.27.0", "1.28.0")
    old = Requirement("onnxruntime>=1.21.0,<1.29")
    narrow = Requirement(ONNXRUNTIME_DEFAULT)
    assert [v for v in versions_without_an_x86_wheel if old.specifier.contains(Version(v))] == list(
        versions_without_an_x86_wheel
    )
    assert [v for v in versions_without_an_x86_wheel if narrow.specifier.contains(Version(v))] == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
