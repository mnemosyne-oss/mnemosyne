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
# The unconditional cap everywhere except macOS x86_64, and the narrow one that platform gets.
ONNXRUNTIME_DEFAULT = "onnxruntime>=1.21.0,<1.29; sys_platform != 'darwin' or platform_machine != 'x86_64'"
ONNXRUNTIME_MACOS_INTEL = "onnxruntime>=1.21.0,<1.24; sys_platform == 'darwin' and platform_machine == 'x86_64'"
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


def _shipped_onnxruntime_requirements() -> list[str]:
    """The onnxruntime requirement strings pyproject.toml actually declares, deduplicated.

    Read from the file rather than from the constants above, so a pairing that drifts out of
    the manifest is caught instead of being asserted against itself.
    """
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = pyproject.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
    return sorted(set(re.findall(r'"(onnxruntime[^"]+)"', block)))


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


def test_only_macos_intel_gets_the_narrow_bound():
    """Nothing the x86_64-macOS branch admits may sit above the last wheel build.

    This is the property the bug is about, so it is what the test asserts. Asserting that a
    marker is present, or that a ceiling holds in isolation, also passes on the bounds
    swapped the wrong way round — which is exactly what the first version of this PR did.

    Microsoft stopped shipping ``macosx_*_x86_64`` wheels after 1.23.2. The macOS branch
    therefore stops before 1.24, while the default branch keeps reaching 1.29 so nothing
    else is capped for a platform that has no problem.
    """
    from packaging.requirements import Requirement
    from packaging.version import Version

    macos_intel_requirement = Requirement(ONNXRUNTIME_MACOS_INTEL)
    narrow = Requirement(ONNXRUNTIME_DEFAULT)

    # 1.23.2 is the last macOS x86_64 release with a wheel, and both bounds admit it.
    assert macos_intel_requirement.specifier.contains(Version("1.23.2"))
    assert narrow.specifier.contains(Version("1.23.2"))

    # Above it only the default bound reaches, and only off that platform.
    for version in ("1.24.0", "1.28.0"):
        assert not macos_intel_requirement.specifier.contains(Version(version))
        assert narrow.specifier.contains(Version(version))

    # 1.29 stays excluded everywhere: its import invokes an unavailable `blkid`.
    for version in ("1.29.0", "1.30.0"):
        assert not macos_intel_requirement.specifier.contains(Version(version))
        assert not narrow.specifier.contains(Version(version))


def test_swapped_bounds_would_still_break_an_intel_mac():
    """Pin the defect itself, so a swap back fails here rather than in a user's install.

    The narrow bound on the wrong marker — what the first version of this PR did — leaves the
    resolver on macOS x86_64 with five versions that have no wheel for it. The correct
    pairing leaves none. This is the arithmetic #1108 reports, asserted rather than described.
    """
    from packaging.markers import Marker
    from packaging.requirements import Requirement
    from packaging.version import Version

    versions_without_an_x86_wheel = ("1.24.0", "1.25.0", "1.26.0", "1.27.0", "1.28.0")
    env = {"sys_platform": "darwin", "platform_machine": "x86_64"}
    swapped_applies = "onnxruntime>=1.21.0,<1.29; sys_platform == 'darwin' and platform_machine == 'x86_64'"

    def admits_uninstallable(requirement: str) -> list[str]:
        specifier = Requirement(requirement).specifier
        return [v for v in versions_without_an_x86_wheel if specifier.contains(Version(v))]

    # With the bounds swapped, the branch that applies on an Intel Mac is the wide one.
    assert Marker(swapped_applies.split(";", 1)[1]).evaluate(env)
    assert admits_uninstallable(swapped_applies) == list(versions_without_an_x86_wheel)

    # Correctly paired, that branch is the narrow one and admits none.
    assert Marker(ONNXRUNTIME_MACOS_INTEL.split(";", 1)[1]).evaluate(env)
    assert admits_uninstallable(ONNXRUNTIME_MACOS_INTEL) == []


def test_nothing_applicable_on_macos_intel_lacks_a_wheel():
    """Read what the shipped requirements actually admit on macOS x86_64.

    Asserting that a marker is present, or counting how many times each requirement string
    appears, passes with the bound and the marker swapped — the file still contains one of
    each, just on the wrong side, which is exactly the defect this PR fixed. So the assertion
    is about behaviour: resolve on the affected platform, and require that nothing it admits
    sits above the last macOS x86_64 release.
    """
    from packaging.markers import Marker
    from packaging.requirements import Requirement
    from packaging.version import Version

    UNWHEELED = ("1.24.0", "1.25.0", "1.26.0", "1.27.0", "1.28.0")
    MACOS_INTEL = {"sys_platform": "darwin", "platform_machine": "x86_64"}
    LINUX = {"sys_platform": "linux", "platform_machine": "x86_64"}

    def admits_unwheeled(environment):
        parsed = [Requirement(r) for r in _shipped_onnxruntime_requirements()]
        applying = [r for r in parsed if Marker(str(r.marker)).evaluate(environment)]
        assert len(applying) == 1, [str(r) for r in applying]
        return [v for v in UNWHEELED if applying[0].specifier.contains(Version(v))]

    assert admits_unwheeled(MACOS_INTEL) == []

    # And nothing else is capped for a platform that has no problem: Linux still reaches
    # 1.28, so this is a split, not a blanket downgrade.
    assert admits_unwheeled(LINUX) == list(UNWHEELED)


def test_swapping_the_bounds_would_reintroduce_the_defect():
    """Guard the pairing itself, so a swap cannot pass unnoticed again.

    On macOS x86_64 the narrow bound has to sit on the darwin/x86_64 marker. With the two
    ceilings exchanged, the branch that applies there is the wide one and every version
    without a wheel is admitted again. Asserted rather than described, because the first
    version of this PR shipped the swap and its own tests passed.
    """
    from packaging.markers import Marker
    from packaging.requirements import Requirement
    from packaging.version import Version

    UNWHEELED = ("1.24.0", "1.25.0", "1.26.0", "1.27.0", "1.28.0")
    MACOS_INTEL = {"sys_platform": "darwin", "platform_machine": "x86_64"}
    narrow = f"onnxruntime>=1.21.0,<1.24; {ONNXRUNTIME_MACOS_INTEL.split(';', 1)[1]}"
    wide = f"onnxruntime>=1.21.0,<1.29; {ONNXRUNTIME_MACOS_INTEL.split(';', 1)[1]}"

    def admits(declaration):
        return [v for v in UNWHEELED if Requirement(declaration).specifier.contains(Version(v))]

    assert Marker(str(Requirement(narrow).marker)).evaluate(MACOS_INTEL)
    assert admits(narrow) == []
    assert admits(wide) == list(UNWHEELED)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
