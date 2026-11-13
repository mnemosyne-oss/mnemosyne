"""Scanner coverage for provider roots (#586 follow-up).

The effective-default scanner used to walk `mnemosyne/` only, so divergent
keys in `hermes_memory_provider/` and `integrations/` were invisible to the
generated configuration reference. These tests pin the coverage and the
cross-root conflict reporting, using fixture trees so they do not rot when
real defaults change.
"""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_gen(tmp_root=None, monkeypatch=None):
    spec = importlib.util.spec_from_file_location(
        "_gendocs_roots", REPO / "scripts" / "generate-docs.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    if tmp_root is not None:
        monkeypatch.setattr(gen, "REPO_ROOT", str(tmp_root))
    return gen


def _write(tree: Path, rel: str, body: str) -> None:
    p = tree / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def test_provider_roots_are_scanned():
    """The three #586 keys are found with their real fallbacks, and the
    roots that read each key are covered."""
    gen = _load_gen()
    env_map, defaults, _ = gen._collect_config()
    effective, conflicts = gen._scan_effective_defaults(
        env_map, defaults,
        roots=("mnemosyne", "hermes_memory_provider", "integrations"),
    )
    assert effective["prefetch_content_chars"][0] == "0"
    assert effective["sync_turn_user_limit"][0] == "500"
    assert effective["sync_turn_assistant_limit"][0] == "800"
    for key in ("sync_turn_user_limit", "sync_turn_assistant_limit"):
        sources = effective[key][1]
        assert any("hermes_memory_provider" in s for s in sources), key
        assert any(s.startswith("integrations") for s in sources), key
    # prefetch_content_chars is no longer read in the packaged provider: it
    # moved to the shared core module (#1077). Both remaining readers must
    # stay covered by the scanner.
    prefetch_sources = effective["prefetch_content_chars"][1]
    assert any(s.startswith("mnemosyne") for s in prefetch_sources), prefetch_sources
    assert any(s.startswith("integrations") for s in prefetch_sources), prefetch_sources
    # The two provider files agree today; no live conflict to report.
    assert conflicts == {}


def test_conflicting_defaults_across_roots_are_reported(tmp_path, monkeypatch):
    """Two roots disagreeing on one key surface as a conflict, not first-wins."""
    _write(
        tmp_path, "pkg_a/mod.py",
        'import os\nLIMIT = int(os.environ.get("MNEMOSYNE_DEMO_LIMIT", "10"))\n',
    )
    _write(
        tmp_path, "pkg_b/mod.py",
        'import os\nLIMIT = int(os.environ.get("MNEMOSYNE_DEMO_LIMIT", "99"))\n',
    )
    gen = _load_gen(tmp_path, monkeypatch)
    env_map = {"demo_limit": "MNEMOSYNE_DEMO_LIMIT"}
    effective, conflicts = gen._scan_effective_defaults(
        env_map, {"demo_limit": 5}, roots=("pkg_a", "pkg_b")
    )
    assert effective["demo_limit"][0] == "10"
    assert "demo_limit" in conflicts
    assert sorted(v for v, _ in conflicts["demo_limit"]) == ["10", "99"]
    # Each value is attributed to the root that set it, not cross-paired.
    by_value = {}
    for v, s in conflicts["demo_limit"]:
        by_value.setdefault(v, []).append(s)
    assert len(by_value["10"]) == 1 and "pkg_a" in by_value["10"][0]
    assert len(by_value["99"]) == 1 and "pkg_b" in by_value["99"][0]


def test_empty_fallback_is_not_a_conflict(tmp_path, monkeypatch):
    """An empty fallback means 'unset', not a competing default."""
    _write(
        tmp_path, "pkg_a/mod.py",
        'import os\nLIMIT = int(os.environ.get("MNEMOSYNE_DEMO_LIMIT", "10"))\n',
    )
    _write(
        tmp_path, "pkg_b/mod.py",
        'import os\nif os.environ.get("MNEMOSYNE_DEMO_LIMIT", ""):\n    pass\n',
    )
    gen = _load_gen(tmp_path, monkeypatch)
    env_map = {"demo_limit": "MNEMOSYNE_DEMO_LIMIT"}
    effective, conflicts = gen._scan_effective_defaults(
        env_map, {"demo_limit": 5}, roots=("pkg_a", "pkg_b")
    )
    assert effective["demo_limit"][0] == "10"
    assert conflicts == {}


def test_single_root_scan_is_unchanged(tmp_path, monkeypatch):
    """Default roots preserve the old single-package behavior."""
    _write(
        tmp_path, "mnemosyne/mod.py",
        'import os\nX = os.environ.get("MNEMOSYNE_DEMO_X", "7")\n',
    )
    gen = _load_gen(tmp_path, monkeypatch)
    effective, conflicts = gen._scan_effective_defaults(
        {"demo_x": "MNEMOSYNE_DEMO_X"}, {"demo_x": 1}
    )
    value, sources = effective["demo_x"]
    assert value == "7"
    assert [s.replace("\\", "/") for s in sources] == ["mnemosyne/mod.py"]
    assert conflicts == {}


def test_divergence_count_header_matches_scan():
    """The hardcoded (N) in the mdx header must equal the live scan count."""
    import re

    gen = _load_gen()
    env_map, defaults, _ = gen._collect_config()
    effective, _ = gen._scan_effective_defaults(
        env_map, defaults,
        roots=("mnemosyne", "hermes_memory_provider", "integrations"),
    )
    text = (REPO / "docs" / "api" / "configuration.mdx").read_text(encoding="utf-8")
    m = re.search(r"### Keys whose effective default bypasses `config\.py` \((\d+)\)", text)
    assert m is not None, "divergence section header missing from configuration.mdx"
    assert int(m.group(1)) == len(effective), (
        f"header says {m.group(1)} but the scan finds {len(effective)}"
    )


def test_sync_turn_descriptions_state_characters_not_turns():
    """Regression for the units error: these caps are per-message characters."""
    gen = _load_gen()
    for key in ("sync_turn_user_limit", "sync_turn_assistant_limit"):
        desc = gen.CONFIG_DESCRIPTIONS[key]
        assert "character" in desc, key
        assert "turns captured" not in desc, key


def test_conflict_footnote_renders(tmp_path, monkeypatch):
    """A reported conflict reaches the rendered footnote, not just the tuple."""
    _write(
        tmp_path, "pkg_a/mod.py",
        'import os\nLIMIT = int(os.environ.get("MNEMOSYNE_DEMO_LIMIT", "10"))\n',
    )
    _write(
        tmp_path, "pkg_b/mod.py",
        'import os\nLIMIT = int(os.environ.get("MNEMOSYNE_DEMO_LIMIT", "99"))\n',
    )
    gen = _load_gen(tmp_path, monkeypatch)
    env_map = {"demo_limit": "MNEMOSYNE_DEMO_LIMIT"}
    effective, conflicts = gen._scan_effective_defaults(
        env_map, {"demo_limit": 5}, roots=("pkg_a", "pkg_b")
    )
    out = gen._render_config(env_map, {"demo_limit": 5}, set(), "0.0.0",
                             effective, conflicts)
    assert "[^demo_limit-conflict]" in out
    assert "`10`" in out and "`99`" in out
    defn = next(
        l for l in out.splitlines() if l.startswith("[^demo_limit-conflict]:")
    ).replace("\\", "/")
    assert "`10` in `pkg_a/mod.py`" in defn
    assert "`99` in `pkg_b/mod.py`" in defn
    row = next(l for l in out.splitlines() if l.startswith("| `demo_limit`"))
    assert "[^demo_limit]" in row and "[^demo_limit-conflict]" in row


def test_empty_first_root_does_not_shadow_later_fallback(tmp_path, monkeypatch):
    """An empty fallback in an earlier root must not hide a later real one."""
    _write(
        tmp_path, "pkg_a/mod.py",
        'import os\nLIMIT = os.environ.get("MNEMOSYNE_DEMO_LIMIT", "")\n',
    )
    _write(
        tmp_path, "pkg_b/mod.py",
        'import os\nLIMIT = int(os.environ.get("MNEMOSYNE_DEMO_LIMIT", "99"))\n',
    )
    gen = _load_gen(tmp_path, monkeypatch)
    env_map = {"demo_limit": "MNEMOSYNE_DEMO_LIMIT"}
    effective, conflicts = gen._scan_effective_defaults(
        env_map, {"demo_limit": 5}, roots=("pkg_a", "pkg_b")
    )
    assert effective["demo_limit"][0] == "99"
    assert conflicts == {}


def test_conflict_without_divergence_renders(tmp_path, monkeypatch):
    """A conflict where the first root matches declared still surfaces."""
    _write(
        tmp_path, "pkg_a/mod.py",
        'import os\nLIMIT = int(os.environ.get("MNEMOSYNE_DEMO_LIMIT", "5"))\n',
    )
    _write(
        tmp_path, "pkg_b/mod.py",
        'import os\nLIMIT = int(os.environ.get("MNEMOSYNE_DEMO_LIMIT", "99"))\n',
    )
    gen = _load_gen(tmp_path, monkeypatch)
    env_map = {"demo_limit": "MNEMOSYNE_DEMO_LIMIT"}
    effective, conflicts = gen._scan_effective_defaults(
        env_map, {"demo_limit": 5}, roots=("pkg_a", "pkg_b")
    )
    assert "demo_limit" not in effective
    assert "demo_limit" in conflicts
    out = gen._render_config(env_map, {"demo_limit": 5}, set(), "0.0.0",
                             effective, conflicts)
    assert "[^demo_limit-conflict]" in out
    row = next(l for l in out.splitlines() if l.startswith("| `demo_limit`"))
    assert "[^demo_limit-conflict]" in row
    # No divergence section: the conflicts-only heading renders instead,
    # so the "(0)"-count bypasses heading can never appear.
    assert "### Keys with conflicting defaults across roots (1)" in out
    assert "### Keys whose effective default bypasses" not in out


def test_build_output_copies_are_not_scanned(tmp_path, monkeypatch):
    """pip install drops build/ copies into the tree; they are not sources.

    The CI test job installs before running the suite, so a stale or
    duplicate copy under build/ (or dist/, egg-info, __pycache__) must
    neither duplicate sources nor report phantom conflicts.
    """
    _write(
        tmp_path, "pkg/mod.py",
        'import os\nLIMIT = int(os.environ.get("MNEMOSYNE_DEMO_LIMIT", "10"))\n',
    )
    _write(
        tmp_path, "pkg/build/lib/mod.py",
        'import os\nLIMIT = int(os.environ.get("MNEMOSYNE_DEMO_LIMIT", "99"))\n',
    )
    gen = _load_gen(tmp_path, monkeypatch)
    env_map = {"demo_limit": "MNEMOSYNE_DEMO_LIMIT"}
    effective, conflicts = gen._scan_effective_defaults(
        env_map, {"demo_limit": 5}, roots=("pkg",)
    )
    assert effective["demo_limit"][0] == "10"
    assert [s.replace("\\", "/") for s in effective["demo_limit"][1]] == [
        "pkg/mod.py"
    ]
    assert conflicts == {}


def test_similarly_named_dirs_are_still_scanned(tmp_path, monkeypatch):
    """Negative control: exclusion matches whole segments, not substrings.

    A directory merely containing an excluded name (rebuild vs build)
    must still be scanned.
    """
    _write(
        tmp_path, "pkg/mod.py",
        'import os\nLIMIT = int(os.environ.get("MNEMOSYNE_DEMO_LIMIT", "10"))\n',
    )
    _write(
        tmp_path, "pkg/rebuild/mod.py",
        'import os\nLIMIT = int(os.environ.get("MNEMOSYNE_DEMO_LIMIT", "99"))\n',
    )
    gen = _load_gen(tmp_path, monkeypatch)
    env_map = {"demo_limit": "MNEMOSYNE_DEMO_LIMIT"}
    effective, conflicts = gen._scan_effective_defaults(
        env_map, {"demo_limit": 5}, roots=("pkg",)
    )
    assert effective["demo_limit"][0] == "10"
    assert "demo_limit" in conflicts
    assert sorted(v for v, _ in conflicts["demo_limit"]) == ["10", "99"]

@pytest.mark.parametrize(
    "copy_rel",
    [
        "pkg/build/lib/mod.py",
        "pkg/dist/mod.py",
        "pkg/__pycache__/mod.py",
        "pkg/demo.egg-info/mod.py",
    ],
)
def test_generated_copy_dirs_are_not_scanned(tmp_path, monkeypatch, copy_rel):
    """Every excluded copy dir behaves like build/: invisible to the scan."""
    _write(
        tmp_path, "pkg/mod.py",
        'import os\nLIMIT = int(os.environ.get("MNEMOSYNE_DEMO_LIMIT", "10"))\n',
    )
    _write(
        tmp_path, copy_rel,
        'import os\nLIMIT = int(os.environ.get("MNEMOSYNE_DEMO_LIMIT", "99"))\n',
    )
    gen = _load_gen(tmp_path, monkeypatch)
    env_map = {"demo_limit": "MNEMOSYNE_DEMO_LIMIT"}
    effective, conflicts = gen._scan_effective_defaults(
        env_map, {"demo_limit": 5}, roots=("pkg",)
    )
    assert effective["demo_limit"][0] == "10"
    assert [s.replace("\\", "/") for s in effective["demo_limit"][1]] == [
        "pkg/mod.py"
    ]
    assert conflicts == {}
