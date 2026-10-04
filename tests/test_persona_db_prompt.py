"""Regression coverage for #854: promoted rows reach the next prompt."""

import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_mixin(path):
    """Load only the shared helper, without importing a live provider."""
    spec = importlib.util.spec_from_file_location("persona_prompt_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HermesPersonaPromptMixin


def make_provider(tmp_path, mixin, name="default"):
    """Create an isolated store with production persona admission constraints."""
    conn = sqlite3.connect(tmp_path / f"{name}.db")
    conn.executescript("""
        CREATE TABLE working_memory (id TEXT PRIMARY KEY, content TEXT);
        CREATE TABLE episodic_memory (id TEXT PRIMARY KEY, content TEXT);
        CREATE TABLE memoria_timelines (source_memory_id TEXT, topic TEXT);
        CREATE TABLE memoria_persona (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT DEFAULT 'default',
            tier TEXT NOT NULL CHECK(tier IN ('permanent','long_term','working')),
            topic TEXT NOT NULL, content TEXT NOT NULL, confidence REAL DEFAULT 0.7,
            source_memory_id TEXT, promotion_reason TEXT,
            reinforcement_count INTEGER DEFAULT 0,
            last_reinforced_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    provider = mixin()
    provider._beam = SimpleNamespace(conn=conn)
    provider.PERSONA_ENABLED = True
    provider.PERSONA_FILE = tmp_path / "missing-persona.md"
    provider.PERSONA_TOKEN_CAP = 1500
    return provider


@pytest.fixture
def mixin():
    """Supply a freshly loaded helper so environment defaults are testable."""
    return load_mixin(ROOT / "mnemosyne/integrations/hermes_persona_prompt.py")


def insert(provider, content, tier="permanent", topic="general", count=0):
    """Insert a complete synthetic persona rule into the isolated store."""
    conn = provider._beam.conn
    conn.execute(
        "INSERT INTO memoria_persona (tier,topic,content,reinforcement_count) VALUES (?,?,?,?)",
        (tier, topic, content, count),
    )
    conn.commit()


def test_actual_promotion_reaches_fresh_process_next_prompt(tmp_path, mixin):
    """Actual promotion persists a non-null topic and reaches a new process."""
    provider = make_provider(tmp_path, mixin)
    conn = provider._beam.conn
    conn.execute("INSERT INTO working_memory VALUES ('source', 'Clean throwaway artifacts after work.')")
    conn.commit()
    spec = importlib.util.spec_from_file_location("persona_adapter_under_test", ROOT / "hermes_memory_provider/persona_adapter.py")
    adapter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter_module)
    result = json.loads(adapter_module.PersonaAdapter(provider._beam)._promote("source", "permanent"))
    assert result["status"] == "ok"
    assert conn.execute("SELECT topic FROM memoria_persona").fetchone()[0] == "general"
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO memoria_persona (tier,topic,content) VALUES ('permanent',NULL,'invalid')")
    conn.close()
    code = """
import importlib.util, sqlite3, sys
from pathlib import Path
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('prompt', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = m.HermesPersonaPromptMixin()
p._beam = SimpleNamespace(conn=sqlite3.connect(sys.argv[2]))
p.PERSONA_ENABLED = True
p.PERSONA_FILE = Path(sys.argv[3])
print(p._with_persona_block('# Mnemosyne Memory'))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "mnemosyne/integrations/hermes_persona_prompt.py"),
         str(tmp_path / "default.db"), str(tmp_path / "missing-persona.md")],
        capture_output=True, text=True, check=True,
    )
    assert "Clean throwaway artifacts after work." in result.stdout
    assert "# L3 Persona (Active Behavioral Rules)" in result.stdout


def test_opt_out_reads_neither_file_nor_db(tmp_path, mixin):
    """Disabled injection must not touch either backing source."""
    p = make_provider(tmp_path, mixin)
    p.PERSONA_ENABLED = False
    p._beam.conn.close()
    p.PERSONA_FILE = None
    assert p._with_persona_block("base") == "base"


def test_opt_in_remains_required(monkeypatch):
    """Storing a persona row does not silently opt users into injection."""
    monkeypatch.delenv("MNEMOSYNE_PERSONA_ENABLED", raising=False)
    assert not load_mixin(ROOT / "mnemosyne/integrations/hermes_persona_prompt.py").PERSONA_ENABLED


def test_profile_bound_stores_do_not_mix(tmp_path, mixin):
    """Reads follow the bound store and never a previous profile cache."""
    a = make_provider(tmp_path, mixin, "alpha")
    b = make_provider(tmp_path, mixin, "beta")
    insert(a, "alpha-only standing rule")
    insert(b, "beta-only standing rule")
    assert "alpha-only" in a._persona_block()
    assert "beta-only" not in a._persona_block()
    assert "beta-only" in b._persona_block()
    assert "alpha-only" not in b._persona_block()
    a._beam = b._beam
    assert "alpha-only" not in a._persona_block()  # no process-global/cache fallback


def test_deterministic_tier_reinforcement_id_order(tmp_path, mixin):
    """Tier, reinforcement, and stable insertion order determine selection."""
    p = make_provider(tmp_path, mixin)
    insert(p, "working-rule", "working", count=99)
    insert(p, "long-rule", "long_term", count=99)
    insert(p, "permanent-first", count=2)
    insert(p, "permanent-second", count=2)
    insert(p, "permanent-low", count=1)
    block = p._persona_block()
    expected = ["permanent-first", "permanent-second", "permanent-low", "long-rule", "working-rule"]
    assert [block.index(text) for text in expected] == sorted(block.index(text) for text in expected)
    assert block == p._persona_block()


def test_hard_entry_bound(tmp_path, mixin):
    """A generous byte budget cannot bypass the twenty-row SQL limit."""
    p = make_provider(tmp_path, mixin)
    for i in range(25):
        insert(p, f"rule-{i:02}")
    p.PERSONA_TOKEN_CAP = 100000
    block = p._persona_block()
    assert block.count("- [permanent]") == 20
    assert "rule-20" not in block


@pytest.mark.parametrize("content", ["x" * 10000, "汉字" * 10000, "word " * 10000], ids=["unbroken", "cjk", "words"])
def test_oversized_entry_skipped_whole_and_small_rule_survives(tmp_path, mixin, content):
    """Oversized ASCII, CJK, and word entries are skipped without slicing."""
    p = make_provider(tmp_path, mixin)
    insert(p, content)
    insert(p, "small rule")
    p.PERSONA_TOKEN_CAP = 100
    block = p._with_persona_block("")
    assert "small rule" in block
    assert len(block.strip().encode("utf-8")) <= 100
    assert content not in block


@pytest.mark.parametrize("cap", [-1, 0, 10])
def test_nonpositive_or_tiny_budget_emits_nothing(tmp_path, mixin, cap):
    """Only the DB path has the documented nonpositive-budget no-op."""
    p = make_provider(tmp_path, mixin)
    insert(p, "rule")
    p.PERSONA_TOKEN_CAP = cap
    assert p._persona_block() == ""


def test_empty_or_missing_table_is_noop(tmp_path, mixin):
    """An empty or older store adds no behavioral section."""
    p = make_provider(tmp_path, mixin)
    assert p._with_persona_block("base") == "base"
    p._beam.conn.execute("DROP TABLE memoria_persona")
    assert p._with_persona_block("base") == "base"


def test_file_path_stays_compatible(tmp_path, mixin):
    """Nonempty legacy file content retains precedence over DB rows."""
    p = make_provider(tmp_path, mixin)
    insert(p, "DB rule")
    p.PERSONA_FILE = tmp_path / "persona.md"
    p.PERSONA_FILE.write_text("explicit file rule", encoding="utf-8")
    assert p._persona_block() == "explicit file rule"
    p.PERSONA_FILE.write_text("", encoding="utf-8")
    assert "DB rule" in p._persona_block()


def test_file_zero_budget_retains_legacy_truncation_notice(tmp_path, mixin):
    """Document compatibility: zero file budget emits a notice, not file content."""
    p = make_provider(tmp_path, mixin)
    p.PERSONA_FILE = tmp_path / "persona.md"
    p.PERSONA_FILE.write_text("legacy file rule", encoding="utf-8")
    p.PERSONA_TOKEN_CAP = 0
    block = p._persona_block()
    assert "truncated" in block
    assert "legacy file rule" not in block


def test_demotion_visible_without_restart(tmp_path, mixin):
    """Committed removal is reflected by the next uncached DB read."""
    p = make_provider(tmp_path, mixin)
    insert(p, "retirable rule")
    assert "retirable" in p._persona_block()
    p._beam.conn.execute("DELETE FROM memoria_persona")
    p._beam.conn.commit()
    assert p._persona_block() == ""


@pytest.mark.parametrize("package", ["hermes_memory_provider", "mnemosyne_hermes"])
def test_real_provider_system_prompt_reads_only_bound_store(tmp_path, mixin, package, monkeypatch):
    """Both provider surfaces emit rules only from the currently bound store."""
    import importlib
    importlib.import_module("mnemosyne.integrations.hermes_persona_prompt")  # core before host discovery
    module = importlib.import_module(package)
    if hasattr(module, "_p1b_current_key"):
        monkeypatch.setattr(module, "_p1b_current_key", lambda: None)
    a = make_provider(tmp_path, mixin, "owner-a")
    b = make_provider(tmp_path, mixin, "owner-b")
    insert(a, "owner-a-only cleanup rule")
    insert(b, "owner-b-only safety rule")
    provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
    provider._beam = a._beam
    provider._init_error = None
    provider._maybe_retry_init = lambda: None
    provider.PERSONA_ENABLED = True
    provider.PERSONA_FILE = tmp_path / "missing.md"
    provider.PERSONA_TOKEN_CAP = 1500
    first = provider.system_prompt_block()
    assert "owner-a-only" in first
    assert "owner-b-only" not in first
    provider._beam = b._beam
    second = provider.system_prompt_block()
    assert "owner-b-only" in second
    assert "owner-a-only" not in second
    provider.PERSONA_ENABLED = False
    assert "# L3 Persona" not in provider.system_prompt_block()
