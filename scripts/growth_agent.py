#!/usr/bin/env python3
"""
Mnemosyne Growth Agent
======================
Daily X growth agent for @mnemosyne_oss. Posts project updates,
stats, feature highlights, and engagement content.

Usage:
    python3 scripts/growth_agent.py           # Full run
    python3 scripts/growth_agent.py --dry-run # Print without posting

Requires: clix0 (authenticated with @mnemosyne account)
"""

import os, sys, json, subprocess, sqlite3, random
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

DATA_DIR = Path(os.environ.get("MNEMOSYNE_DATA_DIR", Path.home() / ".hermes" / "mnemosyne" / "data"))
DB_PATH = DATA_DIR / "mnemosyne.db"
PROJECT_DIR = Path("/root/.hermes/projects/mnemosyne")
DRY_RUN = "--dry-run" in sys.argv

CLIX_ACCOUNT = "mnemosyne"
CLIX_BIN = os.path.expanduser("~/.local/bin/clix")

# ── Topic templates ──────────────────────────────────────────────────────────

TOPICS = [
    "stats",       # DB health metrics
    "feature",     # Highlight a feature
    "community",   # Engagement / call to action
    "benchmark",   # BEAM / performance
    "tip",         # Quick usage tip
]


# ── Duplicate detection ────────────────────────────────────────────────────

def load_posted_texts() -> set:
    """Load all previously posted texts from snapshots to avoid duplicates."""
    posted = set()
    snap_dir = Path.home() / ".hermes" / "mnemosyne" / "growth"
    if not snap_dir.exists():
        return posted
    for f in sorted(snap_dir.glob("run_*.json")):
        try:
            d = json.loads(f.read_text())
            if d.get("posted") and d.get("text"):
                posted.add(d["text"].strip())
        except (json.JSONDecodeError, OSError):
            continue
    return posted

POSTED_TEXTS = set()  # populated on first call to generate_post

def run_cmd(cmd: list, timeout: int = 30) -> tuple[int, str]:
    """Run a command and return (exit_code, stdout+stderr)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except FileNotFoundError:
        return -1, f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -2, "Timed out"

def check_auth() -> Optional[str]:
    """Check if clix is authenticated. Returns account name or None."""
    code, out = run_cmd([CLIX_BIN, "auth", "status"])
    if code != 0:
        return None
    try:
        data = json.loads(out)
        if data.get("authenticated"):
            return data.get("account", "unknown")
    except json.JSONDecodeError:
        pass
    return None

def gather_stats() -> dict:
    """Collect database statistics."""
    stats = {"date": datetime.now(timezone.utc).isoformat()}
    if not DB_PATH.exists():
        stats["error"] = f"DB not found at {DB_PATH}"
        return stats

    conn = sqlite3.connect(str(DB_PATH))
    try:
        for table in ["working_memory", "episodic_memory", "canonical_facts", "triples"]:
            try:
                cnt = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[table] = cnt
            except Exception:
                stats[table] = 0

        # Working memory by importance
        wm_total = stats.get("working_memory", 0)
        if wm_total > 0:
            high_imp = conn.execute(
                "SELECT COUNT(*) FROM working_memory WHERE importance >= 0.7"
            ).fetchone()[0]
            low_imp = conn.execute(
                "SELECT COUNT(*) FROM working_memory WHERE importance < 0.3"
            ).fetchone()[0]
            stats["wm_high_importance"] = high_imp
            stats["wm_low_importance"] = low_imp
            stats["wm_high_pct"] = round(high_imp / wm_total * 100, 1)
            stats["wm_low_pct"] = round(low_imp / wm_total * 100, 1)

        # DB size
        stats["db_size_mb"] = round(DB_PATH.stat().st_size / 1048576, 2)
    finally:
        conn.close()
    return stats

def get_latest_release() -> Optional[str]:
    """Get latest git tag."""
    code, out = run_cmd(
        ["git", "-C", str(PROJECT_DIR), "tag", "--sort=-v:refname"],
        timeout=10
    )
    if code == 0:
        tags = [t.strip() for t in out.split("\n") if t.strip()]
        return tags[0] if tags else None
    return None

def get_recent_commits(days: int = 7) -> list[dict]:
    """Get commits from the last N days."""
    since = datetime.now(timezone.utc)
    code, out = run_cmd(
        ["git", "-C", str(PROJECT_DIR), "log", f"--since={days}.days.ago",
         "--oneline", "--format=%h|%s|%an", "-30"],
        timeout=10
    )
    commits = []
    if code == 0:
        for line in out.split("\n"):
            if "|" in line:
                parts = line.split("|", 2)
                commits.append({"hash": parts[0], "message": parts[1], "author": parts[2] if len(parts) > 2 else "?"})
    return commits

# ── Content generators ──────────────────────────────────────────────────────

def generate_stats_post(stats: dict, release: Optional[str]) -> str:
    """Generate a post about current stats."""
    wm = stats.get("working_memory", 0)
    ep = stats.get("episodic_memory", 0)
    triples = stats.get("triples", 0)
    db_size = stats.get("db_size_mb", 0)
    high_pct = stats.get("wm_high_pct", 0)

    posts = [
        f"🧠 Mnemosyne daily pulse:\n{wm} working memories · {ep} episodes · {triples} triples\n{high_pct}% high-signal memory density · {db_size}MB on disk\n\nAll local. All private. No API calls.",
        f"📊 Mnemosyne by the numbers:\n{triples} knowledge triples indexed\n{ep} episodic traces stored\nDatabase: {db_size}MB\n\nZero-dependency SQLite memory that ships with your app.",
        f"⚡ Memory check: {wm} working memories cached, {high_pct}% above 0.7 importance threshold.\n{stats.get('wm_low_pct', 0)}% idle noise getting consolidated on next sleep cycle.\n\nMnemosyne self-heals. Always.",
    ]
    return random.choice(posts)

def generate_feature_post(stats: dict, release: Optional[str]) -> str:
    """Highlight a feature."""
    features = [
        f"Mnemosyne now ships with L3 behavioral persona rules (v{release or 'latest'}) — always-on, code-configurable constraints that shape memory behavior at the system level.\n\nNo prompts. No wrappers. Just rules.",
        f"BEAM benchmark? Mnemosyne scored top-tier at ICLR 2026.\n\nSub-millisecond recall. Zero dependencies. One SQLite file.\n\nTry it: pip install mnemosyne-memory",
        f"Your agent's memory doesn't need a cloud API.\n\nMnemosyne runs everywhere — Hermes, Claude Code, Cursor, Codex, OpenWebUI, OpenClaw, any MCP client.\n\nOne pip install. One SQLite file. Full privacy.",
        f"Sync your memory across devices with Mnemosyne Sync — optional bidirectional sync with client-side encryption.\n\nYour data. Your keys. Your infra.",
        f"MCP-native since day one. Mnemosyne speaks the Model Context Protocol, so any MCP client gets 23 memory tools out of the box.\n\nCursor + Mnemosyne = persistent context that survives session restarts.",
    ]
    return random.choice(features)

def generate_community_post(stats: dict, release: Optional[str]) -> str:
    """Engagement / CTA."""
    ctas = [
        "Building an AI agent? Memory is the hardest part.\n\nMnemosyne makes it a `pip install`.\n\n⭐ Star us on GitHub: github.com/AxDSan/mnemosyne\n💬 Join the Discord: discord.gg/Cgzpw9x3R",
        "We ship fast. v{rel} just dropped with fixes for the Hermes plugin sync path, sqlite-vec compatibility, and smarter recall.\n\nUpgrade: pip install --upgrade mnemosyne-memory\n\nChangelog: github.com/AxDSan/mnemosyne/releases",
        "What's your agent's memory strategy?\n\nMnemosyne users are building with:\n- BEAM cognitive architecture\n- L3 behavioral rules\n- Multi-device sync with encryption\n- MCP-native tool integration\n\nWe'd love to hear your stack ↓",
    ]
    return random.choice(ctas).format(rel=(release or "latest").lstrip("v"))

def generate_tip_post(stats: dict, release: Optional[str]) -> str:
    """Quick usage tip."""
    tips = [
        "💡 Mnemosyne tip:\nSet MNEMOSYNE_WM_MAX_ITEMS=5000 to cap working memory before consolidation kicks in.\n\nPerfect for edge devices with limited storage.",
        "💡 Want to see what your agent remembers?\n\nmnemosyne diagnose --recall-stats\n\nShows recall frequency, importance distribution, and consolidation readiness at a glance.",
        "💡 Mnemosyne + Cursor = persistent agent memory across sessions.\n\nAdd to .cursor/mcp.json:\n{ \"mcpServers\": { \"mnemosyne\": { \"command\": \"mnemosyne-mcp\" } } }\n\nDone. Your agent now remembers everything.",
    ]
    return random.choice(tips)

def generate_post(topic: str, stats: dict, release: Optional[str]) -> str:
    """Generate content, avoiding text that was already posted."""
    global POSTED_TEXTS
    if not POSTED_TEXTS:
        POSTED_TEXTS.update(load_posted_texts())

    generators = {
        "stats": generate_stats_post,
        "feature": generate_feature_post,
        "community": generate_community_post,
        "benchmark": generate_feature_post,  # reuses feature for now
        "tip": generate_tip_post,
    }

    # Try the requested topic first
    gen = generators.get(topic, generate_stats_post)
    text = gen(stats, release)
    if text.strip() not in POSTED_TEXTS:
        return text

    # If duplicate, try other topics in random order
    import random as _rnd
    topics = [t for t in generators if t != topic]
    _rnd.shuffle(topics)
    for alt_topic in topics:
        alt_gen = generators[alt_topic]
        alt_text = alt_gen(stats, release)
        if alt_text.strip() not in POSTED_TEXTS:
            return alt_text

    # Last resort: append date to make unique
    from datetime import datetime
    date_str = datetime.now(timezone.utc).strftime("%b %d, %Y")
    return f"{text.strip()}\n\n— {date_str}"

def post_to_x(text: str) -> bool:
    """Post to X using clix."""
    if DRY_RUN:
        print(f"[DRY-RUN] Would post:\n{text}\n")
        print(f"[DRY-RUN] Length: {len(text)} chars\n")
        return True

    code, out = run_cmd(
        [CLIX_BIN, "post", text, "--json", "--account", CLIX_ACCOUNT],
        timeout=30
    )
    if code == 0:
        print(f"✓ Posted successfully")
        return True
    else:
        print(f"✗ Post failed (code {code}): {out[:500]}")
        return False

def save_snapshot(stats: dict, release: Optional[str], posted: bool, topic: str, text: str):
    """Save run snapshot to disk."""
    snap_dir = Path.home() / ".hermes" / "mnemosyne" / "growth"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_file = snap_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "release": release,
        "posted": posted,
        "topic": topic,
        "text": text,
        "dry_run": DRY_RUN,
    }
    with open(snap_file, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return snap_file

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"Mnemosyne Growth Agent — {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    # Step 1: Check auth
    print("\n[1/5] Checking authentication...")
    account = check_auth()
    if not account:
        print("  ✗ clix is NOT authenticated.")
        print("  Run: clix auth login  (or restore cookies)")
        print("\n  Possible fixes:")
        print("  1. Export fresh cookies from browser into ~/.hermes/secrets/x_cookies.json")
        print("  2. Run: clix auth import ~/.hermes/secrets/x_cookies.json")
        print("  3. Run: clix auth switch mnemosyne")
        if not DRY_RUN:
            print("\n  Aborting — no valid auth.")
            return
        else:
            print("  (dry-run: continuing anyway)\n")
    else:
        print(f"  ✓ Authenticated as @{account}")

    # Step 2: Gather stats
    print("\n[2/5] Gathering database stats...")
    stats = gather_stats()
    if "error" in stats:
        print(f"  ⚠ {stats['error']}")
    else:
        wm = stats.get("working_memory", 0)
        ep = stats.get("episodic_memory", 0)
        tr = stats.get("triples", 0)
        print(f"  Working memory: {wm} entries")
        print(f"  Episodic memory: {ep} episodes")
        print(f"  Triples: {tr}")
        print(f"  DB size: {stats.get('db_size_mb', 0)} MB")

    # Step 3: Check release
    print("\n[3/5] Checking latest release...")
    release = get_latest_release()
    print(f"  Latest tag: {release or 'unknown'}")

    # Step 4: Generate content
    print("\n[4/5] Generating content...")
    topic = random.choice(TOPICS)
    text = generate_post(topic, stats, release)
    print(f"  Topic: {topic}")
    print(f"  Text:\n{text}\n")

    # Step 5: Post
    posted = False
    if account or DRY_RUN:
        print("\n[5/5] Posting...")
        posted = post_to_x(text)
        if posted:
            print("  ✓ Content live on X")
        else:
            print("  ✗ Failed to post")
    else:
        print("\n[5/5] Skipped — no auth available")

    # Save snapshot
    snap = save_snapshot(stats, release, posted, topic, text)
    print(f"\nSnapshot saved: {snap}")
    print("=" * 60)

if __name__ == "__main__":
    main()
