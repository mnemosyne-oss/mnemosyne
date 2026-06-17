#!/usr/bin/env python3
"""
Mnemosyne Statistics Dashboard
==============================
Terminal-based health check for the local Mnemosyne BEAM memory system.
Reads directly from SQLite — no external dependencies.

Usage:
    python3 mnemosyne-stats.py              # Full dashboard + auto-snapshot
    python3 mnemosyne-stats.py --compact    # Summary only
    python3 mnemosyne-stats.py --json       # JSON output
    python3 mnemosyne-stats.py --save-snapshot  # Save snapshot + show trends
    python3 mnemosyne-stats.py --trends     # Show trend data only
"""

import os
import sqlite3, sys, json
from pathlib import Path
from datetime import datetime

def _default_data_dir() -> Path:
    if data_dir := os.environ.get("MNEMOSYNE_DATA_DIR"):
        return Path(data_dir)
    if hermes_home := os.environ.get("HERMES_HOME"):
        return Path(hermes_home).expanduser() / "mnemosyne" / "data"
    return Path.home() / ".hermes" / "mnemosyne" / "data"


DB_PATH = _default_data_dir() / "mnemosyne.db"
WIKI_PATH = Path.home() / "wiki"
SNAPSHOT_DIR = Path.home() / ".hermes" / "mnemosyne" / "stats"
W = 60

def q(db, sql, params=()):
    try: return db.execute(sql, params).fetchall()
    except: return []

def cnt(db, table):
    r = q(db, f"SELECT COUNT(*) FROM {table}")
    return r[0][0] if r else 0

def pct(a, b): return (a / b * 100) if b > 0 else 0

def collect():
    if not DB_PATH.exists():
        return {"error": f"DB not found at {DB_PATH}"}

    db = sqlite3.connect(str(DB_PATH))
    s = {"db_size_mb": round(DB_PATH.stat().st_size / 1048576, 2)}

    # Working Memory
    wm_total = cnt(db, "working_memory")
    wm = {"total": wm_total, "by_source": {}, "by_scope": {},
          "importance_dist": {}, "recall_dist": {},
          "global_count": 0, "never_recalled": 0, "noise_pct": 0}

    if wm_total:
        for r in q(db, "SELECT source, COUNT(*), AVG(importance), AVG(recall_count) FROM working_memory GROUP BY source ORDER BY COUNT(*) DESC"):
            wm["by_source"][r[0] or "?"] = {"count": r[1], "imp": round(r[2] or 0, 3), "recall": round(r[3] or 0, 1)}
        for r in q(db, "SELECT scope, COUNT(*), AVG(importance) FROM working_memory GROUP BY scope"):
            wm["by_scope"][r[0] or "?"] = {"count": r[1], "imp": round(r[2] or 0, 3)}
        # Buckets cover [0, 1.0]; clamp at upper bound to absorb any
        # historical overflow rows (importance was unconstrained REAL pre-v3.9.0).
        # Use a large upper bound to include historical data anomalies.
        for lo, hi in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1000.0)]:
            c = q(db, "SELECT COUNT(*) FROM working_memory WHERE importance>=? AND importance<?", (lo,hi))
            wm["importance_dist"][f"{lo:.1f}-{hi:.1f}"] = c[0][0]
        # Override last bucket label so it stays user-readable as "0.8-1.0".
        wm["importance_dist"]["0.8-1.0"] = wm["importance_dist"].pop("0.8-1000.0")
        for lo, hi, lbl in [(0,1,"never"),(1,5,"low"),(5,10,"med"),(10,50,"high"),(50,99999,"vhigh")]:
            c = q(db, "SELECT COUNT(*) FROM working_memory WHERE recall_count>=? AND recall_count<?", (lo,hi))
            wm["recall_dist"][lbl] = c[0][0]
        g = q(db, "SELECT COUNT(*) FROM working_memory WHERE scope='global'")
        wm["global_count"] = g[0][0]
        nr = q(db, "SELECT COUNT(*) FROM working_memory WHERE recall_count=0")
        wm["never_recalled"] = nr[0][0]
        noise = q(db, "SELECT COUNT(*) FROM working_memory WHERE importance<0.3 AND recall_count=0")
        wm["noise_pct"] = round(noise[0][0] / wm_total * 100, 1)
    s["working_memory"] = wm

    # Episodic
    ep_total = cnt(db, "episodic_memory")
    ep = {"total": ep_total, "avg_imp": 0, "recent": []}
    if ep_total:
        r = q(db, "SELECT AVG(importance) FROM episodic_memory")
        ep["avg_imp"] = round(r[0][0] or 0, 3)
        for r in q(db, "SELECT id, content, importance, created_at FROM episodic_memory ORDER BY created_at DESC LIMIT 5"):
            ep["recent"].append({"id": r[0], "preview": (r[1] or "")[:70], "imp": r[2], "date": r[3]})
    s["episodic"] = ep

    # Triples (current-truth temporal facts; post-E6 scope)
    tg = {"total": cnt(db, "triples"), "predicates": {}}
    for r in q(db, "SELECT predicate, COUNT(*) FROM triples GROUP BY predicate ORDER BY COUNT(*) DESC LIMIT 10"):
        tg["predicates"][r[0]] = r[1]
    s["triples"] = tg

    # Annotations (post-E6: multi-valued memory annotations — mentions, facts,
    # occurred_on, has_source). Counted separately so operators can see the
    # split clearly. Pre-E6 databases that haven't been migrated yet will
    # show 0 here; the BeamMemory auto-migrate hook moves rows on next init.
    ann = {"total": cnt(db, "annotations"), "kinds": {}}
    for r in q(db, "SELECT kind, COUNT(*) FROM annotations GROUP BY kind ORDER BY COUNT(*) DESC LIMIT 10"):
        ann["kinds"][r[0]] = r[1]
    s["annotations"] = ann

    # Consolidation
    con = {"events": cnt(db, "consolidation_log"), "items": 0, "recent": []}
    r = q(db, "SELECT SUM(items_consolidated) FROM consolidation_log")
    con["items"] = r[0][0] if r and r[0][0] else 0
    for r in q(db, "SELECT session_id, items_consolidated, summary_preview, created_at FROM consolidation_log ORDER BY created_at DESC LIMIT 5"):
        con["recent"].append({"session": r[0][:30], "items": r[1], "summary": r[2], "date": r[3]})
    s["consolidation"] = con

    # Dreamer
    dr = {"runs": cnt(db, "dreamer_runs"), "proposals": 0, "conflicts": 0, "recent": []}
    r = q(db, "SELECT SUM(proposals_generated), SUM(conflicts_detected) FROM dreamer_runs")
    if r and r[0][0]: dr["proposals"] = r[0][0]; dr["conflicts"] = r[0][1] or 0
    for r in q(db, "SELECT started_at, status, memories_scanned, proposals_generated FROM dreamer_runs ORDER BY started_at DESC LIMIT 5"):
        dr["recent"].append({"start": r[0][:19], "status": r[1], "scanned": r[2], "proposals": r[3]})
    s["dreamer"] = dr

    s["embeddings"] = cnt(db, "memory_embeddings")
    s["scratchpad"] = cnt(db, "scratchpad")

    # Wiki
    wk = {"total": 0, "memories": 0, "concepts": 0}
    if WIKI_PATH.exists():
        wk["total"] = len(list(WIKI_PATH.rglob("*.md")))
        mdir = WIKI_PATH / "memories"
        cdir = WIKI_PATH / "concepts"
        wk["memories"] = len(list(mdir.glob("*.md"))) if mdir.exists() else 0
        wk["concepts"] = len(list(cdir.glob("*.md"))) if cdir.exists() else 0
    wk["promotion_pct"] = round(wk["memories"] / wm_total * 100, 1) if wm_total else 0
    s["wiki"] = wk

    # Top recalled
    tr = []
    for r in q(db, "SELECT content, recall_count, importance FROM working_memory WHERE recall_count>5 ORDER BY recall_count DESC LIMIT 10"):
        tr.append({"text": (r[0] or "")[:50].replace("\n", " "), "recalls": r[1], "imp": r[2]})
    s["top_recalled"] = tr

    # Memory age
    age = []
    for r in q(db, """
        SELECT CASE
            WHEN julianday('now')-julianday(created_at)<1 THEN 'Today'
            WHEN julianday('now')-julianday(created_at)<7 THEN 'This week'
            WHEN julianday('now')-julianday(created_at)<30 THEN 'This month'
            ELSE 'Older'
        END, COUNT(*), AVG(importance), AVG(recall_count)
        FROM working_memory GROUP BY 1
    """):
        age.append({"group": r[0], "count": r[1], "imp": round(r[2] or 0, 3), "recall": round(r[3] or 0, 1)})
    s["memory_age"] = age

    # Session activity
    act = []
    for r in q(db, "SELECT DATE(created_at), COUNT(*) FROM working_memory GROUP BY 1 ORDER BY 1 DESC LIMIT 7"):
        act.append({"date": r[0], "count": r[1]})
    s["activity"] = act

    # Quality score
    q_score = sum([
        1 if wm["noise_pct"] < 30 else 0,
        1 if pct(wm["never_recalled"], wm_total) < 70 else 0,
        1 if wm["global_count"] > wm_total * 0.05 else 0,
        1 if s["embeddings"] > 0 or wm_total == 0 else 0,
        1 if con["events"] > 0 else 0,
        1 if dr["runs"] > 0 else 0,
        1 if wk["memories"] > 10 else 0,
    ])
    s["quality_score"] = q_score

    db.close()
    return s

def show(s, compact=False):
    if "error" in s: print(f"ERROR: {s['error']}"); return

    score = s["quality_score"]
    grade = "A" if score >= 6 else "B" if score >= 5 else "C" if score >= 4 else "D" if score >= 3 else "F"
    bar = "█" * score + "░" * (7 - score)

    print("=" * W)
    print("  MNEMOSYNE HEALTH DASHBOARD")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * W)
    print(f"\n  Health: [{bar}] {score}/7 ({grade})   DB: {s['db_size_mb']} MB")

    wm = s["working_memory"]
    print(f"\n{'─'*W}")
    print(f"  WORKING MEMORY: {wm['total']} items")
    print(f"{'─'*W}")

    if not compact:
        print("\n  By Source:")
        for src, d in sorted(wm["by_source"].items(), key=lambda x: -x[1]["count"]):
            p = pct(d["count"], wm["total"])
            print(f"    {src:25s} {d['count']:4d} ({p:4.1f}%) imp={d['imp']:.2f} recall={d['recall']:.0f}")
        print("\n  Importance Distribution:")
        for bucket, c in wm["importance_dist"].items():
            p = pct(c, wm["total"])
            print(f"    {bucket}: {c:4d} {'█' * max(0, int(p / 2.5))}")
        print("\n  Recall Distribution:")
        for bucket, c in wm["recall_dist"].items():
            p = pct(c, wm["total"])
            print(f"    {bucket:8s}: {c:4d} {'█' * max(0, int(p / 2.5))}")

    print(f"\n  Global: {wm['global_count']} ({pct(wm['global_count'], wm['total']):.1f}%)"
          f"  Never recalled: {wm['never_recalled']} ({pct(wm['never_recalled'], wm['total']):.1f}%)"
          f"  Noise: {wm['noise_pct']}%")

    ep = s["episodic"]
    print(f"\n{'─'*W}")
    print(f"  EPISODIC: {ep['total']} items (avg imp: {ep['avg_imp']:.3f})")
    print(f"{'─'*W}")
    if ep["recent"] and not compact:
        for item in ep["recent"]:
            pv = item["preview"][:50] + "..." if len(item["preview"]) > 50 else item["preview"]
            print(f"    [{item['imp']:.2f}] {pv}")

    tg = s["triples"]
    print(f"\n{'─'*W}")
    print(f"  KNOWLEDGE GRAPH: {tg['total']} triples")
    print(f"{'─'*W}")
    if tg["predicates"] and not compact:
        for pred, c in list(tg["predicates"].items())[:8]:
            print(f"    {pred:30s} {c:3d}")

    con = s["consolidation"]
    print(f"\n{'─'*W}")
    print(f"  CONSOLIDATION: {con['events']} events, {con['items']} items")
    print(f"{'─'*W}")
    if con["recent"] and not compact:
        for ev in con["recent"][:3]:
            print(f"    {ev['date']} — {ev['items']} items")

    dr = s["dreamer"]
    print(f"\n{'─'*W}")
    print(f"  DREAMER: {dr['runs']} runs, {dr['proposals']} proposals, {dr['conflicts']} conflicts")
    print(f"{'─'*W}")
    if dr["recent"] and not compact:
        for run in dr["recent"]:
            icon = "✓" if run["status"] == "success" else "✗"
            print(f"    {icon} {run['start']} scanned={run['scanned']} proposals={run['proposals']}")

    emb_status = "OK" if s["embeddings"] > 0 else "EMPTY"
    print(f"\n  Embeddings: {s['embeddings']} ({emb_status})")

    wk = s["wiki"]
    print(f"\n{'─'*W}")
    print(f"  WIKI: {wk['total']} pages ({wk['memories']} memories, {wk['concepts']} concepts)")
    print(f"  Promotion: {wk['promotion_pct']}% of working memory → wiki")
    print(f"{'─'*W}")

    if not compact:
        if s["top_recalled"]:
            print(f"\n  TOP RECALLED:")
            for item in s["top_recalled"]:
                print(f"    {item['recalls']:4d}x [{item['imp']:.2f}] {item['text']}")
        if s["memory_age"]:
            print(f"\n  MEMORY AGE:")
            for item in s["memory_age"]:
                print(f"    {item['group']:12s}: {item['count']:4d} items  imp={item['imp']:.3f}  recall={item['recall']:.1f}")
        if s["activity"]:
            print(f"\n  SESSION ACTIVITY:")
            mx = max(item["count"] for item in s["activity"]) if s["activity"] else 1
            for item in s["activity"]:
                bl = int(item["count"] / mx * 35) if mx > 0 else 0
                print(f"    {item['date']}: {item['count']:4d} {'█' * bl}")

    # Quality
    print(f"\n{'─'*W}")
    print(f"  QUALITY INDICATORS")
    print(f"{'─'*W}")
    indicators = [
        ("Noise < 30%", wm["noise_pct"] < 30, f"{wm['noise_pct']}%"),
        ("Recall > 30%", pct(wm["never_recalled"], wm["total"]) < 70, f"{100 - pct(wm['never_recalled'], wm['total']):.1f}%"),
        ("Global > 5%", wm["global_count"] > wm["total"] * 0.05, f"{pct(wm['global_count'], wm['total']):.1f}%"),
        ("Embeddings OK", s["embeddings"] > 0, f"{s['embeddings']} vectors"),
        ("Consolidation", con["events"] > 0, f"{con['events']} events"),
        ("Dreamer active", dr["runs"] > 0, f"{dr['runs']} runs"),
        ("Wiki promoted", wk["memories"] > 10, f"{wk['memories']} pages"),
    ]
    for label, ok, detail in indicators:
        print(f"    {'✓' if ok else '✗'} {label:25s} {detail}")

    # Recommendations
    print(f"\n{'─'*W}")
    print(f"  RECOMMENDATIONS")
    print(f"{'─'*W}")
    recs = []
    if wm["noise_pct"] > 30: recs.append(f"  ! High noise ({wm['noise_pct']}%) — run cleanup-mnemosyne.py")
    if wm["never_recalled"] > wm["total"] * 0.5: recs.append(f"  ! {wm['never_recalled']} never recalled — prune low-value items")
    if ep["total"] == 0 and wm["total"] > 0: recs.append("  ! No episodic memories — consolidation not promoting")
    if s["embeddings"] == 0 and wm["total"] > 10: recs.append("  ! Embedding pipeline broken — vec_episodes empty")
    if wk["memories"] < 10 and wm["global_count"] > 20: recs.append(f"  ! Only {wk['memories']} wiki pages — promote global memories")
    if not recs: recs.append("  All systems healthy.")
    for r in recs: print(r)

    print(f"\n{'='*W}")

# ─── Snapshot / Trend Tracking ────────────────────────────────────────────────

def save_snapshot(stats):
    """Save a timestamped snapshot for trend tracking."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    snap_file = SNAPSHOT_DIR / f"snap_{ts}.json"
    snap = {
        "timestamp": datetime.now().isoformat(),
        "db_size_mb": stats.get("db_size_mb", 0),
        "wm_total": stats["working_memory"]["total"],
        "wm_noise_pct": stats["working_memory"]["noise_pct"],
        "wm_global_count": stats["working_memory"]["global_count"],
        "wm_never_recalled": stats["working_memory"]["never_recalled"],
        "ep_total": stats["episodic"]["total"],
        "triples_total": stats["triples"]["total"],
        "consolidation_events": stats["consolidation"]["events"],
        "consolidation_items": stats["consolidation"]["items"],
        "dreamer_runs": stats["dreamer"]["runs"],
        "dreamer_proposals": stats["dreamer"]["proposals"],
        "embeddings": stats["embeddings"],
        "wiki_pages": stats["wiki"]["total"],
        "wiki_memories": stats["wiki"]["memories"],
        "quality_score": stats["quality_score"],
    }
    with open(snap_file, "w") as f:
        json.dump(snap, f, indent=2)
    return snap_file

def load_trends():
    """Load recent snapshots and compute deltas."""
    if not SNAPSHOT_DIR.exists():
        return None
    snaps = sorted(SNAPSHOT_DIR.glob("snap_*.json"))
    if len(snaps) < 2:
        return None
    # Find two valid snapshots (skip corrupted ones)
    valid_snaps = []
    for snap in snaps:
        try:
            with open(snap) as f:
                data = json.load(f)
            valid_snaps.append(data)
            if len(valid_snaps) >= 2:
                break
        except (json.JSONDecodeError, IOError):
            continue
    if len(valid_snaps) < 2:
        return None
    prev, curr = valid_snaps[-2], valid_snaps[-1]
    deltas = {}
    for key in curr:
        if key == "timestamp": continue
        if key in prev and isinstance(curr[key], (int, float)):
            delta = curr[key] - prev[key]
            deltas[key] = {
                "prev": prev[key], "curr": curr[key], "delta": delta,
                "pct_change": round(delta / prev[key] * 100, 1) if prev[key] != 0 else 0,
            }
    return {"prev_time": prev.get("timestamp"), "curr_time": curr.get("timestamp"),
            "snapshots_total": len(snaps), "deltas": deltas}

def show_trends(trends):
    """Display trend data."""
    if not trends:
        print(f"\n  No trend data yet (need 2+ snapshots)")
        print(f"  Snapshots dir: {SNAPSHOT_DIR}")
        print(f"  Run: python3 mnemosyne-stats.py --save-snapshot")
        return
    print(f"\n{'─' * W}")
    print(f"  TRENDS (last 2 snapshots)")
    print(f"{'─' * W}")
    print(f"  Previous: {trends['prev_time'][:19]}")
    print(f"  Current:  {trends['curr_time'][:19]}")
    print(f"  Total snapshots: {trends['snapshots_total']}")
    print()
    trend_metrics = [
        ("wm_total", "Working Memory"), ("wm_noise_pct", "Noise %"),
        ("wm_never_recalled", "Never Recalled"), ("wm_global_count", "Global Memories"),
        ("ep_total", "Episodic Memory"), ("triples_total", "Knowledge Triples"),
        ("consolidation_events", "Consolidation Events"), ("dreamer_proposals", "Dreamer Proposals"),
        ("embeddings", "Embeddings"), ("wiki_memories", "Wiki Memories"),
        ("quality_score", "Quality Score"),
    ]
    for key, label in trend_metrics:
        if key in trends["deltas"]:
            d = trends["deltas"][key]
            arrow = "↑" if d["delta"] > 0 else "↓" if d["delta"] < 0 else "="
            print(f"    {label:20s} {d['prev']:>8} → {d['curr']:>8}  {arrow} {d['delta']:+.1f} ({d['pct_change']:+.1f}%)")

def main():
    args = sys.argv[1:]
    compact = "--compact" in args
    s = collect()
    if "--json" in args:
        print(json.dumps(s, indent=2, default=str))
    elif "--save-snapshot" in args:
        snap_file = save_snapshot(s)
        print(f"Snapshot saved: {snap_file}")
        trends = load_trends()
        if trends: show_trends(trends)
    elif "--trends" in args:
        show_trends(load_trends())
    else:
        show(s, compact=compact)
        save_snapshot(s)

if __name__ == "__main__":
    main()
