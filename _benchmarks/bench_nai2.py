#!/usr/bin/env python3
"""NAI-2 Benchmark — Temporal Decay Impact (v2, direct SQL timestamps)"""
import time, hashlib, random
from datetime import datetime, timedelta
from pathlib import Path
from mnemosyne.core.beam import BeamMemory

now = datetime.now()
random.seed(42)

SIGNALS = [
    (30, "user", "I use Python for backend development."),
    (28, "user", "Using MySQL as my primary database."),
    (25, "user", "API throughput is currently 500 requests per second."),
    (23, "user", "Switching backend from Python to Go language."),
    (20, "user", "Migrating database from MySQL to PostgreSQL."),
    (18, "user", "Throughput improved to 5000 rps after Go migration."),
    (15, "user", "Added Redis caching layer for better performance."),
    (12, "user", "Throughput now at 10000 rps with Redis."),
    (10, "user", "Deployed Go backend to production. Response time 250ms."),
    (8, "user", "Running security audit with nmap on all servers."),
    (6, "user", "All services now running as non-root for security."),
    (4, "user", "Going to San Francisco next month for a conference."),
    (2, "user", "Conference is in July. Need to finish deployment first."),
    (1, "user", "Added gin framework for the Go API."),
    (0, "user", "Final stack: Go + PostgreSQL + Redis + gin framework."),
    (29, "assistant", "Python and MySQL noted. Let me know about scaling needs."),
    (24, "assistant", "Go migration confirmed. gin or chi for web framework?"),
    (19, "assistant", "PostgreSQL migration looks good. Updating records."),
    (14, "assistant", "Redis caching is a smart move for throughput."),
    (7, "assistant", "250ms response time is solid for Go backend."),
]

FILLER = []
topics = [
    "checked the weather forecast", "read an article about AI",
    "had lunch at a new restaurant", "updated my laptop OS",
    "watched a tutorial video", "fixed a minor CSS bug",
    "cleaned up old log files", "responded to email thread",
    "updated npm dependencies", "ran a database backup",
    "tested the new API endpoint", "reviewed a pull request",
    "wrote unit tests for module", "configured CI pipeline",
    "debugged a race condition", "optimized a slow query",
    "added error handling", "refactored legacy code",
    "set up monitoring alerts", "documented API changes",
]

for day in range(30, -1, -1):
    n = random.randint(4, 8)
    for _ in range(n):
        topic = random.choice(topics)
        FILLER.append((day, "system", f"Log: {topic} on day {day}."))

TIMELINE = sorted(SIGNALS + FILLER, key=lambda x: x[0], reverse=True)

QUESTIONS = [
    ("What language does user use NOW?", ["Go"], False),
    ("What database does user use NOW?", ["PostgreSQL"], False),
    ("What is the current throughput?", ["10000", "10K"], False),
    ("What caching is used?", ["Redis"], False),
    ("What framework is used with Go?", ["gin"], False),
    ("Where is the conference?", ["San Francisco"], False),
    ("What was the ORIGINAL language?", ["Python"], True),
    ("What was the ORIGINAL database?", ["MySQL"], True),
    ("What was the ORIGINAL throughput?", ["500"], True),
    ("What throughput after Go migration?", ["5000", "5K"], True),
    ("What security tool was recommended?", ["nmap"], False),
    ("What security practice implemented?", ["non-root"], False),
    ("When is the conference?", ["July"], False),
    ("What was the response time?", ["250ms"], False),
    ("What changed: lang, db, or cache?", ["Go", "PostgreSQL", "Redis"], False),
    ("What was added most recently?", ["gin"], False),
    ("What did the user start with?", ["Python", "MySQL"], True),
    ("What does the user use now?", ["Go", "PostgreSQL", "Redis"], False),
    ("BEFORE prod deploy, what was throughput?", ["5000", "5K"], True),
    ("What was the security audit result?", ["non-root", "nmap"], False),
]


def score_answer(predicted, expected):
    hl = predicted.lower()
    hits = sum(1 for kw in expected if kw.lower() in hl)
    return hits / len(expected) if expected else 0.0


def run_bench(label, temporal_weight=0.0, temporal_halflife=168):
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = Path(tmp.name)
    tmp.close()

    beam = BeamMemory(session_id=f"bench_nai2_{label}", db_path=db_path)

    # Direct SQL insertion with correct timestamps
    t0 = time.time()
    for i, (days_ago, role, content) in enumerate(TIMELINE):
        ts = (now - timedelta(days=days_ago)).isoformat()
        mid = hashlib.sha256(f"{label}{i}{content}".encode()).hexdigest()[:16]
        beam.conn.execute(
            """INSERT OR IGNORE INTO working_memory
               (id, content, source, timestamp, session_id, importance, scope)
               VALUES (?, ?, ?, ?, ?, ?, 'global')""",
            (mid, f"[{role}] {content}", role, ts, f"bench_nai2_{label}", 0.5)
        )
    beam.conn.commit()
    ingest_ms = round((time.time()-t0)*1000)

    total = beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]

    all_scores, curr, hist = [], [], []
    first_scores, first_curr, first_hist = [], [], []
    latencies = []

    for question, expected, is_historical in QUESTIONS:
        t0 = time.time()
        results = beam.recall(
            question, top_k=10,
            temporal_weight=temporal_weight,
            temporal_halflife=temporal_halflife,
        )
        latencies.append(round((time.time()-t0)*1000))
        top5 = " ".join(r.get("content","")[:150] for r in results[:5])
        s = score_answer(top5, expected)
        all_scores.append(s)
        (hist if is_historical else curr).append(s)

        # First-answer score: only look at result #1
        first = results[0].get("content","")[:150] if results else ""
        f = score_answer(first, expected)
        first_scores.append(f)
        (first_hist if is_historical else first_curr).append(f)

    beam.conn.close()
    os.unlink(str(db_path))

    return {
        "label": label,
        "ingest_ms": ingest_ms,
        "total": total,
        "avg_top5": round(sum(all_scores)/len(all_scores), 3),
        "avg_current": round(sum(curr)/len(curr), 3) if curr else 0,
        "avg_historical": round(sum(hist)/len(hist), 3) if hist else 0,
        "avg_first": round(sum(first_scores)/len(first_scores), 3),
        "avg_first_curr": round(sum(first_curr)/len(first_curr), 3) if first_curr else 0,
        "p50_ms": sorted(latencies)[len(latencies)//2],
    }


if __name__ == "__main__":
    print(f"Timeline: {len(TIMELINE)} msgs ({len(SIGNALS)} signals + {len(FILLER)} filler)")
    print(f"Questions: {len(QUESTIONS)} (11 current, 9 historical)")
    print()

    bl = run_bench("baseline", temporal_weight=0.0)
    print(f"--- BASELINE (no temporal, k=10, {bl['total']} msgs) ---")
    print(f"  Top-5: {bl['avg_top5']:.3f}  First: {bl['avg_first']:.3f}  Cur: {bl['avg_current']:.3f}/{bl['avg_first_curr']:.3f}  Hist: {bl['avg_historical']:.3f} | {bl['p50_ms']}ms ingest {bl['ingest_ms']}ms")

    t1 = run_bench("temporal", temporal_weight=0.3, temporal_halflife=24)
    print(f"--- TEMPORAL (w=0.3, h=24h, {t1['total']} msgs) ---")
    print(f"  Top-5: {t1['avg_top5']:.3f}  First: {t1['avg_first']:.3f}  Cur: {t1['avg_current']:.3f}/{t1['avg_first_curr']:.3f}  Hist: {t1['avg_historical']:.3f} | {t1['p50_ms']}ms ingest {t1['ingest_ms']}ms")

    t2 = run_bench("strong", temporal_weight=0.6, temporal_halflife=12)
    print(f"--- STRONG (w=0.6, h=12h, {t2['total']} msgs) ---")
    print(f"  Top-5: {t2['avg_top5']:.3f}  First: {t2['avg_first']:.3f}  Cur: {t2['avg_current']:.3f}/{t2['avg_first_curr']:.3f}  Hist: {t2['avg_historical']:.3f} | {t2['p50_ms']}ms ingest {t2['ingest_ms']}ms")

    dc = t1['avg_first_curr'] - bl['avg_first_curr']
    dh = t1['avg_historical'] - bl['avg_historical']
    print()
    print(f"CURRENT: {bl['avg_current']:.3f} → {t1['avg_current']:.3f} → {t2['avg_current']:.3f}")
    print(f"HISTORICAL: {bl['avg_historical']:.3f} → {t1['avg_historical']:.3f} → {t2['avg_historical']:.3f}")
    print(f"DELTA cur={dc:+.3f} hist={dh:+.3f}")
