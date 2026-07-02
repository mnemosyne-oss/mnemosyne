#!/usr/bin/env python3
"""NAI-2 Re-ranking Benchmark — LLM cross-attention vs temporal baseline"""
import time, hashlib, random
from datetime import datetime, timedelta
from pathlib import Path
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.rerank import rerank as llm_rerank, rerank_available

now = datetime.now()
random.seed(42)

TIMELINE = []
# Signals (same as bench_nai2)
signals = [
    (30, "user", "I use Python for backend development."),
    (28, "user", "Using MySQL as primary database."),
    (25, "user", "API throughput currently 500 rps."),
    (23, "user", "Switching backend from Python to Go language."),
    (20, "user", "Migrating database from MySQL to PostgreSQL."),
    (18, "user", "Throughput improved to 5000 rps after Go."),
    (15, "user", "Added Redis caching layer."),
    (12, "user", "Throughput now at 10000 rps with Redis."),
    (10, "user", "Deployed Go backend. Response time 250ms."),
    (8, "user", "Running security audit with nmap."),
    (6, "user", "All services running as non-root."),
    (4, "user", "Going to San Francisco next month."),
    (2, "user", "Conference is in July."),
    (1, "user", "Added gin framework for Go API."),
    (0, "user", "Final stack: Go + PostgreSQL + Redis + gin."),
    (29, "assistant", "Python and MySQL noted."),
    (24, "assistant", "Go migration confirmed. gin or chi?"),
    (19, "assistant", "PostgreSQL migration noted."),
    (14, "assistant", "Redis caching is smart."),
    (7, "assistant", "250ms response time is solid."),
]
# Filler
topics = ["checked weather", "read AI article", "had lunch", "updated OS",
          "watched tutorial", "fixed CSS bug", "cleaned logs", "email thread",
          "npm deps", "db backup", "tested endpoint", "reviewed PR",
          "unit tests", "CI pipeline", "race condition", "slow query",
          "error handling", "refactored code", "monitoring", "API docs"]
for day in range(30, -1, -1):
    for _ in range(random.randint(4,8)):
        TIMELINE.append((day, "system", f"Log: {random.choice(topics)} on day {day}."))
TIMELINE = sorted(signals + TIMELINE, key=lambda x: x[0], reverse=True)

# 10 key questions (speed over coverage)
QUESTIONS = [
    ("What language does user use NOW?", ["Go"], False),
    ("What database NOW?", ["PostgreSQL"], False),
    ("What is current throughput?", ["10000", "10K"], False),
    ("What caching is used?", ["Redis"], False),
    ("What framework with Go?", ["gin"], False),
    ("Where is conference?", ["San Francisco"], False),
    ("What was ORIGINAL language?", ["Python"], True),
    ("What was ORIGINAL database?", ["MySQL"], True),
    ("What was ORIGINAL throughput?", ["500"], True),
    ("What security tool?", ["nmap"], False),
]


def score_answer(predicted, expected):
    hl = predicted.lower()
    hits = sum(1 for kw in expected if kw.lower() in hl)
    return hits / len(expected) if expected else 0.0


def setup_db():
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = Path(tmp.name)
    tmp.close()
    beam = BeamMemory(session_id="bench_rerank", db_path=db_path)
    for days_ago, role, content in TIMELINE:
        ts = (now - timedelta(days=days_ago)).isoformat()
        beam.conn.execute(
            """INSERT OR IGNORE INTO working_memory
               (id, content, source, timestamp, session_id, importance, scope)
               VALUES (?, ?, ?, ?, ?, ?, 'global')""",
            (hashlib.sha256(f"rr{len(TIMELINE)}{content}".encode()).hexdigest()[:16],
             f"[{role}] {content}", role, ts, "bench_rerank", 0.5))
    beam.conn.commit()
    return beam, db_path


def run_bench(label, beam, use_rerank=False):
    all_scores, curr_scores = [], []
    latencies = []
    for question, expected, is_hist in QUESTIONS:
        t0 = time.time()
        results = beam.recall(question, top_k=20, temporal_weight=0.3, temporal_halflife=24)

        if use_rerank and len(results) > 5:
            reranked = llm_rerank(question, results, top_k=5, timeout=60)
            if reranked:
                results = reranked

        lat = round((time.time()-t0)*1000)
        latencies.append(lat)
        top5 = " ".join(r.get("content","")[:150] for r in results[:5])
        s = score_answer(top5, expected)
        all_scores.append(s)
        if not is_hist:
            curr_scores.append(s)

    return {
        "label": label,
        "avg": round(sum(all_scores)/len(all_scores), 3),
        "avg_curr": round(sum(curr_scores)/len(curr_scores), 3) if curr_scores else 0,
        "p50_ms": sorted(latencies)[len(latencies)//2],
        "total_ms": sum(latencies),
    }


if __name__ == "__main__":
    print(f"Timeline: {len(TIMELINE)} msgs, Questions: {len(QUESTIONS)}")
    print(f"LLM available: {rerank_available()}")
    print()

    beam, db_path = setup_db()
    print("--- BASELINE (temporal, no re-rank, k=20) ---")
    bl = run_bench("baseline", beam)
    print(f"  Avg: {bl['avg']:.3f}  Current: {bl['avg_curr']:.3f}  P50: {bl['p50_ms']}ms  Total: {bl['total_ms']}ms")
    beam.conn.close()
    import os; os.unlink(str(db_path))

    beam, db_path = setup_db()
    print()
    print("--- RE-RANKED (temporal + LLM re-rank, k=20→5) ---")
    rr = run_bench("rerank", beam, use_rerank=True)
    print(f"  Avg: {rr['avg']:.3f}  Current: {rr['avg_curr']:.3f}  P50: {rr['p50_ms']}ms  Total: {rr['total_ms']}ms")

    dc = rr['avg_curr'] - bl['avg_curr']
    print()
    print(f"CURRENT Q DELTA: {dc:+.3f}")
    if dc > 0.03:
        print("VERDICT: Keep re-ranking (>3pp improvement)")
    else:
        print(f"VERDICT: Cut re-ranking (<3pp, delta={dc:+.3f})")
    beam.conn.close()
    os.unlink(str(db_path))
