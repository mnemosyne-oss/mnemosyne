#!/usr/bin/env python3
"""NAI-0 Minimal Benchmark — 20 questions across 1 synthetic conversation.

Quick and dirty: ingest 1 fake user+assistant chat, ask 20 questions,
measure keyword-overlap accuracy before and after Phase 0 optimizations.
"""
import json
import time
from pathlib import Path
from mnemosyne.core.beam import BeamMemory

# ── Synthetic Conversation: 15 turns, 5 factual, 5 temporal, 5 multi-hop signals ──

CONVERSATION = [
    ("user", "Hey, can you help me set up my development environment?"),
    ("assistant", "Sure! First, let's install Python 3.12. Download it from python.org."),
    ("user", "I'm on Ubuntu 24.04. I prefer using dark mode for everything."),
    ("assistant", "Got it. For Ubuntu, use apt: sudo apt install python3.12. Also noted your dark mode preference."),
    ("user", "What about Docker? I need to containerize my services."),
    ("assistant", "Docker is separate. sudo apt install docker.io. Make sure to add your user to the docker group."),
    ("user", "The API I'm building needs to handle about 10,000 requests per second."),
    ("assistant", "That's high throughput. You'll want async Python with FastAPI, behind an nginx reverse proxy."),
    ("user", "I changed my mind. I want to use Go instead of Python for the API."),
    ("assistant", "OK, switching to Go. Install Go from golang.org, then use gin or chi for the web framework."),
    ("user", "Last week I deployed a test server and the response time was about 250ms."),
    ("assistant", "250ms is good. For 10K rps in Go, you should get under 50ms. Want help optimizing?"),
    ("user", "Yes please. Is my system secure enough for production?"),
    ("assistant", "Run a security audit: check open ports with nmap, ensure all services run as non-root users."),
    ("user", "Great. One more thing - I'm going to San Francisco next month for a conference."),
    ("assistant", "Nice! SF in June is great. I'll note that down. For the trip, we should finish the deploy first."),
]

# 20 questions: 10 factual, 5 temporal, 5 multi-hop
QUESTIONS = [
    # Factual (direct recall)
    ("What operating system is the user on?", ["Ubuntu", "24.04"]),
    ("What display preference does the user have?", ["dark mode"]),
    ("What tool does the user need to containerize services?", ["Docker"]),
    ("What Python feature was suggested for high throughput?", ["FastAPI", "async"]),
    ("What language did the user switch to for the API?", ["Go", "golang"]),
    ("What web framework was suggested for the new language?", ["gin", "chi"]),
    ("What was the response time of the test server?", ["250ms", "250"]),
    ("What does the user need to install Docker on Ubuntu?", ["apt", "apt install"]),
    ("What city is the user traveling to?", ["San Francisco"]),
    ("What type of application is the API expected to be?", ["high throughput", "10,000", "10K"]),
    # Temporal
    ("When did the user deploy the test server?", ["last week"]),
    ("When is the conference the user mentioned?", ["next month"]),
    ("What OS version is the user on?", ["Ubuntu", "24.04"]),
    ("What was discussed before the user mentioned San Francisco?", ["security", "secure"]),
    ("What was the last topic discussed?", ["conference", "San Francisco", "deploy"]),
    # Multi-hop
    ("What language was chosen after the user changed their mind?", ["Go", "golang"]),
    ("What framework should be used for the new language choice?", ["gin", "chi"]),
    ("How should the user improve throughput in the new language?", ["optimizing", "50ms"]),
    ("What two installation methods were discussed?", ["apt", "python.org", "apt install", "golang.org"]),
    ("What security steps were recommended?", ["nmap", "non-root", "security audit"]),
]


def score_answer(predicted: str, expected: list) -> float:
    """Simple keyword-overlap score. 1.0 = all keywords found."""
    predicted_lower = predicted.lower()
    hits = sum(1 for kw in expected if kw.lower() in predicted_lower)
    return hits / len(expected) if expected else 0.0


def run_benchmark(label: str, top_k: int = 40, use_format: bool = False) -> dict:
    """Run full benchmark pipeline."""
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = Path(tmp.name)
    tmp.close()

    beam = BeamMemory(session_id=f"bench_nai0_{label}", db_path=db_path)

    # Ingest conversation
    t_start = time.time()
    for role, msg in CONVERSATION:
        beam.remember(f"[{role}] {msg}", source=role, importance=0.6 if role == "user" else 0.5)
    ingest_time = time.time() - t_start

    # Answer questions
    scores = []
    coverage_scores = []  # How many of expected keywords appear in ALL retrieved results
    total_latency = 0
    results_preview = []
    total_results = 0

    for question, expected in QUESTIONS:
        t0 = time.time()
        results = beam.recall(question, top_k=top_k)
        recall_time = time.time() - t0
        total_results += len(results)

        # Build full context from all retrieved results
        full_context = " ".join(r.get("content", "")[:200] for r in results)
        
        # Top-5 score (same as before for comparison)
        top5_content = " ".join(r.get("content", "")[:100] for r in results[:5])
        s5 = score_answer(top5_content, expected)
        scores.append(s5)

        # Coverage score: search ALL results, not just top-5
        sc = score_answer(full_context, expected)
        coverage_scores.append(sc)

        total_latency += recall_time

        if len(results_preview) < 3:
            results_preview.append({
                "question": question[:50],
                "top5_score": s5,
                "coverage_score": sc,
                "latency_ms": round(recall_time * 1000),
                "results": len(results),
            })

    avg_score = sum(scores) / len(scores)
    avg_coverage = sum(coverage_scores) / len(coverage_scores)

    beam.conn.close()
    os.unlink(str(db_path))

    return {
        "label": label,
        "avg_top5_score": round(avg_score, 3),
        "avg_coverage": round(avg_coverage, 3),
        "p50_latency_ms": round((total_latency / len(QUESTIONS)) * 1000),
        "ingest_time_ms": round(ingest_time * 1000),
        "avg_results": round(total_results / len(QUESTIONS)),
        "preview": results_preview,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("NAI-0 Minimal Benchmark — Phase 0 Algorithmic Sprint")
    print("=" * 60)
    print(f"Conversation: {len(CONVERSATION)} turns")
    print(f"Questions: {len(QUESTIONS)} (10 factual, 5 temporal, 5 multi-hop)")
    print()

    # Baseline: old k=5, no formatting
    print("--- BASELINE (k=5, no formatting) ---")
    baseline = run_benchmark("baseline", top_k=5, use_format=False)
    print(f"  Top-5 Score: {baseline['avg_top5_score']:.3f}")
    print(f"  Coverage:    {baseline['avg_coverage']:.3f} (full k=5 context)")
    print(f"  P50 Latency: {baseline['p50_latency_ms']}ms")
    print(f"  Avg Results: {baseline['avg_results']}")
    for p in baseline["preview"]:
        print(f"    Q: {p['question']}... => top5={p['top5_score']:.2f} cov={p['coverage_score']:.2f} ({p['latency_ms']}ms, {p['results']}r)")

    print()
    print("--- PHASE 0 (k=40, RRF, sandwich formatting) ---")
    phase0 = run_benchmark("phase0", top_k=40, use_format=True)
    print(f"  Top-5 Score: {phase0['avg_top5_score']:.3f}")
    print(f"  Coverage:    {phase0['avg_coverage']:.3f} (full k=40 context)")
    print(f"  P50 Latency: {phase0['p50_latency_ms']}ms")
    print(f"  Avg Results: {phase0['avg_results']}")
    for p in phase0["preview"]:
        print(f"    Q: {p['question']}... => top5={p['top5_score']:.2f} cov={p['coverage_score']:.2f} ({p['latency_ms']}ms, {p['results']}r)")

    delta_cov = phase0["avg_coverage"] - baseline["avg_coverage"]
    print()
    print("=" * 60)
    print(f"COVERAGE DELTA: {delta_cov:+.3f} ({delta_cov/baseline['avg_coverage']*100:+.1f}% vs baseline)" if baseline["avg_coverage"] > 0 else f"COVERAGE DELTA: {delta_cov:+.3f}")
    print(f"RESULTS SCALE:  {baseline['avg_results']} → {phase0['avg_results']} ({phase0['avg_results']/baseline['avg_results']:.1f}x more context)")
    print("=" * 60)
