#!/usr/bin/env python3
"""NAI-1 Benchmark — Chat Normalization Impact

Generates a messy chat conversation (contractions, emojis, filler, fragments),
ingests it with and without normalization, and measures coverage delta.
"""
import json, time
from pathlib import Path
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.chat_normalize import normalize_chat, extraction_rate

# ── Same factual content as bench_nai0, but in messy chat form ──

MESSY_CONVERSATION = [
    ("user", "yo wassup can u help me setup my dev thingy lol"),
    ("assistant", "yea sure!! first u gotta install python 3.12. download it from python.org"),
    ("user", "im on ubuntu 24.04 bro. also i kinda prefer that dark mode thing for everything tbh fr"),
    ("assistant", "got it. for ubuntu use apt: sudo apt install python3.12. also i noted ur dark pref np"),
    ("user", "wbu that container thing? docker or whatever, i need to containerize my stuff lol"),
    ("assistant", "docker is separate. sudo apt install docker.io. make sure to add ur user to the docker group"),
    ("user", "the api im building gotta handle like 10k requests per sec omg fr fr"),
    ("assistant", "thats a lot! u'll want async python w fastapi behind nginx reverse proxy"),
    ("user", "actually nvm changed my mind lmao. gonna use go instead of python for the api"),
    ("assistant", "ok bet. switching to go. install from golang.org, use gin or chi for the web framework"),
    ("user", "yo last week i deployed a test server n the response time was like ~250ms"),
    ("assistant", "250ms is solid. for 10k rps in go u should get <50ms. want help optimizing?"),
    ("user", "yea pls. btw is my system secure enough for prod lol?"),
    ("assistant", "run a security audit: check open ports w nmap, make sure all services run as non-root tbh"),
    ("user", "aight bet. one more thing fr - im going to sf next month for a conference"),
    ("assistant", "nice!! sf in june is great. i'll note that. for the trip, finish deploy first"),
]

# Same questions but adapted for messy context
QUESTIONS = [
    ("What operating system is the user on?", ["Ubuntu", "24.04"]),
    ("What display preference does the user have?", ["dark mode"]),
    ("What tool does the user need to containerize services?", ["Docker"]),
    ("What language did the user switch to for the API?", ["Go", "golang"]),
    ("What was the response time of the test server?", ["250ms", "250"]),
    ("What city is the user traveling to?", ["San Francisco", "SF"]),
    ("What framework was suggested for Go?", ["gin", "chi"]),
    ("What security steps were recommended?", ["nmap", "non-root"]),
    ("What is the deployment OS?", ["Ubuntu", "24.04"]),
    ("What Python framework was suggested?", ["FastAPI"]),
    ("When was the test server deployed?", ["last week"]),
    ("When is the conference?", ["next month"]),
    ("What was discussed before San Francisco?", ["security", "secure"]),
    ("What language was chosen after mind change?", ["Go", "golang"]),
    ("What throughput is needed?", ["10k", "10,000"]),
    ("What should response time be in Go?", ["50ms", "50"]),
    ("What two installation methods were mentioned?", ["apt", "python.org", "golang.org"]),
    ("What was the last topic?", ["conference", "deploy"]),
    ("What should the user install on Ubuntu?", ["apt", "python3.12"]),
    ("What change did the user make to their API plans?", ["Go", "golang", "go"]),
]


def score_answer(predicted, expected):
    predicted_lower = predicted.lower()
    hits = sum(1 for kw in expected if kw.lower() in predicted_lower)
    return hits / len(expected) if expected else 0.0


def run_bench(label, messages, normalize=False):
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = Path(tmp.name)
    tmp.close()
    
    beam = BeamMemory(session_id=f"bench_nai1_{label}", db_path=db_path)
    
    # Ingest with optional normalization
    t0 = time.time()
    dropped = 0
    for role, msg in messages:
        if normalize:
            cleaned = normalize_chat(msg)
            if cleaned is None:
                dropped += 1
                continue
            beam.remember(f"[{role}] {cleaned}", source=role, importance=0.6 if role=="user" else 0.5)
        else:
            beam.remember(f"[{role}] {msg}", source=role, importance=0.6 if role=="user" else 0.5)
    ingest_ms = round((time.time()-t0)*1000)
    
    # Answer questions
    scores = []
    coverage_scores = []
    latencies = []
    
    for question, expected in QUESTIONS:
        t0 = time.time()
        results = beam.recall(question, top_k=40)
        latencies.append(round((time.time()-t0)*1000))
        
        top5 = " ".join(r.get("content","")[:100] for r in results[:5])
        full = " ".join(r.get("content","")[:200] for r in results)
        
        scores.append(score_answer(top5, expected))
        coverage_scores.append(score_answer(full, expected))
    
    beam.conn.close()
    os.unlink(str(db_path))
    
    return {
        "label": label,
        "ingest_ms": ingest_ms,
        "dropped": dropped,
        "avg_top5": round(sum(scores)/len(scores), 3),
        "avg_coverage": round(sum(coverage_scores)/len(coverage_scores), 3),
        "p50_ms": sorted(latencies)[len(latencies)//2],
    }


if __name__ == "__main__":
    print("=" * 60)
    print("NAI-1 Benchmark — Chat Normalization Impact")
    print("=" * 60)
    
    # Extract rate on the raw messages
    raw_msgs = [m[1] for m in MESSY_CONVERSATION]
    rate = extraction_rate(raw_msgs)
    print(f"Raw messages: {rate['total']}")
    print(f"Survived normalization: {rate['survived']} ({rate['rate']:.0%})")
    print(f"Dropped: {rate['dropped_samples']}")
    print()
    
    print("--- RAW (no normalization) ---")
    raw = run_bench("raw", MESSY_CONVERSATION, normalize=False)
    print(f"  Top-5: {raw['avg_top5']:.3f}  Coverage: {raw['avg_coverage']:.3f}")
    print(f"  Ingest: {raw['ingest_ms']}ms  P50: {raw['p50_ms']}ms")
    
    print()
    print("--- NORMALIZED ---")
    norm = run_bench("norm", MESSY_CONVERSATION, normalize=True)
    print(f"  Top-5: {norm['avg_top5']:.3f}  Coverage: {norm['avg_coverage']:.3f}")
    print(f"  Ingest: {norm['ingest_ms']}ms  P50: {norm['p50_ms']}ms")
    print(f"  Dropped: {norm['dropped']}/{len(MESSY_CONVERSATION)} messages")
    
    delta = norm['avg_coverage'] - raw['avg_coverage']
    print()
    print("=" * 60)
    print(f"COVERAGE DELTA: {delta:+.3f} (normalized vs raw)")
    print("=" * 60)
