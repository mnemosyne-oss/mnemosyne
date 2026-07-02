#!/usr/bin/env python3
"""Quick 100K benchmark: 1 conversation, 20 questions, keyword-overlap scoring."""
import sys, os, tempfile, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from _benchmarks.evaluate_beam_end_to_end import load_beam_dataset
from mnemosyne.core.beam import BeamMemory, init_beam

def score_answer(predicted: str, expected: list) -> float:
    predicted_lower = predicted.lower()
    hits = sum(1 for kw in expected if kw.lower() in predicted_lower)
    return hits / len(expected) if expected else 0.0

def main():
    print("=" * 70)
    print("  BEAM 100K Benchmark — 1 conversation, keyword scoring")
    print("=" * 70)

    data = load_beam_dataset(['100K'], max_conversations=1)
    conv = data['100K'][0]
    questions = conv['questions']
    messages = conv['messages']
    print(f"  Messages: {len(messages)}")
    print(f"  Questions: {len(questions)}")
    print(f"  Abilities: {set(q['ability'] for q in questions)}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "bench_100k.db"
        init_beam(db_path)
        beam = BeamMemory(session_id="bench_100k", db_path=db_path)

        # Ingest
        t0 = time.time()
        for msg in messages:
            beam.remember(msg['content'], source=msg.get('role', 'user'), importance=0.6)
        ingest_time = time.time() - t0
        wm = beam.get_working_stats()
        ep = beam.get_episodic_stats()
        print(f"  Ingest: {ingest_time:.1f}s | WM: {wm['total']} | EP: {ep['total']}")
        print()

        # Query
        results_by_ability = {}
        total_latency = 0
        for q in questions:
            t0 = time.time()
            results = beam.recall(q['question'], top_k=40)
            latency = time.time() - t0
            total_latency += latency

            full_context = " ".join(r.get("content", "")[:200] for r in results)
            top5 = " ".join(r.get("content", "")[:100] for r in results[:5])
            score = score_answer(top5, q.get('rubric_keywords', q.get('expected_keywords', [])))
            coverage = score_answer(full_context, q.get('rubric_keywords', q.get('expected_keywords', [])))

            ab = q['ability']
            if ab not in results_by_ability:
                results_by_ability[ab] = []
            results_by_ability[ab].append({'question': q['question'][:60], 'top5': score, 'coverage': coverage, 'latency_ms': round(latency*1000), 'n': len(results)})

        # Results
        print(f"{'Ability':20s} {'Top-5':>8s} {'Cov':>6s} {'Lat':>6s} {'n':>4s}")
        print("-" * 50)
        overall_scores = []
        overall_cov = []
        for ab in sorted(results_by_ability.keys()):
            items = results_by_ability[ab]
            avg_s = sum(x['top5'] for x in items) / len(items)
            avg_c = sum(x['coverage'] for x in items) / len(items)
            avg_l = sum(x['latency_ms'] for x in items) / len(items)
            overall_scores.append(avg_s)
            overall_cov.append(avg_c)
            print(f"{ab:20s} {avg_s:>7.1%} {avg_c:>5.1%} {avg_l:>5.0f}ms {len(items):>4d}")

        overall = sum(overall_scores) / len(overall_scores)
        overall_c = sum(overall_cov) / len(overall_cov)
        avg_lat = total_latency / len(questions) * 1000
        print("-" * 50)
        print(f"{'OVERALL':20s} {overall:>7.1%} {overall_c:>5.1%} {avg_lat:>5.0f}ms {len(questions):>4d}")
        print()

        # Show 5 lowest-scoring questions
        all_items = [(q['question'], q['top5'], q['coverage']) for items in results_by_ability.values() for q in items]
        all_items.sort(key=lambda x: x[1])
        print("Bottom 5 questions (by top-5 score):")
        for q, s, c in all_items[:5]:
            print(f"  {s:.0%}/{c:.0%}  {q}")

        beam.conn.close()

if __name__ == "__main__":
    main()
