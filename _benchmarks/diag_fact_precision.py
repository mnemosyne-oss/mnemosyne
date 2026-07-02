#!/usr/bin/env python3
"""Diagnose fact_recall precision: what % of returned facts are actually relevant?"""
import sys, os, tempfile, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from _benchmarks.evaluate_beam_end_to_end import load_beam_dataset, init_beam, ingest_conversation
from mnemosyne.core.beam import BeamMemory

# LLM for judging relevance
class JudgeLLM:
    def __init__(self):
        import openai
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    def judge_relevance(self, question: str, fact_text: str) -> bool:
        try:
            resp = self.client.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[{
                    "role": "user",
                    "content": f"QUESTION: {question}\n\nFACT: {fact_text}\n\nIs this fact RELEVANT to answering the question? Answer ONLY 'yes' or 'no'. Relevant means it contains information that helps answer the question, even partially."
                }],
                temperature=0.0,
                max_tokens=5,
            )
            answer = resp.choices[0].message.content.strip().lower()
            return answer.startswith("yes")
        except Exception:
            return False

def main():
    # Load 100K data
    print("Loading 100K BEAM conversation...")
    data = load_beam_dataset(["100K"], max_conversations=1)
    convs = data.get("100K", [])
    if not convs:
        print("ERROR: No 100K data")
        sys.exit(1)
    conv = convs[0]
    questions = conv.get("questions", [])[:16]  # Max 16 questions
    print(f"  {len(conv['messages'])} messages, {len(questions)} questions")

    # Ingest
    print("\nIngesting (use_cloud=True)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "diag.db"
        init_beam(db_path)
        beam = BeamMemory(session_id="diag", db_path=db_path, use_cloud=True)

        t0 = time.perf_counter()
        stats = ingest_conversation(beam, conv["messages"])
        print(f"  Done in {time.perf_counter()-t0:.1f}s, DB: {os.path.getsize(db_path)/1024:.0f}KB")

        facts_count = beam.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        print(f"  Facts stored: {facts_count}")

        # Show random sample facts
        samples = beam.conn.execute(
            "SELECT subject, predicate, object FROM facts ORDER BY RANDOM() LIMIT 5"
        ).fetchall()
        print("  Sample facts:")
        for s in samples:
            print(f"    [{s['predicate']}] {s['subject']}: {s['object'][:80]}")

        # Diagnose each question
        print("\n--- FACT RECALL PRECISION DIAGNOSTIC ---\n")
        judge = JudgeLLM()

        total_precision_5 = []
        total_precision_10 = []
        total_precision_30 = []
        total_coverage = []  # How many relevant facts out of all facts?

        for qi, q in enumerate(questions):
            question = q.get("question", q.get("text", ""))
            rubric = q.get("rubric", q.get("ideal", []))
            ability = q.get("ability", "?")

            facts = beam.fact_recall(question, top_k=30)

            if not facts:
                print(f"[{qi}] {ability}: 0 facts returned, question: {question[:80]}")
                total_precision_5.append(0.0)
                total_precision_10.append(0.0)
                total_precision_30.append(0.0)
                total_coverage.append(0.0)
                continue

            # Judge top 30 for relevance
            relevant_indices = set()
            for fi, f in enumerate(facts[:30]):
                is_rel = judge.judge_relevance(question, f["content"])
                if is_rel:
                    relevant_indices.add(fi)

            p5 = len([i for i in relevant_indices if i < 5]) / min(5, len(facts))
            p10 = len([i for i in relevant_indices if i < 10]) / min(10, len(facts))
            p30 = len([i for i in relevant_indices if i < 30]) / min(30, len(facts))

            total_precision_5.append(p5)
            total_precision_10.append(p10)
            total_precision_30.append(p30)
            total_coverage.append(len(relevant_indices))

            print(f"[{qi}] {ability}: p@5={p5:.0%} p@10={p10:.0%} p@30={p30:.0%} "
                  f"({len(relevant_indices)}/{len(facts)} relevant) "
                  f"Q: {question[:60]}")

        # Aggregate
        print(f"\n{'='*60}")
        print(f"AGGREGATE (n={len(questions)} questions):")
        avg_p5 = sum(total_precision_5) / len(total_precision_5)
        avg_p10 = sum(total_precision_10) / len(total_precision_10)
        avg_p30 = sum(total_precision_30) / len(total_precision_30)
        avg_cov = sum(total_coverage) / len(total_coverage)

        print(f"  Precision@5:  {avg_p5:.1%}")
        print(f"  Precision@10: {avg_p10:.1%}")
        print(f"  Precision@30: {avg_p30:.1%}")
        print(f"  Avg relevant facts: {avg_cov:.1f}")
        print(f"  Zero-recall questions: {sum(1 for c in total_coverage if c == 0)}/{len(total_coverage)}")
        print(f"  p@5 ≥ 50%: {sum(1 for p in total_precision_5 if p >= 0.5)}/{len(total_precision_5)}")

        verdict = "GOOD" if avg_p10 >= 0.5 else "BROKEN"
        print(f"\n  VERDICT: Recall pipeline is {verdict}")

        beam.conn.close()

if __name__ == "__main__":
    main()
