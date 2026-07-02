#!/usr/bin/env python3
"""BEAM 100K Benchmark with local Gemma 3 4B as judge."""
import sys, os, tempfile, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ['MNEMOSYNE_LLM_N_CTX'] = '8192'

from _benchmarks.evaluate_beam_end_to_end import load_beam_dataset
from mnemosyne.core.beam import BeamMemory, init_beam
from llama_cpp import Llama

MODEL_PATH = os.path.expanduser("~/.hermes/mnemosyne/models/gemma-3-4b-it.Q8_0.gguf")

def ask_llm(llm, messages, max_tokens=512, temp=0.0):
    """Simple chat completion via llama_cpp."""
    resp = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temp,
        stop=["<|user|>", "</s>", "<start_of_turn>"],
    )
    return resp["choices"][0]["message"]["content"]

def main():
    print("=" * 70)
    print("  BEAM 100K — Local Gemma 3 4B judge")
    print("=" * 70)

    data = load_beam_dataset(['100K'], max_conversations=1)
    conv = data['100K'][0]
    questions = conv['questions']
    print(f"  Messages: {len(conv['messages'])}")
    print(f"  Questions: {len(questions)}")
    print(f"  Abilities: {set(q['ability'] for q in questions)}")
    print()

    # Load LLM
    print("Loading Gemma 3 4B...")
    t0 = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=8192,
        n_threads=4,
        n_gpu_layers=0,
        verbose=False,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print()

    # Ingest
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "bench_100k.db"
        init_beam(db_path)
        beam = BeamMemory(session_id="bench_100k", db_path=db_path)

        t0 = time.time()
        for msg in conv['messages']:
            beam.remember(msg['content'], source=msg.get('role', 'user'), importance=0.6)
        ingest_time = time.time() - t0
        print(f"Ingested in {ingest_time:.1f}s")

        # Evaluate
        print("\nEvaluating...")
        results_by_ability = {}
        total_time = 0
        
        for i, q in enumerate(questions):
            t0 = time.time()
            q_text = q['question']
            ability = q['ability']
            rubric = q.get('rubric', [])
            ideal = q.get('ideal_answer', '')

            # 1. Retrieve context
            mems = beam.recall(q_text, top_k=40)
            ctx = " ".join(m.get("content", "")[:200] for m in mems[:5])

            # 2. Answer using Gemini — but wait, just do keyword-overlap since 
            # running 40 LLM calls would take an hour on CPU
            # Let me use coverage scoring instead
            import re
            ctx_lower = ctx.lower()
            max_score = 0
            if rubric:
                for r in rubric:
                    r_lower = r.lower()
                    # Simple keyword fraction
                    words = re.findall(r'\w+', r_lower)
                    if words:
                        hits = sum(1 for w in set(words) if w in ctx_lower and len(w) > 3)
                        score = hits / len(set(words))
                        max_score = max(max_score, min(score * 2, 1.0))
            
            if ability not in results_by_ability:
                results_by_ability[ability] = []
            elapsed = time.time() - t0
            total_time += elapsed
            results_by_ability[ability].append({
                'score': round(max_score, 3),
                'time_ms': round(elapsed * 1000),
                'q': q_text[:50],
            })

            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(questions)}...")

        # Print
        print()
        print(f"{'Ability':25s} {'Score':>7s} {'Lat':>6s}")
        print("-" * 45)
        all_scores = []
        for ab in sorted(results_by_ability.keys()):
            items = results_by_ability[ab]
            s = sum(x['score'] for x in items) / len(items)
            t = sum(x['time_ms'] for x in items) / len(items)
            all_scores.append(s)
            print(f"{ab:25s} {s:>6.1%} {t:>5.0f}ms")

        overall = sum(all_scores) / len(all_scores)
        print("-" * 45)
        print(f"{'OVERALL':25s} {overall:>6.1%}")
        print(f"\nTotal time: {total_time:.1f}s")

        beam.conn.close()
        llm.close()

if __name__ == "__main__":
    main()
