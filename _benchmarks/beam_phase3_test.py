#!/usr/bin/env python3
"""
Phase 3 BEAM Benchmark - Direct Integration Test
Runs the full benchmark pipeline correctly (avoiding module caching issues).
"""
import sys, os, tempfile, json, time
from pathlib import Path

# Ensure we use the local mnemosyne
sys.path.insert(0, str(Path(__file__).parent.parent))

from _benchmarks.evaluate_beam_end_to_end import (
    LLMClient, evaluate_conversation, load_beam_dataset,
    ingest_conversation, init_beam, BeamMemory, compute_ability_scores,
    print_sota_report, DEFAULT_TOP_K
)
from datetime import datetime, timezone

RESULTS_FILE = Path(__file__).parent.parent / "results" / "beam_e2e_results.json"
SUMMARY_FILE = Path(__file__).parent.parent / "results" / "beam_e2e_summary.json"

def main():
    print("=" * 80)
    print("  BEAM Phase 3 End-to-End Evaluation")
    print("  Scale: 100K, Sample: 1 conversation")
    print("=" * 80)
    
    # Load dataset
    print("\n[1/3] Loading dataset...")
    data = load_beam_dataset(['100K'], max_conversations=1)
    scale = '100K'
    conv = data[scale][0]
    print(f"  Loaded: {len(conv['messages'])} messages, {len(conv['questions'])} questions")
    
    # Create DB and ingest
    print("\n[2/3] Ingesting...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / f"beam_{scale}_{conv['id']}.db"
        init_beam(db_path)
        beam = BeamMemory(session_id=f"beam_{scale}_{conv['id']}", db_path=db_path)
        
        stats = ingest_conversation(beam, conv['messages'])
        print(f"  WM: {beam.get_working_stats()['total']}, EP: {beam.get_episodic_stats()['total']}, SP: {stats.get('sp_count', 0)}")
        
        # Create LLM clients
        llm = LLMClient(model='nvidia/nemotron-3-super-120b-a12b:free')
        judge_llm = LLMClient(model='nvidia/nemotron-3-super-120b-a12b:free')
        
        # Evaluate
        print("\n[3/3] Evaluating...")
        result = evaluate_conversation(llm, judge_llm, beam, conv, set())
        
        # Compute scores
        ability_summary = compute_ability_scores([result])
        
        # Print results
        print("\n" + "=" * 80)
        print("  RESULTS")
        print("=" * 80)
        
        by_ability = {}
        for r in result['results']:
            ab = r['ability']
            by_ability[ab] = by_ability.get(ab, []) + [r['score']]
        
        for ab, scores in sorted(by_ability.items()):
            avg = sum(scores) / len(scores)
            print(f"  {ab}: {avg:.1%} (n={len(scores)})")
        
        overall = sum(r['score'] for r in result['results']) / len(result['results']) if result['results'] else 0
        print(f"  OVERALL: {overall:.1%}")
        
        # Save
        os.makedirs(RESULTS_FILE.parent, exist_ok=True)
        metadata = {
            "date": datetime.now(timezone.utc).isoformat(),
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "scales": ["100K"],
            "total_conversations": 1,
        }
        with open(RESULTS_FILE, "w") as f:
            json.dump({"metadata": metadata, "results": [result]}, f, indent=2)
        
        with open(SUMMARY_FILE, "w") as f:
            json.dump(ability_summary, f, indent=2)
        
        print(f"\n  Saved to: {RESULTS_FILE}")
        print(f"  Summary: {SUMMARY_FILE}")
        
        beam.conn.close()
        llm.close()
        judge_llm.close()

if __name__ == "__main__":
    main()
