from mnemosyne.core.beam import BeamMemory
from datetime import datetime

# Diagnostic: isolate TR failure
beam = BeamMemory(session_id="diagnostic_tr")
query = "How many weeks do I have between finishing the transaction management sprint and moving?"
print(f"--- TR DIAGNOSTIC ---")
print(f"QUERY: {query}\n")

# Run search with high verbosity in memory layer
# Note: Using standard recall to verify current baseline performance
results = beam.recall(query, top_k=10)

for i, r in enumerate(results):
    # Snippet + timestamp comparison
    print(f"RANK {i+1}: ID={r['id']}")
    print(f"SNIPPET: {r['content'][:150]}")
    print(f"META: {r.get('metadata_json', 'N/A')}\n")
