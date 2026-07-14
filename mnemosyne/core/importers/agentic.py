"""
Agentic (LLM-guided) memory importer.

For providers without programmatic export APIs, this generates:
1. A Python migration script the user can run in their environment
2. Step-by-step instructions for manual extraction
3. Instructions the user can give to their AI agent to perform the migration

This covers the "agent extraction" use case:
- User tells their AI agent (Claude, ChatGPT, etc.) to export from Provider X
- Agent follows Mnemosyne-provided instructions to produce a JSON file
- User runs `hermes mnemosyne import --file export.json` to ingest
"""




# Known provider metadata for script generation
PROVIDER_SCRIPTS = {
    "mem0": {
        "install": "pip install mem0ai",
        "api_import": "from mem0 import MemoryClient",
        "extract_code": """# Connect to Mem0
client = MemoryClient(api_key="YOUR_API_KEY", host="https://api.mem0.ai")

# Extract all memories (paginated)
all_memories = []
page = 1
while True:
    resp = client.get_all(filters={"user_id": "YOUR_USER_ID"}, page=page, page_size=200)
    results = resp.get("results", [])
    all_memories.extend(results)
    if resp.get("next") is None:
        break
    page += 1

# Save to JSON
import json
with open("mem0_export.json", "w") as f:
    json.dump(all_memories, f, indent=2)

print(f"Exported {len(all_memories)} memories to mem0_export.json")""",
        "env_hint": "MEM0_API_KEY",
    },
    "letta": {
        "install": "pip install letta-client",
        "api_import": "from letta_client import Letta",
        "extract_code": """client = Letta(api_key="YOUR_API_KEY")

# Export specific agent or list all
agent_id = "YOUR_AGENT_ID"  # or list agents: agents = client.agents.list()
agent_file = client.agents.export_file(agent_id)

with open("letta_export.json", "w") as f:
    json.dump(agent_file, f, indent=2)

print(f"Exported agent to letta_export.json")""",
        "env_hint": "LETTA_API_KEY",
    },
    "zep": {
        "install": "pip install zep-cloud",
        "api_import": "from zep_cloud.client import Zep",
        "extract_code": """client = Zep(api_key="YOUR_API_KEY")

# List users
users = client.user.list_ordered()
all_data = []

for user in users.get("users", []):
    uid = user["user_id"]
    sessions = client.user.get_sessions(uid)
    for session in sessions:
        sid = session["session_id"]
        mem = client.memory.get(sid)
        all_data.append({
            "user_id": uid,
            "session_id": sid,
            "messages": mem.get("messages", []),
            "summary": mem.get("summary", ""),
            "facts": mem.get("relevant_facts", []),
        })

with open("zep_export.json", "w") as f:
    json.dump(all_data, f, indent=2)

print(f"Exported {len(all_data)} sessions to zep_export.json")""",
        "env_hint": "ZEP_API_KEY",
    },
    "cognee": {
        "install": "pip install cognee",
        "api_import": "import cognee\nimport asyncio",
        "extract_code": """async def extract():
    graph_data = await cognee.graph_db.get_graph_data()
    # graph_data is (nodes, edges) tuple
    nodes, edges = graph_data

    export = {
        "nodes": [{"id": str(n[0]), "properties": n[1]} for n in nodes],
        "edges": [{"source": str(e[0]), "target": str(e[1]), "label": str(e[2]), "properties": e[3] if len(e) > 3 else {}} for e in edges],
    }

    with open("cognee_export.json", "w") as f:
        json.dump(export, f, indent=2)

    print(f"Exported {len(nodes)} nodes and {len(edges)} edges")

asyncio.run(extract())""",
        "env_hint": None,
    },
    "honcho": {
        "install": "pip install honcho-ai",
        "api_import": "from honcho import Honcho",
        "extract_code": """honcho = Honcho(workspace_id="YOUR_WORKSPACE_ID")

# List peers
peers = honcho.list_peers()
all_messages = []

for peer in peers.get("peers", []):
    peer_id = peer["peer_id"]
    sessions = honcho.list_sessions(peer_id=peer_id)
    for session in sessions.get("sessions", []):
        sid = session["session_id"]
        messages = honcho.session(sid).list_messages()
        for msg in messages.get("messages", []):
            all_messages.append({
                "peer_id": peer_id,
                "session_id": sid,
                "content": msg["content"],
                "role": "user",
                "timestamp": msg.get("created_at"),
            })

with open("honcho_export.json", "w") as f:
    json.dump(all_messages, f, indent=2)

print(f"Exported {len(all_messages)} messages")""",
        "env_hint": None,
    },
    "supermemory": {
        "install": "pip install supermemory",
        "api_import": "from supermemory import SuperMemory",
        "extract_code": """client = SuperMemory(api_key="YOUR_API_KEY")

# List documents
docs = client.documents.list()
all_memories = []

for doc in docs:
    all_memories.append({
        "content": doc.get("content", ""),
        "container_tag": doc.get("containerTag", ""),
        "is_static": doc.get("isStatic", False),
        "created_at": doc.get("createdAt"),
    })

# Also search for memories
results = client.search.execute(q="*", containerTags=["YOUR_CONTAINER_TAG"])
for mem in results.get("results", []):
    all_memories.append({
        "content": mem.get("content", mem.get("memory", "")),
        "container_tag": mem.get("containerTag", ""),
        "created_at": mem.get("createdAt"),
    })

with open("supermemory_export.json", "w") as f:
    json.dump(all_memories, f, indent=2)

print(f"Exported {len(all_memories)} items")""",
        "env_hint": "SUPERMEMORY_API_KEY",
    },
}


class AgenticImporter:
    """Generate migration scripts and instructions for any provider.

    This doesn't extract data itself — it produces scripts and prompts
    that the user or their AI agent can use to perform the extraction.

    Usage:
        gen = AgenticImporter()
        script = gen.generate_script("mem0", api_key="sk-xxx", user_id="alice")
        print(script)

        # Or generate agent instructions
        instructions = gen.generate_agent_instructions("zep")
        print(instructions)
    """

    def generate_script(self, provider: str, **kwargs) -> str:
        """Generate a ready-to-run Python migration script.

        Args:
            provider: Provider name (mem0, letta, zep, etc.)
            **kwargs: api_key, user_id, agent_id, workspace_id, container_tag

        Returns:
            Python script as a string.
        """
        meta = PROVIDER_SCRIPTS.get(provider)
        if not meta:
            return self._generate_generic_script(provider, **kwargs)

        api_key = kwargs.get("api_key", "YOUR_API_KEY")
        install = meta["install"]
        api_import = meta["api_import"]
        extract = meta["extract_code"]
        env_hint = meta.get("env_hint", "")

        header = f'''#!/usr/bin/env python3
"""
Mnemosyne Migration Script: {provider.title()} → Mnemosyne
Generated by Mnemosyne Agentic Importer

Prerequisites:
    {install}

Environment:
{f"    Set {env_hint} env var or pass api_key directly." if env_hint else "    No special env vars required."}

Usage:
    python3 migrate_{provider}.py
    # Then: hermes mnemosyne import --file {provider}_export.json
"""

import json
import os

# Configuration
API_KEY = os.environ.get("{env_hint}", "{api_key}")
'''

        script = f'''{header}
{api_import}

{extract}

print("\\nNext step:")
print("  hermes mnemosyne import --file {provider}_export.json")
'''
        return script

    def generate_agent_instructions(self, provider: str) -> str:
        """Generate instructions the user can paste to their AI agent.

        The user gives these instructions to their AI agent (Claude, ChatGPT, etc.)
        and the agent follows them to extract memories and produce a JSON file.
        """
        meta = PROVIDER_SCRIPTS.get(provider)
        install = meta["install"] if meta else f"pip install <{provider}-sdk>"

        return f"""I need to migrate all my memories from {provider.title()} to Mnemosyne.

Please help me extract ALL memories from {provider.title()} and save them as a JSON file.

Requirements:
1. Install the SDK: `{install}`
2. Extract every available memory/message/fact/document
3. Save to a JSON file called `{provider}_export.json`
4. The JSON should be an array of objects, each with at minimum a "content" key
5. Preserve any metadata, timestamps, and user/agent IDs

After extraction, I'll run:
    hermes mnemosyne import --file {provider}_export.json

Please give me the exact Python script to run. Test it for syntax errors before giving it to me."""

    def generate_docs_instructions(self, provider: str) -> str:
        """Generate step-by-step migration instructions for documentation."""
        meta = PROVIDER_SCRIPTS.get(provider)
        if not meta:
            return self._generate_generic_docs(provider)

        install = meta["install"]
        env_hint = meta.get("env_hint", "")

        return f"""## Migrating from {provider.title()} to Mnemosyne

### Step 1: Install dependencies
```bash
pip install mnemosyne-memory
{install}
```

{ "### Step 2: Set API key" + chr(10) + "```bash" + chr(10) + f"export {env_hint}=sk-xxx" + chr(10) + "```" if env_hint else ""}

### Step 3: Run the migration script
Save the script below as `migrate_{provider}.py` and run it:

```bash
python3 migrate_{provider}.py
```

This extracts all memories to `{provider}_export.json`.

### Step 4: Import into Mnemosyne
```bash
hermes mnemosyne import --file {provider}_export.json
```

Or via CLI provider import:
```bash
hermes mnemosyne import --from {provider} --api-key sk-xxx
```

### Step 5: Verify
```bash
hermes mnemosyne stats
```
"""

    def _generate_generic_script(self, provider: str, **kwargs) -> str:
        """Generate a generic extraction script for unknown providers."""
        return f'''#!/usr/bin/env python3
"""
Generic migration script for {provider.title()}

Replace the extraction logic below with your provider's API calls.
"""

import json

all_memories = []

# TODO: Replace with your provider's extraction logic
# Example: client = YourProviderSDK(api_key="...")
# Example: memories = client.list_all()

# Each memory should be a dict with at minimum "content":
# {{"content": "the memory text", "metadata": {{}}, "timestamp": "2026-01-01T00:00:00Z"}}

with open("{provider}_export.json", "w") as f:
    json.dump(all_memories, f, indent=2)

print(f"Exported {{len(all_memories)}} items to {provider}_export.json")
print("Then run: hermes mnemosyne import --file {provider}_export.json")
'''

    def _generate_generic_docs(self, provider: str) -> str:
        """Generate generic docs for unknown providers."""
        return f"""## Migrating from {provider.title()} to Mnemosyne

{provider.title()} doesn't have a built-in importer yet. Use the generic extraction approach:

### Option A: Export manually
Export your memories from {provider.title()} to a JSON file, then import:

```bash
hermes mnemosyne import --file export.json
```

### Option B: Use your AI agent
Paste these instructions to your AI agent:

```
I need to extract ALL memories from {provider.title()} and save them as a JSON file.
The JSON should be an array of objects with a "content" key for each memory.
Save as "{provider}_export.json" so I can import into Mnemosyne with:
hermes mnemosyne import --file {provider}_export.json
```

### Option C: Request an importer
Open an issue on GitHub requesting a {provider.title()} importer:
https://github.com/AxDSan/mnemosyne/issues
"""


# Convenience functions

def generate_migration_script(provider: str, **kwargs) -> str:
    """Generate a ready-to-run migration script for the given provider."""
    gen = AgenticImporter()
    return gen.generate_script(provider, **kwargs)


def generate_agent_instructions(provider: str) -> str:
    """Generate instructions for an AI agent to perform the migration."""
    gen = AgenticImporter()
    return gen.generate_agent_instructions(provider)


def generate_docs_instructions(provider: str) -> str:
    """Generate step-by-step documentation for migration."""
    gen = AgenticImporter()
    return gen.generate_docs_instructions(provider)
