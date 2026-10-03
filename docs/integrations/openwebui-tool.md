# Mnemosyne + OpenWebUI

Add persistent memory to OpenWebUI via a native @tool.

## Setup

1. Install Mnemosyne:

```bash
pip install mnemosyne-memory
```

2. Create a bridge file in OpenWebUI's `data/tools/` directory:

```bash
echo "from openwebui_tool import MnemosyneTool as Mnemosyne" \
  > /path/to/open-webui/data/tools/mnemosyne.py
```

3. In OpenWebUI, go to **Workspace > Tools**. The `mnemosyne` tool appears automatically (hot-reloaded).

4. In a chat, select the Mnemosyne tool from the model's tool dropdown.

## Configuration

The tool provides a settings panel in OpenWebUI with these Valves:

| Setting | Default | Description |
|---------|---------|-------------|
| `db_path` | `~/.hermes/mnemosyne/data/` | Path to memory database directory |
| `bank` | `default` | Memory bank name for isolation |
| `top_k` | `5` | Max results returned by recall |
| `vec_weight` | `null` (`config.yaml` > `MNEMOSYNE_VEC_WEIGHT` > `0.5`) | Vector search weighting |
| `fts_weight` | `null` (`config.yaml` > `MNEMOSYNE_FTS_WEIGHT` > `0.3`) | Full-text search weighting |
| `importance_weight` | `null` (`config.yaml` > `MNEMOSYNE_IMPORTANCE_WEIGHT` > `0.2`) | Importance boost weighting |
| `show_citations` | `true` | Show source citations in output |

These settings are persisted per-user in OpenWebUI's database.

## Usage

The tool provides these operations:

| Operation | Description |
|-----------|-------------|
| `mnemosyne_remember` | Store a memory |
| `mnemosyne_recall` | Search memories by semantic similarity |
| `mnemosyne_forget` | Delete a specific memory |
| `mnemosyne_stats` | View memory statistics |
| `mnemosyne_sleep` | Run consolidation |

The LLM decides when to use each tool based on the conversation context.

## Tips

- **Memory isolation**: Use different banks per chat/workspace for privacy
- **Data location**: Override `MNEMOSYNE_DATA_DIR` env var to change storage path
- **Testing**: Import the tool in Python directly to verify it works:
  ```python
  import asyncio
  from openwebui_tool import MnemosyneTool
  tool = MnemosyneTool()
  result = asyncio.run(tool.mnemosyne_remember("test memory"))
  print(result)
  ```
