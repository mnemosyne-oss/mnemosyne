# Mnemosyne — Deep OpenWebUI Integration

Beyond the basic @tool wrapper, Mnemosyne offers deep integration with OpenWebUI that makes memory automatic and browsable.

## Auto-Conversation Memory

Every chat message can be automatically saved to Mnemosyne without manual `mnemosyne_remember` calls.

### Setup

```bash
pip install mnemosyne-memory

# Start the auto-save service
python -m mnemosyne.integrations.auto_save_openwebui \
    --openwebui-url http://localhost:3000 \
    --api-key your-openwebui-api-key
```

The service polls OpenWebUI's REST API every 60 seconds, saving new messages to Mnemosyne in the `openwebui` bank. Each message is tagged with:
- **source:** `openwebui:user` or `openwebui:assistant`
- **metadata:** chat_id, chat_title, role, message_id, user_id, saved_at
- **importance:** 0.6 for user messages, 0.5 for assistant responses

### Run as a Service

**Systemd:**
```ini
[Unit]
Description=Mnemosyne OpenWebUI Auto-Save
After=network.target

[Service]
ExecStart=/usr/bin/python -m mnemosyne.integrations.auto_save_openwebui --api-key YOUR_KEY
Restart=always
Environment=OPENWEBUI_URL=http://localhost:3000
Environment=MNEMOSYNE_DATA_DIR=/var/lib/mnemosyne

[Install]
WantedBy=multi-user.target
```

**Docker (with Mnemosyne MCP):**
```yaml
services:
  mnemosyne-mcp:
    image: ghcr.io/mnemosyne-oss/mnemosyne-mcp
    volumes:
      - mnemosyne-data:/data
    environment:
      - MNEMOSYNE_DATA_DIR=/data

  auto-save:
    image: python:3.11-slim
    command: python -m mnemosyne.integrations.auto_save_openwebui
             --openwebui-url http://openwebui:3000
             --api-key YOUR_KEY
    volumes:
      - mnemosyne-data:/data
    environment:
      - MNEMOSYNE_DATA_DIR=/data
```

### One-Time Sync

```bash
python -m mnemosyne.integrations.auto_save_openwebui \
    --openwebui-url http://localhost:3000 \
    --api-key your-key \
    --once
```

This imports all existing chat history into Mnemosyne without starting the daemon.

## Memory Browser Dashboard

Browse, search, and inspect all your Mnemosyne memories through a web UI.

```bash
# Install web dependencies (same as MCP SSE)
pip install fastapi uvicorn

# Start the browser
python -m mnemosyne.integrations.memory_browser --port 8081
```

Open http://localhost:8081 to see all memories with filters for source, tier (working/episodic), and sort order. Click a memory to see full detail.

### Features
- **Stats bar:** Live count of memories by tier
- **Search:** Full-text search across all memories
- **Filters:** By source, tier, and sort (recent + importance)
- **Detail view:** Full memory metadata for each entry
- **API:** JSON endpoints at `/api/search` and `/api/stats`

### API Endpoints

```bash
# Get stats
curl http://localhost:8081/api/stats

# Search
curl "http://localhost:8081/api/search?q=preferences&src=openwebui"

# Get memory detail
curl "http://localhost:8081/api/detail/MEMORY_ID"
```

## Cross-Session Memory

Once auto-save is running, the Mnemosyne @tool in OpenWebUI can recall memories from ANY conversation across ALL sessions. This is the core value:

1. Auto-save captures every chat into Mnemosyne
2. The `mnemosyne_recall` tool searches across all saved memories
3. Users can ask "what did we discuss about X last week?" and get accurate results

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENWEBUI_URL` | `http://localhost:3000` | OpenWebUI base URL |
| `OPENWEBUI_API_KEY` | (none) | API key for auth |
| `MNEMOSYNE_DATA_DIR` | `~/.hermes/mnemosyne/data/` | Memory database location |
| `AUTO_SAVE_INTERVAL` | `60` | Polling interval in seconds |
| `AUTO_SAVE_BANK` | `openwebui` | Memory bank name |
