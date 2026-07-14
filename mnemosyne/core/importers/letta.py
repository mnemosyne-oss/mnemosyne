"""
Letta (formerly MemGPT) memory provider importer.

Extracts memories from Letta's AgentFile (.af) format and imports
them into Mnemosyne. Letta uses a hierarchical memory model:
- Memory Blocks (core/working memory) → working_memory
- Archival Memory (long-term vector storage) → episodic_memory
- Message History → episodic with source="letta_message"
- Tools & Agent Config → metadata

Extraction methods:
1. AgentFile (.af) export via Letta SDK: client.agents.export_file(agent_id)
2. Archival memory via SDK search
3. Direct .af file parsing (offline)
"""

import json
from datetime import datetime
from typing import List, Dict
from pathlib import Path

from mnemosyne.core.importers.base import BaseImporter, ImporterResult


class LettaImporter(BaseImporter):
    """Import memories from Letta into Mnemosyne.

    Usage:
        importer = LettaImporter(
            api_key="sk-xxx",         # Letta API key
            agent_id="agent-uuid",    # optional: specific agent
            base_url=None,            # for self-hosted Letta
            agent_file_path=None,     # path to a .af file (offline import)
        )
        result = importer.run(mnemosyne_instance)
    """

    provider_name = "letta"

    def __init__(self, api_key: str = None, agent_id: str = None,
                 base_url: str = None, agent_file_path: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.agent_id = agent_id
        self.base_url = base_url
        self.agent_file_path = agent_file_path

    def extract(self) -> List[Dict]:
        """Extract memories from Letta.

        Priority: 1) offline .af file, 2) SDK export, 3) REST API.
        """
        # Offline .af file
        if self.agent_file_path:
            return self._extract_from_file(self.agent_file_path)

        # Try SDK
        try:
            return self._extract_via_sdk()
        except (ImportError, Exception):
            pass

        # Try REST API
        try:
            return self._extract_via_rest()
        except Exception:
            pass

        raise RuntimeError(
            "Could not extract memories from Letta. "
            "Provide --agent-file-path for offline .af import, "
            "or install the SDK: pip install letta-client"
        )

    def _extract_from_file(self, filepath: str) -> List[Dict]:
        """Parse a Letta AgentFile (.af) directly."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"AgentFile not found: {filepath}")

        # .af files can be JSON, YAML, or TOML
        content = path.read_text()

        # Try JSON first
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try YAML
            try:
                import yaml
                data = yaml.safe_load(content)
            except (ImportError, Exception):
                # Try TOML
                try:
                    import tomllib
                    data = tomllib.loads(content)
                except (ImportError, Exception):
                    raise RuntimeError(
                        f"Could not parse AgentFile: {filepath}. "
                        "Supported formats: JSON, YAML, TOML"
                    )

        return self._parse_agent_data(data)

    def _extract_via_sdk(self) -> List[Dict]:
        """Extract using Letta Python SDK."""
        from letta_client import Letta

        client = Letta(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        if self.agent_id:
            # Export specific agent
            agent_file = client.agents.export_file(self.agent_id)
            return self._parse_agent_data(agent_file)
        else:
            # List all agents and export each
            agents = client.agents.list()
            all_data = []
            for agent in agents:
                try:
                    agent_file = client.agents.export_file(agent.id)
                    all_data.extend(self._parse_agent_data(agent_file))
                except Exception:
                    continue
            return all_data

    def _extract_via_rest(self) -> List[Dict]:
        """Extract using Letta REST API directly."""
        import urllib.request

        base = self.base_url or "https://api.letta.com"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers["Content-Type"] = "application/json"

        if self.agent_id:
            url = f"{base}/v1/agents/{self.agent_id}/export"
        else:
            url = f"{base}/v1/agents/export"

        req = urllib.request.Request(url, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        return self._parse_agent_data(data)

    def _parse_agent_data(self, data: dict) -> List[Dict]:
        """Parse AgentFile data into memory dicts."""
        memories = []
        agent_id = data.get("agent_id", data.get("id", "letta_agent"))
        agent_name = data.get("agent_name", data.get("name", "unknown"))

        # 1. Memory Blocks (core working memory)
        blocks = data.get("memory_blocks", data.get("blocks", {}))
        if isinstance(blocks, list):
            for block in blocks:
                label = block.get("label", block.get("name", "memory"))
                value = block.get("value", block.get("content", ""))
                if not value:
                    continue
                content = f"[{label}] {value}"
                memories.append({
                    "content": content,
                    "source": "letta_block",
                    "importance": 0.8 if block.get("read_only") else 0.6,
                    "metadata": {
                        "letta_label": label,
                        "letta_block_id": block.get("id", block.get("uuid", "")),
                        "letta_agent_id": agent_id,
                        "letta_agent_name": agent_name,
                    },
                })

        # If blocks is a dict of label→value
        elif isinstance(blocks, dict):
            for label, value in blocks.items():
                if not value:
                    continue
                memories.append({
                    "content": f"[{label}] {value}",
                    "source": "letta_block",
                    "importance": 0.7,
                    "metadata": {
                        "letta_label": label,
                        "letta_agent_id": agent_id,
                        "letta_agent_name": agent_name,
                    },
                })

        # 2. Message History
        messages = data.get("messages", data.get("message_history", []))
        for msg in messages:
            role = msg.get("role", msg.get("role_type", "user"))
            content = msg.get("content", msg.get("text", ""))
            if not content:
                continue
            ts = msg.get("created_at", msg.get("timestamp"))
            memories.append({
                "content": content,
                "source": "letta_message",
                "importance": 0.3,
                "metadata": {
                    "letta_role": role,
                    "letta_agent_id": agent_id,
                    "letta_agent_name": agent_name,
                    "_timestamp": ts or datetime.now().isoformat(),
                },
            })

        # 3. System prompt / persona
        sys_prompt = data.get("system_prompt", data.get("persona", ""))
        if sys_prompt:
            memories.append({
                "content": f"[system_prompt] {sys_prompt[:2000]}",
                "source": "letta_system",
                "importance": 0.9,
                "metadata": {
                    "letta_agent_id": agent_id,
                    "letta_agent_name": agent_name,
                },
            })

        return memories

    def transform(self, raw_data: List[Dict]) -> List[Dict]:
        """Transform Letta data to Mnemosyne format."""
        memories = []
        for item in raw_data:
            content = item.get("content", "")
            if not content:
                continue

            memories.append({
                "content": content,
                "source": item.get("source", "letta_import"),
                "importance": float(item.get("importance", 0.5)),
                "metadata": item.get("metadata", {}),
                "valid_until": None,
                "scope": "session",
                "_author_id": f"letta_agent:{item.get('metadata', {}).get('letta_agent_id', 'unknown')}",
                "_author_type": "agent",
                "_channel_id": None,
                "_timestamp": item.get("metadata", {}).get("_timestamp"),
            })

        return memories

    def run(self, mnemosyne, dry_run=False, session_id=None, channel_id=None):
        """Override run to handle identity-aware import."""
        result = ImporterResult(provider=self.provider_name,
                                started_at=datetime.now().isoformat())

        try:
            raw_data = self.extract()
            result.total = len(raw_data)
            if result.total == 0:
                result.errors.append("No memories found in Letta export")
                return result
            if not self.validate(raw_data):
                result.errors.append("Validation failed")
                return result

            memories = self.transform(raw_data)
            if dry_run:
                result.imported = len(memories)
                return result

            for mem_dict in memories:
                try:
                    author_id = mem_dict.pop("_author_id", None)
                    author_type = mem_dict.pop("_author_type", None)
                    chan = mem_dict.pop("_channel_id", None) or channel_id
                    ts = mem_dict.pop("_timestamp", None)
                    meta = mem_dict.get("metadata", {})
                    if ts:
                        meta["imported_at_original"] = ts

                    mid = mnemosyne.remember(
                        content=mem_dict["content"],
                        source=mem_dict.get("source", self.provider_name),
                        importance=mem_dict.get("importance", 0.5),
                        metadata=meta,
                        valid_until=mem_dict.get("valid_until"),
                        scope=mem_dict.get("scope", "session"),
                    )
                    if author_id or author_type or chan:
                        try:
                            mnemosyne.beam.conn.execute("""
                                UPDATE working_memory
                                SET author_id = COALESCE(author_id, ?),
                                    author_type = COALESCE(author_type, ?),
                                    channel_id = COALESCE(channel_id, ?)
                                WHERE id = ?
                            """, (author_id, author_type, chan, mid))
                            mnemosyne.beam.conn.commit()
                        except Exception:
                            pass
                    result.memory_ids.append(mid)
                    result.imported += 1
                except Exception as e:
                    result.failed += 1
                    result.errors.append(f"Failed: {str(e)[:100]}")

        except Exception as e:
            result.errors.append(f"Letta import failed: {e}")

        result.finished_at = datetime.now().isoformat()
        return result
