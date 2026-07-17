"""
Mem0 memory provider importer.

Extracts all memories from a Mem0 instance (cloud or self-hosted)
and imports them into Mnemosyne.

Extraction methods (tried in order):
1. Mem0 Python SDK — `client.get_all()` with pagination
2. Mem0 Platform API — `create_memory_export()` structured export
3. Mem0 OSS REST API — `GET /memories` endpoint

Field mapping:
    memory → content
    user_id / agent_id → author_id
    metadata → metadata
    categories → metadata tags
    created_at → timestamp
    app_id → channel_id
    run_id → preserved in metadata
"""

import json
from datetime import datetime
from typing import List, Dict

from mnemosyne.core.importers.base import BaseImporter, ImporterResult


class Mem0Importer(BaseImporter):
    """Import memories from Mem0 into Mnemosyne.

    Usage:
        importer = Mem0Importer(
            api_key="sk-xxx",
            user_id="alice",      # optional: filter by user
            agent_id=None,        # optional: filter by agent
            base_url=None,        # for self-hosted Mem0 OSS
            page_size=200,        # items per paginated request
        )
        result = importer.run(mnemosyne_instance)
    """

    provider_name = "mem0"

    def __init__(self, api_key: str = None, user_id: str = None,
                 agent_id: str = None, app_id: str = None,
                 base_url: str = None, page_size: int = 200,
                 **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.user_id = user_id
        self.agent_id = agent_id
        self.app_id = app_id
        self.base_url = base_url
        self.page_size = min(page_size, 200)  # Mem0 max is 200

    def extract(self) -> List[Dict]:
        """Extract all memories from Mem0.

        Tries SDK first, falls back to REST API.
        """
        # Try Python SDK first
        try:
            return self._extract_via_sdk()
        except ImportError:
            pass

        # Fall back to REST API
        try:
            return self._extract_via_rest()
        except Exception:
            pass

        # Try structured export (Platform only)
        try:
            return self._extract_via_export()
        except Exception:
            pass

        raise RuntimeError(
            "Could not extract memories from Mem0. "
            "Install the SDK: pip install mem0ai\n"
            "Or provide a base_url for self-hosted REST API."
        )

    def _extract_via_sdk(self) -> List[Dict]:
        """Extract using the Mem0 Python SDK."""
        from mem0 import MemoryClient

        client = MemoryClient(
            api_key=self.api_key,
            host=self.base_url,
        )

        filters = {}
        if self.user_id:
            filters["user_id"] = self.user_id
        if self.agent_id:
            filters["agent_id"] = self.agent_id
        if self.app_id:
            filters["app_id"] = self.app_id

        # If no entity filter, try to get all by using wildcard user_id
        if not filters:
            filters["user_id"] = "*"

        all_memories = []
        page = 1
        while True:
            resp = client.get_all(
                filters=filters,
                page=page,
                page_size=self.page_size,
            )
            results = resp.get("results", [])
            all_memories.extend(results)

            if resp.get("next") is None or len(results) == 0:
                break
            page += 1

        return all_memories

    def _extract_via_rest(self) -> List[Dict]:
        """Extract using Mem0 OSS REST API."""
        import urllib.request
        import urllib.parse

        url = f"{self.base_url or 'http://localhost:8000'}/memories"
        params = {}
        if self.user_id:
            params["user_id"] = self.user_id
        if self.agent_id:
            params["agent_id"] = self.agent_id

        if params:
            url += "?" + urllib.parse.urlencode(params)

        req = urllib.request.Request(url)
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("results", data.get("memories", data.get("data", [])))
        return []

    def _extract_via_export(self) -> List[Dict]:
        """Extract using Mem0 Platform structured export."""
        from mem0 import MemoryClient

        client = MemoryClient(
            api_key=self.api_key,
            host=self.base_url,
        )

        filters = {}
        if self.user_id:
            filters["user_id"] = self.user_id
        if self.agent_id:
            filters["agent_id"] = self.agent_id

        # Create a generic schema to get all memory content
        # The export API requires a Pydantic schema — use a catch-all
        export_job = client.create_memory_export(
            filters=filters,
            export_instructions=(
                "Return ALL memories as a flat JSON array. "
                "Include every memory without filtering or summarizing. "
                "Format: [{'content': '...', 'user_id': '...', 'created_at': '...', 'metadata': {...}}, ...]"
            ),
        )

        export_id = export_job.get("id") or export_job.get("export_id")
        if not export_id:
            raise RuntimeError("Export job created but no export_id returned")

        result = client.get_memory_export(memory_export_id=export_id)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("results", result.get("memories", result.get("data", [])))
        return []

    def transform(self, raw_data: List[Dict]) -> List[Dict]:
        """Transform Mem0 memories to Mnemosyne-compatible format."""
        memories = []

        for item in raw_data:
            # Resolve content field — Mem0 uses 'memory', others use 'content'
            content = item.get("memory", item.get("content", ""))
            if not content:
                continue

            # Author identity: prefer user_id, fall back to agent_id
            mem0_user = item.get("user_id")
            mem0_agent = item.get("agent_id")
            author_id = None
            author_type = "human"
            if mem0_user:
                author_id = f"mem0_user:{mem0_user}"
                author_type = "human"
            elif mem0_agent:
                author_id = f"mem0_agent:{mem0_agent}"
                author_type = "agent"

            # Channel: use app_id as channel
            channel_id = item.get("app_id")

            # Timestamp
            timestamp = item.get("created_at") or item.get("timestamp")
            if not timestamp:
                timestamp = datetime.now().isoformat()

            # Build metadata preserving Mem0-specific fields
            metadata = item.get("metadata", {}) or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {"raw": str(metadata)}

            # Preserve Mem0-native fields in metadata
            metadata["_mem0_id"] = item.get("id", "")
            metadata["_mem0_hash"] = item.get("hash", "")
            metadata["_mem0_run_id"] = item.get("run_id", "")
            if item.get("categories"):
                metadata["_mem0_categories"] = item.get("categories")

            # Importance: infer from metadata or default
            importance = 0.5
            if "importance" in metadata:
                try:
                    importance = float(metadata.pop("importance"))
                except (ValueError, TypeError):
                    pass

            # Updated timestamp
            updated_at = item.get("updated_at")
            if updated_at and updated_at != timestamp:
                metadata["_updated_at"] = updated_at

            memories.append({
                "content": content,
                "source": "mem0_import",
                "importance": importance,
                "metadata": metadata,
                "valid_until": None,
                "scope": "session",
                # Custom fields for Mnemosyne identity layer
                "_author_id": author_id,
                "_author_type": author_type,
                "_channel_id": channel_id,
                "_timestamp": timestamp,
            })

        return memories

    def run(self, mnemosyne, dry_run: bool = False,
            session_id: str = None, channel_id: str = None) -> ImporterResult:
        """Override run() to handle identity-aware import."""
        result = ImporterResult(
            provider=self.provider_name,
            started_at=datetime.now().isoformat(),
        )

        try:
            raw_data = self.extract()
            result.total = len(raw_data)

            if result.total == 0:
                result.errors.append("No memories found to import from Mem0")
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
                    # Extract identity fields before passing to remember()
                    author_id = mem_dict.pop("_author_id", None)
                    author_type = mem_dict.pop("_author_type", None)
                    chan = mem_dict.pop("_channel_id", None) or channel_id
                    ts = mem_dict.pop("_timestamp", None)

                    # Build metadata with timestamp
                    meta = mem_dict.get("metadata", {})
                    if ts:
                        meta["imported_at_original"] = ts

                    # Override session_id if specified
                    if session_id:
                        sid = session_id
                    else:
                        sid = None  # Let Mnemosyne use its own session

                    mid = mnemosyne.remember(
                        content=mem_dict["content"],
                        source=mem_dict.get("source", self.provider_name),
                        importance=mem_dict.get("importance", 0.5),
                        metadata=meta,
                        valid_until=mem_dict.get("valid_until"),
                        scope=mem_dict.get("scope", "session"),
                    )

                    # Store identity via triple or direct beam write
                    # We use the Mnemosyne's beam directly for identity columns
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
                            pass  # Identity update is best-effort

                    result.memory_ids.append(mid)
                    result.imported += 1

                except Exception as e:
                    result.failed += 1
                    result.errors.append(
                        f"Failed to import '{mem_dict.get('content', '')[:80]}': {e}"
                    )

        except Exception as e:
            result.errors.append(f"Mem0 import failed: {e}")

        result.finished_at = datetime.now().isoformat()
        return result


# Convenience function
def import_from_mem0(api_key: str, mnemosyne, user_id: str = None,
                     agent_id: str = None, base_url: str = None,
                     dry_run: bool = False, session_id: str = None,
                     channel_id: str = None) -> ImporterResult:
    """Import all memories from a Mem0 instance into Mnemosyne.

    Args:
        api_key: Mem0 API key (or set MEM0_API_KEY env var).
        mnemosyne: Mnemosyne instance to import into.
        user_id: Optional Mem0 user_id to filter by.
        agent_id: Optional Mem0 agent_id to filter by.
        base_url: Optional base URL for self-hosted Mem0.
        dry_run: If True, validate but don't write.
        session_id: Override session for imported memories.
        channel_id: Channel for imported memories.

    Returns:
        ImporterResult with counts and errors.
    """
    importer = Mem0Importer(
        api_key=api_key,
        user_id=user_id,
        agent_id=agent_id,
        base_url=base_url,
    )
    return importer.run(
        mnemosyne,
        dry_run=dry_run,
        session_id=session_id,
        channel_id=channel_id,
    )
