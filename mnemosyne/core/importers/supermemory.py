"""
SuperMemory memory provider importer.

SuperMemory is a cloud-hosted memory API that organizes memories
with container tags (not sessions). Extraction via:
- client.documents.list() — get all documents
- client.search.execute() — semantic search per container

SuperMemory has documented migration paths from Mem0 and Zep.
"""

import json
from datetime import datetime
from typing import List, Dict

from mnemosyne.core.importers.base import BaseImporter, ImporterResult


class SuperMemoryImporter(BaseImporter):
    """Import memories from SuperMemory into Mnemosyne.

    Usage:
        importer = SuperMemoryImporter(
            api_key="sk-xxx",          # SuperMemory API key
            container_tag="my-app",    # optional: filter by container
        )
        result = importer.run(mnemosyne_instance)
    """

    provider_name = "supermemory"

    def __init__(self, api_key: str = None, container_tag: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.container_tag = container_tag

    def extract(self) -> List[Dict]:
        """Extract memories from SuperMemory."""
        try:
            return self._extract_via_sdk()
        except ImportError:
            pass
        try:
            return self._extract_via_rest()
        except Exception:
            pass
        raise RuntimeError(
            "Could not extract from SuperMemory. Install: pip install supermemory"
        )

    def _extract_via_sdk(self) -> List[Dict]:
        """Extract using SuperMemory Python SDK."""
        from supermemory import SuperMemory

        client = SuperMemory(api_key=self.api_key)

        all_items = []

        # Get documents
        try:
            docs = client.documents.list()
            if isinstance(docs, list):
                for doc in docs:
                    content = doc.get("content", doc.get("text", ""))
                    if content:
                        all_items.append({
                            "content": content,
                            "source": "supermemory_document",
                            "container_tag": doc.get("containerTag", ""),
                            "is_static": doc.get("isStatic", False),
                            "timestamp": doc.get("createdAt"),
                            "metadata": doc.get("metadata", {}),
                        })
        except Exception:
            pass

        # Search for memories by container
        if self.container_tag:
            try:
                resp = client.search.execute(
                    q="*",
                    containerTags=[self.container_tag],
                )
                results = resp.get("results", resp.get("memories", []))
                for mem in results:
                    content = mem.get("content", mem.get("memory", ""))
                    if content:
                        all_items.append({
                            "content": content,
                            "source": "supermemory_memory",
                            "container_tag": mem.get("containerTag", self.container_tag),
                            "is_static": mem.get("isStatic", False),
                            "timestamp": mem.get("createdAt"),
                            "metadata": mem.get("metadata", {}),
                        })
            except Exception:
                pass

        return all_items

    def _extract_via_rest(self) -> List[Dict]:
        """Extract using SuperMemory REST API."""
        import urllib.request

        base = "https://api.supermemory.ai"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        all_items = []

        # List documents
        try:
            req = urllib.request.Request(f"{base}/v4/documents", headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                docs = json.loads(resp.read().decode())
                if isinstance(docs, list):
                    for doc in docs:
                        content = doc.get("content", "")
                        if content:
                            all_items.append({
                                "content": content,
                                "source": "supermemory_document",
                                "container_tag": doc.get("containerTag", ""),
                                "timestamp": doc.get("createdAt"),
                                "metadata": doc.get("metadata", {}),
                            })
        except Exception:
            pass

        # Add memories directly
        try:
            payload = json.dumps({"q": "*"}).encode()
            if self.container_tag:
                payload = json.dumps({
                    "q": "*",
                    "containerTags": [self.container_tag],
                }).encode()
            req = urllib.request.Request(
                f"{base}/v4/search",
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("results", data.get("memories", []))
                for mem in results:
                    content = mem.get("content", mem.get("memory", ""))
                    if content:
                        all_items.append({
                            "content": content,
                            "source": "supermemory_memory",
                            "container_tag": mem.get("containerTag", ""),
                            "is_static": mem.get("isStatic", False),
                            "timestamp": mem.get("createdAt"),
                            "metadata": mem.get("metadata", {}),
                        })
        except Exception:
            pass

        return all_items

    def transform(self, raw_data: List[Dict]) -> List[Dict]:
        """Transform SuperMemory data to Mnemosyne format."""
        memories = []
        for item in raw_data:
            content = item.get("content", "")
            if not content:
                continue

            container = item.get("container_tag", "")
            is_static = item.get("is_static", False)

            # Static memories are more important (identity traits)
            importance = 0.9 if is_static else 0.5

            meta = item.get("metadata", {}) or {}
            meta["_supermemory_container"] = container
            meta["_supermemory_static"] = is_static

            ts = item.get("timestamp")
            if ts:
                meta["_timestamp"] = ts

            memories.append({
                "content": content,
                "source": item.get("source", "supermemory_import"),
                "importance": importance,
                "metadata": meta,
                "valid_until": None,
                "scope": "session",
                "_author_id": "supermemory_import",
                "_author_type": "system",
                "_channel_id": container or self.container_tag,
                "_timestamp": ts,
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
                result.errors.append("No memories found in SuperMemory")
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
            result.errors.append(f"SuperMemory import failed: {e}")
        result.finished_at = datetime.now().isoformat()
        return result
