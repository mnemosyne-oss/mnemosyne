"""
Cognee memory provider importer.

Cognee uses a triple-store architecture: Kùzu graph + LanceDB vectors + SQLite.
Extraction via get_graph_data() returns (nodes, edges) tuples.
Nodes → episodic memories, edges → triples.

Extraction methods:
1. Python SDK: cognee.graph_db.get_graph_data()
2. REST API: GET /datasets/{id}/data
3. Direct addition via add_data_points()
"""

import json
from datetime import datetime
from typing import List, Dict

from mnemosyne.core.importers.base import BaseImporter, ImporterResult


class CogneeImporter(BaseImporter):
    """Import memories from Cognee into Mnemosyne.

    Usage:
        importer = CogneeImporter(
            dataset_id="my-dataset",   # optional: filter by dataset
            data_dir="./.cognee-data", # path to Cognee data directory
            direct_db=True,            # if True, read Kùzu/LanceDB directly
        )
        result = importer.run(mnemosyne_instance)
    """

    provider_name = "cognee"

    def __init__(self, dataset_id: str = None, data_dir: str = None,
                 direct_db: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.dataset_id = dataset_id
        self.data_dir = data_dir
        self.direct_db = direct_db

    def extract(self) -> List[Dict]:
        """Extract memories from Cognee."""
        if self.direct_db and self.data_dir:
            return self._extract_direct()
        try:
            return self._extract_via_sdk()
        except (ImportError, Exception):
            pass
        try:
            return self._extract_via_rest()
        except Exception:
            pass
        raise RuntimeError(
            "Could not extract from Cognee. Install: pip install cognee"
        )

    def _extract_via_sdk(self) -> List[Dict]:
        """Extract using Cognee Python SDK."""
        import cognee
        import asyncio

        async def _extract():
            # Use cognee's adapter to get graph data
            graph_data = await cognee.graph_db.get_graph_data()
            return self._parse_graph_data(graph_data)

        return asyncio.run(_extract())

    def _extract_via_rest(self) -> List[Dict]:
        """Extract using Cognee REST API."""
        import urllib.request

        base = "http://localhost:8000/api/v1"
        if self.dataset_id:
            url = f"{base}/datasets/{self.dataset_id}/data"
        else:
            url = f"{base}/datasets/data"

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return self._parse_api_data(data)

    def _extract_direct(self) -> List[Dict]:
        """Directly read Cognee's file-based stores."""
        from pathlib import Path
        data_dir = Path(self.data_dir or "./.cognee-data")
        items = []

        # Try reading SQLite metadata
        sqlite_path = data_dir / "cognee_db"
        if sqlite_path.exists():
            import sqlite3
            conn = sqlite3.connect(str(sqlite_path))
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT * FROM data_chunks ORDER BY created_at"
                ).fetchall()
                for raw_row in rows:
                    # sqlite3.Row supports bracket access but not .get();
                    # convert to dict so the column-with-default reads
                    # below work. Pre-fix the .get() calls raised
                    # AttributeError and the broad `except` below swallowed
                    # it silently, returning [] for every direct cognee
                    # import even when data_chunks was populated. Same
                    # pattern as the fact_recall fix in C12.a.
                    row = dict(raw_row)
                    items.append({
                        "content": row.get("text") or row.get("content") or "",
                        "source": "cognee_direct",
                        "metadata": {
                            "chunk_id": row.get("id") or "",
                            "document_id": row.get("document_id") or "",
                        },
                        "timestamp": row.get("created_at"),
                    })
            except Exception:
                pass
            finally:
                conn.close()

        return items

    def _parse_graph_data(self, graph_data) -> List[Dict]:
        """Parse Cognee graph (nodes, edges) into memory dicts."""
        if isinstance(graph_data, tuple) and len(graph_data) == 2:
            nodes, edges = graph_data
        else:
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])

        items = []

        # Nodes → episodic-like memories
        for node in nodes:
            if isinstance(node, tuple) and len(node) >= 2:
                node_id, props = node[0], node[1]
            else:
                node_id = node.get("id", node.get("node_id", ""))
                props = node.get("properties", node)

            content_parts = []
            if isinstance(props, dict):
                for k, v in props.items():
                    if k not in ("id", "node_id", "embedding"):
                        content_parts.append(f"{k}: {v}")
            content = "; ".join(content_parts)
            if content:
                items.append({
                    "content": content,
                    "source": "cognee_node",
                    "node_id": str(node_id),
                    "type": "node",
                    "metadata": props if isinstance(props, dict) else {},
                })

        # Edges → triples (subject→predicate→object)
        for edge in edges:
            if isinstance(edge, tuple) and len(edge) >= 4:
                src, tgt, rel, props = edge[0], edge[1], edge[2], edge[3] if len(edge) > 3 else {}
            else:
                src = edge.get("source", edge.get("source_node_id", ""))
                tgt = edge.get("target", edge.get("target_node_id", ""))
                rel = edge.get("relationship", edge.get("label", ""))
                props = edge.get("properties", {})

            fact = f"{src} {rel} {tgt}"
            items.append({
                "content": fact,
                "source": "cognee_edge",
                "type": "edge",
                "metadata": {
                    "source_node": str(src),
                    "target_node": str(tgt),
                    "relationship": str(rel),
                    ** (props if isinstance(props, dict) else {}),
                },
            })

        return items

    def _parse_api_data(self, data) -> List[Dict]:
        """Parse Cognee REST API response."""
        if isinstance(data, list):
            items = data
        else:
            items = data.get("data", data.get("items", data.get("results", [])))

        result = []
        for item in items:
            content = item.get("content", item.get("text", item.get("name", "")))
            if content:
                result.append({
                    "content": content,
                    "source": "cognee_api",
                    "metadata": item.get("metadata", {}),
                    "timestamp": item.get("created_at"),
                })
        return result

    def transform(self, raw_data: List[Dict]) -> List[Dict]:
        """Transform Cognee data to Mnemosyne format."""
        memories = []
        for item in raw_data:
            content = item.get("content", "")
            if not content:
                continue

            item_type = item.get("type", "")
            if item_type == "edge":
                source = "cognee_triple"
                importance = 0.6
            elif item_type == "node":
                source = "cognee_node"
                importance = 0.5
            else:
                source = "cognee_import"
                importance = 0.5

            meta = item.get("metadata", {}) or {}
            if item.get("node_id"):
                meta["_cognee_node_id"] = item["node_id"]

            memories.append({
                "content": content,
                "source": source,
                "importance": importance,
                "metadata": meta,
                "valid_until": None,
                "scope": "session",
                "_author_id": "cognee_system",
                "_author_type": "system",
                "_channel_id": self.dataset_id,
                "_timestamp": item.get("timestamp"),
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
                result.errors.append("No memories found in Cognee")
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
            result.errors.append(f"Cognee import failed: {e}")
        result.finished_at = datetime.now().isoformat()
        return result
