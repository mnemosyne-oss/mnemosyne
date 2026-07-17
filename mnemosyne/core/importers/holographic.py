"""
Hermes Holographic Memory importer.

Hermes Holographic Memory is a local SQLite-backed fact store plugin for
Hermes Agent. It uses FTS5 + HRR (Holographic Reduced Representations) for
compositional queries, with entity linking, trust scoring, and category-based
memory banks.

Schema lives in plugins/memory/holographic/store.py and defaults to
~/.hermes/memory_store.db.

Usage:
    from mnemosyne.core.importers.holographic import HolographicImporter

    importer = HolographicImporter(db_path="~/.hermes/memory_store.db")
    result = importer.run(mnemosyne_instance)

CLI:
    hermes mnemosyne import --from holographic [--db-path ~/.hermes/memory_store.db]
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

from mnemosyne.core.importers.base import BaseImporter, ImporterResult


class HolographicImporter(BaseImporter):
    """Import memories from Hermes Holographic Memory store into Mnemosyne.

    Preserves content, categories, tags, trust scores, timestamps, and
    entity links. HRR vectors are not imported (Mnemosyne uses its own
    embedding/vector search).
    """

    provider_name = "holographic"

    # Entity extraction regexes matching the source store.py
    _RE_CAPITALIZED = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
    _RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
    _RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")
    _RE_AKA = re.compile(
        r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)',
        re.IGNORECASE,
    )

    def __init__(
        self,
        db_path: Optional[str] = None,
        min_trust: float = 0.0,
        category_filter: Optional[str] = None,
        extract_entities: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.db_path = db_path or str(Path.home() / ".hermes" / "memory_store.db")
        self.min_trust = min_trust
        self.category_filter = category_filter
        self.extract_entities = extract_entities

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------

    def extract(self) -> List[Dict]:
        """Extract all facts (with linked entities) from Holographic DB."""
        db_path = str(Path(self.db_path).expanduser())
        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"Holographic memory store not found at: {db_path}\n"
                "Specify --db-path or copy the file to the default location."
            )

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        try:
            # Check if the DB has the holographic schema
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "facts" not in tables:
                raise RuntimeError(
                    f"Database at {db_path} does not contain expected 'facts' table. "
                    "Is this a Holographic memory store?"
                )

            # Build query with optional filters
            where_clauses = ["f.trust_score >= ?"]
            params: list = [self.min_trust]

            if self.category_filter:
                where_clauses.append("f.category = ?")
                params.append(self.category_filter)

            where_sql = " AND ".join(where_clauses)

            # Facts with concatenated entity names
            rows = conn.execute(
                f"""
                SELECT
                    f.fact_id,
                    f.content,
                    f.category,
                    f.tags,
                    f.trust_score,
                    f.retrieval_count,
                    f.helpful_count,
                    f.created_at,
                    f.updated_at,
                    GROUP_CONCAT(DISTINCT e.name) AS entities,
                    GROUP_CONCAT(DISTINCT e.entity_type) AS entity_types
                FROM facts f
                LEFT JOIN fact_entities fe ON f.fact_id = fe.fact_id
                LEFT JOIN entities e ON fe.entity_id = e.entity_id
                WHERE {where_sql}
                GROUP BY f.fact_id
                ORDER BY f.fact_id
                """,
                params,
            ).fetchall()

            results = []
            for row in rows:
                item = dict(row)
                # Parse concatenated entities into a list
                entities_str = item.pop("entities", None)
                entity_types_str = item.pop("entity_types", None)
                item["entity_names"] = (
                    [e for e in entities_str.split("|") if e]
                    if entities_str
                    else []
                )
                item["entity_types_list"] = (
                    [t for t in entity_types_str.split("|") if t]
                    if entity_types_str
                    else []
                )
                results.append(item)

            return results

        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate(self, raw_data: List[Dict]) -> bool:
        if not raw_data:
            return False
        # At minimum each item should have content
        for item in raw_data:
            if not isinstance(item.get("content"), str) or not item["content"].strip():
                return False
        return True

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, raw_data: List[Dict]) -> List[Dict]:
        """Transform Holographic facts to Mnemosyne-compatible dicts.

        Mapping:
            content       → content (preserved verbatim)
            category      → metadata._holographic_category
            tags          → metadata._holographic_tags
            trust_score   → importance (direct 0-1 mapping)
            created_at    → metadata._created_at (preserved timestamp)
            updated_at    → metadata._updated_at
            retrieval_count → metadata._retrieval_count
            helpful_count → metadata._helpful_count
            entity_names  → metadata._entities (list)
        """
        memories = []

        for item in raw_data:
            content = item.get("content", "").strip()
            if not content:
                continue

            # Trust score → importance (both 0-1, trust is the source's signal)
            trust = float(item.get("trust_score", 0.5))
            importance = max(0.0, min(1.0, trust))

            # Build metadata preserving all source fields
            meta: Dict[str, Any] = {
                "_holographic_category": item.get("category", "general"),
                "_holographic_trust_score": trust,
                "_holographic_retrieval_count": item.get("retrieval_count", 0),
                "_holographic_helpful_count": item.get("helpful_count", 0),
            }

            tags = item.get("tags", "").strip()
            if tags:
                meta["_holographic_tags"] = tags

            created = item.get("created_at")
            if created:
                meta["_created_at"] = str(created)

            updated = item.get("updated_at")
            if updated:
                meta["_updated_at"] = str(updated)

            entities = item.get("entity_names", [])
            if entities:
                meta["_entities"] = entities

            memories.append({
                "content": content,
                "source": "holographic_import",
                "importance": importance,
                "metadata": meta,
                "valid_until": None,
                "scope": "session",
            })

        return memories

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, mnemosyne, dry_run=False, session_id=None, channel_id=None):
        result = ImporterResult(
            provider=self.provider_name,
            started_at=datetime.now().isoformat(),
        )

        try:
            raw_data = self.extract()
            result.total = len(raw_data)

            if result.total == 0:
                result.errors.append("No memories found in Holographic store")
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
                    if session_id:
                        mem_dict["session_id"] = session_id
                    if channel_id:
                        mem_dict["channel_id"] = channel_id

                    mid = mnemosyne.remember(
                        content=mem_dict["content"],
                        source=mem_dict.get("source", self.provider_name),
                        importance=mem_dict.get("importance", 0.5),
                        metadata=mem_dict.get("metadata", {}),
                        valid_until=mem_dict.get("valid_until"),
                        scope=mem_dict.get("scope", "session"),
                        extract_entities=self.extract_entities,
                    )
                    result.memory_ids.append(mid)
                    result.imported += 1

                except Exception as e:
                    result.failed += 1
                    result.errors.append(
                        f"Failed to import '{mem_dict.get('content', '')[:80]}': {e}"
                    )

        except Exception as e:
            result.errors.append(f"Holographic import failed: {e}")

        result.finished_at = datetime.now().isoformat()
        return result


def import_from_holographic(mnemosyne, db_path: str = None, **kwargs) -> ImporterResult:
    """Convenience function for importing from Holographic memory store.

    Args:
        mnemosyne: A Mnemosyne instance.
        db_path: Path to Holographic memory_store.db. Defaults to ~/.hermes/memory_store.db.
        **kwargs: Passed to HolographicImporter (min_trust, category_filter, etc.).

    Returns:
        ImporterResult with statistics.
    """
    importer = HolographicImporter(db_path=db_path, **kwargs)
    return importer.run(mnemosyne)
