"""
Hindsight memory provider importer.

Hindsight is a Hermes memory backend that stores consolidated memories with
historical timestamps and fact types. Unlike regular interactive writes, a
migration must preserve that history, so this importer writes to Mnemosyne's
``episodic_memory`` table directly instead of routing every item through
``remember()`` (which would assign the current timestamp and working-memory
session).

Supported inputs:
- JSON export files containing a list of memories
- JSON objects containing ``items``, ``memories``, ``results``, or ``data``
- A running Hindsight API at ``/v1/default/banks/{bank}/memories/list``
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mnemosyne.core.importers.base import BaseImporter, ImporterResult

logger = logging.getLogger(__name__)

try:  # Optional semantic-search backfill for direct episodic imports.
    from mnemosyne.core import embeddings as _embeddings
    from mnemosyne.core.beam import _vec_available, _vec_insert
    from mnemosyne.core.binary_vectors import maximally_informative_binarization as _mib
except Exception:  # pragma: no cover - importer must still work without embeddings extras
    _embeddings = None
    _vec_available = None
    _vec_insert = None
    _mib = None


class HindsightImporter(BaseImporter):
    """Import Hindsight memories into Mnemosyne episodic memory.

    Examples:
        importer = HindsightImporter(file_path="hindsight-export.json")
        result = importer.run(mnemosyne)

        importer = HindsightImporter(base_url="http://127.0.0.1:8888", bank="hermes")
        result = importer.run(mnemosyne)
    """

    provider_name = "hindsight"

    def __init__(self, file_path: str = None, base_url: str = None,
                 bank: str = "hermes", page_size: int = 500,
                 max_items: int = None, namespace: str = None,
                 skip_low_value: bool = False,
                 generate_embeddings: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.file_path = file_path
        self.base_url = base_url.rstrip("/") if base_url else None
        self.bank = bank
        self.page_size = min(max(int(page_size), 1), 1000)
        self.max_items = max_items
        self.namespace = namespace or bank
        self.skip_low_value = skip_low_value
        self.generate_embeddings = generate_embeddings

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------

    def extract(self) -> List[Dict]:
        """Extract memories from a file or Hindsight HTTP API."""
        if self.file_path:
            return self._extract_from_file(Path(self.file_path))
        if self.base_url:
            return self._extract_from_api()
        raise RuntimeError("Provide either file_path or base_url for Hindsight import")

    def _extract_from_file(self, path: Path) -> List[Dict]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return self._unwrap_items(data)

    def _extract_from_api(self) -> List[Dict]:
        items: List[Dict] = []
        offset = 0
        while True:
            query = urllib.parse.urlencode({"limit": self.page_size, "offset": offset})
            url = f"{self.base_url}/v1/default/banks/{self.bank}/memories/list?{query}"
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            page = self._unwrap_items(data)
            if not page:
                break
            items.extend(page)
            offset += len(page)
            if self.max_items and len(items) >= self.max_items:
                return items[:self.max_items]
            total = data.get("total") if isinstance(data, dict) else None
            if total is not None and offset >= int(total):
                break
        return items

    @staticmethod
    def _unwrap_items(data: Any) -> List[Dict]:
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for key in ("items", "memories", "results", "data"):
                value = data.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
            # Accept a single memory object as a one-item export.
            if any(k in data for k in ("text", "content", "memory")):
                return [data]
        return []

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def transform(self, raw_data: List[Dict]) -> List[Dict]:
        """Normalize Hindsight items into episodic import rows."""
        memories = []
        for item in raw_data:
            content = self._content_for(item)
            if not content:
                continue
            fact_type = item.get("fact_type") or item.get("type") or "memory"
            timestamp = self._timestamp_for(item)
            metadata = self._metadata_for(item)
            quality_flags = self._quality_flags(content)
            if quality_flags:
                metadata["import_quality_flags"] = quality_flags
            metadata["import_quality_score"] = self._quality_score(content, quality_flags)
            if self.skip_low_value and self._is_low_value(content, quality_flags):
                continue
            memories.append({
                "id": self._stable_id(item),
                "content": content,
                "source": f"hindsight:{fact_type}",
                "timestamp": timestamp,
                "session_id": self._session_id_for(item),
                "importance": self._importance_for(item),
                "metadata": metadata,
                "valid_until": item.get("valid_until"),
                "scope": "global",
                "channel_id": "hindsight",
                "author_id": None,
                "author_type": None,
                "veracity": "imported",
                "created_at": timestamp,
            })
        return memories

    @staticmethod
    def _content_for(item: Dict) -> str:
        content = item.get("text") or item.get("content") or item.get("memory") or ""
        return str(content).strip()

    def _stable_id(self, item: Dict) -> str:
        raw = item.get("id") or item.get("uuid") or self._content_for(item) or json.dumps(item, sort_keys=True, default=str)
        digest = hashlib.sha256(f"{self.namespace}:{raw}".encode("utf-8")).hexdigest()[:24]
        return f"hs_{digest}"

    @staticmethod
    def _timestamp_for(item: Dict) -> str:
        # mentioned_at is Hindsight's memory timestamp. date/occurred_* may be
        # event dates and can be in the future, so only use them as fallbacks.
        for key in ("mentioned_at", "timestamp", "created_at", "date", "occurred_start"):
            value = item.get(key)
            if value:
                return str(value)
        return datetime.now(timezone.utc).isoformat()

    def _session_id_for(self, item: Dict) -> str:
        tags = item.get("tags") or []
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("session:"):
                    return tag.replace(":", "_", 1)
        chunk_id = item.get("chunk_id")
        if chunk_id:
            digest = hashlib.sha256(str(chunk_id).encode("utf-8")).hexdigest()[:16]
            return f"chunk_{digest}"
        timestamp = self._timestamp_for(item)[:10] or "unknown-date"
        return "hindsight_" + timestamp.replace("-", "")

    @staticmethod
    def _importance_for(item: Dict) -> float:
        explicit = item.get("importance") or item.get("score")
        if explicit is not None:
            try:
                return max(0.0, min(float(explicit), 1.0))
            except (TypeError, ValueError):
                pass
        fact_type = item.get("fact_type") or item.get("type")
        proof_count = item.get("proof_count") or 0
        try:
            proof_bonus = min(float(proof_count), 5.0) * 0.03
        except (TypeError, ValueError):
            proof_bonus = 0.0
        base = {"world": 0.75, "experience": 0.65, "observation": 0.55}.get(fact_type, 0.5)
        return min(1.0, base + proof_bonus)

    def _metadata_for(self, item: Dict) -> Dict:
        metadata = item.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {"raw_metadata": metadata}
        elif not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        preserved = {
            "migration_source": "hindsight",
            "hindsight_bank": self.bank,
            "hindsight_id": item.get("id"),
            "hindsight_fact_type": item.get("fact_type") or item.get("type"),
            "hindsight_context": item.get("context"),
            "hindsight_date": item.get("date"),
            "hindsight_mentioned_at": item.get("mentioned_at"),
            "hindsight_occurred_start": item.get("occurred_start"),
            "hindsight_occurred_end": item.get("occurred_end"),
            "hindsight_entities": item.get("entities"),
            "hindsight_chunk_id": item.get("chunk_id"),
            "hindsight_proof_count": item.get("proof_count"),
            "hindsight_tags": item.get("tags") or [],
            "hindsight_consolidated_at": item.get("consolidated_at"),
            "hindsight_consolidation_failed_at": item.get("consolidation_failed_at"),
        }
        return {**metadata, **preserved}

    @staticmethod
    def _quality_flags(content: str) -> List[str]:
        """Detect common low-value/polluted memories without hiding provenance."""
        flags: List[str] = []
        text = content.strip()
        lower = text.lower()
        if len(text) < 40:
            flags.append("very_short")
        if re.search(r"\bConversation between\b|\bvia Telegram DM\b|\bon Telegram DM platform\b", text, re.I):
            flags.append("generic_conversation_metadata")
        if re.search(r"Review the conversation above|consider saving.*memory|Has the user revealed|Focus on:", text, re.I | re.S):
            flags.append("meta_memory_prompt")
        if re.search(r"\b[A-Za-z0-9+/]{90,}={0,2}\b", text) or "aaak" in lower:
            flags.append("dense_or_encoded_blob")
        if lower in {"no new narrow skills.", "- no new narrow skills.", "soucrce: convo", "source: convo"}:
            flags.append("explicit_low_value")
        return flags

    @staticmethod
    def _quality_score(content: str, flags: List[str]) -> float:
        score = 1.0
        penalties = {
            "very_short": 0.2,
            "generic_conversation_metadata": 0.35,
            "meta_memory_prompt": 0.5,
            "dense_or_encoded_blob": 0.5,
            "explicit_low_value": 0.8,
        }
        for flag in flags:
            score -= penalties.get(flag, 0.1)
        return max(0.0, round(score, 3))

    @classmethod
    def _is_low_value(cls, content: str, flags: List[str]) -> bool:
        return cls._quality_score(content, flags) <= 0.5

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def run(self, mnemosyne, dry_run: bool = False,
            session_id: str = None, channel_id: str = None) -> ImporterResult:
        """Run import, preserving historical fields in episodic memory."""
        result = ImporterResult(
            provider=self.provider_name,
            started_at=datetime.now().isoformat(),
        )
        try:
            raw_data = self.extract()
            result.total = len(raw_data)
            if result.total == 0:
                result.errors.append("No memories found to import from Hindsight")
                result.finished_at = datetime.now().isoformat()
                return result
            if not self.validate(raw_data):
                result.errors.append("Validation failed")
                result.finished_at = datetime.now().isoformat()
                return result

            memories = self.transform(raw_data)
            result.skipped = result.total - len(memories)
            if dry_run:
                result.imported = len(memories)
                result.finished_at = datetime.now().isoformat()
                return result

            conn = mnemosyne.beam.conn
            for mem in memories:
                try:
                    if session_id:
                        mem["session_id"] = session_id
                    if channel_id:
                        mem["channel_id"] = channel_id
                    inserted = self._insert_episodic(
                        conn,
                        mem,
                        generate_embeddings=self.generate_embeddings,
                    )
                    if inserted:
                        result.imported += 1
                        result.memory_ids.append(mem["id"])
                    else:
                        result.skipped += 1
                except Exception as e:
                    result.failed += 1
                    result.errors.append(
                        f"Failed to import '{mem.get('content', '')[:80]}': {e}"
                    )
            conn.commit()

            # --- MEMORIA backfill: extract structured facts from imported content ---
            # Hindsight imports write directly to episodic_memory, bypassing
            # remember() and therefore the regex-based MEMORIA extraction that
            # populates memoria_facts / memoria_timelines / memoria_kg.
            # Without this step, the structured retrieval router sees zero
            # facts for all imported memories and v3's key feature is useless.
            if result.imported > 0:
                logger.info("MEMORIA backfill: extracting facts from %d imported memories",
                            result.imported)
                try:
                    beam = mnemosyne.beam
                    for mem in memories:
                        try:
                            beam.extract_and_store_facts(mem["content"], message_idx=0)
                        except Exception:
                            pass  # Best-effort per memory
                    conn.commit()
                    logger.info("MEMORIA backfill: complete")
                except Exception as e:
                    logger.warning("MEMORIA backfill: failed (%s) — structured router "
                                   "will see 0 facts for imported memories", e)
        except Exception as e:
            result.errors.append(f"Hindsight import failed: {e}")
        result.finished_at = datetime.now().isoformat()
        return result

    @staticmethod
    def _insert_episodic(conn, mem: Dict, generate_embeddings: bool = True) -> bool:
        metadata_json = json.dumps(mem.get("metadata", {}), ensure_ascii=False, sort_keys=True, default=str)
        cur = conn.execute("""
            INSERT OR IGNORE INTO episodic_memory
            (id, content, source, timestamp, session_id, importance, metadata_json,
             summary_of, veracity, created_at, degraded_at, valid_until,
             channel_id, author_id, scope, superseded_by, author_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            mem["id"],
            mem["content"],
            mem.get("source", "hindsight:memory"),
            mem.get("timestamp"),
            mem.get("session_id", "hindsight"),
            mem.get("importance", 0.5),
            metadata_json,
            "",
            mem.get("veracity", "imported"),
            mem.get("created_at") or mem.get("timestamp"),
            None,
            mem.get("valid_until"),
            mem.get("channel_id"),
            mem.get("author_id"),
            mem.get("scope", "global"),
            None,
            mem.get("author_type"),
        ))
        inserted = cur.rowcount > 0
        if inserted and generate_embeddings:
            HindsightImporter._backfill_import_embedding(conn, cur.lastrowid, mem["content"])
        return inserted

    @staticmethod
    def _backfill_import_embedding(conn, rowid: int, content: str) -> None:
        """Best-effort vector generation for direct episodic imports."""
        if not (_embeddings and _embeddings.available()):
            logger.debug("backfill: embeddings unavailable, skipping vector for rowid=%s", rowid)
            return
        vecs = _embeddings.embed([content])
        if vecs is None:
            logger.debug("backfill: embed() returned None for rowid=%s, content=%.50r", rowid, content)
            return
        vec = vecs[0]
        memory_row = conn.execute(
            "SELECT id FROM episodic_memory WHERE rowid = ?",
            (rowid,),
        ).fetchone()
        if memory_row is not None:
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json, model)
                    VALUES (?, ?, ?)
                    """,
                    (memory_row["id"], json.dumps(vec.tolist()), getattr(_embeddings, "MODEL_NAME", "import_backfill")),
                )
            except Exception as exc:
                logger.debug("backfill: memory_embeddings INSERT failed for rowid=%s: %s", rowid, exc)
        if _vec_available and _vec_insert and _vec_available(conn):
            _vec_insert(conn, rowid, vec.tolist())
        if _mib is not None:
            try:
                conn.execute(
                    "UPDATE episodic_memory SET binary_vector = ? WHERE rowid = ?",
                    (_mib(vec), rowid),
                )
            except Exception as exc:
                logger.debug("backfill: binary_vector UPDATE failed for rowid=%s: %s", rowid, exc)


def import_from_hindsight(mnemosyne, file_path: str = None, base_url: str = None,
                          bank: str = "hermes", dry_run: bool = False,
                          session_id: str = None, channel_id: str = None,
                          max_items: int = None,
                          skip_low_value: bool = False,
                          generate_embeddings: bool = True) -> ImporterResult:
    """Convenience wrapper for importing Hindsight memories."""
    importer = HindsightImporter(
        file_path=file_path,
        base_url=base_url,
        bank=bank,
        max_items=max_items,
        skip_low_value=skip_low_value,
        generate_embeddings=generate_embeddings,
    )
    return importer.run(
        mnemosyne,
        dry_run=dry_run,
        session_id=session_id,
        channel_id=channel_id,
    )
