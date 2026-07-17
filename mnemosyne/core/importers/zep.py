"""
Zep memory provider importer.

Zep is a cloud-hosted enterprise memory platform with a temporal
knowledge graph (Neo4j-based). Extraction requires iterating:
users → sessions/threads → memory.get() per session.

Zep has NO bulk export API. Data must be extracted per-session.

Extraction method:
1. List all users via client.user.list_ordered()
2. For each user: list sessions
3. For each session: client.memory.get(session_id) → messages + facts + summary
"""

import json
from datetime import datetime
from typing import List, Dict

from mnemosyne.core.importers.base import BaseImporter, ImporterResult


class ZepImporter(BaseImporter):
    """Import memories from Zep into Mnemosyne.

    Usage:
        importer = ZepImporter(
            api_key="sk-xxx",         # Zep API key
            user_id="alice",          # optional: filter by user
            base_url=None,            # for self-hosted (rare)
            max_sessions=None,        # limit sessions to extract
        )
        result = importer.run(mnemosyne_instance)
    """

    provider_name = "zep"

    def __init__(self, api_key: str = None, user_id: str = None,
                 base_url: str = None, max_sessions: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.user_id = user_id
        self.base_url = base_url
        self.max_sessions = max_sessions

    def extract(self) -> List[Dict]:
        """Extract memories from Zep via session-by-session iteration."""
        try:
            return self._extract_via_sdk()
        except ImportError:
            pass
        try:
            return self._extract_via_rest()
        except Exception:
            pass
        raise RuntimeError(
            "Could not extract from Zep. Install the SDK: pip install zep-cloud"
        )

    def _extract_via_sdk(self) -> List[Dict]:
        """Extract using Zep Python SDK."""
        from zep_cloud.client import Zep
        client = Zep(api_key=self.api_key)

        all_memories = []

        # Step 1: List users
        users = []
        if self.user_id:
            users = [self.user_id]
        else:
            page = 1
            while True:
                resp = client.user.list_ordered(page_size=50, page_number=page)
                batch = resp.get("users", resp.get("results", []))
                if not batch:
                    break
                users.extend(u.get("user_id", u.get("id", "")) for u in batch)
                if resp.get("next") is None:
                    break
                page += 1

        # Step 2: For each user, list sessions and extract
        session_count = 0
        for uid in users:
            if self.max_sessions and session_count >= self.max_sessions:
                break

            try:
                sessions = client.user.get_sessions(uid)
            except Exception:
                sessions = []

            for session in sessions:
                if self.max_sessions and session_count >= self.max_sessions:
                    break

                sid = session.get("session_id", session.get("uuid", ""))
                if not sid:
                    continue

                try:
                    mem = client.memory.get(sid)
                    all_memories.extend(
                        self._parse_session_data(mem, uid, sid)
                    )
                    session_count += 1
                except Exception:
                    continue

        return all_memories

    def _extract_via_rest(self) -> List[Dict]:
        """Extract using Zep REST API directly."""
        import urllib.request

        base = self.base_url or "https://api.getzep.com"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        all_memories = []

        # List users
        users = []
        if self.user_id:
            users = [self.user_id]
        else:
            req = urllib.request.Request(
                f"{base}/api/v2/users?page_size=50",
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                users = [u["user_id"] for u in data.get("users", [])]

        for uid in users:
            req = urllib.request.Request(
                f"{base}/api/v2/users/{uid}/sessions",
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                sessions = json.loads(resp.read().decode())

            for session in sessions:
                sid = session.get("session_id", "")
                if not sid:
                    continue
                req = urllib.request.Request(
                    f"{base}/api/v2/sessions/{sid}/memory",
                    headers=headers,
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    mem = json.loads(resp.read().decode())
                    all_memories.extend(
                        self._parse_session_data(mem, uid, sid)
                    )

        return all_memories

    def _parse_session_data(self, mem: dict, user_id: str,
                            session_id: str) -> List[Dict]:
        """Parse a single Zep session's memory data."""
        items = []

        # 1. Messages
        messages = mem.get("messages", [])
        for msg in messages:
            content = msg.get("content", msg.get("text", ""))
            if not content:
                continue
            role = msg.get("role_type", msg.get("role", "user"))
            ts = msg.get("created_at")
            items.append({
                "content": content,
                "source": "zep_message",
                "user_id": user_id,
                "session_id": session_id,
                "role": role,
                "timestamp": ts,
                "metadata": msg.get("metadata", {}),
            })

        # 2. Summary
        summary = mem.get("summary", "")
        if summary:
            items.append({
                "content": summary,
                "source": "zep_summary",
                "user_id": user_id,
                "session_id": session_id,
                "role": "system",
                "timestamp": None,
                "metadata": {"zep_type": "summary"},
            })

        # 3. Relevant facts (edges)
        facts = mem.get("relevant_facts", mem.get("facts", []))
        for fact in facts:
            fact_text = fact.get("fact", fact.get("text", ""))
            if not fact_text:
                continue
            items.append({
                "content": fact_text,
                "source": "zep_fact",
                "user_id": user_id,
                "session_id": session_id,
                "role": "system",
                "timestamp": fact.get("created_at"),
                "metadata": {
                    "zep_type": "fact",
                    "zep_rating": fact.get("rating"),
                },
            })

        return items

    def transform(self, raw_data: List[Dict]) -> List[Dict]:
        """Transform Zep data to Mnemosyne format."""
        memories = []
        for item in raw_data:
            content = item.get("content", "")
            if not content:
                continue

            # Author: use Zep user_id
            uid = item.get("user_id", "")
            author_id = f"zep_user:{uid}" if uid else None

            # Importance heuristic
            source = item.get("source", "zep_import")
            if source == "zep_summary":
                importance = 0.8
            elif source == "zep_fact":
                importance = 0.7
            else:
                importance = 0.4

            # Build metadata
            meta = item.get("metadata", {}) or {}
            meta["_zep_session_id"] = item.get("session_id", "")
            meta["_zep_role"] = item.get("role", "")

            ts = item.get("timestamp")
            if ts:
                meta["_timestamp"] = ts

            memories.append({
                "content": content,
                "source": source,
                "importance": importance,
                "metadata": meta,
                "valid_until": None,
                "scope": "session",
                "_author_id": author_id,
                "_author_type": "human" if uid else "system",
                "_channel_id": uid,  # user_id as channel for grouping
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
                result.errors.append("No memories found in Zep")
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
            result.errors.append(f"Zep import failed: {e}")

        result.finished_at = datetime.now().isoformat()
        return result
