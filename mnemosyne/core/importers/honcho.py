"""
Honcho memory provider importer.

Honcho (by Plastic Labs) uses an entity-centric model:
Workspaces → Peers → Sessions → Messages.

Extraction requires:
1. List peers via SDK
2. For each peer: list sessions → get context() with summaries
3. For each session: list messages

Honcho uses PostgreSQL with pgvector. No bulk export.
"""

import json
from datetime import datetime
from typing import List, Dict

from mnemosyne.core.importers.base import BaseImporter, ImporterResult


class HonchoImporter(BaseImporter):
    """Import memories from Honcho into Mnemosyne.

    Usage:
        importer = HonchoImporter(
            api_key="sk-xxx",          # Honcho API key
            workspace_id="my-app",     # required
            max_peers=50,              # limit peers to extract
        )
        result = importer.run(mnemosyne_instance)
    """

    provider_name = "honcho"

    def __init__(self, api_key: str = None, workspace_id: str = None,
                 max_peers: int = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.workspace_id = workspace_id or "default"
        self.max_peers = max_peers

    def extract(self) -> List[Dict]:
        """Extract memories from Honcho."""
        try:
            return self._extract_via_sdk()
        except ImportError:
            pass
        try:
            return self._extract_via_rest()
        except Exception:
            pass
        raise RuntimeError(
            "Could not extract from Honcho. Install: pip install honcho-ai"
        )

    def _extract_via_sdk(self) -> List[Dict]:
        """Extract using Honcho Python SDK."""
        from honcho import Honcho

        honcho = Honcho(workspace_id=self.workspace_id)
        if self.api_key:
            honcho.api_key = self.api_key

        all_items = []

        # Step 1: List peers
        try:
            peers_resp = honcho.list_peers()
            peers = peers_resp.get("peers", peers_resp.get("items", []))
        except Exception:
            peers = []

        peer_count = 0
        for peer in peers:
            if self.max_peers and peer_count >= self.max_peers:
                break

            peer_id = peer.get("peer_id", peer.get("id", ""))
            peer_name = peer.get("name", peer_id)

            # Step 2: List sessions for this peer
            try:
                sessions_resp = honcho.list_sessions(peer_id=peer_id)
                sessions = sessions_resp.get("sessions", sessions_resp.get("items", []))
            except Exception:
                sessions = []

            for session in sessions:
                sid = session.get("session_id", session.get("id", ""))
                if not sid:
                    continue

                # Step 3: Get context (includes summary)
                try:
                    ctx = honcho.session(sid).context(summary=True)
                    if isinstance(ctx, dict):
                        summary = ctx.get("summary", ctx.get("context", ""))
                        if summary:
                            all_items.append({
                                "content": str(summary),
                                "source": "honcho_summary",
                                "peer_id": peer_id,
                                "peer_name": peer_name,
                                "session_id": sid,
                                "role": "system",
                            })
                except Exception:
                    pass

                # Step 4: Get messages
                try:
                    messages_resp = honcho.session(sid).list_messages()
                    messages = messages_resp.get("messages", messages_resp.get("items", []))
                    for msg in messages:
                        content = msg.get("content", msg.get("text", ""))
                        if not content:
                            continue
                        all_items.append({
                            "content": content,
                            "source": "honcho_message",
                            "peer_id": msg.get("peer_id", peer_id),
                            "peer_name": peer_name,
                            "session_id": sid,
                            "role": "user",
                            "timestamp": msg.get("created_at"),
                            "metadata": msg.get("metadata", {}),
                        })
                except Exception:
                    pass

            peer_count += 1

        return all_items

    def _extract_via_rest(self) -> List[Dict]:
        """Extract using Honcho REST API."""
        import urllib.request

        # Honcho is primarily SDK-based; REST extraction is similar
        # but depends on server configuration
        base = "http://localhost:8000"
        all_items = []

        # Try listing peers
        req = urllib.request.Request(f"{base}/peers")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                peers = json.loads(resp.read().decode())
        except Exception:
            peers = []

        for peer in peers:
            peer_id = peer.get("peer_id", "")
            # Try to get sessions
            req = urllib.request.Request(f"{base}/peers/{peer_id}/sessions")
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    sessions = json.loads(resp.read().decode())
            except Exception:
                sessions = []

            for session in sessions:
                sid = session.get("session_id", "")
                req = urllib.request.Request(f"{base}/sessions/{sid}/messages")
                try:
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        messages = json.loads(resp.read().decode())
                        for msg in messages:
                            content = msg.get("content", "")
                            if content:
                                all_items.append({
                                    "content": content,
                                    "source": "honcho_message",
                                    "peer_id": peer_id,
                                    "session_id": sid,
                                    "role": "user",
                                    "timestamp": msg.get("created_at"),
                                })
                except Exception:
                    pass

        return all_items

    def transform(self, raw_data: List[Dict]) -> List[Dict]:
        """Transform Honcho data to Mnemosyne format."""
        memories = []
        for item in raw_data:
            content = item.get("content", "")
            if not content:
                continue

            peer_id = item.get("peer_id", "")
            peer_name = item.get("peer_name", peer_id)
            source = item.get("source", "honcho_import")

            # Importance heuristic
            if source == "honcho_summary":
                importance = 0.7
            else:
                importance = 0.4

            meta = item.get("metadata", {}) or {}
            meta["_honcho_session_id"] = item.get("session_id", "")
            meta["_honcho_peer_name"] = peer_name

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
                "_author_id": f"honcho_peer:{peer_id}" if peer_id else None,
                "_author_type": "human",
                "_channel_id": self.workspace_id,
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
                result.errors.append("No memories found in Honcho")
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
            result.errors.append(f"Honcho import failed: {e}")
        result.finished_at = datetime.now().isoformat()
        return result
