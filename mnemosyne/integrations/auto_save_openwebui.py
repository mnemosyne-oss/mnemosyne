"""
Mnemosyne — OpenWebUI Auto Save Service
===========================================
Automatically saves every OpenWebUI chat message to Mnemosyne.

Runs as a lightweight daemon that polls OpenWebUI's REST API
for new messages and stores them in Mnemosyne for persistent,
cross-session recall.

Usage:
    # Start the auto-save service
    python -m mnemosyne.integrations.auto_save_openwebui \\
        --openwebui-url http://localhost:3000 \\
        --api-key your-openwebui-api-key

    # Run once (one-shot sync)
    python -m mnemosyne.integrations.auto_save_openwebui \\
        --openwebui-url http://localhost:3000 \\
        --api-key your-openwebui-api-key --once

    # With custom Mnemosyne data dir
    export MNEMOSYNE_DATA_DIR=/path/to/data
    python -m mnemosyne.integrations.auto_save_openwebui --api-key ...

Configuration via environment variables:
    OPENWEBUI_URL       - OpenWebUI base URL (default: http://localhost:3000)
    OPENWEBUI_API_KEY   - OpenWebUI API key
    MNEMOSYNE_DATA_DIR  - Mnemosyne data directory
    AUTO_SAVE_INTERVAL  - Polling interval in seconds (default: 60)
    AUTO_SAVE_BANK      - Memory bank name (default: openwebui)

Dependencies: None beyond mnemosyne-memory itself (uses stdlib).
"""

import argparse
import asyncio
import json
import logging
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from mnemosyne.core.memory import Mnemosyne

logger = logging.getLogger("mnemosyne-auto-save")

# ── Defaults ──────────────────────────────────────────────────────────

DEFAULT_INTERVAL = 60  # seconds
DEFAULT_BANK = "openwebui"
DEFAULT_OPENWEBUI_URL = "http://localhost:3000"

# Track processed message IDs to avoid duplicates
_PROCESSED_IDS: set[str] = set()
_PROCESSED_FILE = Path.home() / ".mnemosyne" / "openwebui_saved_ids.json"


def _load_processed_ids():
    """Load previously saved message IDs from disk."""
    global _PROCESSED_IDS
    try:
        if _PROCESSED_FILE.exists():
            data = json.loads(_PROCESSED_FILE.read_text())
            _PROCESSED_IDS = set(data.get("saved_ids", []))
            logger.info("Loaded %d previously saved message IDs", len(_PROCESSED_IDS))
    except Exception as e:
        logger.warning("Could not load processed IDs: %s", e)


def _save_processed_ids():
    """Persist processed message IDs to disk (keep last 100K)."""
    try:
        _PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PROCESSED_FILE.write_text(json.dumps({"saved_ids": list(_PROCESSED_IDS)[-100000:]}))
    except Exception as e:
        logger.warning("Could not save processed IDs: %s", e)


# ── Sync Helpers ─────────────────────────────────────────────────────


def _api_get(url: str, api_key: str = "") -> Any:
    """Make a GET request to the OpenWebUI API using stdlib."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


async def _api_get_async(url: str, api_key: str = "") -> Any:
    """Async wrapper around _api_get using run_in_executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _api_get, url, api_key)


async def _health_check(url: str, api_key: str = "") -> bool:
    """Check if OpenWebUI is reachable."""
    try:
        await _api_get_async(f"{url}/api/health", api_key)
        return True
    except Exception:
        return False


# ── Memory Saver ──────────────────────────────────────────────────────


class MemorySaver:
    """Saves OpenWebUI messages to Mnemosyne."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        bank: str = DEFAULT_BANK,
    ):
        db_path = data_dir or os.environ.get(
            "MNEMOSYNE_DATA_DIR",
            str(Path.home() / ".hermes" / "mnemosyne" / "data"),
        )
        db_dir = Path(db_path)
        db_dir.mkdir(parents=True, exist_ok=True)
        self._memory = Mnemosyne(
            session_id=bank,
            bank=bank,
            db_path=Path(str(db_dir / f"{bank}.db")),
        )
        self._bank = bank
        self._stats = {"saved": 0, "skipped": 0, "errors": 0}

    def save_message(
        self,
        message: Dict[str, Any],
        chat_title: str = "",
        chat_id: str = "",
        user_id: str = "",
    ) -> bool:
        """
        Save a single message to Mnemosyne.

        Returns True if saved, False if skipped (duplicate).
        """
        # Build a unique ID from message + chat
        msg_id = str(message.get("id", message.get("message_id", id(message))))
        unique_id = f"{chat_id}:{msg_id}"

        if unique_id in _PROCESSED_IDS:
            self._stats["skipped"] += 1
            return False

        # Extract content
        content = str(message.get("content", message.get("text", "")))
        if not content.strip():
            self._stats["skipped"] += 1
            return False

        role = str(message.get("role", "user"))

        # Extract timestamp
        ts = message.get("timestamp") or message.get("created_at") or int(time.time())
        if isinstance(ts, (int, float)) and ts > 10_000_000_000:
            ts = ts / 1000
        timestamp = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        # Build metadata
        metadata = {
            "chat_id": chat_id,
            "chat_title": chat_title,
            "role": role,
            "message_id": msg_id,
            "platform": "openwebui",
            "user_id": user_id,
            "saved_at": timestamp,
        }

        try:
            self._memory.remember(
                content=content,
                source=f"openwebui:{role}",
                importance=0.5 if role == "assistant" else 0.6,
                metadata=metadata,
            )
            _PROCESSED_IDS.add(unique_id)
            self._stats["saved"] += 1
            return True
        except Exception as e:
            logger.error("Failed to save message %s: %s", unique_id, e)
            self._stats["errors"] += 1
            return False

    def get_stats(self) -> Dict[str, Any]:
        return {**self._stats, "bank": self._bank}

    def persist(self):
        _save_processed_ids()


# ── Sync Engine ───────────────────────────────────────────────────────


class SyncEngine:
    """Syncs OpenWebUI chats to Mnemosyne."""

    def __init__(
        self,
        openwebui_url: str = DEFAULT_OPENWEBUI_URL,
        api_key: str = "",
        data_dir: Optional[str] = None,
        bank: str = DEFAULT_BANK,
        interval: int = DEFAULT_INTERVAL,
    ):
        self.url = openwebui_url.rstrip("/")
        self.api_key = api_key
        self.saver = MemorySaver(data_dir, bank)
        self.interval = interval
        self._running = False

    async def sync_once(self) -> Dict[str, Any]:
        """Run one sync cycle. Returns stats."""
        page = 1
        total_saved = 0

        while True:
            try:
                chats = await _api_get_async(
                    f"{self.url}/api/chats/list?page={page}",
                    self.api_key,
                )
            except Exception as e:
                logger.warning("Failed to fetch chat list (page %d): %s", page, e)
                break

            if not chats:
                break

            for chat in chats if isinstance(chats, list) else (chats or []):
                chat_id = chat.get("id") if isinstance(chat, dict) else None
                chat_title = (chat.get("title", "Untitled") if isinstance(chat, dict) else "Untitled")
                if not chat_id:
                    continue

                try:
                    chat_data = await _api_get_async(
                        f"{self.url}/api/chats/{chat_id}",
                        self.api_key,
                    )
                except Exception as e:
                    logger.debug("Failed messages for %s: %s", chat_id, e)
                    continue

                # Messages can be in different keys depending on OWUI version
                messages = (
                    chat_data.get("messages")
                    or chat_data.get("history")
                    or (chat_data.get("chat") or {}).get("messages", [])
                )
                if not isinstance(messages, list):
                    continue

                for msg in messages:
                    saved = self.saver.save_message(
                        message=msg,
                        chat_title=chat_title,
                        chat_id=chat_id,
                        user_id=chat.get("user_id", "") if isinstance(chat, dict) else "",
                    )
                    if saved:
                        total_saved += 1

            page += 1

        self.saver.persist()
        stats = self.saver.get_stats()
        logger.info(
            "Sync: %d new, %d skipped (total: %d saved, %d errors)",
            total_saved,
            stats["skipped"],
            stats["saved"],
            stats["errors"],
        )
        return {"saved": total_saved, "skipped": stats["skipped"], "total": stats}

    async def run_forever(self):
        self._running = True
        logger.info("Auto-save started (interval=%ds, bank=%s)", self.interval, self.saver._bank)

        while self._running:
            try:
                if not await _health_check(self.url, self.api_key):
                    logger.warning("OpenWebUI unreachable, retry in %ds...", self.interval)
                    await asyncio.sleep(self.interval)
                    continue
                await self.sync_once()
            except Exception as e:
                logger.error("Sync error: %s", e)
            await asyncio.sleep(self.interval)

    def stop(self):
        self._running = False


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Auto-save OpenWebUI convos to Mnemosyne")
    parser.add_argument("--openwebui-url", default=os.environ.get("OPENWEBUI_URL", DEFAULT_OPENWEBUI_URL))
    parser.add_argument("--api-key", default=os.environ.get("OPENWEBUI_API_KEY", ""))
    parser.add_argument("--data-dir", default=os.environ.get("MNEMOSYNE_DATA_DIR", ""))
    parser.add_argument("--bank", default=os.environ.get("AUTO_SAVE_BANK", DEFAULT_BANK))
    parser.add_argument("--interval", type=int, default=int(os.environ.get("AUTO_SAVE_INTERVAL", str(DEFAULT_INTERVAL))))
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    _load_processed_ids()

    engine = SyncEngine(
        openwebui_url=args.openwebui_url,
        api_key=args.api_key,
        data_dir=args.data_dir,
        bank=args.bank,
        interval=args.interval,
    )

    if args.once:
        asyncio.run(engine.sync_once())
    else:
        try:
            asyncio.run(engine.run_forever())
        except KeyboardInterrupt:
            logger.info("Shutdown...")
            engine.stop()
            engine.saver.persist()
            logger.info("Bye.")


if __name__ == "__main__":
    main()
