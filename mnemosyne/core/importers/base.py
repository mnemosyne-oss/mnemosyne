"""
Base classes for memory provider importers.

Each importer implements three methods:
- extract(): Pull all memories from the source provider
- transform(): Convert provider-specific format to Mnemosyne-compatible dicts
- validate(): Check extracted data before import
"""

import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict


@dataclass
class ImporterResult:
    """Result of an import operation."""
    provider: str
    total: int = 0
    imported: int = 0
    skipped: int = 0
    failed: int = 0
    errors: List[str] = field(default_factory=list)
    memory_ids: List[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> Dict:
        return {
            "provider": self.provider,
            "total": self.total,
            "imported": self.imported,
            "skipped": self.skipped,
            "failed": self.failed,
            "errors": self.errors[:20],  # cap at 20
            "memory_ids": self.memory_ids[:50],  # cap at 50
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class BaseImporter(ABC):
    """Abstract base class for memory provider importers.

    Subclass this for each provider. Implement extract(), transform(),
    and optionally validate().

    Usage:
        importer = Mem0Importer(api_key="sk-xxx", user_id="alice")
        result = importer.run(mnemosyne_instance)
    """

    provider_name: str = "unknown"

    def __init__(self, **kwargs):
        self.config = kwargs

    def run(self, mnemosyne, dry_run: bool = False,
            session_id: str = None, channel_id: str = None) -> ImporterResult:
        """Full import pipeline: extract → validate → transform → import.

        Args:
            mnemosyne: A Mnemosyne instance to import into.
            dry_run: If True, validate and transform but don't write.
            session_id: Override session_id for imported memories.
            channel_id: Channel to assign imported memories to.

        Returns:
            ImporterResult with counts and errors.
        """
        result = ImporterResult(
            provider=self.provider_name,
            started_at=datetime.now().isoformat(),
        )

        try:
            # Phase 1: Extract raw data from source
            raw_data = self.extract()
            result.total = len(raw_data)

            if result.total == 0:
                result.errors.append("No memories found to import")
                return result

            # Phase 2: Validate
            if not self.validate(raw_data):
                result.errors.append("Validation failed")
                return result

            # Phase 3: Transform to Mnemosyne format
            memories = self.transform(raw_data)

            if dry_run:
                result.imported = len(memories)
                result.skipped = 0
                result.failed = 0
                return result

            # Phase 4: Import into Mnemosyne
            for mem_dict in memories:
                try:
                    # Allow session/channel overrides from CLI
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
                    )
                    result.memory_ids.append(mid)
                    result.imported += 1
                except Exception as e:
                    result.failed += 1
                    result.errors.append(
                        f"Failed to import '{mem_dict.get('content', '')[:80]}': {e}"
                    )

            if result.skipped:
                result.skipped = result.total - result.imported - result.failed

        except Exception as e:
            result.errors.append(f"Import failed: {e}")

        result.finished_at = datetime.now().isoformat()
        return result

    @abstractmethod
    def extract(self) -> List[Dict]:
        """Extract all memories from the source provider.

        Returns a list of provider-specific raw dicts.
        Each dict should contain at minimum: 'content' (str).
        Common fields: 'timestamp', 'metadata', 'user_id', 'tags'.
        """

    @abstractmethod
    def transform(self, raw_data: List[Dict]) -> List[Dict]:
        """Transform provider-specific dicts to Mnemosyne-compatible dicts.

        Each returned dict should have:
            content: str (required)
            source: str (default: provider_name)
            importance: float (default: 0.5)
            metadata: dict (default: {})
            valid_until: str | None (default: None)
            scope: str (default: "session")

        Return a list of dicts ready for mnemosyne.remember().
        """

    def validate(self, raw_data: List[Dict]) -> bool:
        """Check that extracted data looks valid. Override for provider-specific checks."""
        if not raw_data:
            return False
        if not isinstance(raw_data, list):
            return False
        # At minimum, each item should be a dict with 'content'
        for item in raw_data:
            if not isinstance(item, dict):
                return False
            if "content" not in item and "memory" not in item and "text" not in item:
                # Allow common aliases — transform() handles normalization
                pass
        return True

    @staticmethod
    def _content_hash(content: str) -> str:
        """Generate a deterministic hash for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def import_from_file(filepath: str, mnemosyne, dry_run: bool = False,
                     session_id: str = None, channel_id: str = None) -> ImporterResult:
    """Import memories from a JSON export file.

    The file should contain an array of memory objects, each with at minimum a
    'content' key. Common formats from other providers are auto-detected.

    Args:
        filepath: Path to JSON export file.
        mnemosyne: Mnemosyne instance to import into.
        dry_run: If True, validate but don't write.
        session_id: Override session for imported memories.
        channel_id: Channel to assign imported memories to.

    Returns:
        ImporterResult
    """
    from .base import BaseImporter

    class FileImporter(BaseImporter):
        provider_name = "file"

        def __init__(self, filepath, **kwargs):
            super().__init__(**kwargs)
            self.filepath = filepath

        def extract(self):
            with open(self.filepath) as f:
                data = json.load(f)
            # Handle single object vs array
            if isinstance(data, dict):
                data = [data]
            # Handle wrapped formats: {"results": [...], "memories": [...]}
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                inner = data[0]
                if "results" in inner and isinstance(inner["results"], list):
                    data = inner["results"]
                elif "memories" in inner and isinstance(inner["memories"], list):
                    data = inner["memories"]
                elif "data" in inner and isinstance(inner["data"], list):
                    data = inner["data"]
            return data

        def transform(self, raw_data):
            memories = []
            for item in raw_data:
                content = item.get("content") or item.get("memory") or item.get("text", "")
                if not content:
                    continue
                memories.append({
                    "content": content,
                    "source": item.get("source", "file_import"),
                    "importance": float(item.get("importance", 0.5)),
                    "metadata": item.get("metadata", {}),
                    "valid_until": item.get("valid_until"),
                    "scope": item.get("scope", "session"),
                })
            return memories

    importer = FileImporter(filepath)
    return importer.run(mnemosyne, dry_run=dry_run,
                        session_id=session_id, channel_id=channel_id)
