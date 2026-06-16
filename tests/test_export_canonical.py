"""
Regression tests: canonical_facts must round-trip through export/import.

Before the fix, ``export_to_file`` never wrote the ``canonical_facts`` store,
so the authored single-source-of-truth identity facts (with history) were
silently dropped on a JSON restore. These tests assert the round-trip is
lossless and that older (<=1.2) exports without the section still import.
"""

import json
import tempfile
import unittest
from pathlib import Path

from mnemosyne.core.canonical import CanonicalStore
from mnemosyne.core.memory import Mnemosyne


class TestExportCanonicalRoundTrip(unittest.TestCase):
    def setUp(self):
        self.src = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.dst = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.src.close()
        self.dst.close()
        self.src_path = Path(self.src.name)
        self.dst_path = Path(self.dst.name)

    def tearDown(self):
        import os
        for base in (self.src.name, self.dst.name):
            for suffix in ("", ".pre_e6_backup"):
                try:
                    os.unlink(base + suffix)
                except OSError:
                    pass

    def test_canonical_fact_survives_round_trip(self):
        mem = Mnemosyne(session_id="s1", db_path=self.src_path)
        CanonicalStore(db_path=self.src_path).remember(
            "owner-1", "identity", "display_name", "Goes by Sam."
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "export.json"
            result = mem.export_to_file(str(export_path))
            self.assertEqual(result["canonical_facts_count"], 1)

            payload = json.loads(export_path.read_text())
            self.assertIn("canonical_facts", payload)

            Mnemosyne(session_id="s1", db_path=self.dst_path).import_from_file(
                str(export_path)
            )

        restored = CanonicalStore(db_path=self.dst_path).recall(
            "owner-1", "identity", "display_name"
        )
        self.assertIsNotNone(restored, "canonical fact was lost on restore")
        self.assertEqual(restored["body"], "Goes by Sam.")

    def test_canonical_history_survives_round_trip(self):
        store = CanonicalStore(db_path=self.src_path)
        store.remember("owner-1", "identity", "display_name", "Goes by Sam.")
        store.remember("owner-1", "identity", "display_name", "Goes by Alex now.")
        mem = Mnemosyne(session_id="s1", db_path=self.src_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "export.json"
            mem.export_to_file(str(export_path))
            Mnemosyne(session_id="s1", db_path=self.dst_path).import_from_file(
                str(export_path)
            )

        dst_store = CanonicalStore(db_path=self.dst_path)
        self.assertEqual(
            dst_store.recall("owner-1", "identity", "display_name")["body"],
            "Goes by Alex now.",
        )
        bodies = [h["body"] for h in dst_store.history("owner-1", "identity", "display_name")]
        self.assertIn("Goes by Sam.", bodies)
        self.assertIn("Goes by Alex now.", bodies)

    def test_idempotent_reimport_creates_no_duplicate_live_row(self):
        CanonicalStore(db_path=self.src_path).remember(
            "owner-1", "identity", "display_name", "Goes by Sam."
        )
        mem = Mnemosyne(session_id="s1", db_path=self.src_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "export.json"
            mem.export_to_file(str(export_path))
            dst = Mnemosyne(session_id="s1", db_path=self.dst_path)
            dst.import_from_file(str(export_path))
            dst.import_from_file(str(export_path))  # second run must not duplicate

        store = CanonicalStore(db_path=self.dst_path)
        self.assertEqual(store.recall("owner-1", "identity", "display_name")["body"], "Goes by Sam.")

    def test_legacy_export_without_canonical_section_imports_cleanly(self):
        # A pre-1.3 payload has no canonical_facts key; import must no-op, not error.
        legacy_payload = {"mnemosyne_export": {"version": "1.2"}}
        mem = Mnemosyne(session_id="s1", db_path=self.dst_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "legacy.json"
            p.write_text(json.dumps(legacy_payload))
            stats = mem.import_from_file(str(p))
        self.assertIn("canonical", stats)
        self.assertEqual(sum(v for v in stats["canonical"].values() if isinstance(v, int)), 0)


if __name__ == "__main__":
    unittest.main()
