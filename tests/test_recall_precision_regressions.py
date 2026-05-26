import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.importers import hindsight as hindsight_importer


class TestRecallPrecisionRegressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "mnemosyne.db"
        self.beam = BeamMemory(session_id="precision", db_path=self.db_path)
        self.memories = {
            "deployment_artifact": (
                "Project Orion lab runner starts from the OpenJDK downloads directory "
                "with artifact orion-runner-2026.4.jar and must bind only to 127.0.0.1."
            ),
            "course_names": (
                "For training modules, display full course titles only: Application Security, "
                "Data Analysis, Database Design, Technical Writing, and Product Marketing; "
                "never use abbreviated module codes in user-facing summaries."
            ),
            "automation_policy": (
                "Scheduled automation prompts must discover current context dynamically at runtime "
                "by reading files and querying memory; do not hardcode stale project facts."
            ),
            "travel_plan": (
                "For the conference trip, the attendee stays at Hotel Meridian and the safer running "
                "plan is rideshare to Central Park Loop, then run the 1.6 km park loops."
            ),
            "routing_policy": (
                "Inference routing after Premium Plan: avoid BudgetCloud unless approved; foreground chat "
                "uses Model-A and Model-B is preferred for scheduled and background work."
            ),
            "deadline_noise": (
                "Portfolio checkpoint review is due June 5, 2026, marked lower urgency but useful "
                "to maintain momentum."
            ),
        }
        for content in self.memories.values():
            self.beam.remember(content, source="imported_fixture", importance=0.6, scope="global", veracity="imported")

    def tearDown(self):
        self.tmp.cleanup()

    def assert_top_contains(self, query, expected):
        results = self.beam.recall(query, top_k=5)
        self.assertTrue(results, f"no results for {query!r}")
        top = results[0]["content"].lower()
        self.assertIn(expected.lower(), top, f"wrong top result for {query!r}: {results[0]['content']!r}")

    def test_natural_question_prefers_artifact_memory_over_memoria_or_due_date(self):
        self.assert_top_contains(
            "Where is the Orion runner jar and how should it bind?",
            "orion-runner-2026.4.jar",
        )

    def test_specific_memory_queries_rank_correct_fact_first(self):
        probes = [
            ("What training module naming rule avoids abbreviated codes?", "Application Security"),
            ("How should scheduled automation handle context instead of hardcoding facts?", "dynamically"),
            ("What Hotel Meridian running route plan should be used?", "Central Park Loop"),
            ("What inference routing rule says avoid BudgetCloud?", "avoid BudgetCloud"),
        ]
        for query, expected in probes:
            with self.subTest(query=query):
                self.assert_top_contains(query, expected)

    def test_nonsense_query_abstains_instead_of_returning_low_overlap_memories(self):
        results = self.beam.recall("zxqvplm norf greeble snargle twompset", top_k=5)
        self.assertEqual([], results)

    def test_natural_language_nonsense_abstains_despite_one_real_token(self):
        self.beam.remember(
            "Quantum field theory research notes are stored in the physics archive.",
            source="imported_fixture",
            importance=0.9,
            scope="global",
            veracity="imported",
        )
        self.beam.conn.execute(
            """
            INSERT INTO episodic_memory
                (id, content, source, timestamp, session_id, importance, scope, veracity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "episodic_quantum_note",
                "Quantum field theory research notes are stored in the physics archive.",
                "imported_fixture",
                "2026-05-24T00:00:00",
                "precision",
                0.9,
                "global",
                "imported",
            ),
        )
        self.beam.conn.commit()
        results = self.beam.recall("purple bicycle quantum oatmeal unrelated", top_k=5)
        self.assertEqual([], results)

    def test_three_token_noise_abstains_on_single_shared_token(self):
        self.beam.remember(
            "Invoice drills use the order identifier as the primary key.",
            source="imported_fixture",
            importance=0.9,
            scope="global",
            veracity="imported",
        )
        results = self.beam.recall("customer invoices quantum", top_k=5)
        self.assertEqual([], results)

    def test_memoria_date_or_sequence_fact_does_not_force_top_slot(self):
        results = self.beam.recall("Where is the Orion runner jar and how should it bind?", top_k=5)
        self.assertTrue(results)
        self.assertNotIn("[MEMORIA", results[0]["content"])

    def test_multi_fact_query_keeps_separate_aspects_in_top_results(self):
        self.beam.remember(
            "Ava profile URL is https://example.test/ava for her professional page.",
            source="imported_fixture",
            importance=0.6,
            scope="global",
            veracity="imported",
        )
        self.beam.remember(
            "Ava rejects AI hype positioning and wants grounded software builder wording.",
            source="imported_fixture",
            importance=0.6,
            scope="global",
            veracity="imported",
        )
        for n in range(10):
            self.beam.remember(
                f"Ava profile checklist item {n}: professional photo headline about section skills portfolio connections completed.",
                source="imported_fixture",
                importance=0.8,
                scope="global",
                veracity="imported",
            )

        results = self.beam.recall(
            "What is Ava profile URL and professional branding preference?",
            top_k=5,
        )
        joined = "\n".join(r["content"] for r in results).lower()
        self.assertIn("https://example.test/ava", joined)
        self.assertIn("grounded software builder", joined)

    def test_current_state_query_prefers_newer_correction_over_stale_history(self):
        old_id = self.beam.remember(
            "Project Atlas deployment target was legacy-cluster and should use Model-Old for background work.",
            source="imported_fixture",
            importance=0.7,
            scope="global",
            veracity="imported",
        )
        new_id = self.beam.remember(
            "Current Project Atlas deployment target is stable-cluster and should use Model-New for background work.",
            source="imported_fixture",
            importance=0.7,
            scope="global",
            veracity="imported",
        )
        self.beam.conn.execute(
            "UPDATE working_memory SET timestamp = ? WHERE id = ?",
            ("2025-01-01T00:00:00", old_id),
        )
        self.beam.conn.execute(
            "UPDATE working_memory SET timestamp = ? WHERE id = ?",
            ("2026-05-24T00:00:00", new_id),
        )
        self.beam.conn.commit()

        results = self.beam.recall(
            "What should Project Atlas deployment use now?",
            top_k=3,
        )
        self.assertTrue(results)
        self.assertIn("stable-cluster", results[0]["content"])

    def test_memoria_result_carries_source_memory_and_supplements_raw_row(self):
        source_id = self.beam.remember(
            "Telemetry API latency is 240ms after the cache migration.",
            source="imported_fixture",
            importance=0.6,
            scope="global",
            veracity="imported",
        )
        memoria = self.beam.memoria_retrieve("What is the Telemetry API latency?", top_k=3)
        self.assertIn(source_id, memoria.get("source_memory_ids", []))

        results = self.beam.recall("What is the Telemetry API latency?", top_k=5)
        self.assertTrue(any(r.get("source_memory_id") == source_id for r in results if r.get("tier") == "memoria_source"))


class FakeEmbeddings:
    @staticmethod
    def available():
        return True

    @staticmethod
    def embed(items):
        import numpy as np
        return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)


class TestHindsightImportEmbeddingBackfill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "mnemosyne.db"
        self.beam = BeamMemory(session_id="import", db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_backfill_writes_canonical_memory_embeddings_table(self):
        conn = self.beam.conn
        conn.execute(
            """
            INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("mem_import_1", "Imported artifact memory", "hindsight", "2026-05-24T00:00:00", "import", 0.6),
        )
        rowid = conn.execute("SELECT rowid FROM episodic_memory WHERE id = ?", ("mem_import_1",)).fetchone()[0]

        old_embeddings = hindsight_importer._embeddings
        old_vec_available = hindsight_importer._vec_available
        old_vec_insert = hindsight_importer._vec_insert
        old_mib = hindsight_importer._mib
        try:
            hindsight_importer._embeddings = FakeEmbeddings
            hindsight_importer._vec_available = lambda conn: False
            hindsight_importer._vec_insert = None
            hindsight_importer._mib = None
            hindsight_importer.HindsightImporter._backfill_import_embedding(
                conn, rowid, "Imported artifact memory"
            )
        finally:
            hindsight_importer._embeddings = old_embeddings
            hindsight_importer._vec_available = old_vec_available
            hindsight_importer._vec_insert = old_vec_insert
            hindsight_importer._mib = old_mib

        stored = conn.execute(
            "SELECT embedding_json FROM memory_embeddings WHERE memory_id = ?",
            ("mem_import_1",),
        ).fetchone()
        self.assertIsNotNone(stored)
        self.assertIn("0.1", stored[0])


if __name__ == "__main__":
    unittest.main()
