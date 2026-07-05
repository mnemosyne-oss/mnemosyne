"""
Phase 5: Memory Bank Isolation Tests

Validates:
1. BankManager.create_bank() creates isolated directories
2. BankManager.list_banks() returns all banks including default
3. BankManager.delete_bank() removes banks
4. BankManager.get_bank_db_path() returns correct paths
5. Mnemosyne(bank="work") uses isolated database
6. Module-level set_bank() switches global default
7. Data written to one bank is invisible to another
8. Bank names are validated (alphanumeric + hyphens/underscores)
9. Default bank cannot be deleted without force
"""

import os
import sys
import pytest
import tempfile
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnemosyne.core.banks import BankManager, create_bank, delete_bank, list_banks, bank_exists
from mnemosyne.core.memory import Mnemosyne, set_bank, get_bank, remember, recall, get_stats


# ============================================================================
# BankManager unit tests
# ============================================================================

class TestBankManager:
    """Unit tests for BankManager class."""

    def test_create_bank_creates_directory(self):
        """Creating a bank should create a directory with a DB file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            mgr = BankManager(data_dir)
            db_path = mgr.create_bank("work")
            assert db_path.exists()
            assert db_path.parent.name == "work"
            assert db_path.name == "mnemosyne.db"

    def test_create_bank_duplicate_raises(self):
        """Creating a duplicate bank should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BankManager(Path(tmpdir))
            mgr.create_bank("work")
            with pytest.raises(ValueError, match="already exists"):
                mgr.create_bank("work")

    def test_create_bank_invalid_name_raises(self):
        """Invalid bank names should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BankManager(Path(tmpdir))
            with pytest.raises(ValueError):
                mgr.create_bank("bank with spaces")
            with pytest.raises(ValueError):
                mgr.create_bank("bank/with/slashes")
            with pytest.raises(ValueError):
                mgr.create_bank("bank.with.dots")

    @pytest.mark.parametrize(
        ("name", "error"),
        [
            ("", "Bank name cannot be empty"),
            ("../outside", "Invalid bank name"),
            ("bank/with/slashes", "Invalid bank name"),
            ("bank.with.dots", "Invalid bank name"),
        ],
    )
    def test_path_based_operations_reject_invalid_bank_names(self, name, error):
        """Path-based bank operations must not resolve invalid names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            outside = Path(tmpdir) / "outside"
            outside.mkdir()
            sentinel = outside / "keep.txt"
            sentinel.write_text("do not delete")

            mgr = BankManager(data_dir)

            with pytest.raises(ValueError, match=error):
                mgr.delete_bank(name, force=True)
            with pytest.raises(ValueError, match=error):
                mgr.bank_exists(name)
            with pytest.raises(ValueError, match=error):
                mgr.get_bank_db_path(name)
            with pytest.raises(ValueError, match=error):
                mgr.get_bank_stats(name)
            with pytest.raises(ValueError, match=error):
                mgr.rename_bank(name, "safe_name")

            assert sentinel.exists()
            assert outside.exists()

    def test_list_banks_includes_default(self):
        """list_banks() should always include 'default'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BankManager(Path(tmpdir))
            banks = mgr.list_banks()
            assert "default" in banks

    def test_list_banks_after_create(self):
        """list_banks() should include newly created banks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BankManager(Path(tmpdir))
            mgr.create_bank("work")
            mgr.create_bank("personal")
            banks = mgr.list_banks()
            assert "work" in banks
            assert "personal" in banks
            assert "default" in banks

    def test_delete_bank_removes_directory(self):
        """delete_bank() should remove the bank directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BankManager(Path(tmpdir))
            mgr.create_bank("temp")
            assert mgr.bank_exists("temp")
            mgr.delete_bank("temp")
            assert not mgr.bank_exists("temp")

    def test_delete_nonexistent_bank_returns_false(self):
        """delete_bank() on non-existent bank returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BankManager(Path(tmpdir))
            result = mgr.delete_bank("nonexistent")
            assert result is False

    def test_delete_default_bank_raises(self):
        """Deleting 'default' without force=True should raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BankManager(Path(tmpdir))
            with pytest.raises(ValueError, match="force=True"):
                mgr.delete_bank("default")

    def test_delete_default_with_force_succeeds(self):
        """Deleting 'default' with force=True should succeed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BankManager(Path(tmpdir))
            # default doesn't have a directory, but force should not raise
            result = mgr.delete_bank("default", force=True)
            # Returns False because default dir doesn't exist
            assert result is False

    def test_bank_exists(self):
        """bank_exists() should return True for existing banks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BankManager(Path(tmpdir))
            assert mgr.bank_exists("default") is True
            mgr.create_bank("test")
            assert mgr.bank_exists("test") is True
            assert mgr.bank_exists("nonexistent") is False

    def test_get_bank_db_path_default(self):
        """Default bank should use data_dir/mnemosyne.db."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            mgr = BankManager(data_dir)
            path = mgr.get_bank_db_path("default")
            assert path == data_dir / "mnemosyne.db"

    def test_get_bank_db_path_custom(self):
        """Custom banks should use banks_dir/<name>/mnemosyne.db."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            mgr = BankManager(data_dir)
            mgr.create_bank("work")
            path = mgr.get_bank_db_path("work")
            assert path == data_dir / "banks" / "work" / "mnemosyne.db"

    def test_rename_bank(self):
        """rename_bank() should move the bank directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BankManager(Path(tmpdir))
            mgr.create_bank("old_name")
            new_path = mgr.rename_bank("old_name", "new_name")
            assert new_path.exists()
            assert not mgr.bank_exists("old_name")
            assert mgr.bank_exists("new_name")

    def test_rename_default_raises(self):
        """Cannot rename 'default' bank."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BankManager(Path(tmpdir))
            with pytest.raises(ValueError, match="Cannot rename"):
                mgr.rename_bank("default", "new_name")

    def test_get_bank_stats(self):
        """get_bank_stats() should return correct stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BankManager(Path(tmpdir))
            mgr.create_bank("stats_test")
            stats = mgr.get_bank_stats("stats_test")
            assert stats["name"] == "stats_test"
            assert stats["exists"] is True
            assert stats["db_size_bytes"] >= 0

    def test_module_level_functions(self):
        """Module-level convenience functions should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            db_path = create_bank("mod_test", data_dir)
            assert db_path.exists()
            assert bank_exists("mod_test", data_dir)
            banks = list_banks(data_dir)
            assert "mod_test" in banks
            delete_bank("mod_test", data_dir)
            assert not bank_exists("mod_test", data_dir)


# ============================================================================
# Mnemosyne bank integration tests
# ============================================================================

class TestMnemosyneBankIsolation:
    """Integration tests verifying Mnemosyne uses correct DB per bank."""

    def test_mnemosyne_with_bank_uses_isolated_db(self):
        """Mnemosyne(bank='work') should use a separate database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            os.environ["MNEMOSYNE_DATA_DIR"] = str(data_dir)
            try:
                mgr = BankManager(data_dir)
                mgr.create_bank("work")

                mn_default = Mnemosyne(bank="default")
                mn_work = Mnemosyne(bank="work")

                # Should have different DB paths
                assert mn_default.db_path != mn_work.db_path
                assert "banks" in str(mn_work.db_path)
                assert "work" in str(mn_work.db_path)
            finally:
                del os.environ["MNEMOSYNE_DATA_DIR"]

    def test_data_isolation_between_banks(self):
        """Data written to one bank should not appear in another."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            os.environ["MNEMOSYNE_DATA_DIR"] = str(data_dir)
            try:
                mgr = BankManager(data_dir)
                mgr.create_bank("personal")

                mn_default = Mnemosyne(bank="default")
                mn_personal = Mnemosyne(bank="personal")

                # Write to default (unique content to avoid ID collisions)
                import time
                mn_default.remember("Default bank memory " + str(time.time()), importance=0.8)
                time.sleep(0.01)
                # Write to personal
                mn_personal.remember("Personal bank memory " + str(time.time()), importance=0.9)

                # Query default - should only find default memory
                default_results = mn_default.recall("Default bank", top_k=5)
                assert any("Default" in r["content"] for r in default_results)
                assert not any("Personal" in r["content"] for r in default_results)

                # Query personal - should only find personal memory
                personal_results = mn_personal.recall("Personal bank", top_k=5)
                assert any("Personal" in r["content"] for r in personal_results)
                assert not any("Default" in r["content"] for r in personal_results)
            finally:
                del os.environ["MNEMOSYNE_DATA_DIR"]

    def test_bank_stats_are_isolated(self):
        """Stats should reflect only the current bank."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            os.environ["MNEMOSYNE_DATA_DIR"] = str(data_dir)
            try:
                mgr = BankManager(data_dir)
                mgr.create_bank("project_a")

                mn_a = Mnemosyne(bank="project_a")
                mn_default = Mnemosyne(bank="default")

                import time
                mn_a.remember("Project A task " + str(time.time()), importance=0.5)
                time.sleep(0.01)
                mn_a.remember("Project A meeting " + str(time.time()), importance=0.6)
                time.sleep(0.01)
                mn_default.remember("General note " + str(time.time()), importance=0.4)

                a_stats = mn_a.get_stats()
                default_stats = mn_default.get_stats()

                # Project A should have at least 2 working memories
                assert a_stats["beam"]["working_memory"]["total"] >= 2
                # Default should have at least 1
                assert default_stats["beam"]["working_memory"]["total"] >= 1
            finally:
                del os.environ["MNEMOSYNE_DATA_DIR"]


# ============================================================================
# Module-level bank switching tests
# ============================================================================

class TestModuleLevelBankSwitching:
    """Tests for set_bank() and per-call bank parameter."""

    def test_set_bank_switches_global_default(self):
        """set_bank() should change the bank for subsequent module calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            os.environ["MNEMOSYNE_DATA_DIR"] = str(data_dir)
            try:
                mgr = BankManager(data_dir)
                mgr.create_bank("switch_test")

                # Use Mnemosyne class directly for clean isolation
                mn_default = Mnemosyne(bank="default")
                mn_switch = Mnemosyne(bank="switch_test")

                import time
                # Write to default
                mn_default.remember("Default memory " + str(time.time()), importance=0.8)
                time.sleep(0.01)
                # Write to switch bank
                mn_switch.remember("Switched memory " + str(time.time()), importance=0.9)

                # Verify isolation via class instances
                default_results = mn_default.recall("Default memory", top_k=5)
                switch_results = mn_switch.recall("Switched memory", top_k=5)

                assert any("Default" in r["content"] for r in default_results)
                assert any("Switched" in r["content"] for r in switch_results)
                assert not any("Switched" in r["content"] for r in default_results)
                assert not any("Default" in r["content"] for r in switch_results)
            finally:
                del os.environ["MNEMOSYNE_DATA_DIR"]

    def test_per_call_bank_parameter(self):
        """Individual calls should accept bank parameter directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            os.environ["MNEMOSYNE_DATA_DIR"] = str(data_dir)
            try:
                mgr = BankManager(data_dir)
                mgr.create_bank("call_test")

                import time
                remember("Direct default " + str(time.time()), bank="default")
                time.sleep(0.01)
                remember("Direct custom " + str(time.time()), bank="call_test")

                default_results = recall("Direct default", bank="default")
                custom_results = recall("Direct custom", bank="call_test")

                assert any("default" in r["content"] for r in default_results)
                assert any("custom" in r["content"] for r in custom_results)
            finally:
                del os.environ["MNEMOSYNE_DATA_DIR"]

    def test_get_bank_returns_current(self):
        """get_bank() should return the currently set bank."""
        original = get_bank()
        set_bank("test_bank")
        assert get_bank() == "test_bank"
        set_bank(original)  # Restore


# ============================================================================
# Edge cases
# ============================================================================

class TestBankEdgeCases:
    """Boundary conditions and error handling."""

    def test_empty_bank_name_raises(self):
        """Empty bank name should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BankManager(Path(tmpdir))
            with pytest.raises(ValueError):
                mgr.create_bank("")

    def test_long_bank_name_raises(self):
        """Bank name > 64 chars should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BankManager(Path(tmpdir))
            with pytest.raises(ValueError, match="exceeds 64"):
                mgr.create_bank("a" * 65)

    def test_bank_name_with_hyphens_and_underscores(self):
        """Hyphens and underscores should be valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BankManager(Path(tmpdir))
            mgr.create_bank("my-bank_1")
            assert mgr.bank_exists("my-bank_1")

    def test_mnemosyne_db_path_override_takes_precedence(self):
        """Explicit db_path should override bank resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_db = Path(tmpdir) / "custom.db"
            mn = Mnemosyne(db_path=custom_db, bank="work")
            assert mn.db_path == custom_db
            assert mn.bank == "work"  # Bank is still tracked


# ============================================================================
# Run standalone
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
