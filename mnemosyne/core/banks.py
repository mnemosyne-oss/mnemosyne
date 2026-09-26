"""
Memory Bank Isolation
======================

Provides named memory banks that are fully isolated at the database level.
Each bank gets its own SQLite file under:
    ~/.hermes/mnemosyne/data/banks/<bank_name>/mnemosyne.db

This enables:
- Multi-tenant deployments (one bank per user/tenant)
- Domain separation (work vs personal vs project-specific memories)
- Testing isolation (test bank doesn't pollute production)
- Compliance boundaries (sensitive data in dedicated banks)

API:
    BankManager.create_bank("work")
    BankManager.list_banks() -> ["default", "work", "personal"]
    BankManager.delete_bank("work")
    BankManager.bank_exists("work") -> bool

    Mnemosyne(bank="work")  # All operations isolated to work bank
"""

import shutil
import sqlite3
from pathlib import Path
from typing import List

from mnemosyne.core.paths import _hermes_home, default_data_dir

# On Fly.io and other ephemeral VMs, only ~/.hermes is persisted.
# DEFAULT_DATA_DIR and BANKS_DIR stay importable but resolve on each access
# so a runtime HERMES_HOME change is visible without reimporting.


def _default_data_dir() -> Path:
    """Return the current default data directory, honoring runtime env changes."""
    return default_data_dir()


def __getattr__(name: str):
    """Resolve legacy path constants from the current environment."""
    if name == "_DEFAULT_ROOT":
        return _hermes_home()
    if name == "DEFAULT_DATA_DIR":
        return _default_data_dir()
    if name == "BANKS_DIR":
        return _default_data_dir() / "banks"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"_DEFAULT_ROOT", "DEFAULT_DATA_DIR", "BANKS_DIR"})


class BankManager:
    """
    Manage named memory banks.

    Each bank is a self-contained directory with its own SQLite database,
    enabling complete isolation between banks.
    """

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or _default_data_dir()
        self.banks_dir = self.data_dir / "banks"
        self.banks_dir.mkdir(parents=True, exist_ok=True)

    def create_bank(self, name: str) -> Path:
        """
        Create a new memory bank.

        Args:
            name: Bank name. Must be alphanumeric with hyphens/underscores.

        Returns:
            Path to the bank's database file.

        Raises:
            ValueError: If name is invalid or already exists.
        """
        self._validate_name(name)
        bank_dir = self.banks_dir / name
        if bank_dir.exists():
            raise ValueError(f"Bank '{name}' already exists")
        bank_dir.mkdir(parents=True)
        # Initialize the database by creating it
        db_path = bank_dir / "mnemosyne.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("SELECT 1")
        conn.close()
        return db_path

    def delete_bank(self, name: str, force: bool = False) -> bool:
        """
        Delete a memory bank and all its data.

        Args:
            name: Bank name to delete.
            force: If False, refuses to delete the 'default' bank.

        Returns:
            True if deleted, False if bank didn't exist.

        Raises:
            ValueError: If trying to delete 'default' without force=True.
        """
        self._validate_name(name)
        if name == "default" and not force:
            raise ValueError("Cannot delete 'default' bank without force=True")
        bank_dir = self.banks_dir / name
        if not bank_dir.exists():
            return False
        shutil.rmtree(bank_dir)
        return True

    def list_banks(self) -> List[str]:
        """Return list of all existing bank names."""
        if not self.banks_dir.exists():
            return ["default"]
        banks = [d.name for d in self.banks_dir.iterdir() if d.is_dir()]
        # Ensure 'default' is always present
        if "default" not in banks:
            banks.insert(0, "default")
        return sorted(banks)

    def bank_exists(self, name: str) -> bool:
        """Check if a bank exists."""
        self._validate_name(name)
        if name == "default":
            return True
        return (self.banks_dir / name).is_dir()

    def get_bank_db_path(self, name: str) -> Path:
        """
        Get the database path for a bank.

        The 'default' bank uses the legacy path (data_dir/mnemosyne.db).
        All other banks use banks_dir/<name>/mnemosyne.db.
        """
        if name == "default":
            return self.data_dir / "mnemosyne.db"
        self._validate_name(name)
        return self.banks_dir / name / "mnemosyne.db"

    def rename_bank(self, old_name: str, new_name: str) -> Path:
        """
        Rename a bank.

        Args:
            old_name: Existing bank name.
            new_name: New bank name.

        Returns:
            Path to the new bank's database.

        Raises:
            ValueError: If old_name doesn't exist or new_name is taken.
        """
        self._validate_name(old_name)
        if old_name == "default":
            raise ValueError("Cannot rename 'default' bank")
        self._validate_name(new_name)
        old_dir = self.banks_dir / old_name
        new_dir = self.banks_dir / new_name
        if not old_dir.exists():
            raise ValueError(f"Bank '{old_name}' does not exist")
        if new_dir.exists():
            raise ValueError(f"Bank '{new_name}' already exists")
        old_dir.rename(new_dir)
        return new_dir / "mnemosyne.db"

    def get_bank_stats(self, name: str) -> dict:
        """
        Get statistics for a bank.

        Returns dict with: exists, db_path, db_size_bytes.
        """
        db_path = self.get_bank_db_path(name)
        exists = db_path.exists()
        size = db_path.stat().st_size if exists else 0
        return {
            "name": name,
            "exists": exists,
            "db_path": str(db_path),
            "db_size_bytes": size,
        }

    def _validate_name(self, name: str):
        """Validate bank name format (delegates to module-level helper)."""
        _validate_bank_name(name)


# ---------------------------------------------------------------------------
# Module-level helpers (side-effect-free)
# ---------------------------------------------------------------------------

def _validate_bank_name(name: str):
    """Validate bank name format without touching the filesystem."""
    if not name:
        raise ValueError("Bank name cannot be empty")
    if name == "default":
        return  # 'default' is always valid
    if not all(c.isalnum() or c in "-_" for c in name):
        raise ValueError(
            f"Invalid bank name '{name}'. Use alphanumeric, hyphens, underscores only."
        )
    if len(name) > 64:
        raise ValueError(f"Bank name '{name}' exceeds 64 characters")


def bank_exists_read_only(name: str, data_dir: Path = None) -> bool:
    """Check whether a named bank exists WITHOUT creating any directories.

    Unlike ``BankManager(name).bank_exists(name)``, this never calls
    ``mkdir`` — ``BankManager.__init__`` eagerly creates ``banks/``. Use it
    when rejecting unknown banks before any database initialization, so we never
    silently materialize an empty bank directory or DB for invalid input.

    Resolution follows the same authoritative rules as the rest of the module:
    ``MNEMOSYNE_DATA_DIR`` wins, otherwise ``<HERMES_HOME>/mnemosyne/data``,
    otherwise ``~/.hermes/mnemosyne/data``. Read on every call.
    """
    _validate_bank_name(name)
    if name == "default":
        return True
    resolved: Path = data_dir or _default_data_dir()
    banks_dir = resolved / "banks"
    return (banks_dir / name).is_dir()


def get_bank_db_path_read_only(name: str, data_dir: Path = None) -> Path:
    """Resolve an existing bank database path without initializing bank state.

    This is the read-only counterpart to ``BankManager.get_bank_db_path`` for
    inspection paths. It validates bank names but never constructs a
    ``BankManager`` (whose constructor creates ``banks/``). The default bank
    retains its legacy location; named banks must already have both their bank
    directory and database file on disk.
    """
    _validate_bank_name(name)
    resolved: Path = data_dir or _default_data_dir()
    if name == "default":
        return resolved / "mnemosyne.db"

    if not bank_exists_read_only(name, data_dir=resolved):
        raise ValueError(f"Bank '{name}' does not exist")

    db_path = resolved / "banks" / name / "mnemosyne.db"
    if not db_path.is_file():
        raise FileNotFoundError(f"Database for bank '{name}' does not exist")
    return db_path


# Module-level convenience functions
def create_bank(name: str, data_dir: Path = None) -> Path:
    """Create a new memory bank."""
    return BankManager(data_dir).create_bank(name)


def delete_bank(name: str, data_dir: Path = None, force: bool = False) -> bool:
    """Delete a memory bank."""
    return BankManager(data_dir).delete_bank(name, force=force)


def list_banks(data_dir: Path = None) -> List[str]:
    """List all memory banks."""
    return BankManager(data_dir).list_banks()


def bank_exists(name: str, data_dir: Path = None) -> bool:
    """Check if a bank exists."""
    return BankManager(data_dir).bank_exists(name)
