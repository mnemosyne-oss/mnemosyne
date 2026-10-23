"""The packaged sleep workers must release the handle they opened (mnemosyne#1163).

Both workers open their own ``BeamMemory`` on a short-lived thread and release
it in ``finally``. That release used to import ``mnemosyne.core.memory`` for the
helper, and that module runs ``init_db()`` at import time: it opens a
default-database connection and republishes it into ``beam``'s thread-local
cache, so the worker's own handle survived cleanup. Only a cold process shows
this, hence the fresh interpreter per case.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

_WORKER_RELEASE_PROBE = '''
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
method = sys.argv[2]
repo = Path(sys.argv[3])
path_mode = sys.argv[4]
sleep_mode = sys.argv[5]

sys.path.insert(0, str(repo / "integrations" / "hermes" / "src"))
sys.path.insert(0, str(repo))
os.environ["MNEMOSYNE_DATA_DIR"] = str(root / "default")
os.environ.setdefault("MNEMOSYNE_NO_EMBEDDINGS", "1")

import mnemosyne_hermes as provider_module
from mnemosyne.core.beam import BeamMemory

default_db = (root / "default" / "mnemosyne.db").resolve()
db_path = default_db if path_mode == "match" else (root / "worker" / "mnemosyne.db").resolve()
observed = {}
created = {}


class RecordingBeam(BeamMemory):
    """Real BeamMemory that remembers the handle the worker opened."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        created["conn"] = self.conn

    def sleep(self, *args, **kwargs):
        # The release sits in ``finally`` precisely so a failing sleep still
        # releases; ``raises`` proves cleanup does not depend on success.
        observed["sleep_raised"] = sleep_mode == "raises"
        if sleep_mode == "raises":
            raise RuntimeError("injected sleep failure")
        return super().sleep(*args, **kwargs)


provider_module._get_beam_class = lambda: RecordingBeam
original_release = provider_module._release_thread_connection


def spy_release(db_path_arg=None):
    """Observe the worker's handle at the exact moment cleanup runs."""
    original_release(db_path_arg)
    conn = created.get("conn")
    if conn is not None:
        try:
            conn.execute("SELECT 1").fetchone()
            observed["closed_at_release"] = False
        except Exception:
            observed["closed_at_release"] = True
    observed["memory_imported"] = "mnemosyne.core.memory" in sys.modules
    observed["release_db_path"] = str(db_path_arg)


provider_module._release_thread_connection = spy_release

provider = provider_module.MnemosyneMemoryProvider()
parent = BeamMemory(session_id="parent", db_path=db_path)
provider._beam = parent
provider._agent_context = "primary"
sleep_args = {
    "session_id": "parent",
    "db_path": db_path,
    "author_id": None,
    "author_type": None,
    "channel_id": "parent",
}

if method == "auto_sleep":
    provider._auto_sleep_snapshot_locked = lambda *a, **k: (1, 1, sleep_args, "default", "primary")
    provider._maybe_auto_sleep()
else:
    provider.on_session_end([])

assert "closed_at_release" in observed, "worker finished without releasing its handle"
observed["parent_usable"] = parent.conn.execute("SELECT 1").fetchone()[0] == 1
observed["worker_ran"] = True
print(json.dumps(observed))
'''


def _run_worker_probe(root: Path, method: str, path_mode: str, sleep_mode: str) -> dict:
    """Drive one packaged worker method in a fresh interpreter."""
    script = root / "worker_release_probe.py"
    script.write_text(_WORKER_RELEASE_PROBE)
    env = dict(os.environ)
    env["MNEMOSYNE_NO_EMBEDDINGS"] = "1"
    env["HERMES_HOME"] = str(root / "hermes-home")
    completed = subprocess.run(
        [sys.executable, str(script), str(root), method, str(_REPO_ROOT), path_mode, sleep_mode],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("sleep_mode", ["noop", "raises"])
@pytest.mark.parametrize("path_mode", ["match", "differ"])
@pytest.mark.parametrize("method", ["auto_sleep", "session_end"])
def test_packaged_sleep_worker_releases_the_handle_it_opened(
    tmp_path, method, path_mode, sleep_mode
):
    """Cleanup releases the worker's own handle without importing core.memory.

    A cold process (profile isolation off) is the failing case: the import that
    used to happen inside cleanup displaced the handle first, so the release
    either closed the import's replacement and reported success or closed
    nothing while the worker's handle stayed open. ``match`` keeps the worker's
    database on the default path, ``differ`` moves it elsewhere, and ``raises``
    makes ``sleep()`` fail so the ``finally`` placement is exercised too.
    """
    observed = _run_worker_probe(tmp_path, method, path_mode, sleep_mode)

    assert observed["worker_ran"] is True
    assert observed["release_db_path"].endswith(
        "default/mnemosyne.db" if path_mode == "match" else "worker/mnemosyne.db"
    )
    assert observed.get("sleep_raised") is (sleep_mode == "raises")
    assert observed["closed_at_release"] is True
    assert observed["memory_imported"] is False
    assert observed["parent_usable"] is True
