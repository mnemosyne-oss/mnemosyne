# Mnemosyne PR Handoff — Auto-sleep & Threading Fix

## Status

- **Issue:** [#342](https://github.com/AxDSan/mnemosyne/issues/342) — OPEN
- **Abdiisan response (Jun 17):** "Go ahead and open the PR. Same-day review from me."
- **Next step:** Create PR with the fix

## What Needs To Be Done

1. Read this handoff
2. Read CONTRIBUTING.md in repo root
3. Read the upstream code at `hermes_memory_provider/__init__.py`
4. Apply the patches described below
5. Write tests in `tests/`
6. Follow contributing.md commit convention
7. Commit and push to `origin` (your fork)
8. Molly opens PR from GitHub web

## The Bugs

### Bug 1: `_maybe_auto_sleep()` uses `sleep_all_sessions()` (line 1867)

**File:** `hermes_memory_provider/__init__.py`

**Upstream code (line 1846-1874):**
```python
def _maybe_auto_sleep(self) -> None:
    try:
        stats = self._beam.get_working_stats()
        working = stats.get("total", 0)
        if working > self._auto_sleep_threshold:
            # ... eligibility check ...
            logger.info("Mnemosyne auto-sleep: working=%d, eligible=%d > threshold=%d", working, eligible, self._auto_sleep_threshold)
            sleep_fn = self._beam.sleep_all_sessions if hasattr(self._beam, "sleep_all_sessions") else self._beam.sleep  # <-- BUG: uses sleep_all_sessions
            sleep_thread = threading.Thread(target=sleep_fn, daemon=True)
            sleep_thread.start()
            sleep_thread.join(timeout=self._AUTO_SLEEP_TIMEOUT_SECONDS)
            if sleep_thread.is_alive():
                logger.warning("Mnemosyne auto-sleep timed out after %.0fs — consolidation deferred", self._AUTO_SLEEP_TIMEOUT_SECONDS)
    except Exception:
        pass
```

**Problem:** `sleep_all_sessions()` loops ALL sessions instead of just the current one. With many sessions and memories, this always exceeds the timeout. Auto-sleep never completes.

**Fix:** Use `self._beam.sleep()` directly (session-scoped, like `on_session_end` already does).

### Bug 2: Daemon threads share SQLite connection with main thread

**File:** `hermes_memory_provider/__init__.py`

**Problem in `_maybe_auto_sleep()` (line 1868):**
```python
sleep_thread = threading.Thread(target=sleep_fn, daemon=True)  # <-- uses self._beam
```

**Problem in `on_session_end()` (line 2694-2699):**
```python
def _sleep_with_logging():
    beam.sleep()  # <-- uses self._beam (shared connection)
```

Both functions run `beam.sleep()` in daemon threads but reuse `self._beam.conn` (the same SQLite connection). When the main thread continues doing `sync_turn()` (writing new memories), concurrent writes on the same connection cause episodic INSERT to fail silently (commit rolled back by concurrent main-thread writes).

**Fix:** Create isolated `BeamMemory` instances inside daemon threads so each gets its own SQLite connection via `_thread_local`.

## The Patches

### Patch 1: Fix `_maybe_auto_sleep()` — both bugs

Replace the entire `_maybe_auto_sleep` method (lines 1846-1874) with:

```python
def _maybe_auto_sleep(self) -> None:
    try:
        stats = self._beam.get_working_stats()
        working = stats.get("total", 0)
        if working > self._auto_sleep_threshold:
            # Cheap eligibility check: are there any unconsolidated
            # working memories old enough to consolidate?
            cutoff = (datetime.now() - timedelta(hours=WORKING_MEMORY_TTL_HOURS // 2)).isoformat()
            eligible = self._beam._count_unconsolidated_before(cutoff)
            if eligible == 0:
                return

            skip = self._reserve_reflection_budget("auto_sleep")
            if skip is not None:
                logger.info("Mnemosyne auto-sleep skipped: %s", json.dumps(skip))
                return

            logger.info("Mnemosyne auto-sleep: working=%d, eligible=%d > threshold=%d", working, eligible, self._auto_sleep_threshold)
            # Use session-scoped sleep to avoid timeout on large databases.
            # Create a SEPARATE BeamMemory instance for the daemon thread
            # so it gets its own SQLite connection via _thread_local.
            # Reusing self._beam.conn from a daemon thread races with the
            # main thread's sync_turn() writes, causing episodic INSERT
            # failures (commit rolled back by concurrent main-thread writes).
            beam_ref = self._beam
            def _sleep_isolated():
                try:
                    BeamClass = _get_beam_class()
                    sleep_beam = BeamClass(
                        session_id=beam_ref.session_id,
                        db_path=beam_ref.db_path,
                        author_id=beam_ref.author_id,
                        author_type=beam_ref.author_type,
                        channel_id=beam_ref.channel_id,
                    )
                    sleep_beam.sleep()
                except Exception as inner:
                    logger.debug("Mnemosyne auto-sleep worker failed: %s", inner)
            sleep_thread = threading.Thread(target=_sleep_isolated, daemon=True)
            sleep_thread.start()
            sleep_thread.join(timeout=self._AUTO_SLEEP_TIMEOUT_SECONDS)
            if sleep_thread.is_alive():
                logger.warning("Mnemosyne auto-sleep timed out after %.0fs — consolidation deferred", self._AUTO_SLEEP_TIMEOUT_SECONDS)
    except Exception:
        pass
```

### Patch 2: Fix `on_session_end()` — thread isolation only

Replace the `_sleep_with_logging` function inside `on_session_end` (lines 2694-2701) with:

```python
def _sleep_with_logging():
    # Wrap the target so exceptions get logged at the same
    # severity the previous synchronous version used, instead
    # of bubbling out as an uncaught daemon-thread traceback.
    # Create a SEPARATE BeamMemory so the thread gets its own
    # SQLite connection via _thread_local, avoiding races with
    # the main thread's writes.
    try:
        BeamClass = _get_beam_class()
        sleep_beam = BeamClass(
            session_id=beam_ref.session_id,
            db_path=beam_ref.db_path,
            author_id=beam_ref.author_id,
            author_type=beam_ref.author_type,
            channel_id=beam_ref.channel_id,
        )
        sleep_beam.sleep()
    except Exception as inner:
        logger.debug("Mnemosyne session-end sleep failed: %s", inner)
```

Also change line 2692 from:
```python
beam = self._beam
```
to:
```python
beam_ref = self._beam
```

## Reference: What The Local Patch Looks Like

The installed package at `~/.hermes/hermes-agent/venv/lib/python3.11/site-packages/mnemosyne_hermes/__init__.py` has these patches applied and working. Key differences from upstream:

- `_maybe_auto_sleep` (line 921-949): Uses `beam.sleep()` directly, creates isolated `BeamMemory` in daemon thread
- `on_session_end` (line 1727-1757): Creates isolated `BeamMemory` in daemon thread

## Commit Convention

From CONTRIBUTING.md:
- Use conventional commits: `fix(scope): description`
- Bump version in `mnemosyne/__init__.py`
- Update `CHANGELOG.md`
- Add tests in `tests/`

**Suggested commit message:**
```
fix: auto-sleep uses sleep_all_sessions() causing timeout, daemon thread SQLite race

- _maybe_auto_sleep: use beam.sleep() (session-scoped) instead of
  sleep_all_sessions() which processes ALL sessions and always times out
- _maybe_auto_sleep and on_session_end: create isolated BeamMemory
  instances in daemon threads to avoid SQLite connection races with
  main thread's sync_turn() writes
```

## Testing

1. Run existing tests: `python -m pytest tests/ -v`
2. Add test for auto-sleep using correct function
3. Add test for daemon thread isolation (isolated BeamMemory instances)
4. Verify all tests pass

## Environment

- **Repo:** `~/mnemosyne` (your fork: ruangraung/mnemosyne)
- **Branch:** `main` (create a feature branch first)
- **Python:** 3.11 (venv at `~/.hermes/hermes-agent/venv`)
- **Upstream:** AxDSan/mnemosyne
- **Installed package:** `mnemosyne-hermes 0.1.8` (local patches applied)
- **mnemosyne-memory:** 3.7.0 (latest)

## Key Files

| File | Purpose |
|------|---------|
| `hermes_memory_provider/__init__.py` | Main file to patch (lines 1846-1874, 2677-2713) |
| `mnemosyne/__init__.py` | Version bump |
| `CHANGELOG.md` | Add entry |
| `tests/` | Add tests |
| `CONTRIBUTING.md` | Commit conventions |

## dplush's Suggestion

dplush also suggested making the eligibility check session-scoped. Currently line 1857:
```python
eligible = self._beam._count_unconsolidated_before(cutoff)
```
This counts across ALL sessions. Consider scoping it to the current session only. This is a secondary fix, not blocking the PR.
