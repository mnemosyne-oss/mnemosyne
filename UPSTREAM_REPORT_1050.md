# Provider dies in its constructor under a turn-scoped home — every memory write for that profile is silently lost while the agent reports success

## Summary

`MnemosyneMemoryProvider.__init__` assigns `self._beam = None`. `_beam` is a **property** whose setter routes through `_write_slot()` → `_p1b_current_key()`. That key is a **context-local `ContextVar`**.

When a provider is constructed while a profile's turn-scoped home is set but no binding exists yet for that home, the assignment raises inside the **constructor**:

```
Mnemosyne: no binding for the current home '/…/profiles/markets';
refusing to write through another home's binding.
This session's provider was never initialized for this home.
```

The provider therefore **never reaches `initialize()`**, so it can never create the slot and can never bind. Every subsequent memory write for that profile is lost — **and the agent still reports success.**

This is a regression introduced by the #1050 fail-closed work. The guard itself is correct and should stay; the defect is that construction writes through `_beam` before any binding exists.

## Impact

- **Silent data loss.** The tool call returns success; the fact lands in no store. Verified: after a real turn that reported *"Stored. Fact saved to permanent memory."* the profile's store still held **52 rows with an unchanged mtime from two days earlier**, and the fact was absent from the root store, all six profile stores, and the shared surface.
- **A false success is worse than a visible error** — the caller has no signal that anything went wrong.
- **Latent across every profile.** Only the profile whose home binds last actually trips it; the others escape because an ambient binding already exists for their key. It is an ordering artefact, not a per-profile config fault.
- 9 vendored installs were affected on our fleet (identical copies under separate per-profile venvs).

## Reproduction

```python
import os
os.environ["HERMES_HOME"] = "/…/profiles/markets"
from mnemosyne_hermes import MnemosyneMemoryProvider
MnemosyneMemoryProvider()   # raises RuntimeError: no binding for the current home
```

## Root cause

`integrations/hermes/src/mnemosyne_hermes/__init__.py`, `__init__` (~line 1166):

```python
self._beam: Optional[Any] = None
```

`_beam` is a property (~line 2112) whose setter calls `_write_slot()`, which consults the context-local current key and **raises for an uninitialised home**. The assignment was also **redundant**: every `self._beam` read goes through `_resolve_read_slot()`, which already returns `None` before initialisation. So the line performs a guarded *write* to store a value the read path already defaults to.

## Fix

Drop the construction-time assignment. Keep the fail-closed guard intact for genuine cross-home writes.

```python
def __init__(self):
    # do NOT write `self._beam = None` here — `_beam` is a property whose setter
    # routes through _write_slot() -> _p1b_current_key(), a context-local key.
    self._surface_beam: Optional[Any] = None
```

Read behaviour is unchanged: `_binding_slot()` / `_resolve_read_slot()` return a slot whose `"beam"` default is already `None`, so `self._beam` reads `None` until `initialize()` assigns a real beam.

## Test

`integrations/hermes/test_binding_regression.py` — fails on 0.7.4, passes with the patch. It asserts:

1. the provider constructs under a turn-scoped home;
2. `self._beam` reads `None` before `initialize()` (no `AttributeError`);
3. `initialize()` binds the home;
4. a real `mnemosyne_remember` call **increments the row count in that profile's actual store** and the fact reads back;
5. the fail-closed guard **still blocks** a write from an uninitialised home (i.e. the fix does not weaken #1050).

Check 4 is deliberate: the regression is only real if the row count on disk moves, not if the call returns without raising.

## Deployment note

The fix was propagated by building a wheel from the patched source and installing it into all 9 environments, verifying a single md5 across every copy — deliberately **not** by hand-editing installed files, which a reinstall would silently clobber.

## Related concern

`_init_error` is set in some failure paths, and callers appear to treat "provider unavailable" as a soft no-op. Combined with a tool call that still returns success, any future initialization failure will again be invisible. It would be worth making an uninitialised provider **fail loudly** — or at minimum have the tool surface the reason — so a silent memory write can never again be reported as a success.