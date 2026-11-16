# Fleet census discovery and lock waits

The sleep-time census discovers nonempty `*.db` candidates below the fleet
root. A candidate is counted as a bank only if it can be read and contains a
`conflicts` table. The fixed skip list prunes directories named `backups`,
`cache`, `.cache`, `__pycache__`, `node_modules`, `.git` or `site-packages`.
A directory named `build` is not on that list and is still searched. Directories ending in `.partial` are also pruned when
they are inside a snapshot tree: the selected root resolves to a directory
named `state-snapshots` (including through an alias), or a `state-snapshots`
directory below the root is an ancestor of the `.partial` directory.
A sibling path such as `/fleet/active/data.partial` remains discoverable even
when `/fleet/state-snapshots` exists.
A selected root such as `/fleet/state-snapshots/run` does not prune its
`.partial` children. Discovery otherwise remains layout-independent: a real
bank does not have to be named `mnemosyne.db` or live in a standard directory.

On current source, each SQLite connection is opened read-only with a 50 ms
busy timeout. Three exclusively locked foreign SQLite files therefore do not
incur the former approximately 15 seconds of lock waiting. This timeout
configures SQLite busy-handler sleep for a lock attempt. Separate statements
can each wait again; it is neither an aggregate connection budget nor a strict
wall-clock deadline. Scheduling and filesystem delays can add time. It does
not impose a total census or filesystem-walk deadline. Large trees or many candidates can still cost
significant time.

A readable foreign SQLite file without a `conflicts` table is ignored. An
unreadable candidate cannot safely be classified as foreign: it is reported
under `unreadable` and makes `bound_holds` false. Once the lock is released,
the next census can classify it and measure the fleet again. Do not interpret
an incomplete census as proof that no cross-bank twins exist. Discovery itself
does not descend into directory symlinks below the root, and can silently skip
unlistable directories and candidates whose file metadata cannot be read; these are not recorded under `unreadable`. Together with the
pruned directories, this means `bound_holds=true` describes the discovered,
readable banks, not proof of complete coverage below the selected root.
Banks are read sequentially through separate connections, without a shared
fleet-wide snapshot. Concurrent writes can therefore change the bound between
observations or before the census finishes.

`MNEMOSYNE_FLEET_ROOT` explicitly selects the census root. Narrow it only when
that directory contains the complete intended fleet; otherwise the resulting
report covers less than the deployment. `MNEMOSYNE_FLEET_CENSUS=0` disables the
sleep-time census and therefore also disables that measurement. Neither
setting repairs delayed consolidation work or proves a fleet-wide bound.

These are source guarantees. Existing installed releases must contain the
short-timeout correction to provide them. The regression suite
`tests/test_fleet_conflict_census.py` exercises real foreign and real-bank
locks, recovery after unlocking, read-only access and retained discovery of
unusually located banks.
