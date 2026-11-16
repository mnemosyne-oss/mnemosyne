# Fleet census discovery and lock waits

The sleep-time census discovers nonempty `*.db` candidates below the fleet
root. A candidate is counted as a bank only if it can be read and contains a
`conflicts` table. Cache, backup and package/build directories on the fixed
skip list are pruned. Discovery otherwise remains layout-independent: a real
bank does not have to be named `mnemosyne.db` or live in a standard directory.

On current source, each SQLite connection is opened read-only with a 50 ms
busy timeout. Three exclusively locked foreign SQLite files therefore do not
incur the former approximately 15 seconds of lock waiting. This timeout
bounds SQLite lock waiting per connection; it does not impose a total census
or filesystem-walk deadline. Large trees or many candidates can still cost
significant time.

A readable foreign SQLite file without a `conflicts` table is ignored. An
unreadable candidate cannot safely be classified as foreign: it is reported
under `unreadable` and makes `bound_holds` false. Once the lock is released,
the next census can classify it and measure the fleet again. Do not interpret
an incomplete census as proof that no cross-bank twins exist.

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
