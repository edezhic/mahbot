# Live-WAL/tshm artifacts: snapshot queries, prevention, and the restart prohibition

## Background

MahBot's eight stores (`board`, `chat_history`, `config`, `logs`, `sessions`,
`stats`, `users`, `workspaces`) live under `~/.mahbot/db/*.db` and run under
Limbo's `multiprocess_wal` mode. Each store has three companion files:

| File | Purpose |
|------|---------|
| `*.db-wal` | On-disk WAL frames (written by the daemon; checkpointed every ~5 min) |
| `*.db-tshm` | Limbo shared-memory coordination file (mmap'd; never deleted by the daemon) |
| `*.db-shm` | **Standard-SQLite** shared-memory file — Limbo never creates these; any `-shm` at a live store path is foreign (an external `sqlite3` process, or a stale artifact from one) |

If a foreign standard-SQLite process **deletes or recreates `-wal`/`-shm` files
under the running daemon** (a common "WAL reset" troubleshooting step), the
daemon's WAL file descriptor becomes **orphaned**: it keeps writing to the
unlinked inode and publishing frames through `-tshm`, while the on-disk `-wal`
stays empty. Reads that follow `-tshm` then hit torn-frame errors
(`short read on WAL frame`), and the `-shm`/`-wal` churn can silently break a
store's durable persistence (observed for `logs.db`).

Stock `sqlite3` **cannot open these stores at all** (Limbo-specific DDL:
`USING fts` / `USING backing_btree`). All snapshot queries must go through
`mahbot debug`.

## Snapshot-copy query procedure

Copy the store file set into a temporary HOME's `db/` directory, then run
`mahbot debug` with that HOME. The snapshot may lag live state by up to one
checkpoint interval (5 minutes).

```bash
# 1. Create a temp HOME with the .mahbot layout.
SNAP=$(mktemp -d)
mkdir -p "$SNAP/.mahbot/db"

# 2. Copy the store files — db + wal, and **omit -tshm** (see note below).
#    A single command sequence:
for db in ~/.mahbot/db/*.db; do
  cp "$db"                 "$SNAP/.mahbot/db/"
  cp "$db-wal"             "$SNAP/.mahbot/db/" 2>/dev/null || true
done

# 3. Query the snapshot with a temporary HOME.
HOME="$SNAP" mahbot debug --db sessions "SELECT COUNT(*) FROM messages"

# 4. Clean up.
rm -rf "$SNAP"
```

Notes:

- **Omit the `-tshm` file from the copy.** `mahbot debug` opens read-only; a
  missing `-tshm` makes turso_core degrade to the legacy read-only WAL path,
  reading the on-disk `-wal` + main DB — which is consistent as of the last
  checkpoint (≤5 min lag, exactly the ticket's "one checkpoint window" bound).
  If you instead copy a live `-tshm` that advertises frames for an empty copied
  `-wal` (the orphaned-WAL artifact), the snapshot is itself unreadable and the
  CLI reports the explicit artifact error. Omitting `-tshm` avoids that for
  every store and every window.
- The snapshot is **read-only** for `mahbot debug` (`OpenFlags::ReadOnly`) and
  the CLI never creates `-tshm`/`-wal`/`-shm` files.
- On a live store, `mahbot debug` may fail with the explicit **live instance
  artifact** error when the daemon is actively publishing WAL frames through
  an orphaned fd (see below). That is the expected, actionable behavior —
  query a snapshot in that case.
- **Known limitation — `logs.db`**: the on-disk `logs.db` main file is
  genuinely corrupt (pre-existing since Jul 31, confirmed with
  `sqlite3 quick_check`: `btreeInitPage` errors; byte-identical copy). Table
  data in `logs.db` is therefore **unrecoverable via snapshot** — the copy
  carries the same corruption, and `mahbot debug` reports the explicit
  *database corruption/inconsistency* error on it (never a raw page error,
  and never a misleading "live instance artifact" message, since snapshots
  have no `-tshm`). Recovery requires the (forbidden) restart; until then the
  corruption is surfaced by the Logs-page write-failure banner.

## Prevention rule

- **Never** open live `~/.mahbot/db/*.db` with stock `sqlite3` (it cannot read
  the Limbo schema anyway and its `-shm`/`-wal` handling corrupts the
  multiprocess coordination).
- **Never delete, recreate, rename, or truncate** `-wal`, `-shm`, or `-tshm`
  files while the daemon runs.
- All diagnostic queries go through `mahbot debug` (live stores) or the
  snapshot procedure above.

## Detection guard

The daemon runs a background **wal-guard** task that inspects the store file
set every 60 seconds and warns (visible in the GUI Logs page) when:

- **Orphaned WAL**: a store's `-tshm` advertises live frames (`max_frame > 0`)
  while its on-disk `-wal` is empty. This is the live-instance artifact; it
  fires while the daemon actively writes and is silent in quiet windows right
  after a 5-minute checkpoint (`max_frame` reads 0). It is re-announced every
  10 minutes while persistent.
- **Foreign `-shm`**: a standard-SQLite `-shm` file exists at a live store
  path. This is a secondary signal (healthy stores can carry stale `-shm`
  files) and means an external SQLite process touched the store. It warns once
  per store on appearance only — every store currently carries a stale `-shm`
  that cannot be removed without the forbidden restart window, so periodic
  re-announcement would be permanent log noise.

The guard is detection-only: it never opens, locks, or modifies the files.

> **Note on the live daemon**: the guard task runs inside the daemon process,
> and the binding restart prohibition means the currently running daemon
> (started before this feature landed) does **not** execute it. The guard's
> classification logic is unit-tested with synthetic file states and its
> heuristic was validated live against the real store directory; it becomes
> active the next time the daemon process starts (a step that is outside this
> pipeline's control and may not be performed by any ticket). Until then the
> orphaned-WAL condition is surfaced by `mahbot debug`'s artifact error and,
> for the log store, by the Logs-page write-failure banner.

## The `mahbot debug` live-instance artifact error

When the CLI detects the artifact before opening (tshm advertises frames, on-disk
WAL empty), it fails immediately with:

```
live instance artifact: cannot read '<path>' safely. The running daemon's WAL
file descriptor is orphaned ... Query a snapshot copy instead — see
docs/ops/wal-snapshots.md. Never delete or recreate `-wal`/`-shm`/`-tshm` files
while the daemon runs.
```

Torn-frame errors that race past the pre-check are retried with bounded backoff
(~15 s total). After the retries the CLI re-checks the artifact condition:

- **Artifact confirmed** → the same explicit artifact error above.
- **Artifact absent** → a **database corruption/inconsistency** error that
  distinguishes live stores (a `.tshm` is present; a snapshot copy may still
  read cleanly) from snapshot copies (no `.tshm`; the copied data itself is
  corrupt or was copied mid-checkpoint). Raw engine text (`short read on WAL
  frame`, `Invalid page type: 0`, …) is **never printed** — the retry note and
  both error messages are deliberately sanitized.

`mahbot debug --db all` processes every store and then exits non-zero if any
store failed (the summary counts missing, artifact, and corruption failures),
so scripted diagnostics cannot silently miss a broken store.

## Restart prohibition (binding user directive)

A graceful daemon restart would be the root fix for the orphaned-WAL-fd and
`logs.db` persistence conditions (the exit path checkpoints all stores,
flushing orphaned WAL frames into the main DBs, and the fresh process re-unifies
its WAL fds with the on-disk files). **It is permanently forbidden by user
directive** — no ticket, agent, or operator may restart, reinstall, or
otherwise manipulate the live service. The condition is therefore accepted as
persistent and is mitigated by:

- the wal-guard detection warnings,
- the `mahbot debug` artifact error (fail with the explicit message, never raw
  torn-frame output),
- the log-writer observability surface (the GUI Logs page shows the
  `Log store write failures` banner with the latest insert error),
- the snapshot-copy query procedure above, and
- the prevention rule (stop the foreign `sqlite3` activity that causes the
  condition).
