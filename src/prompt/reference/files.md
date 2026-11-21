# files — reference

Acts on your own personal workspace only — never on a managed workspace, a shared folder or anything outside it.

## Arguments

- `operation` — one of `move`, `rename` or `delete`; required.
- `path` — the one file or folder it acts on, relative to your workspace root (`~` is that root). Required, and it must name an existing item.
- `to` — the destination folder of a `move`; created when it is missing.
- `name` — the new name: the whole point of a `rename`, and an optional rename on arrival for a `move`.

## `operation: "move"`

`to` is the destination folder, created when it is missing. A relative path resolves against your workspace root, `~` and `.` are that root, an empty value is its top level, and `..` is refused. An optional `name` renames the item on arrival; without it the current name is kept.

`to: "archive"` is the special destination: the item goes onto today's shelf, `archive/<year>-<month>-<day>/`, keeping its relative path underneath — what you retire goes there, as opposed to what you file away. An item already on the shelf is shelved under its bare name instead of nested again. `archive/` is not a destination by path: only the `archive` spelling shelves.

## `operation: "rename"`

`name` is the new name, in the same folder; it must be a bare name with no folder in it, and it must differ from the current one. An item inside `generated/` or `uploads/` cannot be renamed — those two are temporary, so move it out under the name you want on the way.

## `operation: "delete"`

The item goes away for good — a folder whole with everything inside it, with no bin to recover it from. When part of it could not be removed the call fails and names what is still there, up to 20 entries.

## The two swept folders

`generated/` (your own outputs) and `uploads/` (attachments you were sent) are swept: the sweep removes a file there that no conversation mentions. Both are fine to move or delete from, and neither is a destination — what you want to keep, move into a folder of your own, where the sweep never looks.

## Refusals

- Nothing outside your personal workspace: a path that leaves it is refused.
- The workspace root itself, the root `MEMORY.md`, and the `generated/`, `uploads/` and `archive/` folders as operands — act on what is inside them instead.
- `.git/`, `.compaction/`, `.hyperframes/`, `.secrets/` (at any depth) and the top-level `shared/` are closed: nothing in them is reachable at all.
- Every symlink, in the item's path or in the destination chain — this tool never follows a link.
- A taken destination name, a folder moved inside itself, and a destination that is already the item's own path.
- A destination that is missing and cannot be created.

## Result

`Moved <from> → <to>`, `Renamed <from> → <to>` or `Deleted <path>` — for a folder, `Deleted <path> — the folder and everything inside it`.
