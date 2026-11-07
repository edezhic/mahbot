Arrange the files in your personal workspace: move a file or folder into a folder of your choosing, rename it in place, or delete it for good. `operation` is required and is one of `move`, `rename`, `delete`; `path` is the one file or folder it acts on.

- `move` — `to` names the destination folder, created when it is missing; `~`, `.` or an empty value is the top of your workspace. An optional `name` gives the item a new name on arrival. The special destination `archive` puts the item on today's shelf, `archive/<year>-<month>-<day>/`, keeping its relative path underneath — that is where what you retire goes, as opposed to what you file away.
- `rename` — `name` is the new name, in the same folder. Not inside `generated/` or `uploads/`: those two are temporary, so move the item out and give it the name you want on the way.
- `delete` — the item goes away for good, a folder whole with everything inside it, with no bin to recover it from. When part of it could not be removed, the answer names what is still there.

`generated/` (your own outputs) and `uploads/` (attachments you were sent) are swept: the sweep removes a file there that no conversation mentions. Both are fine to move or delete from, and neither is a destination — what you want to keep, move into a folder of your own, where the sweep never looks.

Nothing outside your personal workspace is reachable, and a taken destination name is a refusal rather than a replacement: choose another. The workspace root, the root `MEMORY.md`, the `generated/`, `uploads/` and `archive/` folders themselves, `.git/`, `.compaction/`, `.hyperframes/`, `shared/`, `.secrets/` and every symlink are refused. You can act on what is inside those folders, except the closed ones, where nothing is reachable at all.
