# custom — reference

## Arguments

- `tool` — the name of one custom tool listed in the `<custom-tools>` block.
- `args` — a JSON object of that tool's arguments, keyed by the parameter names the block declares: `{"city": "Minsk", "days": 3}` for a tool declaring `city` and `days`.

## What reaches the script

Declared arguments arrive under the names you used, unchanged — an integral number is normalized (`days: 3.0` arrives as `3`), and an optional argument passed as `null` counts as omitted. Types are the declared ones: string, integer, boolean, or a list of strings. A required argument that is missing, or one of the wrong type, is refused before the script runs.

A key the block does not declare is ignored, not refused: the run still happens and the note `[ignored arguments: …]` is appended to its output or failure. A call refused before the tool is resolved (an unknown tool, no usable definition, no grant) carries no such note.

The script runs with the managed runtime, from your workspace root, never through a shell — nothing in the call can be read as shell syntax. It sees the arguments as one JSON argument, so what the tool declares is what it gets.

## Refusals

- The tool does not exist, or its definition cannot be read (a malformed header is never repaired).
- It is not yours to call — availability is per account or per role, and the `<custom-tools>` block
  lists exactly what you hold.
- A required argument is missing, or a value is of the wrong type.

## Failures

Beyond the refusals, a call fails only when the run cannot complete: the script did not finish in time, ran out of memory, or could not start. A background process it leaves holding its output does not fail the run — the run's own exit status decides.

## Result

The script's own output, with the exit status when it is not clean. Output that outgrows the inline budget is kept whole in your `generated/` folder, where your `read` can open it, and deleted when the run ends.
