# send_outcome — reference

## The lookup

`mahbot outcome --id <id>` is the same lookup from a terminal, answered as one line of JSON. The `id` is what every command-line `send` prints in its acknowledgement as `(job <id>)` — except the one-way `guest` address, which prints none because nothing will answer it. An id belonging to anything that was not a command-line handover has no record.

A chain's id is its content: re-issuing an identical chain resumes the run in flight, and the record reports the most recent run of that chain.

## States

- `running` — the handover was accepted and has not finished: either it has not started or it is in flight. A chain that is still running also carries `step`.
- `done` — the addressed turn produced an answer, in `answer`.
- `done_no_answer` — the addressed turn finished with nothing to report. This is a normal `manager` ending, not a failure.
- `failed` — the handover ended without an answer; `reason` says why. The dashboard stopped it, the turn was cancelled, the addressed turn failed, the chain was stopped, or the product stopped mid-flight.
- `unknown` — no record stands under that id: another installation's id, an id that was never a command-line handover, or a record whose retention ran out.

## Body keys

`id`, `state`, then whichever apply: `answer`, `reason`, `shape` when an answer came back (`none`, `json` or `forwarded` — the answer's shape against the schema the `send` asked for), and `step` for a running chain. `answer` is the text the addressed agent produced, after any extraction and before the envelope the conversation receives.

## Retention

A record is kept for 24 hours after it finishes — or after it was created, when it never finished. A lookup after that answers `unknown`.
