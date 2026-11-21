# analyze — reference

Analysts are read-only research agents: they read and search but cannot edit files. Dispatch them when you need an answer that reading the workspace yourself would take too long to assemble.

## Arguments

- `message` — the question, or the part of the workspace to investigate.
- `depth` — `quick` or `deep`, where your schema offers both.
- `place` — another folder for this round, where your schema accepts it: the exact, case-sensitive name of a managed workspace, or an absolute path to a directory. An absolute path that is a managed workspace is treated as that workspace; the filesystem root is refused. Omit it to run in your own workspace. Only a `quick` round honours it — `deep` ignores it.
- `action` — `cancel`, where your schema offers it, to stop a round you dispatched instead of starting one; carries no dispatch argument.
- `job_id` — the round to cancel, taken from its dispatch acknowledgement.

## quick

One round of two decorrelated analysts. Each reports structured, claim-level findings; the results are merged with agreement brackets, and contradictory claims get a fresh-analyst verification pass appended as a `## Verification` section. A focused question one batch can answer directly.

## deep

A long multi-round investigation that can take hours. It decomposes the question into sub-questions, runs rounds of fresh analysts over the remaining gaps, and delivers a compact, source-cited answer with the unresolved questions named, plus a detail file holding the full findings. Because it can take hours, clarify the scope with the user and confirm it explicitly before dispatching.

## Asynchronous delivery

An asynchronous call returns immediately without findings. The result arrives later as an injected follow-up wrapped in an `<analyze-tool-result>` envelope whose body starts with a `Job: <id>` line matching the dispatch acknowledgement.

## Cancelling

`action: "cancel"` with `job_id` stops a quick round instead of starting one. Its analysts stop, its partial findings never become a result, and the round is never delivered — cancel as soon as the answer is no longer needed. An id that is not one of your own dispatched quick analyses is refused and left untouched. A `deep` round cannot be cancelled this way.

## Result

A synchronous round returns the merged findings in the tool answer; an asynchronous one returns only the acknowledgement and delivers the findings later.
