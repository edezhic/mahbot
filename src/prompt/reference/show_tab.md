# show_tab — reference

## Arguments

- `session` — the name you gave `mahbot chrome --session`, exactly as you gave it; the tool applies the CLI's own prefix itself. An empty or missing value raises nothing: the answer asks for the session.

## Which profile

When the machine has several Chrome profiles, pinning the session's profile with `--profile` keeps the reported profile exact. A session left unpinned is reported as the profile that served it, or as unknown when the helper cannot tell.

## The focus, and the tab

This is the product's one browser call that takes the focus. The raise leaves the tab open for the owner: it is taken out of the session's cleanup, so neither the end of the turn, the group collapsing that follows browser steps, nor the session's idle expiry closes it — from then on only the owner closes it. A tab an earlier raise left with the owner is brought forward again by its exact browser handle, never by a loose match.

## Verification

chrome-use reports a raise optimistically, so the raise is checked against the browser itself: the tab in front must be the very tab the session points at, matched by its Chrome tab id, in the focused window. When the evidence cannot settle it — an unanswered, failed or unreadable probe — the answer says the browser did not confirm it, never "raised" and never "not raised". The tab is left open all the same, so look for it before trying again.

## Outcomes

A session that is not running, one with no tab or no page open, a tab that has gone and a window the browser reports as not brought forward all raise nothing and are reported as such, with whatever is known about the profile and page still given. Every ordinary outcome is an answer, not an error.

## Result

The verdict line, then `Profile:`, `Page:` (when the tab has a title) and `Address:`, then an optional note — for instance that the tab may still be closed when the session ends, because chrome-use did not keep it open. Carry the profile, title and address into your message so the owner knows exactly which window and tab to look at.
