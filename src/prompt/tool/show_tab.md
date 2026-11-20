Raise the browser tab you are working in, in front of the owner's other windows, and report where it is: the Chrome profile it lives in, the page title when it has one, and the address.

Call it only when the owner must look at or act in that page — something they have to read, sign in to, approve or finish. Your own convenience is not a reason: you already read the page through `mahbot chrome`, and taking the focus interrupts whatever they were doing.

`session` is exactly the name you gave `mahbot chrome --session` when the work started; the tool applies the CLI's own prefix itself. When the machine has several profiles, pinning that session's profile with `--profile` keeps the reported profile exact; a session left unpinned is reported as the profile that served it, or as unknown when the helper cannot tell.

This is the product's one browser call that takes the focus, and a raise leaves the tab open for the owner: it is taken out of the session's cleanup, so neither the end of the turn, the group collapsing that follows browser steps, nor the session's idle expiry closes it — from then on only the owner closes it. Raise only a page the owner must have.

The answer says plainly what happened. A session that is not running, one with no tab or no page open, a tab that has gone and a window the browser reports as not brought forward all raise nothing and are reported as such, with whatever is known about the profile and page still given. A raise the browser cannot confirm is never passed off as raised — and that tab is left open for the owner all the same, so look for it before trying again. Carry the profile, title and address into your message so the owner knows exactly which window and tab to look at.
