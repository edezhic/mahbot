Drives the owner's real, logged-in Chrome; an answer is a snapshot with refs, page text, an eval value, rows, a PASS/FAIL, an image, or a short confirmation.

Calls on one `tab` are serialized; a second name keeps two pages open.

Refs (`@e1`) belong to the last snapshot on that tab and die on any navigation or DOM change: snapshot again before reusing one, and address a changing page by CSS selector.

Locators decide how a call is written: CSS (`first`, `last`, `nth` with `index`) is the most reliable; `text` matches rendered text only, `exact` narrowing it to the whole string; `role` with `name` matches the accessible name the snapshot prints, so when it finds nothing use the visible text or CSS; `placeholder`/`alt`/`title`/`testid` match HTML attributes the snapshot does not show — read the real one with `eval`.

Trap: in `find`, `value` is what to locate and `text` is what to type into it.

A ⚠ line qualifies the result rather than marking it failed — read it before relying on the answer; a very large answer is cut off and says so, so read a narrower slice.
