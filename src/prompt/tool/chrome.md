Drives the owner's real, logged-in Chrome; an answer is a snapshot with refs, page text, an eval value, rows, a PASS/FAIL, an image, or a short confirmation.

Calls on one `tab` are serialized; a second name keeps two pages open.

`profile` names the Chrome profile a tab must work in: a display name, a directory such as `Profile 8`, a unique name prefix, an e-mail address, or an id. Omit it on a session that has never named one and the helper keeps its own choice. Once named, the profile binds that tab for the session's life — across the helper's routine recycling and a product restart — and a later call on that tab may omit it. Two tabs of one run may name two different profiles. A profile that is not running is started for you, and the result names the profile that actually served the call. Three refusals are distinct: a tab already working in another profile is a conflict naming the profile it is in (use a new `tab` for a different profile); a name matching nothing or several is a bad argument listing the available profiles as display name and directory, which is how you discover them; a profile with no extension, a disabled one, or one that cannot be started is an environment failure naming the profile.

Refs (`@e1`) belong to the last snapshot on that tab and die on any navigation or DOM change: snapshot again before reusing one, and address a changing page by CSS selector.

Locators decide how a call is written: CSS (`first`, `last`, `nth` with `index`) is the most reliable; `text` matches rendered text only, `exact` narrowing it to the whole string; `role` with `name` matches the accessible name the snapshot prints, so when it finds nothing use the visible text or CSS; `placeholder`/`alt`/`title`/`testid` match HTML attributes the snapshot does not show — read the real one with `eval`.

Trap: in `find`, `value` is what to locate and `text` is what to type into it.

A ⚠ line qualifies the result rather than marking it failed — read it before relying on the answer; a very large answer is cut off and says so, so read a narrower slice.
