Drives the owner's real, logged-in Chrome on this machine. One `action` per call; `tab` names the page in this run's session and `profile` pins that tab to one Chrome profile.

Snapshot first and act on the refs it prints (`@e1`) while they last — refs die on any navigation or DOM change, so snapshot again or address a changing page by CSS selector. CSS is the most reliable locator; `text` matches rendered text, `role` the accessible name, and `placeholder`, `alt`, `title` and `testid` the HTML attributes.

An answer is a snapshot with refs, page text, an eval value, rows, a PASS/FAIL, an image, or a short confirmation.
