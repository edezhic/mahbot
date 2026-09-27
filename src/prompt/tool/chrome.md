Chrome browser automation: navigate web pages, interact with elements, extract content, and capture screenshots for visual inspection. Returns an accessibility-tree snapshot for AI consumption.

## Required parameter
`tab` — a logical session name. Missing or empty uses a per-run session that is released when your run ends. Sessions are mapped to a private per-run namespace derived from the run's identity, so a name here never addresses another run's session — and a run resumed under that identity re-attaches the names it was working with. Use an explicit different name (e.g. "docs") only to keep multiple pages open simultaneously. Same tab = serialized operations on that page.

## Workflow
### 1. Navigate: `open <url>`
Opens a page and automatically returns a compact accessibility snapshot. Read this snapshot carefully — it shows the page structure, interactive elements, and element refs (like `@e1`, `@e2`) that you can use in subsequent commands.

### 2. Understand the page: `snapshot`
Re-scan the page to get fresh element refs. Pass `compact: false` to include empty structural elements, `interactive_only: true` to see only buttons/links/inputs, or `depth: N` to limit tree depth (useful for deeply nested pages).

### 3. Interact: `find`, `click`, `fill`, `type`, `press`
Use `find` with locators to click, fill, hover, or check elements. Use `click` with a CSS selector or ref (`@e1`) for simple clicks. Always take a fresh `snapshot` after navigation before using refs — they become stale on any DOM change.

Text input — never set `.value` via eval; use the native actions (they route through React/Vue/Angular setters and rich-editor APIs like CodeMirror/Monaco/ProseMirror):
* `fill { selector: "input#email", text: "user@example.com" }` — clear the field and write the text; the written value is read back and verified, so success means the field really holds it. The default choice, including multiline/large text.
* `type { selector: "#search", text: "hello", key_events: true }` — type character-by-character WITHOUT clearing (appends to existing content). `key_events: true` sends real per-character keyDown/keyUp for autocomplete/combobox fields. Embedded newlines press Enter (which may submit a form) — use `fill` for multiline text. A ⚠ warning in the output means the page rewrote or filtered the typed text.
* `press { key: "Enter", selector: "textarea[name=q]" }` — press a key at the focused element; `selector` tries to focus the element first (use it when focus may have moved), but focus moves only if it is focusable (input, textarea, button, ...) — for a non-focusable selector (e.g. `body`) focus does not move and the key lands wherever focus currently is. `data.target` shows where the key actually landed (may differ from `selector`, may be absent). Press Enter after filling to submit forms. A ⚠ warning means the key provably had no effect — no key listeners for a listener-dependent key (Enter on a bare input, Escape, arrows in text fields; keys with browser defaults like Enter on a button are never probed), an Enter that landed where nothing is focused, or the key going to an iframe boundary — click the target instead.

### 4. Extract content: `get_text`, `get_innertext`, `eval`, or `snapshot`
* `get_text { selector: "body" }` — uses DOM `textContent`, so returns ALL descendant text including content inside `<script>` and `<style>` elements. For visible text only (no script/style), use `get_innertext` or `eval` with `document.querySelector('body').innerText`.
* `get_innertext { selector: "body" }` — uses `innerText()`, returns only visible rendered text. No `<script>`, `<style>`, or hidden content.
* `get_text` or `get_innertext` with a specific selector — extract text from one element.
* `snapshot` — shows page structure (accessibility tree), not full text content. It is naturally compact; use `get_text` for detailed content extraction.
* `eval { js: "..." }` — run arbitrary JavaScript to inspect or extract data. Returns the result serialized as a string. See `## JS eval notes` below.
* `snapshot { compact: false, depth: 5 }` — more detailed accessibility tree.

### 5. Visually inspect the rendered page: `screenshot`
Capture a screenshot of the current tab and inject it into your conversation as a native image, so you can see exactly how the page is rendered. The screenshot is for your own analysis — you do not need to echo it or reference its path. Use it after `open`/`find`/`click` when visual layout, styling, or rendered state matters more than the accessibility tree.

### 6. Wait for a condition: `wait`
* `wait { selector: "#results" }` — wait until a CSS selector matches something.
* `wait { url: "pattern" }` — wait until the URL matches a pattern.
* `wait { text: "Loaded" }` — wait until this text appears in the page.
Give exactly ONE target. A wait declares ~10 s to chrome-use (`expect` declares ~20 s) — a timeout is reported as an error, never an infinite block.

### 7. Assert a condition: `expect`
A bounded PASS/FAIL assertion — prefer it over open-and-eyeball when you need a definite answer:
* `{ condition: "visible", selector: "#main" }` — also "hidden", "present".
* `{ condition: "count", selector: ".card", op: "==", count: 5 }` — op: == != > < >= <=.
* `{ condition: "text", selector: "h1", predicate: "contains", expected: "Hello" }` — predicate: equals | contains | matches.
* `{ condition: "value", selector: "input", expected: "..." }` — input value.
* `{ condition: "attr", selector: "a", name: "href", predicate: "contains", expected: "/docs" }`.
* `{ condition: "url", predicate: "contains", expected: "dashboard" }`.
PASS means the condition held within the deadline; FAIL reports the observed `actual` and whether it timed out. A FAIL is information, not a retry prompt.

### 8. Structured extraction: `extract`
`extract { schema: {...}, limit: N }` — schema-driven row extraction. The schema has an optional `"rows"` CSS selector key for row lists and a required `"fields"` object. Each field maps to a CSS selector string, or to an object for non-text getters: `{"url": {"sel": "a.link", "get": "@href"}}`. Getter is `"text"` (default), `"html"`, `"value"`, or `"@<attribute>"` — the `"@"` prefix is REQUIRED for attributes (get `"@href"`, not `"href"`); inside an extract schema `"@"` always introduces an attribute, unrelated to the `@e1` element refs other actions use. Unknown getters are rejected with a corrective error rather than silently returning text. Example: `{"rows": ".card", "fields": {"title": ".title", "price": ".price"}}`. An empty region is reported honestly (0 rows) instead of returning phantom rows; a failing rows-selector count (e.g. invalid CSS) is an error. `limit` trims the returned rows while `total` keeps the honest full count.

## JS eval notes
* `const`/`let` declarations are scoped to individual `page.evaluate()` calls — they do NOT cause redeclaration errors across separate `eval` calls.
* `var` declarations and `window.*` assignments DO persist across calls (standard JavaScript behavior).
* For multi-step extraction, use `window.__tmp` namespace or wrap in an IIFE: `(() => { const x = ...; return x; })()`.

## Large content handling
* For files over ~10KB, prefer the `read` tool (local files) or `shell` with `curl` (remote files) — they have no truncation.
* If using chrome for large content, use `eval` with chunked extraction: `document.body.innerText.slice(0, 10000)`.
* `get_text` on raw.githubusercontent.com or CDN URLs returns full content but wrapped in page HTML — prefer `read` or `curl` for raw file content.

## `value` vs `text` (critical distinction)
* `value` = the locator search target (CSS selector, button label text, role name). This is what the tool searches for to *find* the element.
* `text` = text to *fill* into the element. Only used with `action: "fill"`.
* **Do NOT swap them** — they serve different purposes.

## Locator types for `find` (ranked by reliability)

### CSS selector locators (`first`, `last`, `nth`) — most reliable
`value` is a CSS selector. Always matches the DOM directly. For `nth` you must also pass `index` (zero-based).
* Use `"input"`, `"textarea"`, `"form input"` for text inputs — role-based textbox locators are unreliable.
* **Important**: CSS selectors must match the actual DOM attributes. If a link's `href` is `connection/index.html` (relative), use `a[href$='connection/index.html']` — not an absolute path.
* Tag names and attribute values are case-sensitive.

### `by: "text"` — second most reliable
Matches visible DOM text content (case-sensitive substring). Use `exact: true` for exact (non-substring) match.
* Does NOT match `aria-label`, `alt`, `placeholder`, or `title` attributes.
* Only matches text visible in the DOM, not hidden/spanned text.
* **Look at the snapshot output to see what visible text is available** before using this locator.

### `by: "role"` — use with caution
Matches by ARIA role (button, link, heading, etc.). Pass `name` to filter by computed accessible name.
* The `name` filter can fail even when the snapshot shows a matching element. When it fails, fall back to `by: "text"` or `by: "first"` with CSS.
* Avoid for textboxes/inputs — use `by: "first"` with CSS instead.

### `by: "placeholder"` — matches HTML placeholder attribute exactly
The snapshot shows the *accessible name* (from `aria-label`), not the `placeholder` attribute. These are often different. Use `eval` with `document.querySelector('selector').getAttribute('placeholder')` to find the real placeholder value.

### `by: "label"`, `by: "testid"`, `by: "alt"`, `by: "title"` — special-purpose
* `label` — matches `<label for='...'>` elements only (not `aria-label`).
* `testid` — matches `data-testid` attribute.
* `alt` — matches HTML `alt` attribute.
* `title` — matches HTML `title` attribute (exact match).

## Valid `find` actions
`click` (click element), `fill` (clears then types, uses `text` param), `hover` (hover over element), `check` (check checkbox/radio button), `text` (get element text content — does NOT use the `text` param). (`focus`, `type`, and `uncheck` are not supported by the runtime.)

## Keyboard shortcuts
* `press { key: "Enter" }` — submit forms after filling inputs.
* `press { key: "Escape" }` — close modals, dialogs, or search overlays.
* `press { key: "/" }` — open search on sites with keyboard-triggered search overlays (documentation, help centers, etc.).
* `press { key: "Tab" }` — move focus to the next focusable element.
* `press { key: "Control+a" }` — select all text in a focused input.
* `press { key: "ArrowDown" }` / `"ArrowUp"` — navigate lists/dropdowns.

## Debugging when selectors fail
When `find` or `click` can't locate an element, use `eval` to inspect the DOM:
* `eval { js: "document.querySelector('selector')?.outerHTML" }` — see the actual element type, attributes, and text.
* `eval { js: "document.querySelectorAll('selector').length" }` — check how many elements match; maybe the wrong one is being targeted.
* `eval { js: "document.querySelector('a')?.href" }` — check actual link URLs (often relative, not absolute as you might expect).
* `eval { js: "document.querySelector('input')?.getAttribute('placeholder')" }` — read the actual placeholder attribute value.

## Limitations
* Refs (`@e1`, `@e2`, …) from a snapshot become stale after any navigation or DOM change — take a fresh snapshot before using them.
* `open` returns a compact snapshot automatically; use a separate `snapshot` call for more detail or different options.
* Elements outside the viewport may not be interactable. If interaction fails, try a `find` with a CSS selector first, or use `eval` to check position.
* Elements with `tabindex="-1"` may not be clickable via role-based locators but can still be found with `by: "first"` CSS selector.
* The snapshot's accessible names may differ from HTML attributes (`aria-label` vs `placeholder`, etc.) — use `eval` to inspect actual attributes.

## Bounds, leftovers and the real browser

* Every call runs to a clock chrome-use itself works to, and the tool's own kill rides above it (plus chrome-use's relay-recovery window and 2 s slack) — chrome-use always gives up first, so its own reason, never a synthetic timeout, is what you normally see. `wait` (~10 s) and `expect` (~20 s) are the verbs that declare a deadline chrome-use honours in full; every other action, `open`'s navigation included, runs to the clock the product declares to chrome-use — the tool's own client tolerance (45 s) less the product's 2 s margin — because a declaration AT that tolerance makes the tool run out of tolerance instead of answering, which then reads as a wedged session (the session is stopped and its open tabs are lost). A timeout is information: with a healthy daemon it usually means a slow page, so retry once before changing approach.
* `mahbot chrome`, the browser command you can run through the shell, works to the same clocks: its `wait`/`expect` default to 8 s, and its `--timeout` is refused (rc 3) at or above chrome-use's 45 s tolerance on the verbs that forward a deadline (`wait`, `expect`, `open` — for `open` the declared deadline is what remains of that flag when the `--expect` wait starts, so a value just under the tolerance still declares less). On the verbs chrome-use takes no per-call deadline for (`count`, `eval`, `extract`, `click`, `fill`, `type`, `press`) the declared clock is the smallest accepted `--timeout` and changes nothing; a value above it raises only mahbot's own kill — the bound `timeout_ms` reports — and anything below it is refused as a usage error rather than silently ignored. Its `--session` names are validated rather than reshaped: a name takes letters, digits, '-' or '_' only (chrome-use's own alphabet — a '.' is refused, because chrome-use will not stop a session whose name has one), and anything else is refused (rc 3) — unlike this tool's `tab`, which is sanitized into a per-run session.
* A leftover note reports chrome-use's own helper still holding the call's output channel. A FINISHED call carries it — its exit status and output are the result — and a FAILING call carries it too: the failure is the failure, and the leftover is a separate process chrome-use deliberately leaves running. The helper is named, left RUNNING, and stopped with the `session stop` that addresses it; no further output from it is collected — mahbot holds at most the 16 most recent such channels, releasing the oldest first, so a helper that writes after its channel is released can end on the broken pipe. The helper is chrome-use's own and also stops itself on chrome-use's idle timeout.
* The tool only drives the owner's real, logged-in browser: before an action it establishes that a real browser is reachable through the relay, and recovers the connection if it is not. A call that ran in a browser chrome-use launched itself is a plain FAILURE, never a quiet success — retry once the connection is back, and read it as an environment fault, not as page state.
* A session that stopped answering is recovered automatically on every path (this tool, the `mahbot chrome` CLI, the watchdog, the ended-run release) — you do not stop and recreate a wedged session yourself.

## chrome-use troubleshooting

When the `chrome` tool fails, its error names the classified cause plus what auto-recovery does on its own. The chrome-use binary lives where its own installer puts it — on macOS and Linux the system-wide programs directory when that directory holds a copy or can be written, otherwise the owner's own per-user programs directory; on Windows the helper's own per-user programs directory under `%LOCALAPPDATA%`, with no system-wide step — and mahbot replaces whatever sits there with the newest release on every start of the product. A copy in a directory mahbot cannot write is the copy the agents use, and mahbot records that it could not be brought up to date rather than working around it; failures are non-fatal and retried on the next start. There is no install tool, and the product always runs its own copy itself rather than the one a shell's search path happens to resolve — diagnose and point at fixes, never claim you reinstalled the CLI.

- **NotInstalled** — the chrome-use extension or native host is missing: enable the extension at `chrome://extensions`. A missing CLI is installed again on the next start.
- **HostBroken** — the native host launcher is broken: run `chrome-use doctor` on the copy the product installed, spelling its full path out when the bare name does not resolve on this shell, or check the logs for the last install error.
- **ExtensionDisabled** — the extension is disabled: enable it at `chrome://extensions`.
- **ExtensionAbsent** — the extension is not installed in Chrome: install it from the Chrome Web Store (listing `chrome-use`, ID `knfcmbamhjmaonkfnjhldjedeobeafmk`; `ab-connect` is its internal codename, seen in runtime messages — same product, don't search the Store for "ab-connect").
- **RelayDown** — Chrome is running but the extension is not republishing: check the extension is enabled and installed in Chrome's DEFAULT profile (a non-default profile won't connect).
- **ChromeNotRunning** — mahbot auto-launches Chrome; if that attempt fails, the user starts Chrome manually.
- **UnreachableTab** — the extension lost its debugger attach to the tab the session was driving: close that leftover tab in Chrome.
- **DaemonWedge** — the chrome daemon is down/unresponsive: auto-recovery restarts it, no user action needed.
- **CLI missing or broken** — the binary is installed on every start of the product, so a broken install only self-heals on the next start: check the logs for the install error and let it retry, or offer the user manual control via the installed chrome-use CLI.

A Chrome-side problem (extension disabled/absent, tab unreachable) is only fixed by the user — daemon restarts can't help there. Right after an update the extension version may briefly skew from the CLI's; if the relay stays down just after an update, advise reloading the extension at `chrome://extensions`. The connection is Chrome native messaging — no debug port, no "Allow remote debugging?" popup.