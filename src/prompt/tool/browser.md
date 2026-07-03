Headless browser for navigating web pages, interacting with elements, and extracting content. Returns an accessibility-tree snapshot for AI consumption.

## Required parameter
`tab` — a logical session name. Use `"default"` for most browsing. Use different names to keep multiple pages open simultaneously. Same tab = serialized operations on that page.

## Workflow
### 1. Navigate: `open <url>`
Opens a page and automatically returns a compact accessibility snapshot. Read this snapshot carefully — it shows the page structure, interactive elements, and element refs (like `@e1`, `@e2`) that you can use in subsequent commands.

### 2. Understand the page: `snapshot`
Re-scan the page to get fresh element refs. Pass `compact: false` to include empty structural elements, `interactive_only: true` to see only buttons/links/inputs, or `depth: N` to limit tree depth (useful for deeply nested pages).

### 3. Interact: `find` or `click`
Use `find` with locators to click, fill, hover, or focus elements. Use `click` with a CSS selector or ref (`@e1`) for simple clicks. Always take a fresh `snapshot` after navigation before using refs — they become stale on any DOM change.

### 4. Extract content: `get_text`, `get_innertext`, `eval`, or `snapshot`
* `get_text { selector: "body" }` — uses DOM `textContent`, so returns ALL descendant text including content inside `<script>` and `<style>` elements. For visible text only (no script/style), use `get_innertext` or `eval` with `document.querySelector('body').innerText`.
* `get_innertext { selector: "body" }` — uses `innerText()`, returns only visible rendered text. No `<script>`, `<style>`, or hidden content.
* `get_text` or `get_innertext` with a specific selector — extract text from one element.
* `snapshot` — shows page structure (accessibility tree), not full text content. It is naturally compact; use `get_text` for detailed content extraction.
* `eval { js: "..." }` — run arbitrary JavaScript to inspect or extract data. Returns the result serialized as a string. See `## JS eval notes` below.
* `snapshot { compact: false, depth: 5 }` — more detailed accessibility tree.

## JS eval notes
* `const`/`let` declarations are scoped to individual `page.evaluate()` calls — they do NOT cause redeclaration errors across separate `eval` calls.
* `var` declarations and `window.*` assignments DO persist across calls (standard JavaScript behavior).
* For multi-step extraction, use `window.__tmp` namespace or wrap in an IIFE: `(() => { const x = ...; return x; })()`.

## Large content handling
* For files over ~10KB, prefer the `read` tool (local files) or `shell` with `curl` (remote files) — they have no truncation.
* If using browser for large content, use `eval` with chunked extraction: `document.body.innerText.slice(0, 10000)`.
* `get_text` on raw.githubusercontent.com or CDN URLs returns full content but wrapped in page HTML — prefer `read` or `curl` for raw file content.

## `value` vs `text` (critical distinction)
* `value` = the locator search target (CSS selector, button label text, role name). This is what the tool searches for to *find* the element.
* `text` = text to *fill* or *type* into the element. Only used with `action: "fill"` or `"type"`.
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
`click` (click element), `fill` (clears then types, uses `text` param), `type` (appends without clearing, uses `text` param), `hover` (hover over element), `focus` (focus element), `check` (check checkbox/radio button), `uncheck` (uncheck checkbox/radio button), `text` (get element text content — does NOT use the `text` param).

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