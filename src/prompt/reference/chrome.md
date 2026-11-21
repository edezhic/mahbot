# chrome — reference

The tool drives the owner's real, logged-in Chrome. `action` is one tagged object per call, e.g. `{"click": {"selector": "@e1"}}`. Calls on one `tab` are serialized; a second name keeps two pages open.

## Actions

- `open` (`url`) — navigate and return the snapshot. A bare URL string is accepted as `{"open": {"url": …}}`.
- `snapshot` (`interactive_only`, `compact`, `depth`) — accessibility tree with refs. `interactive_only` keeps interactive elements; `compact` drops empty structure and defaults to true; `depth` limits the tree depth.
- `click` (`selector`) — click a ref (`@e1`) or a CSS selector.
- `fill` (`selector`, `text`) — write `text` into a field, replacing its content, and read it back.
- `type` (`selector`, `text`, `key_events`) — append `text` without clearing; a newline presses Enter; `key_events` sends real key events for autocomplete fields.
- `press` (`key`, `selector`) — press a key such as `Enter` or `Escape`, optionally after focusing `selector`; a non-focusable selector leaves the key to land where focus is.
- `get_text` (`selector`) — rendered text of the element.
- `get_innertext` (`selector`) — an `innerText` eval; prefer `get_text`. Aliases: `get_inner_text`, `innertext`.
- `get_url` — the current URL.
- `eval` (`js`) — run JavaScript, no top-level `return`.
- `wait` (`selector`, `url`, `text`) — wait for exactly one of the three.
- `expect` (`condition`, `selector`, `count`, `op`, `predicate`, `name`, `expected`) — assert a condition and answer PASS/FAIL. `condition` is one of `visible`, `hidden`, `present`, `count`, `text`, `value`, `attr`, `url`. `selector` applies to every condition except `url`; `count` is the expected element count; `op` is one of `==`, `!=`, `>`, `<`, `>=`, `<=` (default `==`); `predicate` is `equals`, `contains` or `matches` (default `equals`); `name` is the attribute for `attr`; `expected` is the value, or the pattern for `url`.
- `extract` (`schema`, `limit`) — extract rows with a schema. `schema` is `{"rows": "<css>", "fields": {"<name>": "<css>" | {"sel": "<css>", "get": "text|html|value|@attr", "all": true}}}`: an optional `rows` selector plus a `fields` object, a field naming either a CSS selector or a getter object. `get` is `text` (default), `html`, `value` or `@<attribute>` (the `@` is required for attributes); `all` returns an array of every match. A flat schema with no `fields` is normalized. A `rows` selector matching nothing is reported as empty, not as a failure. `limit` trims the rows and keeps the total count.
- `find` (`by`, `value`, `action`, `text`, `name`, `exact`, `index`) — find an element and act on it. `by` is `text`, `role`, `label` (accessible label), `placeholder`, `alt`, `title`, `testid`, `first`, `last` or `nth`; `value` is what to locate, a CSS selector for `first`, `last` and `nth`; `action` is `click`, `fill`, `hover`, `check` or `text` (which reads the element); `text` is what `fill` types; `name` filters by accessible name for `role`; `exact` requires a whole-string match; `index` is zero-based and applies to `nth`. Trap: `value` is what to locate and `text` is what to type into it.
- `upload` (`selector`, `files`) — attach local files to a file input or its label; `files` are absolute or workspace-relative paths.
- `screenshot` — capture the page as an image.

## Tabs

`tab` names the page in this run's private session; an empty or missing value means the default page. A cap-closed tab is gone until an `open` with an address mints a page.

## Profiles

`profile` names the Chrome profile a tab must work in: a display name, a directory such as `Profile 8`, a unique name prefix, an e-mail address, or an id. Omit it on a session that has never named one and the helper keeps its own choice. Once named, the profile binds that tab for the session's life — across the helper's routine recycling and a product restart — and a later call on that tab may omit it. Two tabs of one run may name two different profiles. A profile that is not running is started for you, and the result names the profile that actually served the call.

Three refusals are distinct:

- a tab already working in another profile is a conflict naming the profile it is in — use a new `tab` for a different profile;
- a name matching nothing or several is a bad argument listing the available profiles as display name and directory, which is how you discover them;
- a profile with no extension, a disabled one, or one that cannot be started is an environment failure naming the profile.

## Refs and locators

`first`, `last` and `nth` with `index` are CSS forms. `text` matches rendered text only, with `exact` narrowing it to the whole string. `role` with `name` matches the accessible name the snapshot prints, so when it finds nothing use the visible text or CSS. `placeholder`, `alt`, `title` and `testid` match HTML attributes the snapshot does not show — read the real one with `eval`.

## Answers

A ⚠ line qualifies the result rather than marking it failed — read it before relying on the answer. A very large answer is cut off and says so, so read a narrower slice.
