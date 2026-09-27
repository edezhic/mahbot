//! Single shared source of chrome action descriptions: one table rendered
//! into BOTH the interactive `chrome` tool's LLM-facing parameter schema and
//! the `mahbot chrome` CLI help, so the two consumers cannot drift.

use std::sync::LazyLock;

use serde_json::{Value, json};

use crate::chrome::contract::OutKind;

/// Tool-schema parameter block for one action.
pub(crate) struct ToolParams {
    pub(crate) required: &'static [&'static str],
    pub(crate) properties: Value,
}

/// CLI-help data for one action.
pub(crate) struct CliHelp {
    /// Invocation syntax after the action word, e.g. `<url> [--expect <sel>]`.
    pub(crate) syntax: &'static str,
    /// Per-flag one-liners (flag incl. value placeholder → description).
    pub(crate) flags: &'static [(&'static str, &'static str)],
    /// Whether the global `--session <name>` flag is accepted (renderer adds
    /// its standard row).
    pub(crate) session: bool,
    /// Kinds this action's dispatch can emit (`Ok` first) — rendered into
    /// per-action CLI help and enforced by the CLI's `out_env` debug assertion.
    pub(crate) kinds: &'static [OutKind],
    /// Prose after the shared purpose line.
    pub(crate) details: &'static str,
    /// Shell invocation examples (1-2 per action).
    pub(crate) examples: &'static [&'static str],
}

pub(crate) struct ActionDesc {
    pub(crate) name: &'static str,
    /// One-line purpose — the tool's schema description AND the CLI's
    /// action-list line, from one source.
    pub(crate) purpose: &'static str,
    pub(crate) tool: Option<ToolParams>,
    pub(crate) cli: Option<CliHelp>,
}

/// The `--timeout` description shared by the verbs that forward no `--timeout`
/// chrome-use honours (click, count, eval, extract, fill, type, press): such a
/// call's clock is the one mahbot declares to chrome-use — chrome-use's own
/// client tolerance less mahbot's 2 s margin — and the product's kill
/// rides the relay-recovery window + slack above it — mahbot never cuts the call
/// off before that clock, so chrome-use's own reason, not a synthetic mahbot
/// timeout, is what the caller sees. chrome-use takes no per-call deadline for
/// these verbs, so the product cannot make the call give up earlier: a `--timeout`
/// below that declared clock is REFUSED (rc 3) instead of accepted and silently
/// discarded. A larger one widens the bound mahbot itself kills at — never the
/// clock chrome-use works to, which nothing can raise (declaring a larger deadline
/// to chrome-use would make it run out of its own client tolerance instead of
/// reporting its own reason), so it is a bound the call really runs to. Only
/// `wait`/`expect` forward a deadline chrome-use honours in full, so only there
/// does the `--timeout` become the call's own clock.
const OWN_BOUND_FLAG: &str = "your own bound on the step (default: none — the clock mahbot \
     declares to chrome-use is the call's clock and the product never cuts the call off before it; \
     a value above that clock raises only the bound mahbot itself kills at, by the relay-recovery \
     window and slack, and anything below it is refused as a usage error because chrome-use takes \
     no per-call deadline for this verb — a --timeout is honoured in full by wait/expect only)";

/// The `--timeout` description for the verbs chrome-use honours a `--timeout`
/// on (wait, expect): the declared value IS the chrome-use-side deadline, so
/// chrome-use's own timeout surfaces at it and the product only kills above it
/// plus the relay-recovery window. A value at or above chrome-use's own client
/// tolerance is refused as a usage error. `open` derives the wait's declaration
/// from what REMAINS of the flag when the wait starts, so a larger `--timeout`
/// there never pushes the declaration past the tolerance this parse layer
/// refuses.
const FORWARDED_DEADLINE_FLAG: &str = "condition deadline in seconds (default 8) — chrome-use \
     honours it in full and gives up first, so its own 'condition was not met' verdict is what \
     you see; the product's kill rides above it plus chrome-use's relay-recovery window. Must \
     stay under 45, chrome-use's own client tolerance — a value at or above it is refused, \
     because the tool would run out of tolerance instead of answering";

pub(crate) static ACTIONS: LazyLock<Vec<ActionDesc>> = LazyLock::new(build_actions);

/// Build the shared action registry (registry order = CLI help order).
#[expect(clippy::too_many_lines)]
fn build_actions() -> Vec<ActionDesc> {
    vec![
        ActionDesc {
            name: "status",
            purpose: "report what was established about the owner's real browser",
            tool: None,
            cli: Some(CliHelp {
                syntax: "",
                flags: &[],
                session: false,
                kinds: &[OutKind::Ok, OutKind::Environment, OutKind::Usage],
                details: "Pure preflight — reports what was established (chrome-use, the native host, the extension relay, the real profile being driven, a real browser reachable through the relay, a running Chrome, a usable display), never a file/process health check. Exits 2 (environment) only for a chrome-use that cannot state its version or a fact that rules the connection out; a not-proven report exits 0 with `verdict: not-proven` and an action is still attempted. Never launches Chrome, never recovers, never mutates the environment. Rejects --session.",
                examples: &["mahbot chrome status"],
            }),
        },
        ActionDesc {
            name: "open",
            purpose: "Navigate to a URL (returns page content automatically)",
            tool: Some(ToolParams {
                required: &["url"],
                properties: json!({
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "<url> [--expect <sel>] [--structural] [--timeout <secs>]",
                flags: &[
                    ("--expect <sel>", "wait for this selector after navigation"),
                    (
                        "--structural",
                        "with --expect: a wait timeout is classified as redesign (suspected DOM redesign) instead of timeout",
                    ),
                    (
                        "--timeout <secs>",
                        "the condition deadline for the --expect wait, in seconds (default 20) — \
                         the wait declares what REMAINS of it when it starts, and chrome-use \
                         honours that in full and gives up first, so its own verdict is the \
                         open's; the error-page probe and the settle/content capture are \
                         best-effort and product-bounded; the navigation runs to that same declared \
                         clock, which no value here shortens, so the open may \
                         outlive it; a value at or above chrome-use's own client tolerance (45) \
                         is refused as a usage error, because the tool would then run out of \
                         tolerance instead of answering",
                    ),
                ],
                session: true,
                kinds: &[
                    OutKind::Ok,
                    OutKind::Network,
                    OutKind::Timeout,
                    OutKind::Redesign,
                    OutKind::NotFound,
                    OutKind::Error,
                    OutKind::Environment,
                    OutKind::Usage,
                ],
                details: "The URL must be http(s). Reports the committed final URL plus the page content — a best-effort compact accessibility snapshot, truncated at ~5 KB and absent when the capture fails, the step budget is exhausted, or the page is content-free. An uncommitted navigation (tab still on about:blank) or a Chrome error page is kind network with the requested url; a rendered net error code (e.g. DNS_PROBE_FINISHED_NXDOMAIN, ERR_CONNECTION_REFUSED) surfaces as error_code plus a specific cause in error. An invalid URL is kind usage (rc 3). After navigation (plain path, no --expect) a best-effort network settle runs — capped at ~10s and skipped when the remaining budget cannot also cover content capture — so heavy SPAs (Gmail, YouTube) have largely settled before the next step; a settle timeout never fails the open. The settle reduces, not eliminates, first-step lag: the first count/eval after open can still be slow under contention, but such a step now runs to the clock mahbot declares to chrome-use and reports chrome-use's own reason — pass --expect to have the open itself wait for the page instead. `--expect` is a wait-for-selector convenience after navigation, not a general assertion — use the expect action for condition checks. The `--timeout` deadline bounds the `--expect` wait, which declares to chrome-use what REMAINS of it at the moment it starts — a declaration chrome-use honours in full and gives up first on, so its verdict is the open's — and bounds the product's own best-effort work (the error-page probe, the settle, content capture). A `--timeout` at or above chrome-use's own client tolerance (45s) is refused (rc 3). It does NOT bound the navigation, which runs to the clock mahbot declares to chrome-use like every verb chrome-use accepts no `--timeout` for, so the open may outlive the declared deadline.",
                examples: &[
                    "mahbot chrome open https://example.com",
                    "mahbot chrome open https://example.com --expect \"#main\" --structural --timeout 15",
                ],
            }),
        },
        ActionDesc {
            name: "count",
            purpose: "count elements matching a CSS selector (eval shim over querySelectorAll)",
            tool: None,
            cli: Some(CliHelp {
                syntax: "<selector> [--timeout <secs>]",
                flags: &[("--timeout <secs>", OWN_BOUND_FLAG)],
                session: true,
                kinds: &[
                    OutKind::Ok,
                    OutKind::Empty,
                    OutKind::Timeout,
                    OutKind::Network,
                    OutKind::NotFound,
                    OutKind::Error,
                    OutKind::Environment,
                    OutKind::Usage,
                ],
                details: "A count of 0 reports kind empty (rc 0) — a legitimately empty region, not a failure.",
                examples: &[
                    "mahbot chrome count \".card\"",
                    "mahbot chrome count \"a[href]\" --session docs",
                ],
            }),
        },
        ActionDesc {
            name: "wait",
            purpose: "wait until a CSS selector matches, the URL matches a pattern, or text appears",
            tool: Some(ToolParams {
                required: &[],
                properties: json!({
                    "selector": {
                        "type": "string",
                        "description": "CSS selector to wait for (give exactly ONE of selector/url/text)"
                    },
                    "url": {
                        "type": "string",
                        "description": "URL pattern to wait for (give exactly ONE of selector/url/text)"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to wait for in the page (give exactly ONE of selector/url/text)"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "(<selector> | --url <pattern> | --text <text>) [--timeout <secs>]",
                flags: &[
                    ("--url <pattern>", "wait until the URL matches this pattern"),
                    ("--text <text>", "wait until this text appears in the page"),
                    ("--timeout <secs>", FORWARDED_DEADLINE_FLAG),
                ],
                session: true,
                kinds: &[
                    OutKind::Ok,
                    OutKind::Timeout,
                    OutKind::Network,
                    OutKind::NotFound,
                    OutKind::Error,
                    OutKind::Environment,
                    OutKind::Usage,
                ],
                details: "Exactly one target: a selector positional, --url, or --text. A numeric first token (chrome-use's silent-sleep form) is rejected as usage. --timeout IS chrome-use's deadline, honoured in full — chrome-use's own timeout error surfaces at it, and the product kills only above it plus the relay-recovery window. --fn/--load are not exposed (the one internal --load use is open's post-navigation settle).",
                examples: &[
                    "mahbot chrome wait \"#results\" --timeout 15",
                    "mahbot chrome wait --text \"Loaded\" --timeout 10",
                ],
            }),
        },
        ActionDesc {
            name: "expect",
            purpose: "assert a page condition (visible/hidden/present/count/text/value/attr/url) with a bounded wait",
            tool: Some(ToolParams {
                required: &["condition"],
                properties: json!({
                    "condition": {
                        "type": "string",
                        "description": "Condition to assert: visible | hidden | present | count | text | value | attr | url"
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector the condition applies to (required for all conditions except url)"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Expected element count (condition 'count')"
                    },
                    "op": {
                        "type": "string",
                        "description": "Comparison for condition 'count': == != > < >= <= (default ==)"
                    },
                    "predicate": {
                        "type": "string",
                        "description": "Comparison for text/value/attr/url: equals | contains | matches (default equals)"
                    },
                    "name": {
                        "type": "string",
                        "description": "Attribute name for condition 'attr'"
                    },
                    "expected": {
                        "type": "string",
                        "description": "Expected value for text/value/attr; the URL pattern for url"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "(<selector> <visible|hidden|present> | count <sel> <op> <n> | text|value <sel> <equals|contains|matches> <value> | attr <sel> <name> <pred> <value> | url <pred> <pattern>) [--timeout <secs>]",
                flags: &[("--timeout <secs>", FORWARDED_DEADLINE_FLAG)],
                session: true,
                kinds: &[
                    OutKind::Ok,
                    OutKind::Timeout,
                    OutKind::Network,
                    OutKind::NotFound,
                    OutKind::Error,
                    OutKind::Environment,
                    OutKind::Usage,
                ],
                details: "rc 0 when the condition holds. rc 1 when it does not: kind timeout when the condition never held within the deadline (chrome-use reports every failed expect as a deadline expiration), kind error for other step failures. The allowlist is the safe subset of chrome-use's grammar — gone/request/no-errors and the --not/--regex/--no-wait flags are not exposed.",
                examples: &[
                    "mahbot chrome expect \"#main\" visible",
                    "mahbot chrome expect count \".card\" \"==\" 0",
                    "mahbot chrome expect url contains \"dashboard\"",
                ],
            }),
        },
        ActionDesc {
            name: "eval",
            purpose: "Run JavaScript in the page context. Use to inspect element attributes, check state, or debug.",
            tool: Some(ToolParams {
                required: &["js"],
                properties: json!({
                    "js": {
                        "type": "string",
                        "description": "JavaScript to run in the page context"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "<js> [--timeout <secs>]",
                flags: &[("--timeout <secs>", OWN_BOUND_FLAG)],
                session: true,
                kinds: &[
                    OutKind::Ok,
                    OutKind::Timeout,
                    OutKind::Network,
                    OutKind::NotFound,
                    OutKind::Error,
                    OutKind::Environment,
                    OutKind::Usage,
                ],
                details: "The result is emitted as a JSON value under the result key.",
                examples: &["mahbot chrome eval 'document.title'"],
            }),
        },
        ActionDesc {
            name: "extract",
            purpose: "extract rows from the page with a JSON schema file",
            tool: Some(ToolParams {
                required: &["schema"],
                properties: json!({
                    "schema": {
                        "type": "object",
                        "description": "Extraction schema: an optional \"rows\" CSS selector key for row-list extraction plus a required \"fields\" object. Each field maps to a CSS selector string or {\"sel\": \"<css>\", \"get\": \"<getter>\", \"all\": true}; getter is \"text\" (default), \"html\", \"value\", or \"@<attribute>\" — attributes REQUIRE the \"@\" prefix (e.g. \"@href\"; a bare attribute name is rejected mahbot-side). Example: {\"rows\": \".card\", \"fields\": {\"title\": \".title\", \"url\": {\"sel\": \"a\", \"get\": \"@href\"}}}"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Trim rows mahbot-side; total still reports the honest count"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "--schema-file <path> [--limit <n>] [--timeout <secs>]",
                flags: &[
                    (
                        "--schema-file <path>",
                        "JSON schema file for the extraction (required)",
                    ),
                    (
                        "--limit <n>",
                        "trim rows mahbot-side; total still reports the honest count",
                    ),
                    ("--timeout <secs>", OWN_BOUND_FLAG),
                ],
                session: true,
                kinds: &[
                    OutKind::Ok,
                    OutKind::Empty,
                    OutKind::Timeout,
                    OutKind::Network,
                    OutKind::NotFound,
                    OutKind::Error,
                    OutKind::Environment,
                    OutKind::Usage,
                ],
                details: "The schema follows chrome-use's grammar: an optional \"rows\" selector plus a required \"fields\" object (field name → CSS selector or {sel, get, all}). Field getters: \"text\" (default), \"html\", \"value\", or \"@<attribute>\" (e.g. \"@href\"); attributes require the \"@\" prefix, and unknown getters are rejected kind usage (rc 3) instead of chrome-use's silent textContent fallback. When the schema's rows selector matches 0 elements, the empty region is reported honestly (kind empty, rc 0) without invoking chrome-use's phantom-row extract. An unreadable or invalid schema file is kind usage (rc 3). If the count eval fails (e.g. an invalid CSS rows selector), the action errors explicitly — chrome-use's dominant-container auto-detect fallback is deliberately NOT used, because the gate exists to avoid phantom rows.",
                examples: &["mahbot chrome extract --schema-file products.json --limit 20"],
            }),
        },
        ActionDesc {
            name: "click",
            purpose: "Click an element by ref or CSS selector",
            tool: Some(ToolParams {
                required: &["selector"],
                properties: json!({
                    "selector": {
                        "type": "string",
                        "description": "Element ref (@e1) or CSS selector to click. Refs come from the most recent snapshot on this tab — they become stale after any navigation or re-snapshot"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "<selector> [--if-present] [--timeout <secs>]",
                flags: &[
                    (
                        "--if-present",
                        "a missed click is a no-op success (chrome-use semantics)",
                    ),
                    ("--timeout <secs>", OWN_BOUND_FLAG),
                ],
                session: true,
                kinds: &[
                    OutKind::Ok,
                    OutKind::Timeout,
                    OutKind::Network,
                    OutKind::NotFound,
                    OutKind::Error,
                    OutKind::Environment,
                    OutKind::Usage,
                ],
                details: "The CLI has no snapshot refs — the selector is always CSS.",
                examples: &[
                    "mahbot chrome click \"#submit\"",
                    "mahbot chrome click \".next\" --if-present",
                ],
            }),
        },
        ActionDesc {
            name: "fill",
            purpose: "clear an input/textarea/contenteditable and fill it with text (read-back verified)",
            tool: Some(ToolParams {
                required: &["selector", "text"],
                properties: json!({
                    "selector": {
                        "type": "string",
                        "description": "CSS selector or ref (@e1) of the input/textarea/contenteditable to fill. Refs come from the most recent snapshot on this tab — they become stale after navigation or re-snapshot"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to write — replaces the existing content. Use this for multiline/large text"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "<selector> <text> [--file <path>] [--stdin] [--timeout <secs>]",
                flags: &[
                    (
                        "--file <path>",
                        "read the fill value from a UTF-8 file (large/multiline text — chrome-use reads the file)",
                    ),
                    ("--stdin", "read the fill value from this process's stdin"),
                    ("--timeout <secs>", OWN_BOUND_FLAG),
                ],
                session: true,
                kinds: &[
                    OutKind::Ok,
                    OutKind::Timeout,
                    OutKind::Network,
                    OutKind::NotFound,
                    OutKind::Error,
                    OutKind::Environment,
                    OutKind::Usage,
                ],
                details: "Clears the field and fills it, replacing existing content; the written value is read back and verified before success — rich editors (CodeMirror, Monaco, ProseMirror, contenteditable) and framework inputs (React/Vue/Angular) are handled natively. Exactly one text source: the inline text (multi-word text is joined with spaces), --file <path>, or --stdin. A missing --file is kind usage; a target that does not exist is kind not-found; a read-back verification failure is kind error. A text beginning with '-' is taken as literal text (fill has no single-dash flags), so `fill \"#q\" -tail` fills `-tail`; `--` may still shield one (or use --file/--stdin).",
                examples: &[
                    "mahbot chrome fill \"#email\" \"user@example.com\"",
                    "mahbot chrome fill \".editor\" --file ./post.md",
                ],
            }),
        },
        ActionDesc {
            name: "type",
            purpose: "type text character-by-character into an element (appends, does not clear)",
            tool: Some(ToolParams {
                required: &["selector", "text"],
                properties: json!({
                    "selector": {
                        "type": "string",
                        "description": "CSS selector or ref (@e1) of the input/textarea/contenteditable to type into. Refs come from the most recent snapshot on this tab — they become stale after navigation or re-snapshot"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type character-by-character, appending to existing content. Embedded newlines press Enter (which can submit a form) — use fill for multiline text"
                    },
                    "key_events": {
                        "type": "boolean",
                        "description": "Send real per-character keyDown/keyUp instead of insertText — for autocomplete/combobox fields that only react to key events"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "<selector> <text> [--key-events] [--timeout <secs>]",
                flags: &[
                    (
                        "--key-events",
                        "send real per-character keyDown/keyUp — for autocomplete/combobox fields",
                    ),
                    ("--timeout <secs>", OWN_BOUND_FLAG),
                ],
                session: true,
                kinds: &[
                    OutKind::Ok,
                    OutKind::Timeout,
                    OutKind::Network,
                    OutKind::NotFound,
                    OutKind::Error,
                    OutKind::Environment,
                    OutKind::Usage,
                ],
                details: "Types character-by-character without clearing (appends to existing content). --key-events sends real per-character keyDown/keyUp for autocomplete/combobox fields. Embedded newlines press Enter — they can submit a form; use fill for multiline text. When the page rewrites or filters the typed text, chrome-use's warning makes the action kind error (rc 1) — never a silent success. A text beginning with '-' is taken as literal text (type has no single-dash flags); `--` may still shield one.",
                examples: &[
                    "mahbot chrome type \"#search\" \"hello\"",
                    "mahbot chrome type \"#zip\" \"201-0001\" --key-events",
                ],
            }),
        },
        ActionDesc {
            name: "press",
            purpose: "Press a keyboard key at the current focus (e.g. Enter to submit forms)",
            tool: Some(ToolParams {
                required: &["key"],
                properties: json!({
                    "key": {
                        "type": "string",
                        "description": "Key to press (e.g. Enter, Tab, Escape, Control+a, ArrowDown)"
                    },
                    "selector": {
                        "type": "string",
                        "description": "Try to focus this element (CSS selector or ref @e1) before pressing. Focus moves only if it is focusable (input, textarea, select, button, ...); a non-focusable selector (e.g. body) is a silent no-op and the key lands wherever focus currently is"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "<key> [--selector <sel>] [--hold <ms>] [--timeout <secs>]",
                flags: &[
                    (
                        "--selector <sel>",
                        "try to focus this element before pressing (no-op if not focusable)",
                    ),
                    (
                        "--hold <ms>",
                        "hold the key down for this long before releasing",
                    ),
                    ("--timeout <secs>", OWN_BOUND_FLAG),
                ],
                session: true,
                kinds: &[
                    OutKind::Ok,
                    OutKind::Timeout,
                    OutKind::Network,
                    OutKind::NotFound,
                    OutKind::Error,
                    OutKind::Environment,
                    OutKind::Usage,
                ],
                details: "Sends the key to the focused element. --selector tries to focus its target first (making kind not-found reachable), but focus moves only if the element is focusable (input, textarea, select, button, ...) — for a non-focusable selector (e.g. body) focus does not move and the key lands wherever focus currently is. data.target reports the element where the key actually landed (may differ from --selector, may be absent). For listener-dependent keys (Enter on a bare input, Escape, arrows/PageUp/PageDown in text fields) chrome-use probes for key listeners; with none found, chrome-use's warning makes the action kind error (rc 1) — never a silent success. Keys with browser defaults (e.g. Enter on a button) and command chords are never probed, so they can succeed even if the page ignores them; separately, an Enter that lands where nothing is focused (or on an iframe) also warns and makes the action kind error (rc 1).",
                examples: &[
                    "mahbot chrome press Enter --selector \"textarea[name=q]\"",
                    "mahbot chrome press Escape",
                ],
            }),
        },
        ActionDesc {
            name: "session",
            purpose: "manage named CLI sessions (session stop / session status)",
            tool: None,
            cli: Some(CliHelp {
                syntax: "stop <name> [--force] | status <name>",
                flags: &[(
                    "--force",
                    "also stop protected agent-tab-* / link-enricher-* sessions",
                )],
                session: false,
                kinds: &[
                    OutKind::Ok,
                    OutKind::Empty,
                    OutKind::Timeout,
                    OutKind::Network,
                    OutKind::NotFound,
                    OutKind::Error,
                    OutKind::Environment,
                    OutKind::Usage,
                ],
                details: "Manages named CLI sessions — a `<name>` takes letters, digits, '-' or '_' only (chrome-use's own alphabet: a '.' is refused, since chrome-use refuses to stop such a session). stop refuses protected sessions (agent-tab-* / link-enricher-*) unless --force; names are prefixed mahbot-chrome- unless already prefixed. status is an opt-in liveness probe (worst case ~30s: session-list preflight + a real bounded get-url against the session): a not-running session reports empty (rc 0); a wedged one reports environment (rc 2). The probe only reports — it mutates nothing, so a diagnosis never closes tabs on its own; the recovery itself is automatic everywhere the session is actually driven: every on-demand action verb recovers a session that stopped answering and says so in its error (only when mahbot's own bound ended the step, or chrome-use classified the session unresponsive — a step chrome-use timed out on its own, e.g. an expect whose condition was not met, is not a wedge and costs no session), `session stop <name>` clears it, and the ended-run release and the background watchdog recover one too. After a recovery, re-run the action with `--session <name>` to re-create the session — cookies persist in the profile, open tabs/tab-group identity do not. The global `status` action is session-unaware: it can report the connection ready while a named session is wedged.",
                examples: &[
                    "mahbot chrome session stop docs",
                    "mahbot chrome session status docs",
                ],
            }),
        },
        ActionDesc {
            name: "snapshot",
            purpose: "Get accessibility snapshot with element refs (@e1, @e2, ...)",
            tool: Some(ToolParams {
                required: &[],
                properties: json!({
                    "interactive_only": {
                        "type": "boolean",
                        "description": "Only show interactive elements (buttons, links, inputs)"
                    },
                    "compact": {
                        "type": "boolean",
                        "description": "Remove empty structural elements. Default: true"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Limit tree depth"
                    }
                }),
            }),
            cli: None,
        },
        ActionDesc {
            name: "get_text",
            purpose: "Get text content of an element (uses DOM textContent — includes script/style content)",
            tool: Some(ToolParams {
                required: &["selector"],
                properties: json!({
                    "selector": {
                        "type": "string",
                        "description": "Element ref (@e1) or CSS selector. Refs come from the most recent snapshot — always snapshot before calling get_text with a ref"
                    }
                }),
            }),
            cli: None,
        },
        ActionDesc {
            name: "get_innertext",
            purpose: "Get visible rendered text of an element (uses innerText — no script/style content)",
            tool: Some(ToolParams {
                required: &["selector"],
                properties: json!({
                    "selector": {
                        "type": "string",
                        "description": "Element ref (@e1) or CSS selector. Uses innerText() — returns only visible rendered text, no script/style content"
                    }
                }),
            }),
            cli: None,
        },
        ActionDesc {
            name: "get_url",
            purpose: "Get current URL",
            tool: Some(ToolParams {
                required: &[],
                properties: json!({}),
            }),
            cli: None,
        },
        ActionDesc {
            name: "find",
            purpose: "Find an element by semantic locator and perform an action",
            tool: Some(ToolParams {
                required: &["by", "value", "action"],
                properties: json!({
                    "by": {
                        "type": "string",
                        "description": "Locator type: text (case-sensitive visible text match, second most reliable for buttons/links/headings), role (accessibility tree role, use 'name' field to filter — but name filter can fail even when snapshot shows a match; fall back to 'text' or 'first' if it fails), label (matches <label for='...'> only), placeholder (EXACT match of HTML placeholder attribute — not accessible name shown in snapshot), alt, title (exact HTML title attribute), testid, first (CSS selector — MOST reliable for any element type), last (CSS selector), nth (CSS selector + index). For text inputs: prefer `by: \"first\"` with CSS selector (e.g. `\"input\"`, `\"textarea\"`) — role-based textbox locators are unreliable."
                    },
                    "value": {
                        "type": "string",
                        "description": "Locator match target. For 'text': substring to search for (case-sensitive); for 'placeholder': exact HTML placeholder attribute value (NOT what snapshot shows — check with eval); for 'role': role name ('button', 'link', 'textbox', 'heading'); for 'label': visible <label> text; for 'first'/'last'/'nth': CSS selector (e.g. 'input', 'button', 'form')"
                    },
                    "action": {
                        "type": "string",
                        "description": "Action to perform: click (click element), fill (clear field then type), hover (hover over element), check (check checkbox/radio button), text (get element text content — does NOT use the 'text' param; the 'text' param is only for 'fill'). Press Enter after filling to submit forms."
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to fill into the element (for action 'fill')"
                    },
                    "name": {
                        "type": "string",
                        "description": "Accessible name filter (for role-based finding, e.g. 'Submit'). Note: this filter can fail even when the snapshot shows a matching element. When it fails, retry with `by: \"text\"` or `by: \"first\"` with a CSS selector."
                    },
                    "exact": {
                        "type": "boolean",
                        "description": "Require exact text match"
                    },
                    "index": {
                        "type": "integer",
                        "description": "Zero-based index for `by: \"nth\"`. Required when by is 'nth'."
                    }
                }),
            }),
            cli: None,
        },
        ActionDesc {
            name: "screenshot",
            purpose: "Capture a screenshot of the current page as a PNG and inject it into the conversation as a native image, so you can visually inspect the rendered page",
            tool: Some(ToolParams {
                required: &[],
                properties: json!({}),
            }),
            cli: None,
        },
    ]
}

/// Linear scan for one action's full descriptor.
#[must_use]
pub(crate) fn desc(name: &str) -> Option<&'static ActionDesc> {
    ACTIONS.iter().find(|a| a.name == name)
}

/// Whether `word` is a CLI-dispatchable action (has a `cli` block).
#[must_use]
pub(crate) fn is_cli_action(word: &str) -> bool {
    desc(word).is_some_and(|d| d.cli.is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_action_set(actual: Vec<&str>, expected: &[&str]) {
        let mut a = actual;
        a.sort_unstable();
        let mut e = expected.to_vec();
        e.sort_unstable();
        assert_eq!(a, e);
    }

    #[test]
    fn names_and_purposes_are_valid() {
        let mut seen = std::collections::HashSet::new();
        for a in ACTIONS.iter() {
            assert!(!a.name.is_empty(), "action has an empty name");
            assert!(seen.insert(a.name), "duplicate action name: {}", a.name);
            assert!(
                !a.purpose.is_empty(),
                "{}: purpose must be non-empty",
                a.name
            );
        }
    }

    #[test]
    fn kind_lists_are_wellformed() {
        for a in ACTIONS.iter() {
            let Some(c) = a.cli.as_ref() else {
                continue;
            };
            assert!(!c.kinds.is_empty(), "{}: kinds must be non-empty", a.name);
            assert_eq!(
                c.kinds[0],
                OutKind::Ok,
                "{}: kinds must start with Ok",
                a.name
            );
            for (i, k) in c.kinds.iter().enumerate() {
                assert!(
                    !c.kinds[..i].contains(k),
                    "{}: duplicate kind {}",
                    a.name,
                    k.as_str()
                );
                assert!(!k.as_str().is_empty(), "{}: as_str rendered empty", a.name);
            }
        }
    }

    #[test]
    fn cli_action_names_match_the_cli_dispatch_set() {
        let names: Vec<&str> = ACTIONS
            .iter()
            .filter(|a| a.cli.is_some())
            .map(|a| a.name)
            .collect();
        assert_action_set(
            names,
            &[
                "status", "open", "count", "wait", "expect", "eval", "extract", "click", "fill",
                "type", "press", "session",
            ],
        );
    }

    #[test]
    fn cli_help_entries_are_wellformed() {
        for a in ACTIONS.iter() {
            let Some(c) = a.cli.as_ref() else {
                continue;
            };
            assert!(
                !c.examples.is_empty(),
                "{}: examples must be non-empty",
                a.name
            );
        }
    }
}
