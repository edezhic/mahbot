//! Single shared source of chrome action descriptions: one table rendered
//! into BOTH the interactive `chrome` tool's LLM-facing parameter schema and
//! the `mahbot chrome` CLI help, so the two consumers cannot drift. Each CLI
//! action declares its flags once ([`CliFlag`]), and both the help renderer and
//! the parser read that declaration.

use std::sync::LazyLock;

use serde_json::{Value, json};

use crate::chrome::CHROME_USE_DECLARED_BUDGET;
use crate::chrome::contract::OutKind;

/// Tool-schema parameter block for one action.
pub(crate) struct ToolParams {
    pub(crate) required: &'static [&'static str],
    pub(crate) properties: Value,
}

/// One CLI flag an action accepts — the single declaration the rendered help row
/// and the parser both read.
pub(crate) struct CliFlag {
    /// Printed spelling with its value placeholder (`--expect <sel>`), exactly as
    /// the help row shows it.
    pub(crate) spelling: &'static str,
    /// Whether the flag takes a value. Declared here, never derived from
    /// `spelling`: the placeholder and this field disagreeing is drift.
    pub(crate) takes_value: bool,
    /// The printed description of the row.
    pub(crate) description: String,
    /// The `session` subcommand the flag belongs to (`--force` → `Some("stop")`);
    /// `None` for a flag every invocation of the action accepts.
    pub(crate) scope: Option<&'static str>,
}

impl CliFlag {
    /// The machine name the parser matches (`--expect <sel>` → `expect`).
    #[must_use]
    pub(crate) fn name(&self) -> &str {
        self.spelling
            .trim_start_matches('-')
            .split_whitespace()
            .next()
            .unwrap_or_default()
    }
}

/// A flag that takes a value; `spelling` carries its printed placeholder.
fn value_flag(spelling: &'static str, description: impl Into<String>) -> CliFlag {
    CliFlag {
        spelling,
        takes_value: true,
        description: description.into(),
        scope: None,
    }
}

/// A switch (no value).
fn switch_flag(spelling: &'static str, description: impl Into<String>) -> CliFlag {
    CliFlag {
        spelling,
        takes_value: false,
        description: description.into(),
        scope: None,
    }
}

/// CLI-help data for one action.
pub(crate) struct CliHelp {
    /// Invocation syntax after the action word, e.g. `<url> [--expect <sel>]`.
    pub(crate) syntax: &'static str,
    /// This action's own flag rows, in printed order, each carrying the
    /// subcommand it belongs to (see [`CliFlag::scope`]). The session-wide pair
    /// is not here: [`CliHelp::session_refusal`] decides whether [`Self::flags`]
    /// appends it.
    pub(crate) declared: Vec<CliFlag>,
    /// `None` while the action takes the session-wide `--session`/`--keep` pair;
    /// `Some(message)` on the actions that refuse it, with the message the parser
    /// returns. This one field is what the help renders the pair's rows from and
    /// what the parser accepts them by, so the two cannot disagree.
    pub(crate) session_refusal: Option<&'static str>,
    /// Kinds this action's dispatch can emit (`Ok` first) — rendered into
    /// per-action CLI help and enforced by the CLI's `out_env` debug assertion.
    pub(crate) kinds: &'static [OutKind],
    /// Prose after the shared purpose line.
    pub(crate) details: &'static str,
    /// Shell invocation examples (1-2 per action).
    pub(crate) examples: &'static [&'static str],
}

impl CliHelp {
    /// Every flag this action accepts and its help prints, in row order: the
    /// declared rows, then the session-wide pair when the action takes them.
    pub(crate) fn flags(&self) -> impl Iterator<Item = &CliFlag> {
        let session: &[CliFlag] = if self.session_refusal.is_none() {
            &SESSION_FLAGS[..]
        } else {
            &[]
        };
        self.declared.iter().chain(session.iter())
    }
}

/// The session-wide flags: taken (and printed as rows) by every action whose
/// [`CliHelp::session_refusal`] is `None`, refused by the rest. The top-level
/// help words them its own way; these are the per-action rows.
static SESSION_FLAGS: LazyLock<[CliFlag; 2]> = LazyLock::new(|| {
    [
        value_flag(
            "--session <name>",
            "use/name a session — letters, digits, '-' or '_' only (not valid for status / session subcommands)",
        ),
        switch_flag(
            "--keep",
            "keep the session open until 'mahbot chrome session stop <name>' — only with --session, and only on the call that creates the session",
        ),
    ]
});

pub(crate) struct ActionDesc {
    pub(crate) name: &'static str,
    /// One-line purpose — the tool's schema description AND the CLI's
    /// action-list line, from one source.
    pub(crate) purpose: &'static str,
    pub(crate) tool: Option<ToolParams>,
    pub(crate) cli: Option<CliHelp>,
}

/// The `--timeout` row for the eight verbs chrome-use takes no per-call deadline
/// for (click, count, eval, extract, fill, type, press, upload). Their call runs
/// to the clock mahbot declares to chrome-use and the product kills only above
/// it, so a bound below that clock cannot be honoured — it is refused (rc 3)
/// rather than silently discarded — and a larger one widens only mahbot's own
/// kill. The printed floor is [`CHROME_USE_DECLARED_BUDGET`], the same number
/// [`parse_bound_timeout`](crate::chrome::cli) enforces.
fn own_bound_flag() -> String {
    let floor = CHROME_USE_DECLARED_BUDGET.as_secs();
    format!(
        "your own bound on the step (default: none — the clock mahbot declares to chrome-use is the \
         call's clock and the product never cuts the call off before it; a value above that clock \
         raises only the bound mahbot itself kills at, by the relay-recovery window and slack, and \
         anything below it is refused as a usage error because chrome-use takes no per-call \
         deadline for this verb — {floor} is the smallest accepted value; a --timeout is honoured \
         in full by wait/expect only)"
    )
}

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

/// The kinds an action that works through a chrome-use session can end as,
/// `Ok` first as `CliHelp::kinds` requires: `Timeout` (its own deadline ran
/// out), `Network`, `NotFound`, `Error` (the step failed otherwise),
/// `Environment` (which rules it out), `Usage` (a malformed call). A row with
/// a different vocabulary spells its own list out (`open` alone can emit
/// `Redesign`).
const STEP_KINDS: &[OutKind] = &[
    OutKind::Ok,
    OutKind::Timeout,
    OutKind::Network,
    OutKind::NotFound,
    OutKind::Error,
    OutKind::Environment,
    OutKind::Usage,
];

/// [`STEP_KINDS`] plus `Empty`, kept second so `Ok` stays first, for the three
/// actions that can answer with nothing found instead of a failure: a region
/// no element matches, an extraction with no rows, and a `session status`
/// whose session is not running.
const STEP_KINDS_WITH_EMPTY: &[OutKind] = &[
    OutKind::Ok,
    OutKind::Empty,
    OutKind::Timeout,
    OutKind::Network,
    OutKind::NotFound,
    OutKind::Error,
    OutKind::Environment,
    OutKind::Usage,
];

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
                declared: vec![],
                session_refusal: Some("--session is not valid for status"),
                kinds: &[OutKind::Ok, OutKind::Environment, OutKind::Usage],
                details: "Pure preflight — reports what was established (chrome-use, the native host, the extension relay, the real profile being driven, a real browser reachable through the relay, a running Chrome, a usable display), never a file/process health check. Exits 2 (environment) only for a chrome-use that cannot state its version or a fact that rules the connection out; a not-proven report exits 0 with `verdict: not-proven` and an action is still attempted. Never launches Chrome, never recovers, never mutates the environment. Rejects --session and --keep.",
                examples: &["mahbot chrome status"],
            }),
        },
        ActionDesc {
            name: "open",
            purpose: "navigate to a URL (returns the snapshot)",
            tool: Some(ToolParams {
                required: &["url"],
                properties: json!({
                    "url": {
                        "type": "string",
                        "description": "http(s) URL to open"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "<url> [--expect <sel>] [--structural] [--timeout <secs>]",
                declared: vec![
                    value_flag("--expect <sel>", "wait for this selector after navigation"),
                    switch_flag(
                        "--structural",
                        "with --expect: a wait timeout is classified as redesign (suspected DOM redesign) instead of timeout",
                    ),
                    value_flag(
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
                session_refusal: None,
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
                details: "The URL must be http(s). Reports the committed final URL plus the page content — a best-effort compact accessibility snapshot, truncated at ~5 KB and absent when the capture fails, the step budget is exhausted, or the page is content-free. An uncommitted navigation (tab still on about:blank) or a Chrome error page is kind network with the requested url; a rendered net error code (e.g. DNS_PROBE_FINISHED_NXDOMAIN, ERR_CONNECTION_REFUSED) surfaces as error_code plus a specific cause in error. An invalid URL is kind usage (rc 3). After navigation (plain path, no --expect) a best-effort network settle runs — capped at ~10s and skipped when the remaining budget cannot also cover content capture — so heavy SPAs (Gmail, YouTube) have largely settled before the next step; a settle timeout never fails the open. The settle reduces, not eliminates, first-step lag: the first count/eval after open can still be slow under contention, but such a step now runs to the clock mahbot declares to chrome-use and reports chrome-use's own reason — pass --expect to have the open itself wait for the page instead. `--expect` is a wait-for-selector convenience after navigation, not a general assertion — use the expect action for condition checks. The `--timeout` deadline bounds the `--expect` wait, which declares to chrome-use what REMAINS of it at the moment it starts — a declaration chrome-use honours in full and gives up first on, so its verdict is the open's — and bounds the product's own best-effort work (the error-page probe, the settle, content capture). A `--timeout` at or above chrome-use's own client tolerance (45s) is refused (rc 3). It does NOT bound the navigation, which runs to the clock mahbot declares to chrome-use like every verb chrome-use accepts no `--timeout` for, so the open may outlive the declared deadline. A tab this opens in a named session is taken out of chrome-use's tab group before the result is printed — an ordinary tab a person can work in beside the automation — while the session keeps driving and closing it as before; only a tab this very command opened is moved. Telling that tab apart costs the call one extra passive tab listing before the step (a second or two on a cold helper); when the browser will not answer that listing the tab stays in the group and the open still succeeds.",
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
                declared: vec![value_flag("--timeout <secs>", own_bound_flag())],
                session_refusal: None,
                kinds: STEP_KINDS_WITH_EMPTY,
                details: "A count of 0 reports kind empty (rc 0) — a legitimately empty region, not a failure.",
                examples: &[
                    "mahbot chrome count \".card\"",
                    "mahbot chrome count \"a[href]\" --session docs",
                ],
            }),
        },
        ActionDesc {
            name: "wait",
            purpose: "wait for exactly one of selector/url/text",
            tool: Some(ToolParams {
                required: &[],
                properties: json!({
                    "selector": {
                        "type": "string",
                        "description": "CSS selector"
                    },
                    "url": {
                        "type": "string",
                        "description": "URL pattern"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to appear"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "(<selector> | --url <pattern> | --text <text>) [--timeout <secs>]",
                declared: vec![
                    value_flag("--url <pattern>", "wait until the URL matches this pattern"),
                    value_flag("--text <text>", "wait until this text appears in the page"),
                    value_flag("--timeout <secs>", FORWARDED_DEADLINE_FLAG),
                ],
                session_refusal: None,
                kinds: STEP_KINDS,
                details: "Exactly one target: a selector positional, --url, or --text. A numeric first token (chrome-use's silent-sleep form) is rejected as usage. --timeout IS chrome-use's deadline, honoured in full — chrome-use's own timeout error surfaces at it, and the product kills only above it plus the relay-recovery window. --fn/--load are not exposed (the one internal --load use is open's post-navigation settle).",
                examples: &[
                    "mahbot chrome wait \"#results\" --timeout 15",
                    "mahbot chrome wait --text \"Loaded\" --timeout 10",
                ],
            }),
        },
        ActionDesc {
            name: "expect",
            purpose: "assert a condition, PASS/FAIL",
            tool: Some(ToolParams {
                required: &["condition"],
                properties: json!({
                    "condition": {
                        "type": "string",
                        "description": "visible|hidden|present|count|text|value|attr|url"
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector (except url)"
                    },
                    "count": {
                        "type": "integer",
                        "description": "element count"
                    },
                    "op": {
                        "type": "string",
                        "description": "== != > < >= <= (default ==)"
                    },
                    "predicate": {
                        "type": "string",
                        "description": "equals | contains | matches (default equals)"
                    },
                    "name": {
                        "type": "string",
                        "description": "attribute name"
                    },
                    "expected": {
                        "type": "string",
                        "description": "expected value; pattern for url"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "(<selector> <visible|hidden|present> | count <sel> <op> <n> | text|value <sel> <equals|contains|matches> <value> | attr <sel> <name> <pred> <value> | url <pred> <pattern>) [--timeout <secs>]",
                declared: vec![value_flag("--timeout <secs>", FORWARDED_DEADLINE_FLAG)],
                session_refusal: None,
                kinds: STEP_KINDS,
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
            purpose: "run JavaScript",
            tool: Some(ToolParams {
                required: &["js"],
                properties: json!({
                    "js": {
                        "type": "string",
                        "description": "JavaScript, no top-level return"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "<js> [--timeout <secs>]",
                declared: vec![value_flag("--timeout <secs>", own_bound_flag())],
                session_refusal: None,
                kinds: STEP_KINDS,
                details: "The result is emitted as a JSON value under the result key.",
                examples: &["mahbot chrome eval 'document.title'"],
            }),
        },
        ActionDesc {
            name: "extract",
            purpose: "extract rows with a schema",
            tool: Some(ToolParams {
                required: &["schema"],
                properties: json!({
                    "schema": {
                        "type": "object",
                        "description": "{\"rows\": \"<css>\", \"fields\": {\"<name>\": \"<css>\" | {\"sel\": \"<css>\", \"get\": \"text|html|value|@attr\", \"all\": true}}}; \"all\" returns an array of every match; attributes need the \"@\""
                    },
                    "limit": {
                        "type": "integer",
                        "description": "trim rows; total keeps the count"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "--schema-file <path> [--limit <n>] [--timeout <secs>]",
                declared: vec![
                    value_flag(
                        "--schema-file <path>",
                        "JSON schema file for the extraction (required)",
                    ),
                    value_flag(
                        "--limit <n>",
                        "trim rows mahbot-side; total still reports the honest count",
                    ),
                    value_flag("--timeout <secs>", own_bound_flag()),
                ],
                session_refusal: None,
                kinds: STEP_KINDS_WITH_EMPTY,
                details: "The schema follows chrome-use's grammar: an optional \"rows\" selector plus a \"fields\" object (field name → CSS selector or {sel, get, all}); a flat schema with no \"fields\" is normalized instead of refused. Field getters: \"text\" (default), \"html\", \"value\", or \"@<attribute>\" (e.g. \"@href\"); attributes require the \"@\" prefix, and unknown getters are rejected kind usage (rc 3) instead of chrome-use's silent textContent fallback. When the schema's rows selector matches 0 elements, the empty region is reported honestly (kind empty, rc 0) without invoking chrome-use's phantom-row extract. An unreadable or invalid schema file is kind usage (rc 3). If the count eval fails (e.g. an invalid CSS rows selector), the action errors explicitly — chrome-use's dominant-container auto-detect fallback is deliberately NOT used, because the gate exists to avoid phantom rows.",
                examples: &["mahbot chrome extract --schema-file products.json --limit 20"],
            }),
        },
        ActionDesc {
            name: "click",
            purpose: "click an element",
            tool: Some(ToolParams {
                required: &["selector"],
                properties: json!({
                    "selector": {
                        "type": "string",
                        "description": "Ref (@e1) or CSS selector"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "<selector> [--if-present] [--timeout <secs>]",
                declared: vec![
                    switch_flag(
                        "--if-present",
                        "a missed click is a no-op success (chrome-use semantics)",
                    ),
                    value_flag("--timeout <secs>", own_bound_flag()),
                ],
                session_refusal: None,
                kinds: STEP_KINDS,
                details: "The CLI has no snapshot refs — the selector is always CSS. A tab the page opens is taken out of chrome-use's tab group before the result is printed where chrome-use reports it as this session's own; a popup it reports as the owner's tab is left to the browser's own placement.",
                examples: &[
                    "mahbot chrome click \"#submit\"",
                    "mahbot chrome click \".next\" --if-present",
                ],
            }),
        },
        ActionDesc {
            name: "fill",
            purpose: "fill a field (read back)",
            tool: Some(ToolParams {
                required: &["selector", "text"],
                properties: json!({
                    "selector": {
                        "type": "string",
                        "description": "CSS selector or ref (@e1)"
                    },
                    "text": {
                        "type": "string",
                        "description": "text to write, replacing content"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "<selector> <text> [--file <path>] [--stdin] [--timeout <secs>]",
                declared: vec![
                    value_flag(
                        "--file <path>",
                        "read the fill value from a UTF-8 file (large/multiline text — chrome-use reads the file)",
                    ),
                    switch_flag("--stdin", "read the fill value from this process's stdin"),
                    value_flag("--timeout <secs>", own_bound_flag()),
                ],
                session_refusal: None,
                kinds: STEP_KINDS,
                details: "Clears the field and fills it, replacing existing content; the written value is read back and verified before success — rich editors (CodeMirror, Monaco, ProseMirror, contenteditable) and framework inputs (React/Vue/Angular) are handled natively. Exactly one text source: the inline text (multi-word text is joined with spaces), --file <path>, or --stdin. A missing --file is kind usage; a target that does not exist is kind not-found; a read-back verification failure is kind error. A text beginning with '-' is taken as literal text (fill has no single-dash flags), so `fill \"#q\" -tail` fills `-tail`; `--` shields a leading-dash text (or use --file/--stdin), but an unshielded `-h`/`--help` in the tail is still this help page, not text — and chrome-use answers a help-looking value with its own help, so such a fill fails instead of writing it.",
                examples: &[
                    "mahbot chrome fill \"#email\" \"user@example.com\"",
                    "mahbot chrome fill \".editor\" --file ./post.md",
                ],
            }),
        },
        ActionDesc {
            name: "type",
            purpose: "type, appending (no clear)",
            tool: Some(ToolParams {
                required: &["selector", "text"],
                properties: json!({
                    "selector": {
                        "type": "string",
                        "description": "CSS selector or ref (@e1) of the field"
                    },
                    "text": {
                        "type": "string",
                        "description": "text to append (a newline presses Enter)"
                    },
                    "key_events": {
                        "type": "boolean",
                        "description": "send real key events (autocomplete fields)"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "<selector> <text> [--key-events] [--timeout <secs>]",
                declared: vec![
                    switch_flag(
                        "--key-events",
                        "send real per-character keyDown/keyUp — for autocomplete/combobox fields",
                    ),
                    value_flag("--timeout <secs>", own_bound_flag()),
                ],
                session_refusal: None,
                kinds: STEP_KINDS,
                details: "Types character-by-character without clearing (appends to existing content). --key-events sends real per-character keyDown/keyUp for autocomplete/combobox fields. Embedded newlines press Enter — they can submit a form; use fill for multiline text. When the page rewrites or filters the typed text, chrome-use's warning makes the action kind error (rc 1) — never a silent success. A text beginning with '-' is taken as literal text (type has no single-dash flags); `--` shields one, but an unshielded `-h`/`--help` in the tail is still this help page, not text — and chrome-use answers a help-looking value with its own help, so such a type fails instead of typing it.",
                examples: &[
                    "mahbot chrome type \"#search\" \"hello\"",
                    "mahbot chrome type \"#zip\" \"201-0001\" --key-events",
                ],
            }),
        },
        ActionDesc {
            name: "press",
            purpose: "press a key",
            tool: Some(ToolParams {
                required: &["key"],
                properties: json!({
                    "key": {
                        "type": "string",
                        "description": "key to press (e.g. Enter, Escape)"
                    },
                    "selector": {
                        "type": "string",
                        "description": "focus it first (non-focusable: the key lands where focus is)"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "<key> [--selector <sel>] [--hold <ms>] [--timeout <secs>]",
                declared: vec![
                    value_flag(
                        "--selector <sel>",
                        "try to focus this element before pressing (no-op if not focusable)",
                    ),
                    value_flag(
                        "--hold <ms>",
                        "hold the key down for this long before releasing",
                    ),
                    value_flag("--timeout <secs>", own_bound_flag()),
                ],
                session_refusal: None,
                kinds: STEP_KINDS,
                details: "Sends the key to the focused element. --selector tries to focus its target first (making kind not-found reachable), but focus moves only if the element is focusable (input, textarea, select, button, ...) — for a non-focusable selector (e.g. body) focus does not move and the key lands wherever focus currently is. data.target reports the element where the key actually landed (may differ from --selector, may be absent). For listener-dependent keys (Enter on a bare input, Escape, arrows/PageUp/PageDown in text fields) chrome-use probes for key listeners; with none found, chrome-use's warning makes the action kind error (rc 1) — never a silent success. Keys with browser defaults (e.g. Enter on a button) and command chords are never probed, so they can succeed even if the page ignores them; separately, an Enter that lands where nothing is focused (or on an iframe) also warns and makes the action kind error (rc 1).",
                examples: &[
                    "mahbot chrome press Enter --selector \"textarea[name=q]\"",
                    "mahbot chrome press Escape",
                ],
            }),
        },
        ActionDesc {
            name: "upload",
            purpose: "attach local files",
            tool: Some(ToolParams {
                required: &["selector", "files"],
                properties: json!({
                    "selector": {
                        "type": "string",
                        "description": "file input or its label"
                    },
                    "files": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "absolute or workspace-relative paths"
                    }
                }),
            }),
            cli: Some(CliHelp {
                syntax: "<selector> <path...> [--timeout <secs>]",
                declared: vec![value_flag("--timeout <secs>", own_bound_flag())],
                session_refusal: None,
                kinds: STEP_KINDS,
                details: "Hands each path to the page's file input — the input itself, a hidden one behind a styled control, the label that pairs with it, or the container holding it — the way a real selection reaches it. The CLI has no snapshot refs, so the selector is always CSS. Relative paths resolve against the directory this command was run in, and every path is reported as the absolute file that was used; any file you can read may be attached. Several files need a target that accepts several: a one-file input — named directly, through the label that pairs with it, or through the container holding it — refuses more than one. A success means the files reached the page. The browser side's own warning — it could not confirm the input still holds them (a React-style control consumed or replaced them) — is still a success and is repeated verbatim; it never means the page's own validation accepted them, which is invisible here. A failure means the attach did not complete, and its reason says how far it got. Where the target's shape cannot be judged from the page, the files are sent and the result's `note` says the one-file check was skipped; a target that only accepts drops, with no file input behind it, is refused by the browser side and reported as not delivered.",
                examples: &[
                    "mahbot chrome upload \"#file\" ./report.pdf",
                    "mahbot chrome upload \"input[type=file]\" ./page-1.png ./page-2.png",
                ],
            }),
        },
        ActionDesc {
            name: "session",
            purpose: "manage named CLI sessions (session stop / session status / session list)",
            tool: None,
            cli: Some(CliHelp {
                syntax: "stop <name> [--force] | status <name> | list",
                declared: vec![CliFlag {
                    scope: Some("stop"),
                    ..switch_flag(
                        "--force",
                        "also stop protected agent-tab-* / link-enricher-* sessions — 'session stop' only",
                    )
                }],
                session_refusal: Some("--session is not valid for session stop/status/list"),
                kinds: STEP_KINDS_WITH_EMPTY,
                details: "Manages named CLI sessions — a `<name>` takes letters, digits, '-' or '_' only (chrome-use's own alphabet: a '.' is refused, since chrome-use refuses to stop such a session). stop refuses protected sessions (agent-tab-* / link-enricher-*) unless --force; names are prefixed mahbot-chrome- unless already prefixed. list shows the sessions this CLI created — their names, keep marks, and seconds since last use. A session this CLI creates lives 10 minutes after its last use and is then closed; --keep (only with --session, and only on the call that creates the session) keeps it until session stop, which is also the only thing that ends a kept session. Stopping, restarting or updating mahbot closes no browser tabs. status is an opt-in liveness probe (worst case ~30s: session-list preflight + a real bounded get-url against the session): a not-running session reports empty (rc 0); a wedged one reports environment (rc 2). The probe closes nothing on its own, and for a session this CLI recorded it is still a later call with the same name: it restarts the 10-minute clock, so a diagnosis keeps a session's tabs open rather than closing them; the recovery itself is automatic everywhere the session is actually driven: every on-demand action verb recovers a session that stopped answering and says so in its error (only when mahbot's own bound ended the step, or chrome-use classified the session unresponsive — a step chrome-use timed out on its own, e.g. an expect whose condition was not met, is not a wedge and costs no session), `session stop <name>` clears it, and the background watchdog recovers one too. After a recovery, re-run the action with `--session <name>` to re-create the session — cookies persist in the profile, open tabs/tab-group identity do not. The global `status` action is session-unaware: it can report the connection ready while a named session is wedged.",
                examples: &[
                    "mahbot chrome session stop docs",
                    "mahbot chrome session status docs",
                    "mahbot chrome session list",
                ],
            }),
        },
        ActionDesc {
            name: "snapshot",
            purpose: "accessibility tree with refs",
            tool: Some(ToolParams {
                required: &[],
                properties: json!({
                    "interactive_only": {
                        "type": "boolean",
                        "description": "Only interactive elements"
                    },
                    "compact": {
                        "type": "boolean",
                        "description": "drop empty structure (default true)"
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
            purpose: "get rendered text",
            tool: Some(ToolParams {
                required: &["selector"],
                properties: json!({
                    "selector": {
                        "type": "string",
                        "description": "CSS selector or ref (@e1)"
                    }
                }),
            }),
            cli: None,
        },
        ActionDesc {
            name: "get_innertext",
            purpose: "innerText eval; prefer get_text",
            tool: Some(ToolParams {
                required: &["selector"],
                properties: json!({
                    "selector": {
                        "type": "string",
                        "description": "CSS selector or ref (@e1)"
                    }
                }),
            }),
            cli: None,
        },
        ActionDesc {
            name: "get_url",
            purpose: "get the current URL",
            tool: Some(ToolParams {
                required: &[],
                properties: json!({}),
            }),
            cli: None,
        },
        ActionDesc {
            name: "find",
            purpose: "find an element and act",
            tool: Some(ToolParams {
                required: &["by", "value", "action"],
                properties: json!({
                    "by": {
                        "type": "string",
                        "description": "text | role | label (accessible label) | placeholder | alt | title | testid | first | last | nth"
                    },
                    "value": {
                        "type": "string",
                        "description": "what to locate (CSS for first/last/nth)"
                    },
                    "action": {
                        "type": "string",
                        "description": "click | fill | hover | check | text (reads the element)"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text for action fill"
                    },
                    "name": {
                        "type": "string",
                        "description": "name filter for role"
                    },
                    "exact": {
                        "type": "boolean",
                        "description": "Whole-string match"
                    },
                    "index": {
                        "type": "integer",
                        "description": "Zero-based, for nth"
                    }
                }),
            }),
            cli: None,
        },
        ActionDesc {
            name: "screenshot",
            purpose: "capture the page as an image",
            tool: Some(ToolParams {
                required: &[],
                properties: json!({}),
            }),
            cli: Some(CliHelp {
                syntax: "[--output-dir <dir>]",
                declared: vec![value_flag(
                    "--output-dir <dir>",
                    "directory the picture is written into (default: the directory this command \
                     was run in) — it must already exist and be your own; the file is named \
                     screenshot.png, or screenshot_2.png … when that name is taken, so nothing is \
                     overwritten",
                )],
                session_refusal: None,
                kinds: STEP_KINDS,
                details: "Captures the visible page exactly as the agent-facing capture does — the \
                          same viewport capture, framed and sized by chrome-use the same way. \
                          `path` in the result is the absolute file that was written, in the \
                          platform's own spelling, so it can be opened as it is; `origin` names the \
                          page those pixels came from. The session rule in `mahbot chrome --help` \
                          decides which tab is captured.",
                examples: &[
                    "mahbot chrome screenshot",
                    "mahbot chrome screenshot --output-dir ~/Pictures --session shot",
                ],
            }),
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
                "status",
                "open",
                "count",
                "wait",
                "expect",
                "eval",
                "extract",
                "click",
                "fill",
                "type",
                "press",
                "upload",
                "screenshot",
                "session",
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
