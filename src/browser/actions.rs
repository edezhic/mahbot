//! Single shared source of browser action descriptions: one table rendered
//! into BOTH the interactive `browser` tool's LLM-facing parameter schema and
//! the `mahbot browser` CLI help, so the two consumers cannot drift.

use std::sync::LazyLock;

use serde_json::{Value, json};

use crate::browser::contract::OutKind;

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

pub(crate) static ACTIONS: LazyLock<Vec<ActionDesc>> = LazyLock::new(build_actions);

/// Build the shared action registry (registry order = CLI help order).
#[expect(clippy::too_many_lines)]
fn build_actions() -> Vec<ActionDesc> {
    vec![
        ActionDesc {
            name: "status",
            purpose: "check chrome-use CLI, extension relay, Chrome, and display health",
            tool: None,
            cli: Some(CliHelp {
                syntax: "",
                flags: &[],
                session: false,
                kinds: &[OutKind::Ok, OutKind::Environment, OutKind::Usage],
                details: "Pure preflight — probes chrome-use, the extension relay, a running Chrome, and a usable display. Never launches Chrome and never mutates the environment. Rejects --session.",
                examples: &["mahbot browser status"],
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
                    ("--timeout <secs>", "step deadline in seconds (default 8)"),
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
                details: "The URL must be http(s). Reports the committed final URL. An uncommitted navigation (tab still on about:blank) or a Chrome error page is kind network. An invalid URL is kind usage (rc 3).",
                examples: &[
                    "mahbot browser open https://example.com",
                    "mahbot browser open https://example.com --expect \"#main\" --structural --timeout 15",
                ],
            }),
        },
        ActionDesc {
            name: "count",
            purpose: "count elements matching a CSS selector (eval shim over querySelectorAll)",
            tool: None,
            cli: Some(CliHelp {
                syntax: "<selector> [--timeout <secs>]",
                flags: &[("--timeout <secs>", "step deadline in seconds (default 8)")],
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
                    "mahbot browser count \".card\"",
                    "mahbot browser count \"a[href]\" --session docs",
                ],
            }),
        },
        ActionDesc {
            name: "wait",
            purpose: "wait until a CSS selector matches something",
            tool: None,
            cli: Some(CliHelp {
                syntax: "<selector> [--timeout <secs>]",
                flags: &[("--timeout <secs>", "step deadline in seconds (default 8)")],
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
                details: "The deadline is enforced mahbot-side by bounding the spawned chrome-use step; --timeout is never forwarded to chrome-use.",
                examples: &["mahbot browser wait \"#results\" --timeout 15"],
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
                flags: &[("--timeout <secs>", "step deadline in seconds (default 8)")],
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
                examples: &["mahbot browser eval 'document.title'"],
            }),
        },
        ActionDesc {
            name: "extract",
            purpose: "extract rows from the page with a JSON schema file",
            tool: None,
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
                    ("--timeout <secs>", "step deadline in seconds (default 8)"),
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
                details: "When the schema's rows selector matches 0 elements, the empty region is reported honestly (kind empty, rc 0) without invoking chrome-use's phantom-row extract. An unreadable or invalid schema file is kind usage (rc 3).",
                examples: &["mahbot browser extract --schema-file products.json --limit 20"],
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
                    ("--timeout <secs>", "step deadline in seconds (default 8)"),
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
                    "mahbot browser click \"#submit\"",
                    "mahbot browser click \".next\" --if-present",
                ],
            }),
        },
        ActionDesc {
            name: "session",
            purpose: "stop a named CLI session (session stop)",
            tool: None,
            cli: Some(CliHelp {
                syntax: "stop <name> [--force]",
                flags: &[(
                    "--force",
                    "also stop protected agent-tab-* / link-enricher-* sessions",
                )],
                session: false,
                kinds: &[
                    OutKind::Ok,
                    OutKind::Timeout,
                    OutKind::Network,
                    OutKind::NotFound,
                    OutKind::Error,
                    OutKind::Environment,
                    OutKind::Usage,
                ],
                details: "Refuses to stop protected sessions (the interactive tool's agent-tab-* and the link enricher's link-enricher-*) unless --force is passed. Session names are prefixed with mahbot-browser- unless already prefixed.",
                examples: &["mahbot browser session stop docs"],
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
            name: "press",
            purpose: "Press a keyboard key at the current focus (e.g. Enter to submit forms)",
            tool: Some(ToolParams {
                required: &["key"],
                properties: json!({
                    "key": {
                        "type": "string",
                        "description": "Key to press (e.g. Enter, Tab, Escape, Control+a, ArrowDown)"
                    }
                }),
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
                        "description": "Action to perform: click (click element), fill (clear field then type), type (append text without clearing, uses 'text' parameter), hover (hover over element), focus (focus element), check (check checkbox/radio button), uncheck (uncheck checkbox/radio button), text (get element text content — does NOT use the 'text' param; the 'text' param is only for fill/type). For filling text into inputs, use 'fill' with the 'text' parameter. For typing without clearing first, use 'type'. Press Enter after filling to submit forms."
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to fill/type into the element (for action 'fill' or 'type')"
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
                "status", "open", "count", "wait", "eval", "extract", "click", "session",
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
