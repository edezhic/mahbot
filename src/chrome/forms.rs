//! Safe chrome-use argv forms shared by the interactive `chrome` tool and
//! the `mahbot chrome` CLI. The unsafe shapes are kept out at this shared
//! layer: the numeric `wait <ms>` sleep form has no [`WaitTarget`] variant and
//! [`wait_target`] rejects a numeric selector, and expect conditions are a
//! validated allowlist ([`ExpectCond`] → [`expect_args`]).

use crate::chrome::escape_js_single_quoted;
use serde_json::Value;

/// The inline-text argv element for a `fill`/`type` value: a leading-dash
/// text is shielded with the standard `--` end-of-options marker, which
/// chrome-use's arg preprocessor drops and forwards the value verbatim
/// (otherwise a global/known flag of the same name could swallow it).
pub(crate) fn text_value_argv(text: &str) -> Vec<String> {
    if text.starts_with('-') {
        vec!["--".to_string(), text.to_string()]
    } else {
        vec![text.to_string()]
    }
}

/// What a `wait` waits for. The numeric sleep form (`wait 5000`) has no
/// variant — [`wait_target`] rejects it before one can be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WaitTarget {
    Selector(String),
    Url(String),
    Text(String),
}

impl WaitTarget {
    /// Human-readable target for envelopes and tool output.
    pub(crate) fn describe(&self) -> String {
        match self {
            Self::Selector(s) => format!("selector '{s}'"),
            Self::Url(p) => format!("url pattern '{p}'"),
            Self::Text(t) => format!("text '{t}'"),
        }
    }
}

/// Build `wait` argv for a safe target. `--timeout <ms>` IS forwarded —
/// chrome-use's wait forms honor it (verified against 1.5.101); callers
/// additionally bound the spawned step mahbot-side.
pub(crate) fn wait_args(target: &WaitTarget, timeout_ms: u128) -> Vec<String> {
    let mut args = match target {
        WaitTarget::Selector(s) => vec!["wait".to_string(), s.clone()],
        WaitTarget::Url(p) => vec!["wait".to_string(), "--url".to_string(), p.clone()],
        WaitTarget::Text(t) => vec!["wait".to_string(), "--text".to_string(), t.clone()],
    };
    args.push("--timeout".to_string());
    args.push(timeout_ms.to_string());
    args
}

/// An `expect` condition — the safe allowlist subset of chrome-use's grammar
/// (gone/request/no-errors and the --not/--regex/--no-wait flags are not
/// exposed).
pub(crate) enum ExpectCond {
    State {
        selector: String,
        state: &'static str,
    }, // visible | hidden | present
    Count {
        selector: String,
        op: String,
        n: u64,
    },
    Text {
        selector: String,
        predicate: &'static str,
        value: String,
    },
    Value {
        selector: String,
        predicate: &'static str,
        value: String,
    },
    Attr {
        selector: String,
        name: String,
        predicate: &'static str,
        value: String,
    },
    Url {
        predicate: &'static str,
        pattern: String,
    },
}

/// Validate an element state word → the canonical form.
pub(crate) fn parse_state(s: &str) -> Option<&'static str> {
    match s {
        "visible" => Some("visible"),
        "hidden" => Some("hidden"),
        "present" => Some("present"),
        _ => None,
    }
}

/// Validate a comparison predicate word → the canonical form.
pub(crate) fn parse_predicate(s: &str) -> Option<&'static str> {
    match s {
        "equals" => Some("equals"),
        "contains" => Some("contains"),
        "matches" => Some("matches"),
        _ => None,
    }
}

/// Validate a count comparison operator (symbols or word forms); the word is
/// forwarded verbatim to chrome-use.
pub(crate) fn parse_count_op(s: &str) -> Option<String> {
    let valid = [
        "==", "!=", ">", "<", ">=", "<=", "eq", "ne", "gt", "lt", "ge", "le",
    ];
    valid.contains(&s).then(|| s.to_string())
}

/// Build `expect` argv (selector-first, chrome-use 1.5.101 grammar) plus the
/// forwarded `--timeout <ms>`.
pub(crate) fn expect_args(cond: &ExpectCond, timeout_ms: u128) -> Vec<String> {
    let mut args = match cond {
        ExpectCond::State { selector, state } => {
            vec!["expect".to_string(), selector.clone(), (*state).to_string()]
        }
        ExpectCond::Count { selector, op, n } => vec![
            "expect".to_string(),
            "count".to_string(),
            selector.clone(),
            op.clone(),
            n.to_string(),
        ],
        ExpectCond::Text {
            selector,
            predicate,
            value,
        } => vec![
            "expect".to_string(),
            "text".to_string(),
            selector.clone(),
            (*predicate).to_string(),
            value.clone(),
        ],
        ExpectCond::Value {
            selector,
            predicate,
            value,
        } => vec![
            "expect".to_string(),
            "value".to_string(),
            selector.clone(),
            (*predicate).to_string(),
            value.clone(),
        ],
        ExpectCond::Attr {
            selector,
            name,
            predicate,
            value,
        } => vec![
            "expect".to_string(),
            "attr".to_string(),
            selector.clone(),
            name.clone(),
            (*predicate).to_string(),
            value.clone(),
        ],
        ExpectCond::Url { predicate, pattern } => vec![
            "expect".to_string(),
            "url".to_string(),
            (*predicate).to_string(),
            pattern.clone(),
        ],
    };
    args.push("--timeout".to_string());
    args.push(timeout_ms.to_string());
    args
}

/// Human-readable condition description for envelopes and tool output.
pub(crate) fn describe(cond: &ExpectCond) -> String {
    match cond {
        ExpectCond::State { selector, state } => format!("'{selector}' is {state}"),
        ExpectCond::Count { selector, op, n } => format!("count('{selector}') {op} {n}"),
        ExpectCond::Text {
            selector,
            predicate,
            value,
        } => format!("text of '{selector}' {predicate} \"{value}\""),
        ExpectCond::Value {
            selector,
            predicate,
            value,
        } => format!("value of '{selector}' {predicate} \"{value}\""),
        ExpectCond::Attr {
            selector,
            name,
            predicate,
            value,
        } => format!("attr '{name}' of '{selector}' {predicate} \"{value}\""),
        ExpectCond::Url { predicate, pattern } => format!("url {predicate} \"{pattern}\""),
    }
}

/// Resolve the wait target from the caller-supplied parts — exactly one of
/// selector/url/text, with one policy and one error text for both frontends.
/// A numeric selector is chrome-use's silent-sleep form (rc 0 without
/// waiting): rejected here, never forwarded.
pub(crate) fn wait_target(
    selector: Option<&str>,
    url: Option<&str>,
    text: Option<&str>,
) -> Result<WaitTarget, String> {
    match (selector, url, text) {
        (Some(s), None, None) if s.parse::<u64>().is_err() => {
            Ok(WaitTarget::Selector(s.to_string()))
        }
        (Some(_), None, None) => Err(
            "numeric wait is a silent sleep — not supported; wait for a selector, a URL pattern, or page text"
                .to_string(),
        ),
        (None, Some(u), None) => Ok(WaitTarget::Url(u.to_string())),
        (None, None, Some(t)) => Ok(WaitTarget::Text(t.to_string())),
        _ => Err(
            "give exactly one wait target — a selector, a URL pattern, or page text".to_string(),
        ),
    }
}

/// Build eval JS that returns the count of elements matching `selector`
/// (chrome-use has no standalone count verb reachable from the CLI grammar we
/// use, so both frontends count via this eval shim). Top-document scope: the
/// count gate diverges from chrome-use's frame-aware extract on iframed rows
/// — an accepted heuristic, since the gate exists to keep phantom rows out.
pub(crate) fn count_eval_js(selector: &str) -> String {
    format!(
        "document.querySelectorAll('{}').length",
        escape_js_single_quoted(selector)
    )
}

/// The extract honest-empty gate decision (see [`extract_gate`]).
pub(crate) enum ExtractGate<E> {
    /// 0 matches — report the empty region WITHOUT invoking extract:
    /// chrome-use's rows-mode extract returns phantom rows on a 0-match, and
    /// its invalid-CSS auto-detect fallback manufactures rows too.
    Empty,
    /// Non-zero matches — safe to run the extract.
    Proceed,
    /// The count eval failed (e.g. an invalid CSS rows selector threw) — an
    /// explicit error, never a fallback to extract. Carries the caller's own
    /// classified failure so each frontend renders it in its native channel.
    Fail(E),
}

/// The extract gate, shared by both frontends: count the rows selector
/// ([`count_eval_js`]) and let [`ExtractGate`] decide.
pub(crate) fn extract_gate<E>(count: Result<u64, E>) -> ExtractGate<E> {
    match count {
        Ok(0) => ExtractGate::Empty,
        Ok(_) => ExtractGate::Proceed,
        Err(e) => ExtractGate::Fail(e),
    }
}

/// The corrective suffix shared by every getter-validation error.
const GETTER_GRAMMAR: &str = "valid getters are \"text\" (default), \"html\", \"value\", or \
     \"@<attribute>\" (e.g. \"@href\"); attributes require the \"@\" prefix — a bare \
     attribute name is not a getter";

/// Validate the extract-schema getter vocabulary against chrome-use's
/// documented grammar: `get` = "text" (default) | "@<attribute>" | "html" |
/// "value". chrome-use silently falls back to textContent for unknown
/// getters, which produced plausible-but-wrong `ok:true` extractions (rc 0,
/// no signal) — mahbot rejects them loudly instead. Attributes require the
/// "@" prefix (get "@href", not "href"); "@" alone is invalid. The allowlist
/// is pinned to the documented chrome-use grammar (stable across 1.5.10x) —
/// extend it if a future chrome-use release documents new getters.
pub(crate) fn validate_extract_getters(schema: &Value) -> Result<(), String> {
    let Some(fields) = schema.get("fields").and_then(Value::as_object) else {
        // Missing or non-object "fields" is shape-checked elsewhere; getter
        // validation only applies to the object form.
        return Ok(());
    };
    for (name, field) in fields {
        let Some(obj) = field.as_object() else {
            continue; // plain CSS-selector string fields need no getter check.
        };
        let Some(get) = obj.get("get") else {
            continue;
        };
        let Some(getter) = get.as_str() else {
            return Err(format!(
                "extract field '{name}' has a non-string getter {get} — {GETTER_GRAMMAR}"
            ));
        };
        if let Some(attr) = getter.strip_prefix('@') {
            if attr.is_empty() {
                return Err(format!(
                    "extract field '{name}' has an empty getter \"@\" — {GETTER_GRAMMAR}"
                ));
            }
            continue;
        }
        if !matches!(getter, "text" | "html" | "value") {
            return Err(format!(
                "extract field '{name}' has an invalid getter \"{getter}\" — {GETTER_GRAMMAR}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(args: &[String]) -> Vec<&str> {
        args.iter().map(String::as_str).collect()
    }

    /// The wait args forward the timeout for every target and the expect args cover every variant.
    #[test]
    fn wait_and_expect_args_carry_every_target() {
        {
            let case = "wait args forwards timeout for every target";
            let wait_cases = [
                (
                    "selector",
                    WaitTarget::Selector("#main".into()),
                    vec!["wait", "#main"],
                ),
                (
                    "url",
                    WaitTarget::Url("https://example.com".into()),
                    vec!["wait", "--url", "https://example.com"],
                ),
                (
                    "text",
                    WaitTarget::Text("Loaded".into()),
                    vec!["wait", "--text", "Loaded"],
                ),
            ];
            for (name, target, mut expected) in wait_cases {
                expected.extend(["--timeout", "8000"]);
                assert_eq!(
                    argv(&wait_args(&target, 8000)),
                    expected,
                    "case {case}: {name}"
                );
            }
        }

        {
            let case = "expect args covers every variant";
            let expect_cases = [
                (
                    "state",
                    ExpectCond::State {
                        selector: "#main".into(),
                        state: "visible",
                    },
                    vec!["expect", "#main", "visible"],
                ),
                (
                    "count",
                    ExpectCond::Count {
                        selector: ".card".into(),
                        op: "==".into(),
                        n: 3,
                    },
                    vec!["expect", "count", ".card", "==", "3"],
                ),
                (
                    "text",
                    ExpectCond::Text {
                        selector: "#main".into(),
                        predicate: "contains",
                        value: "hello".into(),
                    },
                    vec!["expect", "text", "#main", "contains", "hello"],
                ),
                (
                    "value",
                    ExpectCond::Value {
                        selector: "input".into(),
                        predicate: "equals",
                        value: "abc".into(),
                    },
                    vec!["expect", "value", "input", "equals", "abc"],
                ),
                (
                    "attr",
                    ExpectCond::Attr {
                        selector: "a".into(),
                        name: "href".into(),
                        predicate: "matches",
                        value: "example.com".into(),
                    },
                    vec!["expect", "attr", "a", "href", "matches", "example.com"],
                ),
                (
                    "url",
                    ExpectCond::Url {
                        predicate: "contains",
                        pattern: "dashboard".into(),
                    },
                    vec!["expect", "url", "contains", "dashboard"],
                ),
            ];
            for (name, cond, mut expected) in expect_cases {
                expected.extend(["--timeout", "8000"]);
                assert_eq!(
                    argv(&expect_args(&cond, 8000)),
                    expected,
                    "case {case}: {name}"
                );
            }
        }
    }

    /// The wait state, predicate and count operators each parse their documented
    /// spellings and reject the rest.
    #[test]
    fn parse_each_wait_predicate_part_accepts_and_rejects() {
        {
            let case = "parse state accepts and rejects";
            for state in ["visible", "hidden", "present"] {
                assert_eq!(
                    parse_state(state),
                    Some(state),
                    "case {case}: {state:?} is a documented state"
                );
            }
            for state in ["gone", "VISIBLE", ""] {
                assert_eq!(
                    parse_state(state),
                    None,
                    "case {case}: {state:?} is not a state"
                );
            }
        }

        {
            let case = "parse predicate accepts and rejects";
            for predicate in ["equals", "contains", "matches"] {
                assert_eq!(
                    parse_predicate(predicate),
                    Some(predicate),
                    "case {case}: {predicate:?} is a documented predicate"
                );
            }
            for predicate in ["regex", "EQUALS", ""] {
                assert_eq!(
                    parse_predicate(predicate),
                    None,
                    "case {case}: {predicate:?} is not a predicate"
                );
            }
        }

        {
            let case = "parse count op accepts symbols and words";
            for op in [
                "==", "!=", ">", "<", ">=", "<=", "eq", "ne", "gt", "lt", "ge", "le",
            ] {
                assert_eq!(
                    parse_count_op(op).as_deref(),
                    Some(op),
                    "case {case}: op {op:?} should parse"
                );
            }
        }

        {
            let case = "parse count op rejects unknowns";
            for op in [">>", "=", "like", " between ", ""] {
                assert_eq!(
                    parse_count_op(op),
                    None,
                    "case {case}: op {op:?} should be rejected"
                );
            }
        }
    }

    /// The count verb's JS wrapper escapes the selector, so a quote or a
    /// backslash in it can neither end nor alter the single-quoted literal.
    #[test]
    fn count_eval_js_escapes_single_quotes_and_backslashes() {
        for (name, selector, js) in [
            ("quote", "a'b", "document.querySelectorAll('a\\'b').length"),
            (
                "backslash",
                "a\\b",
                "document.querySelectorAll('a\\\\b').length",
            ),
            (
                "plain",
                "div.c",
                "document.querySelectorAll('div.c').length",
            ),
        ] {
            assert_eq!(count_eval_js(selector), js, "{name}");
        }
    }

    #[test]
    fn wait_target_rejects_numeric_and_multiple_targets() {
        // Happy targets.
        assert_eq!(
            wait_target(Some("#x"), None, None),
            Ok(WaitTarget::Selector("#x".into()))
        );
        assert_eq!(
            wait_target(None, Some("dash"), None),
            Ok(WaitTarget::Url("dash".into()))
        );
        assert_eq!(
            wait_target(None, None, Some("hi")),
            Ok(WaitTarget::Text("hi".into()))
        );
        // The numeric silent-sleep form is rejected, never forwarded.
        let err = wait_target(Some("5000"), None, None).unwrap_err();
        assert!(err.contains("silent sleep"), "err: {err}");
        // Exactly-one policy.
        assert!(wait_target(Some("#x"), Some("dash"), None).is_err());
        assert!(wait_target(None, None, None).is_err());
        assert!(wait_target(Some("#x"), Some("d"), Some("t")).is_err());
    }

    /// The `get:` grammar: `text`/`html`/`value` and `@<attribute>` pass,
    /// everything else is refused.
    #[test]
    fn validate_extract_getters_accepts_the_grammar_and_rejects_the_rest() {
        struct Case {
            name: &'static str,
            schema: Value,
            /// Substrings the refusal must name; `None` means the schema is valid.
            refused: Option<&'static [&'static str]>,
        }

        let cases = [
            Case {
                // accepts_documented_grammar: text/html/value and "@<attribute>"
                // all pass, and a plain CSS-selector string field needs no getter.
                name: "accepts_documented_grammar",
                schema: serde_json::json!({
                    "rows": ".card",
                    "fields": {
                        "t": {"sel": ".a", "get": "text"},
                        "h": {"sel": ".b", "get": "html"},
                        "v": {"sel": ".c", "get": "value"},
                        "u": {"sel": "a", "get": "@href"},
                        "plain": ".title",
                    },
                }),
                refused: None,
            },
            Case {
                // A field object with no "get" key is fine (defaults to text).
                name: "field_object_without_get",
                schema: serde_json::json!({"fields": {"title": {"sel": ".title"}}}),
                refused: None,
            },
            Case {
                // Plain string fields (CSS selectors) are fine.
                name: "plain_string_fields",
                schema: serde_json::json!({"fields": {"title": ".title", "url": "a"}}),
                refused: None,
            },
            Case {
                // Missing or non-object "fields" is shape-checked elsewhere — Ok here.
                name: "missing_fields",
                schema: serde_json::json!({"rows": ".card"}),
                refused: None,
            },
            Case {
                name: "non_object_fields",
                schema: serde_json::json!({"fields": ".title"}),
                refused: None,
            },
            Case {
                // Bare attribute name (no "@") is rejected with the convention taught.
                name: "bare_attribute_name",
                schema: serde_json::json!({"fields": {"link": {"sel": "a", "get": "href"}}}),
                refused: Some(&["@", "getter", "href"]),
            },
            Case {
                // Empty "@" is rejected.
                name: "empty_getter",
                schema: serde_json::json!({"fields": {"u": {"sel": "a", "get": "@"}}}),
                refused: Some(&["empty getter"]),
            },
            Case {
                // Non-string get is rejected.
                name: "non_string_getter",
                schema: serde_json::json!({"fields": {"u": {"sel": "a", "get": 42}}}),
                refused: Some(&["non-string getter"]),
            },
        ];

        for case in cases {
            match case.refused {
                None => assert!(
                    validate_extract_getters(&case.schema).is_ok(),
                    "{}: must pass",
                    case.name
                ),
                Some(substrings) => {
                    let err = validate_extract_getters(&case.schema)
                        .expect_err("the getter must be refused");
                    for needle in substrings {
                        assert!(err.contains(needle), "{}: {err}", case.name);
                    }
                }
            }
        }
    }
}
