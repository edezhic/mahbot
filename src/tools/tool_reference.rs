//! The on-demand detail behind every tool's short standing description.
//!
//! A tool's standing description answers three things only — how to use it, why, and what it
//! returns — and ends with one pointer here. Everything else (output vocabulary, per-action
//! nuance, edge cases) lives in `src/prompt/reference/{tool}.md`, fetched whole by this tool.
//! One shared entry point rather than a `reference` action on thirty tools: the model is told
//! about exactly one way to ask, and adding the detail costs one optional argument.
//!
//! The text arrives whole — [`Tool::preserve_full_output`] — because the output budget would
//! otherwise truncate a reference mid-sentence.
use serde_json::json;

use crate::Tool;

/// The asset directory the references live in, under the embedded prompt assets.
const PREFIX: &str = "reference/";

/// The one pointer asset a composed description ends with, filled with the tool's own name, so
/// every tool asks the same way.
const POINTER_ASSET: &str = "tool/reference_pointer.md";

/// This tool's own name: it keeps no detail behind itself, so it asks no pointer of its own.
const SELF: &str = "reference";

/// Tools that already answer for themselves: their own `reference` action returns their detail,
/// so the shared entry point refuses to become a second route to the same text.
const SELF_SERVING: &[&str] = &["mahbot_config", "workspace", "hyperframes"];

/// The reference text of one tool, or `None` when that tool keeps none here.
#[must_use]
pub(crate) fn text(tool: &str) -> Option<String> {
    crate::prompt::prompt_asset(&format!("{PREFIX}{tool}.md"))
        .map(|text| text.trim_end().to_owned())
}

/// The tool name a model's argument names: surrounding whitespace, quotes, backticks and a
/// trailing sentence mark are not part of it, and a legacy alias (`read_file`) resolves to the
/// tool that answers for it.
#[must_use]
pub(crate) fn resolve(tool: &str) -> String {
    let bare = tool.trim().trim_matches(|c: char| {
        c.is_whitespace() || matches!(c, '`' | '"' | '\'' | '.' | ',' | ';' | ':')
    });
    crate::tools::normalize_tool_name(&bare.to_ascii_lowercase()).to_owned()
}

/// The line a standing description ends with: the one route to that tool's detail. Rendered from
/// one asset, and shown as the arguments themselves, so a model can pass what it reads unchanged.
#[must_use]
pub(crate) fn pointer(tool: &str) -> String {
    crate::prompt::substitute(
        &crate::prompt::load_prompt(POINTER_ASSET),
        &[("{{tool}}", tool)],
    )
    .trim()
    .to_owned()
}

/// Whether a tool keeps its detail in `src/prompt/reference/{name}.md`: everything but this tool
/// and the three that answer for themselves. A tool whose description is `standing` must have
/// that asset.
#[must_use]
pub(crate) fn keeps_detail(tool: &str) -> bool {
    tool != SELF && !SELF_SERVING.contains(&tool)
}

/// A tool's standing description: its own text, then that pointer. Generated here rather than
/// written into each asset, so every tool asks the same way and the wording lives in one place. A
/// tool that answers for itself carries its own pointer instead, and this one has nothing behind
/// itself to point at.
#[must_use]
pub(crate) fn standing(tool: &str, body: &str) -> String {
    let body = body.trim_end();
    if !keeps_detail(tool) {
        return body.to_owned();
    }
    format!("{body}\n\n{}", pointer(tool))
}

pub(crate) struct ReferenceTool {
    /// The tools the caller holds — the only references it may ask for. Handing out the detail of
    /// a capability a role's toolset withholds would document what that role cannot do.
    holders: Vec<&'static str>,
    /// Whether this caller's `send` takes the owner's own arguments, `extract` and `for`. The
    /// shared `send` reference documents neither, so their detail is appended only for a holder
    /// that can pass them: a holder without them reads everything it can reach and learns nothing
    /// of either.
    owner_arguments: bool,
}

/// The detail a holder of the owner's own `send` arguments may read on top of the shared
/// reference, by tool, in the order it is appended. Kept as a table so the rule reads as
/// deliberate: the shared `send` reference names neither argument, and only a caller whose
/// toolset declares them is handed these assets.
const ARGUMENT_DETAIL: &[(&str, &[&str])] = &[(
    "send",
    &["reference/send_extract.md", "reference/send_for.md"],
)];

impl ReferenceTool {
    #[must_use]
    pub(crate) fn new(holders: Vec<&'static str>, owner_arguments: bool) -> Self {
        Self {
            holders,
            owner_arguments,
        }
    }

    /// One tool's reference, with the detail only a holder of the owner's own arguments may read
    /// appended.
    fn scoped(&self, tool: &str, text: String) -> String {
        let Some((_, assets)) = ARGUMENT_DETAIL.iter().find(|(name, _)| *name == tool) else {
            return text;
        };
        if !self.owner_arguments {
            return text;
        }
        assets.iter().fold(text, |text, asset| {
            format!("{text}\n\n{}", crate::prompt::load_prompt(asset).trim_end())
        })
    }

    /// The index an argument-less call answers with: the names this caller can ask for, so
    /// discovery never needs a guess. A tool that answers for itself is not one of them.
    fn index(&self) -> String {
        let names = self.askable();
        if names.is_empty() {
            return "No references are kept for this tool set.".to_owned();
        }
        format!(
            "References are kept for: {}. Call `reference` with one of those names as `tool`.",
            names.join(", ")
        )
    }

    /// The holders that keep their detail here.
    fn askable(&self) -> Vec<&'static str> {
        self.holders
            .iter()
            .copied()
            .filter(|name| keeps_detail(name))
            .collect()
    }
}

#[async_trait::async_trait]
impl Tool for ReferenceTool {
    fn name(&self) -> &'static str {
        SELF
    }

    fn parameters_schema(&self) -> serde_json::Value {
        crate::tools::tool_params_schema(
            &json!({
                "tool": {
                    "type": "string",
                    "description": "The tool whose detail you want, as its own description spells it. Omit for the list of names."
                }
            }),
            &[],
        )
    }

    fn side_effects(&self) -> bool {
        false
    }

    /// A reference is read to be followed, not skimmed: the ~5 KB budget would cut it mid-rule.
    fn preserve_full_output(&self) -> bool {
        true
    }

    /// A set holding nothing to offer is not advertised: its index would name no tool.
    fn is_advertised(&self) -> bool {
        self.holders.iter().any(|name| keeps_detail(name))
    }

    async fn execute(
        &self,
        _ws: &crate::Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        if !args.is_null() && !args.is_object() {
            anyhow::bail!("arguments must be an object");
        }
        crate::tools::refuse_unknown_fields(
            &self.parameters_schema(),
            &args,
            "this tool takes only `tool`",
        )?;
        // A provider padding an optional argument sends `null` for it; that, an empty name and a
        // name that is only punctuation are all the omitted argument, not a lookup.
        let Some(named) = args.get("tool").filter(|value| !value.is_null()) else {
            return Ok(self.index());
        };
        let named = named
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("`tool` is the tool's name, as a string"))?;
        let tool = resolve(named);
        if tool.is_empty() {
            return Ok(self.index());
        }
        if tool == self.name() {
            anyhow::bail!(
                "`reference` is this tool: call it with no `tool` for the index of names"
            );
        }
        // The holder check comes first: a tool this caller does not have is never named, even
        // when it would answer for itself.
        if !self.holders.contains(&tool.as_str()) {
            anyhow::bail!("no reference for `{tool}` — {}", self.index());
        }
        if SELF_SERVING.contains(&tool.as_str()) {
            anyhow::bail!("`{tool}` answers for itself: ask it with its own `reference` action");
        }
        let text = text(&tool)
            .ok_or_else(|| anyhow::anyhow!("`{tool}` has no reference — {}", self.index()))?;
        Ok(self.scoped(&tool, text))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use strum::IntoEnumIterator;

    use super::*;
    use crate::Role;
    use crate::tools::WebSearchTool;

    /// Every tool the product can offer, unfiltered — `is_advertised` is a runtime filter, not a
    /// property of the set — plus the configured `web_search`, which no static set holds.
    fn all_tools() -> Vec<Box<dyn Tool>> {
        let ws = crate::workspace::test_ws("reference");
        let mut by_name: BTreeMap<&'static str, Box<dyn Tool>> = BTreeMap::new();
        for role in Role::iter() {
            if role == Role::Custom {
                continue;
            }
            for is_admin in [false, true] {
                if role != Role::Assistant && is_admin {
                    continue;
                }
                for tool in role.tools(&ws, is_admin, Arc::default()) {
                    by_name.insert(tool.name(), tool);
                }
            }
        }
        let web_search = WebSearchTool::from_config(
            None,
            Some("test-key".to_owned()),
            Some("test-key".to_owned()),
        )
        .expect("a configured provider builds the tool");
        by_name.insert("web_search", Box::new(web_search));
        by_name.into_values().collect()
    }

    /// A holder set covering every tool, for the route's own tests. The reference itself is never
    /// one of them: a set is built before this tool is pushed into it.
    fn every_holder() -> ReferenceTool {
        ReferenceTool::new(
            all_tools()
                .iter()
                .map(|tool| tool.name())
                .filter(|name| *name != SELF)
                .collect(),
            false,
        )
    }

    /// A tool whose description points at the reference must have a non-empty asset behind it: a
    /// misnamed, missing or emptied `reference/{name}.md` would ship a dead pointer or a blank
    /// answer.
    #[test]
    fn every_tool_that_points_at_a_reference_has_one() {
        for tool in all_tools() {
            if keeps_detail(tool.name()) {
                assert!(
                    text(tool.name()).is_some_and(|text| !text.trim().is_empty()),
                    "`{}` points at a reference asset that does not exist or is empty",
                    tool.name()
                );
            }
        }
    }

    /// Whether a schema declares a `reference` action.
    fn declares_reference_action(schema: &serde_json::Value) -> bool {
        match schema {
            serde_json::Value::Object(map) => map.iter().any(|(key, value)| {
                (key == "action"
                    && value
                        .get("enum")
                        .and_then(serde_json::Value::as_array)
                        .is_some_and(|values| {
                            values.iter().any(|v| v.as_str() == Some("reference"))
                        }))
                    || declares_reference_action(value)
            }),
            serde_json::Value::Array(values) => values.iter().any(declares_reference_action),
            _ => false,
        }
    }

    /// The self-serving list is the schemas' own statement, not a list to keep by hand: a tool
    /// declaring a `reference` action answers for itself, and one that does not cannot.
    #[test]
    fn the_self_serving_list_matches_the_schemas() {
        let mut declared = Vec::new();
        for tool in all_tools() {
            if declares_reference_action(&tool.parameters_schema()) {
                declared.push(tool.name());
            }
        }
        declared.sort_unstable();
        let mut listed = SELF_SERVING.to_vec();
        listed.sort_unstable();
        assert_eq!(
            declared, listed,
            "the tools whose schema declares a `reference` action and the ones this module refuses \
             to answer for have drifted apart"
        );
    }

    /// The route answers, resolves what a copy of its own pointer shows, and refuses a name it
    /// cannot serve with a message that says why.
    #[tokio::test]
    async fn the_reference_answers_and_refuses() {
        let ws = crate::workspace::test_ws("reference");
        let tool = every_holder();
        assert!(
            tool.preserve_full_output(),
            "the ~5 KB budget would cut a rule in half"
        );
        assert!(tool.is_advertised(), "a set with holders offers the route");

        let read = tool
            .execute(&ws, serde_json::json!({"tool": "read"}))
            .await
            .expect("read keeps a reference");
        assert!(!read.trim().is_empty(), "the reference arrived empty");
        // A legacy alias, and the argument a verbatim copy of the pointer shows, resolve to the
        // tool that answers for them.
        for named in ["`read_file`", "Read."] {
            assert_eq!(
                tool.execute(&ws, serde_json::json!({"tool": named}))
                    .await
                    .expect("an alias and a copied argument resolve"),
                read,
                "{named} did not resolve to read"
            );
        }

        let index = tool
            .execute(&ws, serde_json::json!({}))
            .await
            .expect("the index");
        assert!(index.contains("read"), "the index never names `read`");
        assert!(
            !index.contains("workspace"),
            "a tool that answers for itself is not one of the names offered"
        );

        let unknown = tool
            .execute(&ws, serde_json::json!({"tool": "nope"}))
            .await
            .expect_err("an unknown name is refused");
        assert!(unknown.to_string().contains("no reference for `nope`"));
        let self_serving = tool
            .execute(&ws, serde_json::json!({"tool": "workspace"}))
            .await
            .expect_err("a self-serving tool is refused");
        assert!(
            self_serving
                .to_string()
                .contains("its own `reference` action")
        );
        let not_a_name = tool
            .execute(&ws, serde_json::json!({"tool": 3}))
            .await
            .expect_err("a tool that is not a string is refused");
        assert!(not_a_name.to_string().contains("as a string"));
        // A padded optional argument is an omitted one, not a name — and so is a name that is
        // empty or only punctuation.
        for omitted in [
            serde_json::json!({"tool": null}),
            serde_json::json!({"tool": " "}),
            serde_json::json!({"tool": "."}),
        ] {
            assert_eq!(
                tool.execute(&ws, omitted.clone())
                    .await
                    .unwrap_or_else(|err| panic!("{omitted} is the index, not a name: {err}")),
                index
            );
        }
    }

    /// A reference is only handed to a holder of that tool: a role's toolset withholds a
    /// capability by withholding its tool, and its documentation goes with it.
    #[tokio::test]
    async fn the_reference_is_scoped_to_the_callers_tools() {
        let ws = crate::workspace::test_ws("reference");
        let engineer = ReferenceTool::new(vec!["read", "edit", "search", "shell"], false);

        engineer
            .execute(&ws, serde_json::json!({"tool": "read"}))
            .await
            .expect("a held tool's reference arrives");
        for withheld in ["computer", "workspace"] {
            let refused = engineer
                .execute(&ws, serde_json::json!({"tool": withheld}))
                .await
                .expect_err("a tool this holder does not have is refused");
            assert!(
                refused
                    .to_string()
                    .contains(&format!("no reference for `{withheld}`")),
                "a withheld tool is never named as one that answers for itself: {refused}"
            );
        }

        let index = engineer
            .execute(&ws, serde_json::json!({}))
            .await
            .expect("the index");
        assert!(index.contains("read, edit, search, shell"));
        for withheld in ["computer", "mahbot_debug", "photocraft"] {
            assert!(
                !index.contains(withheld),
                "the index offers `{withheld}`, which this holder cannot call"
            );
        }

        // A set with nothing to offer — empty, or only tools that answer for themselves — is not
        // advertised at all: its index would name no tool.
        assert!(!ReferenceTool::new(Vec::new(), false).is_advertised());
        assert!(!ReferenceTool::new(vec!["workspace"], false).is_advertised());
        assert!(engineer.is_advertised());
    }

    /// The owner's argument detail reaches only a holder whose `send` takes those arguments: the
    /// shared `send` reference documents neither `extract` nor `for` — nor where a chain step that
    /// names no folder works — so a holder without them reads everything it can reach and learns
    /// nothing of any of it.
    #[tokio::test]
    async fn the_argument_detail_reaches_only_a_holder_that_can_use_it() {
        let ws = crate::workspace::test_ws("reference");
        let plain = ReferenceTool::new(vec!["send"], false)
            .execute(&ws, serde_json::json!({"tool": "send"}))
            .await
            .expect("a held tool's reference arrives");
        for absent in ["extract", "`for`", "chain"] {
            assert!(
                !plain.contains(absent),
                "a holder without the argument never reads {absent}: {plain}"
            );
        }
        let full = ReferenceTool::new(vec!["send"], true)
            .execute(&ws, serde_json::json!({"tool": "send"}))
            .await
            .expect("a held tool's reference arrives");
        // Every asset the table keeps for `send`, in order, appended whole to the shared reference.
        let mut expected = plain;
        for asset in ARGUMENT_DETAIL.iter().flat_map(|(_, assets)| *assets) {
            expected.push_str("\n\n");
            expected.push_str(crate::prompt::load_prompt(asset).trim_end());
        }
        assert_eq!(
            full, expected,
            "the detail is appended whole to the same reference"
        );
        assert!(
            full.contains("`for`"),
            "the holder of `for` is actually handed its entry: {full}"
        );
        assert!(
            full.contains("the folder the nearest earlier step named"),
            "and the folder a chain step that names none works in: {full}"
        );
    }
}
