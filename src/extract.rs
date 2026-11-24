//! The `extract` schema of `send`: the answer comes back as JSON matching it.
//!
//! A caller supplies a JSON schema; the finished answer is turned into JSON matching it by a
//! separate model call over that answer ([`transform`]). The step's own request is never asked for
//! structured output — no `response_format`, no provider-side schema — so the step keeps its
//! prompt-prefix cache and never changes how it works, and a failed transformation never re-runs
//! it.
//!
//! Only shape is enforced, and only a closed subset of JSON Schema: `type`, `properties`,
//! `required`, `items`, `enum`, `const` and a boolean `additionalProperties`, nested to
//! [`MAX_DEPTH`] levels. Annotation and non-shape keywords are ignored; anything that would change
//! what shape is required and that this product would otherwise silently mis-enforce is refused
//! whole, before anything runs ([`Extract::parse`]).
//!
//! The schema travels with whatever carries the step — a chain step ([`crate::launch::Step`]), a
//! branch or role launch ([`crate::Launch`]), a manager envelope
//! ([`crate::agent::message_router::AgentJob`]), a worker round's row (`jobs.extract`) — so a step
//! resumed after a restart still knows what was asked of it and runs the transformation again. That
//! is not a promise of JSON: the transformation is fail-open, so a replay can still forward the
//! answer unchanged.
//!
//! The transformation runs where the step ran: the model and provider recorded for the step's
//! conversation ([`crate::session::SessionStore::get_provider`]), so its cost does not depend on
//! where the step sits in a chain. Only a step with nothing recorded falls back to the role's
//! configured model, chosen fresh. It is a call of its own: no conversation is replayed, and it
//! carries its own purpose ([`PURPOSE`]) in the stats rows and in Running Agents.

use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::agent::registry::ParentKey;

/// `send`'s argument carrying the schema. Only the owner's Assistant's `send` and its command-line
/// twin declare it (see [`crate::tools::send`]).
pub(crate) const ARG: &str = "extract";

/// The longest schema text accepted — the schema as supplied, or the canonical serialization of an
/// object given to an agent.
pub(crate) const MAX_TEXT_BYTES: usize = 64 * 1024;

/// The deepest a schema may nest: the root is the first level, and every `properties` value and
/// every `items` node adds one.
pub(crate) const MAX_DEPTH: usize = 8;

/// The longest a whole transformation may take: when it is reached the attempts end and the step's
/// whole answer is forwarded unchanged.
pub(crate) const DEADLINE: Duration = Duration::from_mins(5);

/// The purpose this transformation records — in `llm_requests` and in Running Agents.
pub(crate) const PURPOSE: &str = "extract";

/// The enforced keywords: what the product checks the JSON document against.
const ENFORCED: &[&str] = &[
    "type",
    "properties",
    "required",
    "items",
    "enum",
    "const",
    "additionalProperties",
];

/// The tolerated keywords: annotations and bounds, ignored entirely.
const TOLERATED: &[&str] = &[
    "$schema",
    "$id",
    "title",
    "description",
    "default",
    "examples",
    "format",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minProperties",
    "maxProperties",
];

/// Keywords that point at a schema elsewhere. Refused by name, because a schema that only
/// references another is not one this product can check; any other `$`-keyword is refused as
/// simply unsupported.
const REFERENCE_LIKE: &[&str] = &[
    "$ref",
    "$defs",
    "$dynamicRef",
    "$dynamicAnchor",
    "$anchor",
    "$recursiveRef",
    "$recursiveAnchor",
];

/// The JSON types `type` accepts.
const TYPES: &[&str] = &[
    "object", "array", "string", "number", "integer", "boolean", "null",
];

/// A validated `extract` schema. Equality is the value's: two schemas that differ only in key order
/// or whitespace are the same schema, because [`serde_json`] orders object keys.
///
/// `Deserialize` validates, so a schema riding an envelope ([`crate::launch::Step`],
/// [`crate::Launch`], [`crate::agent::message_router::AgentJob`]) is held to the same subset as one
/// given in a call — an unsupported keyword makes the record unreadable rather than applying a
/// looser schema. The two TEXT columns (`jobs.extract`, `session_metadata.extract`) are read back
/// through [`Extract::from_stored`]; an older build never selects a column it does not know, so
/// those two are the one place a schema is dropped quietly by a downgrade.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct Extract(Value);

impl<'de> Deserialize<'de> for Extract {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        Self::parse(&value).map_err(serde::de::Error::custom)
    }
}

impl Extract {
    /// Read a schema given as an agent argument: an object, or JSON text.
    pub(crate) fn from_arg(value: &Value) -> Result<Self> {
        match value {
            Value::String(text) => Self::from_text(text),
            other => Self::parse(other),
        }
    }

    /// Read a schema given as text: the command line's form. The cap measures the text as supplied.
    pub(crate) fn from_text(text: &str) -> Result<Self> {
        if text.len() > MAX_TEXT_BYTES {
            bail!(
                "usage: an `{ARG}` schema is at most {} KiB of text",
                MAX_TEXT_BYTES / 1024
            );
        }
        let text = text.trim();
        if text.is_empty() {
            bail!("usage: `{ARG}` is empty — give the JSON schema the answer must match");
        }
        let value: Value = serde_json::from_str(text)
            .with_context(|| format!("usage: `{ARG}` is not valid JSON"))?;
        Self::parse(&value)
    }

    /// Read a schema given as a JSON value — the one validation every form goes through, an object
    /// argument and an envelope-carried schema included. The cap measures an object's canonical
    /// text, so both forms obey the same bound.
    fn parse(value: &Value) -> Result<Self> {
        let schema = Self(value.clone());
        let canonical = schema.canonical();
        if canonical.len() > MAX_TEXT_BYTES {
            bail!(
                "usage: an `{ARG}` schema is at most {} KiB of text",
                MAX_TEXT_BYTES / 1024
            );
        }
        validate_node(&schema.0, 1)?;
        Ok(schema)
    }

    /// Read a schema back from a stored column (`jobs.extract`, `session_metadata.extract`). The
    /// value is validated like any other — a stored schema this build would refuse is never applied
    /// as a looser one — and a value it cannot read is reported and treated as absent: the answer is
    /// still delivered whole, so a corrupt column never loses a round's work, and it never passes
    /// for a schema that was honoured.
    pub(crate) fn from_stored(raw: Option<String>) -> Option<Self> {
        match serde_json::from_str::<Self>(&raw?) {
            Ok(extract) => Some(extract),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "stored `extract` schema is unreadable — delivering the answer whole",
                );
                None
            }
        }
    }

    /// The schema's canonical text: the prompt's rendering of it and the chain identity's bytes.
    pub(crate) fn canonical(&self) -> String {
        render(&self.0)
    }

    /// Check one JSON document against the schema, answering with what is wrong when it does not
    /// match. This is what the re-ask tells the model.
    fn check(&self, value: &Value) -> std::result::Result<(), String> {
        check_node(&self.0, value, "the answer ")
    }
}

/// The step whose answer is being transformed: where the call runs, what it is grouped under, and
/// which conversation's recorded provider and model it follows.
pub(crate) struct StepContext<'a> {
    pub role: crate::Role,
    pub workspace: &'a str,
    /// The conversation that produced the answer (`session_metadata.agent_id`), when there is one.
    /// Its recorded provider and model are what the transformation runs on.
    pub conversation: Option<&'a str>,
    /// The Running Agents group this call appears under.
    pub parent: Option<ParentKey>,
}

/// Turn one finished answer into JSON matching `extract`, or `None` when it cannot be produced.
///
/// Fail-open by design: the caller forwards the whole answer unchanged, so a failed transformation
/// never loses the step's work and never fails a chain. The call is bounded by its own attempt
/// budget ([`crate::retry::RetryPolicy::extract`]) and [`DEADLINE`]; a reply cut short by the
/// output limit ends it (re-asking cannot succeed), and a re-ask tells the model what was wrong
/// with its previous reply.
async fn transform(extract: &Extract, step: &StepContext<'_>, answer: &str) -> Option<String> {
    // Visible for the whole call, so Running Agents shows it under its own purpose.
    let _visible = crate::agent::registry::NON_AGENT_CALLS.register(
        PURPOSE,
        step.workspace,
        step.parent.clone(),
        false,
        None,
    );
    // ONE user turn carrying both the answer and the schema. Some model families refuse a
    // conversation whose first non-system message is not `user`, and a refusal here would fail
    // open — the feature would silently do nothing on those models.
    let prompt = crate::prompt::substitute(
        &crate::prompt::load_prompt("extraction/schema.md"),
        &[("{{schema}}", &extract.canonical()), ("{{answer}}", answer)],
    );
    let mut request = crate::agent::chat_request(step.role, None, Vec::new());
    // The transformation runs where the step ran: the model and provider recorded for the step's
    // conversation. Nothing recorded — no conversation, or one that predates the columns — falls
    // back to the role's configured model and a fresh pick.
    let recorded = match step.conversation {
        Some(conversation) => crate::session::store().get_provider(conversation).await,
        None => None,
    };
    match recorded {
        Some(stored) => {
            request.model = stored.model;
            request.provider = Some(stored.provider);
        }
        None => request.provider = crate::providers::selection::pick(&request.model),
    }
    request.meta = Some(crate::ChatRequestMeta {
        purpose: PURPOSE,
        agent_id: format!("extract_{}", crate::generate_suffix()),
        role: step.role.as_str().to_string(),
        workspace: step.workspace.to_string(),
        ticket_id: None,
    });
    let schema = extract.clone();
    let validate = move |value: &Value| schema.check(value);
    let rules = crate::agent::extraction::Rules {
        feedback: true,
        truncation_terminal: true,
        deadline: Some(DEADLINE),
        validator_class: crate::retry::FailureClass::SchemaMismatch,
    };
    match crate::agent::extraction::retry_extract_structured_with(
        &[],
        &prompt,
        &request,
        Some(&validate),
        Some(&crate::retry::RetryPolicy::extract()),
        rules,
    )
    .await
    {
        Ok(value) => match serde_json::to_string(&value) {
            Ok(text) => Some(text),
            Err(e) => {
                tracing::warn!(error = %e, "the extracted answer could not be written as JSON");
                None
            }
        },
        Err(e) => {
            tracing::warn!(
                purpose = PURPOSE,
                workspace = step.workspace,
                error = %e,
                "the answer could not be turned into JSON matching its schema — forwarding it whole",
            );
            None
        }
    }
}

/// The answer to deliver, and which of the two things happened: `true` when the answer came back
/// as the schema's JSON, `false` when the whole answer was forwarded unchanged. A transformation
/// that fails forwards the whole answer, so a step's work is never lost and a chain never fails
/// because of it. A turn that produced no answer spends no transformation call — the existing rules
/// for an empty answer apply unchanged.
pub(crate) async fn answer_as_json_reporting(
    extract: Option<&Extract>,
    step: &StepContext<'_>,
    answer: &str,
) -> (String, bool) {
    let Some(extract) = extract.filter(|_| !answer.trim().is_empty()) else {
        return (answer.to_string(), false);
    };
    match transform(extract, step, answer).await {
        Some(json) => (json, true),
        None => (answer.to_string(), false),
    }
}

/// Whether a node's keyword set is one this product can enforce.
fn validate_node(node: &Value, level: usize) -> Result<()> {
    if level > MAX_DEPTH {
        bail!(
            "usage: an `{ARG}` schema nests at most {MAX_DEPTH} levels — the root is the first, and \
             every `properties` value and `items` node adds one"
        );
    }
    let Some(object) = node.as_object() else {
        if node.is_boolean() {
            bail!("usage: a boolean subschema is not supported in an `{ARG}` schema");
        }
        bail!("usage: an `{ARG}` schema node must be a JSON object");
    };
    for key in object.keys() {
        if ENFORCED.contains(&key.as_str()) || TOLERATED.contains(&key.as_str()) {
            continue;
        }
        if REFERENCE_LIKE.contains(&key.as_str()) {
            bail!("usage: `{key}` is reference-like and is not supported in an `{ARG}` schema");
        }
        bail!("usage: `{key}` is not supported in an `{ARG}` schema");
    }
    // Every declared keyword's own shape is checked before "constrains nothing", so a node whose
    // only fault is an empty `enum` is told exactly that.
    if let Some(values) = object.get("enum") {
        match values.as_array() {
            Some(values) if !values.is_empty() => {}
            Some(_) => bail!("usage: `enum` lists at least one allowed value"),
            None => bail!("usage: `enum` is a list of allowed values"),
        }
    }
    let types = declared_types(object)?;
    if object.contains_key("properties") && !types.is_empty() && !types.contains(&"object") {
        bail!("usage: a schema node declaring `properties` must accept `object`");
    }
    if object.contains_key("items") && !types.is_empty() && !types.contains(&"array") {
        bail!("usage: a schema node declaring `items` must accept `array`");
    }
    if !constrains_something(object, &types) {
        bail!(
            "usage: this `{ARG}` schema node constrains nothing — give it a `type`, `properties`, \
             `items`, `required`, `enum` or `const`"
        );
    }
    if let Some(properties) = object.get("properties") {
        let Some(properties) = properties.as_object() else {
            bail!("usage: `properties` is an object of property schemas");
        };
        for schema in properties.values() {
            validate_node(schema, level + 1)?;
        }
    }
    if let Some(required) = object.get("required") {
        let Some(required) = required.as_array() else {
            bail!("usage: `required` is a list of property names");
        };
        let declared = object.get("properties").and_then(Value::as_object);
        for name in required {
            let Some(name) = name.as_str() else {
                bail!("usage: `required` is a list of property names");
            };
            let known = declared.is_some_and(|properties| properties.contains_key(name));
            if !known {
                bail!("usage: `required` names `{name}`, which no `properties` entry declares");
            }
        }
    }
    if let Some(items) = object.get("items") {
        if items.is_array() {
            bail!("usage: array-form `items` is not supported — give one schema for every item");
        }
        validate_node(items, level + 1)?;
    }
    if let Some(additional) = object.get("additionalProperties")
        && !additional.is_boolean()
    {
        bail!("usage: `additionalProperties` is `true` or `false`, not a schema");
    }
    Ok(())
}

/// The types a node accepts — empty when it declares none, which accepts every type.
fn declared_types(object: &Map<String, Value>) -> Result<Vec<&'static str>> {
    let Some(declared) = object.get("type") else {
        return Ok(Vec::new());
    };
    let names = match declared {
        Value::String(name) => vec![name.as_str()],
        Value::Array(names) if !names.is_empty() => names
            .iter()
            .map(|name| {
                name.as_str().ok_or_else(|| {
                    anyhow::anyhow!("usage: `type` is a type name or a list of them")
                })
            })
            .collect::<Result<Vec<_>>>()?,
        _ => bail!("usage: `type` is a type name or a non-empty list of them"),
    };
    names
        .into_iter()
        .map(|name| {
            TYPES
                .iter()
                .copied()
                .find(|known| *known == name)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "usage: `type` names `{name}`, which is not one of {}",
                        TYPES.join(", ")
                    )
                })
        })
        .collect()
}

/// Whether a node requires anything of the document. A node that names a `type` requires that much
/// — except one that accepts `object` with an open `additionalProperties` door and declares nothing
/// else, which permits any object and is refused like an empty node. An empty `properties`,
/// `required` or `enum` declares nothing.
fn constrains_something(object: &Map<String, Value>, types: &[&str]) -> bool {
    let declares = |key: &str| match object.get(key) {
        Some(Value::Array(list)) => !list.is_empty(),
        Some(Value::Object(map)) => !map.is_empty(),
        Some(_) => true,
        None => false,
    };
    let names_something = !types.is_empty()
        || object.contains_key("items")
        || object.contains_key("const")
        || ["properties", "required", "enum"]
            .iter()
            .any(|key| declares(key));
    if !names_something {
        return false;
    }
    let any_object = types.contains(&"object")
        && object.get("additionalProperties") == Some(&Value::Bool(true))
        && !["properties", "required", "enum"]
            .iter()
            .any(|key| declares(key))
        && !object.contains_key("const");
    !any_object
}

/// Check one value against one node, naming what is wrong in the re-ask's own words.
fn check_node(node: &Value, value: &Value, path: &str) -> std::result::Result<(), String> {
    let object = node
        .as_object()
        .ok_or_else(|| format!("{path}is not a schema"))?;
    if let Some(expected) = object.get("const")
        && expected != value
    {
        return Err(format!(
            "{path}must be {}, the only allowed value",
            render(expected)
        ));
    }
    if let Some(values) = object.get("enum").and_then(Value::as_array)
        && !values.contains(value)
    {
        let allowed: Vec<String> = values.iter().map(render).collect();
        return Err(format!("{path}must be one of {}", allowed.join(", ")));
    }
    let types = declared_types(object).expect("a validated schema's `type`");
    if !types.is_empty() && !types.iter().any(|kind| matches_type(kind, value)) {
        return Err(format!(
            "{path}must be {}, not {}",
            types.join(" or "),
            kind(value)
        ));
    }
    if let Some(entries) = value.as_object() {
        if let Some(required) = object.get("required").and_then(Value::as_array) {
            for name in required.iter().filter_map(Value::as_str) {
                if !entries.contains_key(name) {
                    return Err(format!("{path}is missing required property `{name}`"));
                }
            }
        }
        let declared = object.get("properties").and_then(Value::as_object);
        if let Some(properties) = declared {
            for (name, schema) in properties {
                if let Some(entry) = entries.get(name) {
                    check_node(schema, entry, &format!("{path}`{name}` "))?;
                }
            }
        }
        if object.get("additionalProperties") == Some(&Value::Bool(false)) {
            for name in entries.keys() {
                if !declared.is_some_and(|properties| properties.contains_key(name)) {
                    return Err(format!(
                        "{path}has property `{name}`, which the schema does not allow"
                    ));
                }
            }
        }
    }
    if let (Some(items), Some(entries)) = (object.get("items"), value.as_array()) {
        for (index, entry) in entries.iter().enumerate() {
            check_node(items, entry, &format!("{path}item {index} "))?;
        }
    }
    Ok(())
}

/// Whether a JSON value is of one declared `type`.
fn matches_type(kind: &str, value: &Value) -> bool {
    match kind {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => false,
    }
}

/// What a JSON value is, for a mismatch message.
fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// A value's canonical JSON text. `serde_json` orders object keys, so two schemas that differ only
/// in key order or whitespace render the same. A `Value` always serializes: a fallback here would
/// silently make every schema hash alike and render an empty one into the prompt.
fn render(value: &Value) -> String {
    serde_json::to_string(value).expect("a JSON value always serializes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[expect(clippy::needless_pass_by_value)] // every call site hands over a fresh `json!` literal
    fn schema(value: Value) -> Extract {
        Extract::parse(&value).expect("the schema is accepted")
    }

    #[expect(clippy::needless_pass_by_value)] // every call site hands over a fresh `json!` literal
    fn refused(value: Value) -> String {
        Extract::parse(&value)
            .expect_err("the schema is refused")
            .to_string()
    }

    #[test]
    fn the_enforced_subset_is_accepted() {
        for value in [
            json!({ "type": "string" }),
            json!({ "type": ["string", "null"] }),
            json!({ "type": "integer" }),
            json!({ "type": "number" }),
            json!({ "type": "boolean" }),
            json!({ "type": "array", "items": { "type": "string" } }),
            json!({ "type": "object", "properties": { "a": { "type": "string" } }, "required": ["a"] }),
            json!({ "type": "object", "properties": { "a": { "type": "string" } }, "additionalProperties": false }),
            json!({ "enum": ["a", "b"] }),
            json!({ "const": 3 }),
            json!({ "type": "object", "properties": { "a": { "type": "string" } }, "additionalProperties": true }),
        ] {
            let _ = schema(value);
        }
    }

    #[test]
    fn tolerated_keywords_are_ignored() {
        let value = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "x",
            "title": "t",
            "description": "d",
            "default": 1,
            "examples": [1],
            "format": "date",
            "pattern": "^a",
            "minimum": 1,
            "maxLength": 3,
            "type": "integer"
        });
        let schema = schema(value);
        assert!(schema.check(&json!(4)).is_ok(), "a bound is not enforced");
    }

    #[test]
    fn a_stored_schema_reads_back_and_an_unreadable_one_is_absent() {
        let schema = schema(json!({ "type": "string" }));
        assert_eq!(
            Extract::from_stored(Some(schema.canonical())),
            Some(schema),
            "a stored schema reads back as itself"
        );
        assert_eq!(Extract::from_stored(None), None);
        assert_eq!(Extract::from_stored(Some("{not json".to_string())), None);
        assert_eq!(
            Extract::from_stored(Some(r#"{"type":"object","allOf":[]}"#.to_string())),
            None,
            "a stored schema this build would refuse is not applied as a looser one"
        );
    }

    #[test]
    fn unsupported_keywords_are_refused_by_name() {
        for (value, named) in [
            (json!({ "$ref": "#/a" }), "$ref"),
            // A `$`-keyword that references nothing is refused as unsupported, not as a reference.
            (json!({ "type": "object", "$comment": "x" }), "$comment"),
            (json!({ "type": "object", "allOf": [] }), "allOf"),
            (json!({ "type": "object", "anyOf": [] }), "anyOf"),
            (json!({ "type": "object", "oneOf": [] }), "oneOf"),
            (json!({ "type": "object", "not": {} }), "not"),
            (
                json!({ "type": "object", "patternProperties": {} }),
                "patternProperties",
            ),
            (
                json!({ "type": "object", "propertyNames": {} }),
                "propertyNames",
            ),
            (json!({ "type": "array", "contains": {} }), "contains"),
            (
                json!({ "type": "object", "dependencies": {} }),
                "dependencies",
            ),
            (
                json!({ "type": "object", "dependentRequired": {} }),
                "dependentRequired",
            ),
            (
                json!({ "type": "object", "dependentSchemas": {} }),
                "dependentSchemas",
            ),
            (
                json!({ "type": "object", "unevaluatedProperties": false }),
                "unevaluatedProperties",
            ),
            (
                json!({ "type": "array", "unevaluatedItems": false }),
                "unevaluatedItems",
            ),
            (json!({ "type": "object", "$defs": {} }), "$defs"),
            (
                json!({ "type": "object", "definitions": {} }),
                "definitions",
            ),
            (json!({ "type": "array", "prefixItems": [] }), "prefixItems"),
            (
                json!({ "type": "array", "additionalItems": false }),
                "additionalItems",
            ),
            (json!({ "type": "object", "if": {} }), "if"),
            (json!({ "type": "object", "then": {} }), "then"),
            (json!({ "type": "object", "else": {} }), "else"),
        ] {
            let err = refused(value);
            assert!(err.contains(named), "the refusal names {named}: {err}");
        }
    }

    #[test]
    fn a_schema_that_constrains_nothing_is_refused() {
        for value in [
            json!({}),
            json!({ "title": "only annotations" }),
            json!({ "type": "object", "additionalProperties": true }),
            // The union spelling opens the same door.
            json!({ "type": ["object", "null"], "additionalProperties": true }),
            // An empty declaration declares nothing.
            json!({ "required": [] }),
            json!({ "type": "object", "properties": {}, "additionalProperties": true }),
        ] {
            assert!(refused(value).contains("constrains nothing"));
        }
        // The same node without the open door constrains: only `{}` is allowed through.
        assert!(
            Extract::from_arg(&json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false,
            }))
            .is_ok()
        );
        // So does an `enum`, even under an open door: it names the values allowed.
        assert!(
            Extract::from_arg(&json!({
                "type": "object",
                "additionalProperties": true,
                "enum": [{ "a": 1 }],
            }))
            .is_ok()
        );
    }

    #[test]
    fn a_node_must_accept_the_shape_it_declares() {
        assert!(
            refused(json!({ "type": "string", "properties": { "a": { "type": "string" } } }))
                .contains("must accept `object`")
        );
        assert!(
            refused(json!({ "type": "object", "items": { "type": "string" } }))
                .contains("must accept `array`")
        );
        assert!(
            refused(json!({ "type": "object", "required": ["a"] }))
                .contains("no `properties` entry declares")
        );
        assert!(
            refused(json!({ "type": "array", "items": [{ "type": "string" }] }))
                .contains("array-form `items`")
        );
        assert!(
            refused(json!({ "type": "object", "additionalProperties": { "type": "string" } }))
                .contains("not a schema")
        );
        assert!(refused(json!(true)).contains("boolean subschema"));
        assert!(refused(json!({ "type": "bogus" })).contains("not one of"));
        // An empty `enum` is told what is wrong with it, not that the node constrains nothing.
        assert!(refused(json!({ "enum": [] })).contains("`enum` lists at least one allowed value"));
    }

    #[test]
    fn the_nesting_bound_counts_the_root() {
        // Eight levels: the root plus seven nested `items`.
        let mut node = json!({ "type": "string" });
        for _ in 0..(MAX_DEPTH - 1) {
            node = json!({ "type": "array", "items": node });
        }
        let _ = schema(node.clone());
        // One more level is refused.
        let deeper = json!({ "type": "array", "items": node });
        assert!(refused(deeper).contains("nests at most"));
    }

    #[test]
    fn the_text_cap_applies_to_every_form() {
        let oversized = json!({
            "type": "string",
            "description": "x".repeat(MAX_TEXT_BYTES),
        });
        // The text form, the object form and an envelope-carried value obey one cap.
        assert!(Extract::from_text(&oversized.to_string()).is_err());
        assert!(Extract::from_arg(&oversized).is_err());
        assert!(serde_json::from_value::<Extract>(oversized).is_err());
        assert!(Extract::from_text("  ").is_err(), "empty is refused");
        assert!(Extract::from_text("not json").is_err());
    }

    #[test]
    fn key_order_and_whitespace_do_not_change_a_schema() {
        let first = Extract::from_text(r#"{"type":"object","properties":{"a":{"type":"string"}}}"#)
            .expect("accepted");
        let second = Extract::from_text(
            "{ \"properties\": { \"a\": { \"type\": \"string\" } }, \"type\": \"object\" }",
        )
        .expect("accepted");
        assert_eq!(first, second);
        assert_eq!(first.canonical(), second.canonical());
    }

    #[test]
    fn the_check_reports_what_is_wrong() {
        let schema = schema(json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "tags": { "type": "array", "items": { "type": "string" } },
                "kind": { "enum": ["a", "b"] }
            },
            "required": ["name"],
            "additionalProperties": false
        }));
        assert!(
            schema
                .check(&json!({ "name": "x", "tags": ["t"], "kind": "a" }))
                .is_ok()
        );
        assert_eq!(
            schema.check(&json!({})).unwrap_err(),
            "the answer is missing required property `name`"
        );
        assert_eq!(
            schema.check(&json!({ "name": 1 })).unwrap_err(),
            "the answer `name` must be string, not number"
        );
        assert_eq!(
            schema
                .check(&json!({ "name": "x", "tags": [1] }))
                .unwrap_err(),
            "the answer `tags` item 0 must be string, not number"
        );
        assert_eq!(
            schema
                .check(&json!({ "name": "x", "kind": "c" }))
                .unwrap_err(),
            "the answer `kind` must be one of \"a\", \"b\""
        );
        assert_eq!(
            schema
                .check(&json!({ "name": "x", "extra": 1 }))
                .unwrap_err(),
            "the answer has property `extra`, which the schema does not allow"
        );
        assert_eq!(
            schema.check(&json!([1])).unwrap_err(),
            "the answer must be object, not array"
        );
    }

    #[test]
    fn a_union_type_accepts_every_listed_type() {
        let schema = schema(json!({ "type": ["string", "null"] }));
        assert!(schema.check(&json!(null)).is_ok());
        assert!(schema.check(&json!("x")).is_ok());
        assert!(schema.check(&json!(1)).is_err());
    }

    /// The argument arrives in the two forms a call can carry it: the object an agent passes, and
    /// the JSON text the command line passes. Anything that is not a schema object is refused.
    #[test]
    fn the_argument_is_read_from_an_object_or_from_json_text() {
        let object = json!({ "type": "object", "properties": { "a": { "type": "string" } } });
        let from_object = Extract::from_arg(&object).expect("an object is accepted");
        let from_text =
            Extract::from_arg(&json!(object.to_string())).expect("its JSON text is accepted");
        assert_eq!(from_object, from_text, "both forms are one schema");
        for refused in [json!([{ "type": "string" }]), json!(true), json!(3)] {
            assert!(
                Extract::from_arg(&refused).is_err(),
                "must refuse {refused}"
            );
        }
    }

    /// The transformation asks in ONE user turn carrying both the answer and the schema. Some model
    /// families refuse a conversation whose first non-system message is not `user`, and a refusal
    /// here fails open — the feature would silently do nothing on those models.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn the_transformation_asks_in_one_user_turn() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = std::sync::Arc::new(crate::util::test::FakeProvider::new().ok(r#"{"a":"one"}"#));
        let _provider_guard = crate::util::test::install_fake_provider(fake.clone());
        let schema =
            schema(json!({ "type": "object", "properties": { "a": { "type": "string" } } }));
        let step = StepContext {
            role: crate::Role::Worker,
            workspace: "ws",
            conversation: None,
            parent: None,
        };
        assert_eq!(
            answer_as_json_reporting(Some(&schema), &step, "the whole answer").await,
            (r#"{"a":"one"}"#.to_string(), true)
        );
        // `request_messages` holds each request's messages as `Debug`, joined with NUL — one turn
        // has no NUL in it, and a quoted string is matched in its escaped form.
        let messages = fake.request_messages.lock().unwrap();
        let sent = &messages[0];
        assert!(
            !sent.contains('\u{0}'),
            "one turn, not a conversation: {sent}"
        );
        assert!(
            sent.contains("role: User"),
            "the turn is the user's: {sent}"
        );
        assert!(sent.contains("the whole answer"), "{sent}");
        let canonical = format!("{:?}", schema.canonical());
        assert!(
            sent.contains(canonical.trim_matches('"')),
            "the schema is rendered into the same turn: {sent}"
        );
    }

    /// The transformation runs where the step ran: the model and provider recorded for the step's
    /// conversation, not the role's configured model and a freshly picked provider. A conversation
    /// with nothing recorded falls back to the role's model (the test above).
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn the_transformation_runs_on_the_steps_recorded_model_and_provider() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = std::sync::Arc::new(crate::util::test::FakeProvider::new().ok(r#"{"a":"one"}"#));
        let _provider_guard = crate::util::test::install_fake_provider(fake.clone());
        let conversation = crate::session::worker_branch_agent_id("schema_model");
        crate::session::store()
            .set_branch(&conversation, None, None, None, None, None)
            .await
            .expect("the step's conversation exists");
        let stored = crate::providers::selection::StoredProvider {
            model: "step-model".to_string(),
            provider: "step-provider".to_string(),
        };
        assert!(
            crate::session::store()
                .set_provider(&conversation, Some(&stored))
                .await
                .expect("the provider is recorded")
        );
        let schema =
            schema(json!({ "type": "object", "properties": { "a": { "type": "string" } } }));
        let step = StepContext {
            role: crate::Role::Worker,
            workspace: "ws",
            conversation: Some(&conversation),
            parent: None,
        };
        assert_eq!(
            answer_as_json_reporting(Some(&schema), &step, "the answer").await,
            (r#"{"a":"one"}"#.to_string(), true)
        );
        let sent = fake.sent.lock().unwrap();
        assert_eq!(sent[0].model, "step-model");
        assert_eq!(sent[0].provider.as_deref(), Some("step-provider"));
    }
}
