//! Per-endpoint data-sharing policy for the OpenRouter benchmark: does using this endpoint require
//! a non-default setting that lets the provider share the prompts — train on them, or publish them?
//!
//! No documented OpenRouter resource (`/models`, `/models/{author}/{slug}/endpoints`, `/key`)
//! carries that policy, so the source is the model page payload: the undocumented, unversioned
//! Next.js RSC document at `https://openrouter.ai/{model}`, whose endpoint blocks carry a
//! `data_policy` object (`training`, `trainingOpenRouter`, `canPublish`, and retention fields that
//! are not this axis) beside the endpoint's `provider_slug` — the same slug the endpoints API
//! reports as an endpoint `tag`. The column is neutral: it says what has to be switched on to use
//! the endpoint at this price, never whether the endpoint is good, safe or recommended.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::LazyLock;
use std::time::Duration;

use serde_json::Value;

use crate::util::http::install_ring_provider;

/// The `source` value of a snapshot whose page could not be read, or that carried nothing usable.
pub(crate) const UNAVAILABLE: &str = "unavailable";

const PAGE_BASE: &str = "https://openrouter.ai";

/// One decoded `self.__next_f.push([1,"…"])` payload literal: a JSON string, so escaped quotes,
/// newlines and backslashes are the only way its content can look; `\\.` swallows each escape
/// pair so a `\"` inside the payload cannot end the literal early.
static PAYLOAD_CHUNK: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r#"self\.__next_f\.push\(\[1,\s*("(?:[^"\\]|\\.)*")"#)
        .expect("the RSC payload-literal regex compiles")
});

/// Either a block's slug or the start of its policy object. One pass over the payload sees both in
/// document order, which is what pairs a `data_policy` with the slug that precedes it.
static PAYLOAD_MARKER: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r#""provider_slug"\s*:\s*"([^"]*)"|"data_policy"\s*:\s*\{"#)
        .expect("the data-policy marker regex compiles")
});

/// One provider slug's strictest policy reading.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PolicyEntry {
    /// The payload's `provider_slug`; a benchmark row's endpoint tag is matched against it.
    pub tag: String,
    /// Any block for this slug states `training` or `trainingOpenRouter`; `None` when no block
    /// stated either.
    pub trains: Option<bool>,
    /// Any block for this slug states `canPublish`; `None` when no block stated it.
    pub publishes: Option<bool>,
}

/// The privacy snapshot of one run.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PrivacySnapshot {
    /// The canonical model id the page was read for; `None` when the source was unavailable.
    pub source: Option<String>,
    pub entries: Vec<PolicyEntry>,
}

/// Whether a benchmark row's endpoint requires opting into data sharing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OptIn {
    Yes,
    No,
    Unknown,
}

impl OptIn {
    #[must_use]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Yes => "yes",
            Self::No => "no",
            Self::Unknown => "unknown",
        }
    }
}

impl PrivacySnapshot {
    #[must_use]
    pub(crate) fn unavailable() -> Self {
        Self {
            source: None,
            entries: Vec::new(),
        }
    }

    /// The `source` the artifacts carry: the model id, or [`UNAVAILABLE`].
    #[must_use]
    pub(crate) fn source_label(&self) -> &str {
        self.source.as_deref().unwrap_or(UNAVAILABLE)
    }

    /// The reading for a benchmark row's endpoint tag.
    ///
    /// The entry is the LONGEST one whose tag equals the row's tag or is a whole-segment prefix of
    /// it (the row continues with `/` right after the entry tag) — never a partial-word prefix, so
    /// `acme` does not answer for `acme-other`. A known `true` settles the row on its own, so a
    /// block whose training flag is `true` and whose publishing flag is missing reads
    /// [`OptIn::Yes`].
    #[must_use]
    pub(crate) fn value_for(&self, tag: &str) -> OptIn {
        let Some(entry) = self.entry_for(tag) else {
            return OptIn::Unknown;
        };
        match (entry.trains, entry.publishes) {
            (Some(true), _) | (_, Some(true)) => OptIn::Yes,
            (Some(false), Some(false)) => OptIn::No,
            _ => OptIn::Unknown,
        }
    }

    /// The entry [`Self::value_for`] reads, or `None` when no entry answers `tag`.
    fn entry_for(&self, tag: &str) -> Option<&PolicyEntry> {
        self.entries
            .iter()
            .filter(|entry| {
                tag == entry.tag
                    || tag
                        .strip_prefix(entry.tag.as_str())
                        .is_some_and(|rest| rest.starts_with('/'))
            })
            .max_by_key(|entry| entry.tag.len())
    }
}

/// Fetch the model page and extract its per-slug policy entries. Never fails: any failure
/// degrades to [`PrivacySnapshot::unavailable`].
pub(crate) async fn resolve(model: &str, endpoint_tags: &[String]) -> PrivacySnapshot {
    let Some(entries) = fetch_entries(model).await else {
        return PrivacySnapshot::unavailable();
    };
    settle(model.to_string(), entries, endpoint_tags)
}

/// The page URL for a model identifier. Segments are percent-encoded by `path_segments_mut`, the
/// way the endpoints path does it; an id without a slash routes with an empty slug.
#[must_use]
fn page_url(model: &str) -> String {
    let (author, slug) = model.split_once('/').unwrap_or((model, ""));
    let mut url = reqwest::Url::parse(PAGE_BASE).expect("the OpenRouter page base is a valid URL");
    url.path_segments_mut()
        .expect("an https URL has path segments")
        .push(author)
        .push(slug);
    url.to_string()
}

/// Turn parsed entries into the run's snapshot: a source that produced no entry, or no block for
/// any of the run's endpoint tags, is `unavailable` — a dead source must never look like a run
/// where everything happened to be fine. A page whose blocks resolve no endpoint is the
/// unavailable case; a page whose blocks resolve but state no training/publishing flag is not
/// (those rows are `unknown`, which is a row's answer, not the source's).
fn settle(source: String, entries: Vec<PolicyEntry>, endpoint_tags: &[String]) -> PrivacySnapshot {
    let snapshot = PrivacySnapshot {
        source: Some(source),
        entries,
    };
    if snapshot.entries.is_empty()
        || !endpoint_tags
            .iter()
            .any(|tag| snapshot.entry_for(tag).is_some())
    {
        return PrivacySnapshot::unavailable();
    }
    snapshot
}

/// GET the model page and parse it. `None` carries no detail because the snapshot is the report's
/// only channel; the reason goes to stderr, because this command runs before the log is live and a
/// dead source otherwise shows up as a bare `unavailable`.
async fn fetch_entries(model: &str) -> Option<Vec<PolicyEntry>> {
    // reqwest 0.13 ships rustls without a default provider; the client cannot be built before it.
    install_ring_provider();
    let url = page_url(model);
    let client = match reqwest::Client::builder()
        .user_agent(crate::util::http::USER_AGENT)
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(30))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            warn_unavailable(&url, &error);
            return None;
        }
    };
    let response = match client.get(&url).send().await {
        Ok(response) => response,
        Err(error) => {
            warn_unavailable(&url, &error);
            return None;
        }
    };
    let status = response.status();
    if !status.is_success() {
        warn_unavailable(&url, &format!("HTTP {status}"));
        return None;
    }
    match response.text().await {
        Ok(body) => Some(parse_entries(&body)),
        Err(error) => {
            warn_unavailable(&url, &error);
            None
        }
    }
}

/// The run continues on an unavailable privacy source, so the reason is a warning, not an error.
fn warn_unavailable(url: &str, reason: &dyn std::fmt::Display) {
    eprintln!("Warning: privacy policy source unavailable for {url}: {reason}");
}

/// Decode the RSC payload out of a model page, or hand back the document as-is when no chunk
/// decodes (a plain page simply yields no blocks).
fn payload_text(text: &str) -> Cow<'_, str> {
    let mut decoded = String::new();
    let mut any = false;
    for captures in PAYLOAD_CHUNK.captures_iter(text) {
        let Some(literal) = captures.get(1) else {
            continue;
        };
        if let Ok(chunk) = serde_json::from_str::<String>(literal.as_str()) {
            decoded.push_str(&chunk);
            any = true;
        }
    }
    if any {
        Cow::Owned(decoded)
    } else {
        Cow::Borrowed(text)
    }
}

/// Every `(slug, data_policy)` pair the payload carries, in document order. A block without a
/// preceding slug is skipped: the flags alone name no endpoint. The object is parsed by
/// `serde_json` itself, so a `}` inside a value cannot end it early and a payload cut off
/// mid-object simply ends the walk.
fn policy_blocks(payload: &str) -> Vec<(String, Value)> {
    let mut blocks = Vec::new();
    let mut slug: Option<&str> = None;
    for captures in PAYLOAD_MARKER.captures_iter(payload) {
        if let Some(found) = captures.get(1) {
            slug = Some(found.as_str());
            continue;
        }
        let Some(marker) = captures.get(0) else {
            continue;
        };
        // The marker ends just past the `{`, so the object starts one byte back.
        let Some(policy) = first_json_value(&payload[marker.end() - 1..]) else {
            break;
        };
        if let Some(slug) = slug {
            blocks.push((slug.to_string(), policy));
        }
    }
    blocks
}

/// The first complete JSON value at the start of `text` — the same first-value `serde_json` parse
/// as `chrome::contract::parse_first`, over a `&str` slice.
fn first_json_value(text: &str) -> Option<Value> {
    serde_json::Deserializer::from_str(text)
        .into_iter::<Value>()
        .next()?
        .ok()
}

/// One policy flag, or `None` when the block does not state it as a boolean: a missing key, a
/// `null` and a non-boolean value all read as "not stated", so a shape change degrades to
/// `unknown` instead of to `false`.
fn flag(policy: &Value, key: &str) -> Option<bool> {
    policy.get(key).and_then(Value::as_bool)
}

/// One axis folded across every block that carries a slug.
#[derive(Default)]
struct Axis {
    /// Some block stated `true` — the strictest reading wins, whatever the other blocks say.
    any_true: bool,
    /// Some block left the flag null, missing or non-boolean. A false claim about the axis needs
    /// every block to state `false`, so one unstated flag keeps the whole axis `None` — never `no`.
    any_unstated: bool,
}

impl Axis {
    fn add(&mut self, value: Option<bool>) {
        match value {
            Some(value) => self.any_true |= value,
            None => self.any_unstated = true,
        }
    }

    const fn reading(&self) -> Option<bool> {
        if self.any_true {
            Some(true)
        } else if self.any_unstated {
            None
        } else {
            Some(false)
        }
    }
}

#[derive(Default)]
struct Axes {
    trains: Axis,
    publishes: Axis,
}

/// Aggregate the page's policy blocks into one entry per provider slug, sorted by tag so the
/// snapshot and the tests are deterministic. Retention fields are not an axis: blocks that differ
/// only in `retainsPrompts`/`retentionDays` are the same reading, so they must not produce `None`.
fn parse_entries(text: &str) -> Vec<PolicyEntry> {
    let payload = payload_text(text);
    let mut by_slug: BTreeMap<String, Axes> = BTreeMap::new();
    for (slug, policy) in policy_blocks(&payload) {
        let axes = by_slug.entry(slug).or_default();
        axes.trains.add(flag(&policy, "training"));
        axes.trains.add(flag(&policy, "trainingOpenRouter"));
        axes.publishes.add(flag(&policy, "canPublish"));
    }
    by_slug
        .into_iter()
        .map(|(tag, axes)| PolicyEntry {
            tag,
            trains: axes.trains.reading(),
            publishes: axes.publishes.reading(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed capture of `deepseek/deepseek-v4.1-flash-1017`: 101 policy blocks over 49
    /// provider slugs, plus the provider-level copies that must not be read.
    const FIXTURE: &str = include_str!("fixtures/model_page.html");

    const FIXTURE_TAGS: [&str; 49] = [
        "alibaba",
        "atlas-cloud/fp8",
        "baidu/fp8",
        "baseten",
        "baseten/fast",
        "cloudflare",
        "cohere/fp8",
        "coreweave/fp8",
        "crusoe/fp8",
        "deepinfra/fp8",
        "deepinfra/fp8/flex",
        "deepinfra/fp8/priority",
        "deepseek",
        "dekallm",
        "digitalocean",
        "fireworks",
        "fireworks/spot",
        "fireworks/us",
        "gmicloud/fp8",
        "inference-net/fp8",
        "inference-net/us",
        "ionstream",
        "ionstream/flex",
        "makora/fp8",
        "mistral/fp8",
        "modal",
        "morph/fast",
        "morph/fp8",
        "near-ai",
        "nebius/fp8",
        "novita/fp8",
        "open-inference/fp4",
        "parasail/fp8",
        "particle-ai/us",
        "particle-ai/us/flex",
        "primalabs",
        "primeintellect",
        "reka",
        "relace/fp4",
        "river",
        "sail-research/fp4",
        "sail-research/fp4/us",
        "siliconflow/fp8",
        "stepfun/fp8",
        "streamlake/fp8",
        "together",
        "venice/fp8",
        "wafer/mxfp4",
        "wafer/us",
    ];

    fn snapshot_from(text: &str, endpoint_tags: &[&str]) -> PrivacySnapshot {
        let tags: Vec<String> = endpoint_tags.iter().map(|tag| (*tag).to_string()).collect();
        settle("synthetic/model".to_string(), parse_entries(text), &tags)
    }

    #[test]
    fn the_fixture_collapses_101_blocks_into_49_slug_entries() {
        assert_eq!(policy_blocks(&payload_text(FIXTURE)).len(), 101);
        let entries = parse_entries(FIXTURE);
        let tags: Vec<&str> = entries.iter().map(|entry| entry.tag.as_str()).collect();
        assert_eq!(tags, FIXTURE_TAGS);
    }

    #[test]
    fn every_fixture_slug_answers_and_the_official_deepseek_endpoint_is_the_only_opt_in() {
        let snapshot = snapshot_from(FIXTURE, &FIXTURE_TAGS);
        assert_eq!(snapshot.source.as_deref(), Some("synthetic/model"));
        for tag in FIXTURE_TAGS {
            let expected = if tag == "deepseek" {
                OptIn::Yes
            } else {
                OptIn::No
            };
            assert_eq!(snapshot.value_for(tag), expected, "row {tag}");
        }
        let deepseek = snapshot
            .entries
            .iter()
            .find(|entry| entry.tag == "deepseek")
            .expect("the fixture carries the official DeepSeek endpoint");
        assert_eq!(deepseek.trains, Some(true));
        assert_eq!(deepseek.publishes, Some(false));
        assert_eq!(snapshot.value_for("deepseek/fp8"), OptIn::Yes);
        // A row the page never mentioned is unknown, never "no".
        assert_eq!(snapshot.value_for("not-a-provider"), OptIn::Unknown);
    }

    #[test]
    fn repeated_blocks_differing_only_in_retention_keep_the_row_settled() {
        let snapshot = snapshot_from(FIXTURE, &FIXTURE_TAGS);
        for tag in ["sail-research/fp4", "together", "wafer/us"] {
            assert_eq!(snapshot.value_for(tag), OptIn::No, "row {tag}");
        }
    }

    #[test]
    fn a_row_takes_the_longest_whole_segment_prefix() {
        let payload = concat!(
            r#"{"provider_slug":"acme","data_policy":{"training":false,"trainingOpenRouter":false,"canPublish":false}}"#,
            r#"{"provider_slug":"acme/fp8","data_policy":{"training":true,"trainingOpenRouter":false,"canPublish":false}}"#,
        );
        let snapshot = snapshot_from(payload, &["acme/fp8/priority"]);
        assert_eq!(snapshot.entries.len(), 2);
        assert_eq!(snapshot.value_for("acme/fp8/priority"), OptIn::Yes);
        assert_eq!(snapshot.value_for("acme/fp8"), OptIn::Yes);
        assert_eq!(snapshot.value_for("acme"), OptIn::No);
        // No partial-word prefix: `acme` does not answer for `acme-other` or `acmefp8`.
        assert_eq!(snapshot.value_for("acme-other"), OptIn::Unknown);
        assert_eq!(snapshot.value_for("acmefp8"), OptIn::Unknown);
    }

    #[test]
    fn one_block_that_trains_settles_the_whole_slug() {
        let payload = concat!(
            r#"{"provider_slug":"acme","data_policy":{"training":false,"trainingOpenRouter":false,"canPublish":false}}"#,
            r#"{"provider_slug":"acme","data_policy":{"training":true,"trainingOpenRouter":false,"canPublish":false}}"#,
        );
        let entries = parse_entries(payload);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].trains, Some(true));
        assert_eq!(entries[0].publishes, Some(false));
    }

    /// `no` needs every flag of the axis stated false: a null or absent one keeps the axis
    /// unknown, because that is the flag that might allow training.
    #[test]
    fn an_unsettled_training_flag_keeps_the_axis_unknown() {
        for payload in [
            r#"{"provider_slug":"acme","data_policy":{"training":null,"trainingOpenRouter":false,"canPublish":false}}"#,
            r#"{"provider_slug":"acme","data_policy":{"training":false,"canPublish":false}}"#,
        ] {
            let entries = parse_entries(payload);
            assert_eq!(entries[0].trains, None, "payload {payload}");
            assert_eq!(entries[0].publishes, Some(false), "payload {payload}");
            let snapshot = PrivacySnapshot {
                source: Some("synthetic/model".to_string()),
                entries,
            };
            assert_eq!(snapshot.value_for("acme"), OptIn::Unknown, "{payload}");
        }
    }

    #[test]
    fn a_stated_true_settles_a_row_whose_other_axis_is_missing() {
        let payload = r#"{"provider_slug":"acme","data_policy":{"training":true}}"#;
        let entries = parse_entries(payload);
        assert_eq!(entries[0].publishes, None);
        let snapshot = PrivacySnapshot {
            source: Some("synthetic/model".to_string()),
            entries,
        };
        assert_eq!(snapshot.value_for("acme"), OptIn::Yes);
    }

    #[test]
    fn a_brace_inside_a_string_value_does_not_end_the_object() {
        let payload = concat!(
            r#"{"provider_slug":"acme","data_policy":{"training":false,"trainingOpenRouter":false,"#,
            r#""canPublish":false,"#,
            r#""termsOfServiceURL":"https://example.invalid/?q=}"}}"#,
        );
        let entries = parse_entries(payload);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].trains, Some(false));
        assert_eq!(entries[0].publishes, Some(false));
    }

    /// A page whose blocks DO resolve the endpoints, but state no training/publishing flag, is a
    /// live source with `unknown` rows — never an unavailable source, which would drop the
    /// entries and claim the fetch failed.
    #[test]
    fn a_resolved_block_without_flags_is_unknown_not_unavailable() {
        let payload = r#"{"provider_slug":"acme","data_policy":{"retainsPrompts":true}}"#;
        let snapshot = snapshot_from(payload, &["acme"]);
        assert_eq!(snapshot.source.as_deref(), Some("synthetic/model"));
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(snapshot.value_for("acme"), OptIn::Unknown);
    }

    #[test]
    fn a_page_without_usable_blocks_is_unavailable() {
        let snapshot = snapshot_from("<html>nothing</html>", &["acme"]);
        assert!(snapshot.source.is_none());
        assert!(snapshot.entries.is_empty());
        let empty = PrivacySnapshot::unavailable();
        assert!(empty.source.is_none());
        assert!(empty.entries.is_empty());
    }

    #[test]
    fn a_snapshot_matching_none_of_the_endpoint_tags_is_unavailable() {
        let payload =
            r#"{"provider_slug":"acme","data_policy":{"training":false,"canPublish":false}}"#;
        let snapshot = snapshot_from(payload, &["other", "other/fp8"]);
        assert!(snapshot.source.is_none());
        assert!(snapshot.entries.is_empty());
    }

    #[test]
    fn the_page_url_percent_encodes_the_model_segments() {
        assert_eq!(
            page_url("acme/model 1"),
            "https://openrouter.ai/acme/model%201"
        );
        assert_eq!(page_url("acme/m1"), "https://openrouter.ai/acme/m1");
        // An id without a slash routes with an empty slug, like the endpoints path.
        assert_eq!(page_url("acme"), "https://openrouter.ai/acme/");
    }
}
