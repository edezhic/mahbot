//! Which provider serves a model, and when a conversation may leave it.
//!
//! The per-model setting (Settings → Provider → Models → Provider Priority) is the product's own
//! priority list, and a conversation holds exactly ONE entry of it: **choosing** takes the first
//! candidate with no outage on record, while **leaving** the entry it holds needs an outage that has
//! outlasted [`MOVE_AFTER`] — a blip is waited out, because the prompt cache lives at the provider
//! that wrote it. Selection therefore happens in exactly three places: no stored provider yet → the
//! first available from the top; the stored one available or its outage young → keep it, with no
//! re-evaluation against the top; its outage outlasted → select again from the top. Exhaustion is not
//! a special case: choosing falls back to the top entry, so the call is still attempted and the
//! conversation keeps that provider for the next run. Only a listed provider can be held, so an edit
//! that leaves it listed — entries added above it, reordered — changes nothing, while deleting it
//! from the list re-selects on the next call.
//!
//! Availability is learned from our own calls (see [`crate::providers::chat_scoped`]) and lives in
//! memory only, so a restart gives every provider a fresh chance. Every retry ladder of a run applies
//! the same switching rule — [`ProviderSelection::switch_on_failure`] for the ones that own a
//! selection, [`retarget`] for the ones that only have a request — and a request-only move is not
//! recorded durably, so the stored pair then lags by one run (self-correcting: the next request
//! re-selects from the same clock). This module owns the list, the clock and those decisions; it does
//! not call, retry or store anything — it hands the durable [`StoredProvider`] to the caller that
//! does.

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

use crate::config::{CONFIG, is_default_endpoint};
use crate::util::UnwrapPoison;

/// How long a provider must keep failing before a conversation already using it may move on.
pub(crate) const MOVE_AFTER: Duration = Duration::from_secs(300);

/// The provider a conversation holds, and the model it was chosen for — the durable pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoredProvider {
    pub model: String,
    pub provider: String,
}

/// A run's provider state: the model's priority list and the entry in use, so a provider-side
/// failure can move the run to the first available entry from the top.
#[derive(Debug, Default)]
pub(crate) struct ProviderSelection {
    model: String,
    candidates: Vec<String>,
    provider: Option<String>,
}

impl ProviderSelection {
    /// Resolve the selection for `model`, keeping `current` (the conversation's stored provider)
    /// while it still serves the model and its outage is younger than [`MOVE_AFTER`].
    #[must_use]
    pub(crate) fn resolve(model: &str, current: Option<&StoredProvider>) -> Self {
        Self::resolve_with(model, candidates(model), current)
    }

    /// Re-resolve for `model` — the step a caller takes when it builds a request: the provider in
    /// use is kept while it serves that model, and a model change re-selects, because the previous
    /// provider may not serve the new one.
    pub(crate) fn refresh(&mut self, model: &str) {
        *self = Self::resolve(model, self.stored().as_ref());
    }

    /// [`Self::resolve`] with the candidate list supplied instead of read from the process-global
    /// `CONFIG`, so the decision itself can be exercised in isolation.
    fn resolve_with(
        model: &str,
        candidates: Vec<String>,
        current: Option<&StoredProvider>,
    ) -> Self {
        let provider = if candidates.is_empty() {
            // No list → the product does not pin; the router decides.
            None
        } else {
            current
                .filter(|stored| {
                    stored.model == model
                        && candidates.iter().any(|c| c == &stored.provider)
                        && !outage_outlasted(model, &stored.provider)
                })
                .map(|stored| stored.provider.clone())
                .or_else(|| preferred(model, &candidates))
        };
        Self {
            model: model.to_string(),
            candidates,
            provider,
        }
    }

    /// The provider the next request is pinned to; `None` when the product does not pin.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn slug(&self) -> Option<&str> {
        self.provider.as_deref()
    }

    /// Re-resolve for the model `request` carries and pin the request there.
    pub(crate) fn apply(&mut self, request: &mut crate::ChatRequest) {
        self.refresh(&request.model);
        self.pin(request);
    }

    /// The durable form, for the caller that stores it.
    #[must_use]
    pub(crate) fn stored(&self) -> Option<StoredProvider> {
        let provider = self.provider.clone()?;
        Some(StoredProvider {
            model: self.model.clone(),
            provider,
        })
    }

    /// Pin `request` to this selection's provider — the unpinned case included, so a request is
    /// never left on a provider the selection has left.
    pub(crate) fn pin(&self, request: &mut crate::ChatRequest) {
        request.provider.clone_from(&self.provider);
    }

    /// A provider-side failure of the provider in use: once its outage has outlasted [`MOVE_AFTER`],
    /// move to the first available entry from the top of the list and pin `request` there. `true`
    /// when it moved, so the caller can keep its recorded operation in step.
    pub(crate) fn switch_on_failure(&mut self, request: &mut crate::ChatRequest) -> bool {
        self.pin(request);
        if !retarget_with(request, &self.candidates) {
            return false;
        }
        self.provider.clone_from(&request.provider);
        true
    }

    /// A selection over an explicit list, so a test can pin a provider without configuring the
    /// model's routing.
    #[cfg(test)]
    pub(crate) fn for_test(model: &str, candidates: &[&str], provider: &str) -> Self {
        Self {
            model: model.to_string(),
            candidates: candidates.iter().map(|slug| (*slug).to_string()).collect(),
            provider: Some(provider.to_string()),
        }
    }
}

/// The provider a one-off call with no conversation is pinned to: the first available from the top.
#[must_use]
pub(crate) fn pick(model: &str) -> Option<String> {
    preferred(model, &candidates(model))
}

/// Move `request` off its provider once that provider's outage has outlasted [`MOVE_AFTER`] — the
/// rule of [`ProviderSelection::switch_on_failure`] for a ladder that owns no selection
/// (extraction, grouping). `true` when the request now carries another provider.
pub(crate) fn retarget(request: &mut crate::ChatRequest) -> bool {
    let candidates = candidates(&request.model);
    retarget_with(request, &candidates)
}

/// [`retarget`] with the candidate list supplied, so the decision can be exercised in isolation.
fn retarget_with(request: &mut crate::ChatRequest, candidates: &[String]) -> bool {
    let Some(failed) = request.provider.clone() else {
        return false;
    };
    let Some(next) = next_from(&request.model, candidates, &failed) else {
        return false;
    };
    request.provider = Some(next);
    true
}

/// The ordered provider priority list for `model`, highest priority first; empty when the product
/// does not pin (nothing configured, or a custom chat endpoint is in use).
#[must_use]
fn candidates(model: &str) -> Vec<String> {
    // The `provider` block is an OpenRouter concept: a custom endpoint serves whatever its operator
    // configured, so the list is meaningless there (the Settings GUI already hides these rows).
    if !is_default_endpoint(&CONFIG.provider_endpoint()) {
        return Vec::new();
    }
    let Some(order) = CONFIG.model_routing(model).provider_order else {
        return Vec::new();
    };
    parse_priority_list(&order)
}

/// The comma-separated priority list as slugs, highest priority first. Empty entries are dropped, so
/// a trailing comma or a doubled separator never becomes a provider.
fn parse_priority_list(order: &str) -> Vec<String> {
    order
        .split(',')
        .map(str::trim)
        .filter(|slug| !slug.is_empty())
        .map(str::to_string)
        .collect()
}

/// What *choosing* a provider looks at: the first candidate with no outage on record, else the top
/// entry, so an exhausted list still yields one attempt on the highest-priority provider.
fn preferred(model: &str, candidates: &[String]) -> Option<String> {
    candidates
        .iter()
        .find(|provider| failing_since(model, provider).is_none())
        .or_else(|| candidates.first())
        .cloned()
}

/// Whether the provider's outage has lasted long enough to justify leaving it.
fn outage_outlasted(model: &str, provider: &str) -> bool {
    failing_since(model, provider).is_some_and(|since| since.elapsed() >= MOVE_AFTER)
}

/// The provider to continue on after `current` failed — the single switching rule the run state and
/// a bare request share: `None` while the outage is young, and `None` when [`preferred`] hands back
/// `current` itself.
fn next_from(model: &str, candidates: &[String], current: &str) -> Option<String> {
    if !outage_outlasted(model, current) {
        return None;
    }
    preferred(model, candidates).filter(|next| next.as_str() != current)
}

/// When the provider's current outage started, or `None` while it is serving.
fn failing_since(model: &str, provider: &str) -> Option<Instant> {
    let guard = OUTAGES.read().unwrap_poison();
    guard.get(model)?.get(provider).copied()
}

/// Whether the provider is currently failing — the clock readout, for tests that assert what
/// [`crate::providers::chat_scoped`] learned from a call.
#[cfg(test)]
pub(crate) fn is_failing(model: &str, provider: &str) -> bool {
    failing_since(model, provider).is_some()
}

/// A provider-side failure starts the clock (idempotent: the first failure's instant stands).
pub(crate) fn note_failure(model: &str, provider: &str) {
    OUTAGES
        .write()
        .unwrap_poison()
        .entry(model.to_string())
        .or_default()
        .entry(provider.to_string())
        .or_insert_with(Instant::now);
}

/// A successful call clears it.
pub(crate) fn note_success(model: &str, provider: &str) {
    let mut guard = OUTAGES.write().unwrap_poison();
    if let Some(providers) = guard.get_mut(model) {
        providers.remove(provider);
    }
}

/// The in-memory outage clock, keyed by model and then provider — nested so a per-request lookup
/// borrows both keys instead of allocating them. Process-global and deliberately unpersisted: a
/// restart forgets every outage and gives each provider a fresh chance.
static OUTAGES: std::sync::LazyLock<RwLock<HashMap<String, HashMap<String, Instant>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

/// Start the provider's outage clock at `since` — a test's way to make an outage older than
/// [`MOVE_AFTER`] without waiting.
#[cfg(test)]
pub(crate) fn force_outage(model: &str, provider: &str, since: Instant) {
    OUTAGES
        .write()
        .unwrap_poison()
        .entry(model.to_string())
        .or_default()
        .insert(provider.to_string(), since);
}

/// Forget every outage recorded for `model`, so a test starts from a provider set that no other test
/// has marked failing.
#[cfg(test)]
pub(crate) fn clear_outages(model: &str) {
    OUTAGES.write().unwrap_poison().remove(model);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The outage map is process-global and the tests run in parallel, so every test names its own
    /// model (and its own providers where a name could collide with another test's).
    fn candidates_of(slugs: &[&str]) -> Vec<String> {
        slugs.iter().map(|s| (*s).to_string()).collect()
    }

    /// An instant far enough in the past that the outage has outlived [`MOVE_AFTER`].
    fn long_ago() -> Instant {
        Instant::now()
            .checked_sub(MOVE_AFTER + Duration::from_secs(1))
            .expect("a five-minute subtraction stays representable")
    }

    fn stored(model: &str, provider: &str) -> StoredProvider {
        StoredProvider {
            model: model.to_string(),
            provider: provider.to_string(),
        }
    }

    #[test]
    fn a_request_moves_off_a_dead_provider_only_after_the_window() {
        let model = "selection/retarget";
        let mut request = crate::providers::test_request(vec![], None);
        request.model = model.to_string();
        request.provider = Some("first".into());

        force_outage(model, "first", Instant::now());
        assert!(
            !retarget_with(&mut request, &candidates_of(&["first", "second"])),
            "a young outage keeps the retries where the prompt cache is"
        );
        assert_eq!(request.provider.as_deref(), Some("first"));

        force_outage(model, "first", long_ago());
        assert!(retarget_with(
            &mut request,
            &candidates_of(&["first", "second"])
        ));
        assert_eq!(request.provider.as_deref(), Some("second"));

        let mut unpinned = crate::providers::test_request(vec![], None);
        assert!(
            !retarget_with(&mut unpinned, &candidates_of(&["first"])),
            "an unpinned request has nothing to move"
        );
    }

    #[test]
    fn the_priority_list_is_comma_separated_and_trimmed() {
        assert_eq!(
            parse_priority_list(" deepinfra/fp8 , anthropic ,, openai ,"),
            vec!["deepinfra/fp8", "anthropic", "openai"],
            "entries are trimmed and empty ones dropped, so a doubled or trailing comma is harmless"
        );
        assert_eq!(parse_priority_list("   "), Vec::<String>::new());
        assert_eq!(parse_priority_list("only"), vec!["only"]);
    }

    #[test]
    fn a_fresh_conversation_takes_the_first_available_from_the_top() {
        let model = "selection/fresh";
        force_outage(model, "first", Instant::now());
        let resolved =
            ProviderSelection::resolve_with(model, candidates_of(&["first", "second"]), None);
        assert_eq!(resolved.slug(), Some("second"));
        assert_eq!(resolved.stored(), Some(stored(model, "second")));
    }

    /// The default chat model's built-in list parses into the three candidate endpoints in priority
    /// order — the one place the routing string and the selection's split meet.
    #[test]
    fn the_default_chat_model_resolves_to_its_built_in_priority_list() {
        assert_eq!(
            parse_priority_list(
                crate::config::default_model_routing(crate::config::DEFAULT_MANAGER_MODEL)
                    .expect("the default model's built-in list"),
            ),
            candidates_of(&["deepinfra/fp8/flex", "deepinfra/fp8", "stepfun/fp8"]),
            "the built-in list must reach the selection as the three endpoints in priority order"
        );
    }

    #[test]
    fn a_provider_is_kept_while_its_outage_is_younger_than_move_after() {
        let model = "selection/young-outage";
        force_outage(model, "second", Instant::now());
        let current = stored(model, "second");
        let resolved = ProviderSelection::resolve_with(
            model,
            candidates_of(&["first", "second"]),
            Some(&current),
        );
        assert_eq!(
            resolved.slug(),
            Some("second"),
            "a blip does not move a conversation off the provider that holds its prompt cache"
        );
    }

    #[test]
    fn a_provider_is_reselected_once_its_outage_outlasts_move_after() {
        let model = "selection/long-outage";
        force_outage(model, "second", long_ago());
        let current = stored(model, "second");
        let resolved = ProviderSelection::resolve_with(
            model,
            candidates_of(&["first", "second"]),
            Some(&current),
        );
        assert_eq!(resolved.slug(), Some("first"));
    }

    #[test]
    fn a_stored_provider_for_a_different_model_is_reselected() {
        let model = "selection/other-model";
        let current = stored("selection/some-other-model", "second");
        let resolved = ProviderSelection::resolve_with(
            model,
            candidates_of(&["first", "second"]),
            Some(&current),
        );
        assert_eq!(
            resolved.slug(),
            Some("first"),
            "a stored provider is a (model, provider) pair; another model's says nothing about this one"
        );
    }

    #[test]
    fn a_stored_provider_no_longer_on_the_list_is_reselected() {
        let model = "selection/delisted";
        let current = stored(model, "gone");
        let resolved = ProviderSelection::resolve_with(
            model,
            candidates_of(&["first", "second"]),
            Some(&current),
        );
        assert_eq!(
            resolved.slug(),
            Some("first"),
            "only a listed provider can be held, so removing it re-selects"
        );
    }

    #[test]
    fn nothing_available_selects_the_top_entry() {
        let model = "selection/all-down";
        force_outage(model, "first", Instant::now());
        force_outage(model, "second", Instant::now());
        let resolved =
            ProviderSelection::resolve_with(model, candidates_of(&["first", "second"]), None);
        assert_eq!(
            resolved.slug(),
            Some("first"),
            "with every provider down the top entry is the only sensible attempt"
        );
    }

    #[test]
    fn an_empty_list_selects_nothing() {
        let model = "selection/no-list";
        let resolved =
            ProviderSelection::resolve_with(model, Vec::new(), Some(&stored(model, "first")));
        assert_eq!(
            resolved.slug(),
            None,
            "no list configured → the router decides, even over a stored provider"
        );
        assert_eq!(resolved.stored(), None);
    }

    /// A request carrying `provider` — the shape a ladder holds when it reports a failure.
    fn pinned_request(model: &str, provider: &str) -> crate::ChatRequest {
        let mut request = crate::providers::test_request(vec![], None);
        request.model = model.to_string();
        request.provider = Some(provider.to_string());
        request
    }

    #[test]
    fn a_failure_moves_the_run_and_its_request_only_after_move_after() {
        let model = "selection/outage-clock";
        let mut resolved = ProviderSelection::resolve_with(
            model,
            candidates_of(&["first", "second"]),
            Some(&stored(model, "second")),
        );
        let mut request = pinned_request(model, "second");
        assert_eq!(resolved.slug(), Some("second"));
        force_outage(model, "second", Instant::now());
        assert!(
            !resolved.switch_on_failure(&mut request),
            "an outage younger than MOVE_AFTER must not move a run off the provider"
        );
        assert_eq!(resolved.slug(), Some("second"));
        assert_eq!(request.provider.as_deref(), Some("second"));

        force_outage(model, "second", long_ago());
        assert!(
            resolved.switch_on_failure(&mut request),
            "an outage lasting MOVE_AFTER moves the run to the first available candidate"
        );
        assert_eq!(resolved.slug(), Some("first"));
        assert_eq!(
            request.provider.as_deref(),
            Some("first"),
            "the request the ladder retries follows the run"
        );
    }

    #[test]
    fn switch_on_failure_is_a_no_op_without_an_outage_or_a_provider() {
        let model = "selection/no-outage";
        let mut resolved = ProviderSelection::resolve_with(
            model,
            candidates_of(&["first", "second"]),
            Some(&stored(model, "first")),
        );
        let mut request = pinned_request(model, "first");
        assert!(
            !resolved.switch_on_failure(&mut request),
            "a provider with no recorded failure must not move"
        );
        assert_eq!(request.provider.as_deref(), Some("first"));

        let mut unpinned = ProviderSelection::resolve_with(model, Vec::new(), None);
        let mut request = pinned_request(model, "first");
        assert!(!unpinned.switch_on_failure(&mut request));
        assert_eq!(unpinned.slug(), None);
        assert_eq!(
            request.provider, None,
            "a run that pins nothing leaves the request unpinned too"
        );
    }

    #[test]
    fn a_recovered_provider_is_not_skipped_over() {
        let model = "selection/recovered";
        force_outage(model, "first", Instant::now());
        let resolved =
            ProviderSelection::resolve_with(model, candidates_of(&["first", "second"]), None);
        assert_eq!(resolved.slug(), Some("second"));

        note_success(model, "first");
        assert!(!is_failing(model, "first"));
        let reselected =
            ProviderSelection::resolve_with(model, candidates_of(&["first", "second"]), None);
        assert_eq!(
            reselected.slug(),
            Some("first"),
            "a recovered provider is available again, so a fresh selection takes it from the top"
        );
    }

    #[test]
    fn a_switch_to_a_lower_priority_provider_is_not_undone_while_it_works() {
        let model = "selection/sticky-lower";
        force_outage(model, "first", Instant::now());
        let resolved =
            ProviderSelection::resolve_with(model, candidates_of(&["first", "second"]), None);
        assert_eq!(resolved.slug(), Some("second"));

        // The top provider recovers, but the conversation is already warm on the second one.
        note_success(model, "first");
        let kept = ProviderSelection::resolve_with(
            model,
            candidates_of(&["first", "second"]),
            resolved.stored().as_ref(),
        );
        assert_eq!(
            kept.slug(),
            Some("second"),
            "the provider is kept with no re-evaluation against the top while it serves"
        );
    }

    #[test]
    fn reselection_after_a_loss_starts_at_the_top_again() {
        let model = "selection/back-to-top";
        force_outage(model, "first", Instant::now());
        let mut resolved =
            ProviderSelection::resolve_with(model, candidates_of(&["first", "second"]), None);
        assert_eq!(resolved.slug(), Some("second"));

        // The second provider is now the one lost, and the first has recovered.
        note_success(model, "first");
        force_outage(model, "second", long_ago());
        let mut request = pinned_request(model, "second");
        assert!(resolved.switch_on_failure(&mut request));
        assert_eq!(request.provider.as_deref(), Some("first"));
        assert_eq!(
            resolved.slug(),
            Some("first"),
            "re-selection walks the list from the top, not from the entry after the lost one"
        );
    }
}
