//! Provider selection for the OpenRouter benchmark: cost estimation + the
//! selection rules (health gate, outlier filter, top-n, padding).
//!
//! Pure math — no I/O. All functions here are unit-tested against synthetic
//! inputs; the caller (dry-run orchestration) feeds live discovery data in.
//!
//! # Selection rules
//!
//! 1. **Healthy** = status `"0"` AND `context_length >= min_context`
//!    (default 128 000; `None` context → excluded as unknown) AND not a
//!    `:free` variant AND (allowlist absent OR tag allowlisted).
//! 2. An allowlist restricts the candidate pool to its tags first; allowlisted
//!    endpoints that are unhealthy are still padding candidates, but only when
//!    the allowlist itself has fewer than 3 healthy endpoints.
//! 3. Excluded endpoints record the FIRST applicable reason in priority
//!    order: `NotInAllowlist` > `FreeVariant` > `ContextTooSmall`/`ContextUnknown`
//!    > `Status(raw)`.
//! 4. Healthy set H is filtered by cost outlier: est > 3×median is dropped
//!    (`Outlier`) unless dropping would leave fewer than 3 — then the cheapest
//!    dropped outliers are re-admitted until the pool is >= 3.
//! 5. Target count `n = max(3, ceil(0.8 × H.len()))`; the cheapest n of the
//!    filtered pool are selected, the rest get `NotSelected`. When
//!    `H.len() < 3` all of H is selected and padded with the cheapest excluded
//!    candidates to exactly 3 (`Padding` — expected to fail, but measured).
//!    When an allowlist restricts to exactly 1–2 endpoints those are
//!    benchmarked as-is and NO padding happens (user intent wins over the
//!    count rule).
//! 6. `ceil(0.8H) <= H` for H >= 3 makes "never more than the healthy count"
//!    automatic.
//! 7. The plan builder orders selected providers by est cost ascending.

use std::fmt;

use super::discovery::{EndpointInfo, Pricing, parse_price};

// ── Validated token mix ────────────────────────────────────────────

/// Fraction of total tokens served from the prompt cache (validated mix).
pub(crate) const VALIDATED_MIX_CACHED: f64 = 0.977;
/// Fraction of total tokens billed as uncached prompt input.
pub(crate) const VALIDATED_MIX_INPUT: f64 = 0.014;
/// Fraction of total tokens billed as completion output.
pub(crate) const VALIDATED_MIX_OUTPUT: f64 = 0.009;

// ── Cost estimation ────────────────────────────────────────────────

/// Blended per-provider cost estimate for a whole benchmark run.
///
/// The blended price is `cached_share×cache_read + input_share×prompt +
/// output_share×completion`; the token volumes follow
/// [`VALIDATED_MIX_CACHED`]/[`VALIDATED_MIX_INPUT`]/[`VALIDATED_MIX_OUTPUT`].
/// Per-request fees (`pricing.request`) are added once per request. When the
/// provider advertises no `input_cache_read` price the estimate assumes the
/// full prompt price and records that assumption in `flags`.
///
/// Missing/unparseable price fields contribute 0. Market-level price
/// reductions are already reflected in the listed prices and need no
/// adjustment here.
#[must_use]
// usize→f64 is exact for request counts far below 2^53; cost math needs floats.
#[expect(clippy::cast_precision_loss)]
pub(crate) fn estimate_cost(
    price: &Pricing,
    total_tokens: f64,
    requests: usize,
    flags: &mut Vec<String>,
) -> f64 {
    let cache_read = if let Some(p) = price.input_cache_read.as_deref().and_then(parse_price) {
        p
    } else {
        flags
            .push("no cache-read price advertised; estimate assumes full prompt price".to_string());
        price.prompt.as_deref().and_then(parse_price).unwrap_or(0.0)
    };
    let prompt = price.prompt.as_deref().and_then(parse_price).unwrap_or(0.0);
    let completion = price
        .completion
        .as_deref()
        .and_then(parse_price)
        .unwrap_or(0.0);
    let request = price
        .request
        .as_deref()
        .and_then(parse_price)
        .unwrap_or(0.0);

    let blended = VALIDATED_MIX_CACHED * cache_read
        + VALIDATED_MIX_INPUT * prompt
        + VALIDATED_MIX_OUTPUT * completion;
    blended * total_tokens + request * requests as f64
}

// ── Selection types ────────────────────────────────────────────────

/// One candidate endpoint for selection, with its estimated run cost.
///
/// The endpoint and its estimated cost are the only inputs;
/// `select_providers` derives health authoritatively from `endpoint` +
/// `min_context` + `allowlist`.
pub(crate) struct SelectionInput {
    pub endpoint: EndpointInfo,
    pub est_cost: f64,
}

/// Why a candidate was excluded from (or padded into) the selection.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ExclusionReason {
    /// Raw status string was not `"0"`.
    Status(String),
    /// `:free` variant tag.
    FreeVariant,
    /// Context window below `min_context`.
    ContextTooSmall(i64),
    /// No context length advertised.
    ContextUnknown,
    /// Tag not in the allowlist.
    NotInAllowlist,
    /// Cost above 3× the healthy median (dropped).
    Outlier(f64),
    /// Healthy but beyond the target count.
    NotSelected(f64),
    /// Unhealthy but padded in to reach the count (expected to fail).
    Padding,
}

impl fmt::Display for ExclusionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Status(s) => write!(f, "status {s}"),
            Self::FreeVariant => write!(f, "free variant (:free tag)"),
            Self::ContextTooSmall(cl) => write!(f, "context too small ({cl} tokens)"),
            Self::ContextUnknown => write!(f, "context unknown"),
            Self::NotInAllowlist => write!(f, "not in provider allowlist"),
            Self::Outlier(est) => write!(f, "cost outlier (est ${est:.4} > 3×median)"),
            Self::NotSelected(est) => write!(f, "not selected (est ${est:.4})"),
            Self::Padding => write!(f, "padding (expected to fail)"),
        }
    }
}

/// Selection outcome for one candidate (parallel to the input slice).
pub(crate) struct SelectionDecision {
    pub selected: bool,
    pub reason: Option<ExclusionReason>,
}

impl SelectionDecision {
    /// Human-readable selection reason for the plan/report: the exclusion
    /// reason when one is recorded, else "selected"/"not selected".
    #[must_use]
    pub(crate) fn reason_text(&self) -> String {
        match (&self.reason, self.selected) {
            (Some(r), _) => r.to_string(),
            (None, true) => "selected".to_string(),
            (None, false) => "not selected".to_string(),
        }
    }

    /// True iff the endpoint passed the health gate (independent of the cost
    /// rules): selected with no reason, or dropped by a cost rule
    /// (`NotSelected`/`Outlier`). Padding and gate-excluded endpoints are
    /// not healthy.
    #[must_use]
    pub(crate) fn is_healthy(&self) -> bool {
        matches!(
            self.reason,
            None | Some(ExclusionReason::NotSelected(_) | ExclusionReason::Outlier(_))
        )
    }
}

// ── Health classification ──────────────────────────────────────────

/// True iff the raw OpenRouter endpoint status is the healthy `"0"`.
///
/// Refined bands (documented, not enforced here — the raw status is recorded
/// verbatim elsewhere): `-1` unknown, `-2` verify-live, `-3` fallback-only,
/// `-5`/`-10` exclude. None of them is healthy.
#[must_use]
pub(crate) fn is_healthy_status(s: Option<&str>) -> bool {
    s == Some("0")
}

/// True iff the endpoint tag is a `:free` variant.
#[must_use]
pub(crate) fn is_free_variant(tag: &str) -> bool {
    tag.ends_with(":free")
}

/// Classify one endpoint against the health gate (rule 1) and produce the
/// FIRST applicable exclusion reason (rule 3 priority: NotInAllowlist >
/// FreeVariant > ContextTooSmall/ContextUnknown > Status).
///
/// Returns `(healthy, reason)`; `healthy` is `reason.is_none()`.
#[must_use]
pub(crate) fn classify_endpoint(
    endpoint: &EndpointInfo,
    min_context: i64,
    allowlist: Option<&[String]>,
) -> (bool, Option<ExclusionReason>) {
    let in_allowlist = allowlist.is_none_or(|wl| wl.iter().any(|t| t == &endpoint.tag));
    let reason = if !in_allowlist {
        Some(ExclusionReason::NotInAllowlist)
    } else if is_free_variant(&endpoint.tag) {
        Some(ExclusionReason::FreeVariant)
    } else {
        match endpoint.context_length {
            None => Some(ExclusionReason::ContextUnknown),
            Some(cl) if cl < min_context => Some(ExclusionReason::ContextTooSmall(cl)),
            Some(_) if !is_healthy_status(endpoint.status.as_deref()) => Some(
                ExclusionReason::Status(endpoint.status.clone().unwrap_or_default()),
            ),
            Some(_) => None,
        }
    };
    (reason.is_none(), reason)
}

// ── Selection ──────────────────────────────────────────────────────

/// Select benchmark providers per the selection rules (see module docs).
///
/// Returns one [`SelectionDecision`] per input (parallel). The plan builder
/// orders the selected providers by est cost ascending for display.
#[must_use]
pub(crate) fn select_providers(
    input: &[SelectionInput],
    min_context: i64,
    allowlist: Option<&[String]>,
) -> Vec<SelectionDecision> {
    let candidates = candidate_indices(input, allowlist);

    // Allowlist with exactly 1–2 endpoints: benchmark exactly those, no
    // padding beyond the allowlist — user intent wins over the count rule.
    // Only applies when an allowlist was given: with no allowlist the
    // candidate pool is all endpoints and the normal rules govern.
    if allowlist.is_some() && !candidates.is_empty() && candidates.len() <= 2 {
        let mut decisions = base_decisions(input, min_context, allowlist);
        for &idx in &candidates {
            // Keep the natural reason (e.g. "status -10") so the report can
            // show "selected despite <reason>" — the run measures them anyway.
            decisions[idx].selected = true;
        }
        return decisions;
    }

    let mut decisions = base_decisions(input, min_context, allowlist);

    // Healthy set H over the candidate pool (rule 1).
    let mut healthy: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|&idx| decisions[idx].reason.is_none())
        .collect();

    if healthy.is_empty() {
        return decisions; // zero healthy → zero selected (no padding)
    }

    // Rule 5 tail: H < 3 → select all of H, pad with the cheapest excluded
    // candidates to exactly 3.
    if healthy.len() < 3 {
        pad_to_three(&mut decisions, &candidates, input);
        return decisions;
    }

    // Rule 4: cost outliers (est > 3×median), with re-admission to >= 3.
    healthy.sort_by(|&a, &b| input[a].est_cost.total_cmp(&input[b].est_cost));
    let median = median_of(&healthy, input);
    let outliers: Vec<usize> = healthy
        .iter()
        .copied()
        .filter(|&idx| input[idx].est_cost > 3.0 * median)
        .collect();
    let mut pool: Vec<usize> = healthy
        .iter()
        .copied()
        .filter(|&idx| input[idx].est_cost <= 3.0 * median)
        .collect();
    if pool.len() < 3 {
        // Re-admit the cheapest dropped outliers until the pool is >= 3.
        let mut by_cost = outliers.clone();
        by_cost.sort_by(|&a, &b| input[a].est_cost.total_cmp(&input[b].est_cost));
        for idx in by_cost {
            if pool.len() >= 3 {
                break;
            }
            pool.push(idx);
        }
    }

    // Rule 5: target n = max(3, ceil(0.8 × H.len())); cheapest n of the pool.
    let n = selection_target(healthy.len());
    for &idx in pool.iter().take(n) {
        decisions[idx] = SelectionDecision {
            selected: true,
            reason: None,
        };
    }
    // Beyond n → NotSelected (covers re-admitted outliers too).
    for &idx in pool.iter().skip(n) {
        decisions[idx] = SelectionDecision {
            selected: false,
            reason: Some(ExclusionReason::NotSelected(input[idx].est_cost)),
        };
    }
    // Dropped outliers → Outlier.
    for &idx in &outliers {
        if !pool.contains(&idx) {
            decisions[idx] = SelectionDecision {
                selected: false,
                reason: Some(ExclusionReason::Outlier(input[idx].est_cost)),
            };
        }
    }

    decisions
}

/// Rule 2: candidate pool = allowlisted tags (when an allowlist is given),
/// else every endpoint. The output stays parallel to `input`, so candidates
/// carry their input index.
fn candidate_indices(input: &[SelectionInput], allowlist: Option<&[String]>) -> Vec<usize> {
    match allowlist {
        Some(wl) if !wl.is_empty() => input
            .iter()
            .enumerate()
            .filter(|(_, i)| wl.iter().any(|t| t == &i.endpoint.tag))
            .map(|(idx, _)| idx)
            .collect(),
        Some(_) => Vec::new(),
        None => (0..input.len()).collect(),
    }
}

/// Initial per-endpoint classification: nothing selected, reason from the
/// health gate (rule 1 + rule 3).
fn base_decisions(
    input: &[SelectionInput],
    min_context: i64,
    allowlist: Option<&[String]>,
) -> Vec<SelectionDecision> {
    input
        .iter()
        .map(|i| {
            let (_, reason) = classify_endpoint(&i.endpoint, min_context, allowlist);
            SelectionDecision {
                selected: false,
                reason,
            }
        })
        .collect()
}

/// Rule 5 tail: select all healthy candidates (reason `None`), then pad with
/// the cheapest excluded candidates to exactly 3 ([`ExclusionReason::Padding`]
/// — expected to fail, but measured).
fn pad_to_three(
    decisions: &mut [SelectionDecision],
    candidates: &[usize],
    input: &[SelectionInput],
) {
    for &idx in candidates {
        if decisions[idx].reason.is_none() {
            decisions[idx].selected = true;
        }
    }
    let mut by_cost: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|&idx| !decisions[idx].selected)
        .collect();
    by_cost.sort_by(|&a, &b| input[a].est_cost.total_cmp(&input[b].est_cost));
    for idx in by_cost {
        if decisions.iter().filter(|d| d.selected).count() >= 3 {
            break;
        }
        decisions[idx] = SelectionDecision {
            selected: true,
            reason: Some(ExclusionReason::Padding),
        };
    }
}

/// Target selection count: `max(3, ceil(0.8 × healthy))`. For H >= 3 this is
/// never more than the healthy count (`ceil(0.8H) <= H`); the plan builder
/// special-cases H == 0 → 0 separately.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub(crate) fn selection_target(healthy_count: usize) -> usize {
    (3usize).max((healthy_count as f64 * 0.8).ceil() as usize)
}

/// Effective selection target: 0 when nothing is healthy; the allowlist size
/// when a short (1-2 entry) allowlist is given (user intent wins); else the
/// max(3, ceil(0.8 × healthy)) rule.
#[must_use]
pub(crate) fn effective_target_count(
    healthy_count: usize,
    allowlist_matches: Option<usize>,
) -> usize {
    match (healthy_count, allowlist_matches) {
        (0, _) => 0,
        (_, Some(m)) if m > 0 && m <= 2 => m,
        _ => selection_target(healthy_count),
    }
}

/// Median of est costs over the given input indices (average of the two
/// middle values for an even count).
fn median_of(indices: &[usize], input: &[SelectionInput]) -> f64 {
    let mut costs: Vec<f64> = indices.iter().map(|&i| input[i].est_cost).collect();
    costs.sort_by(f64::total_cmp);
    let mid = costs.len() / 2;
    if costs.len().is_multiple_of(2) {
        f64::midpoint(costs[mid - 1], costs[mid])
    } else {
        costs[mid]
    }
}

/// Total estimated cost of the selected providers and the per-provider spend
/// guard: `cap × 2 / max(1, selected_count)`. The ×2 headroom lets a couple of
/// padded (expected-to-fail) providers overshoot individually without burning
/// the whole cap.
#[must_use]
// usize→f64 is exact for counts far below 2^53; the guard is a cost ratio.
#[expect(clippy::cast_precision_loss)]
pub(crate) fn plan_cost(
    selected: &[SelectionDecision],
    inputs: &[SelectionInput],
    cap: f64,
) -> (f64, f64) {
    let selected_count = selected.iter().filter(|d| d.selected).count();
    let total: f64 = selected
        .iter()
        .zip(inputs)
        .filter(|(d, _)| d.selected)
        .map(|(_, i)| i.est_cost)
        .sum();
    let guard = cap * 2.0 / (selected_count.max(1) as f64);
    (total, guard)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench_openrouter::discovery::Pricing;

    /// Build a minimal endpoint with the given tag/context/status.
    fn ep(tag: &str, context: i64, status: &str) -> EndpointInfo {
        ep_ctx(tag, Some(context), status)
    }

    fn ep_ctx(tag: &str, context: Option<i64>, status: &str) -> EndpointInfo {
        EndpointInfo {
            tag: tag.to_string(),
            name: tag.to_string(),
            provider_name: tag.to_string(),
            context_length: context,
            quantization: None,
            status: Some(status.to_string()),
            supports_implicit_caching: Some(true),
            pricing: Some(Pricing {
                prompt: Some("0.000001".to_string()),
                completion: Some("0.000002".to_string()),
                request: Some("0".to_string()),
                input_cache_read: Some("0.0000001".to_string()),
            }),
        }
    }

    fn input(ep: EndpointInfo, est: f64) -> SelectionInput {
        SelectionInput {
            endpoint: ep,
            est_cost: est,
        }
    }

    #[test]
    fn healthy_status_parsing() {
        assert!(is_healthy_status(Some("0")));
        for s in ["-1", "-2", "-3", "-5", "-10", "", "abc"] {
            assert!(!is_healthy_status(Some(s)), "status {s:?}");
        }
        assert!(!is_healthy_status(None));
    }

    #[test]
    fn free_variant_detection() {
        assert!(is_free_variant("groq/llama-3.1-70b:free"));
        assert!(!is_free_variant("groq/llama-3.1-70b"));
        assert!(!is_free_variant(""));
    }

    #[test]
    fn estimate_cost_uses_cache_read_and_request_fees() {
        let price = Pricing {
            prompt: Some("0.000002".to_string()),
            completion: Some("0.000008".to_string()),
            request: Some("0.0005".to_string()),
            input_cache_read: Some("0.0000002".to_string()),
        };
        let mut flags = Vec::new();
        let est = estimate_cost(&price, 1000.0, 4, &mut flags);
        assert!(flags.is_empty(), "cache-read price present → no flag");
        let expected =
            (0.977 * 0.000_000_2 + 0.014 * 0.000_002 + 0.009 * 0.000_008) * 1000.0 + 0.0005 * 4.0;
        assert!((est - expected).abs() < 1e-12);
    }

    #[test]
    fn estimate_cost_falls_back_to_prompt_with_flag() {
        let price = Pricing {
            prompt: Some("0.000002".to_string()),
            completion: Some("0.000008".to_string()),
            request: Some("0".to_string()),
            input_cache_read: None, // provider does not advertise cache pricing
        };
        let mut flags = Vec::new();
        let est = estimate_cost(&price, 1000.0, 1, &mut flags);
        assert_eq!(flags.len(), 1);
        assert!(flags[0].contains("no cache-read price advertised"));
        // All cached tokens billed at the full prompt price.
        let expected = (0.977 + 0.014) * 0.000_002 * 1000.0 + 0.009 * 0.000_008 * 1000.0;
        assert!((est - expected).abs() < 1e-12);
    }

    #[test]
    fn selection_target_count_and_outlier() {
        // 8 healthy with varied costs; one cost is an outlier (>3×median).
        let mut inputs = Vec::new();
        for i in 0..8 {
            let est = if i == 7 { 100.0 } else { f64::from(i + 1) };
            inputs.push(input(ep(&format!("p{i}"), 200_000, "0"), est));
        }
        let decisions = select_providers(&inputs, 128_000, None);
        let selected: Vec<_> = decisions.iter().filter(|d| d.selected).collect();
        // target = max(3, ceil(0.8×8)) = 7; the outlier (>3×median) is dropped.
        assert_eq!(selected.len(), 7);
        assert_eq!(decisions[7].reason, Some(ExclusionReason::Outlier(100.0)));
        // The 6 non-outlier healthy endpoints are all selected.
        for d in &decisions[..6] {
            assert!(d.selected);
        }
    }

    #[test]
    fn selection_pads_healthy_lt_3() {
        // 2 healthy + 1 unhealthy; pad to exactly 3 with the cheapest excluded.
        let mut inputs = Vec::new();
        let h0 = input(ep("h0", 200_000, "0"), 1.0);
        let h1 = input(ep("h1", 200_000, "0"), 2.0);
        let bad = input(ep("bad", 200_000, "-10"), 3.0);
        inputs.extend([h0, h1, bad]);
        let decisions = select_providers(&inputs, 128_000, None);
        let selected: Vec<_> = decisions.iter().filter(|d| d.selected).collect();
        assert_eq!(selected.len(), 3);
        assert!(decisions[0].selected && decisions[1].selected);
        assert!(decisions[2].selected);
        assert_eq!(decisions[2].reason, Some(ExclusionReason::Padding));
    }

    #[test]
    fn selection_zero_healthy_is_empty() {
        let inputs = vec![
            input(ep("a", 200_000, "-10"), 1.0),
            input(ep("b", 200_000, "-2"), 2.0),
        ];
        let decisions = select_providers(&inputs, 128_000, None);
        assert!(decisions.iter().all(|d| !d.selected));
    }

    #[test]
    fn selection_allowlist_restricts_and_short_allowlist_wins() {
        let mut inputs = Vec::new();
        for i in 0..4 {
            inputs.push(input(ep(&format!("p{i}"), 200_000, "0"), f64::from(i + 1)));
        }
        // Allowlist with 2 entries → exactly those two selected, no padding.
        let wl = vec!["p1".to_string(), "p3".to_string()];
        let decisions = select_providers(&inputs, 128_000, Some(&wl));
        assert_eq!(decisions.iter().filter(|d| d.selected).count(), 2);
        assert!(decisions[1].selected && decisions[3].selected);
        assert!(!decisions[0].selected && !decisions[2].selected);

        // Single-entry allowlist → exactly that one.
        let wl = vec!["p0".to_string()];
        let decisions = select_providers(&inputs, 128_000, Some(&wl));
        assert_eq!(decisions.iter().filter(|d| d.selected).count(), 1);
        assert!(decisions[0].selected);
    }

    #[test]
    fn selection_min_context_excludes_small_contexts() {
        // 4 healthy (context ok) + small-context + unknown-context: the two
        // context-excluded endpoints stay out (H >= 3 → no padding).
        let inputs = vec![
            input(ep("small", 32_000, "0"), 5.0),
            input(ep_ctx("none", None, "0"), 5.0),
            input(ep("ok1", 200_000, "0"), 1.0),
            input(ep("ok2", 200_000, "0"), 2.0),
            input(ep("ok3", 200_000, "0"), 3.0),
            input(ep("ok4", 200_000, "0"), 4.0),
        ];
        let decisions = select_providers(&inputs, 128_000, None);
        assert_eq!(
            decisions[0].reason,
            Some(ExclusionReason::ContextTooSmall(32_000))
        );
        assert_eq!(decisions[1].reason, Some(ExclusionReason::ContextUnknown));
        assert!(!decisions[0].selected && !decisions[1].selected);
        // n = max(3, ceil(0.8×4)) = 4 → all 4 healthy selected.
        assert_eq!(decisions.iter().filter(|d| d.selected).count(), 4);
        for (i, d) in decisions.iter().enumerate().take(6).skip(2) {
            assert!(d.selected, "endpoint {i} should be selected");
        }
    }

    #[test]
    fn plan_cost_sums_selected_and_computes_guard() {
        let inputs = vec![
            input(ep("a", 200_000, "0"), 0.1),
            input(ep("b", 200_000, "0"), 0.2),
            input(ep("c", 200_000, "-10"), 0.3),
        ];
        let decisions = select_providers(&inputs, 128_000, None);
        let (total, guard) = plan_cost(&decisions, &inputs, 2.0);
        // a + b selected, c padded → total 0.1 + 0.2 + 0.3.
        assert!((total - 0.6).abs() < 1e-12);
        // guard = cap×2 / max(1, 3) = 4/3.
        assert!((guard - 4.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn exclusion_reason_displays() {
        assert_eq!(
            ExclusionReason::Status("-10".to_string()).to_string(),
            "status -10"
        );
        assert_eq!(
            ExclusionReason::FreeVariant.to_string(),
            "free variant (:free tag)"
        );
        assert_eq!(
            ExclusionReason::ContextTooSmall(32_000).to_string(),
            "context too small (32000 tokens)"
        );
        assert_eq!(
            ExclusionReason::ContextUnknown.to_string(),
            "context unknown"
        );
        assert_eq!(
            ExclusionReason::NotInAllowlist.to_string(),
            "not in provider allowlist"
        );
        assert_eq!(
            ExclusionReason::Padding.to_string(),
            "padding (expected to fail)"
        );
    }

    #[test]
    fn selection_decision_reason_text() {
        let d = SelectionDecision {
            selected: false,
            reason: Some(ExclusionReason::Padding),
        };
        assert_eq!(d.reason_text(), "padding (expected to fail)");
        let selected = SelectionDecision {
            selected: true,
            reason: None,
        };
        assert_eq!(selected.reason_text(), "selected");
        let unselected = SelectionDecision {
            selected: false,
            reason: None,
        };
        assert_eq!(unselected.reason_text(), "not selected");
    }

    #[test]
    fn effective_target_count_rules() {
        // Nothing healthy → 0, allowlist or not.
        assert_eq!(effective_target_count(0, None), 0);
        assert_eq!(effective_target_count(0, Some(2)), 0);
        // No allowlist: max(3, ceil(0.8 × healthy)).
        assert_eq!(effective_target_count(5, None), 4);
        assert_eq!(effective_target_count(20, None), 16);
        // Short allowlist (1-2 matching endpoints) wins over the count rule.
        assert_eq!(effective_target_count(1, Some(2)), 2);
        assert_eq!(effective_target_count(10, Some(1)), 1);
        // A 0-match allowlist is not "short" — the count rule applies.
        assert_eq!(effective_target_count(10, Some(0)), 8);
    }
}
