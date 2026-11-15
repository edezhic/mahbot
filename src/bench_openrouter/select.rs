//! Provider selection for the OpenRouter benchmark: cost estimation + the selection rules
//! (health gate, outlier filter, top-n, padding).
//!
//! Pure math — no I/O. Every function is unit-tested against synthetic inputs; the caller
//! (dry-run orchestration) feeds live discovery data in.
//!
//! # Selection rules
//!
//! 1. **Healthy** = status `"0"` AND `context_length >= min_context` (default 128 000; `None`
//!    context is excluded as unknown) AND not a `:free` variant AND (allowlist absent OR tag
//!    allowlisted).
//! 2. An allowlist restricts the candidate pool to its tags first; allowlisted endpoints that are
//!    unhealthy are still padding candidates, but only when the allowlist itself has fewer than 3
//!    healthy endpoints.
//! 3. Excluded endpoints record the FIRST applicable reason in priority order: `NotInAllowlist` >
//!    `FreeVariant` > `ContextTooSmall`/`ContextUnknown` > `Status(raw)`.
//! 4. Healthy set H is filtered by cost outlier: est > 3×median is dropped (`Outlier`) unless
//!    dropping would leave fewer than 3 — then the cheapest dropped outliers are re-admitted.
//! 5. Target count `n = min(20, max(3, ceil(0.8 × H.len())))` — the [`MAX_SELECTED_ENDPOINTS`]
//!    ceiling caps a large pool (48 healthy → 20, not 39). An explicit provider allowlist is
//!    exempt from the ceiling (a thirty-entry allowlist selects twenty-four). The cheapest n of
//!    the filtered pool are selected. When `H.len() < 3` all of H is selected and padded with
//!    the cheapest excluded candidates to exactly 3 (`Padding`). When an allowlist restricts to
//!    exactly 1–2 endpoints those are benchmarked as-is and NO padding happens. The per-provider
//!    spend guard `cap × 2 / max(1, selected)` follows the selection.
//! 6. `ceil(0.8H) <= H` for H >= 3 makes "never more than the healthy count" automatic.
//! 7. The plan builder orders selected providers by est cost ascending.

use std::fmt;

use super::discovery::{EndpointInfo, Pricing, parse_price};

/// Fraction of total tokens served from the prompt cache (98/1/1 usage mix).
pub(crate) const USAGE_MIX_CACHED: f64 = 0.98;
/// Fraction of total tokens billed as uncached prompt input.
pub(crate) const USAGE_MIX_INPUT: f64 = 0.01;
/// Fraction of total tokens billed as completion output.
pub(crate) const USAGE_MIX_OUTPUT: f64 = 0.01;

/// Hard ceiling on the number of endpoints one run benchmarks. Most discovered endpoints are
/// therefore never attempted, which is what the report's per-row `selected` flag disambiguates:
/// "never selected" and "selected but failed" both read `not measured`. An explicit provider
/// allowlist is exempt.
pub(crate) const MAX_SELECTED_ENDPOINTS: usize = 20;

/// Blended per-token price for one provider: `cached_share×cache_read + input_share×prompt +
/// output_share×completion`. The ONE place the usage blend is computed: [`estimate_cost`] applies
/// it to a token volume, and [`usage_mix_json`] publishes the same constants, so the artifact can
/// never disagree with the math.
#[must_use]
pub(crate) fn blended_price_per_token(cache_read: f64, prompt: f64, completion: f64) -> f64 {
    USAGE_MIX_CACHED * cache_read + USAGE_MIX_INPUT * prompt + USAGE_MIX_OUTPUT * completion
}

/// The usage mix as a self-describing JSON object, taken from the same constants the cost math
/// uses.
#[must_use]
pub(crate) fn usage_mix_json() -> serde_json::Value {
    serde_json::json!({
        "cached": USAGE_MIX_CACHED,
        "input": USAGE_MIX_INPUT,
        "output": USAGE_MIX_OUTPUT,
    })
}

/// The input price a cached prompt token is billed at — the advertised cache-read price, or the
/// full prompt price where none is advertised — and whether that fallback stood in. The ONE place
/// the rule is decided: [`estimate_cost`] flags the fallback, the report's static price does not.
#[must_use]
pub(crate) fn cache_read_price(pricing: &Pricing, prompt: f64) -> (f64, bool) {
    match pricing.input_cache_read.as_deref().and_then(parse_price) {
        Some(cache_read) => (cache_read, false),
        None => (prompt, true),
    }
}

/// Blended per-provider cost estimate for a whole benchmark run: the blended price is
/// [`blended_price_per_token`], with per-request fees added once per request. A provider
/// advertising no `input_cache_read` price is billed at the full prompt price and says so in
/// `flags`. Missing price fields contribute 0. No discount factor is applied anywhere — the
/// listed prices are already net of any advertised discount.
#[must_use]
// usize→f64 is exact for request counts far below 2^53; cost math needs floats.
#[expect(clippy::cast_precision_loss)]
pub(crate) fn estimate_cost(
    price: &Pricing,
    total_tokens: f64,
    requests: usize,
    flags: &mut Vec<String>,
) -> f64 {
    let prompt = price.prompt.as_deref().and_then(parse_price).unwrap_or(0.0);
    let (cache_read, fallback) = cache_read_price(price, prompt);
    if fallback {
        flags
            .push("no cache-read price advertised; estimate assumes full prompt price".to_string());
    }
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

    let blended = blended_price_per_token(cache_read, prompt, completion);
    blended * total_tokens + request * requests as f64
}

/// One candidate endpoint for selection, with its estimated run cost. `select_providers`
/// derives health authoritatively from `endpoint` + `min_context` + `allowlist`.
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
    /// Human-readable reason for the plan/report: the exclusion reason, else "selected"/"not
    /// selected".
    #[must_use]
    pub(crate) fn reason_text(&self) -> String {
        match (&self.reason, self.selected) {
            (Some(r), _) => r.to_string(),
            (None, true) => "selected".to_string(),
            (None, false) => "not selected".to_string(),
        }
    }

    /// True iff the endpoint passed the health gate, independent of the cost rules: selected with
    /// no reason, or dropped by `NotSelected`/`Outlier`.
    #[must_use]
    pub(crate) fn is_healthy(&self) -> bool {
        matches!(
            self.reason,
            None | Some(ExclusionReason::NotSelected(_) | ExclusionReason::Outlier(_))
        )
    }
}

/// True iff the raw OpenRouter endpoint status is the healthy `"0"`. The refined bands (`-1`
/// unknown, `-2` verify-live, `-3` fallback-only, `-5`/`-10` exclude) are none of them healthy.
#[must_use]
pub(crate) fn is_healthy_status(s: Option<&str>) -> bool {
    s == Some("0")
}

/// True iff the endpoint tag is a `:free` variant.
#[must_use]
fn is_free_variant(tag: &str) -> bool {
    tag.ends_with(":free")
}

/// Classify one endpoint against the health gate and produce the FIRST applicable exclusion
/// reason. Returns `(healthy, reason)`; `healthy` is `reason.is_none()`.
#[must_use]
fn classify_endpoint(
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

/// Select benchmark providers per the selection rules (see module docs): one
/// [`SelectionDecision`] per input, in parallel. `allowlist` restricts the candidate pool to the
/// tags it names, so an allowlist naming no tag selects nothing; an empty `--providers` never
/// reaches here as `Some` ([`BenchOptions::parse`] turns it into `None`), which is what keeps it
/// from switching the ceiling off.
#[must_use]
pub(crate) fn select_providers(
    input: &[SelectionInput],
    min_context: i64,
    allowlist: Option<&[String]>,
) -> Vec<SelectionDecision> {
    let candidates = candidate_indices(input, allowlist);

    // An allowlist with exactly 1–2 endpoints: benchmark exactly those, no padding beyond it —
    // user intent wins over the count rule.
    if allowlist.is_some() && !candidates.is_empty() && candidates.len() <= 2 {
        let mut decisions = base_decisions(input, min_context, allowlist);
        for &idx in &candidates {
            // Keep the natural reason (e.g. "status -10") so the report can show "selected
            // despite <reason>" — the run measures them anyway.
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

    // Rule 5 tail: H < 3 → select all of H, pad with the cheapest excluded to exactly 3.
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

    // Rule 5: the cheapest n of the filtered pool.
    let n = selection_target(healthy.len(), allowlist.is_none());
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

/// Rule 2: candidate pool = allowlisted tags, else every endpoint. The output stays parallel to
/// `input`, so candidates carry their input index.
fn candidate_indices(input: &[SelectionInput], allowlist: Option<&[String]>) -> Vec<usize> {
    match allowlist {
        Some(tags) => input
            .iter()
            .enumerate()
            .filter(|(_, i)| tags.contains(&i.endpoint.tag))
            .map(|(idx, _)| idx)
            .collect(),
        None => (0..input.len()).collect(),
    }
}

/// Initial per-endpoint classification: nothing selected, reason from the health gate.
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

/// Rule 5 tail: select all healthy candidates, then pad with the cheapest excluded to exactly 3
/// ([`ExclusionReason::Padding`] — expected to fail, but measured).
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

/// Target selection count: `max(3, ceil(0.8 × healthy))`, never more than the healthy count for
/// H >= 3. `cap_applies` clamps it to [`MAX_SELECTED_ENDPOINTS`] — an explicit provider allowlist
/// turns the clamp off, so the user's list decides. H == 0 → 0 is special-cased separately.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn selection_target(healthy_count: usize, cap_applies: bool) -> usize {
    let n = (3usize).max((healthy_count as f64 * 0.8).ceil() as usize);
    if cap_applies {
        n.min(MAX_SELECTED_ENDPOINTS)
    } else {
        n
    }
}

/// Effective selection target: 0 when nothing is healthy, the match count for a short (1-2
/// matched) allowlist, else the count rule — clamped to [`MAX_SELECTED_ENDPOINTS`] only when no
/// allowlist was given (`allowlist_matches.is_none()`), matching [`select_providers`].
#[must_use]
pub(crate) fn effective_target_count(
    healthy_count: usize,
    allowlist_matches: Option<usize>,
) -> usize {
    match allowlist_matches {
        // A 1-2 match allowlist is benchmarked as-is, healthy or not, so the count rule (and its
        // zero-healthy arm) does not apply to it.
        Some(m) if m > 0 && m <= 2 => m,
        _ if healthy_count == 0 => 0,
        None => selection_target(healthy_count, true),
        Some(_) => selection_target(healthy_count, false),
    }
}

/// Median of est costs over the given input indices (the average of the two middle values).
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

/// Total estimated cost of the selected providers and the per-provider spend guard:
/// `cap × 2 / max(1, selected_count)`. The ×2 headroom lets a couple of padded providers
/// overshoot individually without burning the whole cap.
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

    /// The health gate's two screening predicates: only the raw status `"0"` is healthy, and only
    /// a `:free` suffix is the free variant.
    #[test]
    fn exclusion_predicates() {
        assert!(is_healthy_status(Some("0")));
        for s in ["-1", "-2", "-3", "-5", "-10", "", "abc"] {
            assert!(!is_healthy_status(Some(s)), "status {s:?}");
        }
        assert!(!is_healthy_status(None));

        assert!(is_free_variant("groq/llama-3.1-70b:free"));
        assert!(!is_free_variant("groq/llama-3.1-70b"));
        assert!(!is_free_variant(""));
    }

    /// The estimate blends the 98/1/1 usage mix and adds the per-request fee; a provider with no
    /// advertised cache-read price is billed at the full prompt price and flags it.
    #[test]
    fn estimate_cost_cases() {
        let priced = Pricing {
            prompt: Some("0.000002".to_string()),
            completion: Some("0.000008".to_string()),
            request: Some("0.0005".to_string()),
            input_cache_read: Some("0.0000002".to_string()),
        };
        let mut flags = Vec::new();
        let est = estimate_cost(&priced, 1000.0, 4, &mut flags);
        assert!(flags.is_empty(), "cache-read price present → no flag");
        let expected =
            (0.98 * 0.000_000_2 + 0.01 * 0.000_002 + 0.01 * 0.000_008) * 1000.0 + 0.0005 * 4.0;
        assert!((est - expected).abs() < 1e-12);

        let unpriced_cache = Pricing {
            prompt: priced.prompt.clone(),
            completion: priced.completion.clone(),
            request: Some("0".to_string()),
            input_cache_read: None, // provider does not advertise cache pricing
        };
        let mut flags = Vec::new();
        let est = estimate_cost(&unpriced_cache, 1000.0, 1, &mut flags);
        assert_eq!(flags.len(), 1);
        assert!(flags[0].contains("no cache-read price advertised"));
        // All cached tokens billed at the full prompt price.
        let expected = (0.98 + 0.01) * 0.000_002 * 1000.0 + 0.01 * 0.000_008 * 1000.0;
        assert!((est - expected).abs() < 1e-12);
    }

    /// The 98/1/1 mix at scale: one billion tokens with the three prices, no per-request fee.
    /// The artifact's usage-mix object carries the same three constants.
    #[test]
    fn estimate_cost_uses_the_98_1_1_mix() {
        let priced = Pricing {
            prompt: Some("0.000001".to_string()),
            completion: Some("0.000002".to_string()),
            request: Some("0".to_string()),
            input_cache_read: Some("0.0000001".to_string()),
        };
        let mut flags = Vec::new();
        let est = estimate_cost(&priced, 1e9, 0, &mut flags);
        assert!(flags.is_empty());
        let expected = 1e9 * (0.98 * 1e-7 + 0.01 * 1e-6 + 0.01 * 2e-6);
        assert!((est - expected).abs() < 1e-9, "est {est} vs {expected}");

        let mix = usage_mix_json();
        assert_eq!(mix["cached"], 0.98);
        assert_eq!(mix["input"], 0.01);
        assert_eq!(mix["output"], 0.01);
    }

    /// The gate and cost rules through `select_providers`: an outlier is dropped while the rest are
    /// selected, a short healthy set is padded to three, and context-excluded endpoints stay out.
    #[test]
    fn selection_exclusions_and_padding() {
        struct Case {
            name: &'static str,
            inputs: Vec<SelectionInput>,
            /// Index → the reason recorded for it.
            reasons: &'static [(usize, Option<ExclusionReason>)],
            /// The indices the case must select, in order.
            selected: &'static [usize],
        }
        let mut outlier_inputs = Vec::new();
        for i in 0..8 {
            let est = if i == 7 { 100.0 } else { f64::from(i + 1) };
            outlier_inputs.push(input(ep(&format!("p{i}"), 200_000, "0"), est));
        }
        let cases = [
            Case {
                name: "outlier dropped",
                inputs: outlier_inputs,
                // target = max(3, ceil(0.8×8)) = 7; the outlier (>3×median) is dropped.
                reasons: &[(7, Some(ExclusionReason::Outlier(100.0)))],
                selected: &[0, 1, 2, 3, 4, 5, 6],
            },
            Case {
                name: "padded to three",
                inputs: vec![
                    input(ep("h0", 200_000, "0"), 1.0),
                    input(ep("h1", 200_000, "0"), 2.0),
                    input(ep("bad", 200_000, "-10"), 3.0),
                ],
                reasons: &[(2, Some(ExclusionReason::Padding))],
                selected: &[0, 1, 2],
            },
            Case {
                name: "context excluded",
                inputs: vec![
                    input(ep("small", 32_000, "0"), 5.0),
                    input(ep_ctx("none", None, "0"), 5.0),
                    input(ep("ok1", 200_000, "0"), 1.0),
                    input(ep("ok2", 200_000, "0"), 2.0),
                    input(ep("ok3", 200_000, "0"), 3.0),
                    input(ep("ok4", 200_000, "0"), 4.0),
                ],
                // n = max(3, ceil(0.8×4)) = 4 → all 4 healthy selected.
                reasons: &[
                    (0, Some(ExclusionReason::ContextTooSmall(32_000))),
                    (1, Some(ExclusionReason::ContextUnknown)),
                ],
                selected: &[2, 3, 4, 5],
            },
        ];
        for case in &cases {
            let decisions = select_providers(&case.inputs, 128_000, None);
            let selected: Vec<usize> = decisions
                .iter()
                .enumerate()
                .filter(|(_, d)| d.selected)
                .map(|(idx, _)| idx)
                .collect();
            assert_eq!(selected, case.selected, "case {}: selected", case.name);
            for &(index, ref expected) in case.reasons {
                assert_eq!(
                    decisions[index].reason, *expected,
                    "case {}: reason at index {index}",
                    case.name
                );
            }
        }
    }

    /// `effective_target_count`'s three arms — zero healthy, the 1–2 entry allowlist measured
    /// as-is, and the count rule (capped without an allowlist) — plus the allowlist through
    /// `select_providers`.
    #[test]
    fn allowlist_and_target_count() {
        assert_eq!(effective_target_count(0, None), 0);
        // The 1-2 entry allowlist is measured regardless of health, so its target is its size.
        assert_eq!(effective_target_count(0, Some(2)), 2);
        assert_eq!(effective_target_count(5, None), 4);
        assert_eq!(effective_target_count(20, None), 16);
        assert_eq!(effective_target_count(48, None), 20);
        assert_eq!(effective_target_count(30, Some(30)), 24);
        assert_eq!(effective_target_count(1, Some(2)), 2);
        assert_eq!(effective_target_count(10, Some(1)), 1);

        let mut inputs = Vec::new();
        for i in 0..4 {
            inputs.push(input(ep(&format!("p{i}"), 200_000, "0"), f64::from(i + 1)));
        }
        let wl = vec!["p1".to_string(), "p3".to_string()];
        let decisions = select_providers(&inputs, 128_000, Some(&wl));
        assert_eq!(decisions.iter().filter(|d| d.selected).count(), 2);
        assert!(decisions[1].selected && decisions[3].selected);
        assert!(!decisions[0].selected && !decisions[2].selected);

        let wl = vec!["p0".to_string()];
        let decisions = select_providers(&inputs, 128_000, Some(&wl));
        assert_eq!(decisions.iter().filter(|d| d.selected).count(), 1);
        assert!(decisions[0].selected);
    }

    /// The twenty-endpoint ceiling: a large healthy pool is clamped, a small one is not, and an
    /// explicit allowlist keeps its own count.
    #[test]
    fn selection_ceiling_caps_the_pool() {
        let healthy_pool = |n: usize| -> Vec<SelectionInput> {
            (0..n)
                .map(|i| {
                    let cost = u32::try_from(i).expect("test pool fits u32") + 1;
                    input(ep(&format!("p{i}"), 200_000, "0"), f64::from(cost))
                })
                .collect()
        };
        let selected = |inputs: &[SelectionInput], allowlist: Option<&[String]>| {
            select_providers(inputs, 128_000, allowlist)
                .iter()
                .filter(|d| d.selected)
                .count()
        };

        // 48 healthy → 0.8×48 = 39 clamped to 20.
        assert_eq!(selected(&healthy_pool(48), None), MAX_SELECTED_ENDPOINTS);
        // 30 healthy → 24 clamped to 20.
        assert_eq!(selected(&healthy_pool(30), None), 20);
        // 25 healthy → exactly 20.
        assert_eq!(selected(&healthy_pool(25), None), 20);
        // 5 healthy → 4, no cap in play.
        assert_eq!(selected(&healthy_pool(5), None), 4);
        // An allowlist that names no tag matches nothing. `--providers ""` never reaches the
        // selection as `Some` (parse turns it into `None`), so it cannot exempt a run from the
        // ceiling either.
        assert_eq!(selected(&healthy_pool(48), Some(&[])), 0);

        // A thirty-entry allowlist of healthy endpoints keeps the count rule: 0.8×30 = 24.
        let pool = healthy_pool(30);
        let allowlist: Vec<String> = (0..30).map(|i| format!("p{i}")).collect();
        assert_eq!(selected(&pool, Some(&allowlist)), 24);
    }

    /// The planned spend: the estimate sums every selected provider (padded ones included) and
    /// the guard is `cap × 2 / selected`.
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
}
