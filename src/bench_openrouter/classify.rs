//! Cache classification for the benchmark's TTL ladder.
//!
//! Each ladder round compares the provider's reported cached tokens against what a full cache
//! hold would produce ([`expected_cached_for_round`]) and buckets the result
//! ([`classify_round`]); the per-round classifications collapse into one cache-TTL bucket
//! ([`ttl_bucket`] + [`format_bucket`]) and, through [`cache_hold_result`], into the report's
//! outcome vocabulary: "does not cache", a TTL bucket, "not measured", or one of
//! [`truncated_result`]'s two truncation forms. When a row counts as truncated is
//! [`is_truncated`]'s rule; the formatter that spells it is [`truncated_result`]'s.
//!
//! [`format_duration`] is the single duration formatter — the summary's ladder line and the
//! bucket labels call it, so `3600` can never read `60m` in one place and `1h` in the other.
//!
//! Pure math — no I/O.

use super::discovery::EndpointInfo;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheClassification {
    /// Cached tokens >= 90% of expected — cache held.
    Hit,
    /// Between 10% and 90% — partially served from cache.
    Partial,
    /// Below 10% of expected — cache dropped.
    Drop,
    /// Provider reports no cache at all (expected cached tokens are 0).
    NotSupported,
    /// Round unusable (transport error etc.) — skipped, not a Drop.
    Invalid,
}

impl CacheClassification {
    #[must_use]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Partial => "partial",
            Self::Drop => "drop",
            Self::NotSupported => "not_supported",
            Self::Invalid => "invalid",
        }
    }
}

/// Classify one ladder round: cached tokens vs the full-hold expectation.
///
/// `expected == 0` → [`CacheClassification::NotSupported`]. Otherwise ratio >= 0.9 → Hit,
/// ratio < 0.1 → Drop, in between → Partial.
#[must_use]
// u64→f64 is exact for token counts far below 2^53 — ratio math needs floats.
#[expect(clippy::cast_precision_loss)]
pub(crate) fn classify_round(cached: u64, expected_cached: u64) -> CacheClassification {
    if expected_cached == 0 {
        return CacheClassification::NotSupported;
    }
    let ratio = cached as f64 / expected_cached as f64;
    if ratio >= 0.9 {
        CacheClassification::Hit
    } else if ratio < 0.1 {
        CacheClassification::Drop
    } else {
        CacheClassification::Partial
    }
}

/// True for the classifications that establish a cache hold: a round that observed the expected
/// cached prefix, wholly or partly.
const fn holds(class: CacheClassification) -> bool {
    matches!(
        class,
        CacheClassification::Hit | CacheClassification::Partial
    )
}

/// Collapse the per-round classifications into one TTL bucket string.
///
/// `classifications[i]` is ladder round `i`, `nominal_gaps_secs[i]` its nominal inactivity gap.
/// The arrays cover ladder rounds ONLY — the first entry is the gap-0 probe. Warmup rounds are
/// never passed in.
///
/// - Empty input → `"not measured"`.
/// - [`CacheClassification::Invalid`] rounds are skipped: they neither count as a Drop nor
///   establish a hold; all-Invalid → `"not measured"` (fail-closed).
/// - All non-Invalid rounds [`CacheClassification::NotSupported`] → `"does not cache"`.
/// - The first [`CacheClassification::Drop`] at index `k`: at a `0.0` gap it is
///   `"immediate drop"`; otherwise walk back to the nearest `Hit`/`Partial` at `j` — found →
///   `(gap[j], gap[k]]`, not found → `(0, gap[k]]` rendered `≤{gap[k]}`.
/// - No Drop → `"≥{gap[j]}"` where `j` is the LAST round that established a hold, so trailing
///   unmeasured rounds cannot inflate the bucket. Partial counts as held.
#[must_use]
fn ttl_bucket(classifications: &[CacheClassification], nominal_gaps_secs: &[f64]) -> String {
    let ladder: Vec<(CacheClassification, f64)> = classifications
        .iter()
        .zip(nominal_gaps_secs)
        .map(|(c, g)| (*c, *g))
        .collect();

    if ladder.is_empty() {
        return "not measured".to_string();
    }

    // Every usable round NotSupported → the provider never produces cached tokens; an
    // all-Invalid ladder → nothing was measured.
    let usable: Vec<CacheClassification> = ladder
        .iter()
        .map(|(c, _)| *c)
        .filter(|c| *c != CacheClassification::Invalid)
        .collect();
    if usable.is_empty() {
        return "not measured".to_string();
    }
    if usable
        .iter()
        .all(|c| *c == CacheClassification::NotSupported)
    {
        return "does not cache".to_string();
    }

    // First Drop in the ladder (Invalid/NotSupported skipped by the search).
    if let Some(k) = ladder
        .iter()
        .position(|(c, _)| *c == CacheClassification::Drop)
    {
        // A Drop at a 0-gap rung (the probe at index 0, or a custom ladder's extra one) is an
        // immediate drop.
        if ladder[k].1 == 0.0 {
            return "immediate drop".to_string();
        }
        // Walk back from k−1 to the nearest round that established a hold.
        for (j, (c, _)) in ladder.iter().enumerate().take(k).rev() {
            if holds(*c) {
                return format_bucket(ladder[j].1, ladder[k].1);
            }
        }
        // No prior Hit/Partial — no observed lower bound.
        return format_bucket(0.0, ladder[k].1);
    }

    // No Drop: walk back to the LAST round that established a hold; trailing
    // Invalid/NotSupported rounds were unmeasured and must not inflate the bucket.
    for (j, (c, _)) in ladder.iter().enumerate().rev() {
        if holds(*c) {
            return format_bucket(ladder[j].1, f64::INFINITY);
        }
    }
    // No round ever established a hold (unreachable given the guards, but fail closed).
    "not measured".to_string()
}

/// True iff the ladder ended before its last rung without an observed drop — the only case where
/// the answer could still have changed. [`crate::bench_openrouter::run::run_provider`] sets the
/// report row's truncation flag from this same predicate, so the flag and the suffix cannot
/// disagree.
#[must_use]
pub(crate) fn is_truncated(
    ladder_completed: bool,
    classifications: &[CacheClassification],
) -> bool {
    !ladder_completed && !classifications.contains(&CacheClassification::Drop)
}

/// Suffix marking a row whose ladder was cut short before its last rung (deadline, run abort,
/// per-provider spend guard or cap). Appended by [`truncated_result`] and nowhere else; the
/// report must not re-derive it.
const LADDER_TRUNCATION_SUFFIX: &str = " (ladder truncated)";

/// The truncation form of a ladder that ended before its last rung without an observed drop:
/// `≥{gap}` for the highest rung at which cached tokens were still observed, `not measured` when
/// no rung observed any — a cut-short ladder can never claim `"does not cache"`, which is a
/// statement about the whole ladder. The 0 s probe is a rung like any other, so a hold seen only
/// there reads `≥0s`.
///
/// Takes only the two arrays it reads, so a caller whose ladder never ran (both empty) needs no
/// invented observations.
#[must_use]
pub(crate) fn truncated_result(
    classifications: &[CacheClassification],
    nominal_gaps_secs: &[f64],
) -> String {
    // The highest holding rung, not the last one: a custom ladder need not be sorted.
    let held_gap = classifications
        .iter()
        .zip(nominal_gaps_secs)
        .filter(|(class, _)| holds(**class))
        .map(|(_, gap)| *gap)
        .max_by(f64::total_cmp);
    held_gap.map_or_else(
        || format!("not measured{LADDER_TRUNCATION_SUFFIX}"),
        |gap| format!("≥{}{LADDER_TRUNCATION_SUFFIX}", format_duration(gap)),
    )
}

/// The cache-hold result for a provider: `"not measured"` when no ladder round ran or none
/// measured anything, `"does not cache"` when no round ever reported non-zero cached tokens,
/// otherwise the TTL bucket derived from the ladder.
///
/// `ladder_completed == false` means the ladder was cut short (deadline, run abort, per-provider
/// spend guard or cap). Such a row is truncated unless a [`CacheClassification::Drop`] was
/// observed — a drop settles the bucket, so it is reported as-is; the truncation forms are
/// [`truncated_result`]'s.
#[must_use]
pub(crate) fn cache_hold_result(
    any_cached_observation: bool,
    classifications: &[CacheClassification],
    nominal_gaps_secs: &[f64],
    ladder_completed: bool,
) -> String {
    // Truncated rows report only what the run actually saw. An observed drop is not truncated:
    // the walk-back already brackets the TTL from both sides.
    if is_truncated(ladder_completed, classifications) {
        return truncated_result(classifications, nominal_gaps_secs);
    }
    if classifications.is_empty() {
        return "not measured".to_string();
    }
    // Fail closed: an all-Invalid ladder measured nothing (transport cascade, persistent cache
    // hits, abort). The zero-cache gate must not label such a run "does not cache" on the
    // strength of the single W2 sample the predicate exists to distrust.
    if classifications
        .iter()
        .all(|c| *c == CacheClassification::Invalid)
    {
        return "not measured".to_string();
    }
    // Zero cache evidence across W2 + ladder rounds → the provider does not cache.
    if !any_cached_observation {
        return "does not cache".to_string();
    }
    // Cache evidence exists but every usable round measured no expected hold (all
    // NotSupported) — the hold itself was never measured.
    let usable: Vec<CacheClassification> = classifications
        .iter()
        .copied()
        .filter(|c| *c != CacheClassification::Invalid)
        .collect();
    if usable
        .iter()
        .all(|c| *c == CacheClassification::NotSupported)
    {
        return "not measured".to_string();
    }
    ttl_bucket(classifications, nominal_gaps_secs)
}

/// Format a TTL bucket `(lo, hi]` with human durations: `0s`/`≤5s`/`(5m, 10m]`/`≥30m`.
/// `lo == 0.0` renders `≤{hi}`, `hi == ∞` renders `≥{lo}`, a degenerate `(0, 0]` renders `0s`.
#[must_use]
fn format_bucket(lo: f64, hi: f64) -> String {
    if hi.is_infinite() {
        format!("≥{}", format_duration(lo))
    } else if lo == 0.0 {
        if hi == 0.0 {
            "0s".to_string()
        } else {
            format!("≤{}", format_duration(hi))
        }
    } else {
        format!("({}, {}]", format_duration(lo), format_duration(hi))
    }
}

/// Human duration: `0s`, `{s}s` (<60s), `{m}m` (<3600s), `{h}h` (else), all rounded.
///
/// The one duration formatter for the bench: bucket labels and the summary's ladder line both
/// go through it, so `3600` is `1h` in both places.
#[must_use]
// Inputs are non-negative gap seconds; round-then-truncate is the intent.
#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn format_duration(secs: f64) -> String {
    if secs == 0.0 {
        "0s".to_string()
    } else if secs < 60.0 {
        format!("{}s", secs.round() as u64)
    } else if secs < 3600.0 {
        format!("{}m", (secs / 60.0).round() as u64)
    } else {
        format!("{}h", (secs / 3600.0).round() as u64)
    }
}

/// Tokens a full cache hold would produce for a ladder round: the base cache plus the prompt
/// growth since the base round (saturating — a shrinking prompt cannot lower the base).
#[must_use]
pub(crate) fn expected_cached_for_round(
    base_cached: u64,
    prompt_tokens_round: u64,
    prompt_tokens_base: u64,
) -> u64 {
    base_cached.saturating_add(prompt_tokens_round.saturating_sub(prompt_tokens_base))
}

/// True iff the serving provider (from the response metadata) matches the endpoint's `name` or
/// `provider_name` (case-sensitive exact match).
///
/// This is the tag→name map check: the discovery snapshot maps the pinning `tag` to the names
/// OpenRouter reports, so a pinned run can verify it hit the intended provider.
#[must_use]
pub(crate) fn verify_pinned(serving_provider: Option<&str>, endpoint: &EndpointInfo) -> bool {
    match serving_provider {
        Some(p) => p == endpoint.name || p == endpoint.provider_name,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_round_thresholds() {
        assert_eq!(classify_round(9, 10), CacheClassification::Hit); // ratio 0.9
        assert_eq!(classify_round(899, 1000), CacheClassification::Partial); // 0.899
        assert_eq!(classify_round(1, 10), CacheClassification::Partial); // 0.1
        assert_eq!(classify_round(99, 1000), CacheClassification::Drop); // 0.099
        assert_eq!(classify_round(0, 1000), CacheClassification::Drop);
        assert_eq!(classify_round(5, 0), CacheClassification::NotSupported);
    }

    #[test]
    fn ttl_bucket_derivation() {
        use CacheClassification as C;
        // Ladder-only arrays; the first entry is the gap-0 probe.
        // 1. Drop at index 2; the nearest prior Hit is at gap 5 → (5s, 30s]. The trailing Hit
        //    was still walked and does not move the settled bucket.
        let classes = [C::Hit, C::Hit, C::Drop, C::Hit];
        let gaps = [0.0, 5.0, 30.0, 120.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "(5s, 30s]");

        // 2. No Drop → held through the full ladder → ≥30m (1800s).
        let classes = [C::Hit, C::Hit, C::Hit, C::Hit, C::Hit];
        let gaps = [0.0, 5.0, 30.0, 120.0, 1800.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "≥30m");

        // 3. Shortened ladder (last gap 600s) → ≥10m.
        let classes = [C::Hit, C::Hit, C::Hit];
        let gaps = [0.0, 5.0, 600.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "≥10m");

        // 4. The gap-0 probe itself dropped → genuine immediate drop.
        let classes = [C::Drop, C::Hit, C::Hit];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "immediate drop");

        // 5. Held through the 0s probe, dropped at the 5s rung → ≤5s, NOT an immediate drop.
        let classes = [C::Hit, C::Drop, C::Hit];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "≤5s");

        // 6. All non-Invalid rounds NotSupported → does not cache.
        let classes = [C::NotSupported, C::NotSupported, C::NotSupported];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "does not cache");

        // 7. Invalid rounds are skipped for the NotSupported check too.
        let classes = [C::NotSupported, C::Invalid, C::NotSupported];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "does not cache");

        // 8. All-Invalid ladder fails closed → not measured.
        let classes = [C::Invalid, C::Invalid, C::Invalid];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "not measured");

        // 9. Empty input → not measured.
        assert_eq!(ttl_bucket(&[], &[]), "not measured");

        // 10. The index-1 round is Invalid, so the only observed hold is the gap-0 Hit →
        //     honest lower bound 0 → ≤30s.
        let classes = [C::Hit, C::Invalid, C::Drop, C::Hit];
        let gaps = [0.0, 5.0, 30.0, 120.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "≤30s");

        // 11. No prior Hit/Partial at all → no observed lower bound → ≤30s.
        let classes = [C::Invalid, C::Invalid, C::Drop];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "≤30s");

        // 12. Partial counts as held (not a Drop): the walk-back finds it and buckets the Drop
        //     at 120s against the Partial's 30s gap.
        let classes = [C::Hit, C::Hit, C::Partial, C::Drop];
        let gaps = [0.0, 5.0, 30.0, 120.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "(30s, 2m]");

        // 13. Only the 0s hold was observed; trailing Invalids must NOT inflate the bucket.
        let classes = [C::Hit, C::Invalid, C::Invalid];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "≥0s");

        // 14. Last observed hold at the 5s rung; trailing Invalids → ≥5s.
        let classes = [C::Hit, C::Hit, C::Invalid, C::Invalid];
        let gaps = [0.0, 5.0, 30.0, 120.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "≥5s");

        // 15. Custom ladder with a second 0-gap round: a Drop there is an immediate drop (it
        //     was "0s" before the gap-based check).
        let classes = [C::Hit, C::Drop, C::Hit];
        let gaps = [0.0, 0.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "immediate drop");

        // 16. The full default ladder held to its last rung: the one-hour hold is distinguishable
        //     from the half-hour one (case 2). The trailing Hit after case 1's Drop is walked and
        //     never changes the settled bucket, which is why a drop needs no early stop.
        let classes = [C::Hit; 8];
        let gaps = [0.0, 5.0, 30.0, 120.0, 300.0, 600.0, 1800.0, 3600.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "≥1h");
    }

    #[test]
    fn cache_hold_result_outcomes() {
        use CacheClassification as C;
        // (a) No ladder round ran → not measured.
        assert_eq!(cache_hold_result(true, &[], &[], true), "not measured");
        assert_eq!(cache_hold_result(false, &[], &[], true), "not measured");

        // (b) No round ever reported cached tokens → does not cache, even though the ladder
        //     rounds were usable (the streamlake mislabel regression).
        let classes = [C::Drop, C::NotSupported, C::Drop];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(
            cache_hold_result(false, &classes, &gaps, true),
            "does not cache"
        );

        // (a') Fail closed: an all-Invalid ladder measured nothing, and the zero-cache gate
        //      must not label it "does not cache" on a W2-only zero sample.
        let classes = [C::Invalid, C::Invalid, C::Invalid];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(
            cache_hold_result(false, &classes, &gaps, true),
            "not measured"
        );

        // (b') Mixed: usable Drop rounds exist, but zero cache evidence across them.
        let classes = [C::Invalid, C::Drop, C::Invalid];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(
            cache_hold_result(false, &classes, &gaps, true),
            "does not cache"
        );

        // (c) Cached observations exist → the TTL bucket from the ladder.
        let classes = [C::Hit, C::Hit, C::Drop];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(cache_hold_result(true, &classes, &gaps, true), "(5s, 30s]");

        // (d) Cache evidence exists but the ladder measured no expected hold (every usable
        //     round NotSupported) → the hold itself was never measured.
        let classes = [C::NotSupported, C::NotSupported, C::NotSupported];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(
            cache_hold_result(true, &classes, &gaps, true),
            "not measured"
        );
    }

    /// A cut-short ladder reports only what it observed: the lower bound with the truncation
    /// suffix when a hold was seen, `not measured (ladder truncated)` otherwise — never
    /// `does not cache` and never a bare `not measured`. An observed drop settles the row, so
    /// it keeps its ordinary bucket and takes no suffix.
    #[test]
    fn cache_hold_result_truncated_ladder() {
        use CacheClassification as C;
        // Held through the 1800s rung, cut before the 3600s rung → the lower bound stands. The
        // two arrays are always pushed in lockstep, so the cut-short ladder carries 7 of the 8
        // default rungs.
        let classes = [C::Hit; 7];
        let gaps = [0.0, 5.0, 30.0, 120.0, 300.0, 600.0, 1800.0];
        assert_eq!(
            cache_hold_result(true, &classes, &gaps, false),
            "≥30m (ladder truncated)"
        );

        // No rung ever established a hold → nothing to report about the hold.
        let classes = [C::Invalid, C::Invalid, C::Invalid];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(
            cache_hold_result(false, &classes, &gaps, false),
            "not measured (ladder truncated)"
        );

        // Cached observations existed, but every usable round was NotSupported → the cut-short
        // ladder must not claim "does not cache".
        let classes = [C::NotSupported, C::NotSupported, C::NotSupported];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(
            cache_hold_result(true, &classes, &gaps, false),
            "not measured (ladder truncated)"
        );

        // Zero-cache evidence and a cut-short ladder → still no negative claim.
        let classes = [C::NotSupported, C::Invalid, C::NotSupported];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(
            cache_hold_result(false, &classes, &gaps, false),
            "not measured (ladder truncated)"
        );

        // No ladder round ran at all, yet the ladder was reported as cut short.
        assert_eq!(
            cache_hold_result(true, &[], &[], false),
            "not measured (ladder truncated)"
        );

        // A hold seen only at the 0 s probe: the probe is a ladder rung, so it bounds the hold at
        // `≥0s` rather than claiming nothing was measured.
        let classes = [C::Hit, C::Invalid, C::Invalid];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(
            cache_hold_result(true, &classes, &gaps, false),
            "≥0s (ladder truncated)"
        );

        // A custom ladder need not be sorted: the bound is the highest holding rung, not the last
        // one, so a 0-gap rung above a real one cannot discard the real bound.
        let classes = [C::Hit, C::Hit, C::Hit, C::Invalid];
        let gaps = [0.0, 5.0, 0.0, 30.0];
        assert_eq!(
            cache_hold_result(true, &classes, &gaps, false),
            "≥5s (ladder truncated)"
        );

        // An observed drop settles the bucket: the row is not truncated and takes no suffix.
        let classes = [C::Hit, C::Hit, C::Drop];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(cache_hold_result(true, &classes, &gaps, false), "(5s, 30s]");

        // A drop at the 0s probe is an immediate drop, suffix-free.
        let classes = [C::Drop, C::Invalid];
        let gaps = [0.0, 5.0];
        assert_eq!(
            cache_hold_result(true, &classes, &gaps, false),
            "immediate drop"
        );
    }

    /// The summary's ladder line and the bucket labels share one formatter, so the same gap can
    /// never render `60m` in one place and `1h` in the other.
    #[test]
    fn ladder_line_and_bucket_labels_agree() {
        for (secs, label) in [
            (120.0, "2m"),
            (300.0, "5m"),
            (600.0, "10m"),
            (1800.0, "30m"),
            (3600.0, "1h"),
        ] {
            assert_eq!(format_duration(secs), label, "format_duration({secs})");
            assert_eq!(
                format_bucket(secs, f64::INFINITY),
                format!("≥{label}"),
                "format_bucket({secs}, ∞)"
            );
        }
        // The remaining forms are unchanged: 0s, sub-minute seconds, ≤ and (lo, hi].
        assert_eq!(format_duration(0.0), "0s");
        assert_eq!(format_duration(5.0), "5s");
        assert_eq!(format_duration(59.4), "59s");
        assert_eq!(format_bucket(0.0, 0.0), "0s");
        assert_eq!(format_bucket(0.0, 5.0), "≤5s");
        assert_eq!(format_bucket(300.0, 600.0), "(5m, 10m]");
    }

    #[test]
    fn verify_pinned_matches_name_or_provider_name() {
        let endpoint = EndpointInfo {
            tag: "acme/fp8".to_string(),
            name: "Acme Cloud".to_string(),
            provider_name: "Acme".to_string(),
            context_length: Some(200_000),
            quantization: Some("fp8".to_string()),
            status: Some("0".to_string()),
            supports_implicit_caching: Some(true),
            pricing: None,
        };
        assert!(verify_pinned(Some("Acme Cloud"), &endpoint)); // name
        assert!(verify_pinned(Some("Acme"), &endpoint)); // provider_name
        assert!(!verify_pinned(Some("Other"), &endpoint)); // mismatch
        assert!(!verify_pinned(None, &endpoint)); // no provider reported
        // Case-sensitive: "acme" != "Acme".
        assert!(!verify_pinned(Some("acme"), &endpoint));
    }
}
