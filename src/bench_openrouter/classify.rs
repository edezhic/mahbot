//! Cache classification for the benchmark's TTL ladder.
//!
//! Each ladder round compares the provider's reported cached tokens against
//! what a full cache hold would produce ([`expected_cached_for_round`]) and
//! buckets the result ([`classify_round`]). The per-round classifications are
//! then collapsed into a single cache-TTL bucket ([`ttl_bucket`] +
//! [`format_bucket`]) for the report.
//!
//! Pure math — no I/O. Unit tests cover the threshold boundaries, bucket
//! derivation (Invalid rounds are skipped; an all-Invalid ladder yields
//! "not measured"), formatting, saturation, and the tag→name pin
//! verification.

use super::discovery::EndpointInfo;

// ── Per-round classification ───────────────────────────────────────

/// One ladder round's cache outcome.
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
/// `expected == 0` → [`CacheClassification::NotSupported`] (nothing was ever
/// cached, so a hold can't be measured). Otherwise ratio >= 0.9 → Hit,
/// ratio < 0.1 → Drop, in between → Partial.
#[must_use]
// u64→f64 is exact for token counts far below 2^53 — ratio math needs floats.
#[allow(clippy::cast_precision_loss)]
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

// ── TTL bucket derivation ──────────────────────────────────────────

/// Collapse the per-round classifications into one TTL bucket string.
///
/// `classifications[i]` is ladder round `i`; `nominal_gaps_secs[i]` is the
/// nominal inactivity gap for that round. The arrays cover ladder rounds ONLY:
/// the first entry is the gap-0 probe (nominal gap `0.0`). Warmup rounds are
/// never passed in.
///
/// Rules:
/// - Empty input → `"not measured"`.
/// - [`CacheClassification::Invalid`] rounds are skipped: they neither count
///   as a Drop nor establish a hold.
/// - If every round is Invalid → `"not measured"` (fail-closed — no hold was
///   measured).
/// - If the non-Invalid rounds are all [`CacheClassification::NotSupported`] →
///   `"not supported"` (the provider never produces cached tokens).
/// - The first [`CacheClassification::Drop`] at ladder index `k`:
///   - the Drop is at a round whose nominal gap is `0.0` (the gap-0 probe,
///     or a custom ladder's extra 0-gap rung) → `"immediate drop"`.
///   - otherwise walk back from `k−1` to the nearest index `j` classified
///     `Hit` or `Partial` (the only rounds that establish the cache held
///     through their gap): found → `(gap[j], gap[k]]`; not found (all prior
///     rounds Invalid/NotSupported) → `(0, gap[k]]` — no observed lower
///     bound, rendered `≤{gap[k]}`.
/// - No Drop → `"≥{gap[j]}"` via [`format_bucket`] where `j` is the LAST
///   round that established a hold (`Hit`/`Partial`) — the cache held
///   through `gap[j]`, and trailing `Invalid`/`NotSupported` rounds after it
///   were unmeasured and must not inflate the bucket (e.g. `"≥30m"` for the
///   full default ladder; a shortened ladder reports its actual last gap).
///   Partial rounds count as held (they are not Drops).
#[must_use]
pub(crate) fn ttl_bucket(
    classifications: &[CacheClassification],
    nominal_gaps_secs: &[f64],
) -> String {
    let ladder: Vec<(CacheClassification, f64)> = classifications
        .iter()
        .zip(nominal_gaps_secs)
        .map(|(c, g)| (*c, *g))
        .collect();

    if ladder.is_empty() {
        return "not measured".to_string();
    }

    // Every usable (non-Invalid) round NotSupported → the provider never
    // produces cached tokens; an all-Invalid ladder → nothing was measured.
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
        return "not supported".to_string();
    }

    // First Drop in the ladder (Invalid/NotSupported skipped by the search).
    if let Some(k) = ladder
        .iter()
        .position(|(c, _)| *c == CacheClassification::Drop)
    {
        // A Drop at a 0-gap rung (the gap-0 probe at index 0, or an extra
        // 0-gap round in a custom ladder) is an immediate drop.
        if ladder[k].1 == 0.0 {
            return "immediate drop".to_string();
        }
        // Walk back from k−1 to the nearest round that established a hold.
        for (j, (c, _)) in ladder.iter().enumerate().take(k).rev() {
            if matches!(c, CacheClassification::Hit | CacheClassification::Partial) {
                return format_bucket(ladder[j].1, ladder[k].1);
            }
        }
        // No prior Hit/Partial — no observed lower bound.
        return format_bucket(0.0, ladder[k].1);
    }

    // No Drop: walk back to the LAST round that established a hold — the
    // cache held through its gap; trailing Invalid/NotSupported rounds
    // after it were unmeasured and must not inflate the bucket.
    for (j, (c, _)) in ladder.iter().enumerate().rev() {
        if matches!(c, CacheClassification::Hit | CacheClassification::Partial) {
            return format_bucket(ladder[j].1, f64::INFINITY);
        }
    }
    // No round ever established a hold (the guards above make this
    // unreachable, but fail closed rather than fabricate a bucket).
    "not measured".to_string()
}

/// Format a TTL bucket `(lo, hi]` with human durations:
/// `0s`/`≤5s`/`(5m, 10m]`/`≥30m`. `lo == 0.0` renders `≤{hi}` (e.g. `≤5s`);
/// `hi == ∞` renders `≥{lo}` (`lo == 0.0` → `≥0s`); a degenerate `(0, 0]`
/// bucket renders `0s`.
#[must_use]
pub(crate) fn format_bucket(lo: f64, hi: f64) -> String {
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

/// Human duration: `0s`, `{s}s` (<60s, rounded), `{m}m` (<3600s, rounded),
/// `{h}h` (else, rounded).
#[must_use]
// Inputs are non-negative gap seconds; round-then-truncate is the intent.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn format_duration(secs: f64) -> String {
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

// ── Expected cached tokens ─────────────────────────────────────────

/// Tokens a full cache hold would produce for a ladder round: the base cache
/// plus the prompt growth since the base round (saturating — a shrinking
/// prompt cannot reduce the expectation below the base cache).
#[must_use]
pub(crate) fn expected_cached_for_round(
    base_cached: u64,
    prompt_tokens_round: u64,
    prompt_tokens_base: u64,
) -> u64 {
    base_cached.saturating_add(prompt_tokens_round.saturating_sub(prompt_tokens_base))
}

// ── Provider pin verification ──────────────────────────────────────

/// True iff the serving provider (from the response metadata) matches the
/// endpoint's `name` or `provider_name` (case-sensitive exact match).
///
/// This is the tag→name map check: the discovery snapshot's endpoint list
/// provides the mapping from the provider-pinning `tag` to the human names
/// OpenRouter reports back, so a pinned run can verify it actually hit the
/// intended provider.
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
        // 1. Drop at index 2; the nearest prior Hit is at gap 5 → (5s, 30s].
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

        // 5. Held through the 0s probe, dropped at the 5s rung → ≤5s, NOT an
        //    immediate drop (replaces the old misleading warmup-skew case).
        let classes = [C::Hit, C::Drop, C::Hit];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "≤5s");

        // 6. All non-Invalid rounds NotSupported → not supported.
        let classes = [C::NotSupported, C::NotSupported, C::NotSupported];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "not supported");

        // 7. Invalid rounds are skipped for the NotSupported check too.
        let classes = [C::NotSupported, C::Invalid, C::NotSupported];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "not supported");

        // 8. All-Invalid ladder fails closed → not measured.
        let classes = [C::Invalid, C::Invalid, C::Invalid];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "not measured");

        // 9. Empty input → not measured.
        assert_eq!(ttl_bucket(&[], &[]), "not measured");

        // 10. The index-1 round is Invalid, so the only observed hold is the
        //     gap-0 Hit → honest lower bound 0 → ≤30s.
        let classes = [C::Hit, C::Invalid, C::Drop, C::Hit];
        let gaps = [0.0, 5.0, 30.0, 120.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "≤30s");

        // 11. No prior Hit/Partial at all → no observed lower bound → ≤30s.
        let classes = [C::Invalid, C::Invalid, C::Drop];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "≤30s");

        // 12. Partial counts as held (not a Drop): the walk-back finds it and
        //     buckets the Drop at 120s against the Partial's 30s gap.
        let classes = [C::Hit, C::Hit, C::Partial, C::Drop];
        let gaps = [0.0, 5.0, 30.0, 120.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "(30s, 2m]");

        // 13. Only the 0s hold was observed; trailing Invalids are unmeasured
        //     and must NOT inflate the bucket to ≥30m.
        let classes = [C::Hit, C::Invalid, C::Invalid];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "≥0s");

        // 14. Last observed hold at the 5s rung; trailing Invalids after it
        //     are unmeasured → ≥5s.
        let classes = [C::Hit, C::Hit, C::Invalid, C::Invalid];
        let gaps = [0.0, 5.0, 30.0, 120.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "≥5s");

        // 15. Custom ladder with a second 0-gap round: a Drop at that rung is
        //     an immediate drop (was "0s" before the gap-based check).
        let classes = [C::Hit, C::Drop, C::Hit];
        let gaps = [0.0, 0.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "immediate drop");
    }

    #[test]
    fn format_bucket_ranges() {
        assert_eq!(format_bucket(0.0, 0.0), "0s");
        assert_eq!(format_bucket(0.0, 5.0), "≤5s");
        assert_eq!(format_bucket(0.0, 0.5), "≤1s"); // rounded
        assert_eq!(format_bucket(1800.0, f64::INFINITY), "≥30m");
        assert_eq!(format_bucket(0.0, f64::INFINITY), "≥0s");
        assert_eq!(format_bucket(300.0, 600.0), "(5m, 10m]");
        assert_eq!(format_bucket(5.0, 30.0), "(5s, 30s]");
        assert_eq!(format_bucket(3600.0, 7200.0), "(1h, 2h]");
    }

    #[test]
    fn expected_cached_saturates() {
        assert_eq!(expected_cached_for_round(1000, 1200, 800), 1400); // base + growth
        assert_eq!(expected_cached_for_round(1000, 800, 1200), 1000); // shrink → saturate
        assert_eq!(expected_cached_for_round(0, 500, 500), 0);
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

    #[test]
    fn classification_as_str() {
        use CacheClassification as C;
        assert_eq!(C::Hit.as_str(), "hit");
        assert_eq!(C::Partial.as_str(), "partial");
        assert_eq!(C::Drop.as_str(), "drop");
        assert_eq!(C::NotSupported.as_str(), "not_supported");
        assert_eq!(C::Invalid.as_str(), "invalid");
    }
}
