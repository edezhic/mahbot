//! Cache classification for the benchmark's TTL ladder.
//!
//! Each ladder round compares the provider's reported cached tokens against
//! what a full cache hold would produce ([`expected_cached_for_round`]) and
//! buckets the result ([`classify_round`]). The per-round classifications are
//! then collapsed into a single cache-TTL bucket ([`ttl_bucket`] +
//! [`format_bucket`]) for the report.
//!
//! Pure math — no I/O. Unit tests cover the threshold boundaries, bucket
//! derivation (including warmup handling), formatting, saturation, and the
//! tag→name pin verification.
//!
//! Phase 1 ships the full classifier (tested) ahead of the ladder executor;
//! `#![allow(dead_code)]` covers the gap until Phase 2 runs rounds and the
//! report consumes the buckets.

#![allow(dead_code)]

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
/// nominal inactivity gap for that round. Leading rounds with gap `0` are
/// warmup (they establish the base cache, not the TTL ladder) and are ignored.
///
/// Rules:
/// - If every non-warmup round is [`CacheClassification::NotSupported`] →
///   `"not supported"` (the provider never produces cached tokens).
/// - [`CacheClassification::Invalid`] rounds are skipped for bucketing (they
///   neither count as a Drop nor terminate the search).
/// - The first [`CacheClassification::Drop`] at ladder index `k` gives
///   `(gap[k−1], gap[k]]` — the cache survived the previous gap but not `gap[k]`.
///   `k == 0` (the very first ladder round) → `"immediate drop"`.
/// - No Drop at all → `"≥{last gap}"` via [`format_bucket`] (e.g. `"≥30m"`
///   for the full default ladder; a shortened ladder reports its actual last
///   gap). Partial rounds count as held (they are not Drops).
#[must_use]
pub(crate) fn ttl_bucket(
    classifications: &[CacheClassification],
    nominal_gaps_secs: &[f64],
) -> String {
    // Ladder rounds only: skip leading warmup (gap 0) rounds. Invalid rounds
    // stay in place — they neither count as a Drop nor terminate the search,
    // but the bucket boundaries come from the ladder's own gap values, so a
    // Drop after an Invalid round still buckets against the preceding gap.
    let ladder_start = nominal_gaps_secs
        .iter()
        .position(|&g| g > 0.0)
        .unwrap_or(nominal_gaps_secs.len());

    let ladder: Vec<(CacheClassification, f64)> = classifications
        .iter()
        .zip(nominal_gaps_secs)
        .skip(ladder_start)
        .map(|(c, g)| (*c, *g))
        .collect();

    if ladder.is_empty() {
        return "not supported".to_string();
    }

    // Every usable (non-Invalid) non-warmup round NotSupported → the provider
    // never produces cached tokens.
    let usable: Vec<CacheClassification> = ladder
        .iter()
        .map(|(c, _)| *c)
        .filter(|c| *c != CacheClassification::Invalid)
        .collect();
    if !usable.is_empty()
        && usable
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
        if k == 0 {
            return "immediate drop".to_string();
        }
        return format_bucket(ladder[k - 1].1, ladder[k].1);
    }

    // No Drop: the cache held through the whole ladder.
    let last_gap = ladder.last().map_or(0.0, |(_, g)| *g);
    format_bucket(last_gap, f64::INFINITY)
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
        // Drop at ladder index 1 (round 3 overall): gaps (5s, 30s].
        let classes = [C::Hit, C::Hit, C::Hit, C::Drop, C::Hit];
        let gaps = [0.0, 0.0, 5.0, 30.0, 120.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "(5s, 30s]");

        // All hits → held through the full ladder → ≥30m (1800s).
        let classes = [C::Hit, C::Hit, C::Hit, C::Hit, C::Hit];
        let gaps = [0.0, 0.0, 5.0, 30.0, 1800.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "≥30m");

        // Shortened ladder (last gap 600s) → ≥10m.
        let classes = [C::Hit, C::Hit, C::Hit];
        let gaps = [0.0, 0.0, 600.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "≥10m");

        // First ladder round drops → immediate drop.
        let classes = [C::Hit, C::Hit, C::Drop, C::Hit];
        let gaps = [0.0, 0.0, 5.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "immediate drop");

        // All non-warmup NotSupported → not supported.
        let classes = [C::NotSupported, C::NotSupported, C::NotSupported];
        let gaps = [0.0, 5.0, 30.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "not supported");

        // Invalid rounds are skipped, not Drops.
        let classes = [C::Hit, C::Hit, C::Invalid, C::Drop, C::Hit];
        let gaps = [0.0, 0.0, 5.0, 30.0, 120.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "(5s, 30s]");

        // Partial counts as held (not a Drop): the Drop at 120s buckets
        // against the Partial round's 30s gap, not the 5s gap (a Partial
        // treated as a Drop would give "(5s, 30s]").
        let classes = [C::Hit, C::Hit, C::Hit, C::Partial, C::Drop];
        let gaps = [0.0, 0.0, 5.0, 30.0, 120.0];
        assert_eq!(ttl_bucket(&classes, &gaps), "(30s, 2m]");
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
            model_id: None,
            model_name: None,
            context_length: Some(200_000),
            max_completion_tokens: None,
            quantization: Some("fp8".to_string()),
            status: Some("0".to_string()),
            supports_implicit_caching: Some(true),
            supported_parameters: None,
            pricing: None,
            latency_last_30m: None,
            uptime_last_1d: None,
            uptime_last_30m: None,
            uptime_last_5m: None,
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
