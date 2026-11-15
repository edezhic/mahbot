//! Report artifacts for `mahbot bench-openrouter` full runs.
//!
//! The full-run executor writes four artifacts under `<output-dir>/bench-openrouter/<run-ts>/`:
//!
//! - `report.json` — the complete structured run report: run meta (including the usage mix the
//!   static price is computed at, the privacy source, the ladder-completion flag and the
//!   healthy/selected endpoint counts), one entry per discovered endpoint (`tag` / `selected` /
//!   `static_price_usd_per_1b` / `cache_hold` / `privacy_opt_in_required`), and the artifact paths.
//! - `summary.md` — the human-readable markdown summary (meta table, the counts and privacy notes
//!   above the endpoint table, the four-column endpoint table, a methodology derived from the
//!   constants, and an abort note when applicable).
//! - `providers.json` — the verbatim discovery snapshot, with the key payload redacted, PLUS a
//!   derived `privacy` section: the extracted per-endpoint policy entries, never the whole
//!   upstream document, so the file stays trustworthy as an upstream record.
//! - `manifest.json` — the run manifest: scrubbed argv + env entries + config + outcome. The API
//!   key value is ALWAYS scrubbed (see [`build_manifest`]).
//!
//! `<output-dir>` defaults to `~/.mahbot/benchmarks/<model-slug>/`, so parallel runs of different
//! models write separate folders and take separate output locks. Every artifact is mirrored into
//! `<output-dir>/bench-openrouter/latest/`, so a consumer can read the most recent run from a
//! stable path.
//!
//! All `build_*` functions are pure and unit-tested against synthetic inputs; [`write_artifacts`]
//! and [`acquire_run_lock`] are the only I/O boundaries.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::json;

use crate::bench_openrouter::classify::format_duration;
use crate::bench_openrouter::discovery::{DiscoverySnapshot, EndpointInfo, Pricing};
use crate::bench_openrouter::privacy::{PrivacySnapshot, UNAVAILABLE};
use crate::bench_openrouter::run::ProviderRun;
use crate::bench_openrouter::select::{
    MAX_SELECTED_ENDPOINTS, SelectionDecision, USAGE_MIX_CACHED, USAGE_MIX_INPUT, USAGE_MIX_OUTPUT,
    blended_price_per_token, cache_read_price, usage_mix_json,
};
use crate::bench_openrouter::{
    BenchOptions, WARMUP_ROUNDS, discovery::parse_price, rounds_per_provider,
};

/// Token volume the reported static price is quoted at: one billion.
const PRICE_TOKENS: f64 = 1e9;

/// Price rounding scale: micro-dollars per dollar (see [`static_price_usd_per_1b`]).
const PRICE_SCALE: f64 = 1e6;

/// All artifact paths of one run. `latest_dir` is the `latest/` mirror target.
pub(crate) struct ArtifactPaths {
    pub run_dir: PathBuf,
    pub report: PathBuf,
    pub summary: PathBuf,
    pub providers: PathBuf,
    pub manifest: PathBuf,
    pub latest_dir: PathBuf,
}

/// Run directory name: `%Y%m%d-%H%M%S` (UTC), e.g. `20260820-031200`.
#[must_use]
pub(crate) fn run_dir_name(now: &chrono::DateTime<chrono::Utc>) -> String {
    now.format("%Y%m%d-%H%M%S").to_string()
}

/// Acquire the bench's own run lock: `<output_dir>/bench-openrouter.lock`. It never touches the
/// instance's `mahbot.lock`. Creates `output_dir` first and returns the locked [`File`], which
/// the caller keeps alive for the run (the kernel releases the lock when it closes).
///
/// # Errors
///
/// Fails when the directory cannot be created, the file cannot be opened, or
/// [`try_flock`](crate::util::lock::try_flock) itself errors.
pub(crate) fn acquire_run_lock(output_dir: &Path) -> anyhow::Result<std::fs::File> {
    std::fs::create_dir_all(output_dir)?;
    let path = output_dir.join("bench-openrouter.lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&path)?;
    if !crate::util::lock::try_flock(&file)? {
        anyhow::bail!(
            "another bench-openrouter run is in progress (lock: {})",
            path.display()
        );
    }
    Ok(file)
}

/// Write all four artifacts into `run_dir` and mirror them into `latest_dir`: the three JSON ones
/// pretty-printed, `summary_md` verbatim. The mirror is best-effort per file.
///
/// # Errors
///
/// Fails on any directory creation or file write error.
pub(crate) fn write_artifacts(
    paths: &ArtifactPaths,
    report: &serde_json::Value,
    summary_md: &str,
    providers_json: &serde_json::Value,
    manifest: &serde_json::Value,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(&paths.run_dir)?;
    std::fs::create_dir_all(&paths.latest_dir)?;

    std::fs::write(&paths.report, serde_json::to_string_pretty(report)?)?;
    std::fs::write(&paths.summary, summary_md)?;
    std::fs::write(
        &paths.providers,
        serde_json::to_string_pretty(providers_json)?,
    )?;
    std::fs::write(&paths.manifest, serde_json::to_string_pretty(manifest)?)?;

    // Mirror all four into latest/ (best-effort per file: propagate errors).
    std::fs::copy(&paths.report, paths.latest_dir.join("report.json"))?;
    std::fs::copy(&paths.summary, paths.latest_dir.join("summary.md"))?;
    std::fs::copy(&paths.providers, paths.latest_dir.join("providers.json"))?;
    std::fs::copy(&paths.manifest, paths.latest_dir.join("manifest.json"))?;
    Ok(())
}

/// Build the run manifest. `args` is the raw argv with any API-key material scrubbed: the
/// `--api-key` value becomes `"***"`, and any arg containing `sk-or-` is replaced wholesale. The
/// `env` object records only the three key entries, present-or-absent, values scrubbed.
#[must_use]
// The 9-arg signature is the spec'd manifest surface (scrubbed argv + config +
// outcome); bundling would obscure the manifest's fixed keys.
#[expect(clippy::too_many_arguments)]
pub(crate) fn build_manifest(
    args: &[String],
    opts: &BenchOptions,
    key_source: &str,
    started_at: &str,
    finished_at: &str,
    duration_secs: f64,
    exit_code: i32,
    aborted: bool,
    abort_reason: Option<&str>,
) -> serde_json::Value {
    let mut scrubbed: Vec<String> = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if let Some((name, value)) = arg.split_once('=')
            && name == "--api-key"
        {
            // --api-key=sk-or-xxx → --api-key=***
            scrubbed.push(format!("--api-key={}", scrub_value(value)));
            i += 1;
            continue;
        }
        if arg.contains("sk-or-") {
            scrubbed.push("***".to_string());
            i += 1;
            continue;
        }
        scrubbed.push(arg.clone());
        // --api-key VALUE → --api-key *** (scrub the following argument).
        if arg == "--api-key"
            && let Some(next) = args.get(i + 1)
        {
            scrubbed.push(scrub_value(next));
            i += 2;
            continue;
        }
        i += 1;
    }

    let mut env = serde_json::Map::new();
    if let Ok(v) = std::env::var("MAHBOT_BENCH_MODEL") {
        env.insert("MAHBOT_BENCH_MODEL".to_string(), json!(v));
    }
    if std::env::var("MAHBOT_BENCH_API_KEY").is_ok() {
        env.insert("MAHBOT_BENCH_API_KEY".to_string(), json!("***"));
    }
    if std::env::var("OPENROUTER_API_KEY").is_ok() {
        env.insert("OPENROUTER_API_KEY".to_string(), json!("***"));
    }

    json!({
        "tool": "bench-openrouter",
        "version": env!("CARGO_PKG_VERSION"),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_secs": duration_secs,
        "args": scrubbed,
        "env": env,
        "config": {
            "model": opts.model,
            "key_source": key_source,
            "cap_usd": opts.cap_usd,
            "ladder_secs": opts.ladder_secs,
            "providers_allowlist": opts.providers.clone(),
            "prefix_chars": opts.prefix_chars,
            "output_dir": opts.output_dir.display().to_string(),
        },
        "outcome": {
            "exit_code": exit_code,
            "aborted": aborted,
            "abort_reason": abort_reason,
        },
    })
}

/// The scrubbed stand-in for an API key value: `"***"` for anything non-empty, the original when
/// empty (a `--api-key=` with an empty value carries no secret).
#[must_use]
fn scrub_value(v: &str) -> String {
    if v.is_empty() {
        v.to_string()
    } else {
        "***".to_string()
    }
}

/// Run metadata shared by [`build_report`] and [`build_summary_md`]: what only the executor knows.
/// Everything [`build_report`] can derive from its own inputs — the endpoint counts, the ladder
/// flag, the privacy source — is derived there instead, so no figure can drift from the rows it
/// summarizes.
pub(crate) struct RunMeta {
    pub started_at: String,
    pub finished_at: String,
    pub duration_secs: f64,
    pub model: String,
    pub key_source: String,
    pub cap_usd: f64,
    pub ladder_secs: Vec<u64>,
    pub prefix_chars: usize,
    pub aborted: bool,
    pub abort_reason: Option<String>,
    pub exit_code: i32,
    pub output_dir: PathBuf,
}

/// Build the complete structured report (pure — no I/O). Every discovered endpoint gets a
/// providers-array entry; the cache-hold is the run's `cache_hold_bucket` when a run exists for
/// that tag, else `"not measured"`. The run-level counts, the ladder flag and the privacy source
/// are derived from the same inputs the rows are built from.
#[must_use]
pub(crate) fn build_report(
    meta: &RunMeta,
    endpoints: &[EndpointInfo],
    providers: &[ProviderRun],
    decisions: &[SelectionDecision],
    privacy: &PrivacySnapshot,
    output_dir: &Path,
) -> serde_json::Value {
    let by_tag: HashMap<&str, &ProviderRun> = providers
        .iter()
        .map(|run| (run.tag.as_str(), run))
        .collect();

    let provider_json: Vec<serde_json::Value> = endpoints
        .iter()
        .enumerate()
        .map(|(idx, ep)| {
            let cache_hold = by_tag
                .get(ep.tag.as_str())
                .map_or("not measured", |run| run.cache_hold_bucket.as_str());
            // Selection decisions are built one per discovered endpoint, so a missing one is a
            // contract break, not a data condition to paper over.
            let decision = decisions
                .get(idx)
                .expect("selection decisions parallel the discovered endpoints");
            json!({
                "tag": ep.tag,
                "selected": decision.selected,
                "static_price_usd_per_1b": static_price_usd_per_1b(ep.pricing.as_ref()),
                "cache_hold": cache_hold,
                "privacy_opt_in_required": privacy.value_for(&ep.tag).as_str(),
            })
        })
        .collect();

    let endpoints_healthy = decisions.iter().filter(|d| d.is_healthy()).count();
    let endpoints_selected = decisions.iter().filter(|d| d.selected).count();
    // False exactly when at least one row carries the truncation suffix: `run_provider` sets
    // `truncated` and the suffix from the one predicate.
    let ladder_completed = !providers.iter().any(|run| run.truncated);
    let privacy_source = if privacy.source.is_some() {
        "provider"
    } else {
        UNAVAILABLE
    };

    json!({
        "run": {
            "started_at": meta.started_at,
            "finished_at": meta.finished_at,
            "duration_secs": meta.duration_secs,
            "model": meta.model,
            "key_source": meta.key_source,
            "cap_usd": meta.cap_usd,
            "ladder_secs": meta.ladder_secs,
            "prefix_chars": meta.prefix_chars,
            "aborted": meta.aborted,
            "abort_reason": meta.abort_reason,
            "exit_code": meta.exit_code,
            "usage_mix": usage_mix_json(),
            "privacy_source": privacy_source,
            "ladder_completed": ladder_completed,
            "endpoints_healthy": endpoints_healthy,
            "endpoints_selected": endpoints_selected,
        },
        "providers": provider_json,
        "artifacts": {
            "run_dir": output_dir.display().to_string(),
            "report": "report.json",
            "summary": "summary.md",
            "providers": "providers.json",
            "manifest": "manifest.json",
        },
    })
}

/// Build the human-readable markdown summary (pure — no I/O): header + meta table, the counts and
/// privacy notes above the endpoint table, the four-column endpoint table, a methodology derived
/// from the same constants the run uses, and an abort note when the run was aborted.
#[must_use]
#[expect(
    clippy::too_many_lines,      // one sequential document, in reading order
    clippy::cast_precision_loss  // gap seconds (u64) → f64 for the shared duration formatter
)]
pub(crate) fn build_summary_md(meta: &RunMeta, report: &serde_json::Value) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();

    let _ = writeln!(out, "# OpenRouter provider benchmark — {}\n", meta.model);
    let _ = writeln!(out, "| | |\n|---|---|");
    let _ = writeln!(out, "| Started | {} |", meta.started_at);
    let _ = writeln!(out, "| Duration | {:.1} s |", meta.duration_secs);
    let _ = writeln!(out, "| Key source | {} |", meta.key_source);
    let _ = writeln!(out, "| Spend cap | ${:.2} |", meta.cap_usd);
    // The same formatter the bucket labels use, so a rung can never read `60m` here and `≥1h`
    // there.
    let ladder: Vec<String> = meta
        .ladder_secs
        .iter()
        .map(|&secs| format_duration(secs as f64))
        .collect();
    let _ = writeln!(out, "| Ladder | {} |", ladder.join(", "));
    let _ = writeln!(out, "| Output dir | {} |", meta.output_dir.display());
    if meta.aborted {
        let _ = writeln!(
            out,
            "| Aborted | {} |",
            meta.abort_reason.as_deref().unwrap_or("yes")
        );
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Providers\n");
    // The counts and the privacy source are read from the report, so the summary cannot state a
    // figure the JSON does not carry. Every one of them is written by `build_report`: a missing key
    // is a contract break here, not a data condition to paper over.
    let run = &report["run"];
    let privacy_source = run["privacy_source"]
        .as_str()
        .expect("build_report writes run.privacy_source");
    let endpoints_healthy = run["endpoints_healthy"]
        .as_u64()
        .expect("build_report writes run.endpoints_healthy");
    let endpoints_selected = run["endpoints_selected"]
        .as_u64()
        .expect("build_report writes run.endpoints_selected");
    // A dead privacy source must never look like a run where every endpoint happened to be fine.
    if privacy_source == UNAVAILABLE {
        let _ = writeln!(
            out,
            "Privacy source unavailable: every row reads `unknown` for privacy opt-in.\n"
        );
    }
    // A capped run must not be mistaken for one where every discovered endpoint was measured.
    let _ = writeln!(
        out,
        "Healthy endpoints: {endpoints_healthy}; selected for measurement: {endpoints_selected}.\n"
    );
    let _ = writeln!(
        out,
        "| Provider (tag) | Static price ($/1B) | Cache hold | Privacy opt-in |"
    );
    let _ = writeln!(out, "|---|---|---|---|");
    // Same contract as the run keys above: `build_report` writes every row key, so a missing one is
    // a contract break here, not a cell to paper over.
    let providers = report["providers"]
        .as_array()
        .expect("build_report writes the providers array");
    for p in providers {
        let tag = p["tag"].as_str().expect("build_report writes row.tag");
        let price = p["static_price_usd_per_1b"]
            .as_f64()
            .map_or_else(|| "n/a".to_string(), |v| format!("{v:.2}"));
        let hold = p["cache_hold"]
            .as_str()
            .expect("build_report writes row.cache_hold");
        let opt_in = p["privacy_opt_in_required"]
            .as_str()
            .expect("build_report writes row.privacy_opt_in_required");
        let _ = writeln!(out, "| {tag} | {price} | {hold} | {opt_in} |");
    }
    let _ = writeln!(out);

    // Every count is derived from the ladder, never restated by hand: `rounds_per_provider` owns
    // the gaps→rounds relation and the warmup count.
    let gaps = meta.ladder_secs.len();
    let requests = rounds_per_provider(&meta.ladder_secs);
    let ladder_rounds = requests - WARMUP_ROUNDS;
    let _ = writeln!(
        out,
        "## Methodology\n\n\
         - TTL ladder: per provider, {WARMUP_ROUNDS} byte-identical warmup requests + \
         {ladder_rounds} ladder rounds — the gap-0 probe plus one round per gap ({gaps} gaps, \
         {requests} requests per endpoint including the warmups); the conversation grows one \
         verified tool frame per round. Every selected endpoint walks the whole ladder — there is \
         no early stop on an observed drop.\n\
         - Cache-hold rule: a provider is \"does not cache\" when no warmup-2/ladder round reported \
         cached tokens; otherwise the bucket is the TTL hold from the ladder (interval notation \
         like `(5s, 30s]`, `≤5s`, `≥1h`, `immediate drop`). A row whose ladder ended before its \
         last rung without an observed drop is marked `(ladder truncated)` — it keeps the lower \
         bound the observed rungs give (`≥30m (ladder truncated)`) or reads `not measured (ladder \
         truncated)` when they give none; an observed drop settles the bucket, so such a row is \
         never marked, and a truncated row never claims `does not cache`.\n\
         - Providers that never produced a usable response (warmup or transport failure) are \"not \
         measured\", as are discovered endpoints the selection never picked.\n\
         - Static price: the blended cost of 1B tokens at the {cached_pct}% cached / \
         {input_pct}% input / {output_pct}% output usage mix, volume-independent and without \
         per-request fees; `n/a` when the endpoint advertises no pricing. It assumes the \
         advertised cache-read price applies (the full prompt price stands in where none is \
         advertised), and applies no cache-write fee and no time-of-day price override. The \
         advertised prices are used as listed — they are already net of any discount.\n\
         - Selection: the cheapest `min({MAX_SELECTED_ENDPOINTS}, max(3, ceil(0.8 × healthy)))` \
         healthy endpoints (price outliers excluded unless the pool would fall below three); an \
         explicit `--providers` allowlist keeps the 80% rule without the ceiling.\n\
         - Provider pinning: provider.order pins the endpoint tag, allow_fallbacks=false; the \
         serving provider is verified against the endpoint names (drift retried once, then \
         pin_drift).\n\
         - Retries: bounded (<=3 attempts), fixed sleeps (Retry-After clamped to 5-60 s, else 5 \
         s), never jittered — jitter would corrupt the gap measurement.",
        cached_pct = USAGE_MIX_CACHED * 100.0,
        input_pct = USAGE_MIX_INPUT * 100.0,
        output_pct = USAGE_MIX_OUTPUT * 100.0,
    );
    let _ = writeln!(out);

    if meta.aborted {
        let _ = writeln!(
            out,
            "## Aborted\n\nRun aborted: {}",
            meta.abort_reason.as_deref().unwrap_or("unknown reason")
        );
    }

    out
}

/// Static price per [`PRICE_TOKENS`] tokens at the usage mix: volume-independent, no per-request
/// fee, and no discount factor (the listed prices are already net of any advertised discount).
/// `None` when pricing is entirely absent.
#[must_use]
pub(crate) fn static_price_usd_per_1b(pricing: Option<&Pricing>) -> Option<f64> {
    let p = pricing?;
    let prompt = p.prompt.as_deref().and_then(parse_price)?;
    // The advertised cache-read price, or the full prompt price where none is advertised — the
    // same fallback `estimate_cost` applies, from the one helper that decides it.
    let (cache_read, _) = cache_read_price(p, prompt);
    let completion = p.completion.as_deref().and_then(parse_price)?;
    let price = PRICE_TOKENS * blended_price_per_token(cache_read, prompt, completion);
    // Round to micro-dollars: the blend's float noise (`15.399999999999999`) is an artifact of the
    // arithmetic, not a price, and no consumer of $/1B needs more precision than that. A price
    // below half a micro-dollar per 1B rounds to 0.0 — no real endpoint is that cheap.
    Some((price * PRICE_SCALE).round() / PRICE_SCALE)
}

/// The discovery snapshot for `providers.json`, with the key payload redacted (the key itself is
/// never present in the discovery payloads, but the file is flagged by construction) and a DERIVED
/// `privacy` section appended: the extracted per-slug policy entries, one entry per slug with the
/// strictest reading, or the explicit unavailable marker. The section is derived, so the rest of
/// the file stays a verbatim upstream record.
#[must_use]
pub(crate) fn providers_snapshot_json(
    snapshot: &DiscoverySnapshot,
    privacy: &PrivacySnapshot,
) -> serde_json::Value {
    let entries: Vec<serde_json::Value> = privacy
        .entries
        .iter()
        .map(|entry| {
            json!({
                "tag": entry.tag,
                "trains": entry.trains,
                "publishes": entry.publishes,
            })
        })
        .collect();
    json!({
        "fetched_at": snapshot.fetched_at,
        "models": snapshot.raw_models_json,
        "endpoints": snapshot.raw_endpoints_json,
        "key": snapshot.raw_key_json,
        "key_redacted": true,
        "privacy": {
            "source": privacy.source_label(),
            "entries": entries,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench_openrouter::classify::{CacheClassification, cache_hold_result, is_truncated};
    use crate::bench_openrouter::privacy::PolicyEntry as PrivacyEntry;

    /// The manifest is the artifact that leaves the machine, so the key the run was invoked with
    /// must never appear in it — neither bare nor inside an `--api-key=…` spelling.
    #[test]
    fn build_manifest_redacts_keys() {
        let args = vec![
            "mahbot".to_string(),
            "bench-openrouter".to_string(),
            "--api-key".to_string(),
            "sk-or-abc123".to_string(),
            "--api-key=sk-or-inline".to_string(),
            "--model".to_string(),
            "acme/m1".to_string(),
            "--output-dir".to_string(),
            "/tmp/out".to_string(),
        ];
        let opts = BenchOptions {
            model: "acme/m1".to_string(),
            api_key: Some("sk-or-abc123".to_string()),
            cap_usd: 2.0,
            ladder_secs: vec![0, 5],
            providers: None,
            prefix_chars: 64_000,
            output_dir: PathBuf::from("/tmp/out"),
            dry_run: false,
        };
        let _guard = crate::util::test::env_lock();
        // Ensure the env vars are unset for the test's determinism.
        unsafe {
            std::env::remove_var("MAHBOT_BENCH_MODEL");
            std::env::remove_var("MAHBOT_BENCH_API_KEY");
            std::env::remove_var("OPENROUTER_API_KEY");
        }
        let manifest = build_manifest(&args, &opts, "flag", "s", "f", 1.0, 0, false, None);
        let s = manifest.to_string();
        assert!(!s.contains("sk-or-"), "manifest leaked a key: {s}");
        assert!(s.contains("\"***\""), "expected redaction markers in {s}");
        assert_eq!(manifest["args"][2], "--api-key");
        assert_eq!(manifest["args"][3], "***");
        assert_eq!(manifest["args"][4], "--api-key=***");
        assert_eq!(manifest["config"]["model"], "acme/m1");
        assert_eq!(manifest["outcome"]["exit_code"], 0);
        assert_eq!(manifest["tool"], "bench-openrouter");
    }

    fn endpoint(tag: &str, pricing: &serde_json::Value) -> EndpointInfo {
        serde_json::from_value(serde_json::json!({
            "tag": tag,
            "name": tag,
            "provider_name": tag,
            "context_length": 200_000,
            "status": "0",
            "pricing": pricing.clone(),
        }))
        .expect("endpoint fixture")
    }

    fn meta() -> RunMeta {
        RunMeta {
            started_at: "2026-11-14T00:00:00.000Z".to_string(),
            finished_at: "2026-11-14T01:00:00.000Z".to_string(),
            duration_secs: 3600.0,
            model: "acme/m1".to_string(),
            key_source: "env".to_string(),
            cap_usd: 2.0,
            ladder_secs: crate::bench_openrouter::run::DEFAULT_LADDER_SECS.to_vec(),
            prefix_chars: 64_000,
            aborted: false,
            abort_reason: None,
            exit_code: 0,
            output_dir: PathBuf::from("/tmp/out"),
        }
    }

    /// `prompt 2e-6`, `completion 8e-6`, `cache_read 2e-7` per token → at 98/1/1 the blended
    /// price is 2.96e-7 per token, i.e. $296 per 1B.
    fn pricing() -> serde_json::Value {
        serde_json::json!({
            "prompt": "0.000002",
            "completion": "0.000008",
            "request": "0",
            "input_cache_read": "0.0000002",
        })
    }

    fn decisions(selected: &[bool]) -> Vec<SelectionDecision> {
        selected
            .iter()
            .map(|&selected| SelectionDecision {
                selected,
                reason: if selected {
                    None
                } else {
                    Some(crate::bench_openrouter::select::ExclusionReason::NotSelected(9.9))
                },
            })
            .collect()
    }

    /// A privacy snapshot with one `yes` slug and one `no` slug.
    fn privacy() -> PrivacySnapshot {
        PrivacySnapshot {
            source: Some("acme/m1-20260101".to_string()),
            entries: vec![
                PrivacyEntry {
                    tag: "acme-a".to_string(),
                    trains: Some(true),
                    publishes: Some(false),
                },
                PrivacyEntry {
                    tag: "acme-b".to_string(),
                    trains: Some(false),
                    publishes: Some(false),
                },
            ],
        }
    }

    /// The row contract: one object per discovered endpoint with `tag` / `selected` /
    /// `static_price_usd_per_1b` / `cache_hold` / `privacy_opt_in_required`, plus the run keys the
    /// artifacts promise. The three rows are the three production cases: measured, selected but
    /// truncated, and never selected.
    #[test]
    fn report_carries_the_artifact_contract() {
        let endpoints = vec![
            endpoint("acme-a", &pricing()),
            endpoint("acme-b", &serde_json::json!({})),
            endpoint("acme-c", &pricing()),
        ];
        let runs = vec![
            ProviderRun {
                tag: "acme-a".to_string(),
                cache_hold_bucket: "≥1h".to_string(),
                truncated: false,
            },
            ProviderRun {
                tag: "acme-b".to_string(),
                cache_hold_bucket: "≥30m (ladder truncated)".to_string(),
                truncated: true,
            },
        ];
        let report = build_report(
            &meta(),
            &endpoints,
            &runs,
            &decisions(&[true, true, false]),
            &privacy(),
            Path::new("/tmp/out/run"),
        );

        assert_eq!(
            report["run"]["usage_mix"],
            json!({"cached": 0.98, "input": 0.01, "output": 0.01})
        );
        assert_eq!(report["run"]["privacy_source"], "provider");
        assert_eq!(report["run"]["ladder_completed"], false);
        assert_eq!(report["run"]["endpoints_healthy"], 3);
        assert_eq!(report["run"]["endpoints_selected"], 2);

        let rows = report["providers"].as_array().expect("providers array");
        assert_eq!(rows.len(), 3);
        let mut keys: Vec<&String> = rows[0]
            .as_object()
            .expect("row object")
            .keys()
            .collect::<Vec<_>>();
        keys.sort();
        assert_eq!(
            keys,
            [
                "cache_hold",
                "privacy_opt_in_required",
                "selected",
                "static_price_usd_per_1b",
                "tag"
            ],
            "the row carries exactly the contracted keys"
        );
        assert_eq!(rows[0]["tag"], "acme-a");
        assert_eq!(rows[0]["selected"], true);
        assert_eq!(rows[0]["static_price_usd_per_1b"], 296.0);
        assert_eq!(rows[0]["cache_hold"], "≥1h");
        assert_eq!(rows[0]["privacy_opt_in_required"], "yes");
        // Selected but cut short: the suffix says so, and `selected` is still true.
        assert_eq!(rows[1]["selected"], true);
        assert_eq!(rows[1]["cache_hold"], "≥30m (ladder truncated)");
        assert_eq!(rows[1]["privacy_opt_in_required"], "no");
        // No pricing advertised → null, which the summary renders `n/a`.
        assert!(rows[1]["static_price_usd_per_1b"].is_null());
        // Never selected, so "not measured" — `selected` is what tells that apart, and a slug the
        // privacy source never mentioned reads `unknown`, never `no`.
        assert_eq!(rows[2]["selected"], false);
        assert_eq!(rows[2]["cache_hold"], "not measured");
        assert_eq!(rows[2]["privacy_opt_in_required"], "unknown");
    }

    /// The reported price is rounded to micro-dollars: the blend's float noise (`9.999999999999998`
    /// for a $10 endpoint) is an artifact of the arithmetic, not a price.
    #[test]
    fn static_price_is_rounded_to_micro_dollars() {
        let flat = endpoint(
            "acme-a",
            &json!({
                "prompt": "0.00000001",
                "completion": "0.00000001",
                "request": "0",
                "input_cache_read": "0.00000001",
            }),
        );
        let pricing = flat.pricing.as_ref().expect("pricing");
        let raw = PRICE_TOKENS * blended_price_per_token(1e-8, 1e-8, 1e-8);
        assert_ne!(raw, 10.0, "the fixture exercises the float noise: {raw:?}");
        assert_eq!(
            static_price_usd_per_1b(Some(pricing)),
            Some(10.0),
            "the artifact carries the price, not the noise"
        );
    }

    /// The listed prices are already net of any advertised discount: a payload that carries one
    /// must produce exactly the same figures, in the report and in the preflight estimate. The
    /// key is deliberately one nothing in the product reads — that is the handling, not a gap.
    #[test]
    fn an_advertised_discount_is_not_applied_twice() {
        // The fixture carries the key the real payload uses; nothing in the product reads it.
        let mut discounted_json = pricing();
        discounted_json["discount"] = json!("0.65");
        let plain = endpoint("acme-a", &pricing());
        let discounted = endpoint("acme-a", &discounted_json);
        assert_eq!(
            static_price_usd_per_1b(plain.pricing.as_ref()),
            static_price_usd_per_1b(discounted.pricing.as_ref())
        );
        let mut flags = Vec::new();
        assert_eq!(
            crate::bench_openrouter::select::estimate_cost(
                plain.pricing.as_ref().expect("pricing"),
                1e9,
                11,
                &mut flags
            ),
            crate::bench_openrouter::select::estimate_cost(
                discounted.pricing.as_ref().expect("pricing"),
                1e9,
                11,
                &mut flags
            )
        );
        assert_eq!(static_price_usd_per_1b(plain.pricing.as_ref()), Some(296.0));
    }

    /// `run.ladder_completed` is false exactly when a row is truncated. The rows come from the
    /// production constructors, so the flag is exercised against the same predicate the suffix
    /// comes from rather than a hand-written pair.
    #[test]
    #[expect(clippy::cast_precision_loss)] // gap seconds (u64) → f64 for the shared formatter
    fn run_ladder_completed_follows_the_rows() {
        let meta = meta();
        let gaps: Vec<f64> = meta.ladder_secs.iter().map(|&secs| secs as f64).collect();
        let held = vec![CacheClassification::Hit; gaps.len()];
        let endpoints: Vec<_> = ["acme-a", "acme-b", "acme-c", "acme-d"]
            .iter()
            .map(|tag| endpoint(tag, &pricing()))
            .collect();
        let rows = vec![
            // The whole ladder held: a real `≥1h`.
            ProviderRun {
                tag: "acme-a".to_string(),
                cache_hold_bucket: cache_hold_result(true, &held, &gaps, true),
                truncated: is_truncated(true, &held),
            },
            // Cut short with a hold observed, and cut short without one.
            ProviderRun {
                tag: "acme-b".to_string(),
                cache_hold_bucket: cache_hold_result(true, &held[..2], &gaps[..2], false),
                truncated: is_truncated(false, &held[..2]),
            },
            ProviderRun::ladder_never_ran("acme-c".to_string()),
            // A warmup failure is not truncated, however many such rows exist.
            ProviderRun::not_measured("acme-d".to_string()),
        ];
        let report = build_report(
            &meta,
            &endpoints,
            &rows,
            &decisions(&[true, true, true, true]),
            &privacy(),
            Path::new("/tmp/out/run"),
        );
        assert_eq!(report["run"]["ladder_completed"], false);
        assert_eq!(report["providers"][0]["cache_hold"], "≥1h");

        let complete = build_report(
            &meta,
            &endpoints,
            &rows[3..],
            &decisions(&[true, true, true, true]),
            &privacy(),
            Path::new("/tmp/out/run"),
        );
        assert_eq!(complete["run"]["ladder_completed"], true);
    }

    /// The summary's four columns, the counts line above the table, the price cell scale and the
    /// cells agreeing with the JSON.
    #[test]
    #[expect(clippy::cast_precision_loss)] // gap seconds (u64) → f64 for the shared formatter
    fn summary_table_matches_the_report() {
        let meta = meta();
        let endpoints = vec![
            endpoint("acme-a", &pricing()),
            endpoint("acme-b", &serde_json::json!({})),
        ];
        let runs = vec![ProviderRun {
            tag: "acme-a".to_string(),
            cache_hold_bucket: "≥1h".to_string(),
            truncated: false,
        }];
        let report = build_report(
            &meta,
            &endpoints,
            &runs,
            &decisions(&[true, false]),
            &privacy(),
            Path::new("/tmp/out/run"),
        );
        let summary = build_summary_md(&meta, &report);

        assert!(
            summary
                .contains("| Provider (tag) | Static price ($/1B) | Cache hold | Privacy opt-in |"),
            "the header is the four-column contract: {summary}"
        );
        assert!(
            summary.contains("| acme-a | 296.00 | ≥1h | yes |"),
            "{summary}"
        );
        assert!(
            summary.contains("| acme-b | n/a | not measured | no |"),
            "{summary}"
        );
        assert!(
            summary.contains("Healthy endpoints: 2; selected for measurement: 1."),
            "the counts line is above the table and matches the run keys: {summary}"
        );
        assert_eq!(report["run"]["endpoints_healthy"], 2);
        assert_eq!(report["run"]["endpoints_selected"], 1);
        // The ladder row goes through the same formatter the buckets use, so the hour rung reads
        // `1h` in both places.
        let ladder: Vec<String> = meta
            .ladder_secs
            .iter()
            .map(|&secs| format_duration(secs as f64))
            .collect();
        assert!(
            summary.contains(&format!("| Ladder | {} |", ladder.join(", "))),
            "{summary}"
        );
    }

    /// An unavailable privacy source is one visible line above the table, never a silent run.
    #[test]
    fn summary_names_an_unavailable_privacy_source() {
        let report = build_report(
            &meta(),
            &[endpoint("acme-a", &pricing())],
            &[],
            &decisions(&[true]),
            &PrivacySnapshot::unavailable(),
            Path::new("/tmp/out/run"),
        );
        assert_eq!(report["run"]["privacy_source"], "unavailable");
        assert_eq!(report["providers"][0]["privacy_opt_in_required"], "unknown");
        let summary = build_summary_md(&meta(), &report);
        let note = summary
            .find("Privacy source unavailable")
            .expect("the summary says the source is unavailable");
        let table = summary
            .find("| Provider (tag)")
            .expect("the summary carries the endpoint table");
        assert!(note < table, "the note is above the table: {summary}");
    }

    /// `providers.json` keeps the verbatim discovery payloads and appends the DERIVED privacy
    /// section, with the explicit unavailable marker when the source failed.
    #[test]
    fn providers_snapshot_carries_the_derived_privacy_section() {
        let snapshot = crate::bench_openrouter::discovery::DiscoverySnapshot {
            fetched_at: "2026-11-14T00:00:00.000Z".to_string(),
            catalog: Vec::new(),
            endpoints: serde_json::from_value(serde_json::json!({"data": {"endpoints": []}}))
                .expect("endpoints fixture"),
            key: serde_json::from_value(serde_json::json!({})).expect("key fixture"),
            raw_models_json: serde_json::json!({"data": []}),
            raw_endpoints_json: serde_json::json!({"data": {}}),
            raw_key_json: serde_json::json!({"data": {}}),
        };
        let resolved = providers_snapshot_json(&snapshot, &privacy());
        assert_eq!(resolved["privacy"]["source"], "acme/m1-20260101");
        assert_eq!(
            resolved["privacy"]["entries"],
            json!([
                {"tag": "acme-a", "trains": true, "publishes": false},
                {"tag": "acme-b", "trains": false, "publishes": false},
            ])
        );
        // The upstream record is untouched: the payloads are still verbatim.
        assert_eq!(resolved["models"], snapshot.raw_models_json);
        assert_eq!(resolved["endpoints"], snapshot.raw_endpoints_json);
        assert_eq!(resolved["key_redacted"], true);

        let unavailable = providers_snapshot_json(&snapshot, &PrivacySnapshot::unavailable());
        assert_eq!(
            unavailable["privacy"],
            json!({"source": "unavailable", "entries": []}),
            "the explicit marker, never an absent key"
        );
    }
}
