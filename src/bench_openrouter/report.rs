//! Report artifacts for `mahbot bench-openrouter` full runs.
//!
//! The full-run executor writes four artifacts under
//! `<output-dir>/bench-openrouter/<run-ts>/`:
//!
//! - `report.json` — the complete structured run report: run meta, one
//!   three-column entry per provider (`tag` / `static_price_usd_per_m` /
//!   `cache_hold`), and the artifact paths.
//! - `summary.md` — the human-readable markdown summary (meta table, the
//!   per-provider price + cache-hold table, a short methodology, and an abort
//!   note when applicable).
//! - `providers.json` — the verbatim discovery snapshot (raw models /
//!   endpoints / key payloads with the key redacted).
//! - `manifest.json` — the run manifest: scrubbed argv + env entries +
//!   config + outcome (exit code, abort state). The API key value is ALWAYS
//!   scrubbed (see [`build_manifest`]).
//!
//! By default `<output-dir>` is `~/.mahbot/benchmarks/<model-slug>/` — one
//! folder per model, derived from the model id — so parallel runs of different
//! models write separate folders and take separate output locks; `--output-dir`
//! overrides it exactly.
//!
//! Every artifact is mirrored into `<output-dir>/bench-openrouter/latest/`
//! so a consumer can always read the most recent run of that model from a
//! stable path.
//!
//! All `build_*` functions are pure (no I/O) and unit-tested against
//! synthetic inputs; [`write_artifacts`] and [`acquire_run_lock`] are the only
//! I/O boundaries.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::json;

use crate::bench_openrouter::discovery::{DiscoverySnapshot, EndpointInfo, Pricing};
use crate::bench_openrouter::run::ProviderRun;
use crate::bench_openrouter::{BenchOptions, discovery::parse_price};

// ── Paths ──────────────────────────────────────────────────────────

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

/// Acquire the bench's own run lock: `<output_dir>/bench-openrouter.lock`.
///
/// This is the bench's OWN lock — it never touches the daemon's
/// `mahbot.lock`. Creates `output_dir` first; returns the locked [`File`],
/// which the caller keeps alive for the run duration (the kernel releases the
/// lock when the file is closed / the process exits). Bails when another
/// bench run holds the lock.
///
/// # Errors
///
/// Fails when the directory cannot be created, the file cannot be opened, or
/// `flock` itself errors (as opposed to reporting "already locked").
pub(crate) fn acquire_run_lock(output_dir: &Path) -> anyhow::Result<std::fs::File> {
    std::fs::create_dir_all(output_dir)?;
    let path = output_dir.join("bench-openrouter.lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&path)?;
    if !crate::lock_utils::try_flock(&file)? {
        anyhow::bail!(
            "another bench-openrouter run is in progress (lock: {})",
            path.display()
        );
    }
    Ok(file)
}

// ── Artifact writing ───────────────────────────────────────────────

/// Write all four artifacts into `run_dir` and mirror them into `latest_dir`.
///
/// `create_dir_all` both directories first. The three JSON artifacts are
/// written pretty-printed; `summary_md` is written verbatim. The mirror is
/// best-effort per file — any copy error propagates (the run is still
/// reported as failed, but the run_dir artifacts may be intact).
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

// ── Manifest ───────────────────────────────────────────────────────

/// Build the run manifest. `args` is the raw argv with any API-key material
/// scrubbed: the `--api-key` value (both `--api-key VALUE` and
/// `--api-key=VALUE` forms) becomes `"***"`, and any arg containing the
/// substring `sk-or-` is replaced wholesale with `"***"`. The `env` object
/// records ONLY the three key entries (each present-or-absent), with the key
/// values scrubbed.
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

/// The scrubbed stand-in for an API key value: `"***"` for anything that looks
/// like a key (non-empty), the original when empty (a `--api-key=` with an
/// empty value carries no secret).
#[must_use]
fn scrub_value(v: &str) -> String {
    if v.is_empty() {
        v.to_string()
    } else {
        "***".to_string()
    }
}

// ── Report ─────────────────────────────────────────────────────────

/// Run metadata shared by [`build_report`] and [`build_summary_md`].
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

/// Build the complete structured report (pure — no I/O).
///
/// Every discovered endpoint gets a providers-array entry; the cache-hold is
/// the run's `cache_hold_bucket` when a run exists for that tag, else
/// `"not measured"`.
#[must_use]
pub(crate) fn build_report(
    meta: &RunMeta,
    endpoints: &[EndpointInfo],
    providers: &[ProviderRun],
    output_dir: &Path,
) -> serde_json::Value {
    let by_tag: HashMap<&str, &ProviderRun> = providers
        .iter()
        .map(|run| (run.tag.as_str(), run))
        .collect();

    let provider_json: Vec<serde_json::Value> = endpoints
        .iter()
        .map(|ep| {
            let cache_hold = by_tag
                .get(ep.tag.as_str())
                .map_or("not measured", |run| run.cache_hold_bucket.as_str());
            json!({
                "tag": ep.tag,
                "static_price_usd_per_m": static_price_usd_per_m(ep.pricing.as_ref()),
                "cache_hold": cache_hold,
            })
        })
        .collect();

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

// ── Summary markdown ───────────────────────────────────────────────

/// Ladder gap rendered for the summary: raw seconds below 30 minutes, whole
/// minutes above (the default ladder reads `0s, 5s, 30s, 120s, 300s, 600s, 30m`).
#[must_use]
fn format_ladder_gap(secs: u64) -> String {
    if secs < 1800 {
        format!("{secs}s")
    } else {
        format!("{}m", secs / 60)
    }
}

/// Build the human-readable markdown summary (pure — no I/O). Sections:
/// header + meta table, the per-provider price + cache-hold table, a short
/// methodology, and an abort note when the run was aborted.
#[must_use]
pub(crate) fn build_summary_md(meta: &RunMeta, report: &serde_json::Value) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();

    // 1. Header + run meta table.
    let _ = writeln!(out, "# OpenRouter provider benchmark — {}\n", meta.model);
    let _ = writeln!(out, "| | |\n|---|---|");
    let _ = writeln!(out, "| Started | {} |", meta.started_at);
    let _ = writeln!(out, "| Duration | {:.1} s |", meta.duration_secs);
    let _ = writeln!(out, "| Key source | {} |", meta.key_source);
    let _ = writeln!(out, "| Spend cap | ${:.2} |", meta.cap_usd);
    let ladder: Vec<String> = meta
        .ladder_secs
        .iter()
        .map(|&s| format_ladder_gap(s))
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

    // 2. Per-provider price + cache-hold table (read from the report JSON).
    let _ = writeln!(out, "## Providers\n");
    let _ = writeln!(out, "| Provider (tag) | Static price ($/1M) | Cache hold |");
    let _ = writeln!(out, "|---|---|---|");
    let providers = report["providers"].as_array().cloned().unwrap_or_default();
    for p in providers {
        let tag = p
            .get("tag")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?");
        let price = p
            .get("static_price_usd_per_m")
            .and_then(serde_json::Value::as_f64)
            .map_or_else(|| "n/a".to_string(), |v| format!("{v:.4}"));
        let hold = p
            .get("cache_hold")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("not measured");
        let _ = writeln!(out, "| {tag} | {price} | {hold} |");
    }
    let _ = writeln!(out);

    // 3. Methodology.
    let _ = writeln!(
        out,
        "## Methodology\n\n\
         - TTL ladder: per provider, 2 byte-identical warmup requests + one ladder round per gap \
         (a 7-gap ladder = 8 ladder rounds); the conversation grows one verified tool frame per round.\n\
         - Cache-hold rule: a provider is \"does not cache\" when no warmup-2/ladder round reported \
         cached tokens; otherwise the bucket is the TTL hold from the ladder (interval notation like \
         `(5s, 30s]`, `≤5s`, `≥30m`, `immediate drop`).\n\
         - Providers that did not run (failed to start, skipped the ladder, or were aborted before \
         the ladder) are \"not measured\".\n\
         - Static price: the blended cost per 1M tokens at the validated mix (0.977 cached / 0.014 \
         input / 0.009 output), volume-independent, no per-request fee; `n/a` when the provider \
         advertises no pricing.\n\
         - Provider pinning: provider.order pins the endpoint tag, allow_fallbacks=false; the serving \
         provider is verified against the endpoint names (drift retried once, then pin_drift).\n\
         - Retries: bounded (<=3 attempts), fixed sleeps (Retry-After clamped to 5-60 s, else 5 s), \
         never jittered — jitter would corrupt the gap measurement."
    );
    let _ = writeln!(out);

    // 4. Abort note.
    if meta.aborted {
        let _ = writeln!(
            out,
            "## Aborted\n\nRun aborted: {}",
            meta.abort_reason.as_deref().unwrap_or("unknown reason")
        );
    }

    out
}

/// Static price per 1M tokens at the validated mix (0.977 cached / 0.014
/// input / 0.009 output), volume-independent, no per-request fee. `None` when
/// pricing is entirely absent.
#[must_use]
pub(crate) fn static_price_usd_per_m(pricing: Option<&Pricing>) -> Option<f64> {
    let p = pricing?;
    let cache_read = p
        .input_cache_read
        .as_deref()
        .and_then(parse_price)
        .or_else(|| p.prompt.as_deref().and_then(parse_price))?;
    let prompt = p.prompt.as_deref().and_then(parse_price)?;
    let completion = p.completion.as_deref().and_then(parse_price)?;
    Some(session_cost(
        1e6, cache_read, prompt, completion, 0.977, 0.014, 0.009,
    ))
}

// ── Cost math ──────────────────────────────────────────────────────

/// Session cost in USD for `total_tokens` tokens at the given per-token mix:
/// `total_tokens × (mix_cached×cache_read + mix_input×prompt + mix_output×completion)`.
#[must_use]
pub(crate) fn session_cost(
    total_tokens: f64,
    cache_read: f64,
    prompt: f64,
    completion: f64,
    mix_cached: f64,
    mix_input: f64,
    mix_output: f64,
) -> f64 {
    total_tokens * (mix_cached * cache_read + mix_input * prompt + mix_output * completion)
}

// ── Providers snapshot ─────────────────────────────────────────────

/// The verbatim discovery snapshot for `providers.json`, with the key
/// payload redacted (the raw key JSON may echo the key label / usage — the
/// API key itself is never present in the discovery payloads, but the file is
/// flagged `key_redacted: true` by construction).
#[must_use]
pub(crate) fn providers_snapshot_json(snapshot: &DiscoverySnapshot) -> serde_json::Value {
    json!({
        "fetched_at": snapshot.fetched_at,
        "models": snapshot.raw_models_json,
        "endpoints": snapshot.raw_endpoints_json,
        "key": snapshot.raw_key_json,
        "key_redacted": true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench_openrouter::discovery::{EndpointInfo, Pricing};

    fn endpoint(tag: &str, cache_read: &str, prompt: &str, completion: &str) -> EndpointInfo {
        EndpointInfo {
            tag: tag.to_string(),
            name: format!("Name {tag}"),
            provider_name: format!("Provider {tag}"),
            context_length: Some(200_000),
            quantization: Some("fp8".to_string()),
            status: Some("0".to_string()),
            supports_implicit_caching: Some(true),
            pricing: Some(Pricing {
                prompt: Some(prompt.to_string()),
                completion: Some(completion.to_string()),
                request: Some("0".to_string()),
                input_cache_read: Some(cache_read.to_string()),
            }),
        }
    }

    fn provider_run(tag: &str, cache_hold_bucket: &str) -> ProviderRun {
        ProviderRun {
            tag: tag.to_string(),
            cache_hold_bucket: cache_hold_bucket.to_string(),
        }
    }

    fn meta() -> RunMeta {
        RunMeta {
            started_at: "2026-08-20T00:00:00.000Z".to_string(),
            finished_at: "2026-08-20T01:00:00.000Z".to_string(),
            duration_secs: 3600.0,
            model: "acme/model-1".to_string(),
            key_source: "env".to_string(),
            cap_usd: 2.0,
            ladder_secs: vec![0, 5, 30],
            prefix_chars: 64_000,
            aborted: false,
            abort_reason: None,
            exit_code: 0,
            output_dir: PathBuf::from("/tmp/bench"),
        }
    }

    #[test]
    fn run_dir_name_format() {
        let dt = chrono::DateTime::parse_from_rfc3339("2026-08-20T03:12:42Z")
            .expect("fixture")
            .with_timezone(&chrono::Utc);
        assert_eq!(run_dir_name(&dt), "20260820-031242");
    }

    #[test]
    fn build_report_shape_and_totals() {
        // Endpoint A has a run; endpoint B does not (→ "not measured").
        let endpoints = vec![
            endpoint("acme-a/fp8", "0.0000001", "0.000001", "0.000003"),
            endpoint("acme-b/fp8", "0.0000002", "0.000002", "0.000004"),
        ];
        let providers = vec![provider_run("acme-a/fp8", "≥30m")];
        let report = build_report(&meta(), &endpoints, &providers, Path::new("/tmp/bench"));

        // Exactly the specified top-level keys.
        let expected_keys = ["run", "providers", "artifacts"];
        let object = report.as_object().expect("report is an object");
        assert_eq!(
            object.len(),
            expected_keys.len(),
            "unexpected top-level keys"
        );
        for k in expected_keys {
            assert!(object.contains_key(k), "missing top-level key {k}");
        }

        // Every discovered endpoint has an entry; each entry has EXACTLY the
        // three spec'd keys.
        let arr = report["providers"].as_array().expect("providers array");
        assert_eq!(arr.len(), 2);
        for entry in arr {
            let entry = entry.as_object().expect("provider entry");
            assert_eq!(entry.len(), 3, "provider entries must have 3 keys");
            for k in ["tag", "static_price_usd_per_m", "cache_hold"] {
                assert!(entry.contains_key(k), "missing provider key {k}");
            }
        }
        assert_eq!(arr[0]["tag"], "acme-a/fp8");
        assert_eq!(arr[0]["cache_hold"], "≥30m");
        // static price: cache_read $0.1/M, prompt $1/M, completion $3/M.
        let expected_price = 0.977 * 0.1 + 0.014 * 1.0 + 0.009 * 3.0;
        assert!((arr[0]["static_price_usd_per_m"].as_f64().unwrap() - expected_price).abs() < 1e-9);
        assert!((arr[0]["static_price_usd_per_m"].as_f64().unwrap() - 0.1387).abs() < 1e-9);
        // The endpoint without a run is "not measured".
        assert_eq!(arr[1]["tag"], "acme-b/fp8");
        assert_eq!(arr[1]["cache_hold"], "not measured");
        assert!(
            (arr[1]["static_price_usd_per_m"].as_f64().unwrap()
                - (0.977 * 0.2 + 0.014 * 2.0 + 0.009 * 4.0))
                .abs()
                < 1e-9
        );

        // Run-meta passthrough.
        assert_eq!(report["run"]["model"], "acme/model-1");
        assert_eq!(report["run"]["key_source"], "env");
        assert_eq!(report["run"]["cap_usd"], 2.0);
        assert_eq!(report["run"]["ladder_secs"], json!([0, 5, 30]));
        assert_eq!(report["run"]["exit_code"], 0);
        assert_eq!(report["artifacts"]["run_dir"], "/tmp/bench");

        // A provider with no pricing renders a null static price.
        let mut no_pricing = endpoint("acme-c/fp8", "0", "0", "0");
        no_pricing.pricing = None;
        let report2 = build_report(&meta(), &[no_pricing], &[], Path::new("/tmp/bench"));
        let arr2 = report2["providers"].as_array().expect("providers array");
        assert_eq!(arr2.len(), 1);
        assert!(arr2[0]["static_price_usd_per_m"].is_null());
        assert_eq!(arr2[0]["cache_hold"], "not measured");
    }

    #[test]
    fn session_cost_validated_mix_vs_all_miss() {
        // Arbitrary per-token prices: cache_read 0.1, prompt 1.0, completion 3.0;
        // 1M tokens.
        let validated = session_cost(1e6, 0.1, 1.0, 3.0, 0.977, 0.014, 0.009);
        let expected_validated = 1e6 * (0.977 * 0.1 + 0.014 * 1.0 + 0.009 * 3.0);
        assert!((validated - expected_validated).abs() < 1e-9);
        // All input at the prompt price → 1M tokens × $1/token.
        let all_miss = session_cost(1e6, 1.0, 1.0, 3.0, 1.0, 0.0, 0.0);
        assert!((all_miss - 1e6).abs() < 1e-9);
    }

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
        assert_eq!(manifest["args"][4], "--api-key=***");
        assert_eq!(manifest["config"]["model"], "acme/m1");
        assert_eq!(manifest["outcome"]["exit_code"], 0);
        assert_eq!(manifest["tool"], "bench-openrouter");
    }
}
