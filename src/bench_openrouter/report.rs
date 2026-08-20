//! Report artifacts for `mahbot bench-openrouter` full runs.
//!
//! The full-run executor writes four artifacts under
//! `<output-dir>/bench-openrouter/<run-ts>/`:
//!
//! - `report.json` — the complete structured run report: run meta, the
//!   selection JSON, one [`provider_run_json`] entry per provider (flat, nulls
//!   for absent values), a summary block, and the artifact paths.
//! - `summary.md` — the human-readable markdown summary (comparison table,
//!   cache-hold curves, reliability taxonomy, cost analysis, quantization,
//!   methodology footnote).
//! - `providers.json` — the verbatim discovery snapshot (raw models /
//!   endpoints / key payloads with the key redacted).
//! - `manifest.json` — the run manifest: scrubbed argv + env entries +
//!   config + outcome (exit code, abort state). The API key value is ALWAYS
//!   scrubbed (see [`build_manifest`]).
//!
//! Every artifact is mirrored into `<output-dir>/bench-openrouter/latest/`
//! so a consumer can always read the most recent run from a stable path.
//!
//! All `build_*` functions are pure (no I/O) and unit-tested against
//! synthetic inputs; [`write_artifacts`] and [`acquire_run_lock`] are the only
//! I/O boundaries.

use std::path::{Path, PathBuf};

use serde_json::json;

use crate::bench_openrouter::discovery::DiscoverySnapshot;
use crate::bench_openrouter::run::{ProviderRun, RoundRecord};
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
#[allow(clippy::too_many_arguments)]
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
/// `selection_reasons` is parallel to `providers`: `(selected, selection_reason)`
/// per provider, in the same order as the providers slice.
#[must_use]
pub(crate) fn build_report(
    meta: &RunMeta,
    selection: &serde_json::Value,
    providers: &[ProviderRun],
    selection_reasons: &[(bool, Option<String>)],
    output_dir: &Path,
) -> serde_json::Value {
    let provider_json: Vec<serde_json::Value> = providers
        .iter()
        .zip(selection_reasons)
        .map(|(run, (selected, reason))| provider_run_json(run, *selected, reason.as_deref()))
        .collect();

    let completed_count = providers
        .iter()
        .filter(|p| !p.incomplete && !p.aborted)
        .count();
    let total_billed_usd: f64 = providers.iter().map(|p| p.billed_usd).sum();
    let total_estimated_usd: f64 = providers.iter().map(|p| p.estimated_usd).sum();
    let cache_supported_count = providers.iter().filter(|p| p.cache_supported).count();

    let mut hold_dist: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for p in providers {
        *hold_dist.entry(p.cache_hold_bucket.as_str()).or_insert(0) += 1;
    }
    let hold_distribution: serde_json::Map<String, serde_json::Value> = hold_dist
        .into_iter()
        .map(|(bucket, count)| (bucket.to_string(), json!(count)))
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
        "selection": selection.clone(),
        "providers": provider_json,
        "summary": {
            "selected_count": providers.len(),
            "completed_count": completed_count,
            "total_billed_usd": total_billed_usd,
            "total_estimated_usd": total_estimated_usd,
            "cache_supported_count": cache_supported_count,
            "hold_distribution": hold_distribution,
        },
        "artifacts": {
            "run_dir": output_dir.display().to_string(),
            "report": "report.json",
            "summary": "summary.md",
            "providers": "providers.json",
            "manifest": "manifest.json",
        },
    })
}

/// One provider's flat report object (nulls for absent values).
#[must_use]
// u64→f64 is exact for token counts far below 2^53; blended cost math needs floats.
#[allow(clippy::cast_precision_loss)]
fn provider_run_json(
    run: &ProviderRun,
    selected: bool,
    selection_reason: Option<&str>,
) -> serde_json::Value {
    let ep = &run.endpoint;

    let total = run.total_tokens_reported;
    let blended = if total > 0 {
        Some(run.billed_usd / total as f64 * 1e6)
    } else {
        None
    };
    let per_m_at_validated_mix = price_triple(run).map(|(cache_read, prompt, completion)| {
        session_cost(1e6, cache_read, prompt, completion, 0.977, 0.014, 0.009)
    });
    let per_m_all_miss = price_triple(run).map(|(_, prompt, completion)| {
        session_cost(1e6, prompt, prompt, completion, 1.0, 0.0, 0.0)
    });

    let warmup: Vec<serde_json::Value> = run.warmup.iter().map(round_json).collect();
    let ladder: Vec<serde_json::Value> = run.ladder.iter().map(round_json).collect();

    json!({
        "tag": run.tag,
        "name": ep.name,
        "provider_name": ep.provider_name,
        "quantization": ep.quantization,
        "status": ep.status,
        "supports_implicit_caching": ep.supports_implicit_caching,
        "context_length": ep.context_length,
        "selected": selected,
        "selection_reason": selection_reason,
        "cache_supported": run.cache_supported,
        "contamination_warning": run.contamination_warning,
        "warmup": warmup,
        "ladder": ladder,
        "cache_hold_bucket": run.cache_hold_bucket,
        "cache_hold_curve": run.cache_hold_curve,
        "cost": {
            "estimated_usd": run.estimated_usd,
            "billed_usd": run.billed_usd,
            "delta_usd": run.billed_usd - run.estimated_usd,
            "blended_usd_per_m": blended,
            "per_m_at_validated_mix": per_m_at_validated_mix,
            "per_m_all_miss": per_m_all_miss,
        },
        "latency": run.latency,
        "token_usage": run.token_usage,
        "reliability": run.reliability,
        "incomplete": run.incomplete,
        "incomplete_reason": run.incomplete_reason,
        "aborted": run.aborted,
        "abort_reason": run.abort_reason,
    })
}

/// One round's flat JSON (Option fields render as null automatically).
#[must_use]
fn round_json(rec: &RoundRecord) -> serde_json::Value {
    json!({
        "kind": rec.kind,
        "rung": rec.rung,
        "nominal_gap_secs": rec.nominal_gap_secs,
        "measured_gap_ms": rec.measured_gap_ms,
        "t_send": rec.t_send,
        "t_headers": rec.t_headers,
        "t_body": rec.t_body,
        "prompt_hash": rec.prompt_hash,
        "http_status": rec.http_status,
        "finish_reason": rec.finish_reason,
        "serving_provider": rec.serving_provider,
        "pin_verified": rec.pin_verified,
        "generation_id": rec.generation_id,
        "response_cache_hit": rec.response_cache_hit,
        "usage": {
            "prompt_tokens": rec.usage.prompt_tokens,
            "completion_tokens": rec.usage.completion_tokens,
            "total_tokens": rec.usage.total_tokens,
            "cached_tokens": rec.usage.cached_tokens,
            "cache_write_tokens": rec.usage.cache_write_tokens,
            "miss_tokens": rec.usage.miss_tokens,
            "reasoning_tokens": rec.usage.reasoning_tokens,
            "cost": rec.usage.cost,
        },
        "cache_classification": rec.cache_classification,
        "expected_cached_tokens": rec.expected_cached_tokens,
        "error_class": rec.error_class,
        "retries": rec.retries,
        "cache_status_header": rec.cache_status_header,
    })
}

// ── Summary markdown ───────────────────────────────────────────────

/// Per-provider prices for the cost tables: cache-read (falling back to the
/// full prompt price when the provider omits cache pricing), prompt and
/// completion, all in USD per token. `None` when pricing is entirely absent.
#[must_use]
fn price_triple(run: &ProviderRun) -> Option<(f64, f64, f64)> {
    let p = run.endpoint.pricing.as_ref()?;
    let cache_read = p
        .input_cache_read
        .as_deref()
        .and_then(parse_price)
        .or_else(|| p.prompt.as_deref().and_then(parse_price))?;
    let prompt = p.prompt.as_deref().and_then(parse_price)?;
    let completion = p.completion.as_deref().and_then(parse_price)?;
    Some((cache_read, prompt, completion))
}

/// Median of the round-trip (`full_ms`) latencies, or `None` when no round
/// succeeded.
#[must_use]
fn latency_p50_ms(run: &ProviderRun) -> Option<u64> {
    let mut ms: Vec<u64> = run
        .latency
        .get("full_ms")
        .and_then(serde_json::Value::as_array)
        .map(|arr| arr.iter().filter_map(serde_json::Value::as_u64).collect())
        .unwrap_or_default();
    if ms.is_empty() {
        return None;
    }
    ms.sort_unstable();
    let mid = ms.len() / 2;
    Some(if ms.len().is_multiple_of(2) {
        // Overflow-safe midpoint on the sorted slice (b >= a).
        ms[mid - 1] + (ms[mid] - ms[mid - 1]) / 2
    } else {
        ms[mid]
    })
}

/// Measured cache-hit ratio `cached / (cached + miss)` from the run's token
/// usage (0 when no prompt tokens were reported).
#[must_use]
// u64→f64 is exact for token counts far below 2^53; ratio math needs floats.
#[allow(clippy::cast_precision_loss)]
fn measured_hit_ratio(run: &ProviderRun) -> f64 {
    let cached = run
        .token_usage
        .get("cached")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let miss = run
        .token_usage
        .get("miss")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let denom = cached + miss;
    if denom == 0 {
        0.0
    } else {
        cached as f64 / denom as f64
    }
}

/// Build the human-readable markdown summary (pure — no I/O). Sections:
/// header + meta table, provider comparison, cache-hold curves, reliability
/// taxonomy, cost analysis (+ extrapolation), quantization, methodology
/// footnote, and an abort note when the run was aborted.
#[must_use]
// One fixed-format document builder; splitting it would obscure the section
// ordering. u64→f64 is exact for token counts far below 2^53.
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
pub(crate) fn build_summary_md(
    meta: &RunMeta,
    report: &serde_json::Value,
    providers: &[ProviderRun],
) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();

    // 1. Header + run meta table.
    let _ = writeln!(out, "# OpenRouter provider benchmark — {}\n", meta.model);
    let _ = writeln!(out, "| | |\n|---|---|");
    let _ = writeln!(out, "| Started | {} |", meta.started_at);
    let _ = writeln!(out, "| Duration | {:.1} s |", meta.duration_secs);
    let _ = writeln!(out, "| Key source | {} |", meta.key_source);
    let _ = writeln!(out, "| Spend cap | ${:.2} |", meta.cap_usd);
    if meta.aborted {
        let _ = writeln!(
            out,
            "| Aborted | {} |",
            meta.abort_reason.as_deref().unwrap_or("yes")
        );
    }
    let _ = writeln!(out, "| Output dir | {} |", meta.output_dir.display());
    let _ = writeln!(out);

    // 2. Provider comparison.
    let _ = writeln!(out, "## Provider comparison\n");
    let _ = writeln!(
        out,
        "| Provider (tag) | Quant | Status | Cache hold | Latency p50 (ms) | Est $ | Billed $ | Δ $ | Errors | Incomplete |"
    );
    let _ = writeln!(out, "|---|---|---|---|---|---|---|---|---|---|");
    for run in providers {
        let quant = run.endpoint.quantization.as_deref().unwrap_or("");
        let status = run.endpoint.status.as_deref().unwrap_or("");
        let p50 = latency_p50_ms(run).map_or_else(|| "n/a".to_string(), |ms| ms.to_string());
        let errors = run
            .reliability
            .get("rounds_failed")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {:.4} | {:.4} | {:.4} | {} | {} |",
            run.tag,
            quant,
            status,
            run.cache_hold_bucket,
            p50,
            run.estimated_usd,
            run.billed_usd,
            run.billed_usd - run.estimated_usd,
            errors,
            if run.incomplete { "yes" } else { "" },
        );
    }
    let _ = writeln!(out);

    // 3. Cache-hold curves.
    let _ = writeln!(out, "## Cache-hold curves\n");
    for run in providers {
        let curve: Vec<String> = run
            .cache_hold_curve
            .iter()
            .map(|c| {
                let gap = c
                    .get("gap_secs")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0);
                let cls = c
                    .get("classification")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("invalid");
                format!("{gap:.0}s={cls}")
            })
            .collect();
        let _ = writeln!(out, "`{}`: {}", run.tag, curve.join(" "));
    }
    let _ = writeln!(out);

    // 4. Reliability taxonomy (aggregated error classes, sorted desc).
    let _ = writeln!(
        out,
        "## Reliability taxonomy\n\n| error class | count |\n|---|---|"
    );
    let mut counts: std::collections::BTreeMap<&str, u64> = std::collections::BTreeMap::new();
    for run in providers {
        if let Some(errors) = run
            .reliability
            .get("errors")
            .and_then(serde_json::Value::as_array)
        {
            for e in errors {
                if let (Some(class), Some(count)) = (
                    e.get("class").and_then(serde_json::Value::as_str),
                    e.get("count").and_then(serde_json::Value::as_u64),
                ) {
                    *counts.entry(class).or_insert(0) += count;
                }
            }
        }
    }
    let mut sorted: Vec<(&str, u64)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    for (class, count) in sorted {
        let _ = writeln!(out, "| {class} | {count} |");
    }
    let _ = writeln!(out);

    // 5. Cost analysis.
    let _ = writeln!(out, "## Cost analysis\n");
    let _ = writeln!(
        out,
        "| tag | billed $ | est $ | blended $/M (measured) | $/M @ validated mix | $/M all-miss |"
    );
    let _ = writeln!(out, "|---|---|---|---|---|---|");
    for run in providers {
        let blended = if run.total_tokens_reported > 0 {
            format!(
                "{:.4}",
                run.billed_usd / run.total_tokens_reported as f64 * 1e6
            )
        } else {
            "n/a".to_string()
        };
        let (validated, all_miss) = match price_triple(run) {
            Some((cache_read, prompt, completion)) => (
                format!(
                    "{:.4}",
                    session_cost(1e6, cache_read, prompt, completion, 0.977, 0.014, 0.009)
                ),
                format!(
                    "{:.4}",
                    session_cost(1e6, prompt, prompt, completion, 1.0, 0.0, 0.0)
                ),
            ),
            None => ("n/a".to_string(), "n/a".to_string()),
        };
        let _ = writeln!(
            out,
            "| {} | {:.4} | {:.4} | {} | {} | {} |",
            run.tag, run.billed_usd, run.estimated_usd, blended, validated, all_miss,
        );
    }
    let total_billed = report["summary"]["total_billed_usd"]
        .as_f64()
        .unwrap_or(0.0);
    let total_est = report["summary"]["total_estimated_usd"]
        .as_f64()
        .unwrap_or(0.0);
    let _ = writeln!(
        out,
        "| **Total** | {total_billed:.4} | {total_est:.4} | | | |"
    );
    let _ = writeln!(out);

    // Extrapolation: 100K / 1M / 10M tokens at the validated mix, plus a
    // second row at the provider's measured hit ratio when caching is
    // supported (validated mix when the measured ratio is 0).
    let _ = writeln!(
        out,
        "| tag | 100K tokens | 1M tokens | 10M tokens |\n|---|---|---|---|"
    );
    for run in providers {
        let Some((cache_read, prompt, completion)) = price_triple(run) else {
            continue;
        };
        let validated_row =
            |n: f64| session_cost(n, cache_read, prompt, completion, 0.977, 0.014, 0.009);
        let _ = writeln!(
            out,
            "| {} (validated mix) | {:.4} | {:.4} | {:.4} |",
            run.tag,
            validated_row(1e5),
            validated_row(1e6),
            validated_row(1e7),
        );
        if run.cache_supported {
            let ratio = measured_hit_ratio(run);
            let (mix_cached, mix_input, mix_output) = if ratio == 0.0 {
                (0.977, 0.014, 0.009)
            } else {
                (ratio, 1.0 - ratio, 0.0)
            };
            let measured_row = |n: f64| {
                session_cost(
                    n, cache_read, prompt, completion, mix_cached, mix_input, mix_output,
                )
            };
            let _ = writeln!(
                out,
                "| {} (measured hit {:.3}{}) | {:.4} | {:.4} | {:.4} |",
                run.tag,
                ratio,
                if ratio == 0.0 { "; validated mix" } else { "" },
                measured_row(1e5),
                measured_row(1e6),
                measured_row(1e7),
            );
        }
    }
    let _ = writeln!(out);

    // 6. Quantization.
    let _ = writeln!(out, "## Quantization\n");
    for run in providers {
        let q = run.endpoint.quantization.as_deref().unwrap_or("n/a");
        let _ = writeln!(out, "{}: {}", run.tag, q);
    }
    let _ = writeln!(out);

    // 7. Methodology footnote (fixed text).
    let _ = writeln!(
        out,
        "## Methodology\n\n\
         - TTL ladder: per provider, 2 byte-identical warmup requests + one ladder round per gap \
         (a 7-gap ladder = 8 ladder rounds); the conversation grows one verified tool frame per round.\n\
         - Classification: a round is a hit at >=90% of the W2-anchored expected-cached delta, a drop \
         below 10%, partial in between.\n\
         - Expected-cached delta: base cache (W2) plus prompt growth since W2, saturating.\n\
         - Per-run nonce: a random nonce distinguishes runs in API logs while requests stay \
         byte-identical within a run.\n\
         - Provider pinning: provider.order pins the endpoint tag, allow_fallbacks=false; the serving \
         provider is verified against the endpoint names (drift retried once, then pin_drift).\n\
         - Retries: bounded (<=3 attempts), fixed sleeps (Retry-After clamped to 5-60 s, else 5 s), \
         never jittered — jitter would corrupt the gap measurement.\n\
         - Spend cap: total cap aborts the run; per-provider guard (cap x2 / selected count) stops \
         that provider first.\n\
         - Deadline: 55 minutes for the whole run; the per-provider deadline races every ladder sleep.\n\
         - Timestamps: ms-precision RFC 3339 (t_send / t_headers / t_body) for latency and gaps.\n\
         - Response-cache invalidation: a hit in x-openrouter-cache-status re-sends the round once \
         and marks it invalid (response_cache) if it persists."
    );
    let _ = writeln!(out);

    // 8. Abort note.
    if meta.aborted {
        let _ = writeln!(
            out,
            "## Aborted\n\nRun aborted: {}",
            meta.abort_reason.as_deref().unwrap_or("unknown reason")
        );
    }

    out
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
    use crate::bench_openrouter::run::RoundUsage;

    fn endpoint(tag: &str, cache_read: &str, prompt: &str, completion: &str) -> EndpointInfo {
        EndpointInfo {
            tag: tag.to_string(),
            name: format!("Name {tag}"),
            provider_name: format!("Provider {tag}"),
            model_id: None,
            model_name: None,
            context_length: Some(200_000),
            max_completion_tokens: None,
            quantization: Some("fp8".to_string()),
            status: Some("0".to_string()),
            supports_implicit_caching: Some(true),
            supported_parameters: None,
            pricing: Some(Pricing {
                prompt: Some(prompt.to_string()),
                completion: Some(completion.to_string()),
                request: Some("0".to_string()),
                input_cache_read: Some(cache_read.to_string()),
                input_cache_write: None,
                discount: None,
            }),
            latency_last_30m: None,
            uptime_last_1d: None,
            uptime_last_30m: None,
            uptime_last_5m: None,
        }
    }

    fn provider_run(tag: &str, billed: f64, estimated: f64, total_tokens: u64) -> ProviderRun {
        ProviderRun {
            tag: tag.to_string(),
            endpoint: endpoint(tag, "0.0000001", "0.000001", "0.000003"),
            selection_reason: None,
            cache_supported: true,
            contamination_warning: false,
            warmup: vec![RoundRecord {
                kind: "warmup",
                rung: None,
                nominal_gap_secs: None,
                measured_gap_ms: None,
                t_send: "2026-08-20T00:00:00.000Z".to_string(),
                t_headers: Some("2026-08-20T00:00:00.100Z".to_string()),
                t_body: "2026-08-20T00:00:00.200Z".to_string(),
                prompt_hash: "abc".to_string(),
                http_status: 200,
                finish_reason: Some("tool_calls".to_string()),
                serving_provider: Some("Name p".to_string()),
                pin_verified: true,
                generation_id: Some("gen-1".to_string()),
                response_cache_hit: false,
                usage: RoundUsage {
                    prompt_tokens: Some(1000),
                    completion_tokens: Some(10),
                    total_tokens: Some(1010),
                    cached_tokens: Some(900),
                    cache_write_tokens: Some(100),
                    miss_tokens: Some(100),
                    reasoning_tokens: Some(0),
                    cost: Some(billed / 2.0),
                },
                cache_classification: Some("hit".to_string()),
                expected_cached_tokens: Some(900),
                error_class: None,
                retries: 0,
                cache_status_header: Some("MISS".to_string()),
            }],
            ladder: vec![],
            cache_hold_bucket: "≥30m".to_string(),
            cache_hold_curve: vec![],
            billed_usd: billed,
            estimated_usd: estimated,
            total_tokens_reported: total_tokens,
            token_usage: json!({"cached": 0u64, "miss": 0u64, "output": 0u64,
                                "cache_write": 0u64, "reasoning": 0u64}),
            latency: json!({"header_ms": [100u64], "full_ms": [200u64]}),
            reliability: json!({"errors": [], "retries": 0, "rounds_failed": 0}),
            incomplete: false,
            incomplete_reason: None,
            aborted: false,
            abort_reason: None,
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
        let providers = vec![
            provider_run("acme-a/fp8", 0.5, 0.4, 100_000),
            provider_run("acme-b/fp8", 0.3, 0.2, 50_000),
        ];
        let reasons = vec![(true, Some("selected".to_string())), (false, None)];
        let selection = json!({"healthy_count": 2, "selected_count": 1});
        let report = build_report(
            &meta(),
            &selection,
            &providers,
            &reasons,
            Path::new("/tmp/bench"),
        );

        let expected_keys = ["run", "selection", "providers", "summary", "artifacts"];
        let object = report.as_object().expect("report is an object");
        assert_eq!(
            object.len(),
            expected_keys.len(),
            "unexpected top-level keys"
        );
        for k in expected_keys {
            assert!(object.contains_key(k), "missing top-level key {k}");
        }

        let arr = report["providers"].as_array().expect("providers array");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["tag"], "acme-a/fp8");
        assert_eq!(arr[0]["selected"], true);
        assert_eq!(arr[0]["selection_reason"], "selected");
        assert_eq!(arr[1]["selected"], false);
        assert!(arr[1]["selection_reason"].is_null());
        assert_eq!(arr[0]["cost"]["billed_usd"], 0.5);
        assert_eq!(arr[0]["cost"]["estimated_usd"], 0.4);
        assert!((arr[0]["cost"]["delta_usd"].as_f64().unwrap() - 0.1).abs() < 1e-9);
        // blended = billed / total × 1e6
        assert!((arr[0]["cost"]["blended_usd_per_m"].as_f64().unwrap() - 5.0).abs() < 1e-9);
        // per_m_at_validated_mix with cache_read=$0.1/M, prompt=$1/M, completion=$3/M
        assert!(
            (arr[0]["cost"]["per_m_at_validated_mix"].as_f64().unwrap()
                - (0.977 * 0.1 + 0.014 * 1.0 + 0.009 * 3.0))
                .abs()
                < 1e-9
        );
        assert!((arr[0]["cost"]["per_m_all_miss"].as_f64().unwrap() - 1.0).abs() < 1e-9);
        // warmup round passthrough
        assert_eq!(arr[0]["warmup"][0]["kind"], "warmup");
        assert_eq!(arr[0]["warmup"][0]["usage"]["prompt_tokens"], 1000);
        assert!(arr[0]["warmup"][0]["t_headers"].is_string());
        assert_eq!(report["summary"]["total_billed_usd"], 0.8);
        assert!((report["summary"]["total_estimated_usd"].as_f64().unwrap() - 0.6).abs() < 1e-9);
        assert_eq!(report["summary"]["completed_count"], 2);
        assert_eq!(report["summary"]["cache_supported_count"], 2);
        assert_eq!(report["summary"]["hold_distribution"]["≥30m"], 2);
        assert_eq!(report["artifacts"]["run_dir"], "/tmp/bench");
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
