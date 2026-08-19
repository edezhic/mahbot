//! `mahbot bench-openrouter` — standalone OpenRouter provider benchmark CLI.
//!
//! Dispatched from `main()` BEFORE the instance flock is acquired (like
//! `debug` and `__grep-engine`), so it runs while the live daemon holds the
//! lock. It never opens a live mahbot store for writing: the only database
//! touch is a read-only (ReadOnly|NoLock) `config.db` lookup for the worker
//! model / provider key fallback, identical in spirit to `mahbot debug`.
//!
//! # Phase 1 (this module)
//!
//! Discovery → selection → dry-run plan. `--dry-run` performs the full
//! discovery (models catalog, requested-model endpoints, key envelope) and
//! cost-based provider selection, then emits the canonical plan JSON to
//! stdout with ZERO chat-completions calls. The full-run executor and report
//! writers are Phase 2 — the data model (snapshot, selection, plan) is shaped
//! so they slot in without restructuring.
//!
//! # Exit codes
//!
//! `0` dry-run success; `1` hard error (bad key, unaffordable estimate,
//! discovery failure); `2` usage error (unknown flag / bad value).
//!
//! # Prompt
//!
//! The harness system prompt lives at `src/prompt/bench_openrouter.md`
//! (embedded via rust-embed) and is loaded through [`bench_system_prompt`] so
//! the "all LLM-sent prompts under src/prompt/" rule is enforced by
//! construction.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::bail;
use serde_json::json;

mod classify;
mod discovery;
mod select;

use crate::bench_openrouter::discovery::{
    DiscoveryClient, DiscoverySnapshot, EndpointInfo, Pricing, discover,
};
use crate::bench_openrouter::select::{
    ExclusionReason, SelectionDecision, SelectionInput, classify_endpoint, estimate_cost,
    plan_cost, select_providers, selection_target,
};

// ── Constants ──────────────────────────────────────────────────────

/// Minimum context window (tokens) for a provider to be benchmarked.
const MIN_CONTEXT: i64 = 128_000;

/// Warmup rounds per provider: establish the base cache before the ladder.
const WARMUP_ROUNDS: usize = 2;

/// Characters per token heuristic for the prompt-size estimate.
const CHARS_PER_TOKEN: f64 = 4.0;

/// Fixed prompt-history growth (tokens) added to the prefix estimate: the
/// conversation grows across the 2 warmup + N ladder rounds.
const HISTORY_GROWTH_TOKENS: f64 = 3360.0;

// ── CLI types ──────────────────────────────────────────────────────

/// A CLI failure: `Usage` → exit 2 (with usage on stderr), `Hard` → exit 1.
#[derive(Debug)]
pub(crate) enum CliError {
    Usage(String),
    Hard(String),
}

/// Parsed `bench-openrouter` options.
///
/// `model` is fully resolved at parse time (flag > env `MAHBOT_BENCH_MODEL` >
/// read-only config.db `worker_model` > [`crate::config::DEFAULT_WORKER_MODEL`]);
/// the API key is resolved later (flag > env > config.db, hard error when
/// nothing is configured) so the error path can be reported cleanly.
pub(crate) struct BenchOptions {
    pub model: String,
    pub api_key: Option<String>,
    pub cap_usd: f64,
    pub ladder_secs: Vec<u64>,
    pub providers: Option<Vec<String>>,
    pub prefix_chars: usize,
    pub output_dir: PathBuf,
    pub dry_run: bool,
}

/// Fetch the value of `flag` — either the inline `--flag=value` part or the
/// next argument; advances `i` in the latter case.
fn next_value(
    args: &[String],
    i: &mut usize,
    inline: Option<String>,
    flag: &str,
) -> Result<String, CliError> {
    if let Some(v) = inline {
        if v.is_empty() {
            return Err(CliError::Usage(format!(
                "flag {flag} requires a non-empty value"
            )));
        }
        return Ok(v);
    }
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| CliError::Usage(format!("flag {flag} requires a value")))
}

/// Parse a `--ladder` CSV into gap seconds.
fn parse_ladder(s: &str) -> Result<Vec<u64>, CliError> {
    let vals: Result<Vec<u64>, CliError> = s
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| {
            p.parse::<u64>().map_err(|_| {
                CliError::Usage(format!("invalid --ladder value '{p}' (expected integers)"))
            })
        })
        .collect();
    let vals = vals?;
    if vals.is_empty() {
        return Err(CliError::Usage(
            "--ladder must contain at least one gap (seconds)".to_string(),
        ));
    }
    Ok(vals)
}

impl BenchOptions {
    /// Parse `bench-openrouter` CLI arguments (no clap — manual, mirroring the
    /// debug subcommand's style). Both `--flag value` and `--flag=value` are
    /// accepted.
    pub(crate) fn parse(args: &[String]) -> Result<Self, CliError> {
        let mut model: Option<String> = None;
        let mut api_key: Option<String> = None;
        let mut cap_usd = 2.00;
        let mut ladder_secs: Vec<u64> = vec![0, 5, 30, 120, 300, 600, 1800];
        let mut providers: Option<Vec<String>> = None;
        let mut prefix_chars: usize = 64_000;
        let mut output_dir: Option<PathBuf> = None;
        let mut dry_run = false;

        let mut i = 0;
        while i < args.len() {
            let arg = args[i].clone();
            let (name, inline) = match arg.split_once('=') {
                Some((n, v)) if n.starts_with("--") => (n.to_string(), Some(v.to_string())),
                _ => (arg.clone(), None),
            };
            match name.as_str() {
                "--model" => model = Some(next_value(args, &mut i, inline, "--model")?),
                "--api-key" => api_key = Some(next_value(args, &mut i, inline, "--api-key")?),
                "--cap-usd" => {
                    let v = next_value(args, &mut i, inline, "--cap-usd")?;
                    cap_usd = v.parse::<f64>().map_err(|_| {
                        CliError::Usage(format!(
                            "invalid --cap-usd value '{v}' (expected a number)"
                        ))
                    })?;
                }
                "--ladder" => {
                    let v = next_value(args, &mut i, inline, "--ladder")?;
                    ladder_secs = parse_ladder(&v)?;
                }
                "--providers" => {
                    let v = next_value(args, &mut i, inline, "--providers")?;
                    providers = Some(
                        v.split(',')
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(String::from)
                            .collect(),
                    );
                }
                "--prefix-chars" => {
                    let v = next_value(args, &mut i, inline, "--prefix-chars")?;
                    prefix_chars = v.parse::<usize>().map_err(|_| {
                        CliError::Usage(format!(
                            "invalid --prefix-chars value '{v}' (expected a non-negative integer)"
                        ))
                    })?;
                }
                "--output-dir" => {
                    let v = next_value(args, &mut i, inline, "--output-dir")?;
                    output_dir = Some(PathBuf::from(v));
                }
                "--dry-run" => {
                    if inline.is_some() {
                        return Err(CliError::Usage("--dry-run takes no value".to_string()));
                    }
                    dry_run = true;
                }
                other if other.starts_with("--") => {
                    return Err(CliError::Usage(format!("unknown flag '{other}'")));
                }
                other => {
                    return Err(CliError::Usage(format!("unexpected argument '{other}'")));
                }
            }
            i += 1;
        }

        // Model resolution: --model > env > read-only config.db > default.
        let model = match model {
            Some(m) => m,
            None => model_from_env_or_config(),
        };

        // Output dir default: ~/.mahbot/benchmarks.
        let output_dir = if let Some(d) = output_dir {
            d
        } else {
            let root = crate::config::default_config_dir()
                .map_err(|e| CliError::Hard(format!("cannot resolve config dir: {e:#}")))?;
            root.join("benchmarks")
        };

        Ok(Self {
            model,
            api_key,
            cap_usd,
            ladder_secs,
            providers,
            prefix_chars,
            output_dir,
            dry_run,
        })
    }
}

/// Resolve the benchmark model when `--model` was not given:
/// env `MAHBOT_BENCH_MODEL` > read-only config.db `worker_model` >
/// [`crate::config::DEFAULT_WORKER_MODEL`]. Never fails.
fn model_from_env_or_config() -> String {
    if let Ok(m) = std::env::var("MAHBOT_BENCH_MODEL")
        && !m.is_empty()
    {
        return m;
    }
    if let Ok(root) = crate::config::default_config_dir()
        && let Ok(Some(m)) = read_config_kv(&root, "worker_model")
        && !m.is_empty()
    {
        return m;
    }
    crate::config::DEFAULT_WORKER_MODEL.to_string()
}

/// Resolve the API key: `--api-key` > env `MAHBOT_BENCH_API_KEY` >
/// env `OPENROUTER_API_KEY` > read-only config.db `provider_key` > hard error.
/// Returns the key and its source label for the plan.
fn resolve_key(opts: &BenchOptions) -> Result<(String, &'static str), CliError> {
    if let Some(k) = &opts.api_key
        && !k.is_empty()
    {
        return Ok((k.clone(), "flag"));
    }
    if let Ok(k) = std::env::var("MAHBOT_BENCH_API_KEY")
        && !k.is_empty()
    {
        return Ok((k, "env"));
    }
    if let Ok(k) = std::env::var("OPENROUTER_API_KEY")
        && !k.is_empty()
    {
        return Ok((k, "env"));
    }
    if let Ok(root) = crate::config::default_config_dir()
        && let Ok(Some(k)) = read_config_kv(&root, "provider_key")
        && !k.is_empty()
    {
        return Ok((k, "config"));
    }
    Err(CliError::Hard(
        "no OpenRouter API key — pass --api-key, set MAHBOT_BENCH_API_KEY/OPENROUTER_API_KEY, \
         or configure one in mahbot settings"
            .to_string(),
    ))
}

// ── Read-only config.db helper ─────────────────────────────────────

/// Read one `config_kv` value from the live config store, read-only.
///
/// Opens `<storage_root>/db/config.db` with `ReadOnly|NoLock` (the same path
/// `mahbot debug` uses — never creates or mutates files, reuses the daemon's
/// `.tshm` coordination). Any failure — missing file, unreadable store,
/// missing row — degrades to `Ok(None)` with a `tracing::warn`: this is a
/// fallback resolution path, never fatal.
// The `Result` wrapper is spec'd for the shared data model even though the
// body swallows all errors (Phase 2 callers may surface them).
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn read_config_kv(storage_root: &Path, key: &str) -> anyhow::Result<Option<String>> {
    let result = read_config_kv_inner(storage_root, key);
    match result {
        Ok(v) => Ok(v),
        Err(e) => {
            tracing::warn!(
                key,
                error = %e,
                "bench-openrouter: read-only config lookup failed; ignoring"
            );
            Ok(None)
        }
    }
}

fn read_config_kv_inner(storage_root: &Path, key: &str) -> anyhow::Result<Option<String>> {
    let db_path = crate::turso::store_db_path(storage_root, "config");
    if !db_path.exists() {
        return Ok(None);
    }
    let opts = turso::core::DatabaseOpts::new()
        .with_multiprocess_wal(true)
        .with_index_method(true);
    let (io, db) = crate::debug::open_readonly(&db_path, &db_path, opts)?;
    let conn = crate::debug::connect_readonly(&db, &db_path)?;

    let mut stmt = conn
        .query("SELECT value FROM config_kv WHERE key = ?1")
        .map_err(|e| anyhow::anyhow!("config read query failed: {e}"))?
        .ok_or_else(|| anyhow::anyhow!("config read query produced no statement"))?;
    stmt.bind_at(
        std::num::NonZero::new(1).expect("bound parameter index 1 is non-zero"),
        turso::core::Value::Text(turso::core::types::Text::from(key)),
    )?;

    loop {
        match stmt.step() {
            Ok(turso::core::StepResult::Row) => {
                let row = stmt
                    .row()
                    .ok_or_else(|| anyhow::anyhow!("row missing after StepResult::Row"))?;
                let value = row.get_value(0);
                return Ok(match value {
                    turso::core::Value::Text(t) => Some(t.as_str().to_string()),
                    _ => None,
                });
            }
            Ok(turso::core::StepResult::Done) => return Ok(None),
            Ok(turso::core::StepResult::IO | turso::core::StepResult::Yield) => {
                io.step()
                    .map_err(|e| anyhow::anyhow!("config read I/O step failed: {e}"))?;
            }
            Ok(turso::core::StepResult::Interrupt) => {
                bail!("config read query interrupted");
            }
            Ok(turso::core::StepResult::Busy) => {
                bail!("config store busy; try again later");
            }
            Err(e) => bail!("config read query failed: {e}"),
        }
    }
}

// ── CLI entry ──────────────────────────────────────────────────────

/// Full `bench-openrouter` CLI entry. Called from `main()` on a current-thread
/// runtime, before the instance flock is acquired. `-h`/`--help` prints usage
/// to stdout and exits 0; usage errors print usage to stderr and exit 2; hard
/// errors exit 1; a successful dry-run exits 0.
pub async fn run_cli() -> i32 {
    let args: Vec<String> = std::env::args().skip(2).collect();

    if args
        .iter()
        .any(|a| a == "-h" || a == "--help" || a.starts_with("--help="))
    {
        print!("{}", usage());
        return 0;
    }

    let opts = match BenchOptions::parse(&args) {
        Ok(o) => o,
        Err(CliError::Usage(msg)) => {
            eprintln!("Error: {msg}");
            eprint!("{}", usage());
            return 2;
        }
        Err(CliError::Hard(msg)) => {
            eprintln!("Error: {msg}");
            return 1;
        }
    };

    if opts.dry_run {
        dry_run(&opts).await
    } else {
        run_full(&opts)
    }
}

/// The full-run executor — Phase 2. For now it reports "not yet implemented".
fn run_full(_opts: &BenchOptions) -> i32 {
    eprintln!("Error: bench-openrouter full run is not yet implemented (Phase 2)");
    1
}

/// Usage text shared by `--help` (stdout) and usage errors (stderr).
fn usage() -> &'static str {
    "\
mahbot bench-openrouter — OpenRouter provider benchmark (Phase 1: dry-run)

Usage:
  mahbot bench-openrouter [flags]

Flags:
  --model <slug>         Model id to benchmark. Resolution: --model, env
                         MAHBOT_BENCH_MODEL, config.db worker_model, default.
  --api-key <key>        OpenRouter API key. Resolution: --api-key, env
                         MAHBOT_BENCH_API_KEY, env OPENROUTER_API_KEY,
                         config.db provider_key.
  --cap-usd <f64>        Total cost cap in USD (default: 2.00).
  --ladder <csv>         Inactivity-gap ladder in seconds (default:
                         0,5,30,120,300,600,1800).
  --providers <csv>      Provider tag allowlist (e.g. streamlake/fp8,deepseek/auto).
  --prefix-chars <usize> Estimated prompt prefix size in characters (default: 64000).
  --output-dir <dir>     Report output directory (default: ~/.mahbot/benchmarks).
  --dry-run              Discovery + selection + plan only; ZERO chat calls.
  -h, --help             Print this help and exit.
"
}

// ── Dry-run orchestration ──────────────────────────────────────────

/// Discovery → selection → plan JSON to stdout. Never makes a chat-completions
/// call. Prints nothing secret (the resolved API key is never echoed).
async fn dry_run(opts: &BenchOptions) -> i32 {
    // 1. Resolve key + model.
    let (key, key_source) = match resolve_key(opts) {
        Ok(k) => k,
        Err(CliError::Hard(msg)) => {
            eprintln!("Error: {msg}");
            return 1;
        }
        Err(CliError::Usage(msg)) => {
            eprintln!("Error: {msg}");
            return 2;
        }
    };

    // 2. Discovery (models catalog, endpoints, key envelope).
    let client = DiscoveryClient::new(key);
    let snapshot = match discover(&client, &opts.model).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {e:#}");
            return 1;
        }
    };

    // 3. Resolve the reasoning config + supported parameters from the catalog.
    let catalog_entry = snapshot.catalog.iter().find(|e| e.id == opts.model);
    let reasoning_effort_used = catalog_entry
        .as_ref()
        .and_then(|e| e.reasoning.as_ref())
        .map(|r| {
            // Pick the LAST (lowest) supported effort; if the model advertises
            // reasoning without a supported-efforts list, fall back to "low".
            r.supported_efforts
                .as_ref()
                .and_then(|efforts| efforts.last().cloned())
                .unwrap_or_else(|| "low".to_string())
        });
    let supported_parameters = catalog_entry
        .as_ref()
        .and_then(|e| e.supported_parameters.as_ref())
        .map_or::<&[String], _>(&[], Vec::as_slice);
    let supports_tools = supported_parameters.iter().any(|p| p == "tools");
    let supports_tool_choice = supported_parameters.iter().any(|p| p == "tool_choice");
    let supports_reasoning_effort = supported_parameters.iter().any(|p| p == "reasoning_effort");

    // 4. Selection: per-endpoint cost estimate + health classification.
    let rounds = rounds_per_provider(&opts.ladder_secs);
    let total_tokens = total_tokens_estimate(opts.prefix_chars, rounds);
    let mut inputs = Vec::with_capacity(snapshot.endpoints.data.endpoints.len());
    let mut flags = Vec::with_capacity(snapshot.endpoints.data.endpoints.len());
    let default_price = Pricing::default();
    for ep in &snapshot.endpoints.data.endpoints {
        let mut f = Vec::new();
        let price = if let Some(p) = &ep.pricing {
            p
        } else {
            f.push("no pricing advertised; estimate assumes zero cost".to_string());
            &default_price
        };
        let est_cost = estimate_cost(price, total_tokens, rounds, &mut f);
        let (healthy, reason) = classify_endpoint(ep, MIN_CONTEXT, opts.providers.as_deref());
        inputs.push(SelectionInput {
            endpoint: ep.clone(),
            est_cost,
            healthy,
            excluded_reason: reason.map(|r| r.to_string()),
        });
        flags.push(f);
    }
    let decisions = select_providers(&inputs, MIN_CONTEXT, opts.providers.as_deref());
    let (total_est, per_provider_guard) = plan_cost(&decisions, &inputs, opts.cap_usd);

    // 5. Affordability preflight against the key envelope.
    if let Some(remaining) = snapshot.key.limit_remaining {
        if total_est > remaining {
            eprintln!(
                "Error: estimated cost ${total_est:.4} exceeds the remaining key limit ${remaining:.4}"
            );
            return 1;
        }
        if total_est > 0.25 * remaining {
            eprintln!(
                "Warning: estimated cost ${total_est:.4} is more than 25% of the remaining key limit ${remaining:.4}"
            );
        }
    }

    // 6. Plan JSON → stdout (EPIPE-tolerant).
    let plan = build_plan(&PlanData {
        opts,
        snapshot: &snapshot,
        key_source,
        reasoning_effort_used,
        supports_tools,
        supports_tool_choice,
        supports_reasoning_effort,
        inputs: &inputs,
        flags: &flags,
        decisions: &decisions,
        total_est,
        per_provider_guard,
    });
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, "{plan}");
    0
}

// ── Plan building ──────────────────────────────────────────────────

/// All inputs the plan builder needs (bundled so `build_plan` is pure and
/// testable without I/O).
pub(crate) struct PlanData<'a> {
    pub opts: &'a BenchOptions,
    pub snapshot: &'a DiscoverySnapshot,
    pub key_source: &'a str,
    pub reasoning_effort_used: Option<String>,
    pub supports_tools: bool,
    pub supports_tool_choice: bool,
    pub supports_reasoning_effort: bool,
    pub inputs: &'a [SelectionInput],
    pub flags: &'a [Vec<String>],
    pub decisions: &'a [SelectionDecision],
    pub total_est: f64,
    pub per_provider_guard: f64,
}

/// Rounds per provider: 2 warmup rounds (no gap) + one ladder round per
/// ladder entry.
#[must_use]
pub(crate) fn rounds_per_provider(ladder_secs: &[u64]) -> usize {
    WARMUP_ROUNDS + ladder_secs.len()
}

/// Total tokens per provider for the cost estimate: the prefix-size estimate
/// (`prefix_chars / chars-per-token`) across all rounds, plus the fixed
/// prompt-history growth.
#[must_use]
#[allow(clippy::cast_precision_loss)] // usize→f64 exact below 2^53; estimate math
pub(crate) fn total_tokens_estimate(prefix_chars: usize, rounds: usize) -> f64 {
    (prefix_chars as f64 / CHARS_PER_TOKEN) * rounds as f64 + HISTORY_GROWTH_TOKENS
}

/// Build the canonical dry-run plan JSON (keys sorted — serde_json's default
/// `Map` is a `BTreeMap`).
#[must_use]
pub(crate) fn build_plan(data: &PlanData<'_>) -> serde_json::Value {
    let opts = data.opts;
    let snapshot = data.snapshot;
    let endpoints = &snapshot.endpoints.data.endpoints;
    let decisions = data.decisions;
    let inputs = data.inputs;

    let providers = provider_entries(endpoints, decisions, inputs, data.flags);

    let selected_count = decisions.iter().filter(|d| d.selected).count();
    let padding_count = decisions
        .iter()
        .filter(|d| matches!(d.reason, Some(ExclusionReason::Padding)))
        .count();
    let healthy_count = inputs.iter().filter(|i| i.healthy).count();
    let allowlist_matches = opts.providers.as_ref().map(|wl| {
        wl.iter()
            .filter(|t| inputs.iter().any(|i| i.endpoint.tag == **t))
            .count()
    });
    let target_count = match (healthy_count, allowlist_matches) {
        (0, _) => 0,
        (_, Some(m)) if m > 0 && m <= 2 => m,
        _ => selection_target(healthy_count),
    };

    let rounds = rounds_per_provider(&opts.ladder_secs);
    let total_delay_secs: u64 = opts.ladder_secs.iter().sum();
    let within_cap = data.total_est <= opts.cap_usd;

    let catalog_entry = snapshot.catalog.iter().find(|e| e.id == opts.model);
    let resolved_id = catalog_entry.map_or_else(|| opts.model.clone(), |e| e.id.clone());

    json!({
        "mode": "dry-run",
        "timestamp": crate::turso::now(),
        "model": {
            "requested": opts.model,
            "resolved_id": resolved_id,
            "canonical_slug": catalog_entry.as_ref().and_then(|e| e.canonical_slug.clone()),
            "name": catalog_entry.as_ref().and_then(|e| e.name.clone()),
            "context_length": catalog_entry.as_ref().and_then(|e| e.context_length),
            "reasoning_effort_used": data.reasoning_effort_used,
            "supports_tools": data.supports_tools,
            "supports_tool_choice": data.supports_tool_choice,
            "supports_reasoning_effort": data.supports_reasoning_effort,
        },
        "key": {
            "source": data.key_source,
            // Discovery succeeded, so the key authenticated against the API.
            "valid": true,
            "label": snapshot.key.label,
            "limit": snapshot.key.limit,
            "limit_remaining": snapshot.key.limit_remaining,
            "limit_reset": snapshot.key.limit_reset,
            "is_free_tier": snapshot.key.is_free_tier,
            "redacted": true,
        },
        "discovery": {
            "fetched_at": snapshot.fetched_at,
            "catalog_count": snapshot.catalog.len(),
            "endpoint_count": endpoints.len(),
        },
        "providers": providers,
        "selection": {
            "healthy_count": healthy_count,
            "selected_count": selected_count,
            "padding_count": padding_count,
            "target_count": target_count,
            "min_context": MIN_CONTEXT,
            "allowlist": opts.providers.clone().unwrap_or_default(),
        },
        "schedule": {
            "ladder_secs": opts.ladder_secs,
            "total_delay_secs": total_delay_secs,
            "rounds_per_provider": rounds,
            "warmup_rounds": WARMUP_ROUNDS,
            "concurrent_providers": selected_count,
        },
        "cost": {
            "total_est_usd": data.total_est,
            "cap_usd": opts.cap_usd,
            "per_provider_guard_usd": data.per_provider_guard,
            "within_cap": within_cap,
        },
        "request_count": selected_count * rounds,
        "output_dir": opts.output_dir.display().to_string(),
    })
}

/// Build the `providers` plan array: selected providers first, sorted by est
/// cost ascending (rule 7); unselected providers keep input order.
fn provider_entries(
    endpoints: &[EndpointInfo],
    decisions: &[SelectionDecision],
    inputs: &[SelectionInput],
    flags: &[Vec<String>],
) -> Vec<serde_json::Value> {
    let mut order: Vec<usize> = (0..inputs.len()).collect();
    order.sort_by(|&a, &b| {
        let (sa, sb) = (decisions[a].selected, decisions[b].selected);
        match (sa, sb) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (true, true) => inputs[a].est_cost.total_cmp(&inputs[b].est_cost),
            (false, false) => std::cmp::Ordering::Equal,
        }
    });

    order
        .iter()
        .map(|&i| {
            let ep = &endpoints[i];
            let decision = &decisions[i];
            // A human string from the ExclusionReason when one is recorded
            // (e.g. padded providers show "padding (expected to fail)"),
            // otherwise "selected".
            let selection_reason = match (&decision.reason, decision.selected) {
                (Some(r), _) => r.to_string(),
                (None, true) => "selected".to_string(),
                (None, false) => "not selected".to_string(),
            };
            json!({
                "tag": ep.tag,
                "name": ep.name,
                "provider_name": ep.provider_name,
                "quantization": ep.quantization,
                "status": ep.status,
                "supports_implicit_caching": ep.supports_implicit_caching,
                "context_length": ep.context_length,
                "selected": decision.selected,
                "selection_reason": selection_reason,
                "est_cost_usd": inputs[i].est_cost,
                "flags": flags[i],
            })
        })
        .collect()
}

// ── Harness prompt ─────────────────────────────────────────────────

/// The benchmark harness system prompt (loaded from the embedded asset so the
/// "all LLM-sent prompts live under src/prompt/" rule holds by construction).
/// Phase 2's executor sends it; unused until then.
#[allow(dead_code)]
#[must_use]
pub(crate) fn bench_system_prompt() -> String {
    crate::prompt::load_prompt("bench_openrouter.md")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench_openrouter::select::select_providers;

    /// Synthetic discovery snapshot (no HTTP, no DBs).
    fn test_snapshot() -> DiscoverySnapshot {
        let models = serde_json::json!({"data": [{
            "id": "acme/model-1",
            "canonical_slug": "acme/model-1",
            "name": "Acme Model 1",
            "context_length": 200000,
            "reasoning": {"default_effort": "high", "supported_efforts": ["xhigh","high","medium","low"], "mandatory": false},
            "supported_parameters": ["tools", "tool_choice", "reasoning_effort"]
        }]});
        let endpoints = serde_json::json!({"data": {
            "id": "acme/model-1",
            "name": "Acme Model 1",
            "endpoints": [
                {"tag":"acme-a/fp8","name":"Acme A","provider_name":"Acme","status":"0","context_length":200000,"supports_implicit_caching":true,"quantization":"fp8",
                 "pricing":{"prompt":"0.000002","completion":"0.000008","request":"0","input_cache_read":"0.0000002"}},
                {"tag":"acme-b/fp8","name":"Acme B","provider_name":"Acme","status":"0","context_length":200000,"supports_implicit_caching":true,"quantization":"fp8",
                 "pricing":{"prompt":"0.000003","completion":"0.000009","request":"0","input_cache_read":"0.0000003"}},
                {"tag":"acme-c/fp8","name":"Acme C","provider_name":"Acme","status":"-10","context_length":200000,"supports_implicit_caching":false,"quantization":"fp8",
                 "pricing":{"prompt":"0.000004","completion":"0.000010","request":"0","input_cache_read":"0.0000004"}}
            ]
        }});
        let key = serde_json::json!({"data": {
            "limit": 100.0, "limit_remaining": 74.5, "limit_reset": "monthly",
            "usage": 25.5, "is_free_tier": false, "label": "sk-or-bench"
        }});
        DiscoverySnapshot {
            fetched_at: "2026-08-20T00:00:00.000Z".to_string(),
            catalog: serde_json::from_value(models["data"].clone()).expect("catalog fixture"),
            endpoints: serde_json::from_value(endpoints.clone()).expect("endpoints fixture"),
            key: serde_json::from_value(key["data"].clone()).expect("key fixture"),
            raw_models_json: models,
            raw_endpoints_json: endpoints,
            raw_key_json: key,
        }
    }

    fn test_options() -> BenchOptions {
        BenchOptions {
            model: "acme/model-1".to_string(),
            api_key: Some("sk-or-secret-key".to_string()),
            cap_usd: 2.0,
            // 8 ladder entries → rounds = 2 + 8 = 10 (the spec's nominal shape).
            ladder_secs: vec![0, 5, 30, 120, 300, 600, 1800, 3600],
            providers: None,
            prefix_chars: 64_000,
            output_dir: PathBuf::from("/tmp/benchmarks"),
            dry_run: true,
        }
    }

    /// Build the selection pipeline outputs (owned) from a snapshot, so the
    /// test can hold references for the [`PlanData`] lifetime.
    fn selection_outputs(
        opts: &BenchOptions,
        snapshot: &DiscoverySnapshot,
    ) -> (
        Vec<SelectionInput>,
        Vec<Vec<String>>,
        Vec<SelectionDecision>,
        f64,
        f64,
    ) {
        let rounds = rounds_per_provider(&opts.ladder_secs);
        let total_tokens = total_tokens_estimate(opts.prefix_chars, rounds);
        let mut inputs = Vec::new();
        let mut flags = Vec::new();
        let default_price = Pricing::default();
        for ep in &snapshot.endpoints.data.endpoints {
            let mut f = Vec::new();
            let price = ep.pricing.as_ref().unwrap_or(&default_price);
            let est_cost = estimate_cost(price, total_tokens, rounds, &mut f);
            let (healthy, reason) = classify_endpoint(ep, MIN_CONTEXT, None);
            inputs.push(SelectionInput {
                endpoint: ep.clone(),
                est_cost,
                healthy,
                excluded_reason: reason.map(|r| r.to_string()),
            });
            flags.push(f);
        }
        let decisions = select_providers(&inputs, MIN_CONTEXT, None);
        let (total_est, per_provider_guard) = plan_cost(&decisions, &inputs, opts.cap_usd);
        (inputs, flags, decisions, total_est, per_provider_guard)
    }

    #[test]
    fn dry_run_plan_shape() {
        let opts = test_options();
        let snapshot = test_snapshot();
        let rounds = rounds_per_provider(&opts.ladder_secs);
        let (inputs, flags, decisions, total_est, per_provider_guard) =
            selection_outputs(&opts, &snapshot);
        let data = PlanData {
            opts: &opts,
            snapshot: &snapshot,
            key_source: "env",
            reasoning_effort_used: Some("low".to_string()),
            supports_tools: true,
            supports_tool_choice: true,
            supports_reasoning_effort: true,
            inputs: &inputs,
            flags: &flags,
            decisions: &decisions,
            total_est,
            per_provider_guard,
        };
        let plan = build_plan(&data);

        assert_eq!(plan["mode"], "dry-run");
        // Exactly the specified top-level keys.
        let expected_keys = [
            "mode",
            "timestamp",
            "model",
            "key",
            "discovery",
            "providers",
            "selection",
            "schedule",
            "cost",
            "request_count",
            "output_dir",
        ];
        let object = plan.as_object().expect("plan is an object");
        assert_eq!(
            object.len(),
            expected_keys.len(),
            "unexpected top-level keys"
        );
        for k in expected_keys {
            assert!(object.contains_key(k), "missing top-level key {k}");
        }

        // request_count == selected × rounds (2 warmup + 8 ladder = 10 here).
        let selected_count = decisions.iter().filter(|d| d.selected).count();
        assert_eq!(plan["request_count"], (selected_count * rounds) as u64);
        assert_eq!(plan["request_count"], (selected_count * 10) as u64);

        // providers array mirrors every endpoint once.
        let providers = plan["providers"].as_array().expect("providers array");
        assert_eq!(providers.len(), 3);
        // One endpoint is unhealthy → padded in, so all 3 are selected.
        assert_eq!(plan["selection"]["selected_count"], 3);
        assert_eq!(plan["selection"]["padding_count"], 1);
        // Selected providers come first, sorted by est cost ascending.
        let ests: Vec<f64> = providers
            .iter()
            .take_while(|p| p["selected"] == true)
            .map(|p| p["est_cost_usd"].as_f64().unwrap())
            .collect();
        let mut sorted = ests.clone();
        sorted.sort_by(f64::total_cmp);
        assert_eq!(ests, sorted, "selected providers must be cost-ascending");

        // Redaction marker present; the resolved key value never appears.
        assert_eq!(plan["key"]["redacted"], true);
        assert!(!plan.to_string().contains("sk-or-secret-key"));
    }

    #[test]
    fn harness_prompt_loads() {
        let prompt = bench_system_prompt();
        assert!(!prompt.is_empty());
        assert!(
            prompt.contains("fast_tool"),
            "harness prompt must instruct a fast_tool call"
        );
        assert!(
            !prompt.contains("{{"),
            "harness prompt must not use template variables"
        );
    }

    #[test]
    fn ladder_rounds_derivation() {
        assert_eq!(rounds_per_provider(&[0, 5, 30, 120, 300, 600, 1800]), 9);
        assert_eq!(
            rounds_per_provider(&[0, 5, 30, 120, 300, 600, 1800, 3600]),
            10
        );
    }

    #[test]
    fn cli_parse_basic_flags() {
        let args = vec![
            "--model=acme/m1".to_string(),
            "--api-key".to_string(),
            "sk-or-x".to_string(),
            "--cap-usd".to_string(),
            "1.5".to_string(),
            "--ladder".to_string(),
            "0,10,60".to_string(),
            "--providers".to_string(),
            "a/fp8,b/fp8".to_string(),
            "--prefix-chars=32000".to_string(),
            "--output-dir".to_string(),
            "/tmp/out".to_string(),
            "--dry-run".to_string(),
        ];
        let opts = BenchOptions::parse(&args).expect("parse succeeds");
        assert_eq!(opts.model, "acme/m1");
        assert_eq!(opts.api_key.as_deref(), Some("sk-or-x"));
        assert!((opts.cap_usd - 1.5).abs() < 1e-12);
        assert_eq!(opts.ladder_secs, vec![0, 10, 60]);
        assert_eq!(
            opts.providers.as_deref(),
            Some(&["a/fp8".to_string(), "b/fp8".to_string()][..])
        );
        assert_eq!(opts.prefix_chars, 32_000);
        assert_eq!(opts.output_dir, PathBuf::from("/tmp/out"));
        assert!(opts.dry_run);
    }

    #[test]
    fn cli_parse_errors() {
        // Unknown flag → Usage.
        assert!(matches!(
            BenchOptions::parse(&["--bogus".to_string()]),
            Err(CliError::Usage(_))
        ));
        // Missing value → Usage.
        assert!(matches!(
            BenchOptions::parse(&["--model".to_string()]),
            Err(CliError::Usage(_))
        ));
        // Bad numeric value → Usage.
        assert!(matches!(
            BenchOptions::parse(&["--cap-usd".to_string(), "abc".to_string()]),
            Err(CliError::Usage(_))
        ));
        assert!(matches!(
            BenchOptions::parse(&["--ladder".to_string(), "0,x".to_string()]),
            Err(CliError::Usage(_))
        ));
    }
}
