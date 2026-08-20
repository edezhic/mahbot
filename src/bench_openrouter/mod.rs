//! `mahbot bench-openrouter` — standalone OpenRouter provider benchmark CLI.
//!
//! Dispatched from `main()` BEFORE the instance flock is acquired (like
//! `debug` and `__grep-engine`), so it runs while the live daemon holds the
//! lock. It never opens a live mahbot store for writing: the only database
//! touch is a read-only (ReadOnly|NoLock) `config.db` lookup for the worker
//! model / provider key fallback, identical in spirit to `mahbot debug`.
//!
//! # Modes
//!
//! `--dry-run` performs the full discovery (models catalog, requested-model
//! endpoints, key envelope) and cost-based provider selection, then emits the
//! canonical plan JSON to stdout with ZERO chat-completions calls.
//!
//! The default (full-run) mode adds the TTL-ladder executor: one tokio task
//! per selected provider runs the warmup + ladder rounds against the pinned
//! endpoint, and the run writes report.json / summary.md / providers.json /
//! manifest.json under `<output-dir>/bench-openrouter/<run-ts>/` plus a
//! `latest/` mirror ([`crate::bench_openrouter::report`]). The report is a
//! per-provider list of the static price and the cache-hold result ("does not
//! cache" / TTL hold bucket / "not measured").
//!
//! # Exit codes
//!
//! `0` dry-run success, or a full run whose artifacts were written; `1` hard
//! error (bad key, unaffordable estimate, discovery failure, lock held, write
//! failure) — a full-run hard abort still writes the partial report first;
//! `2` usage error (unknown flag / bad value).
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

use crate::util::UnwrapPoison;

mod classify;
mod discovery;
pub(crate) mod report;
pub(crate) mod run;
mod select;

use crate::bench_openrouter::discovery::{
    DiscoveryClient, DiscoverySnapshot, EndpointInfo, Pricing, discover,
};
use crate::bench_openrouter::select::{
    ExclusionReason, SelectionDecision, SelectionInput, effective_target_count, estimate_cost,
    plan_cost, select_providers,
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

        if !cap_usd.is_finite() || cap_usd <= 0.0 {
            return Err(CliError::Usage(
                "--cap-usd must be a positive number".to_string(),
            ));
        }

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
// The `Result` wrapper is part of the shared helper contract even though the
// body swallows all errors (read-only fallback path).
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
        Err(e) => return cli_error_exit(e),
    };

    if opts.dry_run {
        dry_run(&opts).await
    } else {
        full_run(&opts).await
    }
}

/// Usage text shared by `--help` (stdout) and usage errors (stderr).
fn usage() -> &'static str {
    "\
mahbot bench-openrouter — OpenRouter provider benchmark

Usage:
  mahbot bench-openrouter [flags]

Two modes:
  --dry-run   Discovery + selection + plan JSON to stdout. ZERO chat calls.
  (default)   Full run: TTL ladder over the selected providers, then writes
              report.json / summary.md / providers.json / manifest.json to
              <output-dir>/bench-openrouter/<run-ts>/ with a latest/ mirror.

Flags:
  --model <slug>         Model id to benchmark. Resolution: --model, env
                         MAHBOT_BENCH_MODEL, config.db worker_model, default.
  --api-key <key>        OpenRouter API key. Resolution: --api-key, env
                         MAHBOT_BENCH_API_KEY, env OPENROUTER_API_KEY,
                         config.db provider_key. Never printed; the manifest
                         stores only '***' for its value.
  --cap-usd <f64>        Total cost cap in USD (default: 2.00). Also sets the
                         per-provider guard (cap x2 / selected count).
  --ladder <csv>         Inactivity-gap ladder in seconds (default:
                         0,5,30,120,300,600,1800).
  --providers <csv>      Provider tag allowlist (e.g. streamlake/fp8,deepseek/auto).
  --prefix-chars <usize> Estimated prompt prefix size in characters (default: 64000).
  --output-dir <dir>     Report output directory (default: ~/.mahbot/benchmarks).
  -h, --help             Print this help and exit.

Exit codes:
  0  Success. In a full run, the artifacts (report.json / summary.md /
     providers.json / manifest.json) are complete; each provider's result is
     its static price and cache-hold bucket ('does not cache', a TTL hold like
     '(5s, 30s]', or 'not measured').
  1  Hard failure (bad key, unaffordable estimate, discovery failure, lock
     held, artifact write failure). A full-run abort (spend cap / deadline /
     auth / quota) still writes the partial report before exiting 1.
  2  Usage error (unknown flag / bad value).

Redaction: the API key is never printed. --api-key / env values are recorded
as '***' in manifest.json; providers.json marks the key payload redacted.

Not implemented (design-only): --from-plan, --max-gap-min, --all-endpoints,
the streaming probe, and the /generation audit.
"
}

// ── Shared helpers (dry-run + full-run) ────────────────────────────

/// Model-level capabilities resolved from the discovery catalog: the reasoning
/// effort the run will request and which request parameters the model
/// advertises support for.
pub(crate) struct ModelConfig {
    reasoning_effort_used: Option<String>,
    supports_tools: bool,
    supports_tool_choice: bool,
    supports_reasoning_effort: bool,
}

/// Resolve the reasoning config + supported parameters from the catalog entry
/// for `model`. The reasoning effort picks the LAST (lowest) supported effort;
/// a model that advertises reasoning without a supported-efforts list falls
/// back to `"low"`.
fn resolve_model_config(snapshot: &DiscoverySnapshot, model: &str) -> ModelConfig {
    let catalog_entry = snapshot.catalog.iter().find(|e| e.id == model);
    let reasoning_effort_used = catalog_entry
        .as_ref()
        .and_then(|e| e.reasoning.as_ref())
        .map(|r| {
            r.supported_efforts
                .as_ref()
                .and_then(|efforts| efforts.last().cloned())
                .unwrap_or_else(|| "low".to_string())
        });
    let supported_parameters = catalog_entry
        .as_ref()
        .and_then(|e| e.supported_parameters.as_ref())
        .map_or::<&[String], _>(&[], Vec::as_slice);
    ModelConfig {
        reasoning_effort_used,
        supports_tools: supported_parameters.iter().any(|p| p == "tools"),
        supports_tool_choice: supported_parameters.iter().any(|p| p == "tool_choice"),
        supports_reasoning_effort: supported_parameters.iter().any(|p| p == "reasoning_effort"),
    }
}

/// Selection pipeline outputs for one snapshot: per-endpoint cost estimate +
/// health classification, the estimate flags, the selection decisions, and the
/// plan cost (total estimate + per-provider spend guard).
pub(crate) struct SelectionBundle {
    inputs: Vec<SelectionInput>,
    flags: Vec<Vec<String>>,
    decisions: Vec<SelectionDecision>,
    total_est: f64,
    per_provider_guard: f64,
}

/// Build the selection bundle: per-endpoint cost estimate + health
/// classification → selection decisions → plan cost.
fn build_selection(opts: &BenchOptions, snapshot: &DiscoverySnapshot) -> SelectionBundle {
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
        inputs.push(SelectionInput {
            endpoint: ep.clone(),
            est_cost,
        });
        flags.push(f);
    }
    let decisions = select_providers(&inputs, MIN_CONTEXT, opts.providers.as_deref());
    let (total_est, per_provider_guard) = plan_cost(&decisions, &inputs, opts.cap_usd);
    SelectionBundle {
        inputs,
        flags,
        decisions,
        total_est,
        per_provider_guard,
    }
}

/// Everything the dry-run plan and the full-run executor need from the shared
/// preamble: resolved key, discovery snapshot, model config, and selection.
struct RunPreamble {
    key: String,
    key_source: &'static str,
    snapshot: DiscoverySnapshot,
    model_config: ModelConfig,
    bundle: SelectionBundle,
}

/// Shared preamble for both modes: key resolution → discovery → model config
/// → selection → affordability preflight. Returns a hard/usage error the
/// callers turn into an exit code.
async fn run_preamble(opts: &BenchOptions) -> Result<RunPreamble, CliError> {
    let (key, key_source) = resolve_key(opts)?;
    let client = DiscoveryClient::new(key.clone());
    let snapshot = discover(&client, &opts.model)
        .await
        .map_err(|e| CliError::Hard(format!("{e:#}")))?;
    let model_config = resolve_model_config(&snapshot, &opts.model);
    let bundle = build_selection(opts, &snapshot);
    if let Some(msg) = affordability_preflight(&snapshot, bundle.total_est) {
        return Err(CliError::Hard(msg));
    }
    Ok(RunPreamble {
        key,
        key_source,
        snapshot,
        model_config,
        bundle,
    })
}

/// Print a CLI error the way run_cli does today and return its exit code.
fn cli_error_exit(e: CliError) -> i32 {
    match e {
        CliError::Usage(msg) => {
            eprintln!("Error: {msg}");
            eprint!("{}", usage());
            2
        }
        CliError::Hard(msg) => {
            eprintln!("Error: {msg}");
            1
        }
    }
}

/// The affordability preflight shared by both modes: `Some` with a hard-fail
/// message when the estimate exceeds the remaining key limit (the caller turns
/// it into an error/exit code), `None` otherwise. Warns on stderr (non-fatal)
/// above 25% of the remaining limit.
fn affordability_preflight(snapshot: &DiscoverySnapshot, total_est: f64) -> Option<String> {
    let remaining = snapshot.key.limit_remaining?;
    if total_est > remaining {
        return Some(format!(
            "estimated cost ${total_est:.4} exceeds the remaining key limit ${remaining:.4}"
        ));
    }
    if total_est > 0.25 * remaining {
        eprintln!(
            "Warning: estimated cost ${total_est:.4} is more than 25% of the remaining key limit ${remaining:.4}"
        );
    }
    None
}

// ── Dry-run orchestration ──────────────────────────────────────────

/// Discovery → selection → plan JSON to stdout. Never makes a chat-completions
/// call. Prints nothing secret (the resolved API key is never echoed).
async fn dry_run(opts: &BenchOptions) -> i32 {
    // 1-4. Shared preamble: key → discovery → model config → selection →
    //      affordability preflight.
    let preamble = match run_preamble(opts).await {
        Ok(p) => p,
        Err(e) => return cli_error_exit(e),
    };
    let RunPreamble {
        key: _,
        key_source,
        snapshot,
        model_config,
        bundle,
    } = preamble;

    // 5. Plan JSON → stdout (EPIPE-tolerant).
    let plan = build_plan(&PlanData {
        opts,
        snapshot: &snapshot,
        key_source,
        reasoning_effort_used: model_config.reasoning_effort_used,
        supports_tools: model_config.supports_tools,
        supports_tool_choice: model_config.supports_tool_choice,
        supports_reasoning_effort: model_config.supports_reasoning_effort,
        inputs: &bundle.inputs,
        flags: &bundle.flags,
        decisions: &bundle.decisions,
        total_est: bundle.total_est,
        per_provider_guard: bundle.per_provider_guard,
    });
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, "{plan}");
    0
}

// ── Full-run orchestration ─────────────────────────────────────────

/// The full-run executor: discovery + selection → one task per selected
/// provider (staggered) → TTL-ladder rounds → report artifacts.
///
/// Exit codes: `0` when the artifacts were written and the run was not aborted
/// at run level; `1` on a hard failure (bad key, unaffordable estimate,
/// discovery failure, lock held, write failure) or a run-level abort (spend
/// cap / outer deadline / auth / quota) — the partial report is still written
/// before returning 1; `2` is not used here (parse errors exit before this
/// point).
// One long sequential orchestration (steps 1-15); splitting it
// would obscure the step ordering.
#[allow(clippy::too_many_lines)]
async fn full_run(opts: &BenchOptions) -> i32 {
    use std::sync::Mutex;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    // 1-4. Shared preamble: key → discovery → model config → selection →
    //      affordability preflight.
    let preamble = match run_preamble(opts).await {
        Ok(p) => p,
        Err(e) => return cli_error_exit(e),
    };
    let RunPreamble {
        key,
        key_source,
        snapshot,
        model_config,
        bundle,
    } = preamble;

    // 5. The bench's own run lock (never the daemon's mahbot.lock).
    let _run_lock = match report::acquire_run_lock(&opts.output_dir) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error: {e:#}");
            return 1;
        }
    };

    // 6. Run start + base prompt (per-run nonce, deterministic filler).
    let run_started = std::time::Instant::now();
    let run_ts = chrono::Utc::now();
    let started_at = run_ts.to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    let nonce = format!("{:016x}", rand::random::<u64>());
    let filler = run::generate_filler(opts.prefix_chars);
    let base = run::build_base_prompt(&bench_system_prompt(), &nonce, &filler);

    // 7. Bench reqwest client (run.rs's send_round applies the 120s request
    //    timeout itself).
    crate::util::http::install_ring_provider();
    let client = match reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: failed to build the benchmark HTTP client: {e}");
            return 1;
        }
    };

    // 8. Run context: budget + abort token + abort reason + deadline. Shared
    //    with the spawned tasks via Arc (JoinSet tasks are 'static).
    let ctx = std::sync::Arc::new(run::RunContext {
        budget: run::RunBudget::new(opts.cap_usd, bundle.per_provider_guard),
        abort: CancellationToken::new(),
        abort_reason: Mutex::new(None),
        deadline: run_started + Duration::from_secs(run::RUN_DEADLINE_MINS * 60),
    });

    // 9. One task per SELECTED provider (index = position among selected),
    //    staggered deterministically. The tasks share ctx immutably — the
    //    abort token and budget are internally synchronized.
    let selected: Vec<usize> = bundle
        .decisions
        .iter()
        .enumerate()
        .filter(|(_, d)| d.selected)
        .map(|(i, _)| i)
        .collect();
    if selected.is_empty() {
        eprintln!(
            "Error: no providers selected for model '{}' — nothing to benchmark (the dry-run plan shows why)",
            opts.model
        );
        return 1;
    }
    let mut tasks = tokio::task::JoinSet::new();
    for (idx, &endpoint_idx) in selected.iter().enumerate() {
        let endpoint = snapshot.endpoints.data.endpoints[endpoint_idx].clone();
        let client = client.clone();
        let key = key.clone();
        let model = opts.model.clone();
        let base = base.clone();
        let reasoning_effort = model_config.reasoning_effort_used.clone();
        let ladder = opts.ladder_secs.clone();
        let ctx = std::sync::Arc::clone(&ctx);
        tasks.spawn(async move {
            let stagger =
                Duration::from_millis(u64::try_from(200 * idx).unwrap_or(u64::MAX).min(1000));
            tokio::time::sleep(stagger).await;
            run::run_provider(
                &client,
                &key,
                &endpoint,
                &model,
                &base,
                reasoning_effort.as_deref(),
                &ladder,
                &ctx,
                run_started,
            )
            .await
        });
    }

    // 10. Join with an outer 60-minute guard (the per-provider deadline inside
    //     run_provider is 55 min; the outer guard catches anything stuck).
    let mut provider_runs: Vec<run::ProviderRun> = Vec::new();
    let join = tokio::time::timeout(Duration::from_hours(1), async {
        while let Some(res) = tasks.join_next().await {
            match res {
                Ok(run) => provider_runs.push(run),
                Err(e) => {
                    // A panicked task is exceptional: log and continue — the
                    // report covers the providers that finished.
                    eprintln!("Warning: a provider task panicked: {e}");
                }
            }
        }
    })
    .await;
    if join.is_err() {
        ctx.abort.cancel();
        *ctx.abort_reason.lock().unwrap_poison() = Some("outer 60-minute deadline".to_string());
        while let Some(res) = tasks.join_next().await {
            match res {
                Ok(run) => provider_runs.push(run),
                Err(e) => eprintln!("Warning: a provider task panicked: {e}"),
            }
        }
    }

    // 11. Run-level abort state.
    let abort_reason = ctx.abort_reason.lock().unwrap_poison().clone();
    let aborted = abort_reason.is_some();

    // 12. Artifacts: meta → report/summary/providers/manifest → write.
    let finished_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    let duration_secs = run_started.elapsed().as_secs_f64();
    let exit_code = i32::from(aborted);
    let run_dir = opts
        .output_dir
        .join("bench-openrouter")
        .join(report::run_dir_name(&run_ts));

    let meta = report::RunMeta {
        started_at,
        finished_at,
        duration_secs,
        model: opts.model.clone(),
        key_source: key_source.to_string(),
        cap_usd: opts.cap_usd,
        ladder_secs: opts.ladder_secs.clone(),
        prefix_chars: opts.prefix_chars,
        aborted,
        abort_reason,
        exit_code,
        output_dir: opts.output_dir.clone(),
    };
    let report_json = report::build_report(
        &meta,
        &snapshot.endpoints.data.endpoints,
        &provider_runs,
        &run_dir,
    );
    let summary_md = report::build_summary_md(&meta, &report_json);
    let providers_json = report::providers_snapshot_json(&snapshot);
    let manifest = report::build_manifest(
        &std::env::args().collect::<Vec<_>>(),
        opts,
        &meta.key_source,
        &meta.started_at,
        &meta.finished_at,
        meta.duration_secs,
        meta.exit_code,
        meta.aborted,
        meta.abort_reason.as_deref(),
    );
    let paths = report::ArtifactPaths {
        run_dir: run_dir.clone(),
        report: run_dir.join("report.json"),
        summary: run_dir.join("summary.md"),
        providers: run_dir.join("providers.json"),
        manifest: run_dir.join("manifest.json"),
        latest_dir: opts.output_dir.join("bench-openrouter").join("latest"),
    };
    if let Err(e) = report::write_artifacts(
        &paths,
        &report_json,
        &summary_md,
        &providers_json,
        &manifest,
    ) {
        eprintln!("Error: failed to write benchmark artifacts: {e:#}");
        return 1;
    }

    // 14. Stdout (EPIPE-tolerant).
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(
        out,
        "bench-openrouter: report written to {}",
        run_dir.display()
    );

    // 15. 0 when the artifacts were written; 1 when the run aborted at run
    //     level (the partial report above is already written).
    if meta.aborted {
        eprintln!(
            "Error: benchmark run aborted: {}",
            meta.abort_reason.as_deref().unwrap_or("unknown reason")
        );
    }
    exit_code
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
/// ladder entry plus one — the ladder is a list of inactivity GAPS between
/// ladder rounds, so a 7-entry ladder yields 2 + 8 = 10 requests per provider.
#[must_use]
pub(crate) fn rounds_per_provider(ladder_secs: &[u64]) -> usize {
    WARMUP_ROUNDS + ladder_secs.len() + 1
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
    let healthy_count = decisions.iter().filter(|d| d.is_healthy()).count();
    let allowlist_matches = opts.providers.as_ref().map(|wl| {
        wl.iter()
            .filter(|t| inputs.iter().any(|i| i.endpoint.tag == **t))
            .count()
    });
    let target_count = effective_target_count(healthy_count, allowlist_matches);

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
            // otherwise "selected"/"not selected".
            let selection_reason = decision.reason_text();
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
                "static_price_usd_per_m": crate::bench_openrouter::report::static_price_usd_per_m(
                    ep.pricing.as_ref(),
                ),
                "flags": flags[i],
            })
        })
        .collect()
}

// ── Harness prompt ─────────────────────────────────────────────────

/// The benchmark harness system prompt (loaded from the embedded asset so the
/// "all LLM-sent prompts live under src/prompt/" rule holds by construction).
/// Sent by the full-run executor; the dry run never sends it.
#[must_use]
pub(crate) fn bench_system_prompt() -> String {
    crate::prompt::load_prompt("bench_openrouter.md")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic discovery snapshot (no HTTP, no DBs).
    fn test_snapshot() -> DiscoverySnapshot {
        let models = serde_json::json!({"data": [{
            "id": "acme/model-1",
            "canonical_slug": "acme/model-1",
            "name": "Acme Model 1",
            "context_length": 200_000,
            "reasoning": {"default_effort": "high", "supported_efforts": ["xhigh","high","medium","low"], "mandatory": false},
            "supported_parameters": ["tools", "tool_choice", "reasoning_effort"]
        }]});
        let endpoints = serde_json::json!({"data": {
            "id": "acme/model-1",
            "name": "Acme Model 1",
            "endpoints": [
                {"tag":"acme-a/fp8","name":"Acme A","provider_name":"Acme","status":"0","context_length":200_000,"supports_implicit_caching":true,"quantization":"fp8",
                 "pricing":{"prompt":"0.000002","completion":"0.000008","request":"0","input_cache_read":"0.0000002"}},
                {"tag":"acme-b/fp8","name":"Acme B","provider_name":"Acme","status":"0","context_length":200_000,"supports_implicit_caching":true,"quantization":"fp8",
                 "pricing":{"prompt":"0.000003","completion":"0.000009","request":"0","input_cache_read":"0.0000003"}},
                {"tag":"acme-c/fp8","name":"Acme C","provider_name":"Acme","status":"-10","context_length":200_000,"supports_implicit_caching":false,"quantization":"fp8",
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
            // Default 7-entry ladder (7 gaps between 8 ladder rounds) →
            // rounds = 2 warmup + 8 = 10 requests per provider.
            ladder_secs: vec![0, 5, 30, 120, 300, 600, 1800],
            providers: None,
            prefix_chars: 64_000,
            output_dir: PathBuf::from("/tmp/benchmarks"),
            dry_run: true,
        }
    }

    /// Build the selection pipeline outputs (owned) from a snapshot, so the
    /// test can hold references for the [`PlanData`] lifetime. Uses the real
    /// shared [`build_selection`] so the test exercises the production path.
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
        let bundle = build_selection(opts, snapshot);
        (
            bundle.inputs,
            bundle.flags,
            bundle.decisions,
            bundle.total_est,
            bundle.per_provider_guard,
        )
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

        // request_count == selected × rounds (2 warmup + 8 ladder rounds for
        // the default 7-gap ladder = 10 requests per provider here).
        let selected_count = decisions.iter().filter(|d| d.selected).count();
        assert_eq!(plan["request_count"], (selected_count * rounds) as u64);
        assert_eq!(plan["request_count"], (selected_count * 10) as u64);

        // providers array mirrors every endpoint once.
        let providers = plan["providers"].as_array().expect("providers array");
        assert_eq!(providers.len(), 3);
        // One endpoint is unhealthy → padded in, so all 3 are selected.
        assert_eq!(plan["selection"]["selected_count"], 3);
        assert_eq!(plan["selection"]["padding_count"], 1);
        // Selected providers come first, sorted by static price ascending (the
        // fixture has request=0 for all endpoints, so static-price order ==
        // est-cost order).
        let prices: Vec<f64> = providers
            .iter()
            .take_while(|p| p["selected"] == true)
            .map(|p| {
                p["static_price_usd_per_m"]
                    .as_f64()
                    .expect("selected providers carry a static price")
            })
            .collect();
        assert!(!prices.is_empty(), "selected providers must be priced");
        let mut sorted = prices.clone();
        sorted.sort_by(f64::total_cmp);
        assert_eq!(
            prices, sorted,
            "selected providers must be static-price-ascending"
        );
        // The old est-cost key is gone from the plan.
        assert!(
            providers[0].get("est_cost_usd").is_none(),
            "est_cost_usd must be absent from the plan"
        );

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
        // 7 gaps between 8 ladder rounds → 2 warmup + 8 = 10.
        assert_eq!(rounds_per_provider(&[0, 5, 30, 120, 300, 600, 1800]), 10);
        // 8 gaps → 2 warmup + 9 = 11.
        assert_eq!(
            rounds_per_provider(&[0, 5, 30, 120, 300, 600, 1800, 3600]),
            11
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
        // Non-positive / non-finite caps → Usage.
        for v in ["0", "-1", "nan", "inf"] {
            assert!(
                matches!(
                    BenchOptions::parse(&["--cap-usd".to_string(), v.to_string()]),
                    Err(CliError::Usage(_))
                ),
                "--cap-usd {v} must be rejected"
            );
        }
        assert!(matches!(
            BenchOptions::parse(&["--ladder".to_string(), "0,x".to_string()]),
            Err(CliError::Usage(_))
        ));
    }
}
