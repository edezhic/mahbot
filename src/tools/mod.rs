//! Tool implementations for agent-callable capabilities.

use anyhow::Context;
pub(crate) mod active_models;
pub(crate) mod analyze;
pub(crate) mod automations;
pub mod bun;
pub mod chrome;
pub mod chrome_daemon;
pub(crate) mod chrome_groups;
pub mod chrome_release;
pub(crate) mod chrome_tab_ledger;
pub(crate) mod chrome_tabs;
// Computer-use tool (GUI observe/act); every other platform reaches the stub, never advertised.
pub mod computer;
pub(crate) mod custom;
pub(crate) mod delegation;
pub(crate) mod document;
pub(crate) mod edit;
pub(crate) mod files;
pub(crate) mod grants;
pub(crate) mod hyperframes;
pub(crate) mod image_gen;
pub(crate) mod implement;
pub(crate) mod listing;
pub(crate) mod mahbot_config;
pub(crate) mod mahbot_debug;
pub mod media_catalog;
pub(crate) mod path;
pub(crate) mod rasterize;
pub(crate) mod rasterize_html;
pub(crate) mod read;
pub(crate) mod read_document;
// A guest assistant's report of a capability its user needed and the toolset lacks.
pub(crate) mod report_gap;
pub(crate) mod research;
// One short phrase spoken aloud by the admin's Assistant.
pub(crate) mod say;
pub(crate) mod search;
pub(crate) mod search_archived_tickets;
pub(crate) mod send;
pub(crate) mod shell;
pub(crate) mod sleep;
pub(crate) mod ticket;
pub(crate) mod video_gen;
pub(crate) mod web_search;
pub(crate) mod workspace;

/// Maximum size for edit and search operations and the dashboard editor (10 MB), and the read
/// tool's cap for any read that is not a document container (a container it accepts up to
/// `FILE_MAX_BYTES`, 50 MB). Guards against OOM when agents or the GUI attempt very large files.
pub(crate) const MAX_FILE_SIZE_BYTES: u64 = 10 * 1024 * 1024;

/// Maximum size for a single reference image in bytes. The 2 MB figure is a client-side sanity
/// budget, not a documented OpenRouter number; base64 adds ~33% overhead, so raw image data is
/// capped at 1.5 MB to stay well under. Over-cap references are compressed by the loader. Note: the
/// aggregate body budget re-encodes even under-cap references when the serialized request exceeds
/// it.
const MAX_REFERENCE_IMAGE_BYTES: u64 = 1_500_000;

/// Aggregate budget for the serialized generation request body: the per-image cap does not bound
/// the total, so references are compressed further until the body fits. A client-side sanity
/// budget.
const MAX_REQUEST_BODY_BYTES: usize = 2_000_000;

/// The provider's documented maximum number of reference images in one request (`input_references`
/// allows 16). The live model catalog advertises more for some models, so this also clamps the
/// catalog's declared maximum, and it is the cap when the catalog is unavailable.
pub(crate) const MAX_REFERENCE_IMAGES_PER_REQUEST: usize = 16;

/// The effective reference-image maximum: the catalog's declared value, never above the ceiling.
#[expect(clippy::cast_possible_wrap)] // small compile-time cap
#[must_use]
pub(crate) fn effective_reference_max(catalog_max: i64) -> i64 {
    catalog_max.min(MAX_REFERENCE_IMAGES_PER_REQUEST as i64)
}

/// Argument aliases for file path parameters: `"file"` and `"filename"` are remapped to `"path"` by
/// [`normalize_tool_arguments`] — but not for the document and computer tools, which refuse a stray
/// key.
const PATH_ALIAS_KEYS: &[&str] = &["file", "filename"];

/// Check a file's size against `limit`, naming `label` in the error; the shared gate for reads.
fn check_size_within(meta: &std::fs::Metadata, limit: u64, label: &str) -> anyhow::Result<()> {
    if meta.len() > limit {
        anyhow::bail!("{label}: {} bytes (limit: {limit} bytes)", meta.len());
    }
    Ok(())
}

// The OpenRouter `input_references` array (image_url entries) — the single source of that shape.

fn reference_json(references: &[crate::util::ReferenceImage]) -> serde_json::Value {
    serde_json::json!(
        references
            .iter()
            .map(|r| serde_json::json!({
                "type": "image_url",
                "image_url": { "url": r.data_uri() }
            }))
            .collect::<Vec<_>>()
    )
}

const INPUT_REFERENCES_KEY: &str = "input_references";

/// Ensure the final serialized body fits `max_body_bytes` by compressing references one step at a
/// time, largest first and falling through to the next-largest when a ladder is exhausted: the
/// per-image cap does not bound the total (N refs × cap). The array is re-synced after each step.
pub(crate) fn fit_request_body_budget(
    body: &mut serde_json::Value,
    references: &mut [crate::util::ReferenceImage],
    max_body_bytes: usize,
) -> anyhow::Result<()> {
    if !references.is_empty() {
        body[INPUT_REFERENCES_KEY] = reference_json(references);
    }
    loop {
        if serde_json::to_vec(body)?.len() <= max_body_bytes {
            // The body is final — drop the retained source bytes so they don't sit in memory
            // through the fetch.
            for r in references {
                r.release_source_bytes();
            }
            return Ok(());
        }
        let Some(idx) = references
            .iter()
            .enumerate()
            .filter(|(_, r)| r.has_compression_left())
            .max_by_key(|(_, r)| r.data_uri().len())
            .map(|(i, _)| i)
        else {
            break;
        };
        references[idx].compress_more()?;
        body[INPUT_REFERENCES_KEY] = reference_json(references);
    }
    let body_len = serde_json::to_vec(body)?.len();
    if references.is_empty() {
        anyhow::bail!(
            "Generation request body is too large ({body_len} bytes; limit {max_body_bytes} bytes). \
             Shorten the prompt.",
        );
    }
    let advice = if references.len() > 1 {
        "Reduce the reference image size, the number of references, or the prompt length."
    } else {
        "Reduce the reference image size or shorten the prompt."
    };
    anyhow::bail!(
        "Generation request body is too large ({body_len} bytes after compression; \
         limit {max_body_bytes} bytes). {advice}",
    );
}

pub(crate) use analyze::{AnalyzeDepth, AnalyzeTool, DispatchMode};
pub(crate) use automations::{AddAutomationTool, ListAutomationsTool, RemoveAutomationTool};
pub(crate) use chrome::ChromeTool;
pub(crate) use computer::ComputerTool;
pub(crate) use custom::CustomTool;
pub(crate) use document::DocumentTool;
pub(crate) use edit::EditTool;
pub(crate) use files::FilesTool;
pub(crate) use hyperframes::HyperframesTool;
pub(crate) use image_gen::ImageGenTool;
pub(crate) use mahbot_config::MahbotConfigTool;
pub(crate) use mahbot_debug::MahbotDebugTool;
pub(crate) use rasterize::RasterizeTool;
pub(crate) use read::ReadTool;
pub(crate) use report_gap::ReportGapTool;
pub(crate) use say::SayTool;
pub(crate) use search::SearchTool;
pub(crate) use search_archived_tickets::SearchArchivedTicketsTool;
pub(crate) use send::{Addressee, SendTool};
pub(crate) use shell::{ShellMode, ShellTool};
pub(crate) use sleep::SleepTool;
pub(crate) use ticket::{
    AddCommentTool, CreateTicketTool, GetTicketTool, ListTicketsTool, UpdateTicketTool,
};
pub(crate) use video_gen::VideoGenTool;
pub(crate) use web_search::WebSearchTool;
pub(crate) use workspace::WorkspaceTool;

use crate::agent::message_router::{AgentJob, MessageKind};
use crate::{Tool, Workspace};
use futures_util::FutureExt;
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::util::json::{
    get_bool, get_object, get_opt_bool, get_opt_i64, get_opt_named_str, get_opt_str,
    get_opt_str_strict, get_opt_u64, get_str, get_str_array, get_usize, wrong_type,
};

/// The shared `message` argument of `analyze` and `send`: required, non-empty once trimmed.
pub(crate) fn get_dispatch_message(args: &serde_json::Value) -> anyhow::Result<&str> {
    let message = get_str(args, "message")?;
    anyhow::ensure!(
        !message.trim().is_empty(),
        "usage: the message must not be empty"
    );
    Ok(message)
}

/// Record one corrected argument at debug level, so a reshaped call stays explainable and costs
/// ordinary operation nothing. The note describes the reshaping only, never the payload text.
pub(crate) fn log_normalization_note(tool: &str, note: &str) {
    tracing::debug!(tool, note, "tool arguments were normalized");
}

/// Append a bracketed note as its own paragraph, trimming the text's trailing whitespace first.
pub(crate) fn with_note(text: &str, note: &str) -> String {
    let mut out = text.trim_end().to_string();
    if !out.is_empty() {
        out.push_str("\n\n");
    }
    out.push_str(note);
    out
}

/// Refuse a choice-argument value the holder does not offer, naming what it does. A model can emit
/// a value the declaration never listed, so the choice axes validate rather than merely unadvertise.
pub(crate) fn refuse_choice(argument: &str, value: &str, allowed: &[&str]) -> anyhow::Error {
    anyhow::anyhow!(
        "usage: `{argument}` must be one of {} — got \"{value}\"",
        allowed.join(", ")
    )
}

/// Resolve a choice argument against the holder's kinds, in interface order (narrowest first). An
/// absent or null value takes the narrowest; anything unnamed is refused, never silently narrowed.
pub(crate) fn chosen_kind<T: Copy + PartialEq>(
    args: &serde_json::Value,
    argument: &str,
    held: &[T],
    spelling: impl Fn(T) -> &'static str,
    parse: impl Fn(&str) -> Option<T>,
) -> anyhow::Result<T> {
    debug_assert!(!held.is_empty(), "a holder always has at least one kind");
    match args.get(argument) {
        None | Some(serde_json::Value::Null) => Ok(held[0]),
        Some(serde_json::Value::String(value)) => parse(value)
            .filter(|kind| held.contains(kind))
            .ok_or_else(|| {
                refuse_choice(
                    argument,
                    value,
                    &held.iter().map(|kind| spelling(*kind)).collect::<Vec<_>>(),
                )
            }),
        Some(other) => Err(wrong_type(argument, "a string", other)),
    }
}

/// Build a JSON schema for tool parameters: `properties` in the standard object envelope, with
/// `"required"` added only for a non-empty slice. Tools with non-standard top-level keys should not
/// use this directly, though they may use it for inner entries.
#[must_use]
pub(crate) fn tool_params_schema(
    properties: &serde_json::Value,
    required: &[&str],
) -> serde_json::Value {
    let mut schema = serde_json::json!({
        "type": "object",
        "properties": properties,
    });
    if !required.is_empty() {
        schema["required"] = serde_json::json!(required);
    }
    schema
}

/// Build one `oneOf` action entry: the name wrapped in the object envelope with
/// `"additionalProperties": false` that the chrome and computer action schemas use.
#[must_use]
pub(crate) fn action_entry_schema(
    name: &str,
    description: &str,
    required: &[&str],
    properties: &serde_json::Value,
) -> serde_json::Value {
    let mut inner = tool_params_schema(properties, required);
    inner["additionalProperties"] = serde_json::json!(false);

    serde_json::json!({
        "type": "object",
        "properties": {
            (name): inner
        },
        "required": [name],
        "additionalProperties": false,
        "description": description
    })
}

/// Prefix of a failed tool result — shared so the research wrap-up checks the marker this produces.
pub(crate) const TOOL_FAILURE_MARKER: &str = "Tool call failed.";

/// Byte cap for the argument preview, so a multi-byte argument cannot eat the reason's budget.
const TOOL_FAILURE_ARGS_PREVIEW_BYTES: usize = 1_000;

/// What the caller wraps a tool's own fault text with before it reaches
/// [`format_tool_failure_feedback`] — see [`tool_failure_reason`].
const TOOL_FAILURE_REASON_PREFIX: &str = "Error executing ";

/// The wrapper a fault is reported in, so a tool and the wrapper carrying its text agree on one
/// shape.
#[must_use]
pub(crate) fn tool_failure_reason(tool_name: &str, fault: &str) -> String {
    format!("{TOOL_FAILURE_REASON_PREFIX}{tool_name}: {fault}")
}

/// The room a fault may NOT take inside the delivered result: the wrapper plus everything the
/// formatter spends — the marker, the labels, a tool name of `tool_name_len` bytes, the capped
/// preview.
pub(crate) const fn tool_failure_envelope_bytes(tool_name_len: usize) -> usize {
    TOOL_FAILURE_REASON_PREFIX.len()
        + tool_name_len
        + ": ".len()
        + TOOL_FAILURE_MARKER.len()
        + "\ntool: \narguments: \nreason:\n".len()
        + tool_name_len
        + TOOL_FAILURE_ARGS_PREVIEW_BYTES
}

#[must_use]
pub(crate) fn format_tool_failure_feedback(
    tool_name: &str,
    tool_args: &serde_json::Value,
    reason: &str,
) -> String {
    let args = tool_args.to_string();
    let args_preview = crate::util::cut_bytes(&args, TOOL_FAILURE_ARGS_PREVIEW_BYTES);
    format!(
        "{TOOL_FAILURE_MARKER}\n\
         tool: {tool_name}\n\
         arguments: {args_preview}\n\
         reason:\n{reason}"
    )
}

/// The canonical `internal` error for a mahbot-side fault the model cannot recover from: no
/// argument change or retry can fix it — a user or engineer must investigate the service.
#[must_use]
pub(crate) fn internal_fault(cause: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "internal: {cause} (mahbot-side fault) — \
         hint: no model action can recover this; a user or engineer must \
         investigate the mahbot service"
    )
}

/// Run one durable round's future and catch a panic in it, logging it against `label` and returning
/// its message so the caller decides what a panicked round leaves behind. A panicking round never
/// takes the instance with it: `panic = "unwind"` is load-bearing here.
pub(crate) async fn caught_round<T>(
    label: &str,
    round: impl std::future::Future<Output = T>,
) -> Result<T, String> {
    // Boxed before it is caught: the future is far larger than `clippy::large_futures` tolerates.
    let round = Box::pin(round);
    match std::panic::AssertUnwindSafe(round).catch_unwind().await {
        Ok(value) => Ok(value),
        Err(panic) => {
            let panic = crate::util::panic_message(&*panic);
            tracing::error!(panic = %panic, "{label} panicked");
            Err(panic)
        }
    }
}

/// Gives back a synthetic `dir-…` exec engine on drop, so a round that unwinds releases it exactly
/// as one that returns does. A managed workspace's own engine is never touched.
struct ExecEngineGuard<'a>(&'a crate::Workspace);

impl Drop for ExecEngineGuard<'_> {
    fn drop(&mut self) {
        crate::tools::delegation::release_exec_engine(self.0);
    }
}

/// Carrier for a tool call whose durable work was drain-cut. NEVER persisted: the frame already
/// recorded the call, so the result is simply absent and the resume step re-runs it. Carried as an
/// `anyhow::Error` out of a sync dispatch so the caller can mark the outcome `suspended`.
#[derive(Debug)]
pub(crate) struct CallSuspended;

impl std::fmt::Display for CallSuspended {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tool call suspended by shutdown/drain — durable work left launched for deterministic resume"
        )
    }
}

impl std::error::Error for CallSuspended {}

// The job-scoped parameters every durable sync core takes. Beyond `(ws, task)` these are the values
// the sync wrappers read from the task-locals and job row, so the core never touches
// `CURRENT_TOOL_*` itself.

pub(crate) struct CoreJobArgs<'a> {
    pub job_id: &'a str,
    pub caller_role: crate::Role,
    pub user_name: &'a str,
    pub channel: &'a str,
    /// The fresh-spawn row record — the resume discriminator: `Some` spawns, `None` resumes.
    pub spawn: Option<SpawnRecord<'a>>,
    /// Caller session pin persisted at fresh spawn so the resume-completion step can find
    /// caller-owned launched jobs. Always `None` on resume — the row already carries it.
    pub caller_agent_id: Option<&'a str>,
    /// A checkpoint DB error fails a sync call so it is retried; async and boot-resume warn and go
    /// on.
    pub fail_on_checkpoint_error: bool,
}

/// The jobs-row fields a fresh dispatch writes — read ONLY when the call actually spawns, so a
/// resume can never hand the chosen directory's name to the delivery key.
pub(crate) struct SpawnRecord<'a> {
    /// The CALLER's own workspace name: the durable delivery/routing key
    /// (`jobs.workspace_name`). Never the chosen directory.
    pub delivery_name: &'a str,
    /// The chosen directory the round executes in, when the caller asked for one (including its
    /// own), so the reply names the directory really used and a restart resumes into the same
    /// identity.
    pub exec_dir: Option<&'a str>,
    /// The CALLER's custom-role launch, persisted as `jobs.custom_launch` so a boot resume rebuilds
    /// the envelope addressed to the custom branch. The WHOLE launch travels: a resumed turn opens
    /// its branch from it, so the prompt and tool subset are as load-bearing as the branch address.
    pub custom_launch: Option<&'a crate::Launch>,
}

/// The single outcome form of the durable sync cores: `Terminal` wraps a produced result,
/// `DrainCut` means the round was cut by drain/shutdown — the job stays `launched` for resume and
/// nothing is routed — and `Cancelled` means the round observed its cancel signal and stopped.
pub(crate) enum SyncCoreOutcome {
    Terminal(Result<String, anyhow::Error>),
    DrainCut,
    /// The round observed its cancel signal: nothing is produced, and its rows are the cancel
    /// path's.
    Cancelled,
}

/// The sync analyze/worker durable cores, behind one interface so the sync dispatch, the
/// resume-completion loop and the per-tool wrappers share one body.
#[derive(Copy, Clone)]
pub(crate) enum SyncDurableCore {
    Analyze,
    Worker,
}

impl SyncDurableCore {
    #[must_use]
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Analyze => "Analyze",
            Self::Worker => "Worker",
        }
    }

    pub(crate) fn kind_str(self) -> &'static str {
        match self {
            Self::Analyze => crate::jobs::SpawnChild::Analyze.kind_str(),
            Self::Worker => crate::jobs::SpawnChild::Implement.kind_str(),
        }
    }

    /// Resolve a dangling tool call to the core that settles it, from the tool plus its choice
    /// argument — never by matching a stored round kind. A call that asked for another kind, an
    /// unknown value, a non-string or an unknown (or retired) tool name resolves to `None` and is
    /// re-executed instead.
    #[must_use]
    pub(crate) fn from_call(name: &str, args: &serde_json::Value) -> Option<Self> {
        // A cancel call names an existing round instead of starting one: a
        // dangling cancel frame is re-executed, never resolved to a dispatch.
        if name == "analyze"
            && args
                .get(crate::tools::analyze::ACTION_ARG)
                .is_some_and(|v| !v.is_null())
        {
            return None;
        }
        // Each tool's sync form is its narrowest choice: the quick depth, the worker addressee. An
        // absent or null choice is that kind — how a frame predating the merge reconnects to its
        // unfinished round.
        let (core, argument, own_choice) = match name {
            "analyze" => (
                Self::Analyze,
                AnalyzeDepth::ARG,
                AnalyzeDepth::Quick.as_str(),
            ),
            "send" => (Self::Worker, Addressee::ARG, Addressee::Worker.as_str()),
            _ => return None,
        };
        match args.get(argument) {
            None | Some(serde_json::Value::Null) => {}
            Some(serde_json::Value::String(value)) if value == own_choice => {}
            Some(_) => return None,
        }
        Some(core)
    }

    /// Run the durable core — both per-tool bodies return [`SyncCoreOutcome`] directly.
    pub(crate) async fn run(
        self,
        ws: &crate::Workspace,
        task: &str,
        args: CoreJobArgs<'_>,
    ) -> anyhow::Result<SyncCoreOutcome> {
        match self {
            Self::Analyze => crate::tools::analyze::run_analyze_with_job(ws, task, args).await,
            Self::Worker => crate::tools::implement::run_implement_with_job(ws, task, args).await,
        }
    }

    /// Abandon this dispatch's own row after the call already failed: the failure IS the result, so
    /// the model repeats the call and a leftover row would only be bindable by a later, unrelated
    /// call of the same core. Never an alias: the call is repeated, not reconnected. A drain-cut
    /// keeps the row.
    async fn abandon_failed_dispatch(conn: &crate::db::Connection, job_id: &str, label: &str) {
        if let Err(e) = crate::jobs::terminalize_job(conn, job_id).await {
            tracing::warn!(
                job = %job_id,
                error = %e,
                "Failed to abandon the failed {label} dispatch's row",
            );
        }
    }

    /// The cancel-signal guard for the one core the cancel action can stop, `None` for the rest.
    fn cancel_guard(
        self,
        job_id: &str,
        caller: Option<&str>,
    ) -> Option<crate::round_cancel::CancelGuard> {
        match self {
            Self::Analyze => Some(crate::round_cancel::register(job_id, caller)),
            Self::Worker => None,
        }
    }

    /// Whether the cancel signal fired — the delivery path's last look before the result is handed
    /// over.
    fn stopped_by_cancel(self, job_id: &str) -> bool {
        if !crate::round_cancel::is_cancelled(job_id) {
            return false;
        }
        tracing::info!(job = %job_id, "{} round stopped by cancel — nothing delivered", self.label());
        true
    }

    /// Shared sync dispatch: read the task-locals, generate a fresh job id, run the core as a fresh
    /// spawn (it writes the jobs row), then terminalize or surface [`CallSuspended`] on a
    /// drain-cut. Every outcome but a drain-cut abandons the row with the call's own result.
    pub(crate) async fn run_sync_dispatch(
        self,
        ws: &crate::Workspace,
        task: &str,
        caller_role: crate::Role,
    ) -> anyhow::Result<String> {
        let caller_agent_id = crate::agent::CURRENT_TOOL_AGENT_ID
            .try_with(Clone::clone)
            .unwrap_or(None);
        let user_name = crate::agent::tool_user_name();
        let channel = crate::agent::tool_channel();
        let conn = &crate::session::store().conn;

        // Never binds to prior jobs: a call re-issued after a settled resume spawns fresh.

        let job_id = crate::generate_id();
        let custom = crate::agent::tool_custom_launch();
        let args = CoreJobArgs {
            job_id: &job_id,
            caller_role: caller_role.for_launch(custom.as_ref()),
            user_name: &user_name,
            channel: &channel,
            spawn: Some(SpawnRecord {
                delivery_name: &ws.name,
                exec_dir: None,
                custom_launch: custom.as_ref(),
            }),
            caller_agent_id: caller_agent_id.as_deref(),
            fail_on_checkpoint_error: caller_agent_id.is_some(),
        };
        let outcome = match self.run(ws, task, args).await {
            Ok(outcome) => outcome,
            Err(e) => {
                Self::abandon_failed_dispatch(conn, &job_id, self.label()).await;
                return Err(e);
            }
        };
        match outcome {
            SyncCoreOutcome::Terminal(Ok(text)) => {
                crate::jobs::terminalize_job(conn, &job_id).await?;
                Ok(text)
            }
            SyncCoreOutcome::Terminal(Err(e)) => {
                Self::abandon_failed_dispatch(conn, &job_id, self.label()).await;
                Err(e)
            }
            SyncCoreOutcome::DrainCut => {
                tracing::info!(
                    job = %job_id,
                    "{} round cut short by drain — job stays launched for deterministic resume",
                    self.label(),
                );
                Err(anyhow::Error::new(crate::tools::CallSuspended))
            }
            // Unreachable: only the async entry points register a cancel signal, and whatever
            // cancels a round removes its own rows, so there is nothing to clean up here.
            SyncCoreOutcome::Cancelled => Err(crate::tools::internal_fault(
                "a sync round observed a cancel signal it never registered",
            )),
        }
    }

    /// Shared sync resume: run the resume preamble, then the core as a resume (`spawn: None` — it
    /// reuses the stored roster and outcomes and inserts no row), mapping the outcome to
    /// [`crate::jobs::SyncResumeOutcome`]. `sync_resume` selects checkpoint-error propagation: a
    /// live sync resume passes `true` (fail the round), the async boot-resume rounds `false` (warn
    /// and go on).
    pub(crate) async fn resume_sync_core(
        self,
        ws: &crate::Workspace,
        job_id: &str,
        sync_resume: bool,
    ) -> anyhow::Result<crate::jobs::SyncResumeOutcome> {
        let label = self.label();
        let (caller, caller_role) = match crate::jobs::resume_job_preamble_discrete(
            &crate::session::store().conn,
            job_id,
            &format!("{label} resume"),
            &format!("{label} resume"),
        )
        .await
        {
            crate::jobs::ResumePreamble::Proceed(caller, caller_role) => (caller, caller_role),
            crate::jobs::ResumePreamble::DrainCut => {
                return Ok(crate::jobs::SyncResumeOutcome::DrainCut);
            }
            crate::jobs::ResumePreamble::Gone => {
                return Ok(crate::jobs::SyncResumeOutcome::Gone);
            }
            // An unreadable row is an infra failure, not an abandon: fail the round so the
            // caller-driven path never surfaces an "explicitly abandoned" note for a live job (the
            // next cycle retries).
            crate::jobs::ResumePreamble::Unreadable(e) => {
                return Err(e.context(format!("{label} resume: job row unreadable")));
            }
        };
        let args = CoreJobArgs {
            job_id,
            caller_role,
            user_name: &caller.user_name,
            channel: &caller.channel,
            spawn: None,
            caller_agent_id: None,
            fail_on_checkpoint_error: sync_resume,
        };
        let run = self.run(ws, &caller.task, args).await;
        match run {
            Ok(SyncCoreOutcome::Terminal(result)) => Ok(crate::jobs::SyncResumeOutcome::Terminal(
                caller_role,
                caller,
                result,
            )),
            Ok(SyncCoreOutcome::DrainCut) => {
                tracing::info!(
                    job = %job_id,
                    "{label} resume drain-cut — outcome checkpointed, job stays for next boot",
                );
                Ok(crate::jobs::SyncResumeOutcome::DrainCut)
            }
            // A cancel landed during this resume: the round stopped and its rows are the cancel
            // action's.
            Ok(SyncCoreOutcome::Cancelled) => Ok(crate::jobs::SyncResumeOutcome::Gone),
            Err(e) => Err(e),
        }
    }

    #[must_use]
    pub(crate) fn message_kind(self) -> MessageKind {
        match self {
            Self::Analyze => MessageKind::AnalyzeToolResult,
            Self::Worker => MessageKind::ImplementResult,
        }
    }

    /// Build the async result envelope for this core's round results. The analyze envelope keeps
    /// the caller-generated `job_id` as its first body line so the result is correlatable with the
    /// ack placeholder; the worker's is the `<worker-report>` with the folder the worker worked in.
    #[must_use]
    pub(crate) fn build_async_message(
        self,
        job_id: &str,
        result: &anyhow::Result<String>,
        exec_dir: Option<&str>,
        delivery: &crate::Workspace,
    ) -> String {
        match self {
            Self::Analyze => {
                build_async_result_envelope(job_id, result, "analyze-tool-result", exec_dir)
            }
            Self::Worker => build_worker_envelope(
                "worker_report.md",
                exec_dir.unwrap_or(&delivery.path),
                None,
                &worker_report_text(result),
            ),
        }
    }

    /// Durable async dispatch (SPAWN → run → CHECKPOINT → COMPLETE): run the core as a fresh spawn
    /// with no session pin, then terminalize into the durable envelope under the caller-generated
    /// `job_id`, which `execute()` creates BEFORE the spawn so ack and delivery share one id.
    /// Returns `None` when the round was cut by drain/shutdown (the job stays `launched` for boot
    /// resume; nothing is routed) or observed a cancel (the cancel action owns its rows).
    /// `delivery` is the caller's own workspace, `place` the directory the round executes in,
    /// `custom` the calling custom role's launch.
    #[expect(clippy::too_many_arguments)]
    pub(crate) async fn dispatch_durable(
        self,
        delivery: &crate::Workspace,
        place: Option<&crate::Workspace>,
        task: &str,
        caller_role: crate::Role,
        user_name: &str,
        channel: &str,
        job_id: &str,
        custom: Option<crate::Launch>,
    ) -> Option<AgentJob> {
        let exec = place.unwrap_or(delivery);
        let exec_dir = place.map(|p| p.path.as_str());
        let caller_role = caller_role.for_launch(custom.as_ref());
        let result = match self
            .run(
                exec,
                task,
                CoreJobArgs {
                    job_id,
                    caller_role,
                    user_name,
                    channel,
                    spawn: Some(SpawnRecord {
                        delivery_name: &delivery.name,
                        exec_dir,
                        custom_launch: custom.as_ref(),
                    }),
                    caller_agent_id: None,
                    fail_on_checkpoint_error: false,
                },
            )
            .await
        {
            Ok(SyncCoreOutcome::DrainCut) => {
                // Drain-cut: the outcomes are checkpointed — leave the job `launched` for boot
                // resume. A spurious envelope here would discard them.
                tracing::info!(
                    job = %job_id,
                    "{} round cut short by drain — job stays launched for boot resume",
                    self.label(),
                );
                return None;
            }
            Ok(SyncCoreOutcome::Cancelled) => {
                // Cancelled: the cancel action answered the caller and owns the rows, so nothing is
                // completed, routed or removed — the row write itself is gated in-tx against the
                // cancel signal. Unlike the drain-cut arm, the synthetic `dir-…` engine goes.
                tracing::info!(
                    job = %job_id,
                    "{} round stopped by cancel — not completed, not routed",
                    self.label(),
                );
                if let Some(place) = place {
                    crate::tools::delegation::release_exec_engine(place);
                }
                return None;
            }
            Ok(SyncCoreOutcome::Terminal(result)) => result,
            Err(e) => Err(e),
        };
        // The round is over: drop the synthetic `dir-…` engine it may have created (a managed
        // workspace's own is left alone). The drain-cut path above skips this — that job resumes in
        // the same directory.
        if let Some(place) = place {
            crate::tools::delegation::release_exec_engine(place);
        }
        Some(
            crate::jobs::complete_durable_job(
                job_id,
                self.build_async_message(job_id, &result, exec_dir, delivery),
                self.message_kind(),
                caller_role,
                user_name,
                channel,
                &delivery.name,
                custom,
            )
            .await,
        )
    }

    /// Boot-resume a durable round: resolve the execution workspace (the recorded directory when
    /// there is one, else the caller's own), run the core as a resume through
    /// [`Self::resume_sync_core`], then terminalize into a durable envelope and route it to the
    /// ORIGINAL caller (role/user/channel persisted at spawn), never the Manager; `delivery` stays
    /// the caller's own. A panicking round is caught and reported through the same arm as a round
    /// that produced no outcome. Drain aborts quietly (the row stays for the next boot); a cancel
    /// discards the round's own result.
    pub(crate) async fn resume_durable_round(
        self,
        job_id: &str,
        delivery: &crate::Workspace,
        exec_dir: Option<&str>,
    ) {
        let label = self.label();
        // A resume registers its cancel signal too: its rows exist, so
        // ownership is the row's and no caller pin is needed.
        let _cancel_guard = self.cancel_guard(job_id, None);
        // A cancel can land between the boot scan and the registration above; its mark reaches that
        // registration, so the signal is already fired here — return without running the round.
        if self.stopped_by_cancel(job_id) {
            return;
        }
        // The round's own future is caught: a panic in it is a round that produced no outcome, and
        // it is reported as one rather than left as a launched row nobody is told about.
        let resumed = caught_round(&format!("{label} resume"), async {
            match exec_dir {
                None => self.resume_sync_core(delivery, job_id, false).await,
                Some(dir) => match crate::tools::delegation::workspace_at_dir(dir).await {
                    Ok(exec) => {
                        // Released however this ends: a directory's engine is registered for the
                        // process lifetime, which a panic must not leave behind. Whatever needs it
                        // next re-creates it on demand.
                        let _engine = ExecEngineGuard(&exec);
                        self.resume_sync_core(&exec, job_id, false).await
                    }
                    // The recorded directory is gone: the round ends through the resume-failure arm
                    // below.
                    Err(e) => Err(e),
                },
            }
        })
        .await
        .unwrap_or_else(|panic| Err(anyhow::anyhow!("the round panicked: {panic}")));
        let (caller_role, caller, custom, result) = match resumed {
            Ok(crate::jobs::SyncResumeOutcome::Terminal(caller_role, caller, result)) => {
                let custom = caller.custom_launch.as_deref().cloned();
                (caller_role, caller, custom, result)
            }
            // Drain-cut mid-resume or job row gone: abort quietly — the core logs the reason.
            Ok(crate::jobs::SyncResumeOutcome::DrainCut | crate::jobs::SyncResumeOutcome::Gone) => {
                return;
            }
            // Resume infra failure, an unusable recorded directory, or a panicked round: deliver an
            // error envelope to the ORIGINAL caller, like any round-level failure (the row is still
            // there to re-load it from).
            Err(e) => {
                let Some((caller, caller_role)) = crate::jobs::resume_job_preamble(
                    &crate::session::store().conn,
                    job_id,
                    &format!("{label} resume"),
                    &format!("{label} resume"),
                )
                .await
                else {
                    return;
                };
                let custom = caller.custom_launch.as_deref().cloned();
                let envelope = crate::jobs::complete_durable_job(
                    job_id,
                    self.build_async_message(job_id, &Err(e), exec_dir, delivery),
                    self.message_kind(),
                    caller_role,
                    &caller.user_name,
                    &caller.channel,
                    &delivery.name,
                    custom,
                )
                .await;
                // Last look before the route, mirroring `spawn_dispatch`: a cancel landing
                // during this resume discards its result.
                if self.stopped_by_cancel(job_id) {
                    return;
                }
                crate::agent::message_router::route(
                    &crate::jobs::envelope_target(&envelope),
                    envelope,
                );
                return;
            }
        };
        // Drain/shutdown fired after the round returned: abort quietly without routing or deleting
        // the row — the outcomes are checkpointed, and routing a partial result here would race the
        // exit.
        if crate::shutdown::aborting() {
            tracing::info!(
                job = %job_id,
                "{label} resume aborted after the round completed — job stays for next boot",
            );
            return;
        }
        let envelope = crate::jobs::complete_durable_job(
            job_id,
            self.build_async_message(job_id, &result, exec_dir, delivery),
            self.message_kind(),
            caller_role,
            &caller.user_name,
            &caller.channel,
            &delivery.name,
            custom,
        )
        .await;
        // A cancel landed while this resume ran: its result is never delivered —
        // the last look before the route, mirroring `spawn_dispatch`.
        if self.stopped_by_cancel(job_id) {
            return;
        }
        crate::agent::message_router::route(&crate::jobs::envelope_target(&envelope), envelope);
    }

    /// Async dispatch entry of `analyze`'s quick depth and `send`'s worker addressee: read the
    /// caller identity from the `CURRENT_TOOL_*` task-locals, spawn the core's durable dispatch in
    /// the background, catch its panics, and route the resulting envelope to the caller's agent
    /// channel. The `job_id` comes from `execute()` before this spawn, so ack and delivery share
    /// one id. In order: `tokio::spawn` + `catch_unwind`, so a panic can never leave the caller
    /// waiting (that path rebuilds an error envelope); the dispatch builds the success envelope
    /// itself and it is routed as-is; `Ok(None)` (drain-cut or cancel) routes nothing; the
    /// `aborting()` guard skips routing into a consumer that stopped pulling; the cancel signal is
    /// consulted right before the route. `src/tools/research.rs` deliberately does not share this
    /// scaffold (it omits the pre-route guard).
    pub(crate) fn spawn_dispatch(
        self,
        delivery: &crate::Workspace,
        place: Option<&crate::Workspace>,
        task: &str,
        caller_role: crate::Role,
        job_id: String,
    ) {
        let user_name = crate::agent::tool_user_name();
        let channel = crate::agent::tool_channel();
        // Read synchronously before the spawn (task-local contract): the round's envelope and job
        // row must carry the caller's branch for a custom role, the default construction role
        // otherwise.
        let custom = crate::agent::tool_custom_launch();
        // The dispatching session's agent id, read synchronously: the cancel-signal entry records
        // it, so a cancel naming this id can prove ownership before the `jobs` row exists.
        let caller = crate::agent::CURRENT_TOOL_AGENT_ID
            .try_with(Clone::clone)
            .unwrap_or(None);
        let cancel_guard = self.cancel_guard(&job_id, caller.as_deref());
        let (delivery, task) = (delivery.clone(), task.to_string());
        let place = place.cloned();
        let ws_name = delivery.name.clone();
        tokio::spawn(async move {
            let _cancel_guard = cancel_guard;
            let exec_dir = place.as_ref().map(|p| p.path.as_str());
            let round = caught_round(&format!("{} round dispatch", self.label()), async {
                self.dispatch_durable(
                    &delivery,
                    place.as_ref(),
                    &task,
                    caller_role,
                    &user_name,
                    &channel,
                    &job_id,
                    custom.clone(),
                )
                .await
            })
            .await;
            let envelope = match round {
                Ok(Some(envelope)) => envelope,
                Ok(None) => {
                    return;
                }
                Err(panic) => {
                    // The round is over (as a panic): drop the synthetic
                    // `dir-…` engine it may have created before the envelope.
                    if let Some(exec) = place.as_ref() {
                        crate::tools::delegation::release_exec_engine(exec);
                    }
                    // A custom-role round owns a custom branch address: derive
                    // the envelope's role the same way `dispatch_durable` does.
                    let role = caller_role.for_launch(custom.as_ref());
                    AgentJob {
                        content: self.build_async_message(
                            &job_id,
                            &Err(anyhow::anyhow!("dispatch panicked: {panic}")),
                            exec_dir,
                            &delivery,
                        ),
                        workspace_name: ws_name,
                        user_name,
                        channel,
                        kind: self.message_kind(),
                        role,
                        reply_target: None,
                        pending_job_id: Some(job_id.clone()),
                        originating_workspace: None,
                        custom_role: custom,
                        reply_to_custom: None,
                        urgent: false,
                    }
                }
            };
            if self.stopped_by_cancel(&job_id) {
                return;
            }
            if crate::shutdown::aborting() {
                return;
            }
            crate::agent::message_router::route(&crate::jobs::envelope_target(&envelope), envelope);
        });
    }
}

/// Wrap a sub-agent/tool result in the async `<tag>` envelope. The `job_id` is the first body line
/// so the result is correlatable with the ack placeholder, and a `Directory:` line follows when the
/// round ran in a chosen directory. Failures carry an explicit marker instead of an empty result.
pub(crate) fn build_async_result_envelope(
    job_id: &str,
    result: &anyhow::Result<String>,
    tag: &str,
    exec_dir: Option<&str>,
) -> String {
    // A managed workspace's stored path carries a trailing separator; the
    // reply names the directory in its bare spelling either way.
    let dir_line = exec_dir.map_or_else(String::new, |dir| {
        format!(
            "\n\nDirectory: {}",
            dir.trim_end_matches(std::path::MAIN_SEPARATOR)
        )
    });
    match result {
        Ok(text) => format!("<{tag}>\n\nJob: {job_id}{dir_line}\n\n{text}</{tag}>"),
        Err(e) => {
            tracing::debug!(error = %e, %tag, "async tool result failed");
            format!("<{tag}>\n\nJob: {job_id}{dir_line}\n\nAn error occurred: {e}</{tag}>")
        }
    }
}

/// A worker's report text — the round's own text, or an explicit failure marker
/// so a failed round is never silently dropped.
fn worker_report_text(result: &anyhow::Result<String>) -> String {
    match result {
        Ok(text) => text.clone(),
        Err(e) => {
            tracing::debug!(error = %e, "worker report failed");
            format!("An error occurred: {e}")
        }
    }
}

/// Render a branch-report asset (`worker_report.md` / `worker_failure.md`): one `<worker-report>`
/// envelope for both, with the folder it happened in and, for a named branch, its `branch`
/// attribute.
pub(crate) fn build_worker_envelope(
    asset: &str,
    folder: &str,
    branch: Option<&str>,
    message: &str,
) -> String {
    crate::prompt::substitute(
        &crate::prompt::load_prompt(asset),
        &[
            (
                "{{folder}}",
                envelope_attr(folder.trim_end_matches(std::path::MAIN_SEPARATOR)).as_str(),
            ),
            ("{{branch_attr}}", branch_attr(branch).as_str()),
            ("{{message}}", message),
        ],
    )
}

/// The `branch="id"` attribute a report envelope carries for a named branch, empty otherwise.
pub(crate) fn branch_attr(identifier: Option<&str>) -> String {
    identifier.map_or_else(String::new, |id| {
        format!(" branch=\"{}\"", envelope_attr(id))
    })
}

/// An attribute value interpolated into an envelope. The value is usually a folder path the model
/// chose, so ONLY the three characters that could end the attribute or forge an envelope are
/// replaced: escaping `&` or `'` too would distort an ordinary name (an `o'brien` would name a
/// folder that does not exist).
pub(crate) fn envelope_attr(value: &str) -> String {
    value
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// A tool call's textual result together with the images it produced — what
/// [`crate::Tool::execute_with_payloads`] returns.
#[derive(Debug, Default)]
pub(crate) struct ToolOutput {
    pub text: String,
    /// Images produced by this call, injected by the agent loop as native
    /// image parts after the round's tool-result block.
    pub image_payloads: Vec<ImagePayload>,
    /// Whether `text` is real content the model must keep, rather than a claim ABOUT the images
    /// this call produced: with content, image annotations are APPENDED and any `[IMAGE:path]`
    /// marker is literal text; without it, the annotations REPLACE `text` and a marker may become
    /// an injected image. `true` for the read tool's converted document, the computer tool and the
    /// picture tool's batch report.
    pub text_is_content: bool,
    /// Whether `text` must reach the model whole: a media read is a timeline, not lines, and the
    /// text budget would cut it mid-clip. The call knows its answer is a timeline; the tool's
    /// [`Tool::format_output`] sees only the answer.
    pub preserve_text: bool,
}

/// Pair a textual result with the image payload of [`Tool::image_payload`] — the tail of the
/// default `execute_with_payloads`, shared with the read tool's override so the two cannot drift.
/// The text is flagged as a claim about the image whether or not a payload comes back: the read
/// tool names the file it read. Tools whose text IS the result bypass this helper.
pub(crate) async fn with_image_payload<T: Tool + ?Sized>(
    tool: &T,
    text: String,
    ws: &Workspace,
    args: &serde_json::Value,
) -> ToolOutput {
    ToolOutput {
        text,
        image_payloads: tool.image_payload(ws, args).await.into_iter().collect(),
        text_is_content: false,
        preserve_text: false,
    }
}

#[expect(clippy::struct_excessive_bools)] // independent per-call flags, not states
#[derive(Debug)]
pub(crate) struct ToolExecutionOutcome {
    pub output: String,
    pub success: bool,
    pub image_payloads: Vec<ImagePayload>,
    pub text_is_content: bool,
    pub preserve_text: bool,
    /// `true` when the tool's durable work was drain-cut and its result is intentionally absent:
    /// the commit skips such outcomes, so the call stays dangling until the resume-completion
    /// settles it.
    pub suspended: bool,
    /// `true` when a successful outcome should end the run gracefully after the round's results
    /// commit.
    pub ends_turn: bool,
}

/// Discriminates the tool that produced an [`ImagePayload`], so the loop labels the injected image
/// with the right opening phrase without knowing which tool produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImagePayloadSource {
    Read,
    /// Tool-produced image (image_gen, a rasterized page, an extracted document), or one derived
    /// from a marker.
    Generated,
    /// Screenshot captured by the chrome tool, injected as a native image for the agent's own
    /// analysis.
    Chrome,
    /// Screen capture from the computer tool, injected as a native image for the agent's own
    /// analysis.
    Computer,
    /// A video's timeline as pictures of four frames each.
    Grid,
}

impl ImagePayloadSource {
    /// Opening phrase of the injected-image annotation, e.g. "Read image file" for a raster read.
    #[must_use]
    pub(crate) fn opener(self) -> &'static str {
        match self {
            Self::Read => "Read image file",
            Self::Generated => "Generated image file",
            Self::Chrome => "Chrome screenshot",
            Self::Computer => "Computer screenshot",
            Self::Grid => "Video frame grid",
        }
    }
}

/// Per-call image payload: read from disk, generated or captured by a tool, or derived by the loop
/// from a real `[IMAGE:path]` marker in a successful output. The loop injects `[IMAGE:{data_uri}]`
/// as a synthetic persisted User-role message after the round. `picture_size` is the figure the
/// annotation states for the picture; `copy_size` describes the JPEG the model is given.
#[derive(Debug, Clone)]
pub(crate) struct ImagePayload {
    /// Canonical path of the image, surfaced in the annotation and the debug log. The dedup key is
    /// `data_uri`.
    pub path: String,
    /// Bounded JPEG data-URI (`data:image/jpeg;base64,...`).
    pub data_uri: String,
    /// The size the naming sentence states: the file's own pixels (post-EXIF), or — for a payload
    /// that is not a picture of that file (a video's frame grid) — the payload's own.
    pub picture_size: (u32, u32),
    /// The attached copy's pixels (post-EXIF/post-resize); when it is smaller than `picture_size`
    /// the annotation states it separately as the model's copy.
    pub copy_size: (u32, u32),
    pub format: String,
    /// `Some` when the read resolved a typo'd path to a unique filename match; kept in the
    /// annotation.
    pub recovery_note: Option<String>,
    /// An extra line the annotation carries, for a tool-specific fact that must reach the assistant
    /// with the image itself.
    pub note: Option<String>,
    /// Tool that produced the image, selecting the annotation opener
    /// ("Read image file", "Generated image file", "Chrome screenshot").
    pub source: ImagePayloadSource,
}

/// Prefix `body` with a read's typo-recovery note — the one place `[Recovered path: ...]` is put in
/// front of an answer, so a text read, a raster annotation and a document answer word it
/// identically.
#[must_use]
pub(crate) fn with_recovery_note(note: Option<&str>, body: String) -> String {
    match note {
        Some(note) => format!("{note}\n{body}"),
        None => body,
    }
}

impl ImagePayload {
    /// Tool-result annotation for a FRESH attachment (injected as a new synthetic user message).
    #[must_use]
    pub(crate) fn attached_annotation(&self) -> String {
        self.annotation("Image content attached to the conversation as a native image.")
    }

    /// Tool-result annotation for a DEDUPED repeat read: the image is in the conversation already
    /// and is not attached again. It deliberately claims no presence in the request — the picture
    /// cap may have retired it.
    #[must_use]
    pub(crate) fn already_attached_annotation(&self) -> String {
        self.annotation(
            "Image content is already in the conversation as a native image and is not attached \
             again.",
        )
    }

    /// Render the annotation shared by both variants: `{opener} {path} ({w}x{h}, {format}).
    /// {sentence}`, then — when the attached copy is smaller — the copy's own size as a separate
    /// fact, then the tool's own `note`, with the read's recovery note as a prefix.
    ///
    /// The naming sentence states the picture's own size, because that is what the user receives;
    /// the copy the model is given is a different figure and is never mixed into it.
    #[must_use]
    fn annotation(&self, sentence: &str) -> String {
        let (width, height) = self.picture_size;
        let mut lines = vec![format!(
            "{} {} ({width}x{height}, {}). {sentence}",
            self.source.opener(),
            self.path,
            self.format,
        )];
        if self.copy_size != self.picture_size {
            let (copy_width, copy_height) = self.copy_size;
            lines.push(format!("The model's copy is {copy_width}x{copy_height}."));
        }
        if let Some(note) = &self.note {
            lines.push(note.clone());
        }
        with_recovery_note(self.recovery_note.as_deref(), lines.join("\n"))
    }

    /// Build an [`ImagePayload`] from a post-decode [`crate::util::CompressedImageMeta`] and the
    /// caller's resolved path, recovery note and producing tool: the infallible tail after the
    /// fallible decode. The tool-specific `note` is left `None` for the caller to set.
    #[must_use]
    pub(crate) fn from_compressed_meta(
        path: &std::path::Path,
        meta: crate::util::CompressedImageMeta,
        recovery_note: Option<String>,
        source: ImagePayloadSource,
    ) -> Self {
        let crate::util::CompressedImageMeta {
            data_uri,
            sizes,
            format,
        } = meta;
        Self {
            path: path.display().to_string(),
            data_uri,
            picture_size: sizes.file_size,
            copy_size: sizes.copy_size,
            format,
            recovery_note,
            note: None,
            source,
        }
    }
}

/// Normalize a tool call name and arguments, repairing common agent mistakes. Stats and dispatch
/// should use the result so recovered calls are attributed to the real tool.
#[must_use]
pub(crate) fn normalize_tool_call(
    name: &str,
    mut args: serde_json::Value,
) -> (String, serde_json::Value) {
    if name == "glob"
        && let Some(obj) = args.as_object_mut()
        && !obj.contains_key("mode")
    {
        obj.insert("mode".to_string(), serde_json::json!("files"));
    }
    let normalized_name = normalize_tool_name(name).to_string();
    normalize_tool_arguments(&normalized_name, &mut args);
    (normalized_name, args)
}

/// Map known tool-name aliases to their canonical names — the single source of truth, shared by
/// [`normalize_tool_call`] and [`find_tool`]. The `"glob"` alias resolves to `"search"` regardless
/// of arguments; the parallel `mode:"files"` injection happens in [`normalize_tool_call`].
pub(crate) fn normalize_tool_name(name: &str) -> &str {
    match name {
        "bash" | "run_terminal_cmd" => "shell",
        "grep" | "rg" | "grep_search" | "glob" => "search",
        "read_file" => "read",
        "str_replace" => "edit",
        // pre-rename persisted tool calls (mahbot <0.6) still rebind across restarts
        "browser" => "chrome",
        _ => name,
    }
}

fn normalize_tool_arguments(name: &str, args: &mut serde_json::Value) {
    let Some(obj) = args.as_object_mut() else {
        return;
    };

    // Apply path aliases for every tool call; a tool that does not take a "path" ignores the extra
    // key — except the document and computer tools, whose contract refuses every argument an action
    // does not take, so an alias stays as written and the refusal names that spelling.
    if name != "document" && name != "computer" {
        for &alias in PATH_ALIAS_KEYS {
            remap_arg_key(obj, alias, "path");
        }
    }

    match name {
        "edit" => {
            // Per-tool argument remaps specific to "edit".
            remap_arg_key(obj, "old_str", "old_string");
            remap_arg_key(obj, "new_str", "new_string");
        }
        "shell" => {
            remap_arg_key(obj, "cmd", "command");
            remap_arg_key(obj, "script", "command");
        }
        "get_ticket" | "update_ticket" | "add_comment" | "workspace" => {
            remap_arg_key(obj, "id", "ticket_id");
            remap_arg_key(obj, "ticket", "ticket_id");
        }
        _ => {}
    }
}

/// Move `from` → `to` only when the canonical key is absent; the canonical value always wins.
fn remap_arg_key(obj: &mut serde_json::Map<String, serde_json::Value>, from: &str, to: &str) {
    if !obj.contains_key(to)
        && let Some(v) = obj.remove(from)
    {
        obj.insert(to.to_string(), v);
    }
}

/// A tool's answer to one operation, with the flag a frontend that must treat a partly carried-out
/// operation as a failure reads: the agent path reads only `text`, the command line reads `partial`
/// too and exits non-zero. `partial` is the tool's own honest report, never inferred from the
/// wording.
pub(crate) struct ToolAnswer {
    pub text: String,
    pub partial: bool,
}

impl ToolAnswer {
    pub(crate) fn done(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            partial: false,
        }
    }

    pub(crate) fn partial(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            partial: true,
        }
    }
}

/// Look up a tool by name, resolving aliases via [`normalize_tool_name`].
#[must_use]
pub(crate) fn find_tool<'a>(tools: &'a [Box<dyn Tool>], name: &str) -> Option<&'a dyn Tool> {
    let normalized = normalize_tool_name(name);
    tools
        .iter()
        .find(|t| t.name() == normalized)
        .map(Box::as_ref)
}

/// The workspace's own output subdirectory — the fallback for every caller but a
/// command-line call, which names its destination.
pub(crate) const GENERATED_DIR: &str = "generated";

/// The directory a produced file must land in: the command line's destination, else the workspace's
/// own.
pub(crate) fn output_directory(ws: &Workspace, fallback: &str) -> PathBuf {
    crate::agent::tool_output_dir().unwrap_or_else(|| ws.as_path().join(fallback))
}

/// Canonicalize a local media path and confine it to the caller's own workspace — the shared rule
/// for the video tool's local inputs and the generation tools' reference images. Any image or video
/// the caller keeps in its workspace is a valid reference, wherever the caller filed it; the areas
/// the product keeps to itself ([`path::closed_area`]) are refused. The confinement is the AGENTS'
/// rule: the command line is the owner, whose shell already reaches any file, so it is skipped
/// there. The file must exist, be regular and be canonicalized for every caller: what is given is
/// read with no timeout, so a pipe would hold the daemon and its flight guard forever, and nothing
/// is opened or sniffed to decide it. A relative input resolves against the workspace root.
pub(crate) async fn resolve_local_media_path(
    input: &str,
    ws: &Workspace,
    kind: &str,
) -> anyhow::Result<PathBuf> {
    let root = crate::stored_workspace_path(ws.as_path());
    let spelled = Path::new(input);
    let candidate = if spelled.is_absolute() {
        spelled.to_path_buf()
    } else {
        root.join(spelled)
    };
    // The plain spelling, not the platform's verbatim form: this value is read by
    // the callers and echoed into the refusal text the agent sees.
    let canonical = crate::util::strip_verbatim_prefix(
        &tokio::fs::canonicalize(&candidate)
            .await
            .with_context(|| format!("Local {kind} not found: {input}"))?,
    );
    let meta = tokio::fs::metadata(&canonical)
        .await
        .with_context(|| format!("Local {kind} not found: {input}"))?;
    if !meta.is_file() {
        anyhow::bail!(
            "Local {kind} must be a regular file — read with no timeout of its own, a pipe or a \
             device would hold the operation forever, got: {}",
            canonical.display()
        );
    }
    // The owner's own shell reaches any local file already, so the command line
    // is held to the file checks above and nothing more.
    if crate::agent::tool_command_line() {
        return Ok(canonical);
    }
    if !path::is_path_under_roots(&canonical, std::slice::from_ref(&root)) {
        tracing::warn!(
            path = %canonical.display(),
            "Local {kind} rejected: outside the caller's workspace"
        );
        anyhow::bail!(
            "forbidden: a local {kind} must be a file inside your own workspace, got: {} — hint: \
             copy the file into your workspace first",
            canonical.display()
        );
    }
    if let Some(area) = canonical
        .strip_prefix(&root)
        .ok()
        .and_then(path::closed_area)
    {
        tracing::warn!(
            path = %canonical.display(),
            "Local {kind} rejected: inside the closed `{area}` area"
        );
        anyhow::bail!(
            "forbidden: a local {kind} must be a file inside your own workspace, got: {} — \
             `{area}/` is closed to every tool",
            canonical.display()
        );
    }
    Ok(canonical)
}

/// Save generated media bytes into the caller's output directory ([`output_directory`]), naming the
/// file `{prefix}_{timestamp}.{ext}` and reserving a free name so nothing is overwritten; returns
/// the path of a COMPLETE file. Bytes are content-sniffed and stripped (`util::provenance`) on the
/// way out, so the one written file is what the GUI preview, the Telegram delivery and the model
/// payload consume. Security: deliberately bypasses path security — the output directory is
/// tool-owned or already vetted by the command line. Never use it for user-uploaded or arbitrary
/// content.
async fn save_generated_file(
    ws: &Workspace,
    bytes: &[u8],
    prefix: &str,
    ext: &str,
) -> anyhow::Result<PathBuf> {
    let generated_dir = output_directory(ws, GENERATED_DIR);
    tokio::fs::create_dir_all(&generated_dir)
        .await
        .with_context(|| {
            format!(
                "Failed to create generated directory at {}",
                generated_dir.display()
            )
        })?;

    let timestamp = crate::util::unix_millis();
    let stripped = crate::util::provenance::strip_image_provenance(bytes);
    let output_path = crate::util::write_unique_file(
        &generated_dir,
        &format!("{prefix}_{timestamp}.{ext}"),
        stripped.as_ref(),
    )
    .await
    .with_context(|| {
        format!(
            "Failed to write a generated file into {}",
            generated_dir.display()
        )
    })?;

    if matches!(stripped, Cow::Owned(_)) {
        // Debug: one record per generated image — the strip running is the normal
        // case, not a finding the log store or the issues view needs.
        tracing::debug!(
            path = %output_path.display(),
            "Stripped AI-provenance metadata from a generated image artifact"
        );
    }

    Ok(output_path)
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum TimelinePictures {
    /// As the images the agent loop attaches, after the text.
    Attached,
    /// Not at all: the command line's own run takes the text alone, so it neither encodes the
    /// pictures nor claims them.
    TextOnly,
}

/// A reading as a tool shows it: the timeline and, when the pictures are attached, one payload
/// apiece whose annotation names the frames it holds — so the mapping reaches the model with the
/// image itself, and a picture the loop dedups still says which frames it holds. The one place the
/// two go together, so the read tool and the video tools cannot drift.
pub(crate) async fn timeline_output(
    reading: crate::media::Reading,
    pictures: TimelinePictures,
) -> (String, Vec<ImagePayload>) {
    match pictures {
        TimelinePictures::Attached => {
            let composed = reading.pictures().await;
            let count = composed.len();
            let payloads = composed
                .into_iter()
                .enumerate()
                .map(|(index, picture)| {
                    let note = format!(
                        "picture {} of {count}: {}",
                        index + 1,
                        picture.frames_text()
                    );
                    ImagePayload {
                        path: reading.path.display().to_string(),
                        data_uri: picture.data_uri,
                        // The payload is a grid of the video's frames, not a picture of the file:
                        // the grid's own size is what the annotation states, and the copy is that
                        // same grid, so no copy clause is owed.
                        picture_size: picture.size,
                        copy_size: picture.size,
                        format: "JPEG".to_string(),
                        recovery_note: None,
                        note: Some(note),
                        source: ImagePayloadSource::Grid,
                    }
                })
                .collect();
            (reading.text, payloads)
        }
        TimelinePictures::TextOnly => (reading.text, Vec::new()),
    }
}

/// A produced video's answer: what the surface says about the clip — the `[VIDEO:path]` marker, or
/// a render's marker and facts — then the clip's own timeline.
pub(crate) async fn video_output(
    prefix: String,
    path: &std::path::Path,
    pictures: TimelinePictures,
) -> ToolOutput {
    let (text, image_payloads) =
        match crate::media::read(path, crate::media::Window::Overview).await {
            Ok(reading) => timeline_output(reading, pictures).await,
            Err(error) => (error.unreadable(), Vec::new()),
        };
    ToolOutput {
        text: format!("{prefix}\n\n{text}"),
        image_payloads,
        // The prefix and the timeline are the result; the pictures are attached as payloads after
        // them.
        text_is_content: true,
        // Every caller keeps its whole answer through `preserve_full_output`.
        ..ToolOutput::default()
    }
}

/// Wall-clock deadline for an async-video job (1 hour).
const VIDEO_JOB_TIMEOUT: Duration = Duration::from_hours(1);

/// Polling interval between video job status checks (30 s).
const VIDEO_POLL_INTERVAL: Duration = Duration::from_secs(30);

/// Kind-dependent labels for an OpenRouter async-video job: an edit is named apart from a fresh
/// generation, so every message says which one was asked for.
struct VideoJobLabels {
    /// "Video generation" / "Video edit" — submission/poll HTTP contexts,
    /// tracing, "failed" and timeout bails.
    label: &'static str,
    /// "generation" / "editing" — the 402 credits bail.
    gerund: &'static str,
    /// "Video download" / "Video edit download" — download context + MP4 bail.
    download: &'static str,
}

impl VideoJobLabels {
    const GENERATION: Self = Self {
        label: "Video generation",
        gerund: "generation",
        download: "Video download",
    };
    const EDIT: Self = Self {
        label: "Video edit",
        gerund: "editing",
        download: "Video edit download",
    };
}

/// Submit an OpenRouter async-video job (exactly one POST — no retry: each submission is billable
/// and the endpoint has no idempotency key), poll to the 1-hour deadline, download the result and
/// validate it is a real MP4. Returns the bytes; callers save the file and format the media marker.
#[expect(clippy::too_many_lines)]
async fn fetch_async_video(
    api_base: &str,
    body: &serde_json::Value,
    labels: VideoJobLabels,
) -> anyhow::Result<Vec<u8>> {
    // Each submission is a billable job, so exactly one POST — the endpoint has no idempotency key.
    let submit_url = format!("{api_base}/videos");
    let submit_body: serde_json::Value = match crate::util::http::post_json_to_provider(
        &submit_url,
        body,
        &format!("{} submission", labels.label),
    )
    .await
    {
        Ok(v) => v,
        Err(e) => {
            if e.downcast_ref::<crate::util::error::HttpError>()
                .map(|e| e.status)
                == Some(402)
            {
                anyhow::bail!(
                    "Insufficient OpenRouter credits for video {} (HTTP 402). \
                     Please add credits to your OpenRouter account and try again.",
                    labels.gerund,
                );
            }
            return Err(e);
        }
    };

    // OpenRouter returns: { id, polling_url, status, ... }
    let job_id = match submit_body.get("id").and_then(|v| v.as_str()) {
        Some(id) => id.to_string(),
        None => {
            anyhow::bail!("No job ID in submission response: {submit_body}");
        }
    };

    let polling_url = match submit_body.get("polling_url").and_then(|v| v.as_str()) {
        Some(url) => url.to_string(),
        None => format!("{api_base}/videos/{job_id}"),
    };

    tracing::info!(%job_id, "{} job submitted", labels.label);

    let deadline = Instant::now() + VIDEO_JOB_TIMEOUT;
    let mut result_url: Option<String> = None;
    let mut attempt: u32 = 0;

    while Instant::now() < deadline {
        attempt += 1;

        // Sleep the full interval, but never past the deadline.
        let remaining = deadline.saturating_duration_since(Instant::now());
        tokio::time::sleep(VIDEO_POLL_INTERVAL.min(remaining)).await;
        if Instant::now() >= deadline {
            break;
        }

        // Race the poll against the remaining window so a slow poll request
        // cannot push the total wait past the hour.
        let remaining = deadline.saturating_duration_since(Instant::now());
        let poll_body = match tokio::time::timeout(
            remaining,
            crate::util::http::get_json_from_provider(
                &polling_url,
                &format!("{} poll", labels.label),
            ),
        )
        .await
        {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                tracing::debug!(%job_id, attempt, error = %e, "Poll failed");
                continue;
            }
            Err(_) => break,
        };

        let status = poll_body
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        tracing::debug!(%job_id, %status, attempt, "{} poll", labels.label);

        if status == "completed" {
            // OpenRouter provides an `unsigned_urls` array or a `/content?index=0` endpoint.
            result_url = poll_body
                .get("unsigned_urls")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|v| v.as_str())
                .map(String::from)
                .or_else(|| Some(format!("{api_base}/videos/{job_id}/content?index=0")));
            break;
        }

        if status == "failed" || status == "cancelled" || status == "expired" {
            let err_msg = poll_body
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown error");
            anyhow::bail!("{} failed: {err_msg}", labels.label);
        }
    }

    let Some(download_url) = result_url else {
        anyhow::bail!(
            "{} did not complete within the 1-hour timeout period",
            labels.label
        );
    };

    // The result URL requires the bearer key despite the "unsigned" name. The per-request timeout
    // is the remaining job window, so large files on slow connections are not cut short by the
    // 2-minute client cap.
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        anyhow::bail!(
            "{} did not complete within the 1-hour timeout period",
            labels.label
        );
    }
    let video_bytes =
        crate::util::http::get_bytes_from_provider(&download_url, labels.download, remaining)
            .await?;

    // Validate the payload is a real MP4 (no Content-Length on the response;
    // an error page would slip through otherwise).
    if video_bytes.len() <= 100_000 || &video_bytes[4..8] != b"ftyp" {
        anyhow::bail!(
            "{} returned an invalid file ({} bytes, no ftyp header)",
            labels.download,
            video_bytes.len(),
        );
    }

    Ok(video_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tool;
    use crate::workspace::test_ws_named;
    use tempfile::TempDir;

    #[test]
    fn failure_feedback_bounds_the_argument_preview_it_shows() {
        let tool = "image_gen";
        let fault = "Image generation failed: no configured image model served the request.";
        // A multi-byte argument far longer than the preview's cap: the preview is cut and marked,
        // and the whole result stays inside the budget it is delivered in.
        let args = serde_json::json!({ "prompt": "生成".repeat(600) });
        let args_str = args.to_string();
        assert!(args_str.len() > TOOL_FAILURE_ARGS_PREVIEW_BYTES);
        // The fault reaches the formatter in the caller's wrapper, as it does in production.
        let reason = tool_failure_reason(tool, fault);
        let feedback = format_tool_failure_feedback(tool, &args, &reason);
        let preview = feedback
            .lines()
            .find_map(|line| line.strip_prefix("arguments: "))
            .expect("the preview is shown");
        assert!(
            preview.ends_with('…'),
            "a cut preview says so: {}",
            preview.len()
        );
        assert!(
            preview.len() <= TOOL_FAILURE_ARGS_PREVIEW_BYTES,
            "{preview}"
        );
        assert!(
            args_str.starts_with(preview.trim_end_matches('…')),
            "the preview is the head of the arguments"
        );
        // The reserve covers wrapper and envelope, leaving the fault text a byte or two of slack.
        assert!(
            feedback.len() <= tool_failure_envelope_bytes(tool.len()) + fault.len(),
            "{} bytes for a fault text of {}",
            feedback.len(),
            fault.len()
        );
        assert_eq!(crate::util::truncate_tool_output(&feedback), feedback);
    }

    #[test]
    fn image_payload_generated_verb_annotation() {
        let payload = ImagePayload {
            path: "/gen/img.png".into(),
            data_uri: "data:image/jpeg;base64,aaa".into(),
            picture_size: (4, 4),
            copy_size: (4, 4),
            format: "PNG".into(),
            recovery_note: None,
            note: None,
            source: ImagePayloadSource::Generated,
        };
        let fresh = payload.attached_annotation();
        assert!(
            fresh.starts_with("Generated image file /gen/img.png"),
            "fresh: {fresh}"
        );
        assert!(
            !fresh.starts_with("Read image file"),
            "must not be Read: {fresh}"
        );
        assert!(
            !fresh.contains('\n'),
            "note-free annotation must stay one sentence: {fresh}"
        );
        let dup = payload.already_attached_annotation();
        assert!(
            dup.starts_with("Generated image file /gen/img.png"),
            "dup: {dup}"
        );
        assert!(dup.contains("already in the conversation"), "dup: {dup}");

        // A tool-specific note rides on its own line after the attachment
        // sentence; without one the annotation is exactly the sentence above.
        let mut noted = payload.clone();
        noted.note = Some("Served by fallback model fallback-1".into());
        let fresh_noted = noted.attached_annotation();
        assert!(
            fresh_noted.starts_with(&fresh)
                && fresh_noted.ends_with("\nServed by fallback model fallback-1"),
            "note must be appended as a new line: {fresh_noted}"
        );
        assert!(
            noted
                .already_attached_annotation()
                .ends_with("\nServed by fallback model fallback-1"),
            "note also rides on the already-attached annotation"
        );
    }

    /// A produced picture larger than the model-facing cap states the file's own size in the
    /// naming sentence and the copy's size as a separate, labelled fact — the two figures are
    /// never confusable.
    #[tokio::test]
    async fn a_downscaled_copy_is_labelled_apart_from_the_file_size() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("big.png");
        let (width, height) = (1536, 2048);
        tokio::fs::write(&path, crate::util::test::solid_png(width, height))
            .await
            .unwrap();
        let meta = crate::util::local_image_to_compressed_data_uri_with_meta(&path)
            .await
            .unwrap();
        let payload =
            ImagePayload::from_compressed_meta(&path, meta, None, ImagePayloadSource::Generated);

        assert_eq!(
            payload.picture_size,
            (width, height),
            "the file's own pixels"
        );
        assert_eq!(
            payload.copy_size.1,
            crate::util::INBOUND_IMAGE_MAX_SIDE,
            "the copy is bounded by the attachment cap"
        );
        assert!(
            payload.copy_size.0 < width,
            "the copy is smaller than the file"
        );

        let annotation = payload.attached_annotation();
        assert!(
            annotation.starts_with(&format!(
                "Generated image file {} ({width}x{height}, PNG). Image content attached to the \
                 conversation as a native image.",
                path.display()
            )),
            "the naming sentence states the file's size: {annotation}"
        );
        let (copy_width, copy_height) = payload.copy_size;
        assert!(
            annotation.ends_with(&format!(
                "\nThe model's copy is {copy_width}x{copy_height}."
            )),
            "the copy's size is its own labelled fact: {annotation}"
        );
        assert!(
            payload.already_attached_annotation().ends_with(&format!(
                "\nThe model's copy is {copy_width}x{copy_height}."
            )),
            "a deduped repeat states the copy's size too"
        );
        assert!(
            !annotation.contains(&format!("({copy_width}x{copy_height}, PNG)")),
            "the copy's figure is never printed as the file's: {annotation}"
        );
    }

    /// A picture at or below the cap is attached whole, so it owes no copy clause — the file's
    /// size is the whole story.
    #[tokio::test]
    async fn a_picture_within_the_cap_owes_no_copy_clause() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("small.png");
        let (width, height) = (crate::util::INBOUND_IMAGE_MAX_SIDE, 768);
        tokio::fs::write(&path, crate::util::test::solid_png(width, height))
            .await
            .unwrap();
        let meta = crate::util::local_image_to_compressed_data_uri_with_meta(&path)
            .await
            .unwrap();
        let payload =
            ImagePayload::from_compressed_meta(&path, meta, None, ImagePayloadSource::Generated);

        assert_eq!(payload.picture_size, (width, height));
        assert_eq!(payload.copy_size, (width, height));
        assert_eq!(
            payload.attached_annotation(),
            format!(
                "Generated image file {} ({width}x{height}, PNG). Image content attached to the \
                 conversation as a native image.",
                path.display()
            )
        );
    }

    /// A frame grid is not a picture of the file it came from: the size it states is the grid's
    /// own, and no copy clause is owed.
    #[test]
    fn a_frame_grid_states_its_own_size_and_owes_no_copy_clause() {
        let payload = ImagePayload {
            path: "/clip.mp4".into(),
            data_uri: "data:image/jpeg;base64,grid".into(),
            picture_size: (320, 240),
            copy_size: (320, 240),
            format: "JPEG".into(),
            recovery_note: None,
            note: Some("picture 1 of 2: frames 1-4".into()),
            source: ImagePayloadSource::Grid,
        };
        assert_eq!(
            payload.attached_annotation(),
            "Video frame grid /clip.mp4 (320x240, JPEG). Image content attached to the conversation \
             as a native image.\npicture 1 of 2: frames 1-4"
        );
    }

    /// One [`normalize_tool_call`] call: what the caller sent, the tool the call
    /// must be routed to, and the arguments it must carry afterwards.
    struct NormalizeCallCase {
        name: &'static str,
        tool: &'static str,
        input: serde_json::Value,
        routed: &'static str,
        /// `(key, value)` pairs the arguments must carry.
        fields: &'static [(&'static str, &'static str)],
        /// A key the caller's own spelling must survive under, when it must.
        kept: Option<&'static str>,
    }

    /// Verify [`normalize_tool_call`] repairs the names and arguments a caller sent and remaps
    /// every alias in [`PATH_ALIAS_KEYS`] to `"path"` — sweeping the constant itself, so an alias
    /// added tomorrow is exercised. The document and computer tools are the exemptions and refuse
    /// the alias by its spelling.
    #[expect(clippy::too_many_lines)]
    #[test]
    fn normalize_tool_call_cases() {
        for case in [
            NormalizeCallCase {
                name: "a shell alias sends the command under its own key",
                tool: "bash",
                input: serde_json::json!({"cmd": "echo hi"}),
                routed: "shell",
                fields: &[("command", "echo hi")],
                kept: None,
            },
            NormalizeCallCase {
                name: "a search alias becomes a file listing",
                tool: "glob",
                input: serde_json::json!({"query": "main.rs"}),
                routed: "search",
                fields: &[("mode", "files")],
                kept: None,
            },
            NormalizeCallCase {
                name: "an id argument is the ticket to read",
                tool: "get_ticket",
                input: serde_json::json!({"id": "mahbot-1"}),
                routed: "get_ticket",
                fields: &[("ticket_id", "mahbot-1")],
                kept: None,
            },
            NormalizeCallCase {
                name: "an id argument is the workspace to add",
                tool: "workspace",
                input: serde_json::json!({"id": "ws-1"}),
                routed: "workspace",
                fields: &[("ticket_id", "ws-1")],
                kept: None,
            },
            NormalizeCallCase {
                name: "a path alias becomes the path to read",
                tool: "read",
                input: serde_json::json!({"file": "src/main.rs"}),
                routed: "read",
                fields: &[("path", "src/main.rs")],
                kept: None,
            },
            NormalizeCallCase {
                name: "a read alias is routed to the reader with its own key",
                tool: "read_file",
                input: serde_json::json!({"filename": "lib.rs"}),
                routed: "read",
                fields: &[("path", "lib.rs")],
                kept: None,
            },
            NormalizeCallCase {
                name: "the editor's path and both halves of its replacement are repaired",
                tool: "edit",
                input: serde_json::json!({"file": "main.rs", "old_str": "foo", "new_str": "bar"}),
                routed: "edit",
                fields: &[
                    ("path", "main.rs"),
                    ("old_string", "foo"),
                    ("new_string", "bar"),
                ],
                kept: None,
            },
            // The canonical key wins when both spellings arrive; `remap_arg_key`
            // only moves when the canonical key is absent, so the alias remains.
            NormalizeCallCase {
                name: "a canonical path is never overwritten by an alias",
                tool: "read",
                input: serde_json::json!({"path": "canonical.rs", "file": "alias.rs"}),
                routed: "read",
                fields: &[("path", "canonical.rs")],
                kept: Some("file"),
            },
        ] {
            let (name, args) = normalize_tool_call(case.tool, case.input);
            assert_eq!(name, case.routed, "[{}] routed tool", case.name);
            for (key, value) in case.fields {
                assert_eq!(args[key], *value, "[{}] argument {key}", case.name);
            }
            if let Some(key) = case.kept {
                assert!(
                    args.as_object().unwrap().contains_key(key),
                    "[{}] the caller's own key {key} must survive: {args}",
                    case.name
                );
            }
        }

        for &alias in PATH_ALIAS_KEYS {
            for (tool_name, extra) in &[
                ("read", serde_json::json!({})),
                ("shell", serde_json::json!({"cmd": "ls"})),
            ] {
                let mut input = serde_json::json!({});
                input[alias] = serde_json::json!("src/main.rs");
                if let Some(obj) = extra.as_object() {
                    for (k, v) in obj {
                        input[k] = v.clone();
                    }
                }
                let (name, args) = normalize_tool_call(tool_name, input);
                assert_eq!(
                    name, *tool_name,
                    "tool name should not change for {tool_name} with alias {alias}"
                );
                // The alias is remapped to "path" for every tool but the document and computer
                // tools.
                assert_eq!(
                    args["path"], "src/main.rs",
                    "alias {alias} should be remapped to 'path' for tool {tool_name}"
                );
                assert!(
                    !args.as_object().unwrap().contains_key(alias),
                    "alias key {alias} should be removed after normalization for {tool_name}"
                );
                if *tool_name == "shell" {
                    assert_eq!(args["command"], "ls");
                }
            }
        }
        // The document and computer tools name their arguments: an alias stays as the caller wrote
        // it.
        for tool_name in ["document", "computer"] {
            let (_, args) = normalize_tool_call(tool_name, serde_json::json!({ "file": "a.pdf" }));
            assert_eq!(
                args["file"], "a.pdf",
                "{tool_name} must keep the caller's key"
            );
            assert!(
                args.get("path").is_none(),
                "{tool_name} must not invent 'path'"
            );
        }

        // A dangling `send` frame resolves to the sync worker core: an absent or null address is
        // the narrowest choice, the worker address is itself, and any other address is re-executed.
        for (name, args) in [
            ("an absent address", serde_json::json!({"message": "m"})),
            (
                "an explicit null address",
                serde_json::json!({"message": "m", "addressee": null}),
            ),
            (
                "the worker address",
                serde_json::json!({"message": "m", "addressee": "worker"}),
            ),
        ] {
            assert!(
                matches!(
                    SyncDurableCore::from_call("send", &args),
                    Some(SyncDurableCore::Worker)
                ),
                "{name} still resumes the sync worker"
            );
        }
        assert!(
            SyncDurableCore::from_call(
                "send",
                &serde_json::json!({"message": "m", "addressee": "manager"})
            )
            .is_none(),
            "another address is re-executed, never resolved to this core"
        );
    }

    #[test]
    fn all_media_tools_implement_media_marker() {
        let tools: [(&str, Box<dyn Tool>); 5] = [
            ("ImageGenTool", Box::new(ImageGenTool)),
            ("VideoGenTool", Box::new(VideoGenTool)),
            (
                "DocumentTool",
                Box::new(DocumentTool::new(path::PathAccess::Workspace)),
            ),
            (
                "RasterizeTool",
                Box::new(RasterizeTool::new(path::PathAccess::Workspace)),
            ),
            (
                "HyperframesTool",
                Box::new(HyperframesTool::new(path::PathAccess::Workspace)),
            ),
        ];
        for (name, tool) in &tools {
            let marker = tool.media_marker();
            assert!(
                marker.is_some(),
                "{name} should return Some from media_marker()"
            );
            let marker = marker.unwrap();
            assert!(
                marker.starts_with('['),
                "{name} marker {marker:?} should start with '['"
            );
            assert!(
                marker.ends_with(':'),
                "{name} marker {marker:?} should end with ':'"
            );
            let kind = &marker[1..marker.len() - 1]; // strip [ and :
            assert!(
                !kind.is_empty() && kind.chars().all(char::is_uppercase),
                "{name} marker kind {kind:?} should be non-empty uppercase letters"
            );
            let full_marker = format!("{marker}/some/path]");
            assert!(
                crate::util::MEDIA_MARKER_RE.is_match(&full_marker),
                "{name} marker + path should match MEDIA_MARKER_RE, got: {full_marker:?}"
            );
        }
    }

    // The command line's branch of the rule: the owner IS the machine's own user, so the file the
    // caller named is taken wherever it is — while every other caller keeps the workspace
    // confinement, and the regular-file rule holds for both so the branch cannot be widened into a
    // hang.

    #[tokio::test]
    async fn resolve_local_media_path_takes_any_file_for_a_command_line_call() {
        let tmp = TempDir::new().expect("tempdir");
        let ws = crate::Workspace::from_path(tmp.path());
        let write = |name: &str| {
            let path = tmp.path().join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"media").unwrap();
            path
        };
        let inside = write("generated/clip.mp4");
        // A second workspace stands for everything the caller does not own.
        let elsewhere = TempDir::new().expect("tempdir");
        let outside = elsewhere.path().join("picture.png");
        std::fs::write(&outside, b"media").unwrap();

        assert!(
            resolve_local_media_path(&outside.to_string_lossy(), &ws, "clip")
                .await
                .is_err(),
            "an agent's call must stay confined to the workspace"
        );
        assert!(
            resolve_local_media_path(&inside.to_string_lossy(), &ws, "clip")
                .await
                .is_ok()
        );

        let taken = crate::agent::CURRENT_TOOL_COMMAND_LINE
            .scope(true, async {
                resolve_local_media_path(&outside.to_string_lossy(), &ws, "clip").await
            })
            .await
            .expect("a command-line call takes the file the caller named");
        assert_eq!(
            taken,
            crate::util::strip_verbatim_prefix(&outside.canonicalize().unwrap()),
            "the resolved path is the caller's own file"
        );

        #[cfg(unix)]
        {
            // A pipe with no writer would hold the reading tool — and the operation's flight guard
            // — forever, so it is refused for BOTH callers, the command line's branch included.
            let fifo = tmp.path().join("generated/pipe.mp4");
            let path = std::ffi::CString::new(fifo.to_string_lossy().as_bytes())
                .expect("a path without a NUL");
            // SAFETY: `mkfifo` takes a NUL-terminated path and a mode.
            if unsafe { libc::mkfifo(path.as_ptr(), 0o600) } == 0 {
                let err = resolve_local_media_path(&fifo.to_string_lossy(), &ws, "clip")
                    .await
                    .expect_err("a pipe is no media input");
                assert!(err.to_string().contains("regular file"), "{err}");
                let err = crate::agent::CURRENT_TOOL_COMMAND_LINE
                    .scope(true, async {
                        resolve_local_media_path(&fifo.to_string_lossy(), &ws, "clip").await
                    })
                    .await
                    .expect_err("a pipe is no media input for the command line either");
                assert!(err.to_string().contains("regular file"), "{err}");
            }
        }
    }

    /// The shared rule for the video tool's local inputs and the generation tools' reference images:
    /// any file inside the caller's own workspace is a reference, wherever the caller filed it, and
    /// the workspace's closed areas and everything outside it are refused.
    #[tokio::test]
    async fn resolve_local_media_path_confines_to_the_callers_workspace() {
        let tmp = TempDir::new().expect("tempdir");
        let ws = crate::Workspace::from_path(tmp.path());
        let write = |name: &str| {
            let path = tmp.path().join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"media").unwrap();
            path.to_string_lossy().to_string()
        };

        for accepted in [
            "generated/image_1.png",
            "uploads/clip.mp4",
            "album/photo.png",
            "album/2026/clip.mp4",
            "at-the-top.png",
        ] {
            assert!(
                resolve_local_media_path(&write(accepted), &ws, "clip")
                    .await
                    .is_ok(),
                "{accepted} is inside the workspace and must be taken"
            );
        }

        // A relative spelling resolves against the workspace, not the daemon's own directory.
        assert!(
            resolve_local_media_path("album/photo.png", &ws, "clip")
                .await
                .is_ok()
        );

        for closed in ["shared/tool.ts", ".git/photo.png", ".hyperframes/state.png"] {
            let err = resolve_local_media_path(&write(closed), &ws, "clip")
                .await
                .expect_err("a closed area is never a reference");
            assert!(err.to_string().contains("closed to every tool"), "{err}");
        }

        let elsewhere = TempDir::new().expect("tempdir");
        let outside = elsewhere.path().join("clip.mp4");
        std::fs::write(&outside, b"media").unwrap();
        let err = resolve_local_media_path(&outside.to_string_lossy(), &ws, "clip")
            .await
            .expect_err("a file outside the workspace must be refused");
        assert!(
            err.to_string()
                .contains("must be a file inside your own workspace"),
            "unexpected error: {err}"
        );
    }

    // One [`save_generated_file`] call: the workspace it lands in and the name the produced file
    // must carry.

    struct SaveGeneratedCase {
        name: &'static str,
        /// Nest the workspace root one level down, so its directory does not exist.
        nested: bool,
        prefix: &'static str,
        ext: &'static str,
        data: &'static [u8],
    }

    #[tokio::test]
    async fn save_generated_file_cases() {
        for case in [
            SaveGeneratedCase {
                name: "a file is written under its prefix and extension",
                nested: false,
                prefix: "img",
                ext: "png",
                data: b"hello world",
            },
            SaveGeneratedCase {
                name: "a missing workspace directory is created",
                nested: true,
                prefix: "vid",
                ext: "mp4",
                data: b"test content",
            },
        ] {
            let tmp = TempDir::new().expect("tempdir");
            let root = if case.nested {
                tmp.path().join("nested")
            } else {
                tmp.path().to_path_buf()
            };
            let ws = test_ws_named(&root.to_string_lossy(), "test");

            let path = save_generated_file(&ws, case.data, case.prefix, case.ext)
                .await
                .unwrap_or_else(|err| panic!("[{}] save_generated_file: {err}", case.name));
            assert!(
                path.exists(),
                "[{}] file should exist: {}",
                case.name,
                path.display()
            );
            assert!(
                path.starts_with(&root),
                "[{}] file should be inside the workspace",
                case.name
            );
            // Filename format: {prefix}_{timestamp}.{ext}.
            let file_name = path.file_name().unwrap().to_str().unwrap();
            assert!(
                file_name.starts_with(&format!("{}_", case.prefix)),
                "[{}] filename should start with the prefix: {file_name}",
                case.name
            );
            assert!(
                std::path::Path::new(file_name)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case(case.ext)),
                "[{}] filename should end with .{}: {file_name}",
                case.name,
                case.ext
            );
            if !case.nested {
                let content = tokio::fs::read_to_string(&path).await.unwrap();
                assert_eq!(
                    content,
                    std::str::from_utf8(case.data).unwrap(),
                    "[{}] the bytes are written as given",
                    case.name
                );
            }

            let _ = tokio::fs::remove_dir_all(tmp.path()).await;
        }
    }

    enum BudgetScenario {
        AggregateLadder,
        FallsThroughToSmallerReference,
    }

    struct BodyBudgetCase {
        name: &'static str,
        scenario: BudgetScenario,
    }

    #[tokio::test]
    async fn reference_body_budget_cases() {
        for case in [
            BodyBudgetCase {
                name: "a body past the budget compresses its reference",
                scenario: BudgetScenario::AggregateLadder,
            },
            BodyBudgetCase {
                name: "an exhausted ladder falls through to the next reference",
                scenario: BudgetScenario::FallsThroughToSmallerReference,
            },
        ] {
            match case.scenario {
                BudgetScenario::AggregateLadder => {
                    const BUDGET: usize = 150_000;
                    let tmp = TempDir::new().unwrap();
                    let path = tmp.path().join("ref.png");
                    std::fs::write(&path, crate::util::test::noisy_png(384, 384)).unwrap();
                    let mut refs = vec![
                        crate::util::load_reference_image(&path, MAX_REFERENCE_IMAGE_BYTES)
                            .await
                            .unwrap(),
                    ];
                    let original = refs[0].data_uri().to_string();
                    let mut body = serde_json::json!({
                        "model": "test",
                        "prompt": "test",
                        "input_references": reference_json(&refs),
                    });
                    assert!(
                        serde_json::to_vec(&body).unwrap().len() > BUDGET,
                        "[{}] the body starts past the budget",
                        case.name
                    );
                    fit_request_body_budget(&mut body, &mut refs, BUDGET).unwrap();
                    assert!(
                        serde_json::to_vec(&body).unwrap().len() <= BUDGET,
                        "[{}] the body fits the budget",
                        case.name
                    );
                    assert_ne!(
                        refs[0].data_uri(),
                        original,
                        "[{}] reference should be compressed",
                        case.name
                    );
                }
                BudgetScenario::FallsThroughToSmallerReference => {
                    // When the largest reference's ladder is exhausted, the loop falls through to
                    // the next-largest.
                    let tmp = TempDir::new().unwrap();
                    let big = tmp.path().join("big.png");
                    let small = tmp.path().join("small.png");
                    std::fs::write(&big, crate::util::test::noisy_png(512, 512)).unwrap();
                    std::fs::write(&small, crate::util::test::noisy_png(128, 128)).unwrap();
                    let mut refs = vec![
                        crate::util::load_reference_image(&big, MAX_REFERENCE_IMAGE_BYTES)
                            .await
                            .unwrap(),
                        crate::util::load_reference_image(&small, MAX_REFERENCE_IMAGE_BYTES)
                            .await
                            .unwrap(),
                    ];
                    let small_original = refs[1].data_uri().to_string();
                    let mut body = serde_json::json!({
                        "model": "test",
                        "prompt": "test",
                        "input_references": reference_json(&refs),
                    });
                    // Exhaust the big reference's ladder, then measure the raw body so
                    // the budget can be pinned to a window that only the small
                    // reference can close — deterministically forcing the fall-through path.
                    while refs[0].has_compression_left() {
                        refs[0].compress_more().unwrap();
                    }
                    body[INPUT_REFERENCES_KEY] = reference_json(&refs);
                    let raw_body = serde_json::to_vec(&body).unwrap().len();
                    fit_request_body_budget(&mut body, &mut refs, raw_body - 1).unwrap();
                    assert!(
                        serde_json::to_vec(&body).unwrap().len() < raw_body,
                        "[{}] the body shrinks",
                        case.name
                    );
                    assert_ne!(
                        refs[1].data_uri(),
                        small_original,
                        "[{}] smaller reference should have been compressed",
                        case.name
                    );
                }
            }
        }
    }
}
