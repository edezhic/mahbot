//! Tool implementations for agent-callable capabilities.

use anyhow::Context;
pub(crate) mod active_models;
pub(crate) mod alarms;
pub(crate) mod analyze;
pub(crate) mod archive;
pub mod bun;
pub mod chrome;
pub mod chrome_daemon;
pub(crate) mod chrome_groups;
pub mod chrome_release;
pub(crate) mod chrome_tab_ledger;
pub(crate) mod chrome_tabs;
// Computer-use tool (GUI observe/act). Linux, macOS and Windows offer it; every
// other platform reaches the stub backend, which is never advertised.
pub mod computer;
pub(crate) mod custom;
pub(crate) mod delegation;
pub(crate) mod document;
pub(crate) mod edit;
pub(crate) mod grants;
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
pub(crate) mod research;
pub(crate) mod search;
pub(crate) mod search_archived_tickets;
pub(crate) mod send;
pub(crate) mod shell;
pub(crate) mod sleep;
pub(crate) mod ticket;
pub(crate) mod video_edit;
pub(crate) mod video_gen;
pub(crate) mod web_search;
pub(crate) mod workspace;

/// Maximum file size for edit and search tool operations and the dashboard editor
/// (10 MB), and the read tool's cap for any read that is not a document container;
/// a container it accepts up to `FILE_MAX_BYTES` (50 MB), like an inbound
/// attachment. Guards against OOM when agents or the GUI attempt to read very large
/// files. ReadTool/EditTool enforce it via `check_size_within()`; SearchTool and
/// the Iced Editor use it directly.
pub(crate) const MAX_FILE_SIZE_BYTES: u64 = 10 * 1024 * 1024;

/// Maximum size for a single reference image in bytes. The 2 MB figure is a
/// client-side sanity budget, not a documented OpenRouter number; base64 adds
/// ~33% overhead so we cap raw image data at 1.5 MB (1_500_000 bytes) to stay
/// well under.
/// Over-cap references are compressed by the loader (see `load_reference_image`).
/// Note: the aggregate body budget (`MAX_REQUEST_BODY_BYTES`) re-encodes even
/// under-cap references when the serialized request exceeds it, so literal
/// pass-through holds up to ~1.499 MB per single reference (base64 of 1.5 MB
/// already reaches the 2 MB budget once the data-URI prefix and JSON envelope
/// are added).
const MAX_REFERENCE_IMAGE_BYTES: u64 = 1_500_000;

/// Aggregate budget for the serialized generation request body. The per-image
/// cap does not bound the total body when multiple references are present, so
/// the final serialized body is checked against this budget and references are
/// compressed further until it fits. The 2 MB figure is a client-side sanity
/// budget mirroring the pre-existing ~2 MB provider body-limit premise (not a
/// documented OpenRouter number).
const MAX_REQUEST_BODY_BYTES: usize = 2_000_000;

/// The provider's documented maximum number of reference images in one
/// generation request — the image API declares `input_references` with a
/// maximum array length of 16. The live model catalog advertises more than
/// this for some models (up to 20), so the ceiling also clamps the catalog's
/// declared maximum: both for the reference count the tool sends and for the
/// maximum rendered to the agent. It is also the cap when the catalog is
/// unavailable; the aggregate body budget stays the backstop for what reaches
/// the wire.
pub(crate) const MAX_REFERENCE_IMAGES_PER_REQUEST: usize = 16;

/// The effective reference-image maximum for a model whose catalog declares
/// `catalog_max`: the declared maximum, never above the provider's documented
/// ceiling ([`MAX_REFERENCE_IMAGES_PER_REQUEST`]).
#[expect(clippy::cast_possible_wrap)] // small compile-time cap
#[must_use]
pub(crate) fn effective_reference_max(catalog_max: i64) -> i64 {
    catalog_max.min(MAX_REFERENCE_IMAGES_PER_REQUEST as i64)
}

/// Canonical list of argument aliases for file path parameters.
///
/// The aliases `"file"` and `"filename"` are remapped to `"path"` by
/// [`normalize_tool_arguments`] — everywhere but the document tool, which names
/// its own arguments and refuses one its action does not take, so an alias left
/// as written is refused under the spelling the caller used.
const PATH_ALIAS_KEYS: &[&str] = &["file", "filename"];

/// Check that a file's size is within `limit` bytes, naming `label` in the
/// error. The shared gate for the file-reading paths, whose limits differ: the
/// read/edit cap above, and [`crate::util::FILE_MAX_BYTES`] for a converted
/// document container.
fn check_size_within(meta: &std::fs::Metadata, limit: u64, label: &str) -> anyhow::Result<()> {
    if meta.len() > limit {
        anyhow::bail!("{label}: {} bytes (limit: {limit} bytes)", meta.len());
    }
    Ok(())
}

// ── Generation request reference images ─────────────────────────────────

/// Build the OpenRouter `input_references` array (image_url entries) for
/// generation requests. Single source of truth for the reference request shape.
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

/// Key of the reference-image array in the generation request body.
const INPUT_REFERENCES_KEY: &str = "input_references";

/// Ensure the final serialized request body fits `max_body_bytes`, compressing
/// references (largest first, falling through to the next-largest when a
/// reference's ladder is exhausted) one step at a time until it does. The
/// per-image cap does not bound the total body (N refs × cap), so this is the
/// real guard against oversized generation requests.
///
/// The reference array is always (re)synced from the current reference state
/// when references are present, so a caller cannot silently drop them; the
/// loop rewrites it after each compression step.
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
            // The request body is final — drop the retained source bytes so
            // they don't sit in memory through the generation fetch/poll.
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

// ── Re-exports ─────────────────────────────────────────────────────────

pub(crate) use alarms::{AddAlarmTool, ListAlarmsTool, RemoveAlarmTool};
pub(crate) use analyze::{AnalyzeDepth, AnalyzeTool, DispatchMode};
pub(crate) use archive::ArchiveTool;
pub(crate) use chrome::ChromeTool;
pub(crate) use computer::ComputerTool;
pub(crate) use custom::CustomTool;
pub(crate) use document::DocumentTool;
pub(crate) use edit::EditTool;
pub(crate) use image_gen::ImageGenTool;
pub(crate) use mahbot_config::MahbotConfigTool;
pub(crate) use mahbot_debug::MahbotDebugTool;
pub(crate) use rasterize::RasterizeTool;
pub(crate) use read::ReadTool;
pub(crate) use search::SearchTool;
pub(crate) use search_archived_tickets::SearchArchivedTicketsTool;
pub(crate) use send::{Addressee, SendTool};
pub(crate) use shell::{ShellMode, ShellTool};
pub(crate) use sleep::SleepTool;
pub(crate) use ticket::{
    AddCommentTool, CreateTicketTool, GetTicketTool, ListTicketsTool, UpdateTicketTool,
};
pub(crate) use video_edit::VideoEditTool;
pub(crate) use video_gen::VideoGenTool;
pub(crate) use web_search::{WebSearchBackend, WebSearchTool};
pub(crate) use workspace::WorkspaceTool;

use crate::agent::message_router::{AgentJob, MessageKind};
use crate::{Tool, Workspace};
use futures_util::FutureExt;
use std::borrow::Cow;
use std::path::PathBuf;
use std::time::{Duration, Instant};

// ── JSON helpers ─────────────────────────────────────────────────────────

use crate::util::json::{
    get_bool, get_object, get_opt_bool, get_opt_i64, get_opt_str, get_opt_str_strict, get_opt_u64,
    get_str, get_str_array, get_usize, wrong_type,
};

/// The shared `message` argument of the dispatch tools (`analyze`, `send`):
/// required, and non-empty once trimmed. One rule for both tools, so a holder
/// gets the same refusal whichever one it calls.
pub(crate) fn get_dispatch_message(args: &serde_json::Value) -> anyhow::Result<&str> {
    let message = get_str(args, "message")?;
    anyhow::ensure!(
        !message.trim().is_empty(),
        "usage: the message must not be empty"
    );
    Ok(message)
}

/// Prepend a note about argument normalization so the model can see what was
/// silently corrected (e.g. flattened fields, XML wrapping).
pub(crate) fn with_normalization_notes(output: String, notes: &[String]) -> String {
    if notes.is_empty() {
        return output;
    }
    format!("[normalized] {}\n{output}", notes.join("; "))
}

/// Append a bracketed note (`[exit status: 3]`, `[leftover process] …`) under
/// `text` as its own paragraph: the text's trailing whitespace is trimmed so the
/// note is never preceded by a blank gap, and a run that produced nothing is left
/// as just the note.
pub(crate) fn with_note(text: &str, note: &str) -> String {
    let mut out = text.trim_end().to_string();
    if !out.is_empty() {
        out.push_str("\n\n");
    }
    out.push_str(note);
    out
}

/// Refuse a choice-argument value the holder does not offer, naming what it
/// does. A model can emit a value the declaration never listed, so the choice
/// axes validate rather than merely unadvertise.
pub(crate) fn refuse_choice(argument: &str, value: &str, allowed: &[&str]) -> anyhow::Error {
    anyhow::anyhow!(
        "usage: `{argument}` must be one of {} — got \"{value}\"",
        allowed.join(", ")
    )
}

/// Resolve a choice argument against the kinds a holder has, in interface
/// order (narrowest first). An absent (or explicit null) value takes the
/// narrowest kind; a named value must be one of them — anything else is
/// refused, never silently narrowed. `held` must not be empty (it is the
/// holder's declared set, and a holder always has at least one kind).
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

/// Build a JSON schema for tool parameters.
///
/// Wraps `properties` in the standard `{"type": "object", "properties": {...}}`
/// envelope and conditionally adds `"required"` only when the slice is non-empty.
///
/// This eliminates repetitive boilerplate across tool implementations.
/// Tools with non-standard top-level keys in their top-level schema
/// (e.g., `oneOf` in WebSearchTool) should not use this directly;
/// they may still use it internally as a building block (e.g.,
/// ChromeTool's [`action_entry_schema`] calls it for each inner entry).
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

/// Build a single action entry for a `oneOf` action list.
///
/// Wraps a `"{"name": inner}"` object in the standard `{"type":"object",
/// "properties":{name: inner}, "required":[name], "additionalProperties":false,
/// "description":description}` envelope used by the chrome and computer tools'
/// nested action schemas. `inner` is built by [`tool_params_schema`], which adds
/// `"required"` only when non-empty, and then hard-closes the action object.
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

// ── scrub ────────────────────────────────────────────────────────────

use crate::util::scrub_credentials;

/// Scrub credentials from tool output; delegates the scrubbing policy to the tool.
///
/// Call flow:
///
/// ```text
/// agent::execute_tool
///   └─ scrub_tool_output(tool, args, output)   (tools/mod.rs)
///        ├─ tool.should_scrub_output(args)      (lib.rs, per-tool override)
///        │    ├─ returns true  → scrub_credentials(output)  (util/mod.rs)
///        │    └─ returns false → output as-is
/// ```
///
/// Integration with shell tool's internal scrubbing:
/// The shell tool's `apply_profile_pipeline` (shell/mod.rs) scrubs stdout and stderr
/// at pipeline entry, so `ShellTool::should_scrub_output` returns `false` to prevent
/// this function from double-scrubbing. See the `ShellTool::should_scrub_output` doc
/// comment for the rationale. Any tool that performs its own credential scrubbing
/// internally must follow the same pattern — return `false` from `should_scrub_output`.
#[must_use]
pub(crate) fn scrub_tool_output(
    tool: &dyn Tool,
    call_arguments: &serde_json::Value,
    output: &str,
) -> String {
    if tool.should_scrub_output(call_arguments) {
        scrub_credentials(output)
    } else {
        output.to_string()
    }
}

/// Prefix of a failed tool result — shared so the research wrap-up's
/// success detection checks the same marker [`format_tool_failure_feedback`]
/// produces (a successful result never starts with it).
pub(crate) const TOOL_FAILURE_MARKER: &str = "Tool call failed.";

/// Byte cap for the argument preview in [`format_tool_failure_feedback`]: a byte
/// bound, so a multi-byte argument cannot take the whole budget the reason beside
/// it is measured against.
const TOOL_FAILURE_ARGS_PREVIEW_BYTES: usize = 1_000;

/// What the caller wraps a tool's own fault text with before it reaches
/// [`format_tool_failure_feedback`] — see [`tool_failure_reason`].
const TOOL_FAILURE_REASON_PREFIX: &str = "Error executing ";

/// The reason a tool's fault is reported with: the fault text in the wrapper
/// [`Agent::execute_tool`](crate::agent::Agent) raises it with, so a tool bounding the
/// text it composes and the wrapper that carries it agree on one shape.
#[must_use]
pub(crate) fn tool_failure_reason(tool_name: &str, fault: &str) -> String {
    format!("{TOOL_FAILURE_REASON_PREFIX}{tool_name}: {fault}")
}

/// The room a fault may NOT take inside the result the model receives: the wrapper
/// above plus everything [`format_tool_failure_feedback`] spends on the text itself —
/// the marker, the labels, a tool name of `tool_name_len` bytes and the capped argument
/// preview. A tool composing a fault text subtracts this from the budget a tool result
/// is truncated to rather than re-stating the shape of either wrapper.
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
    // The `reason` parameter is pre-scrubbed by the caller
    // ([`failure_outcome`](crate::agent::Agent::failure_outcome)) and passed
    // through as-is to avoid double-scrubbing. The `tool_args` are scrubbed
    // here since they're formatted for display.
    // Scrubbed before the cap: redaction can lengthen the text.
    let args = scrub_credentials(&tool_args.to_string());
    let args_preview = crate::util::cut_bytes(&args, TOOL_FAILURE_ARGS_PREVIEW_BYTES);
    format!(
        "{TOOL_FAILURE_MARKER}\n\
         tool: {tool_name}\n\
         arguments: {args_preview}\n\
         reason:\n{reason}"
    )
}

/// Build the canonical `internal` tool error for a mahbot-side fault the
/// model cannot recover from (lost state, missing internal rows, all-worker
/// failure): no argument change or retry can fix it — a user or engineer
/// must investigate the mahbot service.
#[must_use]
pub(crate) fn internal_fault(cause: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "internal: {cause} (mahbot-side fault) — \
         hint: no model action can recover this; a user or engineer must \
         investigate the mahbot service"
    )
}

/// Run one durable round's own future and catch a panic in it: the panic is
/// logged against `label` and handed back as its message, so each caller
/// decides what a panicked round leaves behind — a dispatch routes it as the
/// round's error envelope, the resume path as the round's failure. A round that
/// panics never takes the instance with it: `panic = "unwind"` is load-bearing
/// here.
pub(crate) async fn caught_round<T>(
    label: &str,
    round: impl std::future::Future<Output = T>,
) -> Result<T, String> {
    // Boxed before it is caught: the round's own future is far larger than the
    // threshold `clippy::large_futures` warns on, and this is once per durable
    // round.
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

/// Gives back a synthetic `dir-…` exec engine when it is dropped, so a round that
/// unwinds releases it exactly as one that returns does. A managed
/// workspace's own engine is never touched (see
/// [`delegation::release_exec_engine`]).
struct ExecEngineGuard<'a>(&'a crate::Workspace);

impl Drop for ExecEngineGuard<'_> {
    fn drop(&mut self) {
        crate::tools::delegation::release_exec_engine(self.0);
    }
}

/// Internal error-carrier for a tool call whose durable work was drain-cut by
/// shutdown/drain. NEVER persisted — the emission-time frame already recorded
/// the call durably, so the result is simply absent and the universal
/// resume-completion step re-runs/resumes it. Carried as an `anyhow::Error`
/// out of a sync dispatch (analyze, worker) so the caller's tool-execution
/// layer can mark the outcome `suspended`.
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

// ── Sync durable-core glue ──────────────────────────────────────────────
//
// The two dispatch tools (`analyze`, `send`) share one durable lifecycle —
// sync dispatch, async dispatch, boot resume, result envelopes — and ALL of
// that glue lives on [`SyncDurableCore`] so there is exactly one
// implementation: the sync tool paths and the boot-resume rounds both
// delegate to it.

/// The job-scoped parameters every durable sync core takes. Beyond
/// `(ws, task)` these are the values the sync wrappers read from the
/// task-locals / job row, so the core never touches `CURRENT_TOOL_*` itself.
pub(crate) struct CoreJobArgs<'a> {
    pub job_id: &'a str,
    pub caller_role: crate::Role,
    pub user_name: &'a str,
    pub channel: &'a str,
    /// The fresh-spawn row record — the resume discriminator: `Some` spawns
    /// (writes the jobs row), `None` resumes (reuses the stored roster and
    /// outcomes, never inserts a row).
    pub spawn: Option<SpawnRecord<'a>>,
    /// Caller session pin persisted on the jobs row at fresh spawn so the
    /// live resume-completion step can find caller-owned launched jobs.
    /// Always `None` on resume — the row already carries it.
    pub caller_agent_id: Option<&'a str>,
    /// Sync-call discriminator: a checkpoint DB error fails the tool call /
    /// round so it is retried; async & boot-resume paths warn-and-continue
    /// (the outcome is recomputable on the next resume).
    pub fail_on_checkpoint_error: bool,
}

/// The jobs-row fields a fresh dispatch writes. Read ONLY when the call
/// actually spawns (see [`CoreJobArgs::spawn`]); a resume inserts no row and
/// carries no record — so a resume can never hand the chosen directory's name
/// to the delivery key.
pub(crate) struct SpawnRecord<'a> {
    /// The CALLER's own workspace name: the durable delivery/routing key
    /// (`jobs.workspace_name`). Never the chosen directory.
    pub delivery_name: &'a str,
    /// The chosen directory the round executes in, when the caller asked for one
    /// — including one that happens to name the caller's own directory — so the
    /// reply always names the directory really used and a restart resumes into
    /// the same identity. Persisted as `jobs.exec_dir`.
    pub exec_dir: Option<&'a str>,
    /// The CALLER's custom-role launch, when the round was dispatched by a
    /// custom role: persisted as `jobs.custom_launch` so a boot resume rebuilds
    /// the envelope addressed to the custom branch (rather than the tool's
    /// construction role). The WHOLE launch travels — a resumed turn opens its
    /// branch from this snapshot, so the authorial prompt and the tool subset
    /// are as load-bearing as the branch address. `None` for every built-in
    /// caller.
    pub custom_launch: Option<&'a crate::Launch>,
}

/// The single outcome form of the durable sync cores (analyze, worker):
/// `Terminal` wraps a produced result (Ok or Err — both terminalize with a
/// durable envelope), `DrainCut` means the round was cut by drain/shutdown —
/// the job stays status='launched' for resume (boot or live), nothing is
/// routed or terminalized now, `Cancelled` means the round observed its
/// cancel signal and stopped.
pub(crate) enum SyncCoreOutcome {
    Terminal(Result<String, anyhow::Error>),
    DrainCut,
    /// The round observed its cancel signal and stopped: nothing is produced and
    /// nothing is routed. Its durable rows are the cancel path's to remove (see
    /// [`SyncDurableCore::dispatch_durable`]).
    Cancelled,
}

/// The sync analyze/worker durable cores, normalized behind one interface so the
/// sync dispatch, the resume-completion loop and the per-tool wrappers all
/// reuse a single body. [`Worker`](Self::Worker)'s `jobs.kind` is
/// [`crate::jobs::SpawnChild::Implement`]'s (`"implement"`).
#[derive(Copy, Clone)]
pub(crate) enum SyncDurableCore {
    Analyze,
    Worker,
}

impl SyncDurableCore {
    /// The log/site label for this core ("Analyze" / "Worker").
    #[must_use]
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Analyze => "Analyze",
            Self::Worker => "Worker",
        }
    }

    /// The round kind this core's jobs row carries.
    pub(crate) fn kind_str(self) -> &'static str {
        match self {
            Self::Analyze => crate::jobs::SpawnChild::Analyze.kind_str(),
            Self::Worker => crate::jobs::SpawnChild::Implement.kind_str(),
        }
    }

    /// Resolve a dangling tool call to the sync durable core that settles it.
    /// The core is resolved from the tool plus its choice argument, never by
    /// matching a name against a stored round kind: `analyze`'s sync form is the
    /// quick round and `send`'s is the single-run worker, so a call that asked for
    /// another kind (`deep`, `worker_branch`, `manager`, `role`, `guest`), an
    /// unknown value, a non-string, or an unknown tool name — a retired tool's
    /// included — resolves to `None` and is re-executed instead of binding an
    /// unrelated job. A retired tool name is not resolvable either: history is
    /// never rewritten, so a cut-mid-flight retired call is simply repeated.
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
        // Each tool's sync form is its narrowest choice: the quick depth, the
        // worker addressee. An absent (or explicit null) choice is that narrowest
        // kind — how a call frame that predates the merge reconnects to its
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

    /// Run the durable core — both per-tool bodies return [`SyncCoreOutcome`]
    /// directly.
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

    /// Abandon this dispatch's own row after the call has already failed: the
    /// failure IS the call's result, so the model repeats the call and an
    /// unfinished row left behind would only be bindable by a later, unrelated
    /// call of the same core — settling a completed round's stored report as
    /// that call's own result. Never an alias: the call is repeated, not
    /// reconnected. A drain-cut is the only outcome that keeps the row (see
    /// [`Self::run_sync_dispatch`]).
    async fn abandon_failed_dispatch(conn: &crate::db::Connection, job_id: &str, label: &str) {
        if let Err(e) = crate::jobs::terminalize_job(conn, job_id).await {
            tracing::warn!(
                job = %job_id,
                error = %e,
                "Failed to abandon the failed {label} dispatch's row",
            );
        }
    }

    /// The round's cancel-signal guard for the core the cancel action can stop
    /// ([Analyze](Self::Analyze) only), `None` for the rest — see
    /// [`crate::round_cancel`].
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

    /// Whether the round's cancel signal has fired — the delivery path's last look
    /// before the result is handed over, so a cancelled round's result is never
    /// delivered.
    fn stopped_by_cancel(self, job_id: &str) -> bool {
        if !crate::round_cancel::is_cancelled(job_id) {
            return false;
        }
        tracing::info!(job = %job_id, "{} round stopped by cancel — nothing delivered", self.label());
        true
    }

    /// Shared sync dispatch: read the task-locals the way the two wrappers do
    /// today, generate a fresh job id, run the core as a fresh spawn
    /// (`spawn: Some(..)` — the core writes the jobs row), then
    /// terminalize-on-terminal / surface [`CallSuspended`] on drain-cut. Every
    /// outcome but a drain-cut abandons the row with the call's own result, so a
    /// row stays launched only while its call is dangling.
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

        // This dispatch never binds to prior jobs: if the model re-issues a call
        // after the resume-completion already settled and delivered, no launched
        // job remains and this spawns fresh.

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
            // Unreachable: only the async entry points register a cancel signal,
            // and a sync dispatch's id is freshly generated — and whatever
            // cancels a round removes that round's own rows, so there is nothing
            // to clean up here.
            SyncCoreOutcome::Cancelled => Err(crate::tools::internal_fault(
                "a sync round observed a cancel signal it never registered",
            )),
        }
    }

    /// Shared sync resume: run the [`crate::jobs::resume_job_preamble_discrete`]
    /// preamble, then the core as a resume (`spawn: None` — it reuses the
    /// stored roster and outcomes and inserts no row), mapping the outcome to
    /// [`crate::jobs::SyncResumeOutcome`].
    ///
    /// `sync_resume` selects checkpoint-error propagation: a live sync resume
    /// (the caller's resume-completion step, agent/mod.rs) passes `true` — a
    /// checkpoint DB error fails the round; the async boot-resume rounds pass
    /// `false` — warn-and-continue, the outcomes are recomputable on the next
    /// resume.
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
            // An unreadable row is a resume-core infra failure, not an abandon:
            // fail the round so the caller-driven path never surfaces an
            // "explicitly abandoned" note for a live job (the row stays
            // launched and the next cycle retries).
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
            // A cancel landed while this resume ran: the round stopped and its
            // rows are the cancel action's to remove — a quiet abort, exactly
            // like an explicitly abandoned row.
            Ok(SyncCoreOutcome::Cancelled) => Ok(crate::jobs::SyncResumeOutcome::Gone),
            Err(e) => Err(e),
        }
    }

    /// Message kind of the async result envelope routed to the caller.
    #[must_use]
    pub(crate) fn message_kind(self) -> MessageKind {
        match self {
            Self::Analyze => MessageKind::AnalyzeToolResult,
            Self::Worker => MessageKind::ImplementResult,
        }
    }

    /// Build the async result envelope delivered to the caller's agent channel
    /// for this core's round results (success and failure paths).
    ///
    /// The analyze envelope keeps the caller-generated `job_id` as its first body
    /// line so the delivered result is correlatable with the immediate ack
    /// placeholder, and `exec_dir` names the directory the round really ran in
    /// when the caller asked for one. The worker's envelope is the
    /// `<worker-report>` — no job id, and the folder the worker worked in: the
    /// chosen directory when one was given, else the delivery workspace's own
    /// folder.
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

    /// Durable async dispatch (SPAWN → run → CHECKPOINT → COMPLETE): run the
    /// core as a fresh spawn (`spawn: Some(..)` — it writes the jobs row) and no
    /// session pin, then terminalize into the
    /// durable envelope under the caller-generated `job_id` (the id is created
    /// in the tool's `execute()` BEFORE the spawn so the immediate ack and the
    /// delivered envelope share one id). Returns `None` when the round was cut
    /// by drain/shutdown — the job stays status='launched' for boot resume and
    /// NOTHING is routed now — and also when the round observed a cancel: the
    /// cancel action owns the round's rows (it sweeps them, and answers the caller
    /// when it cannot), so nothing is completed, routed or removed here; the
    /// round's own row write is gated in-tx against the cancel signal (see
    /// `jobs::spawn_job`), so the two cannot both stand. On error the envelope
    /// still routes (errors wrapped in the core's envelope tag) and the job is
    /// terminalized.
    ///
    /// `delivery` is the caller's own workspace (the durable delivery key);
    /// `place`, when present, is the directory the round executes in — the
    /// sub-agents' root, context and search key, while delivery stays the
    /// caller's.
    ///
    /// `custom`, when present, is the calling custom role's launch: the round's
    /// envelopes and its jobs row carry `Role::Custom` and the launch so a boot
    /// resume delivers back to the custom branch, never to the tool's own
    /// construction role's address.
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
                // Drain-cut: the round's outcomes are already checkpointed — leave
                // the job status='launched' for boot resume. No terminalization, no
                // error envelope: a spurious envelope here would discard the
                // checkpointed outcomes and contradict "jobs stay status='launched'
                // for boot resume".
                tracing::info!(
                    job = %job_id,
                    "{} round cut short by drain — job stays launched for boot resume",
                    self.label(),
                );
                return None;
            }
            Ok(SyncCoreOutcome::Cancelled) => {
                // Cancelled: the cancel action already answered the caller and
                // owns the round's rows — nothing is completed, routed or removed
                // here, because there is nothing to remove: the row write itself is
                // gated in-tx against the cancel signal (see `jobs::spawn_job`), and
                // a sweep that failed was answered to the caller as a cancellation
                // that did not happen. Unlike the drain-cut arm (which keeps the
                // engine for the resume) the synthetic `dir-…` goes.
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
        // The round is over: drop the synthetic `dir-…` engine it may have
        // created (a managed workspace's own engine is left alone). The
        // drain-cut early return above deliberately skips this — that job
        // resumes in the same directory.
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

    /// Boot-resume a durable round: resolve the execution workspace (the
    /// recorded directory when the round had one, else the caller's own
    /// workspace), run the core as a resume (`spawn: None` — it reuses the stored
    /// roster and outcomes and inserts no row) through
    /// [`Self::resume_sync_core`], then terminalize into a durable envelope and
    /// route it to the ORIGINAL caller (role/user/channel persisted on the job
    /// row at spawn), never the Manager. The delivery workspace stays the
    /// caller's own (`delivery`) so the result lands in the conversation that
    /// asked.
    ///
    /// A round that PANICS is caught here and reported through the same arm as
    /// any other round that produced no outcome: the failure is delivered to the
    /// conversation that was promised a result, and the row is terminalized. A
    /// panicked round is over — leaving the row would make every boot retry a
    /// round that dies the same way, silently.
    ///
    /// Aborts quietly on shutdown/drain: no routing, no terminalization — the
    /// job row stays for the next boot (checkpointed outcomes are reused, so
    /// already-completed LLM work is never lost or duplicated). A cancel landing
    /// during the resume is the one stop that discards the round's own result
    /// instead (the cancel action owns its rows).
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
        // A cancel can land between the boot scan that picked this round and the
        // registration above — and it fires into nothing, because this round had
        // not registered yet. Its mark reaches that registration
        // ([`crate::round_cancel`]), so the signal is already fired here: return
        // before running a round whose caller was answered as stopped.
        if self.stopped_by_cancel(job_id) {
            return;
        }
        // The round's own future is caught: a panic in it is a round that
        // produced no outcome, and it is reported as one (see the doc above)
        // rather than left as a launched row nobody is told about.
        let resumed = caught_round(&format!("{label} resume"), async {
            match exec_dir {
                None => self.resume_sync_core(delivery, job_id, false).await,
                Some(dir) => match crate::tools::delegation::workspace_at_dir(dir).await {
                    Ok(exec) => {
                        // Released however this ends: the round may unwind, and a
                        // directory's engine registered for the process lifetime is
                        // exactly what a panic must not leave behind. Whatever next
                        // needs it re-creates it on demand.
                        let _engine = ExecEngineGuard(&exec);
                        self.resume_sync_core(&exec, job_id, false).await
                    }
                    // The recorded directory is gone: the round ends through
                    // the resume-failure arm below with that plain message — never
                    // left hanging.
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
            // Drain-cut mid-resume / job row gone (explicitly abandoned, or
            // removed by a cancel racing this resume): abort quietly — the core
            // logs the reason; a gone job was already noted.
            Ok(crate::jobs::SyncResumeOutcome::DrainCut | crate::jobs::SyncResumeOutcome::Gone) => {
                return;
            }
            // Resume infra failure (roster load / checkpoint), an unusable
            // recorded directory, or a round that panicked: deliver an error
            // envelope to the ORIGINAL caller, exactly like a round-level
            // failure. The job row is still present (the core never
            // terminalizes), so the caller identity can be re-loaded for the
            // route.
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
        // Drain/shutdown fired after the round returned: abort quietly WITHOUT
        // routing and WITHOUT deleting the row — the outcomes are checkpointed
        // (next boot reuses them) and routing a partial result here would race
        // the exit.
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

    /// Async dispatch entry of [`crate::tools::analyze::AnalyzeTool`]'s quick
    /// depth and [`crate::tools::send::SendTool`]'s worker addressee: read the
    /// caller identity from
    /// the `CURRENT_TOOL_*` task-locals, spawn the core's durable dispatch in the
    /// background, catch its panics, and route the resulting envelope to the
    /// caller's agent channel. The core is the single source of the envelope
    /// kind, tag and builder — no call-site knobs can drift. The `job_id` is
    /// generated by the caller (the tool's `execute()`) BEFORE this spawn so the
    /// immediate ack placeholder and the delivered envelope reference one
    /// identical id.
    ///
    /// Semantics, in order:
    ///
    /// 1. `tokio::spawn` + `catch_unwind` so a panic in the dispatch task can
    ///    never leave the caller waiting forever on a result that can never
    ///    arrive — the panic path rebuilds an error envelope here.
    /// 2. The dispatch builds the success envelope itself (with pending_job_id
    ///    set by the completion tx) — route it as-is so the persisted copy and
    ///    the routed copy can never drift.
    /// 3. `Ok(None)` = drain-cut or cancel: route NOTHING now. For a drain-cut
    ///    the job stays status='launched' for boot resume (a spurious error
    ///    envelope during the drain would discard the checkpointed outcome; the
    ///    result envelope is delivered after boot resume); for a cancel the
    ///    round is gone for good. This silent return is deliberately BEFORE the
    ///    aborting() guard.
    /// 4. `crate::shutdown::aborting()` guard: drain/shutdown fired between the
    ///    dispatch and the route — the pending row (if any) survives for boot
    ///    replay; skip routing rather than deliver into a consumer that has
    ///    stopped pulling. (If the pending INSERT had failed, this also
    ///    suppresses the insert-failure policy's best-effort route — the
    ///    surviving launched job row is resumed at the next boot, so the result
    ///    is deferred, never lost.)
    /// 5. The round's cancel signal is registered BEFORE the spawn and consulted
    ///    immediately before the route: a round the caller cancels is never
    ///    delivered — the one case where a produced result is deliberately
    ///    discarded.
    /// 6. `crate::agent::message_router::route(&crate::jobs::envelope_target(&envelope), envelope)`.
    ///
    /// Note: src/tools/research.rs deliberately does NOT share this scaffold —
    /// it logs "ended without delivery" and omits the pre-route aborting() guard
    /// (at-least-once rationale).
    ///
    /// `delivery` is the caller's own workspace (the durable delivery key);
    /// `place`, when present, is the directory the round executes in.
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
        // Read synchronously before the spawn (task-local contract): the round's
        // envelope and job row must carry the caller's branch when it is a
        // custom role, and the default construction role otherwise — that
        // derivation lives in `dispatch_durable`.
        let custom = crate::agent::tool_custom_launch();
        // The dispatching session's agent id, read synchronously before the
        // spawn (task-local contract) — the round's cancel-signal entry records
        // it, so a cancel naming this id can prove ownership even in the window
        // before the round's `jobs` row is written.
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
                    // Drain-cut or cancel (see doc point 3): route nothing.
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

/// Wrap a sub-agent/tool result in the async `<tag>` envelope delivered to
/// the caller's agent channel. The `job_id` is embedded as the first body line
/// so the delivered result is correlatable with the immediate ack placeholder;
/// when the round ran in a chosen directory (`exec_dir`), a `Directory:` line
/// follows it in the same shape so the caller is never in doubt which directory
/// the round really ran in. Failures carry an explicit marker instead of an
/// empty result. Shared with the deep analysis run.
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

/// Render a branch-report asset (`worker_report.md` / `worker_failure.md`): the
/// same `<worker-report>` envelope for both, with the folder it happened in and —
/// only when the turn ran on a named branch — its `branch` attribute. The single
/// renderer for a worker's report, shared by a single-run dispatch and a worker
/// branch's turn so the two shapes cannot drift.
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

/// The `branch="id"` attribute a report envelope carries for a named branch,
/// empty otherwise — the shape `custom_role_message.md`, `worker_report.md` and
/// `worker_failure.md` all render through `{{branch_attr}}`.
pub(crate) fn branch_attr(identifier: Option<&str>) -> String {
    identifier.map_or_else(String::new, |id| {
        format!(" branch=\"{}\"", envelope_attr(id))
    })
}

/// An attribute value interpolated into an envelope (`<worker-report folder="…">`,
/// `<manager-message workspace="…">`, `<custom-role-message role="…" workspace="…">`).
///
/// The value is usually a folder path the model itself chose, so ONLY the three
/// characters that could end the attribute or forge an envelope are replaced:
/// escaping `&` or `'` too would distort an ordinary name — an account called
/// `o'brien` would become a folder the model is told to trust but that does not
/// exist, and copying it back into a folder argument would not resolve.
pub(crate) fn envelope_attr(value: &str) -> String {
    value
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// A tool call's textual result together with the images it produced — what
/// [`crate::Tool::execute_with_payloads`] returns.
#[derive(Debug)]
pub(crate) struct ToolOutput {
    /// The tool's textual result.
    pub text: String,
    /// Images produced by this call, injected by the agent loop as native
    /// image parts after the round's tool-result block.
    pub image_payloads: Vec<ImagePayload>,
    /// Whether `text` is real content the model must keep, rather than a claim
    /// ABOUT the images this call produced. The single flag behind both
    /// decisions the agent loop makes from a result: with content, image
    /// annotations are APPENDED and any `[IMAGE:path]` marker inside it is
    /// literal text; without it, the annotations REPLACE `text` and such a
    /// marker is a claim that may be turned into an injected image. `true` for
    /// the read tool's converted-document answer (extracted text, or the note
    /// standing in for what a document could not yield) and for the computer
    /// tool, whose result describes the surface and the action rather than the
    /// image.
    pub text_is_content: bool,
}

/// Pair a textual result with the image payload of the tool's
/// [`Tool::image_payload`] hook — the tail of the default
/// [`Tool::execute_with_payloads`], shared with the read tool's override so the
/// two can never drift.
///
/// The text is flagged as a claim about the image (not content) whether or not a
/// payload comes back, which is what every tool routed here has always meant by
/// its result: a media tool names the file it wrote, the read tool the file it
/// read. Tools whose text is the result itself — the read tool's
/// converted-document answer, the computer tool's surface and outcome — bypass
/// this helper.
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
    }
}

/// Outcome for a tool execution.
#[expect(clippy::struct_excessive_bools)] // independent per-call flags, not states
#[derive(Debug)]
pub(crate) struct ToolExecutionOutcome {
    pub output: String,
    pub success: bool,
    /// Images produced by this call, injected after the round's tool-result block.
    pub image_payloads: Vec<ImagePayload>,
    /// The tool's own [`ToolOutput::text_is_content`], carried through unchanged.
    pub text_is_content: bool,
    /// `true` when the tool's durable work was drain-cut and its result is
    /// intentionally absent (the call's frame is already durably persisted at
    /// emission time; the result settles on the next resume-completion). Such
    /// outcomes are SKIPPED by the tool-result commit — no row is written and
    /// the call stays dangling.
    pub suspended: bool,
    /// `true` when a successful outcome should end the agent run gracefully
    /// after the round's results commit (sleep tool); failed/unknown calls
    /// never set it.
    pub ends_turn: bool,
}

/// Discriminates the tool that produced an [`ImagePayload`] so the agent loop can
/// label the injected image with the actual opening phrase ("Read image file",
/// "Generated image file", "Chrome screenshot") without knowing which tool
/// produced it — the loop selects the annotation purely by dedup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImagePayloadSource {
    /// On-disk raster read by the read tool.
    Read,
    /// Tool-produced image (e.g. image_gen, or a page rasterized or extracted
    /// from a converted document), or one derived by the agent loop from a tool's
    /// `[IMAGE:path]` marker.
    Generated,
    /// Screenshot captured by the chrome tool (injected as a native image for
    /// the agent's own visual analysis).
    Chrome,
    /// Screen capture from the computer tool (injected as a native image for
    /// the agent's own visual analysis).
    Computer,
}

impl ImagePayloadSource {
    /// Opening phrase of the injected-image annotation, e.g. "Read image file"
    /// for the read tool's on-disk raster, "Generated image file" for a
    /// tool-produced image, "Chrome screenshot" for the chrome's capture.
    #[must_use]
    pub(crate) fn opener(self) -> &'static str {
        match self {
            Self::Read => "Read image file",
            Self::Generated => "Generated image file",
            Self::Chrome => "Chrome screenshot",
            Self::Computer => "Computer screenshot",
        }
    }
}

/// Per-call image payload produced by a tool that reads an on-disk raster image
/// (the Read tool), by a tool that generated or captured one (image_gen, chrome,
/// computer), or derived by the agent loop from a real on-disk `[IMAGE:path]`
/// marker in a successful tool output that supplied no payload of its own. The
/// agent loop injects `[IMAGE:{data_uri}]` as a synthetic persisted User-role
/// message after the round's tool results.
///
/// `width`/`height` are the post-EXIF/post-resize dimensions of the *encoded*
/// JPEG the data-URI carries, and `format` is the uppercase source-format label
/// (`"PNG" | "JPEG" | "WEBP"`). Together they let the agent loop build the
/// definitive attached/already-attached annotation from the payload metadata
/// rather than rewriting a hardcoded sentence in a tool-result string.
#[derive(Debug, Clone)]
pub(crate) struct ImagePayload {
    /// Canonical (resolved) path of the image — surfaced in the tool-result
    /// annotation (`{opener} {path} ...`) and the debug log. The dedup
    /// key used by the agent loop is `data_uri`, not `path`.
    pub path: String,
    /// Bounded JPEG data-URI (`data:image/jpeg;base64,...`).
    pub data_uri: String,
    /// Post-EXIF/post-resize width of the encoded JPEG (px).
    pub width: u32,
    /// Post-EXIF/post-resize height of the encoded JPEG (px).
    pub height: u32,
    /// Uppercase source-format label: `"PNG" | "JPEG" | "WEBP"`.
    pub format: String,
    /// `Some` when the read resolved a typo'd path to a unique filename match;
    /// preserved in the tool-result annotation so the model keeps the typo
    /// context (mirrors the `[Recovered path: ...]` note shown for text reads).
    pub recovery_note: Option<String>,
    /// An extra line the annotation carries, for a tool-specific fact that must
    /// reach the assistant with the image itself (image_gen names the model that
    /// produced the picture when a fallback served it).
    pub note: Option<String>,
    /// Tool that produced the image, selecting the annotation opener
    /// ("Read image file", "Generated image file", "Chrome screenshot").
    pub source: ImagePayloadSource,
}

/// Prefix `body` with a read's typo-recovery note, when there is one — the one
/// place the `[Recovered path: ...]` line is put in front of a read's answer, so
/// a recovered text read, a recovered raster's annotation and a converted
/// document's answer word it identically.
#[must_use]
pub(crate) fn with_recovery_note(note: Option<&str>, body: String) -> String {
    match note {
        Some(note) => format!("{note}\n{body}"),
        None => body,
    }
}

impl ImagePayload {
    /// Tool-result annotation for a FRESH attachment (the image is injected as a
    /// new synthetic user message).
    #[must_use]
    pub(crate) fn attached_annotation(&self) -> String {
        self.annotation("Image content attached to the conversation as a native image.")
    }

    /// Tool-result annotation for a DEDUPED repeat read (image already attached;
    /// the read returns a reference rather than re-adding it).
    #[must_use]
    pub(crate) fn already_attached_annotation(&self) -> String {
        self.annotation("Image content is already attached to the conversation as a native image.")
    }

    /// Render the annotation shared by the fresh/already-attached variants:
    /// `{opener} {path} ({w}x{h}, {format}). {sentence}`, then the tool's own
    /// `note` on a new line, then the read's recovery note as a prefix.
    #[must_use]
    fn annotation(&self, sentence: &str) -> String {
        let base = format!(
            "{} {} ({}x{}, {}). {}",
            self.source.opener(),
            self.path,
            self.width,
            self.height,
            self.format,
            sentence
        );
        let base = match self.note.as_deref() {
            Some(note) => format!("{base}\n{note}"),
            None => base,
        };
        with_recovery_note(self.recovery_note.as_deref(), base)
    }

    /// Build an [`ImagePayload`] from a post-decode [`crate::util::CompressedImageMeta`]
    /// (the bounded data URI + post-EXIF dims + source-format label) and the
    /// caller's resolved path, recovery note, and producing tool.
    ///
    /// This is the infallible construction tail: the caller performs the
    /// fallible decode separately
    /// (`crate::util::local_image_to_compressed_data_uri_with_meta(...).await.ok()?`)
    /// and passes the decoded meta in to build the payload.
    ///
    /// The annotation `note` (a tool-specific fact carried with the image, e.g.
    /// image_gen naming the model that produced a fallback picture) is left
    /// `None` for the caller to set.
    #[must_use]
    pub(crate) fn from_compressed_meta(
        path: &std::path::Path,
        meta: crate::util::CompressedImageMeta,
        recovery_note: Option<String>,
        source: ImagePayloadSource,
    ) -> Self {
        Self {
            path: path.display().to_string(),
            data_uri: meta.data_uri,
            width: meta.width,
            height: meta.height,
            format: meta.format,
            recovery_note,
            note: None,
            source,
        }
    }
}

/// Normalize a tool call name and arguments, repairing common agent mistakes.
///
/// Returns `(normalized_name, normalized_args)`. Stats and dispatch should use
/// the normalized values so recovered calls are attributed to the real tool.
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

/// Map known tool-name aliases to their canonical names.
///
/// This is the single source of truth for tool-name normalization, shared by
/// [`normalize_tool_call`] (full call normalization) and [`find_tool`] (direct
/// lookup).  Adding a new alias here immediately affects both paths.
///
/// The `"glob"` alias is included because it resolves to `"search"` regardless
/// of arguments; the parallel `mode:"files"` injection is handled separately
/// in [`normalize_tool_call`] when args are available. `pub(crate)` so the
/// research sanitizer can gate on the canonical name before paying for an
/// argument clone.
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

    // Apply path-aliases (file/filename) for every tool call. Tools that do not
    // take a "path" simply ignore the extra key — except the document tool, whose
    // contract refuses every argument an action does not take: an alias is left
    // as the caller wrote it, so a refusal names that spelling rather than a
    // "path" the caller never passed. The `workspace` tool keeps the alias (its
    // `add` takes the path); its `pause` / `resume` refuse a stray field, but by
    // that tool's own address rule rather than by the key's name.
    if name != "document" {
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

/// Move `from` → `to` only when the canonical key is absent.
///
/// When both the alias (`from`) and canonical (`to`) keys are present the
/// canonical value is used.
fn remap_arg_key(obj: &mut serde_json::Map<String, serde_json::Value>, from: &str, to: &str) {
    if !obj.contains_key(to)
        && let Some(v) = obj.remove(from)
    {
        obj.insert(to.to_string(), v);
    }
}

/// A tool's answer to one operation, with the flag a frontend that must treat a
/// partly carried-out operation as a failure reads.
///
/// The agent path reads only [`text`](Self::text) (through [`Tool::execute`]);
/// the command line reads `partial` too, because it must exit non-zero when
/// part of what it asked for did not happen. `partial` is the tool's own honest
/// report — the text names the part that did not succeed — and is never
/// inferred from the wording.
pub(crate) struct ToolAnswer {
    pub text: String,
    pub partial: bool,
}

impl ToolAnswer {
    /// The whole operation was carried out.
    pub(crate) fn done(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            partial: false,
        }
    }

    /// Part of the operation did not succeed, as `text` says.
    pub(crate) fn partial(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            partial: true,
        }
    }
}

/// Look up a tool by name in a slice of boxed `dyn Tool` values.
///
/// Tool-name aliases are resolved via `normalize_tool_name` so that all
/// callers benefit from the same alias mapping.  Prefer [`normalize_tool_call`]
/// before dispatch when full argument normalization is also desired.
#[must_use]
pub(crate) fn find_tool<'a>(tools: &'a [Box<dyn Tool>], name: &str) -> Option<&'a dyn Tool> {
    let normalized = normalize_tool_name(name);
    tools
        .iter()
        .find(|t| t.name() == normalized)
        .map(Box::as_ref)
}

/// The directory a produced file must land in: the destination a
/// command-line caller chose ([`crate::agent::tool_output_dir`]), and the
/// workspace's own `fallback` directory for every other caller.
pub(crate) fn output_directory(ws: &Workspace, fallback: &str) -> PathBuf {
    crate::agent::tool_output_dir().unwrap_or_else(|| ws.as_path().join(fallback))
}

/// Canonicalize a local media path and confine it to the workspace's
/// `uploads/` (received attachments) and `generated/` (tool outputs) dirs — the
/// shared rule for video_edit's local inputs and the generation tools'
/// reference images. `kind` names the input in the messages.
///
/// The confinement is the AGENTS' rule: a command-line caller is the machine's
/// own owner, whose shell can already hand any local file to any program, so the
/// command line adds no capability the owner does not have and this check is
/// skipped for it. Everything else still happens — the file must exist, be a
/// regular file and be canonicalized.
///
/// The regular-file rule holds for every caller, agent or not: what is given here
/// is read by the tool (an upload, a duration probe, an image decode) with no
/// timeout of its own, so a pipe with no writer would hold the daemon — and the
/// operation's own flight guard — forever and a device would read without end.
/// Nothing is opened to decide it: metadata is the whole check.
///
/// Callers must run it before any other filesystem access on the path: a path
/// refused here is only canonicalized and stat'ed — never opened or sniffed.
///
/// Roots are canonicalized when they exist; a missing directory cannot contain
/// a canonicalized (existing) file, so it is skipped — which fails closed when
/// neither exists. A relative input resolves against the process CWD rather
/// than the workspace, matching video_edit's local inputs.
pub(crate) async fn resolve_local_media_path(
    input: &str,
    ws: &Workspace,
    kind: &str,
) -> anyhow::Result<PathBuf> {
    // The plain spelling, not the platform's verbatim form: this value is read by
    // the callers and echoed into the refusal text the agent sees.
    let canonical = crate::util::strip_verbatim_prefix(
        &tokio::fs::canonicalize(input)
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
    let roots: Vec<PathBuf> = ["uploads", "generated"]
        .iter()
        .map(|d| ws.as_path().join(d))
        .filter_map(|r| std::fs::canonicalize(&r).ok())
        .collect();
    if !path::is_path_under_roots(&canonical, &roots) {
        tracing::warn!(
            path = %canonical.display(),
            "Local {kind} rejected: not inside workspace uploads or generated dirs"
        );
        anyhow::bail!(
            "forbidden: local {kind} must be inside the workspace uploads directory \
             (received media attachments) or the generated directory (previously \
             generated media), got: {} — hint: move the file into the workspace \
             uploads directory and retry",
            canonical.display()
        );
    }
    Ok(canonical)
}

/// Save generated media bytes into the caller's output directory
/// ([`output_directory`]: the workspace's `generated/` for every caller but a
/// command-line call, which names its destination), naming the file
/// `{prefix}_{timestamp}.{ext}` and reserving a free name so a file already
/// there is never overwritten (see [`crate::util::write_unique_file`]).
///
/// Creates the directory if needed, writes the file, and returns the full
/// `PathBuf` naming a COMPLETE file. The bytes are content-sniffed and stripped
/// (`util::provenance`) on the way out, so the single written file is what the GUI
/// preview, the Telegram delivery and the model-context payload all consume;
/// non-image outputs pass through untouched.
///
/// # Security note
/// This function deliberately bypasses path security (no `resolve_write_target`
/// check): the output directory is either the tool-owned `generated/` of the
/// caller's workspace or the destination a command-line caller named, which the
/// command line itself has already resolved and vetted (`crate::cli`). Do not
/// use this function for user-uploaded or arbitrary content.
async fn save_generated_file(
    ws: &Workspace,
    bytes: &[u8],
    prefix: &str,
    ext: &str,
) -> anyhow::Result<PathBuf> {
    let generated_dir = output_directory(ws, "generated");
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

/// Format a video tool result: the `[VIDEO:path]` marker first (so the reply
/// path and GUI keep working), then a "Video content:" description line when
/// the shared video transcription succeeds. Fail-open: marker-only on any
/// failure. The transcription's live tracking resolves via the tool-execution
/// task-local (the owning agent's activity indicator); `None` is passed for
/// workspace because the in-agent path never needs a non-agent call row.
pub(crate) async fn format_video_result(marker: String, output_path: &std::path::Path) -> String {
    match crate::providers::transcribe_video_file(output_path, None).await {
        Some(text) => format!("{marker}\n\nVideo content: {text}"),
        None => marker,
    }
}

// ── Async video jobs (video_gen / video_edit) ───────────────────────────

/// Wall-clock deadline for an async-video job (1 hour) — shared by the
/// video_gen and video_edit tools.
const VIDEO_JOB_TIMEOUT: Duration = Duration::from_hours(1);

/// Polling interval between video job status checks (30 s).
const VIDEO_POLL_INTERVAL: Duration = Duration::from_secs(30);

/// Kind-dependent labels for an OpenRouter async-video job.
///
/// Fields stay separate so every user-facing string is byte-identical with the
/// pre-refactor tools (e.g. gen's download context is bare "Video download",
/// not "Video generation download").
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

/// Submit an OpenRouter async-video job (exactly one POST — no retry; each
/// submission is a billable job and the endpoint has no idempotency key),
/// poll for completion (1-hour wall-clock deadline), download the result, and
/// validate it is a real MP4. Returns the video bytes; callers save the file
/// and format the media marker.
#[expect(clippy::too_many_lines)]
async fn fetch_async_video(
    api_base: &str,
    body: &serde_json::Value,
    labels: VideoJobLabels,
) -> anyhow::Result<Vec<u8>> {
    // ── Step 1: Submit video job (exactly one POST — no retry) ─────────
    // Each submission is a billable job; the endpoint has no idempotency key.
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

    // ── Step 2: Poll for completion (1-hour wall-clock deadline) ────────
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
            // Deadline elapsed while the poll request was in flight.
            Err(_) => break,
        };

        let status = poll_body
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        tracing::debug!(%job_id, %status, attempt, "{} poll", labels.label);

        if status == "completed" {
            // Download URL: OpenRouter provides unsigned_urls array or
            // a content endpoint at /api/v1/videos/{jobId}/content?index=0
            result_url = poll_body
                .get("unsigned_urls")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|v| v.as_str())
                .map(String::from)
                .or_else(|| {
                    // Fallback: use content endpoint
                    Some(format!("{api_base}/videos/{job_id}/content?index=0"))
                });
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

    // ── Step 3: Download the video ──────────────────────────────────────
    // The result URL requires the bearer key despite the "unsigned" name.
    // The per-request timeout is the remaining job window, so large files on
    // slow connections are not cut short by the shared client's 2-minute cap.
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

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tool;
    use crate::ToolSpec;
    use crate::workspace::test_ws_named;
    use tempfile::TempDir;

    // ── ToolSpec serde ───────────────────────────────────────────

    #[test]
    fn tool_spec_serde_roundtrip() {
        let spec = ToolSpec {
            name: "test".into(),
            description: "A test tool".into(),
            parameters: serde_json::json!({"type": "object"}),
        };
        let parsed: ToolSpec =
            serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
        assert_eq!(parsed.name, "test");
    }

    #[test]
    fn failure_feedback_bounds_the_argument_preview_it_shows() {
        let tool = "image_gen";
        let fault = "Image generation failed: no configured image model served the request.";
        // A multi-byte argument far longer than the preview's own cap: the preview is
        // cut, marked as cut, and the whole result stays inside the budget it is
        // delivered in — which is what a tool composing a fault text counts on.
        let args = serde_json::json!({ "prompt": "生成".repeat(600) });
        let args_str = args.to_string();
        assert!(args_str.len() > TOOL_FAILURE_ARGS_PREVIEW_BYTES);
        // The fault reaches the formatter in the caller's wrapper, as it does in
        // production.
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
        // The reserve covers the wrapper and the envelope alike, and leaves the fault
        // text a byte or two of slack at most (a char-boundary cut never fills its cap
        // exactly).
        assert!(
            feedback.len() <= tool_failure_envelope_bytes(tool.len()) + fault.len(),
            "{} bytes for a fault text of {}",
            feedback.len(),
            fault.len()
        );
        assert_eq!(crate::util::truncate_tool_output(&feedback), feedback);
    }

    // ── ImagePayload source verb ──────────────────────────────────

    #[test]
    fn image_payload_generated_verb_annotation() {
        let payload = ImagePayload {
            path: "/gen/img.png".into(),
            data_uri: "data:image/jpeg;base64,aaa".into(),
            width: 4,
            height: 4,
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
        assert!(dup.contains("already attached"), "dup: {dup}");

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

    // ── find_tool aliases ──────────────────────────────────────────

    #[test]
    fn find_tool_aliases() {
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(SearchTool),
            Box::new(ShellTool::new(ShellMode::Full)),
            Box::new(ReadTool::general()),
            Box::new(EditTool::confined()),
        ];

        // Each case: (alias, expected_tool_name or None).
        let cases: &[(&str, Option<&str>)] = &[
            // Canonical names
            ("search", Some("search")),
            ("shell", Some("shell")),
            ("read", Some("read")),
            ("edit", Some("edit")),
            // Shell aliases
            ("bash", Some("shell")),
            ("run_terminal_cmd", Some("shell")),
            // Search aliases
            ("grep", Some("search")),
            ("rg", Some("search")),
            ("grep_search", Some("search")),
            ("glob", Some("search")),
            // Read aliases
            ("read_file", Some("read")),
            // Edit aliases
            ("str_replace", Some("edit")),
            // Unknown tool
            ("unknown", None),
        ];

        for &(input, expected) in cases {
            let found = find_tool(&tools, input);
            assert_eq!(found.map(Tool::name), expected, "find_tool({input:?})");
        }
    }

    #[test]
    fn normalize_tool_call_repairs_names_and_args() {
        let (name, args) = normalize_tool_call("bash", serde_json::json!({"cmd": "echo hi"}));
        assert_eq!(name, "shell");
        assert_eq!(args["command"], "echo hi");

        let (name, args) = normalize_tool_call("glob", serde_json::json!({"query": "main.rs"}));
        assert_eq!(name, "search");
        assert_eq!(args["mode"], "files");

        let (name, args) = normalize_tool_call("get_ticket", serde_json::json!({"id": "mahbot-1"}));
        assert_eq!(name, "get_ticket");
        assert_eq!(args["ticket_id"], "mahbot-1");

        let (name, args) = normalize_tool_call("workspace", serde_json::json!({"id": "ws-1"}));
        assert_eq!(name, "workspace");
        assert_eq!(args["ticket_id"], "ws-1");

        // "read" tool: file/filename → path
        let (name, args) = normalize_tool_call("read", serde_json::json!({"file": "src/main.rs"}));
        assert_eq!(name, "read");
        assert_eq!(args["path"], "src/main.rs");

        let (name, args) =
            normalize_tool_call("read_file", serde_json::json!({"filename": "lib.rs"}));
        assert_eq!(name, "read");
        assert_eq!(args["path"], "lib.rs");

        // "edit" tool: file/filename → path, old_str/new_str → old_string/new_string
        let (name, args) = normalize_tool_call(
            "edit",
            serde_json::json!({"file": "main.rs", "old_str": "foo", "new_str": "bar"}),
        );
        assert_eq!(name, "edit");
        assert_eq!(args["path"], "main.rs");
        assert_eq!(args["old_string"], "foo");
        assert_eq!(args["new_string"], "bar");

        // Canonical "path" key is preserved when both "path" and alias are present.
        // remap_arg_key only moves when canonical key is absent, so "file" remains.
        let (name, args) = normalize_tool_call(
            "read",
            serde_json::json!({"path": "canonical.rs", "file": "alias.rs"}),
        );
        assert_eq!(name, "read");
        assert_eq!(args["path"], "canonical.rs");
        assert!(args.as_object().unwrap().contains_key("file"));
    }

    #[test]
    // ── media_marker coverage ────────────────────────────────────────
    fn all_media_tools_implement_media_marker() {
        // Each media-generation tool must return Some from media_marker()
        let tools: [(&str, Box<dyn Tool>); 5] = [
            ("ImageGenTool", Box::new(ImageGenTool)),
            ("VideoGenTool", Box::new(VideoGenTool)),
            ("VideoEditTool", Box::new(VideoEditTool)),
            (
                "DocumentTool",
                Box::new(DocumentTool::new(path::PathAccess::Workspace)),
            ),
            (
                "RasterizeTool",
                Box::new(RasterizeTool::new(path::PathAccess::Workspace)),
            ),
        ];
        for (name, tool) in &tools {
            let marker = tool.media_marker();
            assert!(
                marker.is_some(),
                "{name} should return Some from media_marker()"
            );
            let marker = marker.unwrap();
            // Validate format: `[KIND:` where KIND is uppercase letters
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
            // Validate against the canonical MEDIA_MARKER_RE pattern
            let full_marker = format!("{marker}/some/path]");
            assert!(
                crate::util::MEDIA_MARKER_RE.is_match(&full_marker),
                "{name} marker + path should match MEDIA_MARKER_RE, got: {full_marker:?}"
            );
        }
    }

    // ── normalize_tool_call tests ─────────────────────────────────────

    /// Verify that [`normalize_tool_call`] remaps every alias in
    /// [`PATH_ALIAS_KEYS`] to `"path"` — regardless of whether the tool
    /// explicitly accepts a `"path"` parameter.
    ///
    /// This test verifies path-aliasing for both a path-accepting tool (`read`)
    /// and a non-path tool (`shell`). The remap happens unconditionally;
    /// such tools simply ignore the extra key.
    ///
    /// This test explicitly iterates the constant so the loop-based approach in
    /// [`normalize_tool_arguments`] is verified against all current aliases.
    /// If an alias is added to [`PATH_ALIAS_KEYS`], this test immediately
    /// exercises it — preventing any gap between the lookup path and the
    /// normalization path.
    ///
    /// The document tool is the one exemption: it declares and enforces its own
    /// argument names, so an alias is left as the caller wrote it and refused by
    /// that name instead of being remapped into a `"path"` two of its actions do
    /// not take. Remapped alias and ignored-or-refused extra key are separate
    /// things: `workspace` takes the alias for its `add` while refusing a stray
    /// field on `pause` / `resume` (`tools::workspace::a_misspelled_address_…`).
    #[test]
    fn normalize_tool_call_remaps_all_path_aliases() {
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
                // The alias is remapped to "path" for every tool but the document
                // tool: tools that don't use "path" simply ignore the extra key.
                assert_eq!(
                    args["path"], "src/main.rs",
                    "alias {alias} should be remapped to 'path' for tool {tool_name}"
                );
                // The alias key itself should have been removed since "path" was absent.
                assert!(
                    !args.as_object().unwrap().contains_key(alias),
                    "alias key {alias} should be removed after normalization for {tool_name}"
                );
                // Shell-specific remaps must be unaffected.
                if *tool_name == "shell" {
                    assert_eq!(args["command"], "ls");
                }
            }
        }
        // The document tool's own contract names its arguments: an alias stays as
        // the caller wrote it, for the boundary to refuse that spelling by name.
        let (_, args) = normalize_tool_call("document", serde_json::json!({ "file": "a.pdf" }));
        assert_eq!(args["file"], "a.pdf");
        assert!(args.get("path").is_none());
    }

    // ── resolve_local_media_path tests ─────────────────────────────────────

    /// The command line's own branch of the rule: the owner IS the machine's own
    /// user, so the file the caller named is taken wherever it is — while every
    /// other caller keeps the workspace confinement, and the regular-file rule
    /// holds for both so the branch cannot be widened into a hang.
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
        let outside = write("outside/picture.png");
        let inside = write("generated/clip.mp4");

        // The agents' rule: the workspace's own media dirs, and nowhere else.
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

        // The command line's: the very file the caller gave it.
        let taken = crate::agent::CURRENT_TOOL_COMMAND_LINE
            .scope(true, async {
                resolve_local_media_path(&outside.to_string_lossy(), &ws, "clip").await
            })
            .await
            .expect("a command-line call takes the file the caller named");
        assert!(
            taken.ends_with("outside/picture.png"),
            "the resolved path is the caller's own file: {}",
            taken.display()
        );

        #[cfg(unix)]
        {
            // A pipe with no writer would hold the reading tool — and the
            // operation's flight guard — forever, so it is refused for BOTH
            // callers, the command line's branch included.
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

    /// The rule video_edit's local inputs and the generation tools' reference
    /// images share: only the workspace's `uploads/` and `generated/` dirs are
    /// reachable, and anywhere else is refused with the reason.
    #[tokio::test]
    async fn resolve_local_media_path_confines_to_uploads_and_generated() {
        let tmp = TempDir::new().expect("tempdir");
        let ws = crate::Workspace::from_path(tmp.path());
        let write = |name: &str| {
            let path = tmp.path().join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"media").unwrap();
            path.to_string_lossy().to_string()
        };

        // Both media dirs are accepted, including `generated/` when `uploads/`
        // does not exist yet.
        assert!(
            resolve_local_media_path(&write("generated/image_1.png"), &ws, "image")
                .await
                .is_ok()
        );
        assert!(
            resolve_local_media_path(&write("uploads/clip.mp4"), &ws, "clip")
                .await
                .is_ok()
        );

        // Anywhere else in the workspace is refused, naming both accepted dirs.
        let err = resolve_local_media_path(&write("config.toml"), &ws, "clip")
            .await
            .expect_err("a file outside uploads/generated must be refused");
        assert!(
            err.to_string()
                .contains("must be inside the workspace uploads directory"),
            "unexpected error: {err}"
        );

        // A workspace with neither dir refuses everything — a missing dir
        // cannot contain a canonicalized file, so the rule fails closed.
        let bare = TempDir::new().expect("tempdir");
        let bare_ws = crate::Workspace::from_path(bare.path());
        let root_file = bare.path().join("clip.mp4");
        std::fs::write(&root_file, b"media").unwrap();
        assert!(
            resolve_local_media_path(&root_file.to_string_lossy(), &bare_ws, "clip")
                .await
                .is_err()
        );
    }

    // ── save_generated_file tests ──────────────────────────────────────────

    #[tokio::test]
    async fn save_generated_file_creates_file() {
        let tmp = TempDir::new().expect("tempdir");
        let ws = test_ws_named(&tmp.path().to_string_lossy(), "test");

        let data = b"hello world";
        let path = save_generated_file(&ws, data, "img", "png")
            .await
            .expect("save_generated_file should succeed");

        assert!(path.exists(), "file should exist: {}", path.display());
        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(content, "hello world");

        // Verify filename format: {prefix}_{timestamp}.{ext}
        let file_name = path.file_name().unwrap().to_str().unwrap();
        assert!(
            file_name.starts_with("img_"),
            "filename should start with 'img_': {file_name}",
        );
        assert!(
            std::path::Path::new(file_name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("png")),
            "filename should end with '.png': {file_name}",
        );

        let _ = tokio::fs::remove_dir_all(tmp.path()).await;
    }

    #[tokio::test]
    async fn save_generated_file_creates_directory_if_missing() {
        let tmp = TempDir::new().expect("tempdir");
        let ws = test_ws_named(&tmp.path().join("nested").to_string_lossy(), "test");

        let data = b"test content";
        let path = save_generated_file(&ws, data, "vid", "mp4")
            .await
            .expect("save_generated_file should create dirs");

        assert!(path.exists(), "file should exist: {}", path.display());
        assert!(
            path.starts_with(tmp.path().join("nested")),
            "file should be inside workspace"
        );

        let _ = tokio::fs::remove_dir_all(tmp.path()).await;
    }

    // ── Reference-image body budget ─────────────────────────────────

    #[tokio::test]
    async fn reference_body_budget_compresses_further() {
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
        // The per-image cap passes a 384×384 ref through unchanged (~590 KB
        // base64 body); a small budget forces the aggregate ladder.
        assert!(serde_json::to_vec(&body).unwrap().len() > BUDGET);
        fit_request_body_budget(&mut body, &mut refs, BUDGET).unwrap();
        assert!(serde_json::to_vec(&body).unwrap().len() <= BUDGET);
        assert_ne!(
            refs[0].data_uri(),
            original,
            "reference should be compressed"
        );
    }

    #[tokio::test]
    async fn body_budget_falls_through_to_smaller_reference() {
        // When the largest reference's ladder is exhausted, the loop must
        // fall through to the next-largest instead of bailing.
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
        // Exhaust the big reference's ladder, then measure the raw body so the
        // budget can be pinned to a window that only the small reference can
        // close — deterministically forcing the fall-through path.
        while refs[0].has_compression_left() {
            refs[0].compress_more().unwrap();
        }
        body[INPUT_REFERENCES_KEY] = reference_json(&refs);
        let raw_body = serde_json::to_vec(&body).unwrap().len();
        fit_request_body_budget(&mut body, &mut refs, raw_body - 1).unwrap();
        assert!(serde_json::to_vec(&body).unwrap().len() < raw_body);
        assert_ne!(
            refs[1].data_uri(),
            small_original,
            "smaller reference should have been compressed"
        );
    }
}
