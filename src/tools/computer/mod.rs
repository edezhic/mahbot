//! Computer (GUI observe/act) tool for the full-access Assistant.
//!
//! The platform-agnostic core (types, coordinate math, ref lifecycle, backend
//! trait) lives in [`core`]; [`macos`]/[`linux`]/[`stub`] implement the backend
//! per platform. All GUI actions are serialized through a process-wide lock so
//! two concurrent agent runs can never race on the live GUI.

mod core;
mod linux;
mod macos;
// `stub` is the unsupported-platform backend; on macOS/Linux it would be
// entirely dead (the cfg-gated `backend()` factory never reaches it), so gate
// the module itself rather than carry per-item allows.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod stub;

use self::core::{
    AppInfo, Backend, Capture, ElementAct, MouseButton, Region, ScrollDirection, TargetSpec,
    taxonomy_error,
};
use crate::util::{TOOL_OUTPUT_BUDGET_BYTES, UnwrapPoison};
use crate::{Tool, Workspace};
use anyhow::Context;
use async_trait::async_trait;
use serde::Deserialize;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::{Duration, Instant};
use strum::IntoEnumIterator;
use tracing::debug;

// ── Action enum ─────────────────────────────────────────────────────────

/// One computer tool call. All points/regions are normalized 0-1000 relative to
/// the target surface (see the `## Coordinate contract` prompt section).
#[derive(Debug, Clone, Deserialize, Serialize, strum::EnumIter)]
#[serde(rename_all = "snake_case")]
enum ComputerAction {
    Observe {
        target: Option<String>,
    },
    Screenshot {
        target: Option<String>,
    },
    Zoom {
        target: Option<String>,
        region: Region,
    },
    // NOTE: `Apps`/`Cursor` must stay EMPTY STRUCT variants (`{}`), not unit
    // variants. The canonical wire form is `{"apps": {}}` (empty object), which
    // serde's externally-tagged representation only accepts for a struct variant
    // — a unit variant would reject the map with "invalid type: map, expected
    // unit", making both actions dead on every input path. Serializing an empty
    // struct still yields the same `{"apps": {}}` key, so wire-name extraction
    // and `EnumIter` are unaffected.
    Apps {},
    Windows {
        app: Option<String>,
    },
    Click {
        target: Option<String>,
        #[serde(rename = "ref")]
        reference: Option<String>,
        x: Option<f64>,
        y: Option<f64>,
        button: Option<MouseButton>,
        double: Option<bool>,
        modifiers: Option<Vec<core::Modifier>>,
    },
    Type {
        text: String,
        target: Option<String>,
        #[serde(rename = "ref")]
        reference: Option<String>,
    },
    Press {
        keys: String,
    },
    Scroll {
        direction: ScrollDirection,
        amount: Option<u32>,
        target: Option<String>,
        x: Option<f64>,
        y: Option<f64>,
    },
    Drag {
        from: (f64, f64),
        to: (f64, f64),
        target: Option<String>,
    },
    Cursor {},
    Wait {
        seconds: f64,
    },
}

/// Canonical wire names for every [`ComputerAction`] variant, derived from the
/// enum itself via `strum::EnumIter` + serde external tagging — a new variant
/// is auto-discovered rather than hand-maintained. Used as the
/// [`normalize_action`] allowlist and by the schema lockstep test.
static ACTION_WIRE_NAMES: LazyLock<Vec<String>> = LazyLock::new(|| {
    <ComputerAction as IntoEnumIterator>::iter()
        .map(|action| wire_name(&action))
        .collect()
});

/// Serialize a [`ComputerAction`] and extract its wire name: a lone tag for a
/// fielded variant, or a plain string for a unit variant.
fn wire_name(action: &ComputerAction) -> String {
    let v = serde_json::to_value(action).expect("ComputerAction serializes");
    match v {
        Value::String(s) => s,
        Value::Object(obj) => {
            assert_eq!(obj.len(), 1, "ComputerAction must serialize to one tag");
            obj.keys().next().expect("one tag").clone()
        }
        other => panic!("unexpected ComputerAction serialization: {other}"),
    }
}

const EXPECTED_ACTION_SHAPE: &str = "one of: {\"observe\":{\"target\":\"a1\"}}, \
    {\"screenshot\":{\"target\":\"screen\"}}, \
    {\"zoom\":{\"target\":\"a1\",\"region\":[x0,y0,x1,y1]}}, {\"apps\":{}}, \
    {\"windows\":{\"app\":\"Safari\"}}, \
    {\"click\":{\"ref\":\"e3\"} or {\"x\":<0-1000>,\"y\":<0-1000>}}, \
    {\"type\":{\"text\":\"...\"}}, {\"press\":{\"keys\":\"cmd+shift+t\"}}, \
    {\"scroll\":{\"direction\":\"down\",\"amount\":3}}, \
    {\"drag\":{\"from\":[x,y],\"to\":[x,y]}}, {\"cursor\":{}}, {\"wait\":{\"seconds\":2}}";

/// Corrective error for an unrecoverable action shape, echoing the exact
/// expected form so the model can self-correct in one round-trip.
fn corrective_action_error(received: &Value) -> String {
    format!(
        "Invalid computer action arguments. Expected action to be {EXPECTED_ACTION_SHAPE}. \
         Received: {received}"
    )
}

/// Tolerant action normalization, mirroring the browser tool: a canonical tagged
/// object `{"click":{...}}`, a plain action name with flattened sibling fields
/// `{"action":"click","x":0,"y":0}`, or a stringified JSON object.
fn normalize_action(action: Value, args: &Value) -> Result<(Value, Option<String>), String> {
    match action {
        Value::Object(map) if map.len() == 1 => {
            let (name, inner) = map.into_iter().next().expect("len == 1");
            if !ACTION_WIRE_NAMES.iter().any(|n| n == &name) {
                return Err(corrective_action_error(&json!({name: inner})));
            }
            match inner {
                Value::Object(o) => Ok((json!({name: o}), None)),
                _ => Err(corrective_action_error(&json!({name: inner}))),
            }
        }
        Value::String(s) => {
            let s = s.trim();
            if s.starts_with('{') {
                let parsed: Value = serde_json::from_str(s)
                    .map_err(|_| corrective_action_error(&Value::String(s.to_string())))?;
                return normalize_action(parsed, args);
            }
            if ACTION_WIRE_NAMES.iter().any(|n| n == s) {
                return build_action_from_siblings(s, args);
            }
            Err(corrective_action_error(&Value::String(s.to_string())))
        }
        other => Err(corrective_action_error(&other)),
    }
}

/// Build a tagged action object from a plain action name plus sibling fields,
/// e.g. `{"action":"click","x":0,"y":0}` → `{"click":{"x":0,"y":0}}`.
fn build_action_from_siblings(name: &str, args: &Value) -> Result<(Value, Option<String>), String> {
    let Some(obj) = args.as_object() else {
        return Err(corrective_action_error(args));
    };
    let mut siblings = serde_json::Map::new();
    for (k, v) in obj {
        if k == "action" {
            continue;
        }
        siblings.insert(k.clone(), v.clone());
    }
    if let Some(payload) = siblings.remove(name) {
        let note = format!("sibling '{name}' object used as action payload");
        return match payload {
            Value::Object(o) => Ok((json!({name: o}), Some(note))),
            _ => Err(corrective_action_error(&Value::Object(siblings))),
        };
    }
    Ok((
        json!({name: siblings}),
        Some(format!("flattened fields wrapped into {name} action")),
    ))
}

// ── Backend factory ─────────────────────────────────────────────────────

fn backend() -> &'static dyn Backend {
    #[cfg(target_os = "macos")]
    {
        &macos::MACOS_BACKEND
    }
    #[cfg(target_os = "linux")]
    {
        &linux::LINUX_BACKEND
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        &stub::STUB_BACKEND
    }
}

// ── Capability probe ────────────────────────────────────────────────────

/// Cheap, synchronous availability check: the tool is advertised only when the
/// accessibility channel is trusted. Re-run at every agent construction, so a
/// grant granted later is picked up by newly constructed agents.
pub(crate) fn is_advertised() -> bool {
    backend().accessibility_available()
}

type CaptureProbe = (Instant, bool);

/// Cached capture-channel probe result with its timestamp.
static CAPTURE_CACHE: LazyLock<std::sync::Mutex<Option<CaptureProbe>>> =
    LazyLock::new(|| std::sync::Mutex::new(None));

/// TTL for a cached capture probe; an expired cache is treated as empty so a
/// fresh probe runs.
const CAPTURE_PROBE_TTL: Duration = Duration::from_secs(60);

/// Probe the capture channel and cache the result. Spawned once at daemon start
/// (see `spawn_background_tasks` in `main.rs`); `cached_capture_available()`
/// reads it.
pub async fn warm_capture_probe() {
    let ok = backend().capture_available().await;
    set_capture_cache(ok);
}

fn set_capture_cache(ok: bool) {
    let mut guard = CAPTURE_CACHE.lock().unwrap_poison();
    *guard = Some((Instant::now(), ok));
}

/// Cached capture availability, or `None` when empty/expired (→ re-probe).
pub(crate) fn cached_capture_available() -> Option<bool> {
    let guard = CAPTURE_CACHE.lock().unwrap_poison();
    let (ts, ok) = guard.as_ref()?;
    if ts.elapsed() > CAPTURE_PROBE_TTL {
        return None;
    }
    Some(*ok)
}

// ── Process-wide action serialization ───────────────────────────────────

static ACTION_LOCK: LazyLock<tokio::sync::Mutex<()>> =
    LazyLock::new(|| tokio::sync::Mutex::new(()));

/// Last screenshot/zoom path per agent, keyed by [`core::agent_key`]. A
/// concurrent agent's capture can never be re-attached by another agent's
/// `image_payload`. Entries are replaced per agent on each capture and left
/// stale (the last one is what `image_payload` wants).
static CAPTURE_PATHS: LazyLock<std::sync::Mutex<HashMap<String, String>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

/// Drop an agent's per-agent registry entries (observation/target/capture) at
/// agent-run end, so state never leaks across runs. Called from `run_agent`
/// with the agent's id string (matching [`core::agent_key`]'s task-local value).
pub(crate) fn cleanup_agent_state(agent_key: &str) {
    CAPTURE_PATHS.lock().unwrap_poison().remove(agent_key);
    core::clear_agent_state(agent_key);
}

/// Tool for observing and acting on the local GUI via the OS accessibility
/// channel. Every action is serialized through [`ACTION_LOCK`].
#[derive(Default)]
pub(crate) struct ComputerTool;

#[async_trait]
impl Tool for ComputerTool {
    fn name(&self) -> &'static str {
        "computer"
    }

    /// Every computer output is credential-scrubbed inside the tool: `observe`
    /// scrubs the tree text before its spill; all other actions are scrubbed once
    /// at the dispatch boundary. The agent-level pass must therefore not scrub
    /// again (mirrors the shell/read scrub-internally rule).
    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        false
    }

    fn is_advertised(&self) -> bool {
        self::is_advertised()
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "oneOf": [
                        super::action_entry_schema("observe", "Get the accessibility tree of a target surface with element refs (e1, e2, ...). Refs expire on re-observe.", &[], &json!({
                            "target": {"type": "string", "description": "Target id (a1/w1) or \"screen\". Default: focused window of the frontmost app."}
                        })),
                        super::action_entry_schema("screenshot", "Capture the target surface as a PNG and inject it as a native image; requires a vision-capable model.", &[], &json!({
                            "target": {"type": "string", "description": "Target id (a1/w1) or \"screen\". Default: focused window of the frontmost app."}
                        })),
                        super::action_entry_schema("zoom", "Capture and crop a normalized region of the target surface as a PNG.", &["region"], &json!({
                            "target": {"type": "string", "description": "Target id (a1/w1) or \"screen\". Default: focused window of the frontmost app."},
                            "region": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}, "description": "Normalized [x0, y0, x1, y1] region (0-1000), relative to the target surface."}
                        })),
                        super::action_entry_schema("apps", "List running GUI applications as a1-style targets.", &[], &json!({})),
                        super::action_entry_schema("windows", "List windows, optionally filtered to an app, as w1-style targets.", &[], &json!({
                            "app": {"type": "string", "description": "App name, pid, or a stored a-id to filter windows."}
                        })),
                        super::action_entry_schema("click", "Click by element ref (preferred) or normalized coordinates.", &[], &json!({
                            "target": {"type": "string", "description": "Target id (a1/w1) or \"screen\". Default: focused window of the frontmost app."},
                            "ref": {"type": "string", "description": "Element ref from the most recent observe; preferred over coordinates."},
                            "x": {"type": "number", "description": "Normalized 0-1000 X, relative to the target surface."},
                            "y": {"type": "number", "description": "Normalized 0-1000 Y, relative to the target surface."},
                            "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "Mouse button; coordinate clicks only. Default: left."},
                            "double": {"type": "boolean", "description": "Double-click; coordinate clicks only. Default: false."},
                            "modifiers": {"type": "array", "items": {"enum": ["cmd", "ctrl", "alt", "shift"]}, "description": "Keyboard modifiers; apply only to coordinate clicks."}
                        })),
                        super::action_entry_schema("type", "Set an element's value (with ref) or type into the focused element (without).", &["text"], &json!({
                            "text": {"type": "string", "description": "Text to set/type."},
                            "target": {"type": "string", "description": "Target id (a1/w1) or \"screen\". Default: focused window of the frontmost app."},
                            "ref": {"type": "string", "description": "Element ref from the most recent observe; sets its value."}
                        })),
                        super::action_entry_schema("press", "Press a keyboard chord, e.g. \"cmd+shift+t\", \"return\", \"ctrl+c\".", &["keys"], &json!({
                            "keys": {"type": "string", "description": "Chord to press, e.g. \"cmd+shift+t\", \"return\", \"ctrl+c\"."}
                        })),
                        super::action_entry_schema("scroll", "Scroll the target surface.", &["direction"], &json!({
                            "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                            "amount": {"type": "integer", "description": "Scroll ticks. Default: 3."},
                            "target": {"type": "string", "description": "Target id (a1/w1) or \"screen\". Default: focused window of the frontmost app."},
                            "x": {"type": "number", "description": "Normalized 0-1000 X anchor; provide both x and y together, or omit both."},
                            "y": {"type": "number", "description": "Normalized 0-1000 Y anchor; provide both x and y together, or omit both."}
                        })),
                        super::action_entry_schema("drag", "Drag from one normalized point to another on the target surface.", &["from", "to"], &json!({
                            "from": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}, "description": "[x, y] normalized 0-1000."},
                            "to": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}, "description": "[x, y] normalized 0-1000."},
                            "target": {"type": "string", "description": "Target id (a1/w1) or \"screen\". Default: focused window of the frontmost app."}
                        })),
                        super::action_entry_schema("cursor", "Report the pointer position as normalized and absolute points.", &[], &json!({})),
                        super::action_entry_schema("wait", "Pause before the next action (greater than 0, at most 10 seconds).", &["seconds"], &json!({
                            "seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 10, "description": "Seconds to wait, in (0, 10]."}
                        })),
                    ]
                }
            },
            "required": ["action"],
        })
    }

    async fn execute(&self, _ws: &Workspace, args: Value) -> anyhow::Result<String> {
        let mut notes: Vec<String> = Vec::new();

        let action_value = args
            .get("action")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!(corrective_action_error(&args)))?;
        let (action_value, normalized_note) =
            normalize_action(action_value, &args).map_err(anyhow::Error::msg)?;
        if let Some(note) = normalized_note {
            notes.push(format!("action normalized: {note}"));
        }

        let action: ComputerAction = serde_json::from_value(action_value.clone()).map_err(|e| {
            anyhow::anyhow!(
                "Invalid computer action arguments. Expected action to be {EXPECTED_ACTION_SHAPE}. \
                 Serde error: {e}"
            )
        })?;

        debug!(action = ?action, "computer action");

        // Serialize every GUI-touching action so concurrent agent runs can't race.
        let _lock = ACTION_LOCK.lock().await;

        let output = dispatch(&action).await?;
        Ok(Self::with_normalization_notes(output, &notes))
    }

    async fn image_payload(
        &self,
        _ws: &Workspace,
        args: &serde_json::Value,
    ) -> Option<crate::tools::ImagePayload> {
        // Only a successful `screenshot`/`zoom` produces a payload. Re-parse the
        // action so a stale capture from a prior round is never re-attached to a
        // non-capture call. The path is looked up per agent so two concurrent
        // agents never share a screenshot.
        let action_value = args.get("action")?.clone();
        let (action_value, _) = normalize_action(action_value, args).ok()?;
        let action: ComputerAction = serde_json::from_value(action_value).ok()?;
        if !matches!(
            action,
            ComputerAction::Screenshot { .. } | ComputerAction::Zoom { .. }
        ) {
            return None;
        }
        let path = CAPTURE_PATHS
            .lock()
            .unwrap_poison()
            .get(&core::agent_key())
            .cloned()?;
        let p = PathBuf::from(&path);
        let meta = crate::util::local_image_to_compressed_data_uri_with_meta(&p)
            .await
            .ok()?;
        Some(crate::tools::ImagePayload::from_compressed_meta(
            &p,
            meta,
            None,
            crate::tools::ImagePayloadSource::Computer,
        ))
    }
}

impl ComputerTool {
    /// Prepend a note about argument normalization so the model can see what was
    /// silently corrected.
    fn with_normalization_notes(output: String, notes: &[String]) -> String {
        if notes.is_empty() {
            return output;
        }
        format!("[normalized] {}\n{output}", notes.join("; "))
    }
}

// ── Action dispatch ─────────────────────────────────────────────────────

async fn dispatch(action: &ComputerAction) -> anyhow::Result<String> {
    let output = match action {
        ComputerAction::Apps {} => apps().await?,
        ComputerAction::Windows { app } => windows(app.as_deref()).await?,
        // observe scrubs internally (pre-spill); everything else is scrubbed once
        // at the dispatch boundary below.
        ComputerAction::Observe { target } => return observe(target.as_deref()).await,
        ComputerAction::Screenshot { target } => capture(target.as_deref(), None).await?,
        ComputerAction::Zoom { target, region } => capture(target.as_deref(), Some(region)).await?,
        ComputerAction::Click {
            target,
            reference,
            x,
            y,
            button,
            double,
            modifiers,
        } => {
            click(
                target.as_deref(),
                reference.as_deref(),
                *x,
                *y,
                *button,
                *double,
                modifiers.as_deref(),
            )
            .await?
        }
        ComputerAction::Type {
            text,
            target,
            reference,
        } => type_text(text, target.as_deref(), reference.as_deref()).await?,
        ComputerAction::Press { keys } => {
            backend()
                .raw_input(
                    &resolve_action_target(None)?,
                    core::RawInput::KeyChord {
                        chord: keys.clone(),
                    },
                )
                .await?;
            format!("pressed chord: {keys}")
        }
        ComputerAction::Scroll {
            direction,
            amount,
            target,
            x,
            y,
        } => scroll(*direction, *amount, target.as_deref(), *x, *y).await?,
        ComputerAction::Drag { from, to, target } => {
            let target = resolve_action_target(target.as_deref())?;
            backend()
                .raw_input(
                    &target,
                    core::RawInput::Drag {
                        from: *from,
                        to: *to,
                    },
                )
                .await?;
            format!("dragged from {from:?} to {to:?}")
        }
        ComputerAction::Cursor {} => cursor().await?,
        ComputerAction::Wait { seconds } => {
            if !seconds.is_finite() || *seconds <= 0.0 || *seconds > 10.0 {
                anyhow::bail!("wait seconds must be in (0, 10] — got {seconds}");
            }
            tokio::time::sleep(Duration::from_secs_f64(*seconds)).await;
            format!("waited {seconds}s")
        }
    };
    Ok(crate::util::scrub_credentials(&output))
}

/// Narrow a raw target argument to the surface to act on. `Some(t)` resolves
/// through the target registry as usual; `None` first reuses the target pinned by
/// the calling agent's most recent observation (so a focus shift between observe
/// and the action never lands coordinate actions on a different surface than the
/// refs came from), and only when there is no such pinned target does it fall
/// back to [`core::TargetSpec::Focused`].
fn resolve_action_target(target: Option<&str>) -> anyhow::Result<TargetSpec> {
    if let Some(t) = target {
        return core::resolve_target(Some(t));
    }
    Ok(core::last_observation_target().unwrap_or(TargetSpec::Focused))
}

async fn apps() -> anyhow::Result<String> {
    let apps = backend().list_apps().await?;
    let mut targets = Vec::with_capacity(apps.len());
    let mut lines = Vec::with_capacity(apps.len());
    for (i, app) in apps.iter().enumerate() {
        let id = format!("a{}", i + 1);
        // An app id targets the app's focused window (empty title + index 0).
        targets.push((
            id.clone(),
            TargetSpec::Window {
                app_pid: app.pid,
                app_name: app.name.clone(),
                title: String::new(),
                index: 0,
            },
        ));
        let pid = app.pid.map_or_else(|| "?".to_string(), |p| p.to_string());
        lines.push(format!("[{id}] {} (pid {pid})", app.name));
    }
    core::store_targets(targets);
    if lines.is_empty() {
        Ok("no apps found".to_string())
    } else {
        Ok(lines.join("\n"))
    }
}

async fn windows(app: Option<&str>) -> anyhow::Result<String> {
    let app_info = match app {
        Some(arg) => Some(resolve_app(arg).await?),
        None => None,
    };
    let wins = backend().list_windows(app_info.as_ref()).await?;
    let mut targets = Vec::with_capacity(wins.len());
    let mut lines = Vec::with_capacity(wins.len());
    for (i, win) in wins.iter().enumerate() {
        let id = format!("w{}", i + 1);
        targets.push((
            id.clone(),
            TargetSpec::Window {
                app_pid: win.app_pid,
                app_name: win.app_name.clone(),
                title: win.title.clone(),
                index: win.index,
            },
        ));
        lines.push(format!(
            "[{id}] {} — \"{}\" {}x{} @ ({}, {})",
            win.app_name,
            win.title,
            win.surface.width,
            win.surface.height,
            win.surface.x,
            win.surface.y
        ));
    }
    core::store_targets(targets);
    if lines.is_empty() {
        Ok("no windows found".to_string())
    } else {
        Ok(lines.join("\n"))
    }
}

/// Resolve a `windows` app argument: a stored a-id, a numeric pid, or a raw app
/// name. Always returns a fully-resolved [`AppInfo`] (with a pid) or a clear
/// error, so the macOS backend never silently skips a pid-less app.
async fn resolve_app(arg: &str) -> anyhow::Result<AppInfo> {
    let trimmed = arg.trim();
    if trimmed == "screen" {
        anyhow::bail!("'screen' is a surface, not an app — use the 'apps' action to list apps");
    }
    if let Some(TargetSpec::Window {
        app_pid, app_name, ..
    }) = core::get_target(trimmed)
    {
        return Ok(AppInfo {
            pid: app_pid,
            name: app_name,
        });
    }
    let apps = backend().list_apps().await?;
    if let Ok(pid) = trimmed.parse::<u32>() {
        return apps
            .iter()
            .find(|a| a.pid == Some(pid))
            .cloned()
            .ok_or_else(|| no_app_error(trimmed, &apps));
    }
    if let Some(app) = apps.iter().find(|a| a.name.eq_ignore_ascii_case(trimmed)) {
        return Ok(app.clone());
    }
    Err(no_app_error(trimmed, &apps))
}

fn no_app_error(name: &str, apps: &[AppInfo]) -> anyhow::Error {
    if apps.is_empty() {
        anyhow::anyhow!(
            "no running applications found — start '{name}' or list apps with the 'apps' action"
        )
    } else {
        let candidates = apps
            .iter()
            .take(3)
            .map(|a| a.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        anyhow::anyhow!(
            "no running application matches '{name}' — running apps: {candidates}; use the \
             'apps' action for exact ids"
        )
    }
}

/// A short human label for a target, used in argument-mismatch errors.
fn describe_target(spec: &TargetSpec) -> String {
    match spec {
        TargetSpec::Focused => "the focused window".to_string(),
        TargetSpec::Window {
            app_name, title, ..
        } if !title.is_empty() => {
            format!("window '{title}' of {app_name}")
        }
        TargetSpec::Window { app_name, .. } => format!("an app-level window of {app_name}"),
        TargetSpec::Screen => "the screen".to_string(),
    }
}

/// A ref-based action pins the observed surface; an explicit target that
/// resolves to a DIFFERENT surface is almost always a mistake (the model kept
/// both arguments). An equal target is fine.
fn ensure_ref_target_match(
    reference: &str,
    target: Option<&str>,
    cache_target: &TargetSpec,
) -> anyhow::Result<()> {
    let Some(t) = target else {
        return Ok(());
    };
    let given = core::resolve_target(Some(t))?;
    if given != *cache_target {
        anyhow::bail!(
            "ref {reference} belongs to the surface observed for {} — drop the target argument \
             or re-observe the target you want",
            describe_target(cache_target)
        );
    }
    Ok(())
}

async fn observe(target: Option<&str>) -> anyhow::Result<String> {
    let raw_target = core::resolve_target(target)?;
    // Pin a `Focused` observation to the concrete window at observe time so a
    // later ref-resolved action re-resolves THIS window (and fails stale if it
    // closed) instead of whatever happens to be focused then.
    let target = match raw_target {
        TargetSpec::Focused => TargetSpec::from_window(&backend().focused_window().await?),
        other => other,
    };
    let obs = backend().observe(&target).await?;
    core::store_observation(target, &obs);

    // render_tree emits the "[Surface: ...]" header (with node/interactive
    // counts), so compose the full output here and scrub it BEFORE any spill.
    let mut output = core::render_tree(&obs);
    if core::is_ax_thin(&obs) {
        // An AX-thin surface with an unavailable capture channel combines both
        // diagnoses in one note — the screenshot fallback would dead-end.
        let note = if cached_capture_available() == Some(false) {
            format!(
                "surface exposes few actionable elements AND screen capture is unavailable: {}",
                backend().capture_unavailable_error()
            )
        } else {
            "surface exposes few actionable elements — use screenshot (requires a vision-capable \
             model)"
                .to_string()
        };
        output.push_str(&note);
        output.push('\n');
    }
    // Spill ONLY the fully scrubbed output; try_spill_to_file returns the input
    // unchanged when under budget, so no separate pre-check is needed.
    Ok(crate::tools::shell::try_spill_to_file(
        crate::util::scrub_credentials(&output),
        TOOL_OUTPUT_BUDGET_BYTES,
    ))
}

async fn click(
    target: Option<&str>,
    reference: Option<&str>,
    x: Option<f64>,
    y: Option<f64>,
    button: Option<MouseButton>,
    double: Option<bool>,
    modifiers: Option<&[core::Modifier]>,
) -> anyhow::Result<String> {
    if let Some(reference) = reference {
        // Reject only values that would change behavior vs the canonical element
        // press: a non-left button, a double-click, or non-empty modifiers. An
        // explicit Left button / false double-click / empty modifiers list are
        // redundant with the one-form element press and are accepted.
        if modifiers.is_some_and(|m| !m.is_empty())
            || button.is_some_and(|b| b != MouseButton::Left)
            || double.is_some_and(|d| d)
        {
            anyhow::bail!(
                "{}",
                taxonomy_error(
                    core::ERR_UNSUPPORTED,
                    "button/double/modifiers apply only to coordinate clicks",
                )
            );
        }
        let (locator, cache_target) = core::resolve_ref(reference)?;
        ensure_ref_target_match(reference, target, &cache_target)?;
        backend()
            .act_on_element(&cache_target, &locator, ElementAct::Press)
            .await?;
        return Ok(format!("clicked element {reference}"));
    }
    let (Some(x), Some(y)) = (x, y) else {
        anyhow::bail!("click requires an element ref (preferred) or normalized x/y coordinates");
    };
    let target = resolve_action_target(target)?;
    let point = (x, y);
    backend()
        .raw_input(
            &target,
            core::RawInput::Click {
                point,
                button: button.unwrap_or(MouseButton::Left),
                double: double.unwrap_or(false),
                modifiers: modifiers.unwrap_or_default().to_vec(),
            },
        )
        .await?;
    Ok(format!("clicked at ({x}, {y})"))
}

async fn type_text(
    text: &str,
    target: Option<&str>,
    reference: Option<&str>,
) -> anyhow::Result<String> {
    if let Some(reference) = reference {
        let (locator, cache_target) = core::resolve_ref(reference)?;
        ensure_ref_target_match(reference, target, &cache_target)?;
        backend()
            .act_on_element(
                &cache_target,
                &locator,
                ElementAct::SetValue {
                    text: text.to_string(),
                },
            )
            .await?;
        return Ok(format!("set value on element {reference}"));
    }
    let target = resolve_action_target(target)?;
    backend()
        .raw_input(
            &target,
            core::RawInput::TypeText {
                text: text.to_string(),
            },
        )
        .await?;
    Ok(format!("typed text ({} chars)", text.chars().count()))
}

async fn scroll(
    direction: ScrollDirection,
    amount: Option<u32>,
    target: Option<&str>,
    x: Option<f64>,
    y: Option<f64>,
) -> anyhow::Result<String> {
    let target = resolve_action_target(target)?;
    let amount = amount.unwrap_or(3);
    if amount == 0 {
        anyhow::bail!("scroll amount must be > 0");
    }
    let point = match (x, y) {
        (Some(x), Some(y)) => Some((x, y)),
        (Some(_), None) | (None, Some(_)) => {
            anyhow::bail!("scroll requires both x and y or neither");
        }
        (None, None) => None,
    };
    backend()
        .raw_input(
            &target,
            core::RawInput::Scroll {
                point,
                direction,
                amount,
            },
        )
        .await?;
    Ok(format!("scrolled {amount} ticks {direction:?}"))
}

async fn cursor() -> anyhow::Result<String> {
    let target = resolve_action_target(None)?;
    let (ax, ay) = backend().cursor_position().await?;
    // When the default (Focused) surface has no focused window (desktop
    // frontmost/screensaver) its geometry is unavailable — report the absolute
    // pointer with no normalization basis rather than failing not-matched.
    let Ok(geo) = backend().surface_geometry(&target).await else {
        return Ok(format!(
            "cursor at absolute ({ax:.1}, {ay:.1}) — no focused window to normalize against"
        ));
    };
    if core::contains_point(&geo, ax, ay) {
        let (nx, ny) = core::surface_to_normalized(ax, ay, &geo)?;
        Ok(format!(
            "cursor at normalized ({nx:.1}, {ny:.1}) — absolute ({ax:.1}, {ay:.1})"
        ))
    } else {
        Ok(format!(
            "cursor outside the target surface ({ax:.1}, {ay:.1}) — surface bounds x:[{:.1}, {:.1}] y:[{:.1}, {:.1}]",
            geo.x,
            geo.x + geo.width,
            geo.y,
            geo.y + geo.height
        ))
    }
}

/// Capture the target surface (optionally cropping a normalized region) to a PNG
/// under the managed temp dir, record its path, and return a summary. The capture
/// probe is cached: a successful capture always refreshes the cache; a failed
/// capture on a cold cache probes once and caches the grant state, so a
/// genuinely-grantless setup maps failures to the actionable
/// [`Backend::capture_unavailable_error`] while a real success upgrades the
/// cache immediately — a stale cached negative never locks out a working channel.
async fn capture(target: Option<&str>, region: Option<&Region>) -> anyhow::Result<String> {
    let target = core::resolve_target(target)?;

    let capture_result = backend().capture(&target).await;
    let cap = match capture_result {
        Ok(cap) => {
            // A successful capture proves the channel works — always refresh the
            // cache (overwriting a cached negative) so a fresh probe isn't
            // re-run and later failures map to the real grant state.
            set_capture_cache(true);
            cap
        }
        Err(e) => {
            // First failure in this process: probe once and cache the grant
            // state. A cached negative means the grant is genuinely missing —
            // map subsequent failures to the actionable capture error. A cached
            // positive (or a freshly re-probed positive) means this failure is
            // transient — surface the original error.
            if cached_capture_available().is_none() {
                let ok = backend().capture_available().await;
                set_capture_cache(ok);
            }
            if cached_capture_available() == Some(false) {
                return Err(backend().capture_unavailable_error());
            }
            return Err(e);
        }
    };

    let (cropped, rect_opt) = match region {
        Some(region) => {
            let geo = backend().surface_geometry(&target).await?;
            let rect = core::region_to_pixels(region, &geo, cap.width, cap.height)?;
            (core::crop_rgba(&cap, rect)?, Some(rect))
        }
        None => (cap, None),
    };

    let (cw, ch) = (cropped.width, cropped.height);
    let path = encode_png(cropped)?;
    // Record the path under the owning agent id (owner-deletes-at-end via the
    // existing spill machinery) and stash it in the per-agent capture map.
    crate::tools::shell::record_spill_owner(path.clone());
    CAPTURE_PATHS
        .lock()
        .unwrap_poison()
        .insert(core::agent_key(), path.display().to_string());
    let kind = if rect_opt.is_some() {
        "zoom"
    } else {
        "screenshot"
    };
    let rect_note = rect_opt.map_or_else(String::new, |(x, y, w, h)| {
        format!(" (cropped rect {x},{y},{w},{h})")
    });
    let summary = format!("{kind} capture: {cw}x{ch} — {}{rect_note}", path.display());
    Ok(summary)
}

/// Encode a capture to PNG on disk (CPU + fs work offloaded with
/// [`crate::util::with_block_in_place`]), returning the random-nonce path under
/// the managed agent temp `computer/` dir.
fn encode_png(cap: Capture) -> anyhow::Result<PathBuf> {
    let Capture {
        width,
        height,
        rgba,
    } = cap;
    crate::util::with_block_in_place(move || {
        let img = image::RgbaImage::from_raw(width, height, rgba)
            .ok_or_else(|| anyhow::anyhow!("capture dimensions mismatch"))?;
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .map_err(|e| anyhow::anyhow!("failed to encode screen capture as PNG: {e}"))?;
        let dir = crate::tools::shell::agent_temp_dir()
            .ok_or_else(|| anyhow::anyhow!("no agent temp dir available for screenshots"))?
            .join("computer");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create screenshot dir {}", dir.display()))?;
        let nonce = rand::random::<u64>();
        let path = dir.join(format!("screenshot_{nonce:016x}.png"));
        std::fs::write(&path, &buf)
            .with_context(|| format!("failed to write screenshot {}", path.display()))?;
        Ok::<PathBuf, anyhow::Error>(path)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal payload per variant, used to build a canonical tagged action
    /// (`{"name": payload}`) for the deserialization half of the lockstep check.
    fn action_payload(name: &str) -> Value {
        match name {
            "observe" | "screenshot" | "windows" => json!({}),
            "apps" | "cursor" => json!({}),
            "zoom" => json!({"region": [0, 0, 1, 1]}),
            "click" => json!({"x": 1, "y": 1}),
            "type" => json!({"text": "x"}),
            "press" => json!({"keys": "a"}),
            "scroll" => json!({"direction": "down"}),
            "drag" => json!({"from": [0, 0], "to": [1, 1]}),
            "wait" => json!({"seconds": 1}),
            other => panic!("no lockstep payload for {other}"),
        }
    }

    #[test]
    fn parameters_schema_enum_lockstep() {
        let schema = ComputerTool.parameters_schema();
        let action_schemas = schema["properties"]["action"]["oneOf"]
            .as_array()
            .expect("oneOf should be an array");

        let enum_names: Vec<&str> = ACTION_WIRE_NAMES.iter().map(String::as_str).collect();
        let one_of_names: Vec<&str> = action_schemas
            .iter()
            .filter_map(|s| {
                s.get("properties")
                    .and_then(|p| p.as_object())
                    .and_then(|props| props.keys().next())
                    .map(String::as_str)
            })
            .collect();

        // The schema's oneOf list matches the enum exactly — no extras, no gaps.
        assert_eq!(
            enum_names.len(),
            one_of_names.len(),
            "oneOf count differs from the enum variant count"
        );
        for name in &enum_names {
            assert!(
                one_of_names.contains(name),
                "oneOf entry is missing '{name}'"
            );
        }
        for name in &one_of_names {
            assert!(
                enum_names.contains(name),
                "oneOf has an entry '{name}' that is not an enum variant"
            );
        }

        // Every wire name must normalize (be accepted by the allowlist) and its
        // canonical tagged form must round-trip deserialize. A new
        // `ComputerAction` variant fails here until a oneOf entry + payload are
        // added; the enum is the single source of truth for the name set.
        for name in &enum_names {
            let payload = action_payload(name);

            // Wrapped form, as the schema produces it in the model's call:
            // `{"action": {name: payload}}`.
            let wrapped_args = json!({"action": {*name: payload.clone()}});
            let (wrapped_norm, _) = normalize_action(wrapped_args["action"].clone(), &wrapped_args)
                .unwrap_or_else(|_| panic!("wrapped '{name}' should normalize"));

            // Bare-string form, with the payload spread as siblings — the
            // runtime's `build_action_from_siblings` flattening path:
            // `{"action": name, ...payload}`.
            let mut bare_args = json!({"action": *name});
            if let Value::Object(payload_obj) = &payload {
                bare_args
                    .as_object_mut()
                    .expect("bare args is an object")
                    .extend(payload_obj.clone());
            }
            let (bare_norm, _) = normalize_action(bare_args["action"].clone(), &bare_args)
                .unwrap_or_else(|_| panic!("bare '{name}' should normalize"));

            // Both forms must yield a tagged payload serde accepts for the
            // variant — this is the real runtime path and would fail for a
            // variant whose normalized payload mismatches its serde shape (e.g.
            // apps/cursor as unit variants rejecting the schema's `{}`).
            for normalized in [&wrapped_norm, &bare_norm] {
                let action: ComputerAction = serde_json::from_value(normalized.clone())
                    .unwrap_or_else(|e| {
                        panic!("normalized payload does not deserialize: {normalized} — {e}")
                    });
                assert_eq!(
                    wire_name(&action),
                    *name,
                    "deserialized variant wire name does not match '{name}'"
                );
            }
        }
    }
}
