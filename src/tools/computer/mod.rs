//! Computer (GUI observe/act) tool for the admin's Assistant.
//!
//! The platform-agnostic core (types, coordinate math, ref lifecycle, backend trait) lives in
//! [`core`]; [`linux`], [`macos`], [`windows`] and [`stub`] implement the backend. All GUI
//! actions are serialized through a process-wide lock so two agent runs cannot race on the
//! live GUI.
//!
//! The tool is offered on Linux, macOS and Windows. Linux has no grant model of its own; on
//! macOS the three privacy grants belong to the terminal the daemon was started from and are
//! reported on every action they block ([`macos`]); Windows needs no grant but refuses to
//! actuate on windows of elevated processes and cannot reach the secure desktop
//! ([`windows`]). Mahbot's own windows are targets like any other, but the system answers
//! accessibility requests for them inside this process, so each backend handles them deliberately.
//!
//! The agent-facing description is `src/prompt/tool/computer.md` plus only the running
//! platform's own file ([`PLATFORM_DESCRIPTION`]) — one installation is one platform. It
//! carries only what shapes an ordinary call; a caveat that matters only when a call goes
//! wrong belongs in the failure that raises it, as one line saying what to do next. That next
//! step names only a route that works in the state the call found
//! ([`capture_known_available`]).
//!
//! # Identifiers last for the session
//!
//! `apps` and `windows` answer with opaque ids (`a1754…`, `w1754…`) that later actions take as
//! a target. An id names that one surface for the whole session, is never handed to another
//! surface, and never points at something else. A reset ends them ([`end_session`], called
//! where the session store deletes the conversation) and so does a restart.
//!
//! Every use re-checks the surface against the live system: a surface that has gone is
//! `surface-gone` naming it, while an id the tool no longer knows says exactly that. A failed
//! lookup never claims an application is not running. Element refs are not ids: they belong to
//! the full observation they came from and expire with the turn that minted them
//! ([`core::cleanup_turn`]).
//!
//! # What is in front, what the tool remembers, what an action answers
//!
//! A listing names the frontmost application, marks the focused window and gives a
//! front-to-back order only where the platform can attribute one ([`core::WindowListing`]);
//! one record per surface ([`core::Look`]) is what every statement about that surface is
//! measured against; and input answers what it could observe afterwards, best-effort and
//! without ever claiming the effect ([`answer`]).
//!
//! # Bounded by the product, not by the machine
//!
//! This process aborts on a failed allocation, so no figure the machine reports may decide how
//! much is allocated: one [`core::CaptureBudget`] per capture call bounds every pixel buffer
//! the call holds at once, and every string a read keeps is capped at
//! [`core::MAX_VALUE_CHARS`] or [`core::MAX_NAME_CHARS`] where it is read.
//!
//! # Every argument is bounded
//!
//! Every [`ComputerAction`] argument is a value from a declared set or bounded by a range or a
//! length cap, and the bounds are checked before [`ACTION_LOCK`] is taken — so a bad call
//! costs a round trip, never a wait behind the lock, a capture, a permission prompt or a
//! lookup. An argument the chosen action cannot honour is refused with the form it does accept
//! ([`check_action_fields`], [`validate_action`]).
//!
//! # No call waits forever
//!
//! Every accessibility exchange is bounded by a deadline the product chose, and each backend
//! bounds a whole call on top of that. A bound that runs out always answers — naming the
//! surface and saying whether anything was applied, since an abandoned actuation may have
//! landed — so [`ACTION_LOCK`] is released by an answer and the next call proceeds.
//!
//! # A session that cannot receive the call is refused, not served
//!
//! A locked screen, a session another user switched away from and a machine with no
//! interactive desktop ask [`Backend::desktop_unavailable`] first and fail with
//! `desktop-unavailable`. Listing applications and windows, and `wait`, need no desktop. The
//! tag is not `degraded` on purpose: its guidance is to retry, and a retry into such a state
//! cannot succeed, so the model would loop instead of telling the owner.
//!
//! # Manual check
//!
//! Computer use is offered only to the admin's Assistant, so what is checked automatically is
//! the accounting, not the pixels: the budgets by the tests on [`core::CaptureBudget`], the
//! bounds by those on [`validate_action`] and [`check_action_fields`], and the other platforms
//! by the cross-compile lanes in `scripts/`. One live pass per platform covers the
//! screenshot/zoom/observe shapes, a target stopped with `kill -STOP` ending inside its
//! budget, an out-of-range argument refused before anything happens, and a locked screen
//! answering `desktop-unavailable` while `apps` still lists.

mod core;
mod linux;
mod macos;
mod windows;
// Every other platform: the capability is absent, so the stub is never advertised.
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
mod stub;

use self::core::{
    AppInfo, Backend, Capture, ElementAct, MouseButton, Region, ScrollDirection, TargetSpec,
};
use crate::util::{TOOL_OUTPUT_BUDGET_BYTES, UnwrapPoison};
use crate::{Tool, Workspace};
use anyhow::Context;
use async_trait::async_trait;
use serde::Deserialize;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::{Duration, Instant};
use strum::IntoEnumIterator;
use tracing::debug;

/// One computer tool call. All points/regions are normalized 0-1000 relative to
/// the surface the call names (see the `## Which surface` prompt section).
#[derive(Debug, Clone, Deserialize, Serialize, strum::EnumIter)]
#[serde(rename_all = "snake_case")]
enum ComputerAction {
    /// Read a surface. With no narrowing argument this is a full observation, the only thing
    /// that records a surface and mints refs; `ref`, `depth` and `kind` narrow it to part of the
    /// tree already read, and `changed` answers what differs from that recorded look.
    Observe {
        target: Option<String>,
        #[serde(rename = "ref")]
        reference: Option<String>,
        depth: Option<usize>,
        kind: Option<String>,
        changed: Option<bool>,
    },
    Screenshot {
        target: Option<String>,
    },
    Zoom {
        target: Option<String>,
        region: Region,
    },
    // NOTE: `Apps`/`Cursor` must stay EMPTY STRUCT variants (`{}`), not unit variants. The
    // canonical wire form is `{"apps": {}}`, which serde's externally-tagged representation
    // only accepts for a struct variant — a unit variant would reject the map, making both
    // actions dead on every input path.
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

/// Canonical wire names for every [`ComputerAction`] variant, derived from the enum itself via
/// `strum::EnumIter` + serde external tagging. Used as the [`normalize_action`] allowlist and by
/// the schema lockstep test.
static ACTION_WIRE_NAMES: LazyLock<Vec<String>> = LazyLock::new(|| {
    <ComputerAction as IntoEnumIterator>::iter()
        .map(|action| wire_name(&action))
        .collect()
});

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
    {\"observe\":{\"changed\":true}}, {\"observe\":{\"ref\":\"e3\",\"kind\":\"button\"}}, \
    {\"screenshot\":{\"target\":\"screen\"}}, \
    {\"zoom\":{\"target\":\"a1\",\"region\":[x0,y0,x1,y1]}}, {\"apps\":{}}, \
    {\"windows\":{\"app\":\"Safari\"}}, \
    {\"click\":{\"ref\":\"e3\"} or {\"x\":<0-1000>,\"y\":<0-1000>}}, \
    {\"type\":{\"text\":\"...\"}}, {\"press\":{\"keys\":\"cmd+shift+t\"}}, \
    {\"scroll\":{\"direction\":\"down\",\"amount\":3}}, \
    {\"drag\":{\"from\":[x,y],\"to\":[x,y]}}, {\"cursor\":{}}, {\"wait\":{\"seconds\":2}}";

/// Wheel ticks a `scroll` moves when `amount` is omitted.
const DEFAULT_SCROLL_AMOUNT: u32 = 3;

const MAX_WAIT_SECS: f64 = 10.0;

/// The modifiers a click names, one entry each in the schema — and the most entries a
/// click may carry, since every entry costs a key event under the action lock.
const MODIFIERS: [&str; 4] = ["cmd", "ctrl", "alt", "shift"];

/// One schema entry per [`ComputerAction`], built once: the model-facing schema and
/// [`check_action_fields`] read the same entries, so the arguments a call may carry cannot drift
/// from the interface the model is given. An entry says only what its action does and which of
/// its arguments the name alone does not settle.
static ACTION_ENTRIES: LazyLock<Vec<Value>> = LazyLock::new(|| {
    vec![
        super::action_entry_schema(
            "observe",
            "Read a surface's tree with element refs; narrow it, or ask what changed.",
            &[],
            &json!({
                "target": {"type": "string"},
                "ref": {"type": "string"},
                "depth": {"type": "integer", "minimum": 1, "maximum": core::MAX_TREE_DEPTH},
                "kind": {"type": "string"},
                "changed": {"type": "boolean"}
            }),
        ),
        super::action_entry_schema(
            "screenshot",
            "Capture a surface as an image.",
            &[],
            &json!({
                "target": {"type": "string"}
            }),
        ),
        super::action_entry_schema(
            "zoom",
            "Capture a crop; [x0, y0, x1, y1] normalized.",
            &["region"],
            &json!({
                "target": {"type": "string"},
                "region": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}}
            }),
        ),
        super::action_entry_schema("apps", "List applications with their ids.", &[], &json!({})),
        super::action_entry_schema(
            "windows",
            "List windows with their ids, optionally per app.",
            &[],
            &json!({
                "app": {"type": "string"}
            }),
        ),
        super::action_entry_schema(
            "click",
            "Click a ref, or a point by x and y; button, double and modifiers apply to a \
             point only.",
            &[],
            &json!({
                "target": {"type": "string"},
                "ref": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "button": {"type": "string", "enum": ["left", "right", "middle"]},
                "double": {"type": "boolean"},
                "modifiers": {"type": "array", "maxItems": MODIFIERS.len(), "items": {"enum": MODIFIERS}}
            }),
        ),
        super::action_entry_schema(
            "type",
            "Set an element's value, or type into focus.",
            &["text"],
            &json!({
                "text": {"type": "string"},
                "target": {"type": "string"},
                "ref": {"type": "string"}
            }),
        ),
        super::action_entry_schema(
            "press",
            "Press a key chord, e.g. \"ctrl+c\".",
            &["keys"],
            &json!({
                "keys": {"type": "string"}
            }),
        ),
        super::action_entry_schema(
            "scroll",
            "Scroll, optionally at a point.",
            &["direction"],
            &json!({
                "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                "amount": {"type": "integer", "minimum": 1, "maximum": core::MAX_SCROLL_AMOUNT},
                "target": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"}
            }),
        ),
        super::action_entry_schema(
            "drag",
            "Drag between two normalized points.",
            &["from", "to"],
            &json!({
                "from": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}},
                "to": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}},
                "target": {"type": "string"}
            }),
        ),
        super::action_entry_schema("cursor", "Report the pointer position.", &[], &json!({})),
        super::action_entry_schema(
            "wait",
            "Pause before the next action.",
            &["seconds"],
            &json!({
                "seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": MAX_WAIT_SECS}
            }),
        ),
    ]
});

/// Refuse a field the action does not take, before serde would drop it: the schema declares
/// `additionalProperties: false`, so an unlisted argument is a mistake to report. `action_value`
/// is the normalized tagged form; shape errors are left to the deserializer.
fn check_action_fields(action_value: &Value) -> anyhow::Result<()> {
    let Some((name, payload)) = action_value.as_object().and_then(|o| o.iter().next()) else {
        return Ok(());
    };
    let Some(payload) = payload.as_object() else {
        return Ok(());
    };
    let Some(entry) = ACTION_ENTRIES
        .iter()
        .find(|entry| entry["properties"].get(name.as_str()).is_some())
    else {
        return Ok(());
    };
    let accepted = entry["properties"][name.as_str()]["properties"]
        .as_object()
        .map(|props| props.keys().map(String::as_str).collect::<Vec<_>>())
        .unwrap_or_default();
    let Some(extra) = payload.keys().find(|key| !accepted.contains(&key.as_str())) else {
        return Ok(());
    };
    anyhow::bail!(
        "{name} does not take '{extra}' — its arguments are: {}",
        if accepted.is_empty() {
            "none".to_string()
        } else {
            accepted.join(", ")
        }
    )
}

/// Check every argument an action carries before the action lock is taken and before anything is
/// captured, prompted for or looked up: each string within [`core::MAX_ARG_CHARS`], each number
/// within its stated range, each combination one the action can honour.
fn validate_action(action: &ComputerAction) -> anyhow::Result<()> {
    match action {
        ComputerAction::Observe {
            target,
            reference,
            depth,
            kind,
            changed,
        } => {
            check_opt_arg("target", target.as_deref())?;
            check_observation(reference.as_deref(), *depth, kind.as_deref(), *changed)?;
        }
        ComputerAction::Screenshot { target } => check_opt_arg("target", target.as_deref())?,
        ComputerAction::Zoom { target, region } => {
            check_opt_arg("target", target.as_deref())?;
            core::check_region(region)?;
        }
        ComputerAction::Apps {} | ComputerAction::Cursor {} => {}
        ComputerAction::Windows { app } => check_opt_arg("app", app.as_deref())?,
        ComputerAction::Click {
            target,
            reference,
            x,
            y,
            button,
            double,
            modifiers,
        } => check_click(
            target.as_deref(),
            reference.as_deref(),
            *x,
            *y,
            *button,
            *double,
            modifiers.as_deref(),
        )?,
        ComputerAction::Type {
            text,
            target,
            reference,
        } => {
            core::check_arg_len("text", text)?;
            check_opt_arg("target", target.as_deref())?;
            check_opt_arg("ref", reference.as_deref())?;
        }
        ComputerAction::Press { keys } => core::check_arg_len("keys", keys)?,
        ComputerAction::Scroll {
            amount,
            target,
            x,
            y,
            ..
        } => {
            core::check_scroll_amount(amount.unwrap_or(DEFAULT_SCROLL_AMOUNT))?;
            check_opt_arg("target", target.as_deref())?;
            if coord_pair(*x, *y, "scroll anchor")?.is_none() && target.is_some() {
                anyhow::bail!(
                    "scroll without x and y does not address a surface — the wheel goes to \
                     whatever is under the pointer; pass x and y to scroll the surface you \
                     name, or drop target"
                );
            }
        }
        ComputerAction::Drag { from, to, target } => {
            core::check_point("drag start", from.0, from.1)?;
            core::check_point("drag end", to.0, to.1)?;
            check_opt_arg("target", target.as_deref())?;
        }
        ComputerAction::Wait { seconds } => {
            if !seconds.is_finite() || *seconds <= 0.0 || *seconds > MAX_WAIT_SECS {
                anyhow::bail!("wait seconds must be in (0, {MAX_WAIT_SECS}] — got {seconds}");
            }
        }
    }
    Ok(())
}

/// The three arguments that narrow an observation, and the one that replaces it with a change
/// answer.
fn check_observation(
    reference: Option<&str>,
    depth: Option<usize>,
    kind: Option<&str>,
    changed: Option<bool>,
) -> anyhow::Result<()> {
    check_opt_arg("ref", reference)?;
    check_opt_arg("kind", kind)?;
    if let Some(depth) = depth
        && !(1..=core::MAX_TREE_DEPTH).contains(&depth)
    {
        let max = core::MAX_TREE_DEPTH;
        anyhow::bail!("observe depth must be 1-{max} levels below the surface — got {depth}");
    }
    if changed.unwrap_or(false) && (reference.is_some() || depth.is_some() || kind.is_some()) {
        anyhow::bail!(
            "observe takes changed on its own: it answers what differs from the recorded look, \
             which a ref, depth or kind would narrow — drop them, or drop changed to look at \
             part of the tree"
        );
    }
    Ok(())
}

/// A click is one of two forms, and neither carries the other's arguments.
fn check_click(
    target: Option<&str>,
    reference: Option<&str>,
    x: Option<f64>,
    y: Option<f64>,
    button: Option<MouseButton>,
    double: Option<bool>,
    modifiers: Option<&[core::Modifier]>,
) -> anyhow::Result<()> {
    check_opt_arg("target", target)?;
    check_opt_arg("ref", reference)?;
    match (reference, x, y) {
        // An element press is one form: a button, a second click or a modifier would name a
        // gesture the ref-based path does not perform.
        (Some(reference), None, None) => {
            if modifiers.is_some_and(|m| !m.is_empty())
                || button.is_some_and(|b| b != MouseButton::Left)
                || double.is_some_and(|d| d)
            {
                anyhow::bail!(
                    "click on element {reference} takes no button, double or modifiers beyond the \
                     plain left click — those belong to coordinate clicks; drop them, or drop ref \
                     and give x and y to click a point"
                );
            }
        }
        (Some(_), _, _) => anyhow::bail!(
            "click takes a ref or x/y, never both — drop x and y to click the element, or drop ref \
             to click the point"
        ),
        (None, Some(x), Some(y)) => {
            core::check_point("click point", x, y)?;
            check_modifiers(modifiers)?;
        }
        (None, x, y) => anyhow::bail!(
            "click needs a ref or both x and y — got {}",
            match (x, y) {
                (Some(_), None) => "x alone",
                (None, Some(_)) => "y alone",
                _ => "neither",
            }
        ),
    }
    Ok(())
}

/// A coordinate pair argument: both members, or neither — half a point names no position.
fn coord_pair(x: Option<f64>, y: Option<f64>, what: &str) -> anyhow::Result<Option<(f64, f64)>> {
    match (x, y) {
        (Some(x), Some(y)) => {
            core::check_point(what, x, y)?;
            Ok(Some((x, y)))
        }
        (None, None) => Ok(None),
        (Some(_), None) | (None, Some(_)) => {
            anyhow::bail!("{what} needs both x and y, or neither")
        }
    }
}

fn check_opt_arg(field: &str, value: Option<&str>) -> anyhow::Result<()> {
    value.map_or(Ok(()), |value| core::check_arg_len(field, value))
}

/// Refuse a modifier list longer than the modifiers there are: every entry drives a key event
/// under the action lock.
fn check_modifiers(modifiers: Option<&[core::Modifier]>) -> anyhow::Result<()> {
    let Some(modifiers) = modifiers else {
        return Ok(());
    };
    if modifiers.len() > MODIFIERS.len() {
        anyhow::bail!(
            "click modifiers take at most {} entries, one for each of {} — got {}",
            MODIFIERS.len(),
            MODIFIERS.join(", "),
            modifiers.len()
        );
    }
    Ok(())
}

fn corrective_action_error(received: &Value) -> String {
    format!(
        "Invalid computer action arguments. Expected action to be {EXPECTED_ACTION_SHAPE}. \
         Received: {received}"
    )
}

/// Tolerant action normalization, mirroring the chrome tool: a canonical tagged object
/// `{"click":{...}}`, a plain action name with flattened sibling fields, or a stringified JSON
/// object.
fn normalize_action(action: Value, args: &Value) -> Result<(Value, Option<String>), String> {
    match action {
        Value::Object(map) if map.len() == 1 => {
            let (name, inner) = map.into_iter().next().expect("len == 1");
            if !ACTION_WIRE_NAMES.iter().any(|n| n == &name) {
                return Err(corrective_action_error(&json!({name: inner})));
            }
            if let Some(extra) = args
                .as_object()
                .and_then(|o| o.keys().find(|k| k.as_str() != "action"))
            {
                return Err(format!(
                    "unexpected argument '{extra}' beside the action — send the action and \
                     nothing else at the top level"
                ));
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
            Value::Object(o) => {
                if let Some(extra) = siblings.keys().next() {
                    return Err(format!(
                        "unexpected argument '{extra}' beside the '{name}' payload — send the \
                         payload or the flattened fields, not both"
                    ));
                }
                Ok((json!({name: o}), Some(note)))
            }
            _ => Err(corrective_action_error(&Value::Object(siblings))),
        };
    }
    Ok((
        json!({name: siblings}),
        Some(format!("flattened fields wrapped into {name} action")),
    ))
}

fn backend() -> &'static dyn Backend {
    #[cfg(target_os = "linux")]
    {
        &linux::LINUX_BACKEND
    }
    #[cfg(target_os = "macos")]
    {
        &macos::MACOS_BACKEND
    }
    #[cfg(target_os = "windows")]
    {
        &windows::WINDOWS_BACKEND
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        &stub::STUB_BACKEND
    }
}

/// Cheap, synchronous availability check for advertisement ([`Backend::is_supported`]).
pub(crate) fn is_advertised() -> bool {
    backend().is_supported()
}

type CaptureProbe = (Instant, bool);

static CAPTURE_CACHE: LazyLock<std::sync::Mutex<Option<CaptureProbe>>> =
    LazyLock::new(|| std::sync::Mutex::new(None));

/// TTL for a cached capture probe; an expired cache is treated as empty.
const CAPTURE_PROBE_TTL: Duration = Duration::from_secs(60);

/// Probe the capture channel and cache the result. Spawned once at daemon start
/// (`spawn_background_tasks` in `main.rs`); `cached_capture_available()` reads it.
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

/// Whether a still is known to work right now — the only state in which a result offers a
/// screenshot as the way forward. Unknown counts as "no", so advice must never treat it as
/// permission to recommend a still. Costs no platform work, so it is safe under [`ACTION_LOCK`].
pub(crate) fn capture_known_available() -> bool {
    cached_capture_available() == Some(true)
}

static ACTION_LOCK: LazyLock<tokio::sync::Mutex<()>> =
    LazyLock::new(|| tokio::sync::Mutex::new(()));

/// The last screenshot/zoom of each agent, keyed by [`core::agent_key`], so a concurrent agent's
/// capture is never re-attached by another agent's call.
static CAPTURES: LazyLock<std::sync::Mutex<HashMap<String, PathBuf>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

/// Drop the state one turn owns when its agent run ends: the capture waiting to be attached and
/// the element refs of its last observation. Application and window ids are kept —
/// [`end_session`] ends those. Called from the agent's run-end cleanup guard.
pub(crate) fn cleanup_turn_state(agent_key: &str) {
    CAPTURES.lock().unwrap_poison().remove(agent_key);
    core::cleanup_turn(agent_key);
}

/// End everything a reset session minted: its application and window ids, the surface it last
/// observed, and — on macOS — the window handles those ids resolved through, which have to go
/// with them or a live id would read as a closed window. Called from the session store's delete.
pub(crate) fn end_session(agent_key: &str) {
    core::end_session(agent_key);
    #[cfg(target_os = "macos")]
    macos::forget_agent_windows(agent_key);
}

/// The asset holding this platform's own half of the description. `None` where the tool is
/// never advertised (the stub backend), which leaves the shared text alone.
const PLATFORM_DESCRIPTION: Option<&str> = if cfg!(target_os = "linux") {
    Some("tool/computer_linux.md")
} else if cfg!(target_os = "macos") {
    Some("tool/computer_macos.md")
} else if cfg!(target_os = "windows") {
    Some("tool/computer_windows.md")
} else {
    None
};

/// Tool for observing and acting on the local GUI via the OS accessibility channel.
#[derive(Default)]
pub(crate) struct ComputerTool;

#[async_trait]
impl Tool for ComputerTool {
    fn name(&self) -> &'static str {
        "computer"
    }

    /// The shared contract plus this platform's own half — one installation is one platform.
    fn description(&self) -> String {
        let mut sections = vec![crate::prompt::load_prompt("tool/computer.md")];
        if let Some(asset) = PLATFORM_DESCRIPTION {
            sections.push(crate::prompt::load_prompt(asset));
        }
        sections
            .iter()
            .map(|s| s.trim_end())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Every computer output is credential-scrubbed inside the tool, so the agent-level pass must
    /// not scrub again (mirrors the shell/read scrub-internally rule).
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
                    "description": format!(
                        "One action per call; each string argument is capped at {} characters.",
                        core::MAX_ARG_CHARS
                    ),
                    "oneOf": Value::Array(ACTION_ENTRIES.clone()),
                }
            },
            "required": ["action"],
        })
    }

    async fn execute(&self, _ws: &Workspace, args: Value) -> anyhow::Result<String> {
        let action_value = args
            .get("action")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!(corrective_action_error(&args)))?;
        let (action_value, normalized_note) =
            normalize_action(action_value, &args).map_err(anyhow::Error::msg)?;
        if let Some(note) = normalized_note {
            super::log_normalization_note("computer", &note);
        }

        check_action_fields(&action_value)?;
        let action: ComputerAction = serde_json::from_value(action_value).map_err(|e| {
            anyhow::anyhow!(
                "Invalid computer action arguments. Expected action to be {EXPECTED_ACTION_SHAPE}. \
                 Serde error: {e}"
            )
        })?;
        validate_action(&action)?;

        debug!(action = ?action, "computer action");

        // A session that cannot receive the call is refused before the lock: it costs no
        // wait behind another agent's capture, and nothing was delivered or read.
        if needs_desktop(&action) {
            ensure_interactive_desktop().await?;
        }

        // Serialize every GUI-touching action so concurrent agent runs can't race. The probes an
        // answer makes after the input was delivered run outside it: a slow probe must not
        // lengthen another action's wait, nor count against Windows' confirmation deadline.
        let result = {
            let _lock = ACTION_LOCK.lock().await;
            dispatch(&action).await
        };
        // A failure that confirmed the surface gone drops the id, so the table follows only
        // surfaces that still exist.
        if let Err(err) = &result {
            forget_confirmed_gone(&action, err);
        }
        let delivered = result.map_err(|e| framework_next_step(&action, e))?;
        let answer = answer(delivered).await;
        Ok(crate::util::scrub_credentials(&answer))
    }

    /// The tool's text is a real result, never a claim about an image, so annotations are
    /// appended rather than replacing it. A capture whose image cannot be attached says so here.
    async fn execute_with_payloads(
        &self,
        ws: &crate::Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<crate::tools::ToolOutput> {
        let text = self.execute(ws, args.clone()).await?;
        let (text, image_payloads) = match captured_payload(&args).await {
            Ok(Some(payload)) => (text, vec![payload]),
            Ok(None) => (text, Vec::new()),
            Err(e) => (
                format!("{text}\nno image could be attached to this result: {e}"),
                Vec::new(),
            ),
        };
        Ok(crate::tools::ToolOutput {
            text,
            image_payloads,
            text_is_content: true,
            preserve_text: false,
        })
    }
}

/// The image a `screenshot`/`zoom` call produced, or `None` for every other action: the action
/// is re-parsed so a stale capture is never attached to a non-capture call, and the capture is
/// looked up per agent. Failing to attach is an error the caller must report.
async fn captured_payload(
    args: &serde_json::Value,
) -> anyhow::Result<Option<crate::tools::ImagePayload>> {
    let Some(action_value) = args.get("action").cloned() else {
        return Ok(None);
    };
    let Ok((action_value, _)) = normalize_action(action_value, args) else {
        return Ok(None);
    };
    let Ok(action) = serde_json::from_value::<ComputerAction>(action_value) else {
        return Ok(None);
    };
    if !matches!(
        action,
        ComputerAction::Screenshot { .. } | ComputerAction::Zoom { .. }
    ) {
        return Ok(None);
    }
    let Some(path) = CAPTURES
        .lock()
        .unwrap_poison()
        .get(&core::agent_key())
        .cloned()
    else {
        // Unreachable: a capture records its file before it can return success.
        return Ok(None);
    };
    let meta = crate::util::local_image_to_compressed_data_uri_with_meta(&path).await?;
    Ok(Some(crate::tools::ImagePayload::from_compressed_meta(
        &path,
        meta,
        None,
        crate::tools::ImagePayloadSource::Computer,
    )))
}

/// Whether an action needs the interactive desktop. Listing applications and windows and `wait`
/// describe the session rather than receive anything from it, so everything else is written as
/// an exclusion and an action added later is gated by default.
fn needs_desktop(action: &ComputerAction) -> bool {
    !matches!(
        action,
        ComputerAction::Apps {} | ComputerAction::Windows { .. } | ComputerAction::Wait { .. }
    )
}

/// Refuse a call whose session cannot receive it, before any surface is resolved or captured and
/// before [`ACTION_LOCK`] is taken: a state refusal is the one answer `cursor` must not degrade
/// into "no surface".
async fn ensure_interactive_desktop() -> anyhow::Result<()> {
    let Some(reason) = backend().desktop_unavailable().await else {
        return Ok(());
    };
    Err(core::taxonomy_error(
        core::ERR_DESKTOP_UNAVAILABLE,
        format!(
            "{reason} — this call needs an interactive desktop, so nothing was delivered or read; \
             tell the owner and run it again only once the session is back. Listing apps and \
             windows, and wait, still work"
        ),
    ))
}

/// The one place a `framework-failed` answer gains its next step: a listing or a cursor read is
/// worth one retry, a capture is not helped by being taken again, and the rest check the surface
/// first. The constructor stays plain because a framework failure during an actuation is folded
/// into `applied but unconfirmed`.
fn framework_next_step(action: &ComputerAction, err: anyhow::Error) -> anyhow::Error {
    if !core::is_framework_failed(&err) {
        return err;
    }
    let next = match action {
        ComputerAction::Apps {} | ComputerAction::Windows { .. } | ComputerAction::Cursor {} => {
            "retry it once, and report it if it repeats"
        }
        ComputerAction::Screenshot { .. } | ComputerAction::Zoom { .. } => {
            "re-observe the target if it is a window, then take another route or report it"
        }
        _ => {
            if capture_known_available() {
                "re-observe or screenshot to check the target's state, then take another route \
                 or report it"
            } else {
                "re-observe to check the target's state, then take another route or report it"
            }
        }
    };
    anyhow::anyhow!("{err} — the call was abandoned, so {next}")
}

async fn dispatch(action: &ComputerAction) -> anyhow::Result<Delivered> {
    match action {
        ComputerAction::Apps {} => Ok(Delivered::Read(apps().await?)),
        ComputerAction::Windows { app } => Ok(Delivered::Read(windows(app.as_deref()).await?)),
        // An observation spills and scrubs its own tree; every answer is scrubbed once more at the
        // boundary, which is also where the post-action report is appended.
        ComputerAction::Observe {
            target,
            reference,
            depth,
            kind,
            changed,
        } => {
            observe(
                target.as_deref(),
                reference.as_deref(),
                *depth,
                kind.clone(),
                changed.unwrap_or(false),
            )
            .await
        }
        ComputerAction::Screenshot { target } => {
            Ok(Delivered::Read(capture(target.as_deref(), None).await?))
        }
        ComputerAction::Zoom { target, region } => Ok(Delivered::Read(
            capture(target.as_deref(), Some(region)).await?,
        )),
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
            .await
        }
        ComputerAction::Type {
            text,
            target,
            reference,
        } => type_text(text, target.as_deref(), reference.as_deref()).await,
        ComputerAction::Press { keys } => press(keys).await,
        ComputerAction::Scroll {
            direction,
            amount,
            target,
            x,
            y,
        } => scroll(*direction, *amount, target.as_deref(), *x, *y).await,
        ComputerAction::Drag { from, to, target } => drag(*from, *to, target.as_deref()).await,
        ComputerAction::Cursor {} => Ok(Delivered::Read(cursor().await?)),
        ComputerAction::Wait { seconds } => {
            tokio::time::sleep(Duration::from_secs_f64(*seconds)).await;
            Ok(Delivered::Read(format!("waited {seconds}s")))
        }
    }
}

/// A chord is global input: it goes to the focused application, so no surface is resolved.
async fn press(keys: &str) -> anyhow::Result<Delivered> {
    let actuation = backend()
        .raw_input(
            &TargetSpec::Focused,
            core::RawInput::KeyChord {
                chord: keys.to_string(),
            },
        )
        .await?;
    Ok(Delivered::Global(core::impact_text(
        format!("pressed chord: {keys} — a chord goes to the focused application"),
        actuation.impact,
    )))
}

/// A drag states both ends of its gesture in the surface's own points, and the surface they were
/// mapped through as the backend that posted them reported it.
async fn drag(from: (f64, f64), to: (f64, f64), target: Option<&str>) -> anyhow::Result<Delivered> {
    let surface = resolve_in_place(target).await?;
    let gesture = core::RawInput::Drag { from, to };
    let actuation = backend().raw_input(&surface, gesture.clone()).await?;
    let mut text = match &actuation.placed {
        Some(placed) => format!(
            "dragged {}{} — input {} to {}",
            placed.describe(),
            on_surface(&surface, Some(placed)),
            point_text(from),
            point_text(to)
        ),
        None => format!(
            "dragged{}{UNREAD_POSITION} — input {} to {}",
            on_surface(&surface, None),
            point_text(from),
            point_text(to)
        ),
    };
    text.push_str(unpinned_clause(&surface, &gesture));
    Ok(Delivered::OnSurface {
        text: core::impact_text(text, actuation.impact),
        surface,
    })
}

/// What one call delivered, and therefore what its answer still owes the caller. A read is
/// answered as it is; input delivered to a surface carries that surface, so its answer reports
/// what could be observed after it, read once the action lock is released; input that addresses
/// no surface reports the focus only.
enum Delivered {
    Read(String),
    OnSurface { text: String, surface: TargetSpec },
    Global(String),
}

/// A point as an answer states it: the normalized pair, so a coordinate is never quoted without
/// the scale it was read in.
fn point_text(point: (f64, f64)) -> String {
    format!(
        "({}, {}) of 0-{:.0}",
        point.0,
        point.1,
        core::NORMALIZED_MAX
    )
}

/// The surface clause of an action result: the surface the call named, or the window the input
/// was actually mapped through, named by where it sits. `""` when neither is known.
fn on_surface(target: &TargetSpec, placed: Option<&core::Placement>) -> String {
    match target {
        TargetSpec::Focused => placed.map_or_else(String::new, |placed| {
            let geo = placed.geometry;
            format!(" on the focused window at ({}, {})", geo.x, geo.y)
        }),
        other => format!(" on {}", other.describe()),
    }
}

/// The clause a pointer gesture appends when it is not pinned to the surface: its coordinates
/// were mapped there, but the input lands wherever the pointer is. A coordinate click carries no
/// such clause — what it lands on is checked when it is sent ([`core::UnderPoint`]).
fn unpinned_clause(target: &TargetSpec, input: &core::RawInput) -> &'static str {
    if core::pinned_input(target, input) {
        ""
    } else {
        " — the input goes to whatever is under its coordinates"
    }
}

/// The final text of one call: what it did, and — for input that was delivered — what could be
/// observed after it. Every part is read best-effort: a probe that fails is stated as unread,
/// never turned into a failure of an action that already happened. The answer never claims that
/// the intended effect occurred.
async fn answer(delivered: Delivered) -> String {
    match delivered {
        Delivered::Read(text) => text,
        Delivered::Global(text) => {
            let focus = focus_text(None).await;
            format!("{text}\n[After: no surface addressed — {focus}]")
        }
        Delivered::OnSurface { text, surface } => {
            let focus = focus_text(Some(&surface)).await;
            let state = surface_state(&surface).await;
            format!("{text}\n[After: {} — {focus}; {state}]", surface.describe())
        }
    }
}

/// The element that holds the focus, as the answer states it: by kind and name, or the plain
/// statement that this read did not establish it.
async fn focus_text(surface: Option<&TargetSpec>) -> String {
    match backend().focused_element(surface).await {
        Some(element) => format!("focus: {}", element.describe()),
        // The read failed this time — a target that does not answer, or one with no focused
        // element. Every platform here can answer for some targets, so this is never stated as an
        // incapability of the platform.
        None => "focus: could not be read this time".to_string(),
    }
}

/// Why there is no remembered look of a surface. One clause for every case — never observed,
/// dropped for newer surfaces, discarded after the surface was confirmed gone — because all
/// three are true of it and the answer is not the place to explain the cache.
fn no_look() -> String {
    format!(
        "the tool keeps the {} most recently observed surfaces and holds no remembered look of \
         this one",
        core::MAX_LOOKS
    )
}

/// Whether the surface still sits where its remembered look found it — the one thing an action's
/// answer may say about movement.
async fn surface_state(surface: &TargetSpec) -> String {
    let Some((remembered, taken)) = core::remembered_geometry(surface) else {
        return format!("nothing to compare against — {}", no_look());
    };
    match backend().surface_geometry(surface).await {
        Ok(now) => match core::geometry_change(&remembered, &now) {
            Some(change) => format!(
                "the surface {change} since the observation taken {}",
                core::look_age(taken)
            ),
            None => format!(
                "the surface sits where the observation taken {} found it",
                core::look_age(taken)
            ),
        },
        Err(_) => "whether the surface moved could not be read".to_string(),
    }
}

/// The surface an action addresses: the explicit target, else the surface this session last
/// observed or captured, else whatever the system has focused — so a focus shift between observe
/// and the action never lands it on a different surface than the refs came from.
fn resolve_action_target(target: Option<&str>) -> anyhow::Result<TargetSpec> {
    if let Some(t) = target {
        return core::resolve_target(Some(t));
    }
    Ok(core::last_surface().unwrap_or(TargetSpec::Focused))
}

/// [`resolve_action_target`] with the surface made concrete: a call that ends up on the focused
/// window, or on an application id, names the one window it addresses. Everything recorded about a
/// surface is about one window, so an application address is made concrete before anything is
/// stored.
async fn resolve_surface(target: Option<&str>) -> anyhow::Result<TargetSpec> {
    let surface = resolve_in_place(target).await?;
    if !matches!(surface, TargetSpec::Focused) {
        return Ok(surface);
    }
    let win = backend().surface_window(&surface).await?;
    Ok(TargetSpec::from_window(&win))
}

/// The surface an action that may go to whatever holds the focus addresses — typing, a drag, a
/// wheel at a point. It is resolved exactly once and the input delivered through what it
/// resolved: an `a` id is made concrete, while a call that names and has observed nothing keeps
/// addressing the focused surface.
async fn resolve_in_place(target: Option<&str>) -> anyhow::Result<TargetSpec> {
    match resolve_action_target(target)? {
        app @ TargetSpec::App { .. } => {
            let win = backend().surface_window(&app).await?;
            Ok(TargetSpec::from_window(&win))
        }
        other => Ok(other),
    }
}

/// Drop the id a failed call was asked through when the failure confirmed that surface gone, and
/// forget the remembered look of that surface however the call addressed it. Only a confirmed
/// death counts: a busy or unanswering surface keeps both. This is what keeps the tables
/// following the desktop without polling.
fn forget_confirmed_gone(action: &ComputerAction, err: &anyhow::Error) {
    if !core::is_surface_gone(err) {
        return;
    }
    let named = match action {
        ComputerAction::Windows { app } => app.as_deref(),
        ComputerAction::Observe { target, .. }
        | ComputerAction::Screenshot { target }
        | ComputerAction::Zoom { target, .. }
        | ComputerAction::Click { target, .. }
        | ComputerAction::Type { target, .. }
        | ComputerAction::Scroll { target, .. }
        | ComputerAction::Drag { target, .. } => target.as_deref(),
        ComputerAction::Apps {}
        | ComputerAction::Press { .. }
        | ComputerAction::Cursor {}
        | ComputerAction::Wait { .. } => None,
    };
    let by_ref = match action {
        ComputerAction::Click {
            reference: Some(reference),
            ..
        }
        | ComputerAction::Type {
            reference: Some(reference),
            ..
        } => core::resolve_ref(reference).ok().map(|(_, target)| target),
        _ => None,
    };
    // The surface to forget: the one the ref belongs to, else the id the call named, else that
    // application's own records. The session's last surface is the fallback only when the call
    // named none: forgetting it for a named surface would drop an unrelated record.
    if let Some(surface) = by_ref.or_else(|| named.and_then(|arg| core::get_target(arg.trim()))) {
        core::forget_look(&surface);
    } else if let Some(arg) = named {
        core::forget_look_of_app_argument(arg);
    } else if let Some(surface) = core::last_surface() {
        core::forget_look(&surface);
    }
    if let Some(id) = named {
        core::forget_target(id.trim());
    }
}

async fn apps() -> anyhow::Result<String> {
    let listing = backend().list_apps().await?;
    // An a-id targets the app's own focused window, re-resolved at act time.
    let surfaces = listing
        .apps
        .iter()
        .map(|app| TargetSpec::App {
            app_pid: app.pid,
            app_name: app.name.clone(),
        })
        .collect::<Vec<_>>();
    let ids = core::store_app_targets(&surfaces);
    let mut lines = Vec::with_capacity(listing.apps.len());
    for (app, id) in listing.apps.iter().zip(&ids) {
        let pid = app.pid.map_or_else(|| "?".to_string(), |p| p.to_string());
        lines.push(format!("[{id}] {} (pid {pid})", app.name));
    }
    if lines.is_empty() {
        lines.push("no apps found".to_string());
    }
    let names = listing
        .apps
        .iter()
        .map(|app| app.name.as_str())
        .collect::<Vec<_>>();
    lines.push(frontmost_line(listing.frontmost.as_deref(), &names));
    Ok(lines.join("\n"))
}

/// The line that names what is in front — the one thing a listing must never leave the caller to
/// infer, including when that application is not among those listed.
fn frontmost_line(frontmost: Option<&str>, listed: &[&str]) -> String {
    match frontmost {
        None => "frontmost application: unknown — the application in front could not be named"
            .to_string(),
        Some(name) => {
            if listed.iter().any(|app| app.eq_ignore_ascii_case(name)) {
                format!("frontmost application: {name}")
            } else {
                format!("frontmost application: {name} (not in the list above)")
            }
        }
    }
}

async fn windows(app: Option<&str>) -> anyhow::Result<String> {
    let app_info = match app {
        Some(arg) => {
            let info = resolve_app(arg).await?;
            // An id outlives the run that minted it, so the application it names is confirmed
            // against the running system before the listing presents it as one without windows.
            if backend().app_running(&info).await == Some(false) {
                return Err(core::app_absent(&info.name, core::AppAbsence::Quit));
            }
            Some(info)
        }
        None => None,
    };
    let listing = backend().list_windows(app_info.as_ref()).await?;
    let surfaces = listing
        .windows
        .iter()
        .map(TargetSpec::from_window)
        .collect::<Vec<_>>();
    let ids = core::store_window_targets(&surfaces);
    Ok(render_windows(&listing, &ids, app_info.as_ref()))
}

/// The listing as the model reads it: one line per window, then what the platform could
/// establish about the set, then one line per set of windows that is not listed. A failed read is
/// never presented as an absence.
fn render_windows(
    listing: &core::WindowListing,
    ids: &[String],
    requested: Option<&AppInfo>,
) -> String {
    let mut lines = listing
        .windows
        .iter()
        .zip(ids)
        .map(|(win, id)| {
            // The place in the order and the focused mark sit beside the identity: two windows
            // that share a title and a position still stand apart by the id.
            let mut line = format!("[{id}] ");
            if let Some(rank) = win.rank {
                let _ = write!(line, "#{rank} ");
            }
            let _ = write!(
                line,
                "{} — \"{}\" {}x{} @ ({}, {})",
                win.app_name,
                win.title,
                win.surface.width,
                win.surface.height,
                win.surface.x,
                win.surface.y
            );
            if win.focused {
                line.push_str(" [focused]");
            }
            line
        })
        .collect::<Vec<_>>();
    if lines.is_empty() {
        lines.push(
            if listing.unread.is_empty() {
                "no windows found"
            } else {
                "no windows could be read"
            }
            .to_string(),
        );
    }
    // An order that is not complete says which kind of incompleteness it is.
    if !listing.windows.is_empty() {
        match listing.order {
            core::WindowOrder::All => {}
            core::WindowOrder::Unknown => lines.push(
                "front-to-back order: unknown — the platform could not attribute one to the \
                 windows listed"
                    .to_string(),
            ),
            core::WindowOrder::Unread => lines.push(
                "front-to-back order: unknown — the list it is attributed from could not be read \
                 this time"
                    .to_string(),
            ),
            core::WindowOrder::Partial => {
                let total = listing.windows.len();
                let unplaced = listing
                    .windows
                    .iter()
                    .filter(|window| window.rank.is_none())
                    .count();
                let ranked = total - unplaced;
                lines.push(if ranked == 0 {
                    "front-to-back order: unknown — no listed window has an on-screen entry to be \
                     ranked by"
                        .to_string()
                } else {
                    format!(
                        "front-to-back order: given for {ranked} of the {total} listed windows (#1 \
                         is in front); {unplaced} {} no on-screen entry",
                        if unplaced == 1 { "has" } else { "have" }
                    )
                });
            }
        }
    }
    lines.push(if let Some(app) = requested {
        match listing.frontmost.as_deref() {
            None => format!(
                "whether {} is frontmost: unknown — the application in front could not be named",
                app.name
            ),
            Some(front) if front.eq_ignore_ascii_case(&app.name) => {
                format!("{} is frontmost", app.name)
            }
            Some(front) => format!("{} is not frontmost — {front} is", app.name),
        }
    } else {
        let names = listing
            .windows
            .iter()
            .map(|win| win.app_name.as_str())
            .collect::<Vec<_>>();
        frontmost_line(listing.frontmost.as_deref(), &names)
    });
    lines.extend(listing.unread.iter().map(|unread| {
        format!(
            "windows not listed for {}: {}",
            unread.app_name, unread.reason
        )
    }));
    lines.join("\n")
}

/// Resolve a `windows` app argument: a stored a-id (or w-id, whose app is what is being asked
/// about), a numeric pid, or a raw app name. Always a fully-resolved [`AppInfo`] or a clear
/// error, so no backend silently skips a pid-less app.
async fn resolve_app(arg: &str) -> anyhow::Result<AppInfo> {
    let trimmed = arg.trim();
    if trimmed == "screen" {
        anyhow::bail!("'screen' is a surface, not an app — use the 'apps' action to list apps");
    }
    if let Some((pid, name)) = core::get_target(trimmed)
        .and_then(|target| target.app().map(|(pid, name)| (pid, name.to_string())))
    {
        return Ok(AppInfo { pid, name });
    }
    // Something shaped like one of this tool's ids is read as an id even when the session no
    // longer knows it: calling it a missing application would blame the wrong thing.
    if core::is_identifier(trimmed) {
        return Err(core::unknown_identifier(trimmed));
    }
    let apps = backend().list_apps().await?.apps;
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

/// The refusal for an app argument that matches no running application: the first few candidate
/// names and a count of the rest, so a truncated list never reads as the whole set.
fn no_app_error(name: &str, apps: &[AppInfo]) -> anyhow::Error {
    if apps.is_empty() {
        anyhow::anyhow!(
            "no running applications found — start '{name}' or list apps with the 'apps' action"
        )
    } else {
        const SHOWN: usize = 3;
        let candidates = apps
            .iter()
            .take(SHOWN)
            .map(|a| a.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let more = apps.len().saturating_sub(SHOWN);
        let rest = if more == 0 {
            String::new()
        } else {
            format!(" and {more} more")
        };
        anyhow::anyhow!(
            "no running application matches '{name}' — running apps: {candidates}{rest}; use the \
             'apps' action for exact ids"
        )
    }
}

/// A ref-based action pins the observed surface; an explicit target that resolves to a DIFFERENT
/// surface is almost always a mistake. An equal target is fine, and so is an application id whose
/// application is the ref's — the ref itself names the window the caller saw.
fn ensure_ref_target_match(
    reference: &str,
    target: Option<&str>,
    cache_target: &TargetSpec,
) -> anyhow::Result<()> {
    let Some(t) = target else {
        return Ok(());
    };
    let given = core::resolve_target(Some(t))?;
    let same_app =
        matches!(given, TargetSpec::App { .. }) && core::same_application(&given, cache_target);
    if given != *cache_target && !same_app {
        anyhow::bail!(
            "ref {reference} belongs to the surface observed for {} — drop the target argument \
             or re-observe the target you want",
            cache_target.describe()
        );
    }
    Ok(())
}

/// Read a surface — the whole of it, part of what is already known of it, or what changed on it.
///
/// A **full** observation is the one thing that records a surface: it pins the surface a later
/// target-less call addresses, mints the refs the caller may act on, and replaces the remembered
/// look every later statement is measured against. A failed observation replaces nothing. A
/// **narrow** look (`ref`, `depth`, `kind`) reads the remembered look instead of the platform and
/// replaces nothing; on a surface never fully observed there is nothing to narrow. A **change**
/// answer (`changed`) reads the surface afresh and compares it against the remembered look, also
/// replacing nothing — with no remembered look it is an ordinary full observation.
async fn observe(
    target: Option<&str>,
    reference: Option<&str>,
    depth: Option<usize>,
    kind: Option<String>,
    changed: bool,
) -> anyhow::Result<Delivered> {
    // The surface a ref names is the one it was observed on; every other form resolves
    // concretely, never `Focused`, so what is recorded is the window this call actually read.
    let (surface, at) = match reference {
        Some(reference) => {
            let (locator, cache_target) = core::resolve_ref(reference)?;
            ensure_ref_target_match(reference, target, &cache_target)?;
            (cache_target, Some(locator))
        }
        None => (resolve_surface(target).await?, None),
    };
    let narrow = core::Narrow { at, depth, kind };

    if changed {
        return Ok(Delivered::Read(observe_changed(&surface).await?));
    }
    if narrow.is_narrow() {
        let refs_valid = core::refs_belong_to(&surface);
        // The record is shared, so the view is rendered with no lock held.
        let Some(look) = core::with_look(&surface) else {
            return Ok(Delivered::Read(spill(&format!(
                "nothing to narrow — {}; run observe on it, or drop ref, depth and kind to read \
                 it now",
                no_look()
            ))));
        };
        let answer = core::render_narrow(&look, &narrow, refs_valid)?;
        return Ok(Delivered::Read(spill(&answer)));
    }

    let obs = backend().observe(&surface).await?;
    let counts = core::counts(&obs.root);
    // Rendered before the observation is remembered: the record owns the tree, so nothing is
    // copied.
    let full = render_full(&obs, counts);
    record_observation(&surface, obs, counts);
    Ok(Delivered::Read(spill(&full)))
}

/// The change answer, or an ordinary full observation when there is nothing to compare against.
async fn observe_changed(surface: &TargetSpec) -> anyhow::Result<String> {
    let obs = backend().observe(surface).await?;
    let counts = core::counts(&obs.root);
    if let Some(look) = core::with_look(surface) {
        return Ok(spill(&core::render_changes(&look, &obs, counts)));
    }
    let mut output = render_full(&obs, counts);
    let _ = writeln!(
        output,
        "nothing to compare against — {}; it is now recorded and its refs are the ones to act on",
        no_look()
    );
    record_observation(surface, obs, counts);
    Ok(spill(&output))
}

/// Record a full observation: the surface a later target-less call addresses, the refs the caller
/// may act on, and the remembered look later statements are measured against. One place writes
/// all three, so no path can record one without the others.
fn record_observation(surface: &TargetSpec, obs: core::Observation, counts: core::Counts) {
    core::set_last_surface(surface.clone());
    core::store_observation(surface.clone(), &obs);
    core::remember_look(surface, obs, counts);
}

/// The full observation as the model reads it, with the thin-surface hint when the tree barely
/// describes what is on screen. The hint fires here and nowhere else.
fn render_full(obs: &core::Observation, counts: core::Counts) -> String {
    let mut output = core::render_tree(obs, counts);
    if counts.interactive < core::AX_THIN_INTERACTIVE_ELEMENTS {
        let _ = writeln!(
            output,
            "surface exposes few actionable elements — {} interactive of {} nodes read: {}",
            counts.interactive,
            counts.nodes,
            // A hint may only name a route that works at this moment: a still when stills are
            // known to work, a statement that they are unavailable when that is known, and neither
            // when availability is unknown.
            match cached_capture_available() {
                Some(true) => "use screenshot to see the rest",
                Some(false) => {
                    "stills are unavailable here, so act on the references it lists, or use \
                     apps/windows for another surface"
                }
                None => {
                    "act on the references it lists, or use apps/windows for another surface"
                }
            }
        );
    }
    output
}

/// Hand the answer to the spill budget, scrubbed first: the tree is the one result that can be
/// large enough to be written to a file instead of returned whole.
fn spill(output: &str) -> String {
    crate::tools::shell::try_spill_to_file(
        crate::util::scrub_credentials(output),
        TOOL_OUTPUT_BUDGET_BYTES,
    )
}

async fn click(
    target: Option<&str>,
    reference: Option<&str>,
    x: Option<f64>,
    y: Option<f64>,
    button: Option<MouseButton>,
    double: Option<bool>,
    modifiers: Option<&[core::Modifier]>,
) -> anyhow::Result<Delivered> {
    if let Some(reference) = reference {
        let (locator, cache_target) = core::resolve_ref(reference)?;
        ensure_ref_target_match(reference, target, &cache_target)?;
        let impact = backend()
            .act_on_element(&cache_target, &locator, ElementAct::Press)
            .await?;
        return Ok(Delivered::OnSurface {
            text: core::impact_text(
                format!(
                    "clicked element {reference}{}",
                    on_surface(&cache_target, None)
                ),
                impact,
            ),
            surface: cache_target,
        });
    }
    let Some((x, y)) = coord_pair(x, y, "click point")? else {
        anyhow::bail!("click requires an element ref (preferred) or normalized x/y coordinates");
    };
    // A concrete surface: its own points are what the point is mapped through, what the window
    // under the point is compared against, and what the answer names.
    let target = resolve_surface(target).await?;
    let input = core::RawInput::Click {
        point: (x, y),
        button: button.unwrap_or(MouseButton::Left),
        double: double.unwrap_or(false),
        modifiers: modifiers.unwrap_or_default().to_vec(),
    };
    // What the point lies over is checked by the backend when it sends, after its own raise: only
    // there is "in front" the state the click meets. The call reports the position it put.
    let actuation = backend().raw_input(&target, input.clone()).await?;
    let mut text = match &actuation.placed {
        Some(placed) => format!(
            "clicked at {}{}",
            placed.describe(),
            on_surface(&target, Some(placed))
        ),
        None => format!("clicked{}{UNREAD_POSITION}", on_surface(&target, None)),
    };
    let _ = write!(text, " — input {}", point_text((x, y)));
    Ok(Delivered::OnSurface {
        text: core::impact_text(text, actuation.impact),
        surface: target,
    })
}

/// What a point action's answer says when the call ended without an answer: nothing is claimed
/// about where it went.
const UNREAD_POSITION: &str = " — its position could not be read";

async fn type_text(
    text: &str,
    target: Option<&str>,
    reference: Option<&str>,
) -> anyhow::Result<Delivered> {
    if let Some(reference) = reference {
        let (locator, cache_target) = core::resolve_ref(reference)?;
        ensure_ref_target_match(reference, target, &cache_target)?;
        let impact = backend()
            .act_on_element(
                &cache_target,
                &locator,
                ElementAct::SetValue {
                    text: text.to_string(),
                },
            )
            .await?;
        return Ok(Delivered::OnSurface {
            text: core::impact_text(
                format!(
                    "set value on element {reference} in {}",
                    cache_target.describe()
                ),
                impact,
            ),
            surface: cache_target,
        });
    }
    // Typing without a ref goes to whatever holds the focus. The surface is resolved once and the
    // input delivered through it: an `a` id becomes the window that receives the text.
    let surface = resolve_in_place(target).await?;
    let input = core::RawInput::TypeText {
        text: text.to_string(),
    };
    let actuation = backend().raw_input(&surface, input.clone()).await?;
    Ok(Delivered::OnSurface {
        text: core::impact_text(
            format!(
                "typed text ({} chars){}",
                text.chars().count(),
                if core::pinned_input(&surface, &input) {
                    on_surface(&surface, None)
                } else {
                    " — the text goes to the focused element".to_string()
                }
            ),
            actuation.impact,
        ),
        surface,
    })
}

async fn scroll(
    direction: ScrollDirection,
    amount: Option<u32>,
    target: Option<&str>,
    x: Option<f64>,
    y: Option<f64>,
) -> anyhow::Result<Delivered> {
    let amount = amount.unwrap_or(DEFAULT_SCROLL_AMOUNT);
    let Some(point) = coord_pair(x, y, "scroll anchor")? else {
        // A point-less wheel is global input: no surface is resolved or raised, and the validator
        // has already refused a target that names one.
        let actuation = backend()
            .raw_input(
                &TargetSpec::Focused,
                core::RawInput::Scroll {
                    point: None,
                    direction,
                    amount,
                },
            )
            .await?;
        return Ok(Delivered::Global(core::impact_text(
            format!(
                "scrolled {amount} ticks {direction:?} — a point-less wheel goes to whatever is \
                 under the pointer"
            ),
            actuation.impact,
        )));
    };
    let surface = resolve_in_place(target).await?;
    let input = core::RawInput::Scroll {
        point: Some(point),
        direction,
        amount,
    };
    let actuation = backend().raw_input(&surface, input.clone()).await?;
    let mut text = match &actuation.placed {
        Some(placed) => format!(
            "scrolled {amount} ticks {direction:?} at {}{}",
            placed.describe(),
            on_surface(&surface, Some(placed))
        ),
        None => format!(
            "scrolled {amount} ticks {direction:?}{}{UNREAD_POSITION}",
            on_surface(&surface, None)
        ),
    };
    let _ = write!(text, " — input {}", point_text(point));
    text.push_str(unpinned_clause(&surface, &input));
    Ok(Delivered::OnSurface {
        text: core::impact_text(text, actuation.impact),
        surface,
    })
}

async fn cursor_surface() -> anyhow::Result<(TargetSpec, core::SurfaceGeometry)> {
    let target = resolve_surface(None).await?;
    let geo = backend().surface_geometry(&target).await?;
    Ok((target, geo))
}

async fn cursor() -> anyhow::Result<String> {
    let (ax, ay) = backend().cursor_position().await?;
    // A surface that cannot be read degrades to the absolute pointer, as it always has, and the
    // reason is passed on rather than guessed. The one exception is a grant denial, which is a
    // typed, actionable failure — "no surface" would misname it.
    let (target, geo) = match cursor_surface().await {
        Ok(resolved) => resolved,
        Err(e) if !core::is_permission_denied(&e) => {
            return Ok(format!(
                "cursor at absolute ({ax:.1}, {ay:.1}) — no surface to normalize against: {e}"
            ));
        }
        Err(e) => return Err(e),
    };
    if core::contains_point(&geo, ax, ay) {
        let (nx, ny) = core::surface_to_normalized(ax, ay, &geo)?;
        Ok(format!(
            "cursor at normalized ({nx:.1}, {ny:.1}){} — absolute ({ax:.1}, {ay:.1})",
            on_surface(&target, None)
        ))
    } else {
        Ok(format!(
            "cursor outside {surface} ({ax:.1}, {ay:.1}) — surface bounds x:[{:.1}, {:.1}] \
             y:[{:.1}, {:.1}]",
            geo.x,
            geo.x + geo.width,
            geo.y,
            geo.y + geo.height,
            surface = target.describe()
        ))
    }
}

/// Capture the target surface (optionally cropping a normalized region) to a PNG under the
/// managed temp dir, record it for this call's attachment, and return a summary naming the
/// surface actually captured. The capture probe is cached: a success always refreshes it; a
/// failure on a cold cache probes once and caches the grant state, so a grantless setup maps to
/// the actionable [`Backend::capture_unavailable_error`] while a stale cached negative never
/// locks out a working channel.
async fn capture(target: Option<&str>, region: Option<&Region>) -> anyhow::Result<String> {
    // One resolution for the whole call: the surface captured, the geometry the region is mapped
    // against and the surface pinned afterwards are the same.
    let target = resolve_surface(target).await?;
    // One allowance for everything this call holds at once — the frame, the composite and the
    // crop — so no reported geometry can drive an allocation the process cannot survive.
    let budget = core::CaptureBudget::default();

    let capture_result = backend().capture(&target, &budget).await;
    let cap = match capture_result {
        Ok(cap) => {
            // A successful capture proves the channel works — always refresh the cache so a fresh
            // probe isn't re-run and later failures map to the real grant state.
            set_capture_cache(true);
            cap
        }
        Err(e) => {
            // First failure in this process: probe once and cache the grant state. A cached
            // negative means the grant is genuinely missing; a cached positive means this failure
            // is transient, so the original error is surfaced.
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

    // The region is mapped into the capture's own pixels, so it cannot be taken against a
    // different surface than the one this call captured.
    let (cropped, rect_opt) = match region {
        Some(region) => {
            let rect = core::region_to_pixels(region, cap.width, cap.height)?;
            (
                core::crop_rgba(&cap, rect, &budget, "the zoom crop")?,
                Some(rect),
            )
        }
        None => (cap, None),
    };

    let (cw, ch) = (cropped.width, cropped.height);
    let path = encode_png(cropped)?;
    // A capture that produced pixels is an observation of that surface: later calls with no
    // target address it.
    core::set_last_surface(target.clone());
    // Stash the capture under the owning agent id.
    CAPTURES
        .lock()
        .unwrap_poison()
        .insert(core::agent_key(), path.clone());
    let kind = if rect_opt.is_some() {
        "zoom"
    } else {
        "screenshot"
    };
    let rect_note = rect_opt.map_or_else(String::new, |(x, y, w, h)| {
        format!(" (cropped rect {x},{y},{w},{h})")
    });
    let summary = format!(
        "{kind} of {}: {cw}x{ch} — {}{rect_note}",
        target.describe(),
        path.display()
    );
    Ok(summary)
}

/// Encode a capture to PNG on disk (offloaded with [`crate::util::with_block_in_place`]),
/// returning the random-nonce path under the managed agent temp `computer/` dir. The encode holds
/// no second pixel copy and its output is bounded by the pixels the budget already capped.
fn encode_png(cap: Capture) -> anyhow::Result<PathBuf> {
    // The claim rides with the pixels: the call keeps paying for them until the encode is done.
    let (width, height, rgba, _claim) = cap.into_parts();
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
        crate::tools::shell::write_owned_spill(&path, &buf)
            .with_context(|| format!("failed to write screenshot {}", path.display()))?;
        Ok::<PathBuf, anyhow::Error>(path)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unread_explorer() -> core::UnreadWindows {
        core::UnreadWindows {
            app_name: "Explorer".to_string(),
            reason: "one read timed out".to_string(),
        }
    }

    fn window_info(title: &str, focused: bool, rank: Option<usize>) -> core::WindowInfo {
        core::WindowInfo {
            app_pid: Some(10),
            app_name: "Chrome".to_string(),
            title: title.to_string(),
            key: core::WindowKey::new(format!("k{title}")),
            surface: core::SurfaceGeometry {
                x: 0.0,
                y: 0.0,
                width: 800.0,
                height: 600.0,
            },
            focused,
            rank,
        }
    }

    fn app(name: &str) -> AppInfo {
        AppInfo {
            pid: Some(10),
            name: name.to_string(),
        }
    }

    /// A minimal payload that deserializes each action variant, used by the schema lockstep.
    fn action_payload(name: &str) -> Value {
        match name {
            "observe" | "screenshot" | "windows" | "apps" | "cursor" => json!({}),
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

    /// One listing and what the rendered answer must say: the whole text, or the fragments a
    /// longer one must carry.
    struct ListingCase {
        name: &'static str,
        /// `(title, focused, rank)` of each window, in the order listed.
        windows: &'static [(&'static str, bool, Option<usize>)],
        unread: bool,
        frontmost: Option<&'static str>,
        order: core::WindowOrder,
        refs: &'static [&'static str],
        app: Option<&'static str>,
        exact: Option<&'static str>,
        present: &'static [&'static str],
    }

    #[expect(clippy::too_many_lines)]
    #[test]
    fn window_listing_cases() {
        let cases = [
            // The unread note sits on its own line, so it can neither run into the last window
            // line nor claim that an application has nothing listed when it does.
            ListingCase {
                name: "an_unread_note_has_its_own_line",
                windows: &[("Start", false, Some(2))],
                unread: true,
                frontmost: Some("Chrome"),
                order: core::WindowOrder::All,
                refs: &["w1"],
                app: None,
                exact: Some(
                    "[w1] #2 Chrome — \"Start\" 800x600 @ (0, 0)\n\
                     frontmost application: Chrome\n\
                     windows not listed for Explorer: one read timed out",
                ),
                present: &[],
            },
            // A listing where nothing could be read is a failure, not "no windows".
            ListingCase {
                name: "a_listing_where_nothing_could_be_read",
                windows: &[],
                unread: true,
                frontmost: None,
                order: core::WindowOrder::Unknown,
                refs: &[],
                app: None,
                exact: Some(
                    "no windows could be read\n\
                     frontmost application: unknown — the application in front could not be named\n\
                     windows not listed for Explorer: one read timed out",
                ),
                present: &[],
            },
            // The order is stated only where it is attributable to the windows listed.
            ListingCase {
                name: "an_unordered_listing_says_so_and_names_the_application_in_front",
                windows: &[("Inbox", false, None)],
                unread: false,
                frontmost: Some("Notes"),
                order: core::WindowOrder::Unknown,
                refs: &["w1"],
                app: Some("Chrome"),
                exact: Some(
                    "[w1] Chrome — \"Inbox\" 800x600 @ (0, 0)\n\
                     front-to-back order: unknown — the platform could not attribute one to the windows \
                     listed\n\
                     Chrome is not frontmost — Notes is",
                ),
                present: &[],
            },
            // The frontmost application's own listing marks its focused window.
            ListingCase {
                name: "the_frontmost_applications_own_listing_marks_its_focused_window",
                windows: &[("Inbox", true, Some(1))],
                unread: false,
                frontmost: Some("Chrome"),
                order: core::WindowOrder::All,
                refs: &["w1"],
                app: Some("Chrome"),
                exact: Some(
                    "[w1] #1 Chrome — \"Inbox\" 800x600 @ (0, 0) [focused]\nChrome is frontmost",
                ),
                present: &[],
            },
            // A partial order counts what it covers.
            ListingCase {
                name: "a_partial_order_counts_what_it_covers",
                windows: &[
                    ("Inbox", false, Some(1)),
                    ("Sent", false, None),
                    ("Drafts", false, None),
                ],
                unread: false,
                frontmost: Some("Chrome"),
                order: core::WindowOrder::Partial,
                refs: &["w1", "w2", "w3"],
                app: None,
                exact: None,
                present: &[
                    "front-to-back order: given for 1 of the 3 listed windows (#1 is in front); 2 \
                     have no on-screen entry",
                ],
            },
            ListingCase {
                name: "a_partial_order_with_no_ranked_window_says_so",
                windows: &[
                    ("Inbox", false, None),
                    ("Sent", false, None),
                    ("Drafts", false, None),
                ],
                unread: false,
                frontmost: Some("Chrome"),
                order: core::WindowOrder::Partial,
                refs: &["w1", "w2", "w3"],
                app: None,
                exact: None,
                present: &[
                    "front-to-back order: unknown — no listed window has an on-screen entry to be \
                     ranked by",
                ],
            },
            // An application in front that the listing could not show is named all the same.
            ListingCase {
                name: "a_frontmost_application_the_listing_could_not_show",
                windows: &[("Inbox", false, Some(1))],
                unread: false,
                frontmost: Some("Dock"),
                order: core::WindowOrder::All,
                refs: &["w1"],
                app: None,
                exact: None,
                present: &["frontmost application: Dock (not in the list above)"],
            },
            // A list that could not be read is a failed read, not a fact about the windows.
            ListingCase {
                name: "an_unread_list_is_not_a_fact_about_the_windows",
                windows: &[("Inbox", false, None)],
                unread: false,
                frontmost: Some("Chrome"),
                order: core::WindowOrder::Unread,
                refs: &["w1"],
                app: None,
                exact: None,
                present: &[
                    "front-to-back order: unknown — the list it is attributed from could not be read \
                     this time",
                ],
            },
        ];
        for case in cases {
            let listing = core::WindowListing {
                windows: case
                    .windows
                    .iter()
                    .map(|(title, focused, rank)| window_info(title, *focused, *rank))
                    .collect(),
                unread: if case.unread {
                    vec![unread_explorer()]
                } else {
                    Vec::new()
                },
                frontmost: case.frontmost.map(str::to_string),
                order: case.order,
            };
            let refs: Vec<String> = case.refs.iter().map(ToString::to_string).collect();
            let app = case.app.map(app);
            let answer = render_windows(&listing, &refs, app.as_ref());
            if let Some(exact) = case.exact {
                assert_eq!(answer, exact, "{}: the rendered listing", case.name);
            }
            for expected in case.present {
                assert!(
                    answer.contains(expected),
                    "{}: the listing does not say {expected:?}: {answer}",
                    case.name
                );
            }
        }
    }

    /// One canonical tagged action, normalized and parsed as `execute` does.
    #[expect(clippy::needless_pass_by_value)] // the payload is moved into the tagged form
    fn parsed_action(name: &str, payload: Value) -> ComputerAction {
        let args = json!({"action": {name: payload}});
        let (normalized, _) = normalize_action(args["action"].clone(), &args)
            .unwrap_or_else(|e| panic!("'{name}' should normalize: {e}"));
        serde_json::from_value(normalized).unwrap_or_else(|e| panic!("'{name}' should parse: {e}"))
    }

    fn refusal(action: &ComputerAction) -> String {
        validate_action(action)
            .expect_err("the action should be refused")
            .to_string()
    }

    /// Every numeric argument has a range and every string a cap; the refusal names it.
    #[test]
    fn an_argument_outside_its_range_is_refused_with_the_range() {
        let err = refusal(&parsed_action(
            "scroll",
            json!({"direction": "down", "amount": core::MAX_SCROLL_AMOUNT + 1}),
        ));
        assert!(
            err.contains(&format!("1-{}", core::MAX_SCROLL_AMOUNT)),
            "{err}"
        );
        assert!(
            validate_action(&parsed_action(
                "scroll",
                json!({"direction": "down", "amount": 0})
            ))
            .is_err()
        );
        assert!(validate_action(&parsed_action("scroll", json!({"direction": "down"}))).is_ok());

        let err = refusal(&parsed_action(
            "type",
            json!({"text": "Z".repeat(core::MAX_ARG_CHARS + 1)}),
        ));
        assert!(
            err.contains(&format!("max {}", core::MAX_ARG_CHARS)),
            "{err}"
        );
        assert!(
            !err.contains('Z'),
            "the refused text is counted, never echoed: {err}"
        );
        assert!(validate_action(&parsed_action("type", json!({"text": "ok"}))).is_ok());

        for seconds in [json!(0), json!(-1), json!(MAX_WAIT_SECS + 1.0)] {
            assert!(
                validate_action(&parsed_action("wait", json!({ "seconds": seconds }))).is_err()
            );
        }
        assert!(validate_action(&parsed_action("wait", json!({"seconds": 1}))).is_ok());

        assert!(
            validate_action(&parsed_action(
                "drag",
                json!({"from": [0, core::NORMALIZED_MAX + 1.0], "to": [1, 1]})
            ))
            .is_err()
        );
        // A corner outside 0-1000, and a region no area: both refused before the capture.
        assert!(
            validate_action(&parsed_action(
                "zoom",
                json!({"region": [0, 0, core::NORMALIZED_MAX + 1.0, 10]})
            ))
            .is_err()
        );
        assert!(
            validate_action(&parsed_action(
                "zoom",
                json!({"region": [10, 10, 1000, 10]})
            ))
            .is_err()
        );
        // Reversed corners are the same region, so they stay accepted.
        assert!(
            validate_action(&parsed_action(
                "zoom",
                json!({"region": [1000, 1000, 0, 0]})
            ))
            .is_ok()
        );

        // More modifiers than there are modifiers: each entry costs a key event.
        assert!(
            validate_action(&parsed_action(
                "click",
                json!({"x": 1, "y": 1, "modifiers": ["cmd", "cmd", "ctrl", "alt", "shift"]})
            ))
            .is_err()
        );
        assert!(
            validate_action(&parsed_action(
                "click",
                json!({"x": 1, "y": 1, "modifiers": ["cmd", "shift"]})
            ))
            .is_ok()
        );
    }

    /// An argument the action cannot honour is an error, not something dropped.
    #[test]
    fn an_argument_the_action_cannot_honour_is_refused() {
        let err = {
            let args = json!({"action": {"press": {"keys": "cmd+t", "target": "screen"}}});
            let (normalized, _) = normalize_action(args["action"].clone(), &args).unwrap();
            check_action_fields(&normalized).unwrap_err().to_string()
        };
        assert!(
            err.contains("does not take 'target'") && err.contains("keys"),
            "{err}"
        );

        let err = refusal(&parsed_action(
            "click",
            json!({"ref": "e3", "x": 10, "y": 10}),
        ));
        assert!(err.contains("never both"), "{err}");

        // A gesture only the coordinate form performs, and the same defaults restated on an
        // element click.
        assert!(
            validate_action(&parsed_action(
                "click",
                json!({"ref": "e3", "double": true})
            ))
            .is_err()
        );
        assert!(
            validate_action(&parsed_action(
                "click",
                json!({"ref": "e3", "button": "left", "double": false, "modifiers": []})
            ))
            .is_ok()
        );

        // Half a coordinate pair names no point, and a target with no point addresses no surface.
        assert!(validate_action(&parsed_action("click", json!({"x": 5}))).is_err());
        assert!(
            validate_action(&parsed_action(
                "scroll",
                json!({"direction": "down", "x": 5})
            ))
            .is_err()
        );
        assert!(
            validate_action(&parsed_action(
                "scroll",
                json!({"direction": "down", "target": "a1"})
            ))
            .is_err()
        );

        // An observation either narrows what is already known or asks what changed.
        assert!(
            validate_action(&parsed_action("observe", json!({"depth": 0}))).is_err(),
            "a depth of 0 shows nothing"
        );
        assert!(
            validate_action(&parsed_action(
                "observe",
                json!({"depth": core::MAX_TREE_DEPTH + 1})
            ))
            .is_err()
        );
        let err = refusal(&parsed_action(
            "observe",
            json!({"changed": true, "kind": "button"}),
        ));
        assert!(
            err.contains("changed on its own"),
            "the two questions are refused together: {err}"
        );
        assert!(validate_action(&parsed_action("observe", json!({"changed": true}))).is_ok());
        assert!(
            validate_action(&parsed_action(
                "observe",
                json!({"ref": "e3", "depth": 2, "kind": "button"})
            ))
            .is_ok()
        );

        // A field outside the action, whichever form the call used.
        let flattened = json!({"action": "click", "x": 1, "y": 1, "text": "hi"});
        let (normalized, note) = normalize_action(flattened["action"].clone(), &flattened).unwrap();
        assert!(
            note.is_some(),
            "the flattened form is still tolerated with its note"
        );
        assert!(check_action_fields(&normalized).is_err());

        let tagged = json!({"action": {"click": {"x": 1, "y": 1}}, "foo": 2});
        assert!(normalize_action(tagged["action"].clone(), &tagged).is_err());
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

        // Every field a variant carries is declared in its schema entry, and every declared field
        // is one the variant takes.
        for action_schema in action_schemas {
            let Some((name, inner)) = action_schema["properties"]
                .as_object()
                .and_then(|props| props.iter().next())
            else {
                panic!("every schema entry names its action");
            };
            let declared = inner["properties"]
                .as_object()
                .expect("every action entry declares its arguments")
                .keys()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>();
            let payload = action_payload(name);
            let wrapped = json!({name.as_str(): payload});
            let action: ComputerAction =
                serde_json::from_value(wrapped).expect("the schema payload deserializes");
            let carried = match serde_json::to_value(&action).expect("a parsed action serializes") {
                Value::Object(mut map) => map
                    .remove(name.as_str())
                    .and_then(|inner| {
                        inner
                            .as_object()
                            .map(|fields| fields.keys().cloned().collect())
                    })
                    .unwrap_or_default(),
                other => panic!("an action serializes to one tag, not {other}"),
            };
            assert_eq!(
                declared, carried,
                "'{name}': the schema declares {declared:?} but the variant carries {carried:?}"
            );
        }

        // Every wire name must normalize and its canonical tagged form must round-trip
        // deserialize. A new variant fails here until a oneOf entry and payload are added.
        for name in &enum_names {
            let payload = action_payload(name);

            // Wrapped form, as the schema produces it: `{"action": {name: payload}}`.
            let wrapped_args = json!({"action": {*name: payload.clone()}});
            let (wrapped_norm, _) = normalize_action(wrapped_args["action"].clone(), &wrapped_args)
                .unwrap_or_else(|_| panic!("wrapped '{name}' should normalize"));

            // Bare-string form, with the payload spread as siblings.
            let mut bare_args = json!({"action": *name});
            if let Value::Object(payload_obj) = &payload {
                bare_args
                    .as_object_mut()
                    .expect("bare args is an object")
                    .extend(payload_obj.clone());
            }
            let (bare_norm, _) = normalize_action(bare_args["action"].clone(), &bare_args)
                .unwrap_or_else(|_| panic!("bare '{name}' should normalize"));

            // Both forms must yield a tagged payload serde accepts for the variant — this would
            // fail for a variant whose normalized payload mismatches its serde shape.
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
