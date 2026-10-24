//! Computer (GUI observe/act) tool for the admin's Assistant.
//!
//! The platform-agnostic core (types, coordinate math, ref lifecycle, backend
//! trait) lives in [`core`]; [`linux`], [`macos`], [`windows`] and [`stub`]
//! implement the backend per platform. All GUI actions are serialized through a
//! process-wide lock so two concurrent agent runs can never race on the live GUI.
//!
//! The tool is offered on Linux, macOS and Windows. Linux has no grant model of
//! its own (AT-SPI and the X11/portal channels are what decides); on macOS the
//! three privacy grants belong to the terminal the daemon was started from, are
//! asked for when first needed and reported on every action they block — see
//! [`macos`]. Windows needs no grant at all, but refuses to actuate on windows of
//! elevated processes and cannot reach the secure desktop — see [`windows`].
//!
//! Mahbot's own windows are targets like any other — listed, readable and
//! capturable — but the system answers accessibility requests addressed to them
//! inside this process, so each backend handles them deliberately: on macOS their
//! exchanges run on the main thread under a bounded wait (so they stay
//! actionable), on Windows element actions are refused because there is no way
//! onto the interface thread, and Linux never sees our window at all (no AT-SPI
//! surface).
//!
//! The agent-facing description is the shared `src/prompt/tool/computer.md` plus
//! only the running platform's own file ([`PLATFORM_DESCRIPTION`]): one
//! installation is one platform, so no agent is handed another's caveats.
//!
//! # Bounded by the product, not by the machine
//!
//! This process aborts on a failed allocation, so no figure the machine reports may
//! decide how much is allocated: one [`core::CaptureBudget`] per capture call bounds
//! every pixel buffer the call holds at once, and every string a read keeps is capped
//! at [`core::MAX_VALUE_CHARS`] (values) or [`core::MAX_NAME_CHARS`] (names and
//! labels) where it is read. A refusal is `degraded` and names the ceiling, the
//! geometry and what the call already holds; a cut is stated in the observation
//! header, or marked by the ellipsis where the string is a label. The accounting and
//! its reasoning live in [`core`].
//!
//! # Every argument is bounded
//!
//! Every [`ComputerAction`] argument is either a value from a declared set or bounded by
//! a range or a length cap. Those bounds are stated in the schema and named by the
//! refusal that enforces them, and they are checked before [`ACTION_LOCK`] is taken — so
//! a bad call costs a round trip, never a wait behind the lock, a capture, a permission
//! prompt or a lookup. An argument the chosen action cannot honour is refused with the
//! form the action does accept rather than dropped: a field it does not take fails as
//! [`check_action_fields`] says, and a combination it cannot honour — half a coordinate
//! pair, a ref together with coordinates, a target on a pointer-less wheel — fails in
//! [`validate_action`].
//!
//! # No call waits forever
//!
//! Every accessibility exchange is bounded by a deadline the product chose, and each
//! backend bounds a whole call on top of that: Linux and macOS with the aggregate budget
//! on [`core`]'s deadline constants, checked between exchanges; Windows with a per-shape
//! job deadline and the walk's own. A bound that runs out always answers — naming the
//! surface and saying whether anything was applied, since an abandoned actuation may have
//! landed — so [`ACTION_LOCK`] is released by an answer and the next call proceeds; a
//! responsive target never reaches one. Each backend's header states its mechanism, and
//! the numbers the model is told are in the tool's prompt.
//!
//! # A session that cannot receive the call is refused, not served
//!
//! A locked screen, a session another user switched away from and a machine with no
//! interactive desktop cannot receive input and show nothing to read, so the calls that
//! read or drive a surface ask [`Backend::desktop_unavailable`] first and fail with
//! `desktop-unavailable` — naming the state, and saying that nothing was delivered or
//! read. Listing applications and windows, and `wait`, need no desktop and keep working.
//! The tag is not `degraded` on purpose: its guidance is to retry, and a retry into such
//! a state cannot succeed, so the model would loop instead of telling the owner.
//!
//! # Manual check (run once on a machine of each platform)
//!
//! Computer use is offered only to the admin's Assistant, so what can be checked
//! automatically is the accounting, not the pixels:
//!
//! 1. `screenshot { target: "screen" }` on an ordinary display: the image is the
//!    same as before, the summary names its size and the PNG is attached.
//! 2. `zoom` over a quarter of that screen: the same crop as before, no refusal.
//! 3. `observe` a document or a browser page: the tree reads as before. A field
//!    holding more than 2000 characters shows its first 120 and the header states
//!    both cuts on macOS and Windows (`1 value cut at 2000 chars`, `1 value cut at
//!    120 chars`); Linux reads a value only one character past the dump, so it
//!    states the display cut alone.
//! 4. A refusal needs a call whose live buffers together exceed 128 Mi pixels — a
//!    whole-desktop `zoom` on a wall of four or more 6K displays, or a framework
//!    reporting an absurd geometry — so it is not reachable from an ordinary desktop:
//!    the accounting is verified by the unit tests on [`core::CaptureBudget`] and
//!    [`core::Capture::alloc`], and the other platforms by the cross-compile lanes in
//!    `scripts/`.
//! 5. A target that stops answering: `kill -STOP` a throwaway application, `observe`
//!    it, and the call ends inside its budget with `degraded` naming the window
//!    instead of hanging; `kill -CONT` and the next call works again. Linux only: a
//!    restarted at-spi (`kill` its bus launcher) is picked up by the next call, and a
//!    target advertising an absurd child count is read without its children and says
//!    so in the header.
//! 6. An argument outside its stated bound is refused before anything happens: a
//!    `scroll` with `amount: 1000` fails naming the tick range without moving the wheel,
//!    a `type` over the string cap fails naming the cap without typing a character, and
//!    `click { ref, x, y }` fails naming which one to drop without clicking. The bounds
//!    themselves are covered by the unit tests on [`validate_action`] and
//!    [`check_action_fields`], so this step confirms the live path only.
//! 7. Lock the screen and call `screenshot` / `observe` / `press`: each fails with
//!    `desktop-unavailable` naming the lock, while `apps` still lists. Unlock and the
//!    same calls work again. A machine that cannot be locked (or a platform with no
//!    signal for it) keeps working — that is the intended behaviour, not a gap.

mod core;
mod linux;
mod macos;
mod windows;
// Every other platform: the capability is absent, so the stub is never
// advertised.
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

/// Schema text for the `target` field, shared by every action that takes one so
/// the schema and the tool description cannot drift apart.
const TARGET_FIELD: &str = "Target id (a1/w1) or \"screen\". Default: the surface this agent \
     last observed or captured, initially the focused window of the frontmost app. A w-id names \
     that one window or fails as surface-gone; an a-id means the application's own focused \
     window.";

/// Wheel ticks a `scroll` moves when `amount` is omitted.
const DEFAULT_SCROLL_AMOUNT: u32 = 3;

/// Longest `wait` the tool will sit out, in seconds.
const MAX_WAIT_SECS: f64 = 10.0;

/// The modifiers a click names, one entry each in the schema — and the most entries a
/// click may carry, since every entry costs a key event under the action lock.
const MODIFIERS: [&str; 4] = ["cmd", "ctrl", "alt", "shift"];

/// One schema entry per [`ComputerAction`], built once: the model-facing schema and
/// [`check_action_fields`] read the same entries, so the arguments a call may carry
/// cannot drift from the interface the model is given.
static ACTION_ENTRIES: LazyLock<Vec<Value>> = LazyLock::new(|| {
    vec![
        super::action_entry_schema(
            "observe",
            "Get the accessibility tree of a target surface with element refs (e1, e2, ...). Refs expire on re-observe. The result names the surface it is about.",
            &[],
            &json!({
                "target": {"type": "string", "description": TARGET_FIELD}
            }),
        ),
        super::action_entry_schema(
            "screenshot",
            "Capture the target surface as a PNG and inject it as a native image; requires a vision-capable model. The result names the surface captured.",
            &[],
            &json!({
                "target": {"type": "string", "description": TARGET_FIELD}
            }),
        ),
        super::action_entry_schema(
            "zoom",
            "Capture and crop a normalized region of the target surface as a PNG.",
            &["region"],
            &json!({
                "target": {"type": "string", "description": TARGET_FIELD},
                "region": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}, "description": "Normalized [x0, y0, x1, y1] region (0-1000), relative to the target surface, enclosing a non-zero area."}
            }),
        ),
        super::action_entry_schema(
            "apps",
            "List running GUI applications as a1-style targets.",
            &[],
            &json!({}),
        ),
        super::action_entry_schema(
            "windows",
            "List windows, optionally filtered to an app, as w1-style targets. A w-id names that one window until it is gone.",
            &[],
            &json!({
                "app": {"type": "string", "description": "App name, pid, or a stored a-id to filter windows."}
            }),
        ),
        super::action_entry_schema(
            "click",
            "Click by element ref (preferred) or normalized coordinates — never both. A coordinate click is pointer input: it lands on whatever is under that point.",
            &[],
            &json!({
                "target": {"type": "string", "description": TARGET_FIELD},
                "ref": {"type": "string", "description": "Element ref from the most recent observe; preferred over coordinates."},
                "x": {"type": "number", "description": "Normalized 0-1000 X, relative to the target surface."},
                "y": {"type": "number", "description": "Normalized 0-1000 Y, relative to the target surface."},
                "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "Mouse button; coordinate clicks only. Default: left."},
                "double": {"type": "boolean", "description": "Double-click; coordinate clicks only. Default: false."},
                "modifiers": {"type": "array", "maxItems": MODIFIERS.len(), "items": {"enum": MODIFIERS}, "description": "Keyboard modifiers; apply only to coordinate clicks, one entry per modifier."}
            }),
        ),
        super::action_entry_schema(
            "type",
            "Set an element's value (with ref) or type into the focused element (without). Without a ref the named window is raised first and the call fails instead of typing elsewhere.",
            &["text"],
            &json!({
                "text": {"type": "string", "description": "Text to set/type."},
                "target": {"type": "string", "description": TARGET_FIELD},
                "ref": {"type": "string", "description": "Element ref from the most recent observe; sets its value."}
            }),
        ),
        super::action_entry_schema(
            "press",
            "Press a keyboard chord, e.g. \"cmd+shift+t\", \"return\", \"ctrl+c\". The chord goes to the focused application.",
            &["keys"],
            &json!({
                "keys": {"type": "string", "description": "Chord to press, e.g. \"cmd+shift+t\", \"return\", \"ctrl+c\"."}
            }),
        ),
        super::action_entry_schema(
            "scroll",
            "Scroll the target surface. Without x/y the wheel goes to whatever is under the pointer.",
            &["direction"],
            &json!({
                "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                "amount": {"type": "integer", "minimum": 1, "maximum": core::MAX_SCROLL_AMOUNT, "description": format!("Scroll ticks, 1-{}. Default: {DEFAULT_SCROLL_AMOUNT}.", core::MAX_SCROLL_AMOUNT)},
                "target": {"type": "string", "description": "Target id (a1/w1) or \"screen\", used with x/y; without x/y the wheel goes to whatever is under the pointer and a target is an error."},
                "x": {"type": "number", "description": "Normalized 0-1000 X anchor; provide both x and y together, or omit both."},
                "y": {"type": "number", "description": "Normalized 0-1000 Y anchor; provide both x and y together, or omit both."}
            }),
        ),
        super::action_entry_schema(
            "drag",
            "Drag from one normalized point to another on the target surface.",
            &["from", "to"],
            &json!({
                "from": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}, "description": "[x, y] normalized 0-1000."},
                "to": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}, "description": "[x, y] normalized 0-1000."},
                "target": {"type": "string", "description": TARGET_FIELD}
            }),
        ),
        super::action_entry_schema(
            "cursor",
            "Report the pointer position as normalized and absolute points.",
            &[],
            &json!({}),
        ),
        super::action_entry_schema(
            "wait",
            "Pause before the next action.",
            &["seconds"],
            &json!({
                "seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": MAX_WAIT_SECS, "description": format!("Seconds to wait, in (0, {MAX_WAIT_SECS}].")}
            }),
        ),
    ]
});

/// Refuse a field the action does not take, before serde would drop it: the schema
/// declares `additionalProperties: false` for every action, so an unlisted argument is
/// a mistake to report, not a value to ignore. `action_value` is the normalized tagged
/// form (`{"click": {...}}`); shape errors are left to the deserializer.
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

/// Check every argument an action carries before the action lock is taken and before
/// anything is captured, prompted for or looked up. Each string is within
/// [`core::MAX_ARG_CHARS`], each number within its stated range, and a combination the
/// action cannot honour is refused with the form it does accept.
fn validate_action(action: &ComputerAction) -> anyhow::Result<()> {
    match action {
        ComputerAction::Observe { target } | ComputerAction::Screenshot { target } => {
            check_opt_arg("target", target.as_deref())?;
        }
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
        } => {
            check_opt_arg("target", target.as_deref())?;
            check_opt_arg("ref", reference.as_deref())?;
            match (reference, x, y) {
                // An element press is one form: a button, a second click or a modifier
                // would name a gesture the ref-based path does not perform.
                (Some(reference), None, None) => {
                    if modifiers.as_ref().is_some_and(|m| !m.is_empty())
                        || button.is_some_and(|b| b != MouseButton::Left)
                        || double.is_some_and(|d| d)
                    {
                        anyhow::bail!(
                            "click on element {reference} takes no button, double or modifiers \
                             beyond the plain left click — those belong to coordinate clicks; \
                             drop them, or drop ref and give x and y to click a point"
                        );
                    }
                }
                (Some(_), _, _) => anyhow::bail!(
                    "click takes a ref or x/y, never both — drop x and y to click the element, \
                     or drop ref to click the point"
                ),
                (None, Some(x), Some(y)) => {
                    core::check_point("click point", *x, *y)?;
                    check_modifiers(modifiers.as_deref())?;
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
        }
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

/// A coordinate pair argument: both members, or neither. Half a point cannot be
/// mapped, so it names no position the action could honour.
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

/// Length-check an optional string argument.
fn check_opt_arg(field: &str, value: Option<&str>) -> anyhow::Result<()> {
    value.map_or(Ok(()), |value| core::check_arg_len(field, value))
}

/// Refuse a modifier list longer than the modifiers there are: a click holds each one
/// or it does not, and every entry drives a key event under the action lock.
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

/// Corrective error for an unrecoverable action shape, echoing the exact
/// expected form so the model can self-correct in one round-trip.
fn corrective_action_error(received: &Value) -> String {
    format!(
        "Invalid computer action arguments. Expected action to be {EXPECTED_ACTION_SHAPE}. \
         Received: {received}"
    )
}

/// Tolerant action normalization, mirroring the chrome tool: a canonical tagged
/// object `{"click":{...}}`, a plain action name with flattened sibling fields
/// `{"action":"click","x":0,"y":0}`, or a stringified JSON object.
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

// ── Backend factory ─────────────────────────────────────────────────────

/// The backend for the running platform.
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

// ── Capability probe ────────────────────────────────────────────────────

/// Cheap, synchronous availability check for advertisement — the platform
/// contract behind it is [`Backend::is_supported`].
pub(crate) fn is_advertised() -> bool {
    backend().is_supported()
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

/// The last screenshot/zoom of each agent, keyed by [`core::agent_key`], so a
/// concurrent agent's capture can never be re-attached by another agent's call. An
/// entry is replaced per agent on each capture and left stale (the last one is what
/// the attachment takes).
static CAPTURES: LazyLock<std::sync::Mutex<HashMap<String, CaptureRecord>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

/// One capture as it was produced: the file on disk and the pixel size it was
/// taken at. The attachment is compressed and possibly reduced on its way to the
/// model, and it needs the capture's own size to say so.
#[derive(Clone)]
struct CaptureRecord {
    path: PathBuf,
    width: u32,
    height: u32,
}

/// Drop an agent's per-agent registry entries (observation/target/capture) at
/// agent-run end, so state never leaks across runs. Called from the agent's
/// run-end cleanup guard (`RunEndCleanup`) with the agent's id string (matching
/// [`core::agent_key`]'s task-local value).
pub(crate) fn cleanup_agent_state(agent_key: &str) {
    CAPTURES.lock().unwrap_poison().remove(agent_key);
    core::clear_agent_state(agent_key);
    #[cfg(target_os = "macos")]
    macos::forget_agent_windows(agent_key);
}

// ── Description ────────────────────────────────────────────────────────

/// The asset holding this platform's own half of the description, appended to
/// the shared `tool/computer.md`. `None` where the tool is never advertised (the
/// stub backend), which leaves the shared text alone.
const PLATFORM_DESCRIPTION: Option<&str> = if cfg!(target_os = "linux") {
    Some("tool/computer_linux.md")
} else if cfg!(target_os = "macos") {
    Some("tool/computer_macos.md")
} else if cfg!(target_os = "windows") {
    Some("tool/computer_windows.md")
} else {
    None
};

/// Tool for observing and acting on the local GUI via the OS accessibility
/// channel. Every action is serialized through [`ACTION_LOCK`].
#[derive(Default)]
pub(crate) struct ComputerTool;

#[async_trait]
impl Tool for ComputerTool {
    fn name(&self) -> &'static str {
        "computer"
    }

    /// The shared contract plus this platform's own half — an installation is one
    /// platform, so the other platforms' caveats never reach an agent.
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
                    "description": format!(
                        "One action per call. Every string argument is capped at {} characters \
                         and every coordinate is normalized 0-{}; an argument outside a stated \
                         range, or one the action does not take, fails the call.",
                        core::MAX_ARG_CHARS, core::NORMALIZED_MAX
                    ),
                    "oneOf": Value::Array(ACTION_ENTRIES.clone()),
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

        // Serialize every GUI-touching action so concurrent agent runs can't race.
        let _lock = ACTION_LOCK.lock().await;

        let output = dispatch(&action).await?;
        Ok(super::with_normalization_notes(output, &notes))
    }

    /// The tool's text is a real result (a surface summary, an action's outcome),
    /// never a claim about an image, so annotations are appended rather than
    /// replacing it. A capture whose image cannot be attached says so here instead
    /// of leaving the call looking like a successful observation.
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
        })
    }
}

/// The image a `screenshot`/`zoom` call produced, or `None` for every other action.
///
/// The action is re-parsed so a stale capture from a prior round is never attached
/// to a non-capture call, and the record is looked up per agent so two concurrent
/// agents never share a screenshot. Failing to produce the attachment is an error
/// the caller must report, never a silent `None`.
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
    let Some(record) = CAPTURES
        .lock()
        .unwrap_poison()
        .get(&core::agent_key())
        .cloned()
    else {
        // Unreachable: a capture records its file before it can return success.
        return Ok(None);
    };
    let meta = crate::util::local_image_to_compressed_data_uri_with_meta(&record.path).await?;
    let mut payload = crate::tools::ImagePayload::from_compressed_meta(
        &record.path,
        meta,
        None,
        crate::tools::ImagePayloadSource::Computer,
    );
    // A capture larger than the shared attachment cap reaches the model reduced, so
    // the model is told: an image it measured at its capture size would otherwise
    // silently describe pixels it was never shown.
    let longest = record.width.max(record.height);
    if longest > crate::util::INBOUND_IMAGE_MAX_SIDE {
        payload.note = Some(format!(
            "reduced from the captured {}x{} to {}x{} — an attached image is capped at {} px on \
             its long edge",
            record.width,
            record.height,
            payload.width,
            payload.height,
            crate::util::INBOUND_IMAGE_MAX_SIDE
        ));
    }
    Ok(Some(payload))
}

// ── Action dispatch ─────────────────────────────────────────────────────

/// Whether an action needs the interactive desktop. Listing applications and windows and
/// `wait` describe the session rather than receive anything from it, so they stay available
/// while it is locked; everything else is written as an exclusion so an action added later
/// is gated by default.
fn needs_desktop(action: &ComputerAction) -> bool {
    !matches!(
        action,
        ComputerAction::Apps {} | ComputerAction::Windows { .. } | ComputerAction::Wait { .. }
    )
}

/// Refuse a call whose session cannot receive it, before any surface is resolved or
/// captured and before [`ACTION_LOCK`] is taken. Deliberately outside the per-action
/// code: a state refusal is the one answer `cursor` must not degrade into "no surface".
async fn ensure_interactive_desktop() -> anyhow::Result<()> {
    let Some(reason) = backend().desktop_unavailable().await else {
        return Ok(());
    };
    Err(core::taxonomy_error(
        core::ERR_DESKTOP_UNAVAILABLE,
        format!(
            "{reason} — this call needs an interactive desktop, so nothing was delivered or read; \
             tell the owner and run it again only once the session is back"
        ),
    ))
}

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
            // A chord is global input: it goes to the focused application, so no
            // surface is resolved and none is raised on its behalf.
            let impact = backend()
                .raw_input(
                    &TargetSpec::Focused,
                    core::RawInput::KeyChord {
                        chord: keys.clone(),
                    },
                )
                .await?;
            core::impact_text(
                format!("pressed chord: {keys} — a chord goes to the focused application"),
                impact,
            )
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
            let gesture = core::RawInput::Drag {
                from: *from,
                to: *to,
            };
            let impact = backend().raw_input(&target, gesture.clone()).await?;
            core::impact_text(
                format!(
                    "dragged from {from:?} to {to:?}{}{}",
                    on_surface(&target),
                    unpinned_clause(&target, &gesture)
                ),
                impact,
            )
        }
        ComputerAction::Cursor {} => cursor().await?,
        ComputerAction::Wait { seconds } => {
            tokio::time::sleep(Duration::from_secs_f64(*seconds)).await;
            format!("waited {seconds}s")
        }
    };
    Ok(crate::util::scrub_credentials(&output))
}

/// The surface clause of an action result: `""` when the call named no surface
/// (the focused window), else `" on <surface>"`. Every action states the surface it
/// addressed this one way, so a result never varies by action.
fn on_surface(target: &TargetSpec) -> String {
    match target {
        TargetSpec::Focused => String::new(),
        other => format!(" on {}", other.describe()),
    }
}

/// The clause a pointer gesture appends when it is not pinned to the surface: its
/// coordinates were mapped there, but the input lands wherever the pointer is.
/// Pinned input omits it, having been delivered to the surface it named.
fn unpinned_clause(target: &TargetSpec, input: &core::RawInput) -> &'static str {
    if core::pinned_input(target, input) {
        ""
    } else {
        " — the input goes to whatever is under its coordinates"
    }
}

/// The surface an action addresses: the explicit target, else the surface this
/// agent last observed or captured, else — with nothing pinned yet — whatever the
/// system has focused. Coordinate actions use it so a focus shift between observe
/// and the action never lands them on a different surface than the refs came from.
fn resolve_action_target(target: Option<&str>) -> anyhow::Result<TargetSpec> {
    if let Some(t) = target {
        return core::resolve_target(Some(t));
    }
    Ok(core::last_surface().unwrap_or(TargetSpec::Focused))
}

/// [`resolve_action_target`] with the last resort made concrete: a call that ends
/// up on the focused window names that window, so its result — and the surface it
/// pins — is about the one surface it actually used.
async fn resolve_surface(target: Option<&str>) -> anyhow::Result<TargetSpec> {
    match resolve_action_target(target)? {
        TargetSpec::Focused => {
            let win = backend().focused_window().await?;
            Ok(TargetSpec::from_window(&win))
        }
        other => Ok(other),
    }
}

async fn apps() -> anyhow::Result<String> {
    let apps = backend().list_apps().await?;
    // An a-id targets the app's own focused window, re-resolved at act time.
    let surfaces = apps
        .iter()
        .map(|app| TargetSpec::App {
            app_pid: app.pid,
            app_name: app.name.clone(),
        })
        .collect::<Vec<_>>();
    let ids = core::store_app_targets(&surfaces);
    let mut lines = Vec::with_capacity(apps.len());
    for (app, id) in apps.iter().zip(&ids) {
        let pid = app.pid.map_or_else(|| "?".to_string(), |p| p.to_string());
        lines.push(format!("[{id}] {} (pid {pid})", app.name));
    }
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
    let listing = backend().list_windows(app_info.as_ref()).await?;
    let surfaces = listing
        .windows
        .iter()
        .map(TargetSpec::from_window)
        .collect::<Vec<_>>();
    let ids = core::store_window_targets(&surfaces);
    Ok(render_windows(&listing, &ids))
}

/// The listing as the model reads it: one line per window, then one line per set of
/// windows that is not listed. A failed read is never presented as an absence, and
/// the note never claims anything about the windows listed above it — an application
/// can have both.
fn render_windows(listing: &core::WindowListing, ids: &[String]) -> String {
    let mut lines = listing
        .windows
        .iter()
        .zip(ids)
        .map(|(win, id)| {
            format!(
                "[{id}] {} — \"{}\" {}x{} @ ({}, {})",
                win.app_name,
                win.title,
                win.surface.width,
                win.surface.height,
                win.surface.x,
                win.surface.y
            )
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
    lines.extend(listing.unread.iter().map(|unread| {
        format!(
            "windows not listed for {}: {}",
            unread.app_name, unread.reason
        )
    }));
    lines.join("\n")
}

/// Resolve a `windows` app argument: a stored a-id (or w-id, whose app is what is
/// being asked about), a numeric pid, or a raw app name. Always returns a
/// fully-resolved [`AppInfo`] (with a pid) or a clear error, so the macOS backend
/// never silently skips a pid-less app.
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
            cache_target.describe()
        );
    }
    Ok(())
}

async fn observe(target: Option<&str>) -> anyhow::Result<String> {
    // A concrete surface, never `Focused`: the observation pins THIS window, so a
    // later action re-resolves it (and fails honestly if it is gone) instead of
    // whatever happens to be focused then.
    let target = resolve_surface(target).await?;
    let obs = backend().observe(&target).await?;
    // Only a successful observation moves the pin; a failed one leaves the
    // surface this agent last observed in place.
    core::set_last_surface(target.clone());
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
        let (locator, cache_target) = core::resolve_ref(reference)?;
        ensure_ref_target_match(reference, target, &cache_target)?;
        let impact = backend()
            .act_on_element(&cache_target, &locator, ElementAct::Press)
            .await?;
        return Ok(core::impact_text(
            format!("clicked element {reference}{}", on_surface(&cache_target)),
            impact,
        ));
    }
    let Some((x, y)) = coord_pair(x, y, "click point")? else {
        anyhow::bail!("click requires an element ref (preferred) or normalized x/y coordinates");
    };
    let target = resolve_action_target(target)?;
    let input = core::RawInput::Click {
        point: (x, y),
        button: button.unwrap_or(MouseButton::Left),
        double: double.unwrap_or(false),
        modifiers: modifiers.unwrap_or_default().to_vec(),
    };
    let impact = backend().raw_input(&target, input.clone()).await?;
    Ok(core::impact_text(
        format!(
            "clicked at ({x}, {y}){}{}",
            on_surface(&target),
            unpinned_clause(&target, &input)
        ),
        impact,
    ))
}

async fn type_text(
    text: &str,
    target: Option<&str>,
    reference: Option<&str>,
) -> anyhow::Result<String> {
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
        return Ok(core::impact_text(
            format!(
                "set value on element {reference} in {}",
                cache_target.describe()
            ),
            impact,
        ));
    }
    let target = resolve_action_target(target)?;
    let input = core::RawInput::TypeText {
        text: text.to_string(),
    };
    let impact = backend().raw_input(&target, input.clone()).await?;
    // Typing is pinned only on a named window; with `screen` or nothing observed
    // yet it goes to the focused element, so the result must not claim a surface
    // it never reached.
    Ok(core::impact_text(
        format!(
            "typed text ({} chars){}",
            text.chars().count(),
            if core::pinned_input(&target, &input) {
                on_surface(&target)
            } else {
                " — the text goes to the focused element".to_string()
            }
        ),
        impact,
    ))
}

async fn scroll(
    direction: ScrollDirection,
    amount: Option<u32>,
    target: Option<&str>,
    x: Option<f64>,
    y: Option<f64>,
) -> anyhow::Result<String> {
    let amount = amount.unwrap_or(DEFAULT_SCROLL_AMOUNT);
    let Some(point) = coord_pair(x, y, "scroll anchor")? else {
        // A point-less wheel is global input: no surface is resolved or raised, and the
        // validator has already refused a target that names one.
        let impact = backend()
            .raw_input(
                &TargetSpec::Focused,
                core::RawInput::Scroll {
                    point: None,
                    direction,
                    amount,
                },
            )
            .await?;
        return Ok(core::impact_text(
            format!(
                "scrolled {amount} ticks {direction:?} — a point-less wheel goes to whatever is \
                 under the pointer"
            ),
            impact,
        ));
    };
    let target = resolve_action_target(target)?;
    let input = core::RawInput::Scroll {
        point: Some(point),
        direction,
        amount,
    };
    let impact = backend().raw_input(&target, input.clone()).await?;
    Ok(core::impact_text(
        format!(
            "scrolled {amount} ticks {direction:?} at ({}, {}){}{}",
            point.0,
            point.1,
            on_surface(&target),
            unpinned_clause(&target, &input)
        ),
        impact,
    ))
}

/// The surface a `cursor` reading is relative to, with its geometry: a concrete
/// one, so the numbers say what they are relative to.
async fn cursor_surface() -> anyhow::Result<(TargetSpec, core::SurfaceGeometry)> {
    let target = resolve_surface(None).await?;
    let geo = backend().surface_geometry(&target).await?;
    Ok((target, geo))
}

async fn cursor() -> anyhow::Result<String> {
    let (ax, ay) = backend().cursor_position().await?;
    // A surface that cannot be read degrades to the absolute pointer, as it always
    // has (no focused window, an unreachable AT-SPI bus, a Wayland portal failure, a
    // self-target read the main thread did not answer), and the reason is passed on
    // rather than guessed. The one exception is a grant denial, which is a typed
    // failure: it is actionable (its text names the terminal and the pane), and "no
    // surface" would misname it.
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
            on_surface(&target)
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

/// Capture the target surface (optionally cropping a normalized region) to a PNG
/// under the managed temp dir, record it for this call's attachment, and return a
/// summary naming the surface actually captured. The capture probe is cached: a
/// successful capture always refreshes the cache; a failed capture on a cold cache
/// probes once and caches the grant state, so a genuinely-grantless setup maps
/// failures to the actionable [`Backend::capture_unavailable_error`] while a real
/// success upgrades the cache immediately — a stale cached negative never locks
/// out a working channel.
async fn capture(target: Option<&str>, region: Option<&Region>) -> anyhow::Result<String> {
    // One resolution for the whole call: the surface captured, the geometry the
    // region is mapped against and the surface pinned afterwards are the same.
    let target = resolve_surface(target).await?;
    // One allowance for everything this call holds at once — the frame, the composite
    // and the crop — so no reported geometry can drive an allocation the process
    // cannot survive.
    let budget = core::CaptureBudget::default();

    let capture_result = backend().capture(&target, &budget).await;
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

    // The region is mapped into the capture's own pixels, so it cannot be taken
    // against a different surface than the one this call captured.
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
    // A capture that produced pixels is an observation of that surface: later
    // calls with no target address it.
    core::set_last_surface(target.clone());
    // Stash the capture under the owning agent id, with the size it was taken at so
    // the attachment can say when it had to reduce it.
    CAPTURES.lock().unwrap_poison().insert(
        core::agent_key(),
        CaptureRecord {
            path: path.clone(),
            width: cw,
            height: ch,
        },
    );
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

/// Encode a capture to PNG on disk (CPU + fs work offloaded with
/// [`crate::util::with_block_in_place`]), returning the random-nonce path under
/// the managed agent temp `computer/` dir. The encode holds no second pixel copy
/// ([`image::RgbaImage::from_raw`] takes the buffer over) and its compressed output
/// is bounded by the pixels it encodes, which the call's budget already capped.
fn encode_png(cap: Capture) -> anyhow::Result<PathBuf> {
    // The claim rides with the pixels: the call keeps paying for them until the encode
    // is done with them.
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

    /// Minimal payload per variant, used to build a canonical tagged action
    /// (`{"name": payload}`) for the deserialization half of the lockstep check.
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

    fn unread_explorer() -> core::UnreadWindows {
        core::UnreadWindows {
            app_name: "Explorer".to_string(),
            reason: "one read timed out".to_string(),
        }
    }

    /// The unread note sits on its own line, so it can neither run into the last
    /// window line nor claim that an application has nothing listed when it does.
    #[test]
    fn window_listing_puts_each_unread_note_on_its_own_line() {
        let listing = core::WindowListing {
            windows: vec![core::WindowInfo {
                app_pid: Some(10),
                app_name: "Chrome".to_string(),
                title: "Start".to_string(),
                key: core::WindowKey::new("k"),
                surface: core::SurfaceGeometry {
                    x: 0.0,
                    y: 0.0,
                    width: 800.0,
                    height: 600.0,
                },
            }],
            unread: vec![unread_explorer()],
        };
        assert_eq!(
            render_windows(&listing, &["w1".to_string()]),
            "[w1] Chrome — \"Start\" 800x600 @ (0, 0)\n\
             windows not listed for Explorer: one read timed out"
        );
        // A listing where nothing could be read is a failure, not "no windows".
        let nothing = core::WindowListing {
            windows: Vec::new(),
            unread: vec![unread_explorer()],
        };
        assert_eq!(
            render_windows(&nothing, &[]),
            "no windows could be read\nwindows not listed for Explorer: one read timed out"
        );
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

        // A gesture only the coordinate form performs, and the same defaults restated
        // on an element click (the form the tool has always tolerated).
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

        // Half a coordinate pair names no point, and a target with no point addresses
        // no surface.
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

        // Every field a variant carries is declared in its schema entry, and every
        // declared field is one the variant takes: a field serde would drop, and a
        // field refused without a slot in the variant, both fail here.
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
