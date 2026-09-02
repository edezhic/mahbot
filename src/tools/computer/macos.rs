//! macOS [`Backend`] for the Computer tool: AX accessibility (observe/act),
//! CGEvent raw input, and ScreenCaptureKit capture. Non-`Send` platform objects
//! are created and dropped INSIDE `with_block_in_place` closures — never in
//! struct state or across an await.

#![cfg(target_os = "macos")]
// The many casts and raw-pointer borrows are inherent to the C ABI/geometry
// types the FFI exposes (u32 display IDs, usize pixel dims, CGFloat floats);
// suppressing the pedantic cast/pointer lints here keeps the file focused.
#![allow(
    clippy::borrow_as_ptr,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::ptr_as_ptr,
    clippy::ref_as_ptr,
    clippy::semicolon_if_nothing_returned
)]

use super::core::{
    self, AppInfo, Backend, Capture, ElementAct, Locator, Modifier, MouseButton, Observation,
    RawInput, ScrollDirection, SurfaceGeometry, TargetSpec, UiNode, WindowInfo,
};
use crate::util::{UnwrapPoison, with_block_in_place};
use anyhow::anyhow;
use block2::RcBlock;
use dispatch2::{DispatchRetained, DispatchSemaphore, DispatchTime};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2_app_kit::{NSApplicationActivationPolicy, NSWorkspace};
use objc2_application_services::{
    self, AXError, AXIsProcessTrusted, AXUIElement, AXValue, AXValueType,
};
use objc2_core_foundation::{
    CFArray, CFBoolean, CFRetained, CFString, CFType, CGPoint, CGRect, CGSize, Type,
};
use objc2_core_graphics::{
    CGBitmapContextCreate, CGBitmapContextGetData, CGColorSpace, CGContext, CGDirectDisplayID,
    CGDisplayBounds, CGError, CGEvent, CGEventField, CGEventFlags, CGEventSource,
    CGEventSourceStateID, CGEventTapLocation, CGEventType, CGGetActiveDisplayList, CGImage,
    CGImageAlphaInfo, CGImageByteOrderInfo, CGMouseButton, CGScrollEventUnit,
};
use objc2_foundation::{NSArray, NSError, NSInteger};
use objc2_screen_capture_kit::{
    SCContentFilter, SCScreenshotConfiguration, SCScreenshotManager, SCScreenshotOutput,
    SCShareableContent, SCStreamErrorCode, SCWindow,
};
use std::sync::{Arc, Mutex};

pub(crate) static MACOS_BACKEND: MacOsBackend = MacOsBackend;

pub(crate) struct MacOsBackend;

// ── kAX* attribute/action string constants ───────────────────────────────
//
// The objc2 bindings do not ship the kAX* string constants, so the ones we use
// are declared here. `CFString::from_static_str` reuses the static buffer, so
// the fresh retain per call is cheap; drop happens at the end of the statement.

const AX_ROLE: &str = "AXRole";
const AX_TITLE: &str = "AXTitle";
const AX_VALUE: &str = "AXValue";
const AX_SUBROLE: &str = "AXSubrole";
const AX_POSITION: &str = "AXPosition";
const AX_SIZE: &str = "AXSize";
const AX_ACTIONS: &str = "AXActions";
const AX_CHILDREN: &str = "AXChildren";
const AX_WINDOWS: &str = "AXWindows";
const AX_FOCUSED_WINDOW: &str = "AXFocusedWindow";
const AX_FOCUSED_APPLICATION: &str = "AXFocusedApplication";
const AX_FOCUSED: &str = "AXFocused";
const AX_PRESS: &str = "AXPress";
const AX_OPEN: &str = "AXOpen";
const AX_RAISE: &str = "AXRaise";

#[inline]
fn axc(s: &'static str) -> CFRetained<CFString> {
    CFString::from_static_str(s)
}

// ── AX attribute readers ─────────────────────────────────────────────────

fn attribute_value(el: &AXUIElement, attr: &CFString) -> Result<CFRetained<CFType>, AXError> {
    let mut value: *const CFType = std::ptr::null();
    // SAFETY: `value` points to valid storage the binding fills on success.
    let err = unsafe {
        el.copy_attribute_value(
            attr,
            std::ptr::NonNull::new(&mut value as *mut *const CFType).unwrap(),
        )
    };
    if err != AXError::Success {
        return Err(err);
    }
    // SAFETY: AXUIElementCopyAttributeValue returns a +1 retained pointer (Create
    // rule); null indicates a missing value.
    match std::ptr::NonNull::new(value.cast_mut()) {
        Some(p) => Ok(unsafe { CFRetained::from_raw(p) }),
        None => Err(AXError::NoValue),
    }
}

fn attribute_string(el: &AXUIElement, attr: &CFString) -> Option<String> {
    let val = attribute_value(el, attr).ok()?;
    Some(val.downcast_ref::<CFString>()?.to_string())
}

fn attribute_point(el: &AXUIElement, attr: &CFString) -> Option<CGPoint> {
    let val = attribute_value(el, attr).ok()?;
    let axval = val.downcast_ref::<AXValue>()?;
    let mut p = CGPoint::new(0.0, 0.0);
    // SAFETY: `axval` is an AXValue; the destination matches AXValueType::CGPoint.
    let ok = unsafe {
        axval.value(
            AXValueType::CGPoint,
            std::ptr::NonNull::new(&mut p as *mut _ as *mut _).unwrap(),
        )
    };
    ok.then_some(p)
}

fn attribute_size(el: &AXUIElement, attr: &CFString) -> Option<CGSize> {
    let val = attribute_value(el, attr).ok()?;
    let axval = val.downcast_ref::<AXValue>()?;
    let mut s = CGSize::new(0.0, 0.0);
    // SAFETY: `axval` is an AXValue; the destination matches AXValueType::CGSize.
    let ok = unsafe {
        axval.value(
            AXValueType::CGSize,
            std::ptr::NonNull::new(&mut s as *mut _ as *mut _).unwrap(),
        )
    };
    ok.then_some(s)
}

fn attribute_bool(el: &AXUIElement, attr: &CFString) -> Option<bool> {
    let val = attribute_value(el, attr).ok()?;
    Some(val.downcast_ref::<CFBoolean>()?.value())
}

/// Raw (AX-prefixed) action names; empty on error.
fn raw_action_names(el: &AXUIElement) -> Vec<String> {
    let Ok(val) = attribute_value(el, &axc(AX_ACTIONS)) else {
        return Vec::new();
    };
    let Ok(array) = val.downcast::<CFArray>() else {
        return Vec::new();
    };
    // SAFETY: AXActions is a CFArray of CFString objects.
    let typed: CFRetained<CFArray<CFString>> = unsafe { CFRetained::cast_unchecked(array) };
    typed.iter().map(|s| s.to_string()).collect()
}

fn attribute_children(el: &AXUIElement) -> Vec<CFRetained<AXUIElement>> {
    let Ok(val) = attribute_value(el, &axc(AX_CHILDREN)) else {
        return Vec::new();
    };
    let Ok(array) = val.downcast::<CFArray>() else {
        return Vec::new();
    };
    // SAFETY: AXChildren is a CFArray of AXUIElement objects.
    let typed: CFRetained<CFArray<AXUIElement>> = unsafe { CFRetained::cast_unchecked(array) };
    typed.iter().collect()
}

fn normalize_action_name(name: &str) -> String {
    name.strip_prefix("AX").unwrap_or(name).to_lowercase()
}

/// Map an AXError to a taxonomy error per the contract.
fn ax_error(err: AXError, ctx: impl std::fmt::Display) -> anyhow::Error {
    match err {
        AXError::CannotComplete => core::taxonomy_error(
            core::ERR_DEGRADED,
            format!("transient AX failure, retry: {ctx}"),
        ),
        AXError::InvalidUIElement => core::taxonomy_error(
            core::ERR_STALE_ELEMENT,
            format!("window closed or changed — re-enumerate: {ctx}"),
        ),
        AXError::APIDisabled => core::taxonomy_error(
            core::ERR_PERMISSION_DENIED,
            format!("Accessibility TCC grant missing: {ctx}"),
        ),
        AXError::AttributeUnsupported
        | AXError::ActionUnsupported
        | AXError::NoValue
        | AXError::NotImplemented
        | AXError::ParameterizedAttributeUnsupported => core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            format!("unsupported accessibility operation: {ctx}"),
        ),
        // Plain (non-taxonomy) errors — not actionable by the model.
        _ => anyhow!("AX error {err:?}: {ctx}"),
    }
}

/// Window-level error mapping (CannotComplete/InvalidUIElement are fatal).
fn ax_window_error(err: AXError, ctx: impl std::fmt::Display) -> anyhow::Error {
    match err {
        AXError::CannotComplete => core::taxonomy_error(
            core::ERR_DEGRADED,
            format!("transient AX failure, retry: {ctx}"),
        ),
        AXError::InvalidUIElement => core::taxonomy_error(
            core::ERR_STALE_ELEMENT,
            format!("window closed or changed — re-enumerate: {ctx}"),
        ),
        AXError::APIDisabled => core::taxonomy_error(
            core::ERR_PERMISSION_DENIED,
            format!("Accessibility TCC grant missing: {ctx}"),
        ),
        _ => core::taxonomy_error(
            core::ERR_NOT_MATCHED,
            format!("no windows enumerated — re-enumerate: {ctx}"),
        ),
    }
}

// ── AX tree walk ─────────────────────────────────────────────────────────

const MAX_DEPTH: usize = 25;

/// True when the element is a secure (e.g. password) field. macOS reports the
/// secure marker in either AXRole or AXSubrole — a password field is
/// AXTextField with the marker in AXSubrole — so both are checked. One predicate
/// keeps the value-masking in [`walk_element`] and the SetValue refusal in
/// [`set_element_value`] in agreement.
fn is_secure_field(role: &str, subrole: Option<&str>) -> bool {
    role.to_lowercase().contains("secure")
        || subrole.is_some_and(|s| s.to_lowercase().contains("secure"))
}

fn walk_element(
    element: &AXUIElement,
    origin: (f64, f64),
    depth: usize,
    handles: &mut Vec<CFRetained<AXUIElement>>,
    nodes: &mut usize,
) -> Option<UiNode> {
    if *nodes >= core::MAX_RENDER_NODES {
        return None;
    }
    *nodes += 1;
    handles.push(element.retain());

    let role = attribute_string(element, &axc(AX_ROLE)).unwrap_or_else(|| "unknown".to_string());
    // Never read a value out of a secure element. A password field reports
    // AXRole="AXTextField" with the secure marker in AXSubrole, so check the
    // subrole (or role) for "secure" before pulling the value.
    let subrole = attribute_string(element, &axc(AX_SUBROLE));
    let value = if is_secure_field(&role, subrole.as_deref()) {
        None
    } else {
        attribute_string(element, &axc(AX_VALUE))
    };
    let frame = match (
        attribute_point(element, &axc(AX_POSITION)),
        attribute_size(element, &axc(AX_SIZE)),
    ) {
        (Some(p), Some(s)) => Some((p.x - origin.0, p.y - origin.1, s.width, s.height)),
        _ => None,
    };
    let actions = raw_action_names(element)
        .iter()
        .map(|a| normalize_action_name(a))
        .collect();
    let focused = attribute_bool(element, &axc(AX_FOCUSED)).unwrap_or(false);

    let mut children = Vec::new();
    if depth < MAX_DEPTH {
        for child in attribute_children(element) {
            if let Some(child_node) = walk_element(&child, origin, depth + 1, handles, nodes) {
                children.push(child_node);
            }
        }
    }
    Some(UiNode {
        role,
        name: attribute_string(element, &axc(AX_TITLE)),
        value,
        frame,
        actions,
        focused,
        children,
    })
}

/// Build the accessibility tree, returning the root node plus a pre-order-aligned
/// `Vec` of platform handles (used by `act_on_element` to map a matched node back
/// to its element).
fn build_tree(root: &AXUIElement, origin: (f64, f64)) -> (UiNode, Vec<CFRetained<AXUIElement>>) {
    let mut handles = Vec::new();
    let mut nodes = 0usize;
    let node = walk_element(root, origin, 0, &mut handles, &mut nodes).unwrap_or_else(|| UiNode {
        role: "unknown".to_string(),
        name: None,
        value: None,
        frame: None,
        actions: Vec::new(),
        focused: false,
        children: Vec::new(),
    });
    (node, handles)
}

// ── Window resolution ────────────────────────────────────────────────────

fn focused_app(system: &AXUIElement) -> Option<CFRetained<AXUIElement>> {
    attribute_value(system, &axc(AX_FOCUSED_APPLICATION))
        .ok()?
        .downcast::<AXUIElement>()
        .ok()
}

fn focused_window_of_app(app: &AXUIElement) -> Option<CFRetained<AXUIElement>> {
    attribute_value(app, &axc(AX_FOCUSED_WINDOW))
        .ok()?
        .downcast::<AXUIElement>()
        .ok()
}

fn running_app_pid(name: &str) -> Option<u32> {
    let needle = name.to_lowercase();
    NSWorkspace::sharedWorkspace()
        .runningApplications()
        .iter()
        .find(|app| {
            app.activationPolicy() == NSApplicationActivationPolicy::Regular
                && app
                    .localizedName()
                    .is_some_and(|n| n.to_string().to_lowercase() == needle)
        })
        .map(|app| app.processIdentifier() as u32)
}

/// App display name for a pid: the workspace's authoritative localized name (the
/// same source `list_apps` uses) with an AXTitle fallback when the running-app
/// lookup misses. An AXTitle read on the AXApplication element is usually empty,
/// so relying on it alone yields an app="".
fn app_display_name(pid: u32, app: &AXUIElement) -> String {
    NSWorkspace::sharedWorkspace()
        .runningApplications()
        .iter()
        .find(|a| a.processIdentifier() == pid as i32)
        .and_then(|a| a.localizedName().map(|n| n.to_string()))
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| attribute_string(app, &axc(AX_TITLE)).unwrap_or_default())
}

fn app_element(pid: Option<u32>, name: &str) -> Result<CFRetained<AXUIElement>, anyhow::Error> {
    let pid = match pid {
        Some(p) => p,
        None => running_app_pid(name).ok_or_else(|| {
            anyhow!("could not resolve running application '{name}' to a process id")
        })?,
    };
    // SAFETY: `pid` is a running app (or stale); AXUIElementCreateApplication
    // always returns a non-null element — attribute reads surface InvalidUIElement.
    Ok(unsafe { AXUIElement::new_application(pid as libc::pid_t) })
}

fn read_windows(app: &AXUIElement) -> Result<Vec<CFRetained<AXUIElement>>, anyhow::Error> {
    let val = attribute_value(app, &axc(AX_WINDOWS))
        .map_err(|e| ax_window_error(e, "enumerate windows"))?;
    let array: CFRetained<CFArray> = val
        .downcast::<CFArray>()
        .map_err(|_| anyhow!("AXWindows was not a CFArray"))?;
    // SAFETY: AXWindows is a CFArray of AXUIElement objects.
    let typed: CFRetained<CFArray<AXUIElement>> = unsafe { CFRetained::cast_unchecked(array) };
    Ok(typed.iter().collect())
}

fn resolve_window_in_app(
    app: &AXUIElement,
    title: &str,
    index: usize,
) -> Result<CFRetained<AXUIElement>, anyhow::Error> {
    // App-level target (empty title, index 0) resolves the app's focused window
    // at ACT time via kAXFocusedWindowAttribute first; only if that is
    // unavailable does it fall back to the first window in the AXWindows array.
    // This is deliberately re-resolved per action, so a ref observed against an
    // app-level target follows the currently focused window. A locator mismatch
    // inside the so-resolved window then degrades to not-matched (a safe error),
    // never to a wrong-window action.
    if title.is_empty()
        && index == 0
        && let Some(w) = focused_window_of_app(app)
    {
        return Ok(w);
    }
    let windows = read_windows(app)?;
    if windows.is_empty() {
        return Err(core::taxonomy_error(
            core::ERR_NOT_MATCHED,
            "no windows found for this app — re-enumerate windows",
        ));
    }
    if !title.is_empty()
        && let Some(w) = windows
            .iter()
            .find(|w| attribute_string(w, &axc(AX_TITLE)).as_deref() == Some(title))
    {
        return Ok(w.retain());
    }
    if index < windows.len() {
        return Ok(windows[index].retain());
    }
    Err(core::taxonomy_error(
        core::ERR_NOT_MATCHED,
        format!("window '{title}' (index {index}) not found — re-enumerate windows"),
    ))
}

fn resolve_window(
    target: &TargetSpec,
) -> Result<(CFRetained<AXUIElement>, CFRetained<AXUIElement>), anyhow::Error> {
    match target {
        TargetSpec::Focused => {
            // SAFETY: AXUIElementCreateSystemWide never returns null.
            let system = unsafe { AXUIElement::new_system_wide() };
            let app = focused_app(&system).ok_or_else(|| {
                core::taxonomy_error(
                    core::ERR_NOT_MATCHED,
                    "no focused application — enumerate apps/windows first",
                )
            })?;
            let window = focused_window_of_app(&app).ok_or_else(|| {
                core::taxonomy_error(core::ERR_NOT_MATCHED, "no focused window (desktop frontmost/screensaver) — use 'apps'/'windows' to pick a surface")
            })?;
            Ok((window, app))
        }
        TargetSpec::Window {
            app_pid,
            app_name,
            title,
            index,
        } => {
            let app = app_element(*app_pid, app_name)?;
            Ok((resolve_window_in_app(&app, title, *index)?, app))
        }
        TargetSpec::Screen => Err(core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            "observe targets an accessibility surface — use screenshot for the screen",
        )),
    }
}

fn window_origin(window: &AXUIElement) -> Result<(f64, f64), anyhow::Error> {
    let p = attribute_point(window, &axc(AX_POSITION))
        .ok_or_else(|| anyhow!("window position unavailable"))?;
    Ok((p.x, p.y))
}

fn window_frame(window: &AXUIElement) -> Result<SurfaceGeometry, anyhow::Error> {
    let p = attribute_point(window, &axc(AX_POSITION))
        .ok_or_else(|| anyhow!("window position unavailable"))?;
    let s =
        attribute_size(window, &axc(AX_SIZE)).ok_or_else(|| anyhow!("window size unavailable"))?;
    Ok(SurfaceGeometry {
        x: p.x,
        y: p.y,
        width: s.width,
        height: s.height,
    })
}

fn app_pid(app: &AXUIElement) -> Result<u32, anyhow::Error> {
    let mut pid: libc::pid_t = 0;
    // SAFETY: `pid` points to valid storage the binding fills on success.
    let err = unsafe { app.pid(std::ptr::NonNull::new(&mut pid).unwrap()) };
    if err != AXError::Success {
        return Err(ax_error(err, "read application pid"));
    }
    Ok(pid as u32)
}

// ── Raw input (CGEvent) ──────────────────────────────────────────────────

fn cg_mouse_button(button: MouseButton) -> CGMouseButton {
    match button {
        MouseButton::Left => CGMouseButton::Left,
        MouseButton::Right => CGMouseButton::Right,
        MouseButton::Middle => CGMouseButton::Center,
    }
}

fn cg_type_for(button: MouseButton, down: bool) -> CGEventType {
    match (button, down) {
        (MouseButton::Left, true) => CGEventType::LeftMouseDown,
        (MouseButton::Left, false) => CGEventType::LeftMouseUp,
        (MouseButton::Right, true) => CGEventType::RightMouseDown,
        (MouseButton::Right, false) => CGEventType::RightMouseUp,
        (MouseButton::Middle, true) => CGEventType::OtherMouseDown,
        (MouseButton::Middle, false) => CGEventType::OtherMouseUp,
    }
}

fn cg_flags(modifiers: &[Modifier]) -> CGEventFlags {
    modifiers.iter().fold(CGEventFlags::empty(), |acc, m| {
        acc | match m {
            Modifier::Cmd => CGEventFlags::MaskCommand,
            Modifier::Ctrl => CGEventFlags::MaskControl,
            Modifier::Alt => CGEventFlags::MaskAlternate,
            Modifier::Shift => CGEventFlags::MaskShift,
        }
    })
}

fn post_event(event: &CGEvent) {
    CGEvent::post(CGEventTapLocation::HIDEventTap, Some(event));
}

fn source() -> Option<CFRetained<CGEventSource>> {
    CGEventSource::new(CGEventSourceStateID::CombinedSessionState)
}

fn click(point: CGPoint, button: MouseButton, double: bool, modifiers: &[Modifier]) {
    let src = source();
    let flags = cg_flags(modifiers);
    let flap = |down: bool, state: i64| {
        if let Some(ev) = CGEvent::new_mouse_event(
            src.as_deref(),
            cg_type_for(button, down),
            point,
            cg_mouse_button(button),
        ) {
            CGEvent::set_flags(Some(&ev), flags);
            CGEvent::set_integer_value_field(Some(&ev), CGEventField::MouseEventClickState, state);
            post_event(&ev);
        }
    };
    if double {
        // Both halves carry click-state 2; two down/up flaps make a double click.
        flap(true, 2);
        flap(false, 2);
        flap(true, 2);
        flap(false, 2);
    } else {
        flap(true, 1);
        flap(false, 1);
    }
}

fn sleep_millis(ms: u64) {
    std::thread::sleep(std::time::Duration::from_millis(ms));
}

fn ascii_key(c: char) -> Option<i64> {
    // kVK_ANSI_* for the 26 letters and top-row digits.
    Some(match c {
        '0' => 29,
        '1' => 18,
        '2' => 19,
        '3' => 20,
        '4' => 21,
        '5' => 23,
        '6' => 22,
        '7' => 26,
        '8' => 28,
        '9' => 25,
        'a' => 0,
        's' => 1,
        'd' => 2,
        'f' => 3,
        'h' => 4,
        'g' => 5,
        'z' => 6,
        'x' => 7,
        'c' => 8,
        'v' => 9,
        'b' => 11,
        'q' => 12,
        'w' => 13,
        'e' => 14,
        'r' => 15,
        'y' => 16,
        't' => 17,
        'i' => 34,
        'o' => 31,
        'p' => 35,
        'l' => 37,
        'j' => 38,
        'k' => 40,
        'n' => 45,
        'm' => 46,
        'u' => 32,
        _ => return None,
    })
}

/// Key name → macOS virtual keycode. Unknown names return `None`.
fn key_code(key: &str) -> Option<i64> {
    let k = key.to_lowercase();
    if k.len() == 1
        && let Some(c) = k.chars().next().and_then(ascii_key)
    {
        return Some(c);
    }
    Some(match k.as_str() {
        "return" | "enter" => 36,
        "tab" => 48,
        "space" => 49,
        "delete" => 51, // backspace
        "forwarddelete" => 117,
        "escape" => 53,
        "home" => 115,
        "end" => 119,
        "pageup" => 116,
        "pagedown" => 121,
        "up" => 126,
        "down" => 125,
        "left" => 123,
        "right" => 124,
        "f1" => 122,
        "f2" => 120,
        "f3" => 99,
        "f4" => 118,
        "f5" => 96,
        "f6" => 97,
        "f7" => 98,
        "f8" => 100,
        "f9" => 101,
        "f10" => 109,
        "f11" => 103,
        "f12" => 111,
        "minus" => 27,
        "equal" => 24,
        "comma" => 43,
        "period" => 47,
        "slash" => 44,
        "semicolon" => 41,
        "quote" => 39,
        "backslash" => 42,
        "backtick" => 50,
        "leftbracket" => 33,
        "rightbracket" => 30,
        "capslock" => 57,
        _ => return None,
    })
}

const KNOWN_KEYS: &str = "return, enter, tab, space, delete, forwarddelete, escape, home, end, \
     pageup, pagedown, up, down, left, right, f1..f12, minus, equal, comma, period, slash, \
     semicolon, quote, backslash, backtick, leftbracket, rightbracket, capslock, a-z, 0-9";

fn press_key(key: &str) -> Result<(), anyhow::Error> {
    let Some(code) = key_code(key) else {
        return Err(anyhow!(
            "unknown key '{key}' — supported keys: {KNOWN_KEYS}"
        ));
    };
    let src = source();
    if let Some(down) = CGEvent::new_keyboard_event(src.as_deref(), code as u16, true) {
        post_event(&down);
    }
    if let Some(up) = CGEvent::new_keyboard_event(src.as_deref(), code as u16, false) {
        post_event(&up);
    }
    Ok(())
}

fn type_text(text: &str) {
    let src = source();
    for ch in text.chars() {
        if ch == '\n' {
            let _ = press_key("return");
            continue;
        }
        let mut buf = [0u16; 2];
        let encoded = ch.encode_utf16(&mut buf);
        if let Some(down) = CGEvent::new_keyboard_event(src.as_deref(), 0, true) {
            // SAFETY: `encoded` is a valid UTF-16 buffer.
            unsafe {
                CGEvent::keyboard_set_unicode_string(
                    Some(&down),
                    encoded.len() as u64,
                    encoded.as_ptr(),
                )
            };
            post_event(&down);
        }
        if let Some(up) = CGEvent::new_keyboard_event(src.as_deref(), 0, false) {
            // SAFETY: `encoded` is a valid UTF-16 buffer.
            unsafe {
                CGEvent::keyboard_set_unicode_string(
                    Some(&up),
                    encoded.len() as u64,
                    encoded.as_ptr(),
                )
            };
            post_event(&up);
        }
        sleep_millis(1);
    }
}

fn key_chord(chord: &str) -> Result<(), anyhow::Error> {
    let (modifiers, key) = core::parse_key_chord(chord)?;
    let Some(code) = key_code(&key) else {
        return Err(anyhow!(
            "unknown key '{key}' — supported keys: {KNOWN_KEYS}"
        ));
    };
    let src = source();
    let flags = cg_flags(&modifiers);
    if let Some(down) = CGEvent::new_keyboard_event(src.as_deref(), code as u16, true) {
        CGEvent::set_flags(Some(&down), flags);
        post_event(&down);
    }
    if let Some(up) = CGEvent::new_keyboard_event(src.as_deref(), code as u16, false) {
        // Clear flags on key-up so the modifier doesn't stick.
        CGEvent::set_flags(Some(&up), CGEventFlags::empty());
        post_event(&up);
    }
    Ok(())
}

/// Scroll: wheel 1 = vertical (down positive), wheel 2 = horizontal (right
/// positive). CGEvent deltas use +ve for down/right on macOS.
fn scroll(point: Option<CGPoint>, direction: ScrollDirection, amount: u32) {
    let src = source();
    let (w1, w2) = match direction {
        ScrollDirection::Down => (amount as i32, 0),
        ScrollDirection::Up => (-(amount as i32), 0),
        ScrollDirection::Right => (0, amount as i32),
        ScrollDirection::Left => (0, -(amount as i32)),
    };
    if let Some(ev) =
        CGEvent::new_scroll_wheel_event2(src.as_deref(), CGScrollEventUnit::Line, 2, w1, w2, 0)
    {
        if let Some(p) = point {
            CGEvent::set_location(Some(&ev), p);
        }
        post_event(&ev);
    }
}

fn drag(from: CGPoint, to: CGPoint) {
    let src = source();
    if let Some(down) = CGEvent::new_mouse_event(
        src.as_deref(),
        CGEventType::LeftMouseDown,
        from,
        CGMouseButton::Left,
    ) {
        post_event(&down);
    }
    for i in 1..=12 {
        let t = i as f64 / 12.0;
        if let Some(drag) = CGEvent::new_mouse_event(
            src.as_deref(),
            CGEventType::LeftMouseDragged,
            CGPoint::new(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t),
            CGMouseButton::Left,
        ) {
            post_event(&drag);
        }
        sleep_millis(2);
    }
    if let Some(up) = CGEvent::new_mouse_event(
        src.as_deref(),
        CGEventType::LeftMouseUp,
        to,
        CGMouseButton::Left,
    ) {
        post_event(&up);
    }
}

// ── Surface geometry ─────────────────────────────────────────────────────

fn active_display_ids() -> Vec<CGDirectDisplayID> {
    let mut displays = [0u32; 32];
    let mut count = 0u32;
    // SAFETY: `displays` and `count` are valid storage the binding fills.
    let err =
        unsafe { CGGetActiveDisplayList(displays.len() as u32, displays.as_mut_ptr(), &mut count) };
    if err != CGError::Success {
        return Vec::new();
    }
    displays[..count as usize].to_vec()
}

fn screen_surface() -> Result<SurfaceGeometry, anyhow::Error> {
    let ids = active_display_ids();
    if ids.is_empty() {
        anyhow::bail!("no active displays found");
    }
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for id in ids {
        let b = CGDisplayBounds(id);
        min_x = min_x.min(b.origin.x);
        min_y = min_y.min(b.origin.y);
        max_x = max_x.max(b.origin.x + b.size.width);
        max_y = max_y.max(b.origin.y + b.size.height);
    }
    Ok(SurfaceGeometry {
        x: min_x,
        y: min_y,
        width: max_x - min_x,
        height: max_y - min_y,
    })
}

fn surface_geometry_blocking(target: &TargetSpec) -> Result<SurfaceGeometry, anyhow::Error> {
    match target {
        TargetSpec::Screen => screen_surface(),
        _ => window_frame(&resolve_window(target)?.0),
    }
}

fn activate_window(target: &TargetSpec) -> Result<(), anyhow::Error> {
    let (TargetSpec::Focused | TargetSpec::Window { .. }) = target else {
        return Ok(());
    };
    // Best-effort raise: no focused window (e.g. desktop frontmost) is fine for
    // key chords/typing — geometry is only required by the point-based actions.
    let Ok((window, _app)) = resolve_window(target) else {
        return Ok(());
    };
    if raw_action_names(&window).iter().any(|a| a == AX_RAISE) {
        // SAFETY: AXRaise was verified to exist on the window.
        let err = unsafe { window.perform_action(&axc(AX_RAISE)) };
        if err != AXError::Success {
            return Err(ax_error(err, "raise window"));
        }
        sleep_millis(80);
    }
    Ok(())
}

fn raw_input_blocking(target: &TargetSpec, input: &RawInput) -> Result<(), anyhow::Error> {
    if !matches!(target, TargetSpec::Screen) {
        activate_window(target)?;
    }
    match input {
        RawInput::Click {
            point,
            button,
            double,
            modifiers,
        } => {
            let (x, y) =
                core::normalized_to_surface(point.0, point.1, &surface_geometry_blocking(target)?)?;
            click(CGPoint::new(x, y), *button, *double, modifiers);
        }
        RawInput::TypeText { text } => type_text(text),
        RawInput::KeyChord { chord } => key_chord(chord)?,
        RawInput::Scroll {
            point,
            direction,
            amount,
        } => {
            let p = if let Some((nx, ny)) = point {
                let (x, y) =
                    core::normalized_to_surface(*nx, *ny, &surface_geometry_blocking(target)?)?;
                Some(CGPoint::new(x, y))
            } else {
                None
            };
            scroll(p, *direction, *amount);
        }
        RawInput::Drag { from, to } => {
            let geometry = surface_geometry_blocking(target)?;
            let (fx, fy) = core::normalized_to_surface(from.0, from.1, &geometry)?;
            let (tx, ty) = core::normalized_to_surface(to.0, to.1, &geometry)?;
            drag(CGPoint::new(fx, fy), CGPoint::new(tx, ty));
        }
    }
    Ok(())
}

// ── Capture (ScreenCaptureKit) ───────────────────────────────────────────

/// Hand-off wrappers letting SCK completion handlers deliver non-`Send` retained
/// objects across the handler-to-waiter thread boundary.
struct SendShareable(Retained<SCShareableContent>);
unsafe impl Send for SendShareable {}

struct SendImage(Retained<CGImage>);
unsafe impl Send for SendImage {}

struct ScErr {
    code: NSInteger,
    message: String,
}

impl From<ScErr> for anyhow::Error {
    fn from(err: ScErr) -> anyhow::Error {
        if err.code == SCStreamErrorCode::UserDeclined.0
            || err.code == SCStreamErrorCode::MissingEntitlements.0
        {
            core::taxonomy_error(
                core::ERR_PERMISSION_DENIED,
                format!(
                    "Screen Recording permission is not granted — {}",
                    err.message
                ),
            )
        } else if err.code == SCStreamErrorCode::NoWindowList.0 {
            core::taxonomy_error(
                core::ERR_NOT_MATCHED,
                "no window found — re-enumerate windows",
            )
        } else {
            anyhow!("screen capture failed (code {}): {}", err.code, err.message)
        }
    }
}

fn sc_err_from_nserror(err: &NSError) -> ScErr {
    ScErr {
        code: err.code(),
        message: err.localizedDescription().to_string(),
    }
}

/// Block on an SCK completion handler (5s timeout) and return the handed-off
/// result. The caller spawns the provider call; this consumes the shared slot.
fn sc_wait<T: Send>(
    slot: &Arc<Mutex<Option<Result<T, ScErr>>>>,
    semaphore: &Arc<DispatchRetained<DispatchSemaphore>>,
    what: &str,
) -> Result<T, anyhow::Error> {
    if semaphore.wait(DispatchTime::NOW.time(5_000_000_000)) != 0 {
        return Err(anyhow!("timed out waiting for {what}"));
    }
    slot.lock()
        .unwrap_poison()
        .take()
        .ok_or_else(|| anyhow!("{what} handler never invoked"))?
        .map_err(anyhow::Error::from)
}

/// Fetch the shareable-content snapshot (availability probe + window/display
/// capture source).
fn get_shareable_content() -> Result<Retained<SCShareableContent>, anyhow::Error> {
    let semaphore = Arc::new(DispatchSemaphore::new(0));
    let sem2 = Arc::clone(&semaphore);
    let slot: Arc<Mutex<Option<Result<SendShareable, ScErr>>>> = Arc::new(Mutex::new(None));
    let slot2 = Arc::clone(&slot);
    let block = RcBlock::new(move |content: *mut SCShareableContent, err: *mut NSError| {
        let result = if !err.is_null() {
            // SAFETY: `err` is valid for the handler's duration.
            Err(sc_err_from_nserror(unsafe { &*err }))
        } else if content.is_null() {
            Err(ScErr {
                code: 0,
                message: "no shareable content".to_string(),
            })
        } else {
            // SAFETY: a retained object the handler hands us; `retain` takes
            // ownership beyond the callback.
            match unsafe { Retained::retain(content) } {
                Some(c) => Ok(SendShareable(c)),
                None => Err(ScErr {
                    code: 0,
                    message: "null shareable content".to_string(),
                }),
            }
        };
        *slot2.lock().unwrap_poison() = Some(result);
        sem2.signal();
    });
    // SAFETY: block is valid; SCK does not retain the block across the call.
    unsafe {
        SCShareableContent::getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler(false, true, &block);
    }
    sc_wait::<SendShareable>(&slot, &semaphore, "shareable content").map(|SendShareable(c)| c)
}

fn capture_filter_to_rgba(
    filter: &SCContentFilter,
    px_w: usize,
    px_h: usize,
    ignore_shadows: bool,
) -> Result<Capture, anyhow::Error> {
    let config = unsafe { SCScreenshotConfiguration::new() };
    // SAFETY: the config setters are ObjC messages against a live object.
    // sourceRect is intentionally NOT set: SCK interprets it filter-locally and
    // would crop off-content, so capture the full filter content and let
    // `core::crop_rgba` do the zoom cropping instead.
    unsafe {
        config.setWidth(px_w as NSInteger);
        config.setHeight(px_h as NSInteger);
        config.setShowsCursor(false);
        config.setIgnoreShadows(ignore_shadows);
    }

    let semaphore = Arc::new(DispatchSemaphore::new(0));
    let sem2 = Arc::clone(&semaphore);
    let slot: Arc<Mutex<Option<Result<SendImage, ScErr>>>> = Arc::new(Mutex::new(None));
    let slot2 = Arc::clone(&slot);
    let block = RcBlock::new(move |output: *mut SCScreenshotOutput, err: *mut NSError| {
        let result = if !err.is_null() {
            // SAFETY: `err` is valid for the handler's duration.
            Err(sc_err_from_nserror(unsafe { &*err }))
        } else if output.is_null() {
            Err(ScErr {
                code: 0,
                message: "no screenshot output".to_string(),
            })
        } else {
            // SAFETY: `output` is valid for the handler's duration.
            let out = unsafe { &*output };
            match unsafe { out.sdrImage() } {
                Some(img) => Ok(SendImage(img)),
                None => Err(ScErr {
                    code: 0,
                    message: "no SDR image in output".to_string(),
                }),
            }
        };
        *slot2.lock().unwrap_poison() = Some(result);
        sem2.signal();
    });
    // SAFETY: filter/config/block are valid; SCK does not retain the block.
    unsafe {
        SCScreenshotManager::captureScreenshotWithFilter_configuration_completionHandler(
            filter,
            &config,
            Some(&block),
        );
    }
    let SendImage(image) = sc_wait::<SendImage>(&slot, &semaphore, "screenshot")?;
    capture_image_to_rgba(&image)
}

/// Turn a CGImage into top-down RGBA bytes. CG contexts use a bottom-left
/// origin, so the image is flipped (translate+scale) before drawing to yield
/// top-down rows.
fn capture_image_to_rgba(image: &CGImage) -> Result<Capture, anyhow::Error> {
    let w = CGImage::width(Some(image));
    let h = CGImage::height(Some(image));
    if w == 0 || h == 0 {
        anyhow::bail!("captured image has zero size");
    }
    let space = CGColorSpace::new_device_rgb()
        .ok_or_else(|| anyhow!("failed to create RGB color space"))?;
    let bitmap_info = CGImageAlphaInfo::PremultipliedLast.0 | CGImageByteOrderInfo::Order32Big.0;
    // SAFETY: sizes/stride are valid; NULL data → CG allocates the buffer.
    let ctx = unsafe {
        CGBitmapContextCreate(
            std::ptr::null_mut(),
            w,
            h,
            8,
            w * 4,
            Some(&space),
            bitmap_info,
        )
    }
    .ok_or_else(|| anyhow!("failed to create bitmap context"))?;
    CGContext::translate_ctm(Some(&ctx), 0.0, h as f64);
    CGContext::scale_ctm(Some(&ctx), 1.0, -1.0);
    CGContext::draw_image(
        Some(&ctx),
        CGRect::new(CGPoint::new(0.0, 0.0), CGSize::new(w as f64, h as f64)),
        Some(image),
    );
    let data = CGBitmapContextGetData(Some(&ctx));
    if data.is_null() {
        anyhow::bail!("bitmap context returned no data");
    }
    // SAFETY: `data` points to `w*h*4` bytes owned by the context; copied before drop.
    let rgba = unsafe { std::slice::from_raw_parts(data as *const u8, w * h * 4) }.to_vec();
    Ok(Capture {
        width: w as u32,
        height: h as u32,
        rgba,
    })
}

fn find_sc_window(
    content: &SCShareableContent,
    pid: u32,
    title: Option<&str>,
    frame: SurfaceGeometry,
) -> Option<Retained<SCWindow>> {
    // SAFETY: SCK getter methods are ObjC messages against live objects.
    unsafe { content.windows() }.iter().find(|window| {
        // SAFETY: ObjC message against a live SCWindow/SCRunningApplication.
        let owner = unsafe { window.owningApplication() };
        let owner_pid = owner.as_ref().map(|a| unsafe { a.processID() } as u32);
        if owner_pid != Some(pid) {
            return false;
        }
        if let Some(t) = title {
            // SAFETY: ObjC message against a live SCWindow.
            if unsafe { window.title() }.map(|s| s.to_string()).as_deref() == Some(t) {
                return true;
            }
        }
        // Fall back to frame proximity when the title didn't match.
        // SAFETY: ObjC message against a live SCWindow.
        let f = unsafe { window.frame() };
        (f.origin.x - frame.x).abs() < 2.0
            && (f.origin.y - frame.y).abs() < 2.0
            && (f.size.width - frame.width).abs() < 2.0
            && (f.size.height - frame.height).abs() < 2.0
    })
}

fn capture_window_blocking(target: &TargetSpec) -> Result<Capture, anyhow::Error> {
    let (window, app) = resolve_window(target)?;
    let pid = app_pid(&app)?;
    let title = window_title(&window);
    let frame = window_frame(&window)?;
    let content = get_shareable_content()?;
    let sc_window = find_sc_window(&content, pid, title.as_deref(), frame).ok_or_else(|| {
        core::taxonomy_error(
            core::ERR_NOT_MATCHED,
            "window not found in shareable content — re-enumerate windows",
        )
    })?;
    // SAFETY: `alloc()` is the fresh allocation consumed by the init.
    let filter = unsafe {
        SCContentFilter::initWithDesktopIndependentWindow(SCContentFilter::alloc(), &sc_window)
    };
    let scale = unsafe { filter.pointPixelScale() } as f64;
    let px_w = (frame.width * scale).round();
    let px_h = (frame.height * scale).round();
    capture_filter_to_rgba(&filter, px_w as usize, px_h as usize, true)
}

/// Composite each active display into one RGBA buffer at the virtual-desktop
/// bounding rect, each display drawn at its global-frame offset. Each display is
/// captured and placed at ITS OWN pixel scale, so a mixed-density setup renders
/// each screen at its native density; when densities differ the composite is
/// sized at the first display's reference scale and a lower-density display
/// occupies a proportionally smaller region of the buffer (an accepted residual
/// limitation of a single-buffer composite).
fn capture_screen_blocking() -> Result<Capture, anyhow::Error> {
    let content = get_shareable_content()?;
    // SAFETY: SCK getter is an ObjC message against a live object.
    let displays = unsafe { content.displays() };
    if displays.is_empty() {
        anyhow::bail!("no shareable displays found");
    }
    let empty: Retained<NSArray<SCWindow>> = NSArray::new();
    let first = displays.iter().next().expect("displays non-empty");
    // SAFETY: `alloc()` is the fresh allocation consumed by the init.
    let first_filter = unsafe {
        SCContentFilter::initWithDisplay_excludingWindows(SCContentFilter::alloc(), &first, &empty)
    };
    let ref_scale = unsafe { first_filter.pointPixelScale() } as f64;
    let bbox = screen_surface()?;
    let out_w = (bbox.width * ref_scale).round() as usize;
    let out_h = (bbox.height * ref_scale).round() as usize;
    if out_w == 0 || out_h == 0 {
        anyhow::bail!("virtual desktop has zero size");
    }
    let mut composite = vec![0u8; out_w * out_h * 4];
    for display in displays {
        // SAFETY: `alloc()` is the fresh allocation consumed by the init.
        let filter = unsafe {
            SCContentFilter::initWithDisplay_excludingWindows(
                SCContentFilter::alloc(),
                &display,
                &empty,
            )
        };
        // SAFETY: ObjC message against a live SCDisplay.
        let f = unsafe { display.frame() };
        let scale = unsafe { filter.pointPixelScale() } as f64;
        let px_w = (f.size.width * scale).round() as usize;
        let px_h = (f.size.height * scale).round() as usize;
        let cap = capture_filter_to_rgba(&filter, px_w, px_h, false)?;
        let dx = ((f.origin.x - bbox.x) * scale).round();
        let dy = ((f.origin.y - bbox.y) * scale).round();
        if dx >= 0.0 && dy >= 0.0 {
            core::blit_rgba(&mut composite, out_w, out_h, &cap, dx as usize, dy as usize);
        }
    }
    Ok(Capture {
        width: out_w as u32,
        height: out_h as u32,
        rgba: composite,
    })
}

fn capture_blocking(target: &TargetSpec) -> Result<Capture, anyhow::Error> {
    match target {
        TargetSpec::Screen => capture_screen_blocking(),
        _ => capture_window_blocking(target),
    }
}

fn capture_available_blocking() -> bool {
    get_shareable_content().is_ok()
}

// ── Element actions ──────────────────────────────────────────────────────

fn press_element(element: &AXUIElement) -> Result<(), anyhow::Error> {
    let names = raw_action_names(element);
    if names.iter().any(|a| a == AX_PRESS) {
        // SAFETY: AXPress was verified to exist on the element.
        map_action_err(
            unsafe { element.perform_action(&axc(AX_PRESS)) },
            "perform press",
        )
    } else if names.iter().any(|a| a == AX_OPEN) {
        // SAFETY: AXOpen was verified to exist on the element.
        map_action_err(
            unsafe { element.perform_action(&axc(AX_OPEN)) },
            "perform open",
        )
    } else {
        let available = names
            .iter()
            .map(|a| normalize_action_name(a))
            .collect::<Vec<_>>()
            .join(", ");
        Err(core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            format!("element has no press/open action; available actions: {available}"),
        ))
    }
}

fn set_element_value(element: &AXUIElement, text: &str) -> Result<(), anyhow::Error> {
    let role = attribute_string(element, &axc(AX_ROLE)).unwrap_or_default();
    let subrole = attribute_string(element, &axc(AX_SUBROLE));
    if is_secure_field(&role, subrole.as_deref()) {
        return Err(core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            "element is a secure text field — SetValue refused (use click by ref then type)",
        ));
    }
    let cf: CFRetained<CFType> = CFRetained::from(CFString::from_str(text));
    // SAFETY: `cf` is a valid CFString (a CFType).
    map_action_err(
        unsafe { element.set_attribute_value(&axc(AX_VALUE), &cf) },
        "set value",
    )
}

fn map_action_err(err: AXError, ctx: &str) -> Result<(), anyhow::Error> {
    match err {
        AXError::Success => Ok(()),
        AXError::AttributeUnsupported => Err(core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            format!(
                "{ctx} — element does not support SetValue; use click by ref then type without ref"
            ),
        )),
        AXError::InvalidUIElement => Err(core::taxonomy_error(
            core::ERR_STALE_ELEMENT,
            format!("{ctx} — window closed or changed — re-enumerate"),
        )),
        AXError::APIDisabled => Err(core::taxonomy_error(
            core::ERR_PERMISSION_DENIED,
            format!("{ctx} — Accessibility TCC grant missing"),
        )),
        AXError::CannotComplete => Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            format!("{ctx} — transient AX failure, retry"),
        )),
        _ => Err(anyhow!("AX error {err:?} while {ctx}")),
    }
}

// ── Backend impl ─────────────────────────────────────────────────────────

#[async_trait::async_trait]
impl Backend for MacOsBackend {
    fn accessibility_available(&self) -> bool {
        // SAFETY: AXIsProcessTrusted is a cheap sync query (no prompt variant).
        unsafe { AXIsProcessTrusted() }
    }

    async fn capture_available(&self) -> bool {
        with_block_in_place(capture_available_blocking)
    }

    fn capture_unavailable_error(&self) -> anyhow::Error {
        core::taxonomy_error(
            core::ERR_PERMISSION_DENIED,
            "Screen Recording permission is not granted (System Settings → Privacy & Security → \
             Screen Recording). On macOS a plain unbundled binary may not be grantable — wrapping \
             the binary in a minimal .app bundle is the known workaround (provisioning is a \
             separate follow-up). Accessibility observe/act remain available without this grant.",
        )
    }

    async fn list_apps(&self) -> Result<Vec<AppInfo>, anyhow::Error> {
        with_block_in_place(|| {
            let mut out = Vec::new();
            for app in NSWorkspace::sharedWorkspace().runningApplications() {
                if app.activationPolicy() != NSApplicationActivationPolicy::Regular {
                    continue;
                }
                let name = app.localizedName().map_or(String::new(), |s| s.to_string());
                if !name.is_empty() {
                    out.push(AppInfo {
                        pid: Some(app.processIdentifier() as u32),
                        name,
                    });
                }
            }
            out.sort_by(|a, b| a.name.cmp(&b.name));
            out.dedup_by(|a, b| a.pid == b.pid);
            Ok(out)
        })
    }

    async fn list_windows(&self, app: Option<&AppInfo>) -> Result<Vec<WindowInfo>, anyhow::Error> {
        with_block_in_place(|| {
            let app_infos: Vec<AppInfo> = if let Some(a) = app {
                vec![a.clone()]
            } else {
                let mut v = Vec::new();
                for run in NSWorkspace::sharedWorkspace().runningApplications() {
                    if run.activationPolicy() != NSApplicationActivationPolicy::Regular {
                        continue;
                    }
                    let name = run.localizedName().map_or(String::new(), |s| s.to_string());
                    if !name.is_empty() {
                        v.push(AppInfo {
                            pid: Some(run.processIdentifier() as u32),
                            name,
                        });
                    }
                }
                v
            };
            let mut out = Vec::new();
            for ai in app_infos {
                let Some(pid) = ai.pid else { continue };
                let app_el = unsafe { AXUIElement::new_application(pid as libc::pid_t) };
                let Ok(windows) = read_windows(&app_el) else {
                    continue;
                };
                for (idx, window) in windows.iter().enumerate() {
                    let Ok(frame) = window_frame(window) else {
                        continue;
                    };
                    let title = window_title(window).unwrap_or_default();
                    if title.is_empty() && (frame.width <= 0.0 || frame.height <= 0.0) {
                        continue;
                    }
                    out.push(WindowInfo {
                        app_pid: Some(pid),
                        app_name: ai.name.clone(),
                        title,
                        index: idx,
                        surface: frame,
                    });
                }
            }
            Ok(out)
        })
    }

    async fn focused_window(&self) -> Result<WindowInfo, anyhow::Error> {
        with_block_in_place(|| {
            let (window, app) = resolve_window(&TargetSpec::Focused)?;
            let pid = app_pid(&app)?;
            Ok(WindowInfo {
                app_pid: Some(pid),
                app_name: app_display_name(pid, &app),
                title: window_title(&window).unwrap_or_default(),
                index: 0,
                surface: window_frame(&window)?,
            })
        })
    }

    async fn surface_geometry(
        &self,
        target: &TargetSpec,
    ) -> Result<SurfaceGeometry, anyhow::Error> {
        with_block_in_place(|| surface_geometry_blocking(target))
    }

    async fn observe(&self, target: &TargetSpec) -> Result<Observation, anyhow::Error> {
        with_block_in_place(|| {
            let (window, app) = resolve_window(target)?;
            let origin = window_origin(&window)?;
            let surface = window_frame(&window)?;
            let app_name = match target {
                TargetSpec::Window { app_name, .. } => app_name.clone(),
                _ => attribute_string(&app, &axc(AX_TITLE)).unwrap_or_default(),
            };
            let (root, _handles) = build_tree(&window, origin);
            Ok(Observation {
                app_name,
                window_title: window_title(&window),
                surface,
                root,
            })
        })
    }

    async fn act_on_element(
        &self,
        target: &TargetSpec,
        locator: &Locator,
        act: ElementAct,
    ) -> Result<(), anyhow::Error> {
        with_block_in_place(|| {
            let (window, _app) = resolve_window(target)?;
            let (root, handles) = build_tree(&window, window_origin(&window)?);
            let matched = match core::resolve_locator(&root, locator) {
                core::LocatorMatch::Path(node) | core::LocatorMatch::Unique(node) => node,
                core::LocatorMatch::Ambiguous => {
                    return Err(core::taxonomy_error(
                        core::ERR_AMBIGUOUS_LOCATOR,
                        "locator matches multiple elements — re-observe and pick a more specific ref",
                    ));
                }
                core::LocatorMatch::NotFound => {
                    return Err(core::taxonomy_error(
                        core::ERR_NOT_MATCHED,
                        "element no longer matches its locator — re-observe",
                    ));
                }
            };
            let element = &handles[core::pre_order_index(&root, matched)];
            match act {
                ElementAct::Press => press_element(element),
                ElementAct::SetValue { text } => set_element_value(element, &text),
            }
        })
    }

    async fn raw_input(&self, target: &TargetSpec, input: RawInput) -> Result<(), anyhow::Error> {
        with_block_in_place(|| raw_input_blocking(target, &input))
    }

    async fn cursor_position(&self) -> Result<(f64, f64), anyhow::Error> {
        with_block_in_place(|| {
            let event =
                CGEvent::new(None).ok_or_else(|| anyhow!("failed to create dummy cursor event"))?;
            let p = CGEvent::location(Some(&event));
            Ok((p.x, p.y))
        })
    }

    async fn capture(&self, target: &TargetSpec) -> Result<Capture, anyhow::Error> {
        with_block_in_place(|| capture_blocking(target))
    }
}

fn window_title(window: &AXUIElement) -> Option<String> {
    attribute_string(window, &axc(AX_TITLE))
}

// ── Tests (pure logic only; no FFI) ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_table_covers_chord_parser_keys() {
        for key in [
            "a",
            "z",
            "5",
            "0",
            "return",
            "space",
            "backslash",
            "f1",
            "f12",
            "capslock",
        ] {
            let (mods, parsed) = core::parse_key_chord(key).unwrap();
            assert!(mods.is_empty());
            assert_eq!(parsed, key.to_lowercase(), "chord parser mangles '{key}'");
            assert!(key_code(key).is_some(), "kVK table missing '{key}'");
        }
        let (mods, parsed) = core::parse_key_chord("cmd+shift+t").unwrap();
        assert_eq!(mods, vec![Modifier::Cmd, Modifier::Shift]);
        assert_eq!(parsed, "t");
    }

    #[test]
    fn virtual_desktop_bounding_rect_union() {
        let rects = [
            (0.0, 0.0, 1920.0, 1080.0),
            (1920.0, 0.0, 1920.0, 1080.0),
            (0.0, -100.0, 800.0, 600.0),
        ];
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        for (x, y, w, h) in rects {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x + w);
            max_y = max_y.max(y + h);
        }
        let geo = SurfaceGeometry {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        };
        assert_eq!(
            (geo.x, geo.y, geo.width, geo.height),
            (0.0, -100.0, 3840.0, 1180.0)
        );
        let (px, py) = core::normalized_to_surface(500.0, 500.0, &geo).unwrap();
        assert!((px - 1920.0).abs() < 1e-9, "x = {px}");
        assert!((py - 490.0).abs() < 1e-9, "y = {py}");
    }

    #[test]
    fn blit_rgba_clips_and_copies() {
        let mut dest = vec![0u8; 4 * 4 * 4];
        let src = Capture {
            width: 2,
            height: 2,
            rgba: vec![1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255],
        };
        core::blit_rgba(&mut dest, 4, 4, &src, 1, 1);
        assert_eq!(&dest[4 * 4 + 4..4 * 4 + 8], &[1, 2, 3, 255]);
        assert_eq!(&dest[4 * 4 + 8..4 * 4 + 12], &[4, 5, 6, 255]);
    }
}
