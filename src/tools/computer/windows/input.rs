//! Pointer and keyboard synthesis for the Windows backend, through `SendInput`.
//!
//! `SendInput` is session-global: it delivers to whatever window has focus, so a
//! window target is first brought to the foreground (best-effort — Windows may
//! refuse a foreground change to a background process). Input that is pinned to a
//! named window — typing, a drag, a wheel at a point — then verifies the window is
//! actually in front and refuses rather than reporting a success that landed
//! elsewhere; a coordinate click, a point-less wheel and a key chord are
//! deliberately unpinned and go where the system sends them. Coordinates are
//! physical pixels throughout, matching the UIA tree and the capture.
//!
//! Every batch is refused before it is sent when the secure desktop is up or the
//! window belongs to an elevated process (see [`super::window`]); a batch Windows
//! refuses outright is an error, because nothing was delivered. A batch Windows
//! drops part-way — the UIPI case, where `SendInput` inserts fewer events and reports
//! no error — comes back applied-but-unconfirmed, naming how much of the call's input
//! was delivered.
//!
//! Chord syntax is the shared one: `cmd` is the Windows key here, `win` is also
//! accepted as a key name for a plain Windows-key tap.

use super::super::core::{
    self, Impact, Modifier, MouseButton, RawInput, ScrollDirection, SurfaceGeometry, TargetSpec,
};
use super::window;
use anyhow::anyhow;
use windows::Win32::Foundation::HWND;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBD_EVENT_FLAGS, KEYBDINPUT, KEYEVENTF_KEYUP,
    KEYEVENTF_UNICODE, MOUSE_EVENT_FLAGS, MOUSEEVENTF_HWHEEL, MOUSEEVENTF_LEFTDOWN,
    MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, MOUSEEVENTF_RIGHTDOWN,
    MOUSEEVENTF_RIGHTUP, MOUSEEVENTF_WHEEL, MOUSEINPUT, SendInput, VIRTUAL_KEY, VK_A, VK_BACK,
    VK_CAPITAL, VK_CONTROL, VK_DELETE, VK_DOWN, VK_END, VK_ESCAPE, VK_F1, VK_HOME, VK_INSERT,
    VK_LEFT, VK_LWIN, VK_MENU, VK_NEXT, VK_OEM_1, VK_OEM_2, VK_OEM_3, VK_OEM_4, VK_OEM_5, VK_OEM_6,
    VK_OEM_7, VK_OEM_COMMA, VK_OEM_MINUS, VK_OEM_PERIOD, VK_OEM_PLUS, VK_PRIOR, VK_RETURN,
    VK_RIGHT, VK_SHIFT, VK_SPACE, VK_TAB, VK_UP,
};
use windows::Win32::UI::WindowsAndMessaging::{GetCursorPos, SetCursorPos};

/// Points of interpolation in a drag, and the pause between them — the same shape
/// the other backends use, so a drag reads as one gesture to the application.
const DRAG_STEPS: usize = 12;
const DRAG_STEP_PAUSE: std::time::Duration = std::time::Duration::from_millis(2);

/// Events per `SendInput` batch, so typing a long text never builds one huge array.
const INPUT_BATCH: usize = 256;

/// One wheel notch. `SendInput` carries the wheel delta in `mouseData`.
const WHEEL_DELTA: i32 = 120;

const KNOWN_KEYS: &str = "return/enter, tab, space, delete/backspace, forwarddelete, escape/esc, \
     home, end, pageup/pagedown, up/down/left/right, f1..f24, capslock, insert, minus, equal, \
     comma, period, slash, semicolon, quote, backslash, backtick, leftbracket, rightbracket, \
     win/windows/super/meta (a plain Windows-key tap), a-z, 0-9";

// ── Raw input ───────────────────────────────────────────────────────────

pub(super) fn raw_input(target: &TargetSpec, input: RawInput) -> anyhow::Result<Impact> {
    let pinned = core::pinned_input(target, &input);
    match input {
        RawInput::Click {
            point,
            button,
            double,
            modifiers,
        } => click(target, point, button, double, &modifiers),
        RawInput::TypeText { text } => {
            let resolved = resolve_window(target)?;
            // Typing goes to whatever holds focus, so that is the window the refusal
            // is about.
            activate_target(&Aim::of(resolved.as_ref(), window::foreground), pinned)?;
            send(&type_events(&text))
        }
        RawInput::KeyChord { chord } => {
            // A chord is global: it goes to whatever already has focus (as on
            // Linux), so the target is never resolved or raised here — only the
            // refusals are checked.
            let (modifiers, key) = core::parse_key_chord(&chord)?;
            let key = vk_for(&key)?;
            window::ensure_actuation_allowed(window::foreground())?;
            send(&chord_events(&modifiers, key))
        }
        RawInput::Scroll {
            point,
            direction,
            amount,
        } => scroll(target, point, direction, amount, pinned),
        RawInput::Drag { from, to } => drag(target, from, to, pinned),
    }
}

/// Pointer position in absolute physical pixels.
pub(super) fn cursor_position() -> anyhow::Result<(f64, f64)> {
    let mut point = windows::Win32::Foundation::POINT::default();
    // SAFETY: `point` is a live local of exactly the type the call fills; failure (no
    // attached desktop) becomes the error below.
    unsafe { GetCursorPos(&raw mut point) }.map_err(|e| {
        core::taxonomy_error(
            core::ERR_DEGRADED,
            format!("failed to read the pointer position: {e}"),
        )
    })?;
    Ok((f64::from(point.x), f64::from(point.y)))
}

/// A left click at an absolute physical point — the element-action fallback for an
/// element that supports no accessibility pattern at all.
pub(super) fn click_absolute(x: i32, y: i32) -> anyhow::Result<Impact> {
    window::ensure_actuation_allowed(window::top_level_at(x, y))?;
    set_cursor(x, y)?;
    send(&[
        mouse_event(0, 0, 0, MOUSEEVENTF_LEFTDOWN),
        mouse_event(0, 0, 0, MOUSEEVENTF_LEFTUP),
    ])
}

/// Type text into whatever has focus (used by the element-action fallback after
/// the element itself was focused).
pub(super) fn type_text_into_focus(text: &str) -> anyhow::Result<Impact> {
    send(&type_events(text))
}

/// Select everything in the focused element — the prelude to typing over a value
/// the value pattern refused to set.
pub(super) fn select_all() -> anyhow::Result<Impact> {
    send(&chord_events(&[Modifier::Ctrl], VK_A))
}

/// What one input batch is aimed at, as the refusals need to see it. A window target
/// is checked as a whole and raised; the `screen` target names no window, so the check
/// falls to the window the input touches — `None` there is the desktop itself, which no
/// process owns and which cannot be elevated.
enum Aim<'a> {
    Window(&'a window::TargetWindow),
    Screen(Option<HWND>),
}

impl<'a> Aim<'a> {
    /// The aim for a resolved window target, or — for the screen — whatever the
    /// caller's probe finds where the input lands. The one place the pairing is
    /// decided, and the probe runs only when there is no window target.
    fn of(
        resolved: Option<&'a window::TargetWindow>,
        screen: impl FnOnce() -> Option<HWND>,
    ) -> Self {
        match resolved {
            Some(resolved) => Self::Window(resolved),
            None => Self::Screen(screen()),
        }
    }
}

/// Check the two refusals before input, and raise a window target. Pinned input —
/// typed text, a drag, a wheel at a point on a named window — is meant for that very
/// window, so a foreground change Windows refused fails the call here instead of
/// reporting a success that would land in whatever is in front; unpinned input (a
/// coordinate click, a point-less wheel) only raises, because it goes where the
/// system sends it.
fn activate_target(aim: &Aim<'_>, pinned: bool) -> anyhow::Result<()> {
    match aim {
        Aim::Window(resolved) => {
            window::ensure_actuation_allowed(Some(resolved.hwnd))?;
            window::activate(resolved.hwnd);
            if pinned {
                window::ensure_foreground(resolved.hwnd)?;
            }
            Ok(())
        }
        Aim::Screen(landing) => window::ensure_actuation_allowed(*landing),
    }
}

/// The target's window, resolved once. `Screen` is the whole virtual desktop and
/// names no window.
fn resolve_window(target: &TargetSpec) -> anyhow::Result<Option<window::TargetWindow>> {
    match target {
        TargetSpec::Screen => Ok(None),
        other => window::resolve(other).map(Some),
    }
}

/// The surface the target's normalized points are expressed in: the window's own
/// frame, or the whole virtual desktop.
fn surface_of(resolved: Option<&window::TargetWindow>) -> anyhow::Result<SurfaceGeometry> {
    match resolved {
        Some(resolved) => Ok(resolved.surface),
        None => window::screen_geometry(),
    }
}

fn click(
    target: &TargetSpec,
    point: (f64, f64),
    button: MouseButton,
    double: bool,
    modifiers: &[Modifier],
) -> anyhow::Result<Impact> {
    let resolved = resolve_window(target)?;
    let geometry = surface_of(resolved.as_ref())?;
    let (x, y) = core::normalized_to_surface(point.0, point.1, &geometry)?;
    let (x, y) = (round_i32(x), round_i32(y));
    // A coordinate click is pointer input, never pinned: raise, but the click lands
    // on whatever is under the point.
    activate_target(
        &Aim::of(resolved.as_ref(), || window::top_level_at(x, y)),
        false,
    )?;
    set_cursor(x, y)?;
    let (down, up) = match button {
        MouseButton::Left => (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
        MouseButton::Right => (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
        MouseButton::Middle => (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
    };
    let mut events = Vec::with_capacity(modifiers.len() * 2 + 4);
    push_modifiers(&mut events, modifiers, false);
    for _ in 0..if double { 2 } else { 1 } {
        events.push(mouse_event(0, 0, 0, down));
        events.push(mouse_event(0, 0, 0, up));
    }
    push_modifiers(&mut events, modifiers, true);
    send(&events)
}

fn scroll(
    target: &TargetSpec,
    point: Option<(f64, f64)>,
    direction: ScrollDirection,
    amount: u32,
    pinned: bool,
) -> anyhow::Result<Impact> {
    let resolved = resolve_window(target)?;
    let at = match point {
        Some((nx, ny)) => {
            let geometry = surface_of(resolved.as_ref())?;
            let (x, y) = core::normalized_to_surface(nx, ny, &geometry)?;
            Some((round_i32(x), round_i32(y)))
        }
        None => None,
    };
    // The elevation refusal is about the window the wheel lands on: the one under the
    // point. Only a `screen`-targeted wheel reaches this probe (a point-less wheel is
    // dispatched to the focused surface, whose window the window arm checks), and that
    // shape always carries a point.
    activate_target(
        &Aim::of(resolved.as_ref(), || {
            at.and_then(|(x, y)| window::top_level_at(x, y))
        }),
        pinned,
    )?;
    if let Some((x, y)) = at {
        set_cursor(x, y)?;
    }
    let ticks = i32::try_from(amount.max(1)).unwrap_or(i32::MAX);
    let delta = ticks.saturating_mul(WHEEL_DELTA);
    let vertical = matches!(direction, ScrollDirection::Up | ScrollDirection::Down);
    let sign = match direction {
        ScrollDirection::Up | ScrollDirection::Right => 1,
        ScrollDirection::Down | ScrollDirection::Left => -1,
    };
    let data = delta.saturating_mul(sign).cast_unsigned();
    let flags = if vertical {
        MOUSEEVENTF_WHEEL
    } else {
        MOUSEEVENTF_HWHEEL
    };
    send(&[mouse_event(0, 0, data, flags)])
}

fn drag(
    target: &TargetSpec,
    from: (f64, f64),
    to: (f64, f64),
    pinned: bool,
) -> anyhow::Result<Impact> {
    let resolved = resolve_window(target)?;
    let geometry = surface_of(resolved.as_ref())?;
    let (fx, fy) = core::normalized_to_surface(from.0, from.1, &geometry)?;
    let (tx, ty) = core::normalized_to_surface(to.0, to.1, &geometry)?;
    // The gesture starts at `from`, so that is the landing a screen-target drag is
    // checked against — the same rule a click uses.
    activate_target(
        &Aim::of(resolved.as_ref(), || {
            window::top_level_at(round_i32(fx), round_i32(fy))
        }),
        pinned,
    )?;
    set_cursor(round_i32(fx), round_i32(fy))?;
    // A single-event batch is either delivered or refused outright, so a refused
    // press means nothing of the gesture is in the system yet.
    send(&[mouse_event(0, 0, 0, MOUSEEVENTF_LEFTDOWN)])?;
    for step in 1..=DRAG_STEPS {
        let progress = step as f64 / DRAG_STEPS as f64;
        let x = round_i32(fx + (tx - fx) * progress);
        let y = round_i32(fy + (ty - fy) * progress);
        if let Err(e) = set_cursor(x, y) {
            // The gesture is already under way: let the button go, then report what
            // landed rather than "nothing happened", which would invite a second drag.
            let released = match send(&[mouse_event(0, 0, 0, MOUSEEVENTF_LEFTUP)]) {
                Ok(_) => "the button was released again".to_string(),
                Err(err) => format!("the button may still be held down ({err:#})"),
            };
            return Ok(Impact::Unconfirmed(format!(
                "the drag started (the button went down at ({}, {})) but the pointer could not be \
                 moved further — {e}; {released}",
                round_i32(fx),
                round_i32(fy)
            )));
        }
        std::thread::sleep(DRAG_STEP_PAUSE);
    }
    // The gesture ran to its end, so only the release is in question: a refusal
    // leaves the button this call pressed down, which the model must know.
    let released = match send(&[mouse_event(0, 0, 0, MOUSEEVENTF_LEFTUP)]) {
        Ok(impact) => return Ok(impact),
        Err(err) => format!("{err:#}"),
    };
    Ok(Impact::Unconfirmed(format!(
        "the drag ran to its end but the button's release was refused ({released}) — the button \
         may still be held down"
    )))
}

fn set_cursor(x: i32, y: i32) -> anyhow::Result<()> {
    // SAFETY: two integers in, no pointer or handle; a refusal is the returned error.
    unsafe { SetCursorPos(x, y) }.map_err(|e| {
        core::taxonomy_error(
            core::ERR_DEGRADED,
            format!("failed to move the pointer to ({x}, {y}): {e}"),
        )
    })
}

fn round_i32(value: f64) -> i32 {
    value.round() as i32
}

// ── Event construction ──────────────────────────────────────────────────

fn mouse_event(dx: i32, dy: i32, data: u32, flags: MOUSE_EVENT_FLAGS) -> INPUT {
    INPUT {
        r#type: INPUT_MOUSE,
        Anonymous: INPUT_0 {
            mi: MOUSEINPUT {
                dx,
                dy,
                mouseData: data,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

fn key_event(key: VIRTUAL_KEY, flags: KEYBD_EVENT_FLAGS) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: key,
                wScan: 0,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

/// A Unicode code unit (surrogate pairs are sent as their two halves).
fn unicode_event(unit: u16, flags: KEYBD_EVENT_FLAGS) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(0),
                wScan: unit,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

fn push_modifiers(events: &mut Vec<INPUT>, modifiers: &[Modifier], release: bool) {
    // Released in the reverse of the press order, like a human lets a chord go.
    let ordered: Vec<Modifier> = if release {
        modifiers.iter().rev().copied().collect()
    } else {
        modifiers.to_vec()
    };
    for modifier in ordered {
        let key = modifier_key(modifier);
        events.push(key_event(
            key,
            if release {
                KEYEVENTF_KEYUP
            } else {
                KEYBD_EVENT_FLAGS(0)
            },
        ));
    }
}

fn chord_events(modifiers: &[Modifier], key: VIRTUAL_KEY) -> Vec<INPUT> {
    let mut events = Vec::with_capacity(modifiers.len() * 2 + 2);
    push_modifiers(&mut events, modifiers, false);
    events.push(key_event(key, KEYBD_EVENT_FLAGS(0)));
    events.push(key_event(key, KEYEVENTF_KEYUP));
    push_modifiers(&mut events, modifiers, true);
    events
}

/// Text as keyboard events: a newline or tab is a real key press (a raw `\r` is
/// ignored by most edit controls), everything else is a Unicode code unit.
fn type_events(text: &str) -> Vec<INPUT> {
    let mut events = Vec::with_capacity(text.len() * 2);
    let mut previous = '\0';
    for character in text.chars() {
        let was_carriage_return = previous == '\r';
        previous = character;
        match character {
            // A CRLF pair is one Enter, not two.
            '\n' if was_carriage_return => {}
            '\n' | '\r' => events.extend(chord_events(&[], VK_RETURN)),
            '\t' => events.extend(chord_events(&[], VK_TAB)),
            other => {
                let mut units = [0u16; 2];
                for unit in other.encode_utf16(&mut units) {
                    events.push(unicode_event(*unit, KEYEVENTF_UNICODE));
                    events.push(unicode_event(*unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP));
                }
            }
        }
    }
    events
}

fn modifier_key(modifier: Modifier) -> VIRTUAL_KEY {
    match modifier {
        // Windows' own convention: `cmd` is the Windows key on this platform.
        Modifier::Cmd => VK_LWIN,
        Modifier::Ctrl => VK_CONTROL,
        Modifier::Alt => VK_MENU,
        Modifier::Shift => VK_SHIFT,
    }
}

/// The virtual key for a chord's key name. Unknown names are a plain argument
/// error listing what is accepted.
fn vk_for(key: &str) -> anyhow::Result<VIRTUAL_KEY> {
    let key = key.trim();
    let named = match key {
        "return" | "enter" => Some(VK_RETURN),
        "tab" => Some(VK_TAB),
        "space" => Some(VK_SPACE),
        // The shared vocabulary: `delete` is the Backspace key (as on macOS and
        // Linux); the forward-delete key is `forwarddelete`.
        "delete" | "backspace" => Some(VK_BACK),
        "forwarddelete" => Some(VK_DELETE),
        "escape" | "esc" => Some(VK_ESCAPE),
        "home" => Some(VK_HOME),
        "end" => Some(VK_END),
        "pageup" => Some(VK_PRIOR),
        "pagedown" => Some(VK_NEXT),
        "up" => Some(VK_UP),
        "down" => Some(VK_DOWN),
        "left" => Some(VK_LEFT),
        "right" => Some(VK_RIGHT),
        "capslock" => Some(VK_CAPITAL),
        "insert" => Some(VK_INSERT),
        "minus" => Some(VK_OEM_MINUS),
        "equal" => Some(VK_OEM_PLUS),
        "comma" => Some(VK_OEM_COMMA),
        "period" => Some(VK_OEM_PERIOD),
        "slash" => Some(VK_OEM_2),
        "semicolon" => Some(VK_OEM_1),
        "quote" => Some(VK_OEM_7),
        "backslash" => Some(VK_OEM_5),
        "backtick" => Some(VK_OEM_3),
        "leftbracket" => Some(VK_OEM_4),
        "rightbracket" => Some(VK_OEM_6),
        "win" | "windows" | "super" | "meta" => Some(VK_LWIN),
        _ => None,
    };
    if let Some(key) = named {
        return Ok(key);
    }
    let lower = key.to_ascii_lowercase();
    if let Some(number) = lower.strip_prefix('f')
        && let Ok(index) = number.parse::<u16>()
        && (1..=24).contains(&index)
    {
        return Ok(VIRTUAL_KEY(VK_F1.0 + index - 1));
    }
    if lower.len() == 1 {
        let character = lower.as_bytes()[0];
        match character {
            b'a'..=b'z' => return Ok(VIRTUAL_KEY(u16::from(character - b'a') + 0x41)),
            b'0'..=b'9' => return Ok(VIRTUAL_KEY(u16::from(character - b'0') + 0x30)),
            _ => {}
        }
    }
    Err(anyhow!(
        "unknown key name '{key}' — accepted keys: {KNOWN_KEYS}"
    ))
}

// ── Sending ─────────────────────────────────────────────────────────────

/// Send one batch. `SendInput` inserts fewer events than offered when UIPI blocks
/// the input — it reports no error, only a short count. Delivering nothing is a
/// refusal; a short batch is [`Impact::Unconfirmed`], naming how much of the whole
/// call was delivered, after what this batch left held is let go.
fn send(events: &[INPUT]) -> anyhow::Result<Impact> {
    if events.is_empty() {
        return Ok(Impact::Done);
    }
    for (index, batch) in events.chunks(INPUT_BATCH).enumerate() {
        // SAFETY: the binding passes `batch`'s own length as the event count and
        // `size_of::<INPUT>()` as the stride, and the slice outlives the call.
        let sent = unsafe { SendInput(batch, std::mem::size_of::<INPUT>() as i32) };
        if sent as usize != batch.len() {
            // Everything this call delivered so far, earlier batches included: a
            // modifier pressed there is still down.
            let delivered = index * INPUT_BATCH + sent as usize;
            if delivered == 0 {
                return Err(core::taxonomy_error(
                    core::ERR_DEGRADED,
                    "Windows refused the input before any of it was delivered — the foreground \
                     window may belong to an elevated process, or the secure desktop is up; \
                     nothing was applied, so bring the window to the foreground and act again",
                ));
            }
            let mut detail = format!(
                "{delivered} of {} events were delivered and the rest were dropped",
                events.len()
            );
            if let Some(repair) = release_held(&events[..delivered]) {
                detail.push_str("; ");
                detail.push_str(&repair);
            }
            return Ok(Impact::Unconfirmed(detail));
        }
    }
    Ok(Impact::Done)
}

/// Release whatever the landed events left held, buttons before keys and in reverse
/// order, and name what a refused repair leaves down. Without this a partial
/// insertion (a desktop switch mid-batch) leaves a modifier held, or — for a click
/// truncated between down and up — the mouse button down, until the owner presses
/// and releases it again.
fn release_held(landed: &[INPUT]) -> Option<String> {
    let mut keys: Vec<VIRTUAL_KEY> = Vec::new();
    let mut buttons: Vec<MOUSE_EVENT_FLAGS> = Vec::new();
    for input in landed {
        if input.r#type == INPUT_KEYBOARD {
            // SAFETY: `r#type` says this union holds a keyboard event, which is what
            // was pushed; reading `ki` copies plain integers out.
            let key = unsafe { input.Anonymous.ki };
            if key.dwFlags.contains(KEYEVENTF_KEYUP) {
                keys.retain(|vk| *vk != key.wVk);
            } else if !key.dwFlags.contains(KEYEVENTF_UNICODE) {
                keys.push(key.wVk);
            }
        } else if input.r#type == INPUT_MOUSE {
            // SAFETY: as above for the mouse arm — plain integers only.
            let flags = unsafe { input.Anonymous.mi.dwFlags };
            for (down, up) in [
                (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
                (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
                (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
            ] {
                if flags.contains(down) {
                    buttons.push(up);
                } else if flags.contains(up) {
                    buttons.retain(|held| *held != up);
                }
            }
        }
    }
    let mut releases: Vec<INPUT> = buttons
        .iter()
        .rev()
        .map(|up| mouse_event(0, 0, 0, *up))
        .collect();
    releases.extend(
        keys.iter()
            .rev()
            .map(|key| key_event(*key, KEYEVENTF_KEYUP)),
    );
    if releases.is_empty() {
        return None;
    }
    // SAFETY: same contract as `send` — the slice's own length and element size, and it
    // outlives the call.
    let sent = unsafe { SendInput(&releases, std::mem::size_of::<INPUT>() as i32) };
    (sent as usize != releases.len()).then(|| {
        "the release of what landed was itself refused, so a key or button may still be held down"
            .to_string()
    })
}
