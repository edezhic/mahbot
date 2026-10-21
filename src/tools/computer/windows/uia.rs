//! The accessibility half of the Windows backend: reading a window's UI Automation
//! tree and acting on the element a ref points at.
//!
//! Runs on the apartment thread (see [`super::thread`]) and only ever holds
//! platform handles inside one call.
//!
//! The tree is the CONTROL view (`ControlViewWalker`) — the structure an
//! accessibility client is meant to consume: real controls, not the raw provider
//! scaffolding. Roles are UIA control types mapped onto the shared vocabulary
//! [`super::super::core::INTERACTIVE_ROLES`] already understands (`edit` →
//! `textfield`, `hyperlink` → `link`, `tabitem` → `tab`), so a ref's role matches
//! the same way on every platform. `actions` is derived from the control type as a
//! hint for the model; the real pattern support is probed when an action runs.
//!
//! Reading is capped lower than on the other platforms (2000 nodes) because UIA
//! charges one cross-process call per property. A node or depth cap stops the walk
//! with the header still reporting the real counts; a provider that runs out of
//! TIME is different and abandons the whole read with an honest `degraded` timeout
//! rather than returning half of a tree. A provider that refuses a single child or
//! property only ends that subtree or leaves one field absent, the way a broken
//! accessibility attribute does on the other platforms.

use super::super::core::{
    self, ElementAct, Locator, Observation, SurfaceGeometry, TargetSpec, UiNode,
};
use super::{input, window};
use std::time::{Duration, Instant};
use windows::Win32::Foundation::{E_ACCESSDENIED, E_NOTIMPL};
use windows::Win32::Foundation::{HWND, POINT, RECT};
use windows::Win32::UI::Accessibility::{
    ExpandCollapseState_Expanded, IUIAutomation, IUIAutomationElement,
    IUIAutomationExpandCollapsePattern, IUIAutomationInvokePattern,
    IUIAutomationLegacyIAccessiblePattern, IUIAutomationSelectionItemPattern,
    IUIAutomationTogglePattern, IUIAutomationTreeWalker, IUIAutomationValuePattern,
    UIA_AppBarControlTypeId, UIA_ButtonControlTypeId, UIA_CalendarControlTypeId,
    UIA_CheckBoxControlTypeId, UIA_ComboBoxControlTypeId, UIA_CustomControlTypeId,
    UIA_DataGridControlTypeId, UIA_DataItemControlTypeId, UIA_DocumentControlTypeId,
    UIA_E_ELEMENTNOTAVAILABLE, UIA_E_ELEMENTNOTENABLED, UIA_E_NOTSUPPORTED, UIA_EditControlTypeId,
    UIA_ExpandCollapsePatternId, UIA_GroupControlTypeId, UIA_HeaderControlTypeId,
    UIA_HeaderItemControlTypeId, UIA_HyperlinkControlTypeId, UIA_ImageControlTypeId,
    UIA_InvokePatternId, UIA_LegacyIAccessiblePatternId, UIA_ListControlTypeId,
    UIA_ListItemControlTypeId, UIA_MenuBarControlTypeId, UIA_MenuControlTypeId,
    UIA_MenuItemControlTypeId, UIA_PATTERN_ID, UIA_PaneControlTypeId, UIA_ProgressBarControlTypeId,
    UIA_RadioButtonControlTypeId, UIA_ScrollBarControlTypeId, UIA_SelectionItemPatternId,
    UIA_SemanticZoomControlTypeId, UIA_SeparatorControlTypeId, UIA_SliderControlTypeId,
    UIA_SpinnerControlTypeId, UIA_SplitButtonControlTypeId, UIA_StatusBarControlTypeId,
    UIA_TabControlTypeId, UIA_TabItemControlTypeId, UIA_TableControlTypeId, UIA_TextControlTypeId,
    UIA_ThumbControlTypeId, UIA_TitleBarControlTypeId, UIA_TogglePatternId,
    UIA_ToolBarControlTypeId, UIA_ToolTipControlTypeId, UIA_TreeControlTypeId,
    UIA_TreeItemControlTypeId, UIA_ValuePatternId, UIA_WindowControlTypeId,
};
use windows::core::BSTR;

/// How long one walk may spend before it gives up — inside the caller's
/// [`super::OBSERVE_TIMEOUT`], so a stuck provider still ends as a clean timeout.
pub(super) const WALK_DEADLINE: Duration = Duration::from_secs(10);

/// Nodes per walk. UIA pays one cross-process read per property, so this is lower
/// than the other backends' render cap; the header the model sees carries the real
/// counts of what was read.
const MAX_NODES: usize = 2000;
/// Depth per walk — a guard against provider cycles, like the other backends'.
const MAX_DEPTH: usize = 25;
/// Cap on a value pulled into a node, mirroring the Linux backend's.
const MAX_VALUE_CHARS: usize = 2000;
/// Cap on an element name; a name is a label, not a document.
const MAX_NAME_CHARS: usize = 200;

/// Roles whose value is worth a second cross-process read.
const VALUE_ROLES: &[&str] = &[
    "textfield",
    "combobox",
    "text",
    "document",
    "spinner",
    "slider",
];

// ── Reading ─────────────────────────────────────────────────────────────

pub(super) fn observe(uia: &IUIAutomation, target: &TargetSpec) -> anyhow::Result<Observation> {
    if matches!(target, TargetSpec::Screen) {
        return Err(core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            "observe targets an accessibility surface — use screenshot for the screen",
        ));
    }
    let resolved = window::resolve(target)?;
    let element = window_element(uia, &resolved)?;
    let tree = build_tree(uia, &element, resolved.surface)?;
    Ok(Observation {
        app_name: resolved.app_name,
        window_title: Some(resolved.title),
        surface: resolved.surface,
        root: tree.root,
    })
}

pub(super) fn act_on_element(
    uia: &IUIAutomation,
    target: &TargetSpec,
    locator: &Locator,
    act: &ElementAct,
) -> anyhow::Result<()> {
    let resolved = window::resolve(target)?;
    window::ensure_actuation_allowed(Some(resolved.hwnd))?;
    window::ensure_not_own_window(resolved.hwnd)?;
    let element = window_element(uia, &resolved)?;
    let tree = build_tree(uia, &element, resolved.surface)?;
    let node = core::resolve_locator_checked(&tree.root, locator)?;
    let index = core::pre_order_index(&tree.root, node);
    let element = tree.handles.get(index).ok_or_else(|| {
        core::taxonomy_error(
            core::ERR_STALE_ELEMENT,
            "element is no longer in the window's tree — re-observe and use a fresh ref",
        )
    })?;
    match act {
        ElementAct::Press => press(element, resolved.hwnd),
        ElementAct::SetValue { text } => set_value(element, resolved.hwnd, text),
    }
}

fn window_element(
    uia: &IUIAutomation,
    resolved: &window::TargetWindow,
) -> anyhow::Result<IUIAutomationElement> {
    // SAFETY: `resolved.hwnd` was resolved by this job and is not owned here, and `uia`
    // is the client built on this apartment thread; the element comes back owned.
    unsafe { uia.ElementFromHandle(resolved.hwnd) }.map_err(|e| {
        map_uia_error(
            &format!("open the accessibility tree of \"{}\"", resolved.title),
            &e,
        )
    })
}

struct Tree {
    root: UiNode,
    /// One element per node, in the same pre-order as `root` — the mapping
    /// [`core::pre_order_index`] lands in.
    handles: Vec<IUIAutomationElement>,
}

struct Walk {
    walker: IUIAutomationTreeWalker,
    origin: (f64, f64),
    deadline: Instant,
    nodes: usize,
    handles: Vec<IUIAutomationElement>,
}

fn build_tree(
    uia: &IUIAutomation,
    root: &IUIAutomationElement,
    surface: SurfaceGeometry,
) -> anyhow::Result<Tree> {
    // SAFETY: `uia` is the live client owned by this thread's `Ctx`; the walker it
    // yields is an owned interface, held by the `Walk` below.
    let walker = unsafe { uia.ControlViewWalker() }
        .map_err(|e| map_uia_error("open the accessibility tree walker", &e))?;
    let mut walk = Walk {
        walker,
        origin: (surface.x, surface.y),
        deadline: Instant::now() + WALK_DEADLINE,
        nodes: 0,
        handles: Vec::new(),
    };
    let root = walk.node(root, 0)?;
    Ok(Tree {
        root,
        handles: walk.handles,
    })
}

impl Walk {
    fn node(&mut self, element: &IUIAutomationElement, depth: usize) -> anyhow::Result<UiNode> {
        if Instant::now() > self.deadline {
            return Err(core::taxonomy_error(
                core::ERR_DEGRADED,
                format!(
                    "the application did not finish exposing its accessibility tree within {}s — \
                     it may be busy or hung; retry, or use screenshot",
                    WALK_DEADLINE.as_secs()
                ),
            ));
        }
        self.nodes += 1;
        self.handles.push(element.clone());
        let mut node = read_node(element, self.origin);
        if depth >= MAX_DEPTH {
            return Ok(node);
        }
        // SAFETY: `element` is a live element of this walk and `self.walker` is the
        // walker of the same client; the child comes back owned, and a provider that
        // refuses it just ends this level.
        let mut child = unsafe { self.walker.GetFirstChildElement(element) }.ok();
        while let Some(element) = child {
            if self.nodes >= MAX_NODES {
                break;
            }
            node.children.push(self.node(&element, depth + 1)?);
            // SAFETY: as above — `element` is the child the previous call returned and
            // is still alive here; the sibling comes back owned.
            child = unsafe { self.walker.GetNextSiblingElement(&element) }.ok();
        }
        Ok(node)
    }
}

fn read_node(element: &IUIAutomationElement, origin: (f64, f64)) -> UiNode {
    // SAFETY: `element` is a live element handle; the control type is returned by value.
    let control_type = unsafe { element.CurrentControlType() }.map_or(0, |id| id.0);
    let (role, actions) = role_and_actions(control_type);
    // SAFETY: `element` is live; the name comes back as an owned `BSTR`.
    let name = unsafe { element.CurrentName() }
        .ok()
        .map(|name| name.to_string())
        .filter(|name| !name.is_empty())
        .map(|name| crate::util::truncate(&name, MAX_NAME_CHARS));
    // SAFETY: `element` is live; the rectangle is returned by value.
    let frame = unsafe { element.CurrentBoundingRectangle() }
        .ok()
        .and_then(|rect| surface_frame(rect, origin));
    let focused =
        // SAFETY: `element` is live and the focus flag comes back by value.
        unsafe { element.CurrentHasKeyboardFocus() }.is_ok_and(windows::core::BOOL::as_bool);
    let value = if VALUE_ROLES.contains(&role) {
        read_value(element)
    } else {
        None
    };
    UiNode {
        role: role.to_string(),
        name,
        value,
        frame,
        actions: actions.iter().map(|action| (*action).to_string()).collect(),
        focused,
        children: Vec::new(),
    }
}

fn read_value(element: &IUIAutomationElement) -> Option<String> {
    // SAFETY: `element` is live; the password flag is returned by value.
    if unsafe { element.CurrentIsPassword() }.is_ok_and(windows::core::BOOL::as_bool) {
        return None;
    }
    let pattern =
        // SAFETY: `element` is live and the pattern id is a constant; the pattern comes
        // back as an owned `IUIAutomationValuePattern` that outlives the read below.
        unsafe { element.GetCurrentPatternAs::<IUIAutomationValuePattern>(UIA_ValuePatternId) }
            .ok()?;
    // SAFETY: `pattern` is that owned `IUIAutomationValuePattern`; the value is an
    // owned `BSTR`.
    let value = unsafe { pattern.CurrentValue() }.ok()?.to_string();
    if value.is_empty() {
        None
    } else {
        Some(crate::util::truncate(&value, MAX_VALUE_CHARS))
    }
}

/// A global physical rect as the surface-local `(x, y, w, h)` a [`UiNode`] carries.
fn surface_frame(rect: RECT, origin: (f64, f64)) -> Option<(f64, f64, f64, f64)> {
    let width = f64::from(rect.right - rect.left);
    let height = f64::from(rect.bottom - rect.top);
    if width <= 0.0 || height <= 0.0 {
        return None;
    }
    Some((
        f64::from(rect.left) - origin.0,
        f64::from(rect.top) - origin.1,
        width,
        height,
    ))
}

/// The role name and action hints for a UIA control type. Names that the shared
/// interactive-role list knows keep its spelling (`edit` → `textfield`,
/// `hyperlink` → `link`, `tabitem` → `tab`).
fn role_and_actions(control_type: i32) -> (&'static str, &'static [&'static str]) {
    const PRESS: &[&str] = &["press"];
    const SELECT: &[&str] = &["select"];
    const VALUE: &[&str] = &["set-value"];
    match control_type {
        x if x == UIA_ButtonControlTypeId.0 => ("button", PRESS),
        x if x == UIA_SplitButtonControlTypeId.0 => ("button", &["press", "expand"]),
        x if x == UIA_CheckBoxControlTypeId.0 => ("checkbox", &["toggle"]),
        x if x == UIA_RadioButtonControlTypeId.0 => ("radiobutton", SELECT),
        x if x == UIA_ComboBoxControlTypeId.0 => ("combobox", &["expand", "set-value"]),
        x if x == UIA_EditControlTypeId.0 => ("textfield", VALUE),
        x if x == UIA_HyperlinkControlTypeId.0 => ("link", PRESS),
        x if x == UIA_MenuItemControlTypeId.0 => ("menuitem", PRESS),
        x if x == UIA_MenuControlTypeId.0 => ("menu", &[]),
        x if x == UIA_MenuBarControlTypeId.0 => ("menubar", &[]),
        x if x == UIA_ListItemControlTypeId.0 => ("listitem", SELECT),
        x if x == UIA_DataItemControlTypeId.0 => ("listitem", SELECT),
        x if x == UIA_ListControlTypeId.0 => ("list", &[]),
        x if x == UIA_TabItemControlTypeId.0 => ("tab", SELECT),
        x if x == UIA_TabControlTypeId.0 => ("tabgroup", &[]),
        x if x == UIA_TreeItemControlTypeId.0 => ("treeitem", &["select", "expand"]),
        x if x == UIA_TreeControlTypeId.0 => ("tree", &[]),
        x if x == UIA_TableControlTypeId.0 => ("table", &[]),
        x if x == UIA_DataGridControlTypeId.0 => ("table", &[]),
        x if x == UIA_HeaderControlTypeId.0 => ("header", &[]),
        x if x == UIA_HeaderItemControlTypeId.0 => ("headeritem", SELECT),
        x if x == UIA_SliderControlTypeId.0 => ("slider", VALUE),
        x if x == UIA_SpinnerControlTypeId.0 => ("spinner", VALUE),
        x if x == UIA_ScrollBarControlTypeId.0 => ("scrollbar", &[]),
        x if x == UIA_ProgressBarControlTypeId.0 => ("progressbar", &[]),
        x if x == UIA_StatusBarControlTypeId.0 => ("statusbar", &[]),
        x if x == UIA_ToolBarControlTypeId.0 => ("toolbar", &[]),
        x if x == UIA_ToolTipControlTypeId.0 => ("tooltip", &[]),
        x if x == UIA_TitleBarControlTypeId.0 => ("titlebar", &[]),
        x if x == UIA_SeparatorControlTypeId.0 => ("separator", &[]),
        x if x == UIA_ThumbControlTypeId.0 => ("thumb", &[]),
        x if x == UIA_ImageControlTypeId.0 => ("image", &[]),
        x if x == UIA_TextControlTypeId.0 => ("text", &[]),
        // A document is typed into like a text field (Windows 11 Notepad's editor is
        // one), so it carries the action hint that gets it a ref.
        x if x == UIA_DocumentControlTypeId.0 => ("document", VALUE),
        x if x == UIA_GroupControlTypeId.0 => ("group", &[]),
        x if x == UIA_PaneControlTypeId.0 => ("pane", &[]),
        x if x == UIA_WindowControlTypeId.0 => ("window", &[]),
        x if x == UIA_CalendarControlTypeId.0 => ("calendar", &[]),
        x if x == UIA_CustomControlTypeId.0 => ("custom", &[]),
        x if x == UIA_AppBarControlTypeId.0 => ("toolbar", &[]),
        x if x == UIA_SemanticZoomControlTypeId.0 => ("group", &[]),
        _ => ("unknown", &[]),
    }
}

// ── Acting ──────────────────────────────────────────────────────────────

/// Press an element the way a user would: the first accessibility action it actually
/// supports, in the order a screen reader tries them, and — only when it supports
/// none at all — a click on its own point.
fn press(element: &IUIAutomationElement, hwnd: HWND) -> anyhow::Result<()> {
    // SAFETY: every `unsafe` block below is one method on the pattern that `attempt`
    // obtained from this element — each is owned here, alive for its call, and runs on
    // the thread that owns it.
    let pressed = attempt(
        element,
        UIA_InvokePatternId,
        |p: &IUIAutomationInvokePattern| unsafe { p.Invoke() },
    )
    .or_else(|| {
        attempt(
            element,
            UIA_TogglePatternId,
            |p: &IUIAutomationTogglePattern| unsafe { p.Toggle() },
        )
    })
    .or_else(|| {
        attempt(
            element,
            UIA_SelectionItemPatternId,
            |p: &IUIAutomationSelectionItemPattern| unsafe { p.Select() },
        )
    })
    .or_else(|| {
        attempt(
            element,
            UIA_ExpandCollapsePatternId,
            |p: &IUIAutomationExpandCollapsePattern| unsafe {
                // Expanded goes to collapsed and back; anything else opens.
                if p.CurrentExpandCollapseState()
                    .is_ok_and(|state| state == ExpandCollapseState_Expanded)
                {
                    p.Collapse()
                } else {
                    p.Expand()
                }
            },
        )
    })
    .or_else(|| {
        attempt(
            element,
            UIA_LegacyIAccessiblePatternId,
            |p: &IUIAutomationLegacyIAccessiblePattern| unsafe { p.DoDefaultAction() },
        )
    });
    if let Some(result) = pressed {
        return result;
    }
    let point = clickable_point(element).ok_or_else(|| {
        core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            "element supports no press action and has no visible point — re-observe, or click it by \
             coordinates",
        )
    })?;
    // The click lands on whatever is at that point, so the window must be in front.
    window::activate(hwnd);
    window::ensure_foreground(hwnd)?;
    input::click_absolute(point.x, point.y)
}

/// One pattern's action, when the element supports the pattern.
///
/// `None` means "try the next one": the pattern is absent, or the element reports
/// the action itself as unsupported. A pattern that *fails* — a disabled control, a
/// provider that ran out of time — is the element's own answer and comes back as
/// `Some(Err)`, so it is reported instead of being replaced by a coordinate click
/// that would claim a click the element never received.
fn attempt<T: windows::core::Interface>(
    element: &IUIAutomationElement,
    id: UIA_PATTERN_ID,
    action: impl FnOnce(&T) -> windows::core::Result<()>,
) -> Option<anyhow::Result<()>> {
    // SAFETY: `GetCurrentPatternAs` yields an owned interface for this element; the
    // pattern object is released when it goes out of scope here.
    let pattern = unsafe { element.GetCurrentPatternAs::<T>(id) }.ok()?;
    match action(&pattern) {
        Ok(()) => Some(Ok(())),
        Err(error) if is_unsupported(&error) => None,
        Err(error) => Some(Err(map_uia_error("press the element", &error))),
    }
}

/// Whether a UIA call failed because the element does not support that action at
/// all, rather than because it refused it.
fn is_unsupported(error: &windows::core::Error) -> bool {
    let code = error.code().0.cast_unsigned();
    code == UIA_E_NOTSUPPORTED || code == E_NOTIMPL.0.cast_unsigned()
}

/// Whether the element is a control a person would type into: an edit field, a
/// numeric spinner, or a document surface (Windows 11 Notepad exposes its editor as
/// one). A combo box is deliberately absent — a dropdown-only one shows no text
/// field, and typing into it would silently do nothing.
fn is_editable(element: &IUIAutomationElement) -> bool {
    // SAFETY: `element` is live; the control type is returned by value.
    let control_type = unsafe { element.CurrentControlType() }.map_or(0, |id| id.0);
    [
        UIA_EditControlTypeId.0,
        UIA_SpinnerControlTypeId.0,
        UIA_DocumentControlTypeId.0,
    ]
    .contains(&control_type)
}

fn clickable_point(element: &IUIAutomationElement) -> Option<POINT> {
    let mut point = POINT::default();
    // SAFETY: `point` is a live `POINT` local that the call fills in place, and the
    // related success flag comes back by value.
    let reported = unsafe { element.GetClickablePoint(&raw mut point) };
    if reported.is_ok_and(windows::core::BOOL::as_bool) {
        return Some(point);
    }
    // SAFETY: `element` is live; the rectangle is returned by value.
    let rect = unsafe { element.CurrentBoundingRectangle() }.ok()?;
    if rect.right <= rect.left || rect.bottom <= rect.top {
        return None;
    }
    Some(POINT {
        x: rect.left + (rect.right - rect.left) / 2,
        y: rect.top + (rect.bottom - rect.top) / 2,
    })
}

/// Set an element's value: the value pattern when it is writable, otherwise — for
/// an EDITABLE control only — focus the element and type over its selected
/// contents, which is what a user does. Anything else is refused, like the other
/// backends: typing into an arbitrary control is destructive, not helpful.
fn set_value(element: &IUIAutomationElement, hwnd: HWND, text: &str) -> anyhow::Result<()> {
    // SAFETY: `element` is live; the password flag is returned by value.
    if unsafe { element.CurrentIsPassword() }.is_ok_and(windows::core::BOOL::as_bool) {
        return Err(core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            "element is a password field — its value cannot be set through the accessibility channel",
        ));
    }
    // SAFETY: `element` is live on this apartment thread; the pattern, its read-only
    // flag and the `BSTR` handed to `SetValue` are locals owned by this block, and the
    // string outlives the call that reads it.
    unsafe {
        if let Ok(pattern) =
            element.GetCurrentPatternAs::<IUIAutomationValuePattern>(UIA_ValuePatternId)
        {
            let read_only = pattern
                .CurrentIsReadOnly()
                .is_ok_and(windows::core::BOOL::as_bool);
            if !read_only {
                match pattern.SetValue(&BSTR::from(text)) {
                    Ok(()) => return Ok(()),
                    // The pattern cannot serve a value here: fall through to typing.
                    Err(e) if is_unsupported(&e) => {}
                    Err(e) => return Err(map_uia_error("set the element's value", &e)),
                }
            }
        }
    }
    if !is_editable(element) {
        return Err(core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            "element has no writable value and is not an editable field — use click by \
             coordinates and raw typing instead",
        ));
    }
    // Typing goes to the foreground window, so focus follows the window first.
    window::activate(hwnd);
    window::ensure_foreground(hwnd)?;
    // SAFETY: `element` is live, and focusing it takes no pointer of ours; it runs on
    // the thread that owns the element.
    unsafe { element.SetFocus() }.map_err(|e| map_uia_error("focus the element", &e))?;
    input::select_all()?;
    input::type_text_into_focus(text)
}

// ── Errors ──────────────────────────────────────────────────────────────

/// Map a UIA failure onto the shared taxonomy. The provider's own codes carry the
/// distinction the model needs: an element that is gone, one that is disabled and a
/// provider that ran out of time each call for a different next step.
fn map_uia_error(action: &str, error: &windows::core::Error) -> anyhow::Error {
    // Access denied is the typed permission failure, so a caller that must branch
    // on it (see `core::is_permission_denied`) does not read the text.
    if error.code() == E_ACCESSDENIED {
        return core::permission_denied(format!("failed to {action}: {error}"));
    }
    match error.code().0.cast_unsigned() {
        // The element is gone: its ref no longer resolves, so re-observing is the way
        // back — the `stale-element` guidance.
        UIA_E_ELEMENTNOTAVAILABLE => core::taxonomy_error(
            core::ERR_STALE_ELEMENT,
            format!(
                "failed to {action}: the element is no longer there — re-observe and act on the \
                 element the tree shows now ({error})"
            ),
        ),
        // The element is there but disabled: the action is not available for this
        // target, so another element or a coordinate click is the way round it — the
        // `unsupported` guidance, not a retry.
        UIA_E_ELEMENTNOTENABLED => core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            format!(
                "failed to {action}: the control is disabled, so the action is not available for \
                 it until its state changes — act on an enabled element, or use coordinates \
                 ({error})"
            ),
        ),
        UIA_E_NOTSUPPORTED => core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            format!("failed to {action}: the control does not support it ({error})"),
        ),
        _ => core::taxonomy_error(core::ERR_DEGRADED, format!("failed to {action}: {error}")),
    }
}
