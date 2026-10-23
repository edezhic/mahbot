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
//! and is recorded in the observation's truncation, which the header names; a name or
//! a value is kept to `core::MAX_NAME_CHARS` / `core::MAX_VALUE_CHARS` where the tree
//! takes it, the cut stated in that same header, because a document-shaped field
//! reports a whole file as its value. A provider that runs out of TIME is different
//! and abandons the whole read with an honest `degraded` timeout rather than returning
//! half of a tree. A provider that refuses a single child or property only ends that
//! subtree or leaves one field absent, the way a broken accessibility attribute does
//! on the other platforms.

use super::super::core::{
    self, ElementAct, Impact, Locator, Observation, SurfaceGeometry, TargetSpec, Truncation, UiNode,
};
use super::{input, window};
use std::time::Duration;
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
    UIA_E_ELEMENTNOTAVAILABLE, UIA_E_ELEMENTNOTENABLED, UIA_E_NOTSUPPORTED, UIA_E_TIMEOUT,
    UIA_EditControlTypeId, UIA_ExpandCollapsePatternId, UIA_GroupControlTypeId,
    UIA_HeaderControlTypeId, UIA_HeaderItemControlTypeId, UIA_HyperlinkControlTypeId,
    UIA_ImageControlTypeId, UIA_InvokePatternId, UIA_LegacyIAccessiblePatternId,
    UIA_ListControlTypeId, UIA_ListItemControlTypeId, UIA_MenuBarControlTypeId,
    UIA_MenuControlTypeId, UIA_MenuItemControlTypeId, UIA_PATTERN_ID, UIA_PaneControlTypeId,
    UIA_ProgressBarControlTypeId, UIA_RadioButtonControlTypeId, UIA_ScrollBarControlTypeId,
    UIA_SelectionItemPatternId, UIA_SemanticZoomControlTypeId, UIA_SeparatorControlTypeId,
    UIA_SliderControlTypeId, UIA_SpinnerControlTypeId, UIA_SplitButtonControlTypeId,
    UIA_StatusBarControlTypeId, UIA_TabControlTypeId, UIA_TabItemControlTypeId,
    UIA_TableControlTypeId, UIA_TextControlTypeId, UIA_ThumbControlTypeId,
    UIA_TitleBarControlTypeId, UIA_TogglePatternId, UIA_ToolBarControlTypeId,
    UIA_ToolTipControlTypeId, UIA_TreeControlTypeId, UIA_TreeItemControlTypeId, UIA_ValuePatternId,
    UIA_WindowControlTypeId,
};
use windows::core::BSTR;

/// How long one walk may spend before it gives up — inside the caller's
/// [`super::OBSERVE_TIMEOUT`], so a stuck provider still ends as a clean timeout. A UIA
/// read is a cross-process property call under the apartment thread's transaction
/// timeout, which is this same figure: one stalled read spends the whole walk, and the
/// check before the next read is what stops it there instead of paying another.
pub(super) const WALK_DEADLINE: Duration = Duration::from_secs(10);

/// Nodes per walk. UIA pays one cross-process read per property, so this is lower
/// than the renderer's `core::MAX_TREE_NODES`; reaching it stops the walk and is
/// recorded as the walk's own ceiling.
const MAX_NODES: usize = 2000;
/// Depth per walk — a guard against provider cycles, like the other backends'.
const MAX_DEPTH: usize = 25;

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
    // The title is read before the walk: a walk that spends its budget has to name
    // the window it abandoned, and this read is one it can no longer afford then.
    let title = window::title_text(resolved.hwnd);
    let element = window_element(uia, &resolved)?;
    let budget = core::Budget::new(
        WALK_DEADLINE,
        "tree walk",
        core::window_description(&resolved.app_name, &title),
    );
    let tree = build_tree(uia, &element, resolved.surface, &budget)?;
    let mut truncation = tree.truncation;
    // The header keeps the whole title to the same cap as a name and, unlike a
    // listing's label, states the cut.
    let window_title = Some(truncation.keep_name(&title));
    Ok(Observation {
        app_name: resolved.app_name,
        window_title,
        surface: resolved.surface,
        root: tree.root,
        truncation,
    })
}

pub(super) fn act_on_element(
    uia: &IUIAutomation,
    target: &TargetSpec,
    locator: &Locator,
    act: &ElementAct,
) -> anyhow::Result<Impact> {
    let resolved = window::resolve(target)?;
    window::ensure_actuation_allowed(Some(resolved.hwnd))?;
    window::ensure_not_own_window(resolved.hwnd)?;
    let element = window_element(uia, &resolved)?;
    // Resolution already read the title, so the walk's own bound is named from it
    // rather than paying a second cross-process read.
    let budget = core::Budget::new(
        WALK_DEADLINE,
        "tree walk",
        core::window_description(&resolved.app_name, &resolved.title),
    );
    let tree = build_tree(uia, &element, resolved.surface, &budget)?;
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
    /// Where this walk stopped short of the whole tree.
    truncation: Truncation,
}

struct Walk<'a> {
    walker: IUIAutomationTreeWalker,
    origin: (f64, f64),
    /// The walk's own bound, checked before every provider read it makes.
    budget: &'a core::Budget,
    nodes: usize,
    handles: Vec<IUIAutomationElement>,
    truncation: Truncation,
}

fn build_tree(
    uia: &IUIAutomation,
    root: &IUIAutomationElement,
    surface: SurfaceGeometry,
    budget: &core::Budget,
) -> anyhow::Result<Tree> {
    // SAFETY: `uia` is the live client owned by this thread's `Ctx`; the walker it
    // yields is an owned interface, held by the `Walk` below.
    let walker = unsafe { uia.ControlViewWalker() }
        .map_err(|e| map_uia_error("open the accessibility tree walker", &e))?;
    let mut walk = Walk {
        walker,
        origin: (surface.x, surface.y),
        budget,
        nodes: 0,
        handles: Vec::new(),
        truncation: Truncation::default(),
    };
    let root = walk.node(root, 0)?;
    Ok(Tree {
        root,
        handles: walk.handles,
        truncation: walk.truncation,
    })
}

impl Walk<'_> {
    fn node(&mut self, element: &IUIAutomationElement, depth: usize) -> anyhow::Result<UiNode> {
        self.budget.check()?;
        self.nodes += 1;
        self.handles.push(element.clone());
        let mut node = read_node(element, self.origin, &mut self.truncation, self.budget)?;
        // The child probe below is another provider call, and so is every sibling read
        // after it: none may start once the walk has spent its budget.
        self.budget.check()?;
        if depth >= MAX_DEPTH {
            // The node's subtree is not read. Only a node that has one is a cut, so a
            // leaf at the ceiling is not reported as a truncation.
            // SAFETY: `element` is a live element of this walk and `self.walker` is the
            // walker of the same client; the probe discards the child it returns.
            if unsafe { self.walker.GetFirstChildElement(element) }.is_ok() {
                self.truncation.cut_depth(MAX_DEPTH);
            }
            return Ok(node);
        }
        // SAFETY: `element` is a live element of this walk and `self.walker` is the
        // walker of the same client; the child comes back owned, and a provider that
        // refuses it just ends this level.
        let mut child = unsafe { self.walker.GetFirstChildElement(element) }.ok();
        while let Some(element) = child {
            if self.nodes >= MAX_NODES {
                self.truncation.hit_node_ceiling(MAX_NODES);
                break;
            }
            self.budget.check()?;
            node.children.push(self.node(&element, depth + 1)?);
            // SAFETY: as above — `element` is the child the previous call returned and
            // is still alive here; the sibling comes back owned.
            child = unsafe { self.walker.GetNextSiblingElement(&element) }.ok();
        }
        Ok(node)
    }
}

/// One node's own fields. Each provider read below is preceded by the budget check,
/// so a target that stalls mid-node costs at most one transaction past the walk's
/// bound instead of one per field.
fn read_node(
    element: &IUIAutomationElement,
    origin: (f64, f64),
    truncation: &mut Truncation,
    budget: &core::Budget,
) -> anyhow::Result<UiNode> {
    budget.check()?;
    // SAFETY: `element` is a live element handle; the control type is returned by value.
    let control_type = unsafe { element.CurrentControlType() }.map_or(0, |id| id.0);
    let (role, actions) = role_and_actions(control_type);
    budget.check()?;
    // SAFETY: `element` is live; the name comes back as an owned `BSTR`.
    let name = unsafe { element.CurrentName() }
        .ok()
        .map(|name| name.to_string())
        .filter(|name| !name.is_empty())
        .map(|name| truncation.keep_name(&name));
    budget.check()?;
    // SAFETY: `element` is live; the rectangle is returned by value.
    let frame = unsafe { element.CurrentBoundingRectangle() }
        .ok()
        .and_then(|rect| surface_frame(rect, origin));
    budget.check()?;
    let focused =
        // SAFETY: `element` is live and the focus flag comes back by value.
        unsafe { element.CurrentHasKeyboardFocus() }.is_ok_and(windows::core::BOOL::as_bool);
    let value = if VALUE_ROLES.contains(&role) {
        read_value(element, budget)?.map(|value| truncation.keep_value(&value))
    } else {
        None
    };
    Ok(UiNode {
        role: role.to_string(),
        name,
        value,
        frame,
        actions: actions.iter().map(|action| (*action).to_string()).collect(),
        focused,
        children: Vec::new(),
    })
}

/// One element's value, whole, as the marshaller handed it over — the caller keeps it
/// through [`Truncation::keep_value`], where the read's cap applies and the cut is
/// stated in the observation's header. `Err` is the walk's budget spent; an absent
/// pattern or an unreadable value is `Ok(None)`.
fn read_value(
    element: &IUIAutomationElement,
    budget: &core::Budget,
) -> anyhow::Result<Option<String>> {
    budget.check()?;
    // SAFETY: `element` is live; the password flag is returned by value.
    if unsafe { element.CurrentIsPassword() }.is_ok_and(windows::core::BOOL::as_bool) {
        return Ok(None);
    }
    budget.check()?;
    // SAFETY: `element` is live and the pattern id is a constant; the pattern comes
    // back as an owned `IUIAutomationValuePattern` that outlives the read below.
    let Ok(pattern) =
        (unsafe { element.GetCurrentPatternAs::<IUIAutomationValuePattern>(UIA_ValuePatternId) })
    else {
        return Ok(None);
    };
    budget.check()?;
    // SAFETY: `pattern` is that owned `IUIAutomationValuePattern`; the value is an
    // owned `BSTR`.
    let Ok(value) = (unsafe { pattern.CurrentValue() }) else {
        return Ok(None);
    };
    let value = value.to_string();
    Ok(if value.is_empty() { None } else { Some(value) })
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
fn press(element: &IUIAutomationElement, hwnd: HWND) -> anyhow::Result<Impact> {
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
/// `Some`, so it is reported instead of being replaced by a coordinate click that
/// would claim a click the element never received.
fn attempt<T: windows::core::Interface>(
    element: &IUIAutomationElement,
    id: UIA_PATTERN_ID,
    action: impl FnOnce(&T) -> windows::core::Result<()>,
) -> Option<anyhow::Result<Impact>> {
    // SAFETY: `GetCurrentPatternAs` yields an owned interface for this element; the
    // pattern object is released when it goes out of scope here.
    let pattern = unsafe { element.GetCurrentPatternAs::<T>(id) }.ok()?;
    match action(&pattern) {
        Ok(()) => Some(Ok(Impact::Done)),
        Err(error) if is_unsupported(&error) => None,
        Err(error) => Some(map_uia_action_error("press the element", &error)),
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
fn set_value(element: &IUIAutomationElement, hwnd: HWND, text: &str) -> anyhow::Result<Impact> {
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
                    Ok(()) => return Ok(Impact::Done),
                    // The pattern cannot serve a value here: fall through to typing.
                    Err(e) if is_unsupported(&e) => {}
                    Err(e) => return map_uia_action_error("set the element's value", &e),
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
    // A select-all that only partly landed leaves the field's old text in place and
    // a selection nobody can vouch for, so the value is unchanged: that is a failure,
    // not an unconfirmed set, and the text is deliberately not typed over it.
    if let Impact::Unconfirmed(detail) = input::select_all()? {
        return Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            format!(
                "the element's value was not set and no text was typed — selecting its existing \
                 text was only partly delivered ({detail}), so the field's selection is now \
                 unknown"
            ),
        ));
    }
    input::type_text_into_focus(text)
}

// ── Errors ──────────────────────────────────────────────────────────────

/// Map a UIA failure of an ACTUATING call. Everything [`map_uia_error`] classifies
/// means the action was not performed, so it stays an error; a provider that ran out
/// of time is the one case the element may have acted on anyway, and it answers
/// [`Impact::Unconfirmed`] rather than inviting a second press.
fn map_uia_action_error(action: &str, error: &windows::core::Error) -> anyhow::Result<Impact> {
    if error.code().0.cast_unsigned() == UIA_E_TIMEOUT {
        return Ok(Impact::Unconfirmed(format!(
            "the element's provider did not answer {action} within its own timeout, so the call \
             was abandoned while it may still complete — {error}"
        )));
    }
    Err(map_uia_error(action, error))
}

/// Map a UIA failure onto the shared taxonomy. The provider's own codes carry the
/// distinction the model needs: an element that is gone and one that is disabled
/// each call for a different next step; anything else — a provider that ran out of
/// time, an unknown code — is `degraded`.
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
