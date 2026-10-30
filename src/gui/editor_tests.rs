//! Editor tests: text matching, selection, buffer operations and the
//! file-backed editing paths.

use super::*;
use crate::gui::editor_widget::{EditorAction, EditorBuffer};

fn fs_entry(name: &str, full_path: &str, is_dir: bool) -> FsEntry {
    FsEntry {
        name: name.to_string(),
        full_path: full_path.to_string(),
        is_dir,
        error: None,
    }
}

type MatchCase<'a> = (&'a str, &'a str, &'a str, bool, &'a [(usize, usize)]);

#[test]
fn test_compute_text_matches() {
    // (name, text, query, case-sensitive, expected match ranges)
    let cases: &[MatchCase] = &[
        ("empty_query", "hello", "", true, &[]),
        (
            "basic",
            "hello world hello",
            "hello",
            true,
            &[(0, 5), (12, 17)],
        ),
        ("no_match", "hello world", "xyz", true, &[]),
        // Non-overlapping matches.
        ("non_overlapping", "aaaaa", "aa", true, &[(0, 2), (2, 4)]),
        (
            "case_insensitive",
            "Hello World hello",
            "hello",
            false,
            &[(0, 5), (12, 17)],
        ),
        (
            "case_insensitive_no_match",
            "Hello World",
            "xyz",
            false,
            &[],
        ),
        // Single-char queries return empty (2-char min enforcement).
        ("single_char", "hello", "h", true, &[]),
        ("shortest_match", "ab", "ab", true, &[(0, 2)]),
        ("consecutive", "abab", "ab", true, &[(0, 2), (2, 4)]),
    ];
    for &(name, text, query, sensitive, expected) in cases {
        let result = compute_text_matches(text, query, sensitive);
        assert_eq!(result.len(), expected.len(), "case {name}: match count");
        for (i, &(start, end)) in expected.iter().enumerate() {
            assert_eq!(result[i], start..end, "case {name}: match {i}");
        }
    }
}

// ── validate_file_content ─────────────────────────────────────

#[test]
fn test_validate_file_content() {
    let big = || vec![b'a'; usize::try_from(MAX_FILE_SIZE).unwrap() + 1];
    let big_bytes = big();
    let mut big_with_null = big();
    big_with_null.push(0);
    // (input, expected error prefix or None for ok)
    let cases: &[(&[u8], Option<&str>)] = &[
        (b"", None),
        (b"hello world", None),
        ("Привет мир 👋".as_bytes(), None),
        (&big_bytes, Some("File too large")),
        (b"hello\0world", Some("Binary file detected")),
        (&big_with_null, Some("File too large")),
    ];
    for &(bytes, expected) in cases {
        match expected {
            Some(prefix) => {
                let err = validate_file_content(bytes).unwrap_err();
                assert!(err.starts_with(prefix), "unexpected error: {err}");
            }
            None => assert!(validate_file_content(bytes).is_ok()),
        }
    }
}

#[test]
fn test_byte_offset_to_line_byte_col_unicode() {
    let text = "Привет **мир**";
    let (line, byte_col, line_start) = byte_offset_to_line_byte_col(text, 13).unwrap();
    assert_eq!(line, 0);
    assert_eq!(byte_col, 13);
    assert_eq!(line_start, 0);
    // End of match on "мир" — byte offset 21.
    let (_, byte_end_col, line_start) = byte_offset_to_line_byte_col(text, 21).unwrap();
    assert_eq!(byte_end_col, 21 - line_start);
}

#[test]
fn test_build_tab_records_dirty_content() {
    let tabs = |is_dirty: bool| {
        vec![Tab {
            path: "/tmp/foo.md".to_string(),
            file_name: "foo.md".to_string(),
            is_dirty,
            line_ending: LineEnding::Lf,
        }]
    };
    let mut tab_contents = HashMap::new();
    let buffer = EditorBuffer::with_text("unsaved edits", None);
    tab_contents.insert(
        "/tmp/foo.md".to_string(),
        TabData {
            content: buffer,
            undo_stack: RefCell::new(UndoStack::new()),
            find_replace_state: None,
            saved_text_hash: 0,
        },
    );
    let records = build_tab_records(&tabs(true), 0, &tab_contents);
    assert_eq!(records.len(), 1, "dirty tab: record count");
    assert!(records[0].is_dirty, "dirty tab: flag kept");
    assert_eq!(
        records[0].dirty_content.as_deref(),
        Some("unsaved edits"),
        "dirty tab: unsaved content persisted"
    );

    let records = build_tab_records(&tabs(false), 0, &HashMap::new());
    assert!(
        records[0].dirty_content.is_none(),
        "clean tab: no content stored"
    );
}

#[test]
fn test_save_result_ignores_stale_save() {
    let mut state = EditorState::new();
    let path = "/tmp/stale.md".to_string();
    state.tabs.push(Tab {
        path: path.clone(),
        file_name: "stale.md".to_string(),
        is_dirty: true,
        line_ending: LineEnding::Lf,
    });
    state.tab_contents.insert(
        path.clone(),
        TabData {
            content: EditorBuffer::with_text("edited after save started", None),
            undo_stack: RefCell::new(UndoStack::new()),
            find_replace_state: None,
            saved_text_hash: hash_text("on disk"),
        },
    );
    let saved_hash = hash_text("saved snapshot");
    let _ = state.save_result(&path, Ok(()), saved_hash);
    assert!(
        state.tabs[0].is_dirty,
        "stale save must not clear dirty flag"
    );
}

// ── saved-tabs restore failure ─────────────────────────────

#[test]
fn test_failed_saved_tabs_restore_blocks_persistence() {
    let mut state = EditorState::new();
    state.selected_workspace_name = Some("ws".to_string());

    // A failed restore keeps the failure visible — with the persistence block
    // it implies — and the session uninitialized.
    let _ = state.saved_tabs_loaded(Err("db down".to_string()), state.saved_tabs_gen);
    let err = state
        .tabs_error
        .as_deref()
        .expect("a failed restore is recorded");
    assert!(err.starts_with("db down"), "got: {err}");
    assert!(
        err.contains(TABS_SAVE_PAUSED),
        "the shown failure must state the paused saving: {err}"
    );
    assert!(!state.session_initialized);
    assert!(state.try_save_current_tabs().is_none());

    // Opening a file marks the session initialized, but the failed restore
    // still blocks persistence so the persisted tab set is never overwritten.
    state.session_initialized = true;
    assert!(state.try_save_current_tabs().is_none());

    // A successful reload clears the failure so saving is re-enabled.
    let _ = state.saved_tabs_loaded(Ok(Vec::new()), state.saved_tabs_gen);
    assert!(state.tabs_error.is_none());
    assert!(state.session_initialized);
}

// ── byte_offset_to_line_col ────────────────────────────────

#[test]
fn test_byte_offset_to_line_col() {
    // (name, text, byte offset, expected (line, column))
    let cases = [
        ("unicode_multibyte", "Привет мир", 13, (0, 7)),
        ("start", "hello\nworld", 0, (0, 0)),
        ("second_line", "hello\nworld", 6, (1, 0)),
        ("middle_of_line", "hello\nworld", 8, (1, 2)),
        // Beyond the text length clamps to the end.
        ("beyond_end", "hello", 100, (0, 5)),
        ("empty", "", 0, (0, 0)),
        // Empty content with an out-of-bounds offset clamps to the start.
        ("empty_beyond_end", "", 100, (0, 0)),
    ];
    for &(name, text, byte_offset, expected) in &cases {
        assert_eq!(
            byte_offset_to_line_col(text, byte_offset),
            expected,
            "case {name}: (line, column)"
        );
    }
}

/// Fluent builder for an [`EditorState`] with tab fixtures.
struct EditorFixture(EditorState);

impl EditorFixture {
    fn new() -> Self {
        Self(EditorState::new())
    }

    /// Adds a tab holding `text` (starting saved) and makes it active.
    fn tab(mut self, path: &str, text: &str) -> Self {
        self.0.tabs.push(Tab {
            path: path.to_string(),
            file_name: path.rsplit('/').next().unwrap_or(path).to_string(),
            is_dirty: false,
            line_ending: LineEnding::Lf,
        });
        self.0.tab_contents.insert(
            path.to_string(),
            TabData {
                content: EditorBuffer::with_text(text, None),
                undo_stack: RefCell::new(UndoStack::new()),
                find_replace_state: None,
                saved_text_hash: hash_text(text),
            },
        );
        self.0.active_tab_index = self.0.tabs.len() - 1;
        self
    }

    /// `count` empty unsaved tabs under `/tmp` (tab-switch fixture).
    fn tabs(self, count: usize, active: usize) -> Self {
        assert!(active < count, "active out of range");
        let mut f = (0..count).fold(self, |f, i| {
            f.tab(&format!("/tmp/test_{i}.rs"), "").unsaved()
        });
        f.0.active_tab_index = active;
        f
    }

    /// Marks the active tab's content unsaved.
    fn unsaved(mut self) -> Self {
        self.active_tab_data().saved_text_hash = 0;
        self
    }

    /// Installs a case-sensitive find/replace state on the active tab.
    fn find_replace(mut self, query: &str, matches: Vec<Range<usize>>, match_idx: usize) -> Self {
        self.active_tab_data().find_replace_state = Some(FindReplaceState {
            query: crate::gui::common::SingleLineEditorState::new(query),
            replace: crate::gui::common::SingleLineEditorState::new(""),
            matches,
            current_match_idx: match_idx,
            case_sensitive: true,
        });
        self
    }

    fn active_tab_data(&mut self) -> &mut TabData {
        let path = self.0.tabs[self.0.active_tab_index].path.clone();
        self.0
            .tab_contents
            .get_mut(&path)
            .expect("active tab has TabData")
    }

    fn build(self) -> EditorState {
        self.0
    }
}

// ── Tree keyboard navigation focus state tests ──────────────────

fn make_editor_with_tree() -> EditorState {
    let mut state = EditorState::new();
    state.selected_workspace_path = Some("/tmp".to_string());
    // Root entries, plus "src" children so they show when expanded.
    state.dir_entries.insert(
        String::new(),
        vec![
            fs_entry("src", "src", true),
            fs_entry("Cargo.toml", "Cargo.toml", false),
        ],
    );
    state.dir_entries.insert(
        "src".to_string(),
        vec![fs_entry("main.rs", "src/main.rs", false)],
    );
    state.rebuild_tree();
    state
}

#[test]
fn test_misc_focus_actions() {
    type Setup = fn(&mut EditorState);
    type Check = fn(&EditorState) -> Option<&'static str>;
    // (name, message, setup, check)
    let cases: &[(&str, EditorMessage, Setup, Check)] = &[
        (
            "escape_clears_tree_focus",
            EditorMessage::Escape,
            |s| s.file_tree.tree_focused = true,
            |s| s.file_tree.tree_focused.then_some("tree focus not cleared"),
        ),
        (
            "toggle_dir_sets_tree_focus",
            EditorMessage::ToggleDir("src".to_string()),
            |s| {
                s.selected_file = Some("Cargo.toml".to_string());
            },
            |s| {
                if !s.file_tree.tree_focused {
                    return Some("tree focus not set");
                }
                s.selected_file.is_some().then_some("selection not cleared")
            },
        ),
        (
            "select_file_keeps_tree_focus",
            EditorMessage::SelectFile("src/main.rs".to_string()),
            |s| s.file_tree.tree_focused = true,
            |s| (!s.file_tree.tree_focused).then_some("tree focus not kept"),
        ),
        // A mouse-originated EditorAction (like MoveTo from a click)
        // should transfer focus from the file tree to the editor.
        (
            "editor_action_clears_tree_focus",
            EditorMessage::EditorAction(EditorAction::MoveTo { line: 0, col: 0 }),
            |s| {
                s.file_tree.tree_focused = true;
                s.pending_enter_dir = Some("src".to_string());
                s.active_modal = Some(ModalKind::Rename(RenameTarget {
                    path: "src/main.rs".to_string(),
                    abs_path: String::new(),
                    is_dir: false,
                    ws_root: String::new(),
                    input_text: crate::gui::common::SingleLineEditorState::new("main.rs"),
                    error: None,
                }));
            },
            |s| {
                if s.file_tree.tree_focused {
                    return Some("tree focus not cleared");
                }
                if s.pending_enter_dir.is_some() {
                    return Some("pending enter dir not cleared");
                }
                s.active_modal.is_some().then_some("modal not closed")
            },
        ),
    ];
    for &(name, ref msg, setup, check) in cases {
        let mut state = make_editor_with_tree();
        setup(&mut state);
        let _ = state.update(msg.clone());
        if let Some(what) = check(&state) {
            panic!("case {name}: {what}");
        }
    }
}

#[test]
fn test_async_dir_load_flow() {
    use EditorMessage::{DirExpanded, ToggleDir, TreeNavEnter};
    type Build = fn() -> EditorState;
    type Check = fn(&mut EditorState) -> Option<&'static str>;
    fn check(ok: bool, what: &'static str) -> Option<&'static str> {
        (!ok).then_some(what)
    }
    fn focused_src_state() -> EditorState {
        let mut state = EditorState::new();
        state.selected_workspace_path = Some("/tmp".to_string());
        state
            .dir_entries
            .insert(String::new(), vec![fs_entry("src", "src", true)]);
        state.rebuild_tree();
        state.file_tree.tree_focused = true;
        state
    }
    fn expanded(generation: u64) -> EditorMessage {
        DirExpanded {
            dir_path: "src".to_string(),
            r#gen: generation,
            entries: Ok(vec![fs_entry("main.rs", "src/main.rs", false)]),
            quiet: false,
        }
    }
    // (name, fixture, check)
    let cases: &[(&str, Build, Check)] = &[
        (
            "async_enter_dir_pending_then_advances",
            focused_src_state,
            |s| {
                let _ = s.update(TreeNavEnter);
                let pending = s.pending_enter_dir.as_deref();
                check(pending == Some("src"), "pending enter dir not recorded")?;
                check(
                    s.file_tree.expanded_dirs.contains("src"),
                    "dir not expanded by the toggle",
                )?;
                check(
                    s.file_tree.tree_focus_index == 0,
                    "focus moved before the listing arrived",
                )?;
                let _ = s.update(expanded(s.generation));
                check(s.pending_enter_dir.is_none(), "pending not cleared")?;
                check(s.file_tree.tree_focus_index == 1, "focus not advanced")
            },
        ),
        ("toggle_dir_loads_and_completes", focused_src_state, |s| {
            let _ = s.update(ToggleDir("src".to_string()));
            check(s.loading_dirs.contains("src"), "load not started")?;
            check(
                s.dir_generations.contains_key("src"),
                "generation not registered",
            )?;
            check(s.pending_enter_dir.is_none(), "pending set by a toggle")?;
            check(s.file_tree.tree_focused, "tree not focused")?;
            check(s.file_tree.tree_focus_index == 0, "focus not on src")?;
            let _ = s.update(expanded(s.dir_generations["src"]));
            check(s.dir_entries.contains_key("src"), "listing not cached")?;
            check(s.dir_entries["src"].len() == 1, "unexpected listing length")?;
            check(!s.loading_dirs.contains("src"), "loading not cleared")?;
            check(s.pending_enter_dir.is_none(), "pending left behind")?;
            let nodes = &s.file_tree.visible_tree_nodes;
            check(nodes.len() >= 2, "tree not rebuilt")?;
            check(nodes[0].0.as_str() == "src", "first node is not src")?;
            check(nodes[1].0.as_str() == "src/main.rs", "not src/main.rs")
        }),
        ("toggle_dir_without_workspace", EditorState::new, |s| {
            let tree = &s.file_tree;
            check(!tree.expanded_dirs.contains("src"), "already expanded")?;
            let _ = s.update(ToggleDir("src".to_string()));
            check(
                s.file_tree.expanded_dirs.contains("src"),
                "dir not expanded by the toggle",
            )?;
            check(s.generation == 0, "generation bumped without a workspace")?;
            check(s.loading_dirs.is_empty(), "load started, no workspace")?;
            check(
                s.dir_generations.is_empty(),
                "generation registered without a workspace",
            )
        }),
    ];
    for &(name, build, check) in cases {
        let mut state = build();
        if let Some(what) = check(&mut state) {
            panic!("case {name}: {what}");
        }
    }
}

// ── Find/Replace tests ───────────────────────────────────────────

#[test]
fn test_find_bar_and_match_navigation() {
    type Build = fn() -> EditorState;
    type Check = fn(&mut EditorState) -> Option<&'static str>;
    fn with_find(text: &str, query: &str, matches: &[Range<usize>]) -> EditorState {
        let fixture = EditorFixture::new().tab("/test.rs", text).unsaved();
        fixture.find_replace(query, matches.to_vec(), 0).build()
    }
    /// The same tab, opened with no find state at all.
    fn find_closed() -> EditorState {
        EditorFixture::new()
            .tab("/test.rs", "fn hello() {}")
            .build()
    }
    /// Two space matches, the second one current.
    fn two_matches() -> EditorState {
        with_find("a b c", " ", &[1..2, 3..4])
    }
    fn match_idx(state: &EditorState) -> usize {
        let frs = state.tab_contents["/test.rs"].find_replace_state.as_ref();
        frs.unwrap().current_match_idx
    }
    const SOLE_MATCH: Range<usize> = 4..9;
    // (name, fixture, check)
    let cases: &[(&str, Build, Check)] = &[
        (
            "find_bar_open",
            || with_find("fn hello() {}", "hello", &[SOLE_MATCH]),
            |s| (!s.is_find_bar_open()).then_some("find bar not open"),
        ),
        ("find_bar_closed_without_find_state", find_closed, |s| {
            s.is_find_bar_open().then_some("find bar open")
        }),
        ("find_bar_closed_without_tabs", EditorState::new, |s| {
            s.is_find_bar_open().then_some("find bar open")
        }),
        ("navigate_wraps_next", two_matches, |s| {
            for (want, what) in [(1, "Next did not advance"), (0, "Next did not wrap")] {
                let _ = s.navigate_find_match(FindDirection::Next);
                if match_idx(s) != want {
                    return Some(what);
                }
            }
            None
        }),
        ("navigate_wraps_prev", two_matches, |s| {
            for (want, what) in [(1, "Prev did not advance"), (0, "Prev did not wrap")] {
                let _ = s.navigate_find_match(FindDirection::Prev);
                if match_idx(s) != want {
                    return Some(what);
                }
            }
            None
        }),
        (
            "navigate_no_matches",
            || with_find("no matches", "zzz", &[]),
            |s| {
                let _ = s.navigate_find_match(FindDirection::Next);
                let _ = s.navigate_find_match(FindDirection::Prev);
                (match_idx(s) != 0).then_some("index moved without matches")
            },
        ),
        ("navigate_without_find_state", find_closed, |s| {
            let _ = s.navigate_find_match(FindDirection::Next);
            let _ = s.navigate_find_match(FindDirection::Prev);
            let find = &s.tab_contents["/test.rs"].find_replace_state;
            find.is_some().then_some("find state created")
        }),
        (
            "replace_all_preserves_cursor",
            || {
                let mut state = EditorFixture::new().tab("/test.rs", "ab cd ab").build();
                let tab = state.tab_contents.get_mut("/test.rs").unwrap();
                tab.content.move_to(0, 5);
                tab.find_replace_state = Some(FindReplaceState {
                    query: crate::gui::common::SingleLineEditorState::new("ab"),
                    replace: crate::gui::common::SingleLineEditorState::new("xy"),
                    matches: vec![0..2, 6..8],
                    current_match_idx: 0,
                    case_sensitive: true,
                });
                state
            },
            |s| {
                let _ = s.update(EditorMessage::FindReplaceAll);
                let cursor = s.tab_contents["/test.rs"].content.cursor();
                if cursor.line != 0 {
                    return Some("cursor line moved");
                }
                (cursor.column != 5).then_some("cursor column moved")
            },
        ),
    ];
    for &(name, build, check) in cases {
        let mut state = build();
        if let Some(what) = check(&mut state) {
            panic!("case {name}: {what}");
        }
    }
}

type ReplaceCase<'a> = (
    &'a str,
    &'a str,
    &'a str,
    &'a str,
    &'a str,
    &'a [(usize, usize)],
);

#[test]
fn test_find_replace_auto_advance() {
    // (name, text, query, replace, expected text, expected matches)
    let cases: &[ReplaceCase] = &[
        // Same-length replacement: remaining "ab" at 6..8 is found by advancing
        // past replace_end (= 0 + 2 = 2).
        ("same_length", "ab cd ab", "ab", "xy", "xy cd ab", &[(6, 8)]),
        // replace_end = 0 + 1 = 1. Remaining "aaa" at byte 6 in the new text.
        (
            "shorter_replacement",
            "aaa bbb aaa",
            "aaa",
            "a",
            "a bbb aaa",
            &[(6, 9)],
        ),
        // Using range.end (= 2) as the advance point would skip the remaining
        // match at 1..3, replace_end (= 1) finds it.
        ("adjacent_matches", "aaaa", "aa", "x", "xaa", &[(1, 3)]),
        // Longer replacement: replace_end = 3, so the remaining "ab" wraps to 0.
        ("longer_replacement", "ab", "ab", "abc", "abc", &[(0, 2)]),
        // No remaining matches: matches reset to empty, cursor at replace_end.
        ("no_more_matches", "ab", "ab", "xy", "xy", &[]),
    ];
    let path = "/test.rs";
    for &(name, text, query, replace, expected_text, expected_matches) in cases {
        let mut state = EditorFixture::new().tab(path, text).build();
        let tab = state.tab_contents.get_mut(path).unwrap();
        // The state before the replace holds every match the query has.
        tab.find_replace_state = Some(FindReplaceState {
            query: crate::gui::common::SingleLineEditorState::new(query),
            replace: crate::gui::common::SingleLineEditorState::new(replace),
            matches: compute_text_matches(text, query, true),
            current_match_idx: 0,
            case_sensitive: true,
        });
        let _ = state.update(EditorMessage::FindReplace);
        let tab = state.tab_contents.get(path).unwrap();
        let frs = tab.find_replace_state.as_ref().unwrap();
        assert_eq!(tab.content.text(), expected_text, "case {name}: text");
        assert_eq!(
            frs.matches.len(),
            expected_matches.len(),
            "case {name}: match count"
        );
        for (i, &(s, e)) in expected_matches.iter().enumerate() {
            assert_eq!(frs.matches[i], s..e, "case {name}: match {i} range");
        }
        // The cursor lands on the next match, or right after the replacement
        // when none is left.
        let (line, col) = expected_matches
            .first()
            .map_or((0, replace.chars().count()), |&(start, _)| (0, start));
        let cursor = tab.content.cursor();
        assert_eq!(cursor.line, line, "case {name}: cursor line");
        assert_eq!(cursor.column, col, "case {name}: cursor col");
        assert_eq!(frs.current_match_idx, 0, "case {name}: current_match_idx");
    }
}

// ── Modal / overlay gating ───────────────────────────────────────

#[expect(clippy::too_many_lines)]
#[test]
fn test_overlay_gates_editor_shortcuts() {
    fn quick_open() -> ModalKind {
        ModalKind::QuickOpen(QuickOpenState {
            filter: crate::gui::common::SingleLineEditorState::new(""),
            selected_index: 0,
            results: Vec::new(),
        })
    }

    fn tab_state() -> EditorState {
        EditorFixture::new().tab("/test.rs", "hello").build()
    }

    fn undoable() -> EditorState {
        let mut state = tab_state();
        let tab = state.tab_contents.get_mut("/test.rs").unwrap();
        tab.undo_stack.borrow_mut().snap_before_edit(&tab.content);
        tab.content.perform_action(EditorAction::Insert('!'));
        state
    }

    fn focusable_tree() -> EditorState {
        let mut state = make_editor_with_tree();
        state.file_tree.expanded_dirs.insert("src".to_string());
        state.file_tree.nodes =
            build_hierarchical_tree(&state.dir_entries, &state.file_tree.expanded_dirs, "");
        state.file_tree.rebuild_visible();
        state
    }

    type Build = fn() -> EditorState;
    type Check = fn(&mut EditorState) -> Result<(), String>;
    // (name, fixture, check)
    let cases: &[(&str, Build, Check)] = &[
        ("undo_noop_when_quick_open_active", undoable, |s| {
            s.active_modal = Some(quick_open());
            let _ = s.update(EditorMessage::Undo);
            let text = s.tab_contents["/test.rs"].content.text();
            if text != "!hello" {
                return Err(format!("undo applied under a modal: {text:?}"));
            }
            Ok(())
        }),
        (
            "refresh_file_tree_noop_when_quick_open_active",
            make_editor_with_tree,
            |s| {
                if s.selected_workspace_path.is_none() {
                    return Err("no workspace selected".to_string());
                }
                let gen_count = s.dir_generations.len();
                s.active_modal = Some(quick_open());
                let _ = s.update(EditorMessage::RefreshFileTree);
                if s.dir_generations.len() != gen_count {
                    return Err("directory refreshed under a modal".to_string());
                }
                Ok(())
            },
        ),
        (
            "tree_focus_toggled_noop_during_modal_overlay",
            make_editor_with_tree,
            |s| {
                let _ = s.update(EditorMessage::TreeFocusToggled);
                if !s.file_tree.tree_focused {
                    return Err("first toggle did not focus the tree".to_string());
                }
                s.active_modal = Some(quick_open());
                let _ = s.update(EditorMessage::TreeFocusToggled);
                if !s.file_tree.tree_focused {
                    return Err("toggle applied under a modal".to_string());
                }
                Ok(())
            },
        ),
        (
            "tree_nav_suppressed_during_goto_line_overlay",
            focusable_tree,
            |s| {
                s.file_tree.tree_focused = true;
                s.file_tree.tree_focus_index = 0; // "src"
                s.active_modal = Some(ModalKind::GotoLine);
                let prev_focus = s.file_tree.tree_focus_index;
                for msg in [
                    EditorMessage::TreeNavUp,
                    EditorMessage::TreeNavDown,
                    EditorMessage::TreeNavEnter,
                    EditorMessage::TreeNavLeft,
                    EditorMessage::TreeNavRight,
                ] {
                    let _ = s.update(msg.clone());
                    if s.file_tree.tree_focus_index != prev_focus {
                        return Err(format!("{msg:?} moved the focus"));
                    }
                }
                let _ = s.update(EditorMessage::TreeFocusToggled);
                if !s.file_tree.tree_focused {
                    return Err("TreeFocusToggled cleared the focus".to_string());
                }
                Ok(())
            },
        ),
        ("tree_nav_suppressed_during_rename", focusable_tree, |s| {
            s.selected_workspace_path = Some("/tmp".to_string());
            s.file_tree.tree_focused = true;
            s.file_tree.tree_focus_index = 0; // "src"
            let _ = s.update(EditorMessage::RenameRequested("Cargo.toml".into()));
            let prev_focus = s.file_tree.tree_focus_index;
            for msg in [
                EditorMessage::TreeNavUp,
                EditorMessage::TreeNavDown,
                EditorMessage::TreeNavEnter,
                EditorMessage::TreeNavLeft,
                EditorMessage::TreeNavRight,
                EditorMessage::TreeFocusToggled,
            ] {
                let _ = s.update(msg.clone());
                if s.file_tree.tree_focus_index != prev_focus {
                    return Err(format!("{msg:?} moved the focus"));
                }
            }
            let _ = s.update(EditorMessage::RenameCancel);
            let _ = s.update(EditorMessage::TreeNavDown);
            if s.file_tree.tree_focus_index == prev_focus {
                return Err("navigation did not resume after cancel".to_string());
            }
            Ok(())
        }),
        (
            "quick_open_toggle_blocked_when_goto_line_open",
            tab_state,
            |s| {
                s.active_modal = Some(ModalKind::GotoLine);
                let _ = s.update(EditorMessage::QuickOpenToggle);
                if matches!(s.active_modal, Some(ModalKind::QuickOpen(_))) {
                    return Err("quick open replaced the goto-line overlay".to_string());
                }
                Ok(())
            },
        ),
        (
            "quick_open_toggle_closes_when_already_open",
            tab_state,
            |s| {
                s.active_modal = Some(ModalKind::QuickOpen(QuickOpenState {
                    filter: crate::gui::common::SingleLineEditorState::new("foo"),
                    selected_index: 0,
                    results: Vec::new(),
                }));
                let _ = s.update(EditorMessage::QuickOpenToggle);
                if s.active_modal.is_some() {
                    return Err("quick open not closed".to_string());
                }
                Ok(())
            },
        ),
        (
            "global_search_toggle_blocked_when_quick_open_open",
            tab_state,
            |s| {
                s.selected_workspace_name = Some("ws".to_string());
                s.selected_workspace_path = Some("/tmp/ws".to_string());
                s.active_modal = Some(quick_open());
                let _ = s.update(EditorMessage::GlobalSearchToggle);
                if matches!(s.active_modal, Some(ModalKind::GlobalSearch(_))) {
                    return Err("global search opened under quick open".to_string());
                }
                Ok(())
            },
        ),
    ];
    for &(name, build, check) in cases {
        let mut state = build();
        if let Err(what) = check(&mut state) {
            panic!("case {name}: {what}");
        }
    }
}

// ── Inline rename tests ────────────────────────────────────

#[test]
fn test_rename_request_and_input() {
    fn tree_state() -> EditorState {
        let mut state = make_editor_with_tree();
        state.selected_workspace_path = Some("/tmp".to_string());
        state
    }

    type Build = fn() -> EditorState;
    type Check = fn(&mut EditorState) -> Option<&'static str>;
    // (name, fixture, check)
    let cases: &[(&str, Build, Check)] = &[
        ("rename_request_sets_target", tree_state, |s| {
            let _ = s.update(EditorMessage::RenameRequested("Cargo.toml".into()));
            let Some(ModalKind::Rename(rt)) = &s.active_modal else {
                return Some("expected Rename modal");
            };
            if rt.path != "Cargo.toml" {
                return Some("wrong path");
            }
            if rt.input_text.text() != "Cargo.toml" {
                return Some("wrong prefilled name");
            }
            rt.is_dir.then_some("marked as a directory")
        }),
        (
            "rename_request_on_directory_sets_is_dir",
            EditorState::new,
            |s| {
                // A real temp directory so Path::is_dir() returns true.
                let tmp_dir = tempfile::tempdir().unwrap();
                std::fs::create_dir(tmp_dir.path().join("src")).unwrap();
                s.selected_workspace_path = Some(tmp_dir.path().to_string_lossy().to_string());
                s.dir_entries
                    .insert(String::new(), vec![fs_entry("src", "src", true)]);
                let _ = s.update(EditorMessage::RenameRequested("src".into()));
                let Some(ModalKind::Rename(rt)) = &s.active_modal else {
                    return Some("expected Rename modal");
                };
                if rt.path != "src" {
                    return Some("wrong path");
                }
                if rt.input_text.text() != "src" {
                    return Some("wrong prefilled name");
                }
                (!rt.is_dir).then_some("not marked as a directory")
            },
        ),
        ("rename_request_on_root_dir_rejected", tree_state, |s| {
            let _ = s.update(EditorMessage::RenameRequested(String::new()));
            matches!(s.active_modal, Some(ModalKind::Rename(_)))
                .then_some("root accepted as a rename target")
        }),
        (
            "rename_input_updates_text_and_clears_error",
            tree_state,
            |s| {
                let _ = s.update(EditorMessage::RenameRequested("Cargo.toml".into()));
                if let Some(ModalKind::Rename(rt)) = s.active_modal.as_mut() {
                    rt.error = Some("bad".into());
                    // The buffer pre-fills with the file name; clear it before typing.
                    rt.input_text.set_text("");
                }
                let _ = s.update(EditorMessage::RenameInput(EditorAction::Paste(
                    "new_name".into(),
                )));
                match &s.active_modal {
                    Some(ModalKind::Rename(rt)) => {
                        if rt.input_text.text() != "new_name" {
                            return Some("typed text lost");
                        }
                        rt.error.is_some().then_some("error not cleared")
                    }
                    _ => Some("expected Rename modal"),
                }
            },
        ),
    ];
    for &(name, build, check) in cases {
        let mut state = build();
        if let Some(what) = check(&mut state) {
            panic!("case {name}: {what}");
        }
    }
}

#[test]
fn test_rename_cancel_and_mutual_exclusion() {
    fn rename_state() -> EditorState {
        let mut state = make_editor_with_tree();
        state.selected_workspace_path = Some("/tmp".to_string());
        state
    }
    fn start(state: &mut EditorState) {
        let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
        assert!(
            matches!(state.active_modal, Some(ModalKind::Rename(_))),
            "rename starts"
        );
    }
    fn no_rename(state: &EditorState) -> Option<&'static str> {
        matches!(state.active_modal, Some(ModalKind::Rename(_))).then_some("rename not cancelled")
    }
    type Run = fn(&mut EditorState) -> Option<&'static str>;
    // (name, run after a rename was started)
    let cases: &[(&str, Run)] = &[
        ("rename_cancel_and_escape", |s| {
            for msg in [EditorMessage::RenameCancel, EditorMessage::Escape] {
                start(s);
                let _ = s.update(msg.clone());
                if s.active_modal.is_some() {
                    return Some("modal not closed");
                }
            }
            None
        }),
        ("rename_mutual_exclusion_with_new_item", |s| {
            let _ = s.update(EditorMessage::NewFileRequested("src".into()));
            no_rename(s)?;
            if !matches!(s.active_modal, Some(ModalKind::NewItem(_))) {
                return Some("new item modal not opened");
            }
            let _ = s.update(EditorMessage::NewFileRequested(String::new()));
            if !matches!(s.active_modal, Some(ModalKind::NewItem(_))) {
                return Some("new item modal not reopened");
            }
            let _ = s.update(EditorMessage::RenameRequested("Cargo.toml".into()));
            (!matches!(s.active_modal, Some(ModalKind::Rename(_)))).then_some("rename lost")
        }),
        ("rename_cancelled_by_toggle_dir_click", |s| {
            let _ = s.update(EditorMessage::ToggleDir("src".into()));
            no_rename(s)
        }),
        ("rename_cancelled_by_select_file_click", |s| {
            let _ = s.update(EditorMessage::SelectFile("src/main.rs".into()));
            no_rename(s)
        }),
        ("rename_mutual_exclusion_cancelled_by_new_file", |s| {
            let _ = s.update(EditorMessage::NewFileRequested("src".into()));
            no_rename(s)?;
            (!matches!(s.active_modal, Some(ModalKind::NewItem(_))))
                .then_some("new item modal missing")
        }),
        ("rename_mutual_exclusion_cancelled_by_new_directory", |s| {
            let _ = s.update(EditorMessage::NewDirectoryRequested("src".into()));
            no_rename(s)?;
            (!matches!(s.active_modal, Some(ModalKind::NewItem(_))))
                .then_some("new item modal missing")
        }),
        ("rename_mutual_exclusion_cancelled_by_delete_file", |s| {
            let _ = s.update(EditorMessage::DeleteFileRequested("other.rs".into()));
            no_rename(s)?;
            match &s.active_modal {
                Some(ModalKind::DeleteConfirm(target)) => {
                    (target.path != "other.rs").then_some("wrong delete target")
                }
                _ => Some("delete confirmation missing"),
            }
        }),
        ("rename_mutual_exclusion_cancelled_by_delete_dir", |s| {
            let _ = s.update(EditorMessage::DeleteDirectoryRequested("src".into()));
            no_rename(s)?;
            match &s.active_modal {
                Some(ModalKind::DeleteConfirm(target)) => {
                    (target.path != "src").then_some("wrong delete target")
                }
                _ => Some("delete confirmation missing"),
            }
        }),
    ];
    for &(name, run) in cases {
        let mut state = rename_state();
        start(&mut state);
        if let Some(what) = run(&mut state) {
            panic!("case {name}: {what}");
        }
    }
}

// ── Rename validation tests ────────────────────────────────

fn setup_rename_state(state: &mut EditorState, input_text: &str) {
    state.selected_workspace_path = Some("/tmp".to_string());
    let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
    if let Some(ModalKind::Rename(ref mut rt)) = state.active_modal {
        rt.input_text.set_text(input_text);
    }
}

/// Helper: set up a rename with `input`, submit it, and assert the error.
fn assert_rename_rejects(name: &str, input: &str, expected: Option<&'static str>) {
    let mut state = make_editor_with_tree();
    setup_rename_state(&mut state, input);
    let _ = state.update(EditorMessage::RenameSubmit);
    let err = match &state.active_modal {
        Some(ModalKind::Rename(rt)) => rt.error.as_deref(),
        _ => None,
    };
    assert_eq!(err, expected, "case {name}: rejection of {input:?}");
}

#[test]
fn test_rename_validation() {
    const SEPARATOR: &str = "Name cannot contain path separators";
    // (name, input, expected rejection)
    let cases = [
        ("empty", "   ", Some("Name cannot be empty")),
        ("slash", "foo/bar.rs", Some(SEPARATOR)),
        ("backslash", "foo\\bar.rs", Some(SEPARATOR)),
        ("nul", "foo\0bar.rs", Some(SEPARATOR)),
        ("dot", ".", Some("Invalid name")),
        ("dot_dot", "..", Some("Invalid name")),
    ];
    for &(name, input, expected) in &cases {
        assert_rename_rejects(name, input, expected);
    }
}

#[cfg(target_os = "windows")]
#[test]
fn test_rename_validation_os_reserved_names() {
    let reserved = ["con", "NUL", "prn", "AUX", "com1", "lpt3"];
    for name in &reserved {
        assert_rename_rejects(name, name, Some("Name is reserved by the operating system"));
    }
}

#[test]
fn test_rename_validation_target_already_exists() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let ws = tmp_dir.path().to_string_lossy().to_string();
    // Create a file that would conflict.
    let existing = tmp_dir.path().join("existing.txt");
    std::fs::write(&existing, "").unwrap();

    let mut state = make_editor_with_tree();
    state.selected_workspace_path = Some(ws.clone());
    let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
    if let Some(ModalKind::Rename(ref mut rt)) = state.active_modal {
        rt.input_text.set_text("existing.txt");
    }
    let _ = state.update(EditorMessage::RenameSubmit);
    let err = match &state.active_modal {
        Some(ModalKind::Rename(rt)) => rt.error.as_deref(),
        _ => None,
    };
    assert_eq!(
        err,
        Some("A file or directory with that name already exists")
    );
}

#[test]
fn test_rename_stale_generation_discarded() {
    let mut state = make_editor_with_tree();
    state.selected_workspace_path = Some("/tmp".to_string());
    // Expand src so it's visible for RenameRequested.
    state.file_tree.expanded_dirs.insert("src".to_string());
    state.rebuild_tree();

    // Dispatch a rename for a non-root path (src/main.rs) so that the
    // staleness check in RenameCompleted (which only applies when the
    // parent dir is non-empty) is actually exercised.
    let _ = state.update(EditorMessage::RenameRequested("src/main.rs".into()));
    if let Some(ModalKind::Rename(ref mut rt)) = state.active_modal {
        rt.input_text.set_text("lib.rs");
    }
    let _ = state.update(EditorMessage::RenameSubmit);

    // Simulate a stale RenameCompleted whose rename_gen does not
    // match the current dir_generations entry for the parent dir ("src").
    // It passes dir_entries: Ok(vec![]) — if the staleness guard fails and
    // this result is applied, it would overwrite dir_entries["src"] with
    // an empty vec, losing the original children.
    let task = state.update(EditorMessage::RenameCompleted {
        old_path: "src/main.rs".into(),
        new_path: "src/lib.rs".into(),
        is_dir: false,
        result: Ok(()),
        dir_entries: Ok(vec![]),
        rename_gen: 0, // stale — doesn't match dir_generations["src"]
    });
    // The stale result should be a no-op (discarded silently).
    let _ = task;
    // dir_entries["src"] must still contain its original entries — if
    // the stale result were applied, the empty vec would have replaced them.
    let src_entries = state.dir_entries.get("src");
    assert!(
        src_entries.is_some(),
        "dir_entries[\"src\"] should still exist"
    );
    if let Some(entries) = src_entries {
        assert_eq!(entries.len(), 1, "should still have one entry");
        assert_eq!(entries[0].name, "main.rs");
        assert_eq!(entries[0].full_path, "src/main.rs");
    }
    // selected_file should not have been updated.
    assert_eq!(state.selected_file, None);
}

// ── rekey helpers ──────────────────────────────────────────

type RebaseCase<'a> = (
    &'a str,
    &'a str,
    &'a str,
    &'a [&'a str],
    &'a [(&'a str, &'a str)],
);

#[test]
fn test_rebase_tree_key() {
    // (name, old prefix, new prefix, keys, expected rebased keys)
    let cases: &[RebaseCase] = &[
        ("empty", "old/", "new/", &[], &[]),
        ("no_match", "old/", "new/", &["a", "b"], &[]),
        (
            "some_match",
            "old/",
            "new",
            &["old/foo", "other", "old/bar/baz"],
            &[("old/bar/baz", "new/bar/baz"), ("old/foo", "new/foo")],
        ),
        (
            "exact_prefix",
            "dir",
            "newdir",
            &["dir"],
            &[("dir", "newdir")],
        ),
    ];
    for &(name, old_prefix, new_prefix, keys, expected) in cases {
        let mut pairs: Vec<(&str, String)> = keys
            .iter()
            .filter_map(|k| Some((*k, rebase_tree_key(k, old_prefix, new_prefix)?)))
            .collect();
        pairs.sort_by(|a, b| a.0.cmp(b.0));
        assert_eq!(
            pairs
                .iter()
                .map(|(a, b)| (*a, b.as_str()))
                .collect::<Vec<_>>(),
            expected,
            "case {name}: rebased keys"
        );
    }
}

/// The absolute helpers split paths into components, so a trailing base separator still matches.
#[test]
fn test_abs_path_helpers_are_component_wise() {
    assert!(is_within_abs("C:/ws/a.txt", "C:/ws/"));

    let moved = |rest: &str| {
        Path::new("C:/moved")
            .join(rest)
            .to_string_lossy()
            .into_owned()
    };

    assert_eq!(
        rebase_abs_path("C:/ws/a.txt", "C:/ws", "C:/moved"),
        Some(moved("a.txt"))
    );
    assert_eq!(
        rebase_abs_path("C:/ws", "C:/ws", "C:/moved"),
        Some("C:/moved".to_string())
    );
    assert_eq!(rebase_abs_path("C:/other/a.txt", "C:/ws", "C:/moved"), None);

    let mut map = HashMap::from([
        ("C:/ws/a.txt".to_string(), 1),
        ("C:/other.txt".to_string(), 2),
    ]);
    rekey_abs_map(&mut map, "C:/ws", "C:/moved");
    assert_eq!(
        map,
        HashMap::from([(moved("a.txt"), 1), ("C:/other.txt".to_string(), 2)]),
        "only what lies under the moved directory is re-keyed"
    );

    let mut toasts = HashSet::from(["C:/ws/a.txt".to_string(), "C:/other".to_string()]);
    rekey_abs_set(&mut toasts, "C:/ws", "C:/moved");
    assert_eq!(
        toasts,
        HashSet::from([moved("a.txt"), "C:/other".to_string()])
    );
}

#[test]
fn test_rekey_map_prefix_no_modify() {
    let mut map = HashMap::from([
        ("dir/file.rs".to_string(), "content_a".to_string()),
        ("dir/sub/file.rs".to_string(), "content_b".to_string()),
        ("other".to_string(), "content_c".to_string()),
    ]);
    rekey_map_prefix(&mut map, "dir/", "newdir", |_| {});
    assert_eq!(map.len(), 3);
    assert_eq!(map.get("newdir/file.rs"), Some(&"content_a".to_string()));
    assert_eq!(
        map.get("newdir/sub/file.rs"),
        Some(&"content_b".to_string())
    );
    assert_eq!(map.get("other"), Some(&"content_c".to_string()));
    assert!(!map.contains_key("dir/file.rs"));
}

#[test]
fn test_rekey_map_prefix_with_modify() {
    let mut map = HashMap::from([
        ("old/key".to_string(), vec![1, 2]),
        ("old/other".to_string(), vec![3]),
        ("keep".to_string(), vec![4]),
    ]);
    rekey_map_prefix(&mut map, "old/", "new", |v: &mut Vec<i32>| v.push(99));
    assert_eq!(map.len(), 3);
    assert_eq!(map.get("new/key"), Some(&vec![1, 2, 99]));
    assert_eq!(map.get("new/other"), Some(&vec![3, 99]));
    assert_eq!(map.get("keep"), Some(&vec![4]));
}

#[test]
fn test_rekey_set_prefix_basic() {
    let mut set = HashSet::from(["a/x".to_string(), "a/y".to_string(), "b/z".to_string()]);
    rekey_set_prefix(&mut set, "a/", "b");
    assert_eq!(set.len(), 3);
    assert!(set.contains("b/x"));
    assert!(set.contains("b/y"));
    assert!(set.contains("b/z"));
}

#[test]
fn test_rekey_set_prefix_exact() {
    let mut set = HashSet::from(["dir".to_string()]);
    rekey_set_prefix(&mut set, "dir", "newdir");
    assert_eq!(set.len(), 1);
    assert!(set.contains("newdir"));
    assert!(!set.contains("dir"));
}

#[test]
fn test_rename_dir_entries_migration_own_entry_and_full_path() {
    // Verify that after a directory rename completes, the renamed
    // directory's own dir_entries key is migrated (old_path -> new_path)
    // and child entries have their full_path fields updated.
    let mut state = make_editor_with_tree();
    state.selected_workspace_path = Some("/tmp".to_string());

    // Set up state as if the user expanded "src" and we have its children.
    state.file_tree.expanded_dirs.insert("src".to_string());
    // Add a subdirectory entry for recursive testing.
    state.dir_entries.insert(
        "src/subdir".to_string(),
        vec![fs_entry("helper.rs", "src/subdir/helper.rs", false)],
    );

    // Simulate a rename of "src" -> "lib" completing successfully.
    // Pre-populate dir_generations so the staleness guard passes
    // (rename_submit would have registered this generation before
    // firing the async operation).
    state.dir_generations.insert(String::new(), 0);
    let _ = state.update(EditorMessage::RenameCompleted {
        old_path: "src".into(),
        new_path: "lib".into(),
        is_dir: true,
        result: Ok(()),
        dir_entries: Ok(vec![fs_entry("lib", "lib", true)]),
        rename_gen: 0,
    });

    // The directory's own dir_entries entry should be migrated.
    assert!(
        !state.dir_entries.contains_key("src"),
        "old path key should be removed"
    );
    let own_entries = state.dir_entries.get("lib");
    assert!(
        own_entries.is_some(),
        "new path key should exist for the renamed directory"
    );
    // The own-key entry's children must have their full_path updated.
    if let Some(entries) = own_entries {
        assert_eq!(entries.len(), 1, "src had one child (main.rs)");
        assert_eq!(entries[0].full_path, "lib/main.rs");
    }

    // The child directory entry should be migrated with updated full_path.
    let child_entries = state.dir_entries.get("lib/subdir");
    assert!(
        child_entries.is_some(),
        "child dir_entries key should be migrated"
    );
    if let Some(entries) = child_entries {
        if let Some(entry) = entries.first() {
            assert_eq!(
                entry.full_path, "lib/subdir/helper.rs",
                "entry full_path should be updated to new prefix"
            );
        }
    }

    // The expanded_dirs should have been migrated.
    assert!(
        !state.file_tree.expanded_dirs.contains("src"),
        "old expanded_dir should be removed"
    );
    assert!(
        state.file_tree.expanded_dirs.contains("lib"),
        "new expanded_dir should exist"
    );
}

#[test]
fn test_switch_tab_relative() {
    // (name, tabs, start, direction, expected active index)
    let cases: &[(&str, usize, usize, TabDirection, usize)] = &[
        ("single tab next", 1, 0, TabDirection::Next, 0),
        ("single tab prev", 1, 0, TabDirection::Prev, 0),
        ("next wraps to first", 3, 2, TabDirection::Next, 0),
        ("prev wraps to last", 3, 0, TabDirection::Prev, 2),
        ("middle next", 3, 1, TabDirection::Next, 2),
        ("middle prev", 3, 1, TabDirection::Prev, 0),
        // With exactly two tabs, Next and Prev toggle between them.
        ("two tabs next from first", 2, 0, TabDirection::Next, 1),
        ("two tabs next from second", 2, 1, TabDirection::Next, 0),
        ("two tabs prev from first", 2, 0, TabDirection::Prev, 1),
    ];
    for &(name, tabs, start, direction, expected) in cases {
        let mut state = EditorFixture::new().tabs(tabs, start).build();
        let _ = state.switch_tab_relative(direction);
        assert_eq!(state.active_tab_index, expected, "case {name}: active tab");
    }
}

// ── Directory forgetting (deleted directories) ─────────────────

#[test]
fn test_classify_read_dir_error_not_found_vs_other() {
    use std::io::ErrorKind;
    // ENOENT / ENOTDIR mean the directory is gone or was never a
    // directory — normal and forgettable.
    assert!(matches!(
        classify_read_dir_error(&std::io::Error::new(ErrorKind::NotFound, "gone")),
        ReadDirError::NotFound
    ));
    assert!(matches!(
        classify_read_dir_error(&std::io::Error::new(ErrorKind::NotADirectory, "file")),
        ReadDirError::NotFound
    ));
    // Everything else means the directory exists but cannot be read —
    // a real problem that must keep warning.
    assert!(matches!(
        classify_read_dir_error(&std::io::Error::new(ErrorKind::PermissionDenied, "locked")),
        ReadDirError::Other(_)
    ));
}

fn make_editor_with_expanded_subdir() -> EditorState {
    let mut state = make_editor_with_tree();
    state.file_tree.expanded_dirs.insert("src".to_string());
    state.file_tree.expanded_dirs.insert("src/sub".to_string());
    state.dir_entries.insert(
        "src/sub".to_string(),
        vec![fs_entry("deep.rs", "src/sub/deep.rs", false)],
    );
    state
        .dir_entries
        .get_mut("src")
        .unwrap()
        .push(fs_entry("sub", "src/sub", true));
    state.loading_dirs.insert("src/sub".to_string());
    state.dir_generations.insert("src".to_string(), 3);
    state.dir_generations.insert("src/sub".to_string(), 7);
    state.selected_file = Some("src/sub/deep.rs".to_string());
    state.all_workspace_files = vec![
        "Cargo.toml".to_string(),
        "src/main.rs".to_string(),
        "src/sub/deep.rs".to_string(),
    ];
    // Absolute-path-keyed caches (workspace root is "/tmp").
    state
        .file_mtimes
        .insert("/tmp/src/main.rs".to_string(), std::time::SystemTime::now());
    state
        .file_generations
        .insert("/tmp/src/main.rs".to_string(), 1);
    state
        .deleted_file_toasted
        .insert("/tmp/src/main.rs".to_string());
    state
}

#[test]
fn test_dir_expanded_error_cases() {
    fn forgotten(state: &EditorState) -> Result<(), String> {
        let tree = &state.file_tree;
        let root_entries = state.dir_entries.get("").unwrap();
        for (what, present) in [
            ("src", tree.expanded_dirs.contains("src")),
            ("src/sub", tree.expanded_dirs.contains("src/sub")),
            ("src listing", state.dir_entries.contains_key("src")),
            ("src/sub listing", state.dir_entries.contains_key("src/sub")),
            ("src gen", state.dir_generations.contains_key("src")),
            ("src/sub gen", state.dir_generations.contains_key("src/sub")),
        ] {
            if present {
                return Err(format!("{what} not forgotten"));
            }
        }
        if state.loading_dirs.contains("src/sub") {
            return Err("load flag not dropped".to_string());
        }
        if state.selected_file.is_some() {
            return Err("selection not cleared".to_string());
        }
        if state
            .all_workspace_files
            .iter()
            .any(|p| p.starts_with("src/"))
        {
            return Err("quick-open not pruned".to_string());
        }
        if root_entries.iter().any(|e| e.full_path == "src") {
            return Err("root entry not pruned".to_string());
        }
        if tree.visible_tree_nodes.iter().any(|(p, _)| p == "src") {
            return Err("visible node not pruned".to_string());
        }
        Ok(())
    }
    fn kept(state: &EditorState) -> Result<(), String> {
        let tree = &state.file_tree;
        for (what, present) in [
            ("src", tree.expanded_dirs.contains("src")),
            ("src/sub", tree.expanded_dirs.contains("src/sub")),
            ("src listing", state.dir_entries.contains_key("src")),
            ("src/sub listing", state.dir_entries.contains_key("src/sub")),
        ] {
            if !present {
                return Err(format!("{what} dropped"));
            }
        }
        if state.selected_file.as_deref() != Some("src/sub/deep.rs") {
            return Err("selection lost".to_string());
        }
        if !state.all_workspace_files.iter().any(|p| p == "src/main.rs") {
            return Err("quick-open pruned".to_string());
        }
        Ok(())
    }
    type Check = fn(&EditorState) -> Result<(), String>;
    // (name, error, quiet, check)
    let cases: &[(&str, ReadDirError, bool, Check)] = &[
        ("not_found_quiet", ReadDirError::NotFound, true, forgotten),
        ("not_found_loud", ReadDirError::NotFound, false, forgotten),
        (
            "other_error_keeps_path",
            ReadDirError::Other("Permission denied".to_string()),
            true,
            kept,
        ),
    ];
    for &(name, ref error, quiet, check) in cases {
        let mut state = make_editor_with_expanded_subdir();
        let r#gen = state.bump_generation();
        state.dir_generations.insert("src".to_string(), r#gen);
        let _ = state.update(EditorMessage::DirExpanded {
            dir_path: "src".to_string(),
            r#gen,
            entries: Err(error.clone()),
            quiet,
        });
        assert_eq!(check(&state), Ok(()), "case {name}");
    }
}

#[test]
fn test_forget_directory_prunes_path_descendants_and_root() {
    // Phase 1: forgetting a subdirectory prunes it and its descendants from
    // every refresh/cache structure while preserving the parent's state.
    let mut state = make_editor_with_expanded_subdir();
    state.dir_generations.insert(String::new(), 9); // parent (root) gen must survive
    state.forget_directory("src");

    assert!(state.file_tree.expanded_dirs.is_empty());
    assert!(!state.dir_entries.contains_key("src"));
    assert!(!state.dir_entries.contains_key("src/sub"));
    assert!(state.loading_dirs.is_empty());
    assert!(!state.dir_generations.contains_key("src"));
    assert!(!state.dir_generations.contains_key("src/sub"));
    assert_eq!(state.dir_generations.get(""), Some(&9));
    assert_eq!(state.selected_file, None);
    assert_eq!(state.all_workspace_files, vec!["Cargo.toml".to_string()]);

    // Root listing keeps unrelated entries but drops the deleted node.
    let root_entries = state.dir_entries.get("").unwrap();
    assert_eq!(root_entries.len(), 1);
    assert_eq!(root_entries[0].full_path, "Cargo.toml");

    // Absolute-path-keyed caches are pruned as well.
    assert!(state.file_mtimes.is_empty());
    assert!(state.file_generations.is_empty());
    assert!(state.deleted_file_toasted.is_empty());

    state.rebuild_tree();
    assert!(
        !state
            .file_tree
            .visible_tree_nodes
            .iter()
            .any(|(p, _)| p == "src")
    );

    // Phase 2: forgetting the root (workspace itself gone) clears the whole
    // tree — including the surviving root listing and, per the doc contract,
    // the absolute-path-keyed caches under the workspace.
    state
        .file_mtimes
        .insert("/tmp/Cargo.toml".to_string(), std::time::SystemTime::now());
    state
        .file_generations
        .insert("/tmp/Cargo.toml".to_string(), 4);
    state
        .deleted_file_toasted
        .insert("/tmp/Cargo.toml".to_string());
    state.forget_directory("");

    assert!(state.dir_entries.is_empty());
    assert!(state.file_tree.expanded_dirs.is_empty());
    assert!(state.loading_dirs.is_empty());
    assert!(state.dir_generations.is_empty());
    assert_eq!(state.selected_file, None);
    assert!(state.all_workspace_files.is_empty());
    assert!(state.file_mtimes.is_empty());
    assert!(state.file_generations.is_empty());
    assert!(state.deleted_file_toasted.is_empty());

    state.rebuild_tree();
    assert!(state.file_tree.nodes.is_empty());
    assert!(state.file_tree.visible_tree_nodes.is_empty());
}

#[test]
fn test_dir_deleted_cases() {
    type Check = fn(&EditorState) -> Result<(), String>;

    /// The GUI-delete success path prunes the directory and registers a re-read.
    fn pruned_and_refreshed(state: &EditorState) -> Result<(), String> {
        let tree = &state.file_tree;
        for (what, kept) in [
            ("src", tree.expanded_dirs.contains("src")),
            ("src/sub", tree.expanded_dirs.contains("src/sub")),
            ("src listing", state.dir_entries.contains_key("src")),
            ("src/sub listing", state.dir_entries.contains_key("src/sub")),
        ] {
            if kept {
                return Err(format!("{what} not pruned"));
            }
        }
        if state.selected_file.is_some() {
            return Err("selection not cleared".to_string());
        }
        if state
            .all_workspace_files
            .iter()
            .any(|p| p.starts_with("src/"))
        {
            return Err("quick-open not pruned".to_string());
        }

        // The parent (root) listing survives but no longer shows the deleted dir.
        let root_entries = state.dir_entries.get("").unwrap();
        if root_entries.iter().any(|e| e.full_path == "src") {
            return Err("root still lists src".to_string());
        }
        if root_entries.len() != 1 {
            return Err("root listing size".to_string());
        }
        if root_entries[0].full_path != "Cargo.toml" {
            return Err("root survivor".to_string());
        }

        // A fresh generation slot for the parent re-read task was registered.
        if !state.dir_generations.contains_key("") {
            return Err("parent re-read not registered".to_string());
        }
        Ok(())
    }

    /// A completion from another workspace must not prune this workspace.
    fn untouched(state: &EditorState) -> Result<(), String> {
        let tree = &state.file_tree;
        for (what, kept) in [
            ("src", tree.expanded_dirs.contains("src")),
            ("src/sub", tree.expanded_dirs.contains("src/sub")),
            ("src listing", state.dir_entries.contains_key("src")),
            ("src/sub listing", state.dir_entries.contains_key("src/sub")),
        ] {
            if !kept {
                return Err(format!("{what} dropped"));
            }
        }
        if state.selected_file.as_deref() != Some("src/sub/deep.rs") {
            return Err("selection changed".to_string());
        }
        if state.all_workspace_files.len() != 3 {
            return Err("quick-open changed".to_string());
        }
        if state.dir_generations.contains_key("") {
            return Err("parent re-read registered".to_string());
        }
        if !tree.visible_tree_nodes.iter().any(|(p, _)| p == "src") {
            return Err("visible src dropped".to_string());
        }
        Ok(())
    }

    let elsewhere = "/some/other/workspace";
    // (name, workspace path reported by the completion, check)
    let cases: &[(&str, &str, Check)] = &[
        ("matching_workspace_prunes", "/tmp", pruned_and_refreshed),
        ("stale_workspace_ignored", elsewhere, untouched),
    ];
    for &(name, workspace_path, check) in cases {
        let mut state = make_editor_with_expanded_subdir();
        let _ = state.update(EditorMessage::DirDeleted {
            dir_path: "src".to_string(),
            workspace_path: workspace_path.to_string(),
        });
        assert_eq!(check(&state), Ok(()), "case {name}");
    }
}

#[test]
fn test_confirm_delete_dir_defers_prune_until_success() {
    // ConfirmDelete starts the async remove_dir_all but must NOT prune tree
    // state yet: if the delete fails (e.g. permission denied), the
    // still-existing directory must be fully intact in the tree. The prune
    // only happens in the DirDeleted success message, so asserting the tree
    // is untouched at delete-start is the testable proxy for the failure
    // path (the async task is not executed in tests).
    let mut state = make_editor_with_expanded_subdir();
    state.active_modal = Some(ModalKind::DeleteConfirm(DeleteConfirmTarget {
        path: "src".to_string(),
        is_dir: true,
        dirty_tab_count: 0,
        abs_path: "/tmp/src".to_string(),
    }));

    let _ = state.update(EditorMessage::ConfirmDelete);

    // Modal closed, but all tree/cache state survives the delete-start.
    assert!(state.active_modal.is_none());
    assert!(state.file_tree.expanded_dirs.contains("src"));
    assert!(state.file_tree.expanded_dirs.contains("src/sub"));
    assert!(state.dir_entries.contains_key("src"));
    assert!(state.dir_entries.contains_key("src/sub"));
    assert_eq!(state.selected_file.as_deref(), Some("src/sub/deep.rs"));
    assert_eq!(state.all_workspace_files.len(), 3);
    assert!(state.file_mtimes.contains_key("/tmp/src/main.rs"));
    assert!(state.file_generations.contains_key("/tmp/src/main.rs"));

    // No orphaned parent generation: the dir-delete path registers the
    // parent slot only when the delete succeeds (via DirDeleted).
    assert!(!state.dir_generations.contains_key(""));
    assert_eq!(state.dir_generations.len(), 2); // fixture: "src" and "src/sub"
}

// ── Gitignore dimming (in-process matcher) ──────────────────────

fn find_node<'a>(nodes: &'a [widgets::TreeNode], path: &str) -> &'a widgets::TreeNode {
    find_in_tree(nodes, path).unwrap_or_else(|| panic!("node not found in tree: {path}"))
}

fn find_in_tree<'a>(nodes: &'a [widgets::TreeNode], path: &str) -> Option<&'a widgets::TreeNode> {
    for node in nodes {
        if node.full_path == path {
            return Some(node);
        }
        if let Some(found) = find_in_tree(&node.children, path) {
            return Some(found);
        }
    }
    None
}

#[test]
fn test_ignore_dimming_survives_collapse_and_expand() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let ws = dir.path();
    // Dummy `.git` dir so `require_git` is satisfied — no `git init` needed.
    std::fs::create_dir(ws.join(".git")).unwrap();
    std::fs::write(ws.join(".gitignore"), "secret/\n").unwrap();
    std::fs::create_dir(ws.join("secret")).unwrap();
    std::fs::write(ws.join("secret/hidden.txt"), "x").unwrap();
    std::fs::write(ws.join("visible.txt"), "x").unwrap();

    let mut matcher = build_ignore_matcher(ws);
    assert!(matcher.is_some());

    let mut dir_entries = HashMap::new();
    dir_entries.insert(
        String::new(),
        vec![
            fs_entry("secret", "secret", true),
            fs_entry("visible.txt", "visible.txt", false),
        ],
    );
    dir_entries.insert(
        "secret".into(),
        vec![fs_entry("hidden.txt", "secret/hidden.txt", false)],
    );

    // Expanded tree: secret dir fully populated and stamped.
    let expanded = HashSet::from(["secret".to_string()]);
    let mut nodes = build_hierarchical_tree(&dir_entries, &expanded, "");
    stamp_ignored_nodes(&mut nodes, &mut matcher);
    assert!(find_node(&nodes, "secret").ignored);
    assert!(find_node(&nodes, "secret/hidden.txt").ignored);
    assert!(!find_node(&nodes, "visible.txt").ignored);

    // Collapse path (raw rebuild, no visible-list rebuild): dimming survives.
    let collapsed = HashSet::new();
    let mut nodes = build_hierarchical_tree(&dir_entries, &collapsed, "");
    stamp_ignored_nodes(&mut nodes, &mut matcher);
    assert!(find_node(&nodes, "secret").ignored);

    // Re-expand: children come back and are still dimmed.
    let mut nodes = build_hierarchical_tree(&dir_entries, &expanded, "");
    stamp_ignored_nodes(&mut nodes, &mut matcher);
    assert!(find_node(&nodes, "secret").ignored);
    assert!(find_node(&nodes, "secret/hidden.txt").ignored);
}

#[test]
fn test_negated_patterns_do_not_dim() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let ws = dir.path();
    std::fs::create_dir(ws.join(".git")).unwrap();
    // `build/*` ignores the *contents* of build but not build itself, so a
    // negated child can be re-included. (A plain `build/` + `!build/keep.txt`
    // would NOT un-ignore keep.txt — git can't re-include a file inside an
    // excluded directory, and the ignore crate follows that.)
    std::fs::write(ws.join(".gitignore"), "build/*\n!build/keep.txt\n").unwrap();
    std::fs::create_dir(ws.join("build")).unwrap();
    std::fs::write(ws.join("build/keep.txt"), "x").unwrap();
    std::fs::write(ws.join("build/other.txt"), "x").unwrap();

    let mut matcher = build_ignore_matcher(ws);
    assert!(matcher.is_some());

    let mut dir_entries = HashMap::new();
    dir_entries.insert(String::new(), vec![fs_entry("build", "build", true)]);
    dir_entries.insert(
        "build".into(),
        vec![
            fs_entry("keep.txt", "build/keep.txt", false),
            fs_entry("other.txt", "build/other.txt", false),
        ],
    );

    let expanded = HashSet::from(["build".to_string()]);
    let mut nodes = build_hierarchical_tree(&dir_entries, &expanded, "");
    stamp_ignored_nodes(&mut nodes, &mut matcher);

    let build = find_node(&nodes, "build");
    assert!(
        !build.ignored,
        "build/* does not match the build dir itself"
    );
    for child in &build.children {
        if child.full_path == "build/keep.txt" {
            assert!(!child.ignored, "negated keep.txt must not be dimmed");
        } else {
            assert!(child.ignored, "non-negated other.txt must be dimmed");
        }
    }
}

#[test]
fn test_matcher_covers_subdirectory_of_repo() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let root = dir.path();
    std::fs::create_dir(root.join(".git")).unwrap();
    std::fs::create_dir(root.join("sub")).unwrap();
    std::fs::write(root.join("sub/hidden.txt"), "x").unwrap();
    std::fs::write(root.join("sub/kept.txt"), "x").unwrap();
    // A root-level gitignore rule that targets the sub workspace applies to a
    // subdirectory-of-repo workspace even though the workspace's matcher root
    // is `sub` (parents default).
    std::fs::write(root.join(".gitignore"), "sub/hidden.txt\n").unwrap();

    let ws = root.join("sub");
    let mut matcher = build_ignore_matcher(&ws).unwrap();
    assert!(
        matcher.matched("hidden.txt", false).is_ignore(),
        "parent repo .gitignore should apply to a subdirectory workspace"
    );
    assert!(
        !matcher.matched("kept.txt", false).is_ignore(),
        "a non-matching file should not be ignored"
    );

    // Without a `.git` dir in the ancestry, nothing is dimmed (require_git).
    std::fs::remove_dir_all(root.join(".git")).unwrap();
    let mut matcher = build_ignore_matcher(&ws).unwrap();
    assert!(
        !matcher.matched("hidden.txt", false).is_ignore(),
        "no .git in the repo ancestry means nothing is dimmed"
    );
}

// ── Commit/staging mtime detector ───────────────────────────────

/// Bump a file's mtime by rewriting it after a pause for coarse timestamps.
fn bump_file_mtime(path: &Path) {
    std::thread::sleep(Duration::from_millis(20));
    std::fs::write(path, "changed\n").unwrap();
}

fn make_editor_with_git_ws() -> (tempfile::TempDir, EditorState) {
    let dir = tempfile::tempdir().expect("create temp dir");
    let ws = dir.path();
    let git_dir = ws.join(".git");
    std::fs::create_dir(&git_dir).unwrap();
    std::fs::create_dir(git_dir.join("logs")).unwrap();
    std::fs::write(git_dir.join("index"), "").unwrap();
    std::fs::write(git_dir.join("logs/HEAD"), "").unwrap();

    let mut state = EditorState::new();
    state.selected_workspace_name = Some("ws".to_string());
    state.selected_workspace_path = Some(ws.to_string_lossy().to_string());
    state.git_dir = find_git_dir(ws);
    (dir, state)
}

#[test]
fn test_git_status_update_re_latches_mtimes() {
    let (dir, mut state) = make_editor_with_git_ws();
    let ws = dir.path();
    let index = ws.join(".git").join("index");

    // First probe latches the baseline without triggering a refresh.
    let _ = state.update(EditorMessage::CheckGitDirty);
    assert!(!state.git_status_loading);
    assert!(state.git_dirty_mtimes.is_some());

    // Staging grew: bump `.git/index` mtime to a distinct value.
    bump_file_mtime(&index);
    let _ = state.update(EditorMessage::CheckGitDirty);
    assert!(
        state.git_status_loading,
        "an index mtime change must request a git-status refresh"
    );

    // The refresh resolves via the existing GitState → GitStatusUpdated forward.
    let mut statuses = HashMap::new();
    statuses.insert("src/main.rs".to_string(), GitFileStatus::Modified);
    let _ = state.update(EditorMessage::GitStatusUpdated {
        workspace: Some("ws".to_string()),
        statuses,
    });
    assert!(state.git_status_cache.contains_key("src/main.rs"));
    assert!(
        !state.git_status_loading,
        "GitStatusUpdated clears the guard"
    );

    // The GitStatusUpdated handler re-latched the (now-changed) mtimes, so a
    // subsequent probe sees no delta and must NOT re-trigger (loop prevented).
    let _ = state.update(EditorMessage::CheckGitDirty);
    assert!(
        !state.git_status_loading,
        "re-latched baseline must not re-trigger the detector"
    );
}

#[test]
fn test_page_visibility_gates_tree_ping() {
    let mut state = make_editor_with_tree();
    assert!(state.selected_workspace_path.is_some());
    let gen_count_before = state.dir_generations.len();

    // Hidden page: pings are dropped, so no directory refresh is dispatched.
    state.page_visible = false;
    let _ = state.update(EditorMessage::FileTreePinged);
    assert_eq!(
        state.dir_generations.len(),
        gen_count_before,
        "hidden editor must drop file-tree pings"
    );

    // Visible page: the ping triggers a refresh (a new root dir generation).
    state.page_visible = true;
    let _ = state.update(EditorMessage::FileTreePinged);
    assert!(
        state.dir_generations.len() > gen_count_before,
        "visible editor must dispatch a tree refresh on ping"
    );
}

#[test]
fn test_modal_cleared_on_page_switch() {
    let mut state = make_editor_with_tree();
    state.active_modal = Some(ModalKind::QuickOpen(QuickOpenState {
        filter: crate::gui::common::SingleLineEditorState::new(""),
        selected_index: 0,
        results: Vec::new(),
    }));

    let _ = state.update(EditorMessage::PageVisibilityChanged(false));
    assert!(
        state.active_modal.is_none(),
        "modals must close on a page switch"
    );
    assert!(!state.page_visible);
}
