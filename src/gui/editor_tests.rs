use super::*;
use crate::gui::editor_widget::{EditorAction, EditorBuffer};

// ── compute_text_matches ────────────────────────────────────

#[test]
fn test_compute_text_matches() {
    struct Case {
        text: &'static str,
        query: &'static str,
        sensitive: bool,
        expected: &'static [(usize, usize)],
    }
    let cases: &[Case] = &[
        // Empty query
        Case {
            text: "hello",
            query: "",
            sensitive: true,
            expected: &[],
        },
        // Basic match
        Case {
            text: "hello world hello",
            query: "hello",
            sensitive: true,
            expected: &[(0, 5), (12, 17)],
        },
        // No match
        Case {
            text: "hello world",
            query: "xyz",
            sensitive: true,
            expected: &[],
        },
        // Non-overlapping
        Case {
            text: "aaaaa",
            query: "aa",
            sensitive: true,
            expected: &[(0, 2), (2, 4)],
        },
        // Case-insensitive
        Case {
            text: "Hello World hello",
            query: "hello",
            sensitive: false,
            expected: &[(0, 5), (12, 17)],
        },
        // Case-insensitive no match
        Case {
            text: "Hello World",
            query: "xyz",
            sensitive: false,
            expected: &[],
        },
        // Single-char queries return empty (2-char min enforcement)
        Case {
            text: "hello",
            query: "h",
            sensitive: true,
            expected: &[],
        },
        // Boundary: shortest possible match
        Case {
            text: "ab",
            query: "ab",
            sensitive: true,
            expected: &[(0, 2)],
        },
        // Boundary: consecutive matches
        Case {
            text: "abab",
            query: "ab",
            sensitive: true,
            expected: &[(0, 2), (2, 4)],
        },
    ];
    for case in cases {
        let result = compute_text_matches(case.text, case.query, case.sensitive);
        assert_eq!(
            result.len(),
            case.expected.len(),
            "text={:?} query={:?} sensitive={}",
            case.text,
            case.query,
            case.sensitive
        );
        for (i, &(start, end)) in case.expected.iter().enumerate() {
            assert_eq!(
                result[i],
                start..end,
                "match[{i}] text={:?} query={:?} sensitive={}",
                case.text,
                case.query,
                case.sensitive
            );
        }
    }
}

// ── validate_file_content ─────────────────────────────────────

#[test]
fn test_validate_file_content_accepts_valid_input() {
    assert!(validate_file_content(b"").is_ok());
    assert!(validate_file_content(b"hello world").is_ok());
    assert!(validate_file_content("Привет мир 👋".as_bytes()).is_ok());
}

#[test]
fn test_validate_file_content_rejects_invalid_input() {
    let big = vec![b'a'; usize::try_from(MAX_FILE_SIZE).unwrap() + 1];
    let err = validate_file_content(&big).unwrap_err();
    assert!(err.starts_with("File too large"), "unexpected error: {err}");

    let bytes = b"hello\0world";
    let err = validate_file_content(bytes).unwrap_err();
    assert!(
        err.starts_with("Binary file detected"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_validate_file_content_both_conditions_reports_size_first() {
    let mut big_with_null = vec![b'a'; usize::try_from(MAX_FILE_SIZE).unwrap() + 1];
    big_with_null.push(0);
    let err = validate_file_content(&big_with_null).unwrap_err();
    assert!(
        err.starts_with("File too large"),
        "size check should be reported before null-byte check: {err}"
    );
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
fn test_build_tab_records_persists_dirty_content() {
    let tabs = vec![Tab {
        path: "/tmp/foo.md".to_string(),
        file_name: "foo.md".to_string(),
        is_dirty: true,
        has_trailing_newline: true,
        line_ending: LineEnding::Lf,
    }];
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
    let records = build_tab_records(&tabs, 0, &tab_contents);
    assert_eq!(records.len(), 1);
    assert!(records[0].is_dirty);
    assert_eq!(records[0].dirty_content.as_deref(), Some("unsaved edits"));
}

#[test]
fn test_build_tab_records_clears_dirty_content_when_clean() {
    let tabs = vec![Tab {
        path: "/tmp/foo.md".to_string(),
        file_name: "foo.md".to_string(),
        is_dirty: false,
        has_trailing_newline: false,
        line_ending: LineEnding::Lf,
    }];
    let records = build_tab_records(&tabs, 0, &HashMap::new());
    assert!(records[0].dirty_content.is_none());
}

#[test]
fn test_save_result_ignores_stale_save() {
    let mut state = EditorState::new();
    let path = "/tmp/stale.md".to_string();
    state.tabs.push(Tab {
        path: path.clone(),
        file_name: "stale.md".to_string(),
        is_dirty: true,
        has_trailing_newline: false,
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

// ── byte_offset_to_cursor_pos ───────────────────────────────

#[test]
fn test_byte_offset_to_cursor_pos() {
    struct Case {
        text: &'static str,
        byte_offset: usize,
        expected: Option<(usize, usize)>,
    }
    let cases: &[Case] = &[
        // Unicode multi-byte chars
        Case {
            text: "Привет мир",
            byte_offset: 13, // start of "м"
            expected: Some((0, 7)),
        },
        // Start of content
        Case {
            text: "hello\nworld",
            byte_offset: 0,
            expected: Some((0, 0)),
        },
        // Second line
        Case {
            text: "hello\nworld",
            byte_offset: 6, // after "hello\n"
            expected: Some((1, 0)),
        },
        // Middle of a line
        Case {
            text: "hello\nworld",
            byte_offset: 8, // "wo"
            expected: Some((1, 2)),
        },
        // Beyond text length
        Case {
            text: "hello",
            byte_offset: 100,
            expected: None,
        },
        // Empty content
        Case {
            text: "",
            byte_offset: 0,
            expected: Some((0, 0)),
        },
    ];
    for case in cases {
        let content = EditorBuffer::with_text(case.text, None);
        let pos = byte_offset_to_cursor_pos(&content, case.byte_offset);
        assert_eq!(
            pos, case.expected,
            "text={:?} offset={}",
            case.text, case.byte_offset
        );
    }
}

// ── UndoStack ───────────────────────────────────────────────

fn setup_undo_stack(text: &str) -> (EditorBuffer, UndoStack) {
    (EditorBuffer::with_text(text, None), UndoStack::new())
}

#[test]
fn test_undo_stack_snap_and_undo() {
    let (content, mut stack) = setup_undo_stack("original");
    stack.snap_before_edit(&content);

    // Simulate edit
    let modified = EditorBuffer::with_text("modified", None);
    let snapshot = stack.undo(&modified).unwrap();
    assert_eq!(snapshot.text, "original");
}

#[test]
fn test_undo_stack_redo() {
    let (content, mut stack) = setup_undo_stack("original");
    stack.snap_before_edit(&content);

    let modified = EditorBuffer::with_text("modified", None);
    let _ = stack.undo(&modified);

    let snapshot = stack.redo(&modified).unwrap();
    assert_eq!(snapshot.text, "modified");
}

#[test]
fn test_undo_stack_new_edit_clears_redo() {
    let (content, mut stack) = setup_undo_stack("v1");
    stack.snap_before_edit(&content);

    let v2 = EditorBuffer::with_text("v2", None);
    let _ = stack.undo(&v2);

    // New edit after undo should clear redo.
    let v3 = EditorBuffer::with_text("v3", None);
    stack.snap_before_edit(&v3);

    assert!(stack.redo(&v3).is_none());
}

#[test]
fn test_undo_stack_cursor_restoration() {
    let (content, mut stack) = setup_undo_stack("line1\nline2\nline3");
    // Move cursor to (1, 2) — line 1, column 2 ("ne2")
    content.move_to(1, 2);
    stack.snap_before_edit(&content);

    let modified = EditorBuffer::with_text("changed", None);
    let snapshot = stack.undo(&modified).unwrap();
    assert_eq!(snapshot.cursor_line, 1);
    assert_eq!(snapshot.cursor_col, 2);
}

// ── Tree keyboard navigation focus state tests ──────────────────

/// Helper to create a minimal EditorState with a simple tree.
fn make_editor_with_tree() -> EditorState {
    let mut state = EditorState::new();
    state.selected_workspace_path = Some("/tmp".to_string());
    // Populate root dir_entries so build_hierarchical_tree works.
    state.dir_entries.insert(
        String::new(),
        vec![
            FsEntry {
                name: "src".to_string(),
                full_path: "src".to_string(),
                is_dir: true,
                error: None,
            },
            FsEntry {
                name: "Cargo.toml".to_string(),
                full_path: "Cargo.toml".to_string(),
                is_dir: false,
                error: None,
            },
        ],
    );
    // Populate "src" dir_entries so children show when expanded.
    state.dir_entries.insert(
        "src".to_string(),
        vec![FsEntry {
            name: "main.rs".to_string(),
            full_path: "src/main.rs".to_string(),
            is_dir: false,
            error: None,
        }],
    );
    // Build the tree from dir_entries (consistent with real behavior).
    state.rebuild_tree();
    state
}

#[test]
fn test_rebuild_visible_tree_flattens_nodes() {
    let state = make_editor_with_tree();
    assert_eq!(state.file_tree.visible_tree_nodes.len(), 2);
    assert_eq!(state.file_tree.visible_tree_nodes[0].0, "src");
    assert!(state.file_tree.visible_tree_nodes[0].1); // is_dir
    assert_eq!(state.file_tree.visible_tree_nodes[1].0, "Cargo.toml");
    assert!(!state.file_tree.visible_tree_nodes[1].1); // not is_dir
}

#[test]
fn test_rebuild_visible_tree_with_expanded_dir() {
    let mut state = make_editor_with_tree();
    state.file_tree.expanded_dirs.insert("src".to_string());
    // Rebuild tree from dir_entries with expanded state, then flatten.
    state.file_tree.nodes =
        build_hierarchical_tree(&state.dir_entries, &state.file_tree.expanded_dirs, "");
    state.file_tree.rebuild_visible();
    assert_eq!(state.file_tree.visible_tree_nodes.len(), 3);
    assert_eq!(state.file_tree.visible_tree_nodes[0].0, "src");
    assert_eq!(state.file_tree.visible_tree_nodes[1].0, "src/main.rs");
    assert_eq!(state.file_tree.visible_tree_nodes[2].0, "Cargo.toml");
}

#[test]
fn test_tree_focus_toggled_sets_focus() {
    let mut state = make_editor_with_tree();
    assert!(!state.file_tree.tree_focused);

    // Toggle on
    let _ = state.update(EditorMessage::TreeFocusToggled);
    assert!(state.file_tree.tree_focused);

    // Toggle off
    let _ = state.update(EditorMessage::TreeFocusToggled);
    assert!(!state.file_tree.tree_focused);
}

#[test]
fn test_tree_focus_toggled_empty_tree_stays_off() {
    let mut state = EditorState::new();
    assert!(!state.file_tree.tree_focused);

    let _ = state.update(EditorMessage::TreeFocusToggled);
    assert!(!state.file_tree.tree_focused); // No visible nodes, focus rejected
}

#[test]
fn test_misc_focus_actions() {
    struct Case {
        name: &'static str,
        msg: EditorMessage,
        setup: fn(&mut EditorState),
        check: fn(&EditorState, name: &str),
    }
    let cases: &[Case] = &[
        Case {
            name: "escape_clears_tree_focus",
            msg: EditorMessage::Escape,
            setup: |s| s.file_tree.tree_focused = true,
            check: |s, name| assert!(!s.file_tree.tree_focused, "case: {name}"),
        },
        Case {
            name: "toggle_dir_sets_tree_focus",
            msg: EditorMessage::ToggleDir("src".to_string()),
            setup: |s| {
                s.selected_file = Some("Cargo.toml".to_string());
            },
            check: |s, name| {
                assert!(s.file_tree.tree_focused, "case: {name}");
                assert!(s.selected_file.is_none(), "case: {name}");
            },
        },
        Case {
            name: "select_file_keeps_tree_focus",
            msg: EditorMessage::SelectFile("src/main.rs".to_string()),
            setup: |s| s.file_tree.tree_focused = true,
            check: |s, name| assert!(s.file_tree.tree_focused, "case: {name}"),
        },
        // A mouse-originated EditorAction (like MoveTo from a click)
        // should transfer focus from the file tree to the editor.
        Case {
            name: "editor_action_clears_tree_focus",
            msg: EditorMessage::EditorAction(EditorAction::MoveTo { line: 0, col: 0 }),
            setup: |s| {
                s.file_tree.tree_focused = true;
                s.pending_enter_dir = Some("src".to_string());
                s.active_modal = Some(ModalKind::Rename(RenameTarget {
                    path: "src/main.rs".to_string(),
                    abs_path: String::new(),
                    is_dir: false,
                    ws_root: String::new(),
                    input_text: "main.rs".to_string(),
                    error: None,
                }));
            },
            check: |s, name| {
                assert!(!s.file_tree.tree_focused, "case: {name}");
                assert_eq!(s.pending_enter_dir, None, "case: {name}");
                assert!(s.active_modal.is_none(), "case: {name}");
            },
        },
    ];
    for case in cases {
        let mut state = make_editor_with_tree();
        (case.setup)(&mut state);
        let _ = state.update(case.msg.clone());
        (case.check)(&state, case.name);
    }
}

#[test]
fn test_tree_nav_enter() {
    struct Case {
        name: &'static str,
        focused: bool,
        start_idx: usize,
        /// Set selected_file before the message
        pre_select_file: bool,
        /// Expected tree_focused after
        expect_focused: bool,
        /// Expected focus index after (None = skip check)
        expected_idx: Option<usize>,
        /// Additional per-case assertions
        check: Option<fn(&EditorState, name: &str)>,
    }
    let cases: &[Case] = &[
        // TreeNavEnter on a file dispatches an async load task, but
        // tree_focused stays true in the same-turn state update.
        Case {
            name: "on_file_dispatches_task",
            focused: true,
            start_idx: 1,
            pre_select_file: false,
            expect_focused: true,
            expected_idx: None,
            check: None,
        },
        Case {
            name: "not_focused_ignored",
            focused: false,
            start_idx: 1,
            pre_select_file: false,
            expect_focused: false,
            expected_idx: Some(1),
            check: None,
        },
        Case {
            name: "on_dir_expands_and_advances",
            focused: true,
            start_idx: 0,
            pre_select_file: true,
            expect_focused: true,
            expected_idx: Some(1),
            check: Some(|s, name| {
                assert!(s.file_tree.expanded_dirs.contains("src"), "case: {name}");
                assert!(s.selected_file.is_none(), "case: {name}");
                assert_eq!(s.file_tree.visible_tree_nodes[1].0, "src/main.rs");
            }),
        },
    ];
    for case in cases {
        let mut state = make_editor_with_tree();
        state.file_tree.tree_focused = case.focused;
        state.file_tree.tree_focus_index = case.start_idx;
        if case.pre_select_file {
            state.selected_file = Some("Cargo.toml".to_string());
        }
        let _ = state.update(EditorMessage::TreeNavEnter);
        assert_eq!(
            state.file_tree.tree_focused, case.expect_focused,
            "case: {}",
            case.name
        );
        if let Some(idx) = case.expected_idx {
            assert_eq!(state.file_tree.tree_focus_index, idx, "case: {}", case.name);
        }
        if let Some(check) = case.check {
            check(&state, case.name);
        }
    }
}

#[test]
fn test_visible_tree_clamps_focus_on_rebuild() {
    let mut state = make_editor_with_tree();
    state.file_tree.tree_focus_index = 999; // Way out of range
    state.file_tree.rebuild_visible();
    assert_eq!(
        state.file_tree.tree_focus_index,
        state.file_tree.visible_tree_nodes.len() - 1
    );
}

#[test]
fn test_async_enter_dir_sets_pending_then_advances() {
    let mut state = EditorState::new();
    state.selected_workspace_path = Some("/tmp".to_string());
    // Set up a tree where "src" dir_entries are empty (needs async load).
    state.dir_entries.insert(
        String::new(),
        vec![FsEntry {
            name: "src".to_string(),
            full_path: "src".to_string(),
            is_dir: true,
            error: None,
        }],
    );
    state.rebuild_tree();
    state.file_tree.tree_focused = true;
    state.file_tree.tree_focus_index = 0; // "src"

    let _task = state.update(EditorMessage::TreeNavEnter);
    // "src" needs async loading — pending_enter_dir is set.
    assert_eq!(state.pending_enter_dir.as_deref(), Some("src"));
    assert!(state.file_tree.expanded_dirs.contains("src"));
    // Focus stays on "src" until children load.
    assert_eq!(state.file_tree.tree_focus_index, 0);

    // Simulate DirExpanded completing with children.
    let entries = vec![FsEntry {
        name: "main.rs".to_string(),
        full_path: "src/main.rs".to_string(),
        is_dir: false,
        error: None,
    }];
    let dir_gen = state.generation;
    let _task = state.update(EditorMessage::DirExpanded {
        dir_path: "src".to_string(),
        r#gen: dir_gen,
        entries: Ok(entries),
        quiet: false,
    });
    // Focus should have advanced to the first child.
    assert_eq!(state.pending_enter_dir, None);
    assert_eq!(state.file_tree.tree_focus_index, 1); // "src/main.rs"
}

#[test]
fn test_toggle_dir_async_load_and_complete() {
    let mut state = EditorState::new();
    state.selected_workspace_path = Some("/tmp".to_string());
    // "src" dir has no cached entries → needs async load.
    state.dir_entries.insert(
        String::new(),
        vec![FsEntry {
            name: "src".to_string(),
            full_path: "src".to_string(),
            is_dir: true,
            error: None,
        }],
    );
    state.rebuild_tree();
    state.file_tree.tree_focused = true;
    state.file_tree.tree_focus_index = 0; // "src"

    let _task = state.update(EditorMessage::ToggleDir("src".to_string()));
    // ToggleDir sets loading_dirs and dir_generations.
    assert!(state.loading_dirs.contains("src"));
    assert!(state.dir_generations.contains_key("src"));
    // ToggleDir does NOT set pending_enter_dir.
    assert_eq!(state.pending_enter_dir, None);
    // Focus is on "src".
    assert!(state.file_tree.tree_focused);
    assert_eq!(state.file_tree.tree_focus_index, 0);

    // Simulate DirExpanded completing with children.
    let dir_gen = *state.dir_generations.get("src").unwrap();
    let entries = vec![FsEntry {
        name: "main.rs".to_string(),
        full_path: "src/main.rs".to_string(),
        is_dir: false,
        error: None,
    }];
    let _task = state.update(EditorMessage::DirExpanded {
        dir_path: "src".to_string(),
        r#gen: dir_gen,
        entries: Ok(entries),
        quiet: false,
    });
    // Entries are now cached.
    assert!(state.dir_entries.contains_key("src"));
    assert_eq!(state.dir_entries["src"].len(), 1);
    // loading_dirs is cleared.
    assert!(!state.loading_dirs.contains("src"));
    // pending_enter_dir was never set.
    assert_eq!(state.pending_enter_dir, None);
    // visible_tree_nodes is correctly rebuilt (rebuild_tree was called).
    assert!(state.file_tree.visible_tree_nodes.len() >= 2);
    assert_eq!(state.file_tree.visible_tree_nodes[0].0, "src");
    assert_eq!(state.file_tree.visible_tree_nodes[1].0, "src/main.rs");
}

#[test]
fn test_toggle_dir_no_workspace_returns_none() {
    let mut state = EditorState::new();
    // Precondition: "src" is not yet in expanded_dirs before the call.
    assert!(!state.file_tree.expanded_dirs.contains("src"));
    // No workspace set — async load should return None (early return).
    let _task = state.update(EditorMessage::ToggleDir("src".to_string()));
    // expanded_dirs is modified (insert happens before the workspace guard),
    // but no async load was spawned since there's no workspace path.
    assert!(state.file_tree.expanded_dirs.contains("src"));
    assert_eq!(state.generation, 0);
    assert!(state.loading_dirs.is_empty());
    assert!(state.dir_generations.is_empty());
}

// ── Git status porcelain parsing tests ─────────────────────────

#[allow(clippy::too_many_lines)]
#[test]
fn test_parse_git_status_porcelain() {
    struct Case {
        /// Short label for failure messages.
        name: &'static str,
        /// Raw git status --porcelain output.
        input: &'static str,
        /// Expected entries: (path, Some(status)) asserts the file has that
        /// status; (path, None) asserts the file is absent from the map.
        /// An empty slice asserts the entire map is empty.
        expected: &'static [(&'static str, Option<GitFileStatus>)],
    }
    let cases: &[Case] = &[
        Case {
            name: "unstaged modified file",
            input: " M src/main.rs\n",
            expected: &[("src/main.rs", Some(GitFileStatus::Modified))],
        },
        Case {
            name: "staged added file",
            input: "A  new_file.rs\n",
            expected: &[("new_file.rs", Some(GitFileStatus::Added))],
        },
        Case {
            name: "untracked file",
            input: "?? new_file.rs\n",
            expected: &[("new_file.rs", Some(GitFileStatus::Added))],
        },
        Case {
            name: "staged and unstaged modified (MM)",
            input: "MM both.rs\n",
            expected: &[("both.rs", Some(GitFileStatus::Modified))],
        },
        Case {
            name: "staged added + unstaged modified (AM)",
            input: "AM partial.rs\n",
            expected: &[("partial.rs", Some(GitFileStatus::Modified))],
        },
        Case {
            name: "rename (old -> new)",
            input: "R  old.rs -> new.rs\n",
            expected: &[("new.rs", Some(GitFileStatus::Modified))],
        },
        Case {
            name: "rename with arrow in old path",
            input: "R  \"old -> name.rs\" -> \"new -> name.rs\"\n",
            expected: &[("new -> name.rs", Some(GitFileStatus::Modified))],
        },
        Case {
            name: "untracked directory (trailing slash stripped)",
            input: "?? new_dir/\n",
            expected: &[("new_dir", Some(GitFileStatus::Added))],
        },
        Case {
            name: "quoted path with spaces",
            input: " M \"path with spaces.rs\"\n",
            expected: &[("path with spaces.rs", Some(GitFileStatus::Modified))],
        },
        Case {
            name: "deleted file (unstaged) skipped",
            input: " D gone.rs\n",
            expected: &[],
        },
        Case {
            name: "deleted file (staged) skipped",
            input: "D  gone.rs\n",
            expected: &[],
        },
        Case {
            name: "clean (unrecognized status) not present",
            input: "   clean.rs\n",
            expected: &[("clean.rs", None)],
        },
        Case {
            name: "multiple entries same file — modified wins over added",
            input: "A  dup.rs\n M dup.rs\n",
            expected: &[("dup.rs", Some(GitFileStatus::Modified))],
        },
        Case {
            name: "multiple entries same file — added sticks",
            input: "?? dup.rs\nA  dup.rs\n",
            expected: &[("dup.rs", Some(GitFileStatus::Added))],
        },
        Case {
            name: "empty output",
            input: "",
            expected: &[],
        },
        Case {
            name: "mixed statuses",
            input: concat!(
                " M src/main.rs\n",
                "?? new_file.rs\n",
                "A  staged.rs\n",
                " D deleted.rs\n",
            ),
            expected: &[
                ("src/main.rs", Some(GitFileStatus::Modified)),
                ("new_file.rs", Some(GitFileStatus::Added)),
                ("staged.rs", Some(GitFileStatus::Added)),
                ("deleted.rs", None),
            ],
        },
    ];

    for case in cases {
        let map = parse_git_status_porcelain(case.input);

        if case.expected.is_empty() {
            assert!(
                map.is_empty(),
                "case '{}' (input={:?}): expected empty map, got {:#?}",
                case.name,
                case.input,
                map
            );
        } else {
            let expected_count = case.expected.iter().filter(|(_, s)| s.is_some()).count();
            assert_eq!(
                map.len(),
                expected_count,
                "case '{}' (input={:?}): map has unexpected entries",
                case.name,
                case.input,
            );
            for &(path, expected_status) in case.expected {
                match expected_status {
                    Some(status) => {
                        assert_eq!(
                            map.get(path),
                            Some(&status),
                            "case '{}' (input={:?}): path={:?}",
                            case.name,
                            case.input,
                            path
                        );
                    }
                    None => {
                        assert!(
                            !map.contains_key(path),
                            "case '{}' (input={:?}): path={:?} should be absent, got {:?}",
                            case.name,
                            case.input,
                            path,
                            map.get(path)
                        );
                    }
                }
            }
        }
    }
}

// ── Find/Replace tests ───────────────────────────────────────────

#[test]
#[allow(clippy::single_range_in_vec_init)]
fn test_is_find_bar_open_true_when_active() {
    let state = make_editor_with_find_state("fn hello() {}", "hello", vec![4..9], 0);
    assert!(state.is_find_bar_open());
}

#[test]
fn test_is_find_bar_open_false_when_closed() {
    let state = make_editor_with_single_tab("fn hello() {}");
    assert!(!state.is_find_bar_open());
}

#[test]
fn test_is_find_bar_open_no_tabs() {
    let state = EditorState::new();
    assert!(!state.is_find_bar_open());
}

#[allow(clippy::too_many_lines)]
#[test]
fn test_find_replace_auto_advance() {
    // Verifies cursor auto-advance after find_replace across five scenarios:
    // same-length replacement, shorter replacement, adjacent matches,
    // longer replacement (wrap-around), and no remaining matches.
    struct Case {
        name: &'static str,
        /// Initial buffer text.
        text: &'static str,
        /// Search query.
        query: &'static str,
        /// Replacement text.
        replace: &'static str,
        /// Pre-seeded match byte-range pairs.
        initial_matches: &'static [(usize, usize)],
        /// Expected text after replacement.
        expected_text: &'static str,
        /// Expected remaining match byte-range pairs.
        expected_matches: &'static [(usize, usize)],
        /// Expected cursor line after replacement.
        expected_cursor_line: usize,
        /// Expected cursor column after replacement.
        expected_cursor_col: usize,
        /// Expected current_match_idx after replacement.
        expected_current_match_idx: usize,
    }

    let cases: &[Case] = &[
        // Same-length replacement: "ab cd ab" → "xy cd ab".
        // After replacing first "ab" (0..2) with "xy" (len=2), remaining "ab"
        // at 6..8 is found by advancing past replace_end (= 0 + 2 = 2).
        Case {
            name: "same_length",
            text: "ab cd ab",
            query: "ab",
            replace: "xy",
            initial_matches: &[(0, 2), (6, 8)],
            expected_text: "xy cd ab",
            expected_matches: &[(6, 8)],
            expected_cursor_line: 0,
            expected_cursor_col: 6,
            expected_current_match_idx: 0,
        },
        // replace_end = 0 + 1 = 1. Remaining "aaa" at byte 6 in new text.
        Case {
            name: "shorter_replacement",
            text: "aaa bbb aaa",
            query: "aaa",
            replace: "a",
            initial_matches: &[(0, 3), (8, 11)],
            expected_text: "a bbb aaa",
            expected_matches: &[(6, 9)],
            expected_cursor_line: 0,
            expected_cursor_col: 6,
            expected_current_match_idx: 0,
        },
        // Adjacent matches: "aaaa" → "xaa".
        // Using range.end (= 2) as the advance point would incorrectly skip
        // the remaining match at 1..3 (1 >= 2 is false, so position() returns
        // None → wraps to 0, not 1). Using replace_end (= 1) correctly finds
        // the match at position 1.
        Case {
            name: "adjacent_matches",
            text: "aaaa",
            query: "aa",
            replace: "x",
            initial_matches: &[(0, 2), (2, 4)],
            expected_text: "xaa",
            expected_matches: &[(1, 3)],
            expected_cursor_line: 0,
            expected_cursor_col: 1,
            expected_current_match_idx: 0,
        },
        // Longer replacement: "ab" → "abc".
        // replace_end = 0 + 3 = 3. Remaining "ab" at 0..2 has start=0 < 3,
        // so position() returns None → wraps to index 0.
        Case {
            name: "longer_replacement",
            text: "ab",
            query: "ab",
            replace: "abc",
            initial_matches: &[(0, 2)],
            expected_text: "abc",
            expected_matches: &[(0, 2)],
            expected_cursor_line: 0,
            expected_cursor_col: 0,
            expected_current_match_idx: 0,
        },
        // No remaining matches: "ab" → "xy".
        // Matches is empty, current_match_idx resets to 0, cursor moves to
        // the end of the replacement (replace_end = 0 + 2 = 2).
        Case {
            name: "no_more_matches",
            text: "ab",
            query: "ab",
            replace: "xy",
            initial_matches: &[(0, 2)],
            expected_text: "xy",
            expected_matches: &[],
            expected_cursor_line: 0,
            expected_cursor_col: 2,
            expected_current_match_idx: 0,
        },
    ];

    let path = "/test.rs".to_string();
    for c in cases {
        let mut state = make_editor_with_single_tab(c.text);
        if let Some(tab) = state.tab_contents.get_mut(&path) {
            tab.find_replace_state = Some(FindReplaceState {
                query: c.query.to_string(),
                replace: c.replace.to_string(),
                matches: c.initial_matches.iter().map(|&(s, e)| s..e).collect(),
                current_match_idx: 0,
                case_sensitive: true,
            });
        }
        let _ = state.update(EditorMessage::FindReplace);
        let tab = state.tab_contents.get(&path).unwrap();
        let frs = tab.find_replace_state.as_ref().unwrap();

        assert_eq!(tab.content.text(), c.expected_text, "{}: text", c.name);
        assert_eq!(
            frs.matches.len(),
            c.expected_matches.len(),
            "{}: match count",
            c.name,
        );
        for (i, &(s, e)) in c.expected_matches.iter().enumerate() {
            assert_eq!(frs.matches[i], s..e, "{}: match {i} range", c.name);
        }
        let cursor = tab.content.cursor();
        assert_eq!(
            cursor.line, c.expected_cursor_line,
            "{}: cursor line",
            c.name
        );
        assert_eq!(
            cursor.column, c.expected_cursor_col,
            "{}: cursor col",
            c.name
        );
        assert_eq!(
            frs.current_match_idx, c.expected_current_match_idx,
            "{}: current_match_idx",
            c.name,
        );
    }
}

/// Helper to create an [`EditorState`] with a single tab at `/test.rs`
/// that has an active [`FindReplaceState`].
fn make_editor_with_find_state(
    text: &str,
    query: &str,
    matches: Vec<Range<usize>>,
    current_match_idx: usize,
) -> EditorState {
    let mut state = EditorState::new();
    state.tabs.push(Tab {
        path: "/test.rs".to_string(),
        file_name: "test.rs".to_string(),
        is_dirty: false,
        has_trailing_newline: true,
        line_ending: LineEnding::Lf,
    });
    state.active_tab_index = 0;
    state.tab_contents.insert(
        "/test.rs".to_string(),
        TabData {
            content: EditorBuffer::with_text(text, None),
            undo_stack: RefCell::new(UndoStack::new()),
            find_replace_state: Some(FindReplaceState {
                query: query.to_string(),
                replace: String::new(),
                matches,
                current_match_idx,
                case_sensitive: true,
            }),
            saved_text_hash: 0,
        },
    );
    state
}

#[test]
fn test_navigate_find_match_wraps_next() {
    let mut state = make_editor_with_find_state("a b c", " ", vec![1..2, 3..4], 0);

    // Navigate next from index 0 → 1.
    let _ = state.navigate_find_match(FindDirection::Next);
    let frs = state.tab_contents.get("/test.rs").unwrap();
    let s = frs.find_replace_state.as_ref().unwrap();
    assert_eq!(s.current_match_idx, 1);

    // Navigate next from index 1 → wraps to 0.
    let _ = state.navigate_find_match(FindDirection::Next);
    let s = state.tab_contents.get("/test.rs").unwrap();
    let s = s.find_replace_state.as_ref().unwrap();
    assert_eq!(s.current_match_idx, 0);
}

#[test]
fn test_navigate_find_match_wraps_prev() {
    let mut state = make_editor_with_find_state("a b c", " ", vec![1..2, 3..4], 0);

    // Navigate prev from index 0 → wraps to 1 (last).
    let _ = state.navigate_find_match(FindDirection::Prev);
    let s = state.tab_contents.get("/test.rs").unwrap();
    let s = s.find_replace_state.as_ref().unwrap();
    assert_eq!(s.current_match_idx, 1);

    // Navigate prev from index 1 → 0.
    let _ = state.navigate_find_match(FindDirection::Prev);
    let s = state.tab_contents.get("/test.rs").unwrap();
    let s = s.find_replace_state.as_ref().unwrap();
    assert_eq!(s.current_match_idx, 0);
}

#[test]
fn test_navigate_find_match_no_matches() {
    let mut state = make_editor_with_find_state("no matches", "zzz", vec![], 0);

    // Navigating with no matches should not crash.
    let _ = state.navigate_find_match(FindDirection::Next);
    let _ = state.navigate_find_match(FindDirection::Prev);
    let s = state.tab_contents.get("/test.rs").unwrap();
    let s = s.find_replace_state.as_ref().unwrap();
    assert_eq!(s.current_match_idx, 0);
}

#[test]
fn test_navigate_find_match_only_affects_find_tab() {
    // Tab without find state should not be affected.
    let mut state = make_editor_with_single_tab("hello");

    // Should not panic.
    let _ = state.navigate_find_match(FindDirection::Next);
    let _ = state.navigate_find_match(FindDirection::Prev);
}

// ── Tree arrow-key navigation tests ─────────────────────────────

#[allow(clippy::too_many_lines)]
#[test]
fn test_tree_nav_left_right() {
    struct Case {
        name: &'static str,
        msg: EditorMessage,
        start_idx: usize,
        /// Pre-expand "src" before sending the message
        pre_expand_src: bool,
        /// Set selected_file to Some("Cargo.toml") before sending the message
        pre_select_file: bool,
        /// Expected focus index after the message
        expected_idx: usize,
        /// Additional per-case assertions beyond focus index
        check: Option<fn(&EditorState, name: &str)>,
    }
    let cases: &[Case] = &[
        Case {
            name: "left_on_expanded_dir_collapses",
            msg: EditorMessage::TreeNavLeft,
            start_idx: 0,
            pre_expand_src: true,
            pre_select_file: false,
            expected_idx: 0,
            check: Some(|s, name| {
                assert!(!s.file_tree.expanded_dirs.contains("src"), "case: {name}");
            }),
        },
        Case {
            name: "left_on_file_navigates_to_parent",
            msg: EditorMessage::TreeNavLeft,
            start_idx: 1,
            pre_expand_src: true,
            pre_select_file: false,
            expected_idx: 0,
            check: Some(|s, name| {
                assert_eq!(s.file_tree.visible_tree_nodes[0].0, "src", "case: {name}");
            }),
        },
        Case {
            name: "left_on_root_collapsed_dir_noop",
            msg: EditorMessage::TreeNavLeft,
            start_idx: 0,
            pre_expand_src: false,
            pre_select_file: false,
            expected_idx: 0,
            check: None,
        },
        Case {
            name: "left_on_root_file_noop",
            msg: EditorMessage::TreeNavLeft,
            start_idx: 1,
            pre_expand_src: false,
            pre_select_file: false,
            expected_idx: 1,
            check: None,
        },
        Case {
            name: "right_on_collapsed_dir_expands_and_advances",
            msg: EditorMessage::TreeNavRight,
            start_idx: 0,
            pre_expand_src: false,
            pre_select_file: true,
            expected_idx: 1,
            check: Some(|s, name| {
                assert!(s.file_tree.expanded_dirs.contains("src"), "case: {name}");
                assert!(s.selected_file.is_none(), "case: {name}");
                assert_eq!(s.file_tree.visible_tree_nodes[1].0, "src/main.rs");
            }),
        },
        Case {
            name: "right_on_expanded_dir_moves_to_first_child",
            msg: EditorMessage::TreeNavRight,
            start_idx: 0,
            pre_expand_src: true,
            pre_select_file: false,
            expected_idx: 1,
            check: Some(|s, name| {
                assert_eq!(
                    s.file_tree.visible_tree_nodes[1].0, "src/main.rs",
                    "case: {name}"
                );
            }),
        },
        Case {
            name: "right_on_file_noop",
            msg: EditorMessage::TreeNavRight,
            start_idx: 1,
            pre_expand_src: false,
            pre_select_file: false,
            expected_idx: 1,
            check: None,
        },
    ];
    for case in cases {
        let mut state = make_editor_with_tree();
        if case.pre_expand_src {
            state.file_tree.expanded_dirs.insert("src".to_string());
            state.file_tree.nodes =
                build_hierarchical_tree(&state.dir_entries, &state.file_tree.expanded_dirs, "");
            state.file_tree.rebuild_visible();
        }
        state.file_tree.tree_focused = true;
        state.file_tree.tree_focus_index = case.start_idx;
        if case.pre_select_file {
            state.selected_file = Some("Cargo.toml".to_string());
        }
        let _ = state.update(case.msg.clone());
        assert_eq!(
            state.file_tree.tree_focus_index, case.expected_idx,
            "case: {}",
            case.name
        );
        if let Some(check) = case.check {
            check(&state, case.name);
        }
    }
}

// ── Click-to-select focus index tests ────────────────────────────

#[test]
fn test_toggle_dir_sets_tree_focus_index() {
    let mut state = make_editor_with_tree();
    // Select a file first so we can verify it gets cleared.
    state.selected_file = Some("Cargo.toml".to_string());
    let _ = state.update(EditorMessage::ToggleDir("src".to_string()));
    // ToggleDir should set tree_focus_index to "src"'s position
    assert!(state.file_tree.tree_focused);
    assert_eq!(state.file_tree.tree_focus_index, 0);
    assert_eq!(state.file_tree.visible_tree_nodes[0].0, "src");
    assert!(
        state.selected_file.is_none(),
        "ToggleDir should clear selected_file"
    );
}

#[test]
fn test_select_file_sets_tree_focus_index() {
    let mut state = make_editor_with_tree();
    // Expand "src" so "src/main.rs" is visible in the flat list.
    state.file_tree.expanded_dirs.insert("src".to_string());
    state.file_tree.nodes =
        build_hierarchical_tree(&state.dir_entries, &state.file_tree.expanded_dirs, "");
    state.file_tree.rebuild_visible();
    state.file_tree.tree_focused = true;
    let _ = state.update(EditorMessage::SelectFile("src/main.rs".to_string()));
    // SelectFile keeps tree_focused and remembers focus index.
    assert!(state.file_tree.tree_focused);
    // tree_focus_index should point to "src/main.rs" for Ctrl+B re-focus.
    assert_eq!(
        state.file_tree.visible_tree_nodes[state.file_tree.tree_focus_index].0,
        "src/main.rs"
    );
}

#[test]
fn test_select_file_sets_tree_focused_when_not_focused() {
    // When tree_focused starts false, clicking a file should set it true.
    let mut state = make_editor_with_tree();
    state.file_tree.expanded_dirs.insert("src".to_string());
    state.file_tree.nodes =
        build_hierarchical_tree(&state.dir_entries, &state.file_tree.expanded_dirs, "");
    state.file_tree.rebuild_visible();
    state.file_tree.tree_focused = false;
    let _ = state.update(EditorMessage::SelectFile("src/main.rs".to_string()));
    assert!(
        state.file_tree.tree_focused,
        "SelectFile should set tree_focused to true"
    );
}

// ── Focus gating and find/replace cursor tests ───────────────────

fn make_editor_with_single_tab(text: &str) -> EditorState {
    let mut state = EditorState::new();
    state.tabs.push(Tab {
        path: "/test.rs".to_string(),
        file_name: "test.rs".to_string(),
        is_dirty: false,
        has_trailing_newline: true,
        line_ending: LineEnding::Lf,
    });
    state.active_tab_index = 0;
    state.tab_contents.insert(
        "/test.rs".to_string(),
        TabData {
            content: EditorBuffer::with_text(text, None),
            undo_stack: RefCell::new(UndoStack::new()),
            find_replace_state: None,
            saved_text_hash: hash_text(text),
        },
    );
    state
}

#[test]
fn test_undo_noop_when_quick_open_active() {
    let mut state = make_editor_with_single_tab("hello");
    let path = "/test.rs".to_string();
    if let Some(tab_data) = state.tab_contents.get_mut(&path) {
        tab_data
            .undo_stack
            .borrow_mut()
            .snap_before_edit(&tab_data.content);
        tab_data.content.perform_action(EditorAction::Insert('!'));
    }
    state.active_modal = Some(ModalKind::QuickOpen(QuickOpenState {
        filter: String::new(),
        selected_index: 0,
        results: Vec::new(),
    }));
    let _ = state.update(EditorMessage::Undo);
    assert_eq!(
        state.tab_contents.get(&path).unwrap().content.text(),
        "!hello"
    );
}

#[test]
fn test_refresh_file_tree_noop_when_quick_open_active() {
    let mut state = make_editor_with_tree();
    // Pre-populate dir_generations so we can detect new entries.
    let initial_gen_count = state.dir_generations.len();
    assert!(state.selected_workspace_path.is_some());

    // Activate a modal overlay (QuickOpen).
    state.active_modal = Some(ModalKind::QuickOpen(QuickOpenState {
        filter: String::new(),
        selected_index: 0,
        results: Vec::new(),
    }));

    // RefreshFileTree should be suppressed — no new dir generations added.
    let _ = state.update(EditorMessage::RefreshFileTree);
    assert_eq!(
        state.dir_generations.len(),
        initial_gen_count,
        "RefreshFileTree must not spawn directory refreshes when a modal overlay is active"
    );
}

#[test]
fn test_tree_focus_toggled_noop_during_modal_overlay() {
    let mut state = make_editor_with_tree();
    // First toggle tree focus ON.
    let _ = state.update(EditorMessage::TreeFocusToggled);
    assert!(state.file_tree.tree_focused);

    // Activate a modal overlay (QuickOpen).
    state.active_modal = Some(ModalKind::QuickOpen(QuickOpenState {
        filter: String::new(),
        selected_index: 0,
        results: Vec::new(),
    }));

    // TreeFocusToggled should be suppressed — focus stays ON.
    let _ = state.update(EditorMessage::TreeFocusToggled);
    assert!(
        state.file_tree.tree_focused,
        "TreeFocusToggled must not toggle focus when a modal overlay is active"
    );
}

#[test]
fn test_tree_nav_suppressed_during_goto_line_overlay() {
    let mut state = make_editor_with_tree();
    state.file_tree.expanded_dirs.insert("src".to_string());
    state.file_tree.nodes =
        build_hierarchical_tree(&state.dir_entries, &state.file_tree.expanded_dirs, "");
    state.file_tree.rebuild_visible();
    state.file_tree.tree_focused = true;
    state.file_tree.tree_focus_index = 0; // "src"

    // Activate a non-search modal overlay (GotoLine).
    state.active_modal = Some(ModalKind::GotoLine(String::new()));

    let prev_focus = state.file_tree.tree_focus_index;
    // Up/Down/Enter/Left/Right — assert tree_focus_index unchanged.
    let nav_msgs: &[EditorMessage] = &[
        EditorMessage::TreeNavUp,
        EditorMessage::TreeNavDown,
        EditorMessage::TreeNavEnter,
        EditorMessage::TreeNavLeft,
        EditorMessage::TreeNavRight,
    ];
    for msg in nav_msgs {
        let _ = state.update(msg.clone());
        assert_eq!(
            state.file_tree.tree_focus_index, prev_focus,
            "{msg:?} should be suppressed during GotoLine overlay"
        );
    }

    // TreeFocusToggled is handled separately because it toggles
    // tree_focused, not tree_focus_index.
    let _ = state.update(EditorMessage::TreeFocusToggled);
    assert!(
        state.file_tree.tree_focused,
        "TreeFocusToggled should be suppressed during GotoLine overlay"
    );
}

#[test]
fn test_find_replace_all_preserves_cursor() {
    let mut state = make_editor_with_single_tab("ab cd ab");
    let path = "/test.rs".to_string();
    if let Some(tab_data) = state.tab_contents.get_mut(&path) {
        tab_data.content.move_to(0, 5);
        tab_data.find_replace_state = Some(FindReplaceState {
            query: "ab".to_string(),
            replace: "xy".to_string(),
            matches: vec![0..2, 6..8],
            current_match_idx: 0,
            case_sensitive: true,
        });
    }
    let _ = state.update(EditorMessage::FindReplaceAll);
    let cursor = state.tab_contents.get(&path).unwrap().content.cursor();
    assert_eq!(cursor.line, 0);
    assert_eq!(cursor.column, 5);
}

#[test]
fn test_quick_open_toggle_blocked_when_goto_line_open() {
    let mut state = make_editor_with_single_tab("hello");
    state.active_modal = Some(ModalKind::GotoLine(String::new()));
    let _ = state.update(EditorMessage::QuickOpenToggle);
    assert!(!matches!(state.active_modal, Some(ModalKind::QuickOpen(_))));
}

#[test]
fn test_quick_open_toggle_closes_when_already_open() {
    let mut state = make_editor_with_single_tab("hello");
    state.active_modal = Some(ModalKind::QuickOpen(QuickOpenState {
        filter: "foo".to_string(),
        selected_index: 0,
        results: Vec::new(),
    }));
    let _ = state.update(EditorMessage::QuickOpenToggle);
    assert!(state.active_modal.is_none());
}

#[test]
fn test_global_search_toggle_blocked_when_quick_open_open() {
    let mut state = make_editor_with_single_tab("hello");
    state.selected_workspace_name = Some("ws".to_string());
    state.selected_workspace_path = Some("/tmp/ws".to_string());
    state.active_modal = Some(ModalKind::QuickOpen(QuickOpenState {
        filter: String::new(),
        selected_index: 0,
        results: Vec::new(),
    }));
    let _ = state.update(EditorMessage::GlobalSearchToggle);
    assert!(!matches!(
        state.active_modal,
        Some(ModalKind::GlobalSearch(_))
    ));
}

// ── Inline rename tests ────────────────────────────────────

#[test]
fn test_rename_request_sets_target() {
    let mut state = make_editor_with_tree();
    state.selected_workspace_path = Some("/tmp".to_string());
    let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
    assert!(matches!(state.active_modal, Some(ModalKind::Rename(_))));
    let rt = match state.active_modal {
        Some(ModalKind::Rename(ref rt)) => rt.clone(),
        _ => panic!("expected Rename modal"),
    };
    assert_eq!(rt.path, "Cargo.toml");
    assert_eq!(rt.input_text, "Cargo.toml");
    assert!(!rt.is_dir);
}

#[test]
fn test_rename_request_on_directory_sets_is_dir() {
    // Use a real temp directory so Path::is_dir() returns true.
    let tmp_dir = tempfile::tempdir().unwrap();
    let dir_path = tmp_dir.path().join("src");
    std::fs::create_dir(&dir_path).unwrap();
    let mut state = EditorState::new();
    state.selected_workspace_path = Some(tmp_dir.path().to_string_lossy().to_string());
    state.dir_entries.insert(
        String::new(),
        vec![FsEntry {
            name: "src".to_string(),
            full_path: "src".to_string(),
            is_dir: true,
            error: None,
        }],
    );
    let _ = state.update(EditorMessage::RenameRequested("src".into()));
    assert!(matches!(state.active_modal, Some(ModalKind::Rename(_))));
    let rt = match state.active_modal {
        Some(ModalKind::Rename(ref rt)) => rt.clone(),
        _ => panic!("expected Rename modal"),
    };
    assert_eq!(rt.path, "src");
    assert_eq!(rt.input_text, "src");
    assert!(rt.is_dir);
}

#[test]
fn test_rename_request_on_root_dir_rejected() {
    let mut state = make_editor_with_tree();
    state.selected_workspace_path = Some("/tmp".to_string());
    let _ = state.update(EditorMessage::RenameRequested(String::new()));
    assert!(
        state.active_modal.is_none() || !matches!(state.active_modal, Some(ModalKind::Rename(_)))
    );
}

#[test]
fn test_rename_input_updates_text_and_clears_error() {
    let mut state = make_editor_with_tree();
    state.selected_workspace_path = Some("/tmp".to_string());
    let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
    // Simulate a validation error
    if let Some(ModalKind::Rename(ref mut rt)) = state.active_modal {
        rt.error = Some("bad".into());
    }
    // Type new text
    let _ = state.update(EditorMessage::RenameInput("new_name".into()));
    if let Some(ModalKind::Rename(ref rt)) = state.active_modal {
        assert_eq!(rt.input_text, "new_name");
        // Error should be cleared when user types
        assert!(rt.error.is_none());
    } else {
        panic!("expected Rename modal");
    }
}

#[test]
fn test_rename_cancel_clears_target() {
    let mut state = make_editor_with_tree();
    state.selected_workspace_path = Some("/tmp".to_string());
    let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
    assert!(matches!(state.active_modal, Some(ModalKind::Rename(_))));
    let _ = state.update(EditorMessage::RenameCancel);
    assert!(state.active_modal.is_none());
}

#[test]
fn test_escape_cancels_rename() {
    let mut state = make_editor_with_tree();
    state.selected_workspace_path = Some("/tmp".to_string());
    let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
    assert!(matches!(state.active_modal, Some(ModalKind::Rename(_))));
    let _ = state.update(EditorMessage::Escape);
    assert!(state.active_modal.is_none());
}

#[test]
fn test_tree_nav_suppressed_during_rename() {
    let mut state = make_editor_with_tree();
    state.selected_workspace_path = Some("/tmp".to_string());
    // Expand "src" so TreeNavEnter/TreeNavLeft/TreeNavRight have targets.
    state.file_tree.expanded_dirs.insert("src".to_string());
    state.file_tree.nodes =
        build_hierarchical_tree(&state.dir_entries, &state.file_tree.expanded_dirs, "");
    state.file_tree.rebuild_visible();
    state.file_tree.tree_focused = true;
    // Focus on "src" so TreeNavLeft (collapse) and TreeNavRight (expand)
    // have an effect when not suppressed.
    state.file_tree.tree_focus_index = 0; // "src"

    let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
    let prev_focus = state.file_tree.tree_focus_index;
    // All 6 tree-navigation messages must be suppressed during rename.
    let nav_msgs: &[EditorMessage] = &[
        EditorMessage::TreeNavUp,
        EditorMessage::TreeNavDown,
        EditorMessage::TreeNavEnter,
        EditorMessage::TreeNavLeft,
        EditorMessage::TreeNavRight,
        EditorMessage::TreeFocusToggled,
    ];
    for msg in nav_msgs {
        let _ = state.update(msg.clone());
        assert_eq!(
            state.file_tree.tree_focus_index, prev_focus,
            "{msg:?} should be suppressed during rename"
        );
    }
    // After the rename is cancelled, navigation should work again.
    let _ = state.update(EditorMessage::RenameCancel);
    let _ = state.update(EditorMessage::TreeNavDown);
    // Focus should have moved now that rename is gone.
    assert_ne!(state.file_tree.tree_focus_index, prev_focus);
}

#[test]
fn test_rename_mutual_exclusion_with_new_item() {
    let mut state = make_editor_with_tree();
    state.selected_workspace_path = Some("/tmp".to_string());

    // Start rename, then NewFileRequested should cancel it.
    let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
    assert!(matches!(state.active_modal, Some(ModalKind::Rename(_))));
    let _ = state.update(EditorMessage::NewFileRequested("src".into()));
    assert!(
        state.active_modal.is_none() || !matches!(state.active_modal, Some(ModalKind::Rename(_)))
    );
    assert!(matches!(state.active_modal, Some(ModalKind::NewItem(_))));

    // Start new item again — and confirm rename cancels new_item.
    let _ = state.update(EditorMessage::NewFileRequested(String::new()));
    assert!(matches!(state.active_modal, Some(ModalKind::NewItem(_))));
    let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
    assert!(matches!(state.active_modal, Some(ModalKind::Rename(_))));
}

// ── Rename validation tests ────────────────────────────────

/// Helper: set up state for rename validation tests.
fn setup_rename_state(state: &mut EditorState, input_text: &str) {
    state.selected_workspace_path = Some("/tmp".to_string());
    let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
    if let Some(ModalKind::Rename(ref mut rt)) = state.active_modal {
        rt.input_text = input_text.to_string();
    }
}

/// Helper: set up a rename with `input`, submit it, and assert that
/// the resulting error equals `expected`.
fn assert_rename_rejects(input: &str, expected: Option<&'static str>) {
    let mut state = make_editor_with_tree();
    setup_rename_state(&mut state, input);
    let _ = state.update(EditorMessage::RenameSubmit);
    let err = match &state.active_modal {
        Some(ModalKind::Rename(rt)) => rt.error.as_deref(),
        _ => None,
    };
    assert_eq!(err, expected, "rejection of {input:?}");
}

#[test]
fn test_rename_validation() {
    struct Case {
        input: &'static str,
        expected: Option<&'static str>,
    }
    let cases: &[Case] = &[
        // Empty / whitespace-only
        Case {
            input: "   ",
            expected: Some("Name cannot be empty"),
        },
        // Path separators
        Case {
            input: "foo/bar.rs",
            expected: Some("Name cannot contain path separators"),
        },
        Case {
            input: "foo\\bar.rs",
            expected: Some("Name cannot contain path separators"),
        },
        Case {
            input: "foo\0bar.rs",
            expected: Some("Name cannot contain path separators"),
        },
        // Dot / dot-dot
        Case {
            input: ".",
            expected: Some("Invalid name"),
        },
        Case {
            input: "..",
            expected: Some("Invalid name"),
        },
    ];
    for case in cases {
        assert_rename_rejects(case.input, case.expected);
    }
}

#[cfg(target_os = "windows")]
#[test]
fn test_rename_validation_os_reserved_names() {
    let reserved = ["con", "NUL", "prn", "AUX", "com1", "lpt3"];
    for name in &reserved {
        assert_rename_rejects(name, Some("Name is reserved by the operating system"));
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
        rt.input_text = "existing.txt".to_string();
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
        rt.input_text = "lib.rs".to_string();
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

// ── Click-outside cancel tests (consolidated) ───────────────

#[test]
fn test_rename_cancelled_by_tree_click() {
    // Both ToggleDir and SelectFile should cancel a pending rename.
    let triggers: &[EditorMessage] = &[
        EditorMessage::ToggleDir("src".into()),
        EditorMessage::SelectFile("src/main.rs".into()),
    ];
    for trigger in triggers {
        let mut state = make_editor_with_tree();
        state.selected_workspace_path = Some("/tmp".to_string());
        let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
        assert!(
            matches!(state.active_modal, Some(ModalKind::Rename(_))),
            "rename should be active before {trigger:?}"
        );
        let _ = state.update(trigger.clone());
        assert!(
            state.active_modal.is_none()
                || !matches!(state.active_modal, Some(ModalKind::Rename(_))),
            "rename should be cancelled by {trigger:?}"
        );
    }
}

#[test]
fn test_rename_mutual_exclusion_cancelled_by_other_modals() {
    // Starting a different modal operation should cancel an active rename.
    // Each test case carries a message to dispatch and a check closure
    // that verifies the expected modal state after the message fires.
    struct Case {
        msg: EditorMessage,
        /// Assert the expected modal state after the message fires.
        check: fn(&EditorState),
    }
    let cases: &[Case] = &[
        Case {
            msg: EditorMessage::NewFileRequested("src".into()),
            check: |s| assert!(matches!(s.active_modal, Some(ModalKind::NewItem(_)))),
        },
        Case {
            msg: EditorMessage::NewDirectoryRequested("src".into()),
            check: |s| assert!(matches!(s.active_modal, Some(ModalKind::NewItem(_)))),
        },
        Case {
            msg: EditorMessage::DeleteFileRequested("other.rs".into()),
            check: |s| {
                assert!(matches!(s.active_modal, Some(ModalKind::DeleteConfirm(_))));
                if let Some(ModalKind::DeleteConfirm(ref target)) = s.active_modal {
                    assert_eq!(target.path, "other.rs");
                }
            },
        },
        Case {
            msg: EditorMessage::DeleteDirectoryRequested("src".into()),
            check: |s| {
                assert!(matches!(s.active_modal, Some(ModalKind::DeleteConfirm(_))));
                if let Some(ModalKind::DeleteConfirm(ref target)) = s.active_modal {
                    assert_eq!(target.path, "src");
                }
            },
        },
    ];
    for case in cases {
        let mut state = make_editor_with_tree();
        state.selected_workspace_path = Some("/tmp".to_string());

        // Start rename.
        let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
        assert!(
            matches!(state.active_modal, Some(ModalKind::Rename(_))),
            "case {:?}",
            case.msg
        );

        // Fire the competing modal message.
        let _ = state.update(case.msg.clone());
        assert!(
            !matches!(state.active_modal, Some(ModalKind::Rename(_))),
            "rename should be cancelled by {:?}",
            case.msg
        );
        (case.check)(&state);
    }
}

// ── rekey helpers ──────────────────────────────────────────

#[test]
fn test_rekey_keys_empty() {
    let pairs = rekey_keys("old/", "new/", Vec::<String>::new());
    assert!(pairs.is_empty());
}

#[test]
fn test_rekey_keys_no_match() {
    let keys = vec!["a".to_string(), "b".to_string()];
    let pairs = rekey_keys("old/", "new/", keys);
    assert!(pairs.is_empty());
}

#[test]
fn test_rekey_keys_some_match() {
    let keys = vec![
        "old/foo".to_string(),
        "other".to_string(),
        "old/bar/baz".to_string(),
    ];
    let mut pairs = rekey_keys("old/", "new", keys);
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(pairs.len(), 2);
    assert_eq!(
        pairs[0],
        ("old/bar/baz".to_string(), "new/bar/baz".to_string())
    );
    assert_eq!(pairs[1], ("old/foo".to_string(), "new/foo".to_string()));
}

#[test]
fn test_rekey_keys_exact_prefix() {
    let keys = vec!["dir".to_string()];
    let pairs = rekey_keys("dir", "newdir", keys);
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0], ("dir".to_string(), "newdir".to_string()));
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
        vec![FsEntry {
            name: "helper.rs".to_string(),
            full_path: "src/subdir/helper.rs".to_string(),
            is_dir: false,
            error: None,
        }],
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
        dir_entries: Ok(vec![FsEntry {
            name: "lib".to_string(),
            full_path: "lib".to_string(),
            is_dir: true,
            error: None,
        }]),
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

/// Creates an [`EditorState`] with `count` tabs, each with a unique path and
/// file name. The active tab is set to `active`. The caller must ensure
/// `active < count`.
fn make_editor_with_tabs(count: usize, active: usize) -> EditorState {
    assert!(
        active < count,
        "active tab index must be less than tab count"
    );
    let mut state = EditorState::new();
    for i in 0..count {
        state.tabs.push(Tab {
            path: format!("/tmp/test_{i}.rs"),
            file_name: format!("test_{i}.rs"),
            is_dirty: false,
            has_trailing_newline: true,
            line_ending: LineEnding::Lf,
        });
        state.tab_contents.insert(
            format!("/tmp/test_{i}.rs"),
            TabData {
                content: EditorBuffer::with_text("", None),
                undo_stack: RefCell::new(UndoStack::new()),
                find_replace_state: None,
                saved_text_hash: 0,
            },
        );
    }
    state.active_tab_index = active;
    state
}

#[test]
fn test_switch_tab_relative() {
    struct Case {
        name: &'static str,
        tabs: usize,
        start: usize,
        direction: TabDirection,
        expected: usize,
    }
    let cases: &[Case] = &[
        Case {
            name: "single tab next",
            tabs: 1,
            start: 0,
            direction: TabDirection::Next,
            expected: 0,
        },
        Case {
            name: "single tab prev",
            tabs: 1,
            start: 0,
            direction: TabDirection::Prev,
            expected: 0,
        },
        Case {
            name: "next wraps to first",
            tabs: 3,
            start: 2,
            direction: TabDirection::Next,
            expected: 0,
        },
        Case {
            name: "prev wraps to last",
            tabs: 3,
            start: 0,
            direction: TabDirection::Prev,
            expected: 2,
        },
        Case {
            name: "middle next",
            tabs: 3,
            start: 1,
            direction: TabDirection::Next,
            expected: 2,
        },
        Case {
            name: "middle prev",
            tabs: 3,
            start: 1,
            direction: TabDirection::Prev,
            expected: 0,
        },
    ];
    for case in cases {
        let mut state = make_editor_with_tabs(case.tabs, case.start);
        let _ = state.switch_tab_relative(case.direction);
        assert_eq!(state.active_tab_index, case.expected, "{}", case.name);
    }
}

#[test]
fn test_switch_tab_relative_two_tabs() {
    // With exactly two tabs, Next and Prev toggle between them.
    let mut state = make_editor_with_tabs(2, 0);
    let _ = state.switch_tab_relative(TabDirection::Next);
    assert_eq!(state.active_tab_index, 1);

    let _ = state.switch_tab_relative(TabDirection::Next);
    assert_eq!(state.active_tab_index, 0);

    let _ = state.switch_tab_relative(TabDirection::Prev);
    assert_eq!(state.active_tab_index, 1);
}
