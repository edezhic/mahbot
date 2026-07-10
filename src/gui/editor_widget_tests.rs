use super::*;

#[test]
fn test_new_empty() {
    let buf = EditorBuffer::new();
    assert_eq!(buf.text(), "");
    assert_eq!(buf.line_count(), 1);
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 0);
    assert_eq!(cursor.column, 0);
    assert!(cursor.selection.is_none());
}

#[test]
fn test_with_text() {
    let buf = EditorBuffer::with_text("hello\nworld", None);
    assert_eq!(buf.text(), "hello\nworld");
    assert_eq!(buf.line_count(), 2);
    assert_eq!(buf.line(0).as_deref(), Some("hello"));
    assert_eq!(buf.line(1).as_deref(), Some("world"));
}

#[test]
fn test_line_out_of_bounds() {
    let buf = EditorBuffer::with_text("hello", None);
    assert!(buf.line(1).is_none());
    assert!(buf.line(usize::MAX).is_none());
}

#[test]
fn test_cursor_move_to() {
    let buf = EditorBuffer::with_text("hello\nworld", None);
    buf.move_to(1, 2);
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 1);
    assert_eq!(cursor.column, 2);
    assert!(cursor.selection.is_none());
}

#[test]
fn test_cursor_move_to_beyond_end() {
    let buf = EditorBuffer::with_text("hello", None);
    buf.move_to(999, 999);
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 0);
    assert_eq!(cursor.column, 5);
}

#[test]
fn test_select_all_and_selection() {
    let buf = EditorBuffer::with_text("hello\nworld", None);
    buf.select_all();
    let cursor = buf.cursor();
    assert!(cursor.selection.is_some());
    let sel = buf.selection();
    assert_eq!(sel, Some("hello\nworld".to_string()));
}

#[test]
fn test_no_selection() {
    let buf = EditorBuffer::with_text("hello", None);
    assert!(buf.selection().is_none());
}

#[test]
fn test_set_text() {
    let buf = EditorBuffer::with_text("hello", None);
    buf.set_text("world");
    assert_eq!(buf.text(), "world");
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 0);
    assert_eq!(cursor.column, 0);
}

#[test]
fn test_insert_character() {
    let buf = EditorBuffer::with_text("helo", None);
    buf.move_to(0, 2);
    buf.perform_action(EditorAction::Insert('l'));
    assert_eq!(buf.text(), "hello");
}

#[test]
fn test_insert_at_end() {
    let buf = EditorBuffer::with_text("hello", None);
    buf.move_to(0, 5);
    buf.perform_action(EditorAction::Insert('!'));
    assert_eq!(buf.text(), "hello!");
}

#[test]
fn test_enter() {
    let buf = EditorBuffer::with_text("hello world", None);
    buf.move_to(0, 5);
    buf.perform_action(EditorAction::Enter);
    assert_eq!(buf.text(), "hello\n world");
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 1);
    // "hello world" has no leading whitespace, so auto-indent produces
    // column 0 (empty indent on the new line).
    assert_eq!(cursor.column, 0);
}

#[test]
fn test_enter_at_end_of_highlighted_file_with_trailing_newline() {
    // Regression test: Enter at end of a highlighted file
    // that *has* a trailing newline must place cursor on a new blank line,
    // not jump to the start of the last content line.
    let buf = EditorBuffer::with_text("fn main() {}\n", Some(HighlightLanguage::Rust));
    // Buffer should have a trailing empty sentinel line (2 lines).
    assert_eq!(buf.line_count(), 2);
    let cursor_before = buf.cursor();
    assert_eq!(cursor_before.line, 0);
    assert_eq!(cursor_before.column, 0);
    // Move cursor to end of the content line (right after '}').
    let content_len = "fn main() {}".chars().count();
    buf.move_to(0, content_len);
    buf.perform_action(EditorAction::Enter);
    // Text should be: original line + inserted newline + trailing sentinel.
    assert_eq!(buf.text(), "fn main() {}\n\n");
    let cursor = buf.cursor();
    // Cursor must be on the newly created blank line (line 1).
    assert_eq!(cursor.line, 1);
    assert_eq!(cursor.column, 0);
}

#[test]
fn test_enter_at_end_of_highlighted_file_no_trailing_newline() {
    // Same as above but the file has *no* trailing newline — the bug
    // would clamp cursor to (line=0, col=0) instead of (line=1, col=0).
    let buf = EditorBuffer::with_text("fn main() {}", Some(HighlightLanguage::Rust));
    assert_eq!(buf.line_count(), 1);
    let content_len = "fn main() {}".chars().count();
    buf.move_to(0, content_len);
    buf.perform_action(EditorAction::Enter);
    // Enter inserts \n; buffer_text now includes a trailing newline.
    assert_eq!(buf.text(), "fn main() {}\n");
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 1);
    assert_eq!(cursor.column, 0);
}

#[test]
fn test_backspace() {
    let buf = EditorBuffer::with_text("hello", None);
    buf.move_to(0, 5);
    buf.perform_action(EditorAction::Backspace);
    assert_eq!(buf.text(), "hell");
    let cursor = buf.cursor();
    assert_eq!(cursor.column, 4);
}

#[test]
fn test_backspace_at_start() {
    let buf = EditorBuffer::with_text("hello", None);
    buf.perform_action(EditorAction::Backspace);
    assert_eq!(buf.text(), "hello");
}

#[test]
fn test_backspace_newline() {
    let buf = EditorBuffer::with_text("hello\nworld", None);
    buf.move_to(1, 0);
    buf.perform_action(EditorAction::Backspace);
    assert_eq!(buf.text(), "helloworld");
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 0);
    assert_eq!(cursor.column, 5);
}

#[test]
fn test_delete() {
    let buf = EditorBuffer::with_text("hello", None);
    buf.perform_action(EditorAction::Delete);
    assert_eq!(buf.text(), "ello");
}

#[test]
fn test_delete_at_end() {
    let buf = EditorBuffer::with_text("hello", None);
    buf.move_to(0, 5);
    buf.perform_action(EditorAction::Delete);
    assert_eq!(buf.text(), "hello");
}

#[test]
fn test_delete_multibyte_accent() {
    let buf = EditorBuffer::with_text("café", None);
    buf.move_to(0, 3); // before 'é'
    buf.perform_action(EditorAction::Delete);
    assert_eq!(buf.text(), "caf");
}

#[test]
fn test_delete_multibyte_cyrillic() {
    let buf = EditorBuffer::with_text("привет", None);
    buf.move_to(0, 0);
    buf.perform_action(EditorAction::Delete);
    assert_eq!(buf.text(), "ривет");
}

#[test]
fn test_delete_emoji_scalar() {
    // Editor tracks scalar-value columns, not full grapheme clusters.
    let buf = EditorBuffer::with_text("a🎉b", None);
    buf.move_to(0, 1);
    buf.perform_action(EditorAction::Delete);
    assert_eq!(buf.text(), "ab");
}

#[test]
fn test_char_col_to_byte_offset_multibyte() {
    let line = "héllo";
    assert_eq!(char_col_to_byte_offset_in_line(line, 0), 0);
    assert_eq!(char_col_to_byte_offset_in_line(line, 1), 1);
    assert_eq!(char_col_to_byte_offset_in_line(line, 2), 3);
    assert_eq!(char_col_to_byte_range_in_line(line, 1), (1, 3));
}

#[test]
fn test_multiline_indent_preserves_selection() {
    let buf = EditorBuffer::with_text("- item one\n- item two\n- item three", None);
    buf.move_to(0, 0);
    buf.perform_action(EditorAction::SelectTo { line: 1, col: 100 });
    assert!(buf.cursor().selection.is_some());

    buf.perform_action(EditorAction::Indent);
    assert_eq!(buf.text(), "\t- item one\n\t- item two\n- item three");
    assert!(
        buf.cursor().selection.is_some(),
        "selection should survive first indent"
    );
    assert_eq!(
        buf.selection(),
        Some("\t- item one\n\t- item two".to_string())
    );

    buf.perform_action(EditorAction::Indent);
    assert_eq!(buf.text(), "\t\t- item one\n\t\t- item two\n- item three");
    assert!(
        buf.cursor().selection.is_some(),
        "selection should survive second indent"
    );

    buf.perform_action(EditorAction::Unindent);
    assert_eq!(buf.text(), "\t- item one\n\t- item two\n- item three");
    assert!(
        buf.cursor().selection.is_some(),
        "selection should survive unindent"
    );
}

#[test]
fn test_paste() {
    let buf = EditorBuffer::with_text("heo", None);
    buf.move_to(0, 2);
    buf.perform_action(EditorAction::Paste("ll".to_string()));
    assert_eq!(buf.text(), "hello");
}

#[test]
fn test_select_to_duplicate_endpoint_preserves_selection() {
    let buf = EditorBuffer::with_text("hello world", None);
    buf.move_to(0, 0);
    buf.perform_action(EditorAction::SelectTo { line: 0, col: 5 });
    assert_eq!(buf.selection(), Some("hello".to_string()));

    // Repeated SelectTo at the drag endpoint (duplicate CursorMoved).
    buf.perform_action(EditorAction::SelectTo { line: 0, col: 5 });
    assert_eq!(
        buf.selection(),
        Some("hello".to_string()),
        "duplicate SelectTo must not clear an existing selection"
    );
}

#[test]
fn test_indent() {
    let buf = EditorBuffer::with_text("hello", None);
    buf.perform_action(EditorAction::Indent);
    assert_eq!(buf.text(), "\thello");
}

#[test]
fn test_insert_with_selection_replaces() {
    let buf = EditorBuffer::with_text("hello world", None);
    buf.select_all();
    buf.perform_action(EditorAction::Insert('X'));
    assert_eq!(buf.text(), "X");
}

#[test]
fn test_enter_with_selection_replaces() {
    let buf = EditorBuffer::with_text("hello", None);
    buf.move_to(0, 3);
    buf.perform_action(EditorAction::SelectTo { line: 0, col: 5 });
    buf.perform_action(EditorAction::Enter);
    assert_eq!(buf.text(), "hel\n");
}

#[test]
fn test_paste_with_selection_replaces() {
    let buf = EditorBuffer::with_text("hello world", None);
    buf.move_to(0, 6);
    buf.perform_action(EditorAction::SelectTo { line: 0, col: 11 });
    buf.perform_action(EditorAction::Paste("there".to_string()));
    assert_eq!(buf.text(), "hello there");
}

#[test]
fn test_line_col_roundtrip() {
    let text = "hello\nworld\nfoo";
    for (line, line_text) in text.lines().enumerate() {
        for (col, _) in line_text.chars().enumerate() {
            let offset = line_col_to_byte_offset(text, line, col);
            let (rl, rc) = byte_offset_to_line_col(text, offset);
            assert_eq!(rl, line, "line mismatch at ({line},{col})");
            assert_eq!(rc, col, "col mismatch at ({line},{col})");
        }
        // End of line
        let col = line_text.chars().count();
        let offset = line_col_to_byte_offset(text, line, col);
        let (rl, rc) = byte_offset_to_line_col(text, offset);
        assert_eq!(rl, line, "line mismatch at end of line {line}");
        assert_eq!(rc, col, "col mismatch at end of line {line}");
    }
}

#[test]
fn test_selection_after_select_to_then_move() {
    let buf = EditorBuffer::with_text("abcdef", None);
    // Select "bcd"
    buf.move_to(0, 1);
    buf.perform_action(EditorAction::SelectTo { line: 0, col: 4 });
    assert_eq!(buf.selection(), Some("bcd".to_string()));
    // Performing a move clears selection
    buf.move_to(0, 0);
    assert!(buf.selection().is_none());
}

#[test]
fn test_unindent() {
    let buf = EditorBuffer::with_text("    hello", None);
    buf.perform_action(EditorAction::Unindent);
    assert_eq!(buf.text(), "   hello");
}

#[test]
fn test_multi_byte_character() {
    let buf = EditorBuffer::with_text("héllo", None);
    buf.move_to(0, 1);
    buf.perform_action(EditorAction::Insert('é'));
    assert_eq!(buf.text(), "hééllo");
    let cursor = buf.cursor();
    assert_eq!(cursor.column, 2);
}

#[test]
fn test_select_all_empty() {
    let buf = EditorBuffer::new();
    buf.select_all();
    assert!(buf.selection().is_none());
}

// ── Line comment prefix ───────────────────────────────────────

#[test]
fn test_line_comment_prefix_by_language() {
    use super::highlight::HighlightLanguage::*;
    assert_eq!(line_comment_prefix(Some(Rust), None), Some("//"));
    assert_eq!(line_comment_prefix(Some(Python), None), Some("#"));
    assert_eq!(line_comment_prefix(Some(Sql), None), Some("--"));
    assert_eq!(line_comment_prefix(Some(Json), None), None);
    assert_eq!(line_comment_prefix(Some(Html), None), None);
    assert_eq!(line_comment_prefix(Some(Markdown), None), None);
}

#[test]
fn test_line_comment_prefix_by_extension() {
    assert_eq!(line_comment_prefix(None, Some("yaml")), Some("#"));
    assert_eq!(line_comment_prefix(None, Some("yml")), Some("#"));
    assert_eq!(line_comment_prefix(None, Some("rs")), None); // falls to ext-only
}

#[test]
fn test_line_comment_prefix_none() {
    assert_eq!(line_comment_prefix(None, None), None);
    assert_eq!(line_comment_prefix(None, Some("xyz")), None);
}

// ── Toggle line comment ───────────────────────────────────────

#[allow(clippy::too_many_lines)]
#[test]
fn test_toggle_line_comment() {
    struct Case {
        name: &'static str,
        input: &'static str,
        expected: &'static str,
        language: Option<HighlightLanguage>,
        /// Set file extension instead of language (fallback path).
        file_ext: Option<&'static str>,
        /// Cursor position before toggling (defaults to 0, 0).
        cursor_line: usize,
        cursor_col: usize,
    }

    let cases: &[Case] = &[
        Case {
            name: "add",
            input: "hello",
            expected: "// hello",
            language: Some(HighlightLanguage::Rust),
            file_ext: None,
            cursor_line: 0,
            cursor_col: 0,
        },
        Case {
            name: "remove",
            input: "// hello",
            expected: "hello",
            language: Some(HighlightLanguage::Rust),
            file_ext: None,
            cursor_line: 0,
            cursor_col: 0,
        },
        Case {
            name: "remove_with_space",
            input: "//  hello",
            expected: " hello",
            language: Some(HighlightLanguage::Rust),
            file_ext: None,
            cursor_line: 0,
            cursor_col: 0,
        },
        Case {
            name: "preserves_whitespace",
            input: "    hello",
            expected: "    // hello",
            language: Some(HighlightLanguage::Rust),
            file_ext: None,
            cursor_line: 0,
            cursor_col: 0,
        },
        Case {
            name: "noop_unknown",
            input: "hello",
            expected: "hello",
            language: None,
            file_ext: None,
            cursor_line: 0,
            cursor_col: 0,
        },
        Case {
            name: "rust_hash",
            input: "hello",
            expected: "# hello",
            language: Some(HighlightLanguage::Python),
            file_ext: None,
            cursor_line: 0,
            cursor_col: 0,
        },
        Case {
            name: "yaml_via_extension",
            input: "hello",
            expected: "# hello",
            language: None,
            file_ext: Some("yaml"),
            cursor_line: 0,
            cursor_col: 0,
        },
        Case {
            name: "unknown_extension_noop",
            input: "hello",
            expected: "hello",
            language: None,
            file_ext: Some("xyz"),
            cursor_line: 0,
            cursor_col: 0,
        },
        Case {
            name: "preserves_neighbor_lines",
            input: "first\nsecond\nthird",
            expected: "first\n// second\nthird",
            language: Some(HighlightLanguage::Rust),
            file_ext: None,
            cursor_line: 1,
            cursor_col: 0,
        },
    ];

    for case in cases {
        let buf = EditorBuffer::with_text(case.input, case.language);
        if let Some(ext) = case.file_ext {
            buf.set_file_extension(Some(ext));
        }
        if case.cursor_line != 0 || case.cursor_col != 0 {
            buf.move_to(case.cursor_line, case.cursor_col);
        }
        buf.perform_action(EditorAction::ToggleLineComment);
        assert_eq!(buf.text(), case.expected, "case: {}", case.name);
    }
}

// ── Jump to matching bracket ───────────────────────────────────

#[test]
fn test_jump_to_matching_bracket() {
    struct Case {
        name: &'static str,
        input: &'static str,
        cursor_col: usize,
        expected_line: usize,
        expected_col: usize,
    }

    let cases: &[Case] = &[
        Case {
            name: "forward_paren",
            input: "(hello)",
            cursor_col: 1,
            expected_line: 0,
            expected_col: 6,
        },
        Case {
            name: "backward_paren",
            input: "(hello)",
            cursor_col: 6,
            expected_line: 0,
            expected_col: 1,
        },
        Case {
            name: "square_bracket",
            input: "[hello]",
            cursor_col: 1,
            expected_line: 0,
            expected_col: 6,
        },
        Case {
            name: "brace",
            input: "{hello}",
            cursor_col: 1,
            expected_line: 0,
            expected_col: 6,
        },
        Case {
            name: "none",
            input: "hello",
            cursor_col: 3,
            expected_line: 0,
            expected_col: 3,
        },
    ];

    for case in cases {
        let buf = EditorBuffer::with_text(case.input, None);
        buf.move_to(0, case.cursor_col);
        buf.perform_action(EditorAction::JumpToMatchingBracket);
        let cursor = buf.cursor();
        assert_eq!(
            cursor.line, case.expected_line,
            "case: {} (line)",
            case.name
        );
        assert_eq!(
            cursor.column, case.expected_col,
            "case: {} (col)",
            case.name
        );
    }
}

// ── Delete line ────────────────────────────────────────────────

#[test]
fn test_delete_current_line() {
    let buf = EditorBuffer::with_text("line1\nline2\nline3", None);
    buf.move_to(1, 0);
    buf.perform_action(EditorAction::DeleteLine);
    assert_eq!(buf.text(), "line1\nline3");
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 1); // Stayed at index 1 (now "line3")
    assert_eq!(cursor.column, 0);
}

#[test]
fn test_delete_first_line() {
    let buf = EditorBuffer::with_text("line1\nline2\nline3", None);
    buf.perform_action(EditorAction::DeleteLine);
    assert_eq!(buf.text(), "line2\nline3");
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 0);
    assert_eq!(cursor.column, 0);
}

#[test]
fn test_delete_last_line() {
    let buf = EditorBuffer::with_text("line1\nline2\nline3", None);
    buf.move_to(2, 0);
    buf.perform_action(EditorAction::DeleteLine);
    assert_eq!(buf.text(), "line1\nline2");
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 1);
}

#[test]
fn test_delete_single_line() {
    let buf = EditorBuffer::with_text("hello", None);
    buf.perform_action(EditorAction::DeleteLine);
    assert_eq!(buf.text(), "");
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 0);
    assert_eq!(cursor.column, 0);
}

#[test]
fn test_delete_selected_lines() {
    let buf = EditorBuffer::with_text("a\nb\nc\nd\ne", None);
    buf.move_to(1, 0);
    buf.perform_action(EditorAction::SelectTo { line: 3, col: 0 });
    buf.perform_action(EditorAction::DeleteLine);
    assert_eq!(buf.text(), "a\ne");
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 1);
}

// ── Duplicate line ─────────────────────────────────────────────

#[test]
fn test_duplicate_current_line() {
    let buf = EditorBuffer::with_text("hello\nworld", None);
    buf.move_to(0, 0);
    buf.perform_action(EditorAction::DuplicateLine);
    assert_eq!(buf.text(), "hello\nhello\nworld");
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 1); // Cursor on duplicated line
    assert_eq!(cursor.column, 0);
}

#[test]
fn test_duplicate_last_line() {
    let buf = EditorBuffer::with_text("hello\nworld", None);
    buf.move_to(1, 0);
    buf.perform_action(EditorAction::DuplicateLine);
    assert_eq!(buf.text(), "hello\nworld\nworld");
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 2);
}

#[test]
fn test_duplicate_selected_lines() {
    let buf = EditorBuffer::with_text("a\nb\nc\nd", None);
    buf.move_to(1, 0);
    buf.perform_action(EditorAction::SelectTo { line: 2, col: 0 });
    buf.perform_action(EditorAction::DuplicateLine);
    assert_eq!(buf.text(), "a\nb\nc\nb\nc\nd");
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 3);
}

// ── Move line up/down ──────────────────────────────────────────

#[test]
fn test_move_line_up() {
    let buf = EditorBuffer::with_text("a\nb\nc", None);
    buf.move_to(1, 0);
    buf.perform_action(EditorAction::MoveLineUp);
    assert_eq!(buf.text(), "b\na\nc");
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 0);
    assert_eq!(cursor.column, 0);
}

#[test]
fn test_move_line_down() {
    let buf = EditorBuffer::with_text("a\nb\nc", None);
    buf.move_to(1, 0);
    buf.perform_action(EditorAction::MoveLineDown);
    assert_eq!(buf.text(), "a\nc\nb");
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 2);
    assert_eq!(cursor.column, 0);
}

#[test]
fn test_move_line_up_at_top() {
    let buf = EditorBuffer::with_text("a\nb", None);
    buf.perform_action(EditorAction::MoveLineUp);
    assert_eq!(buf.text(), "a\nb"); // No change
}

#[test]
fn test_move_line_down_at_bottom() {
    let buf = EditorBuffer::with_text("a\nb", None);
    buf.move_to(1, 0);
    buf.perform_action(EditorAction::MoveLineDown);
    assert_eq!(buf.text(), "a\nb"); // No change
}

#[test]
fn test_has_trailing_newline() {
    assert!(has_trailing_newline("hello\n"));
    assert!(!has_trailing_newline("hello"));
    assert!(!has_trailing_newline(""));
}

#[test]
fn test_detect_line_ending_lf() {
    assert_eq!(detect_line_ending("hello\nworld\n"), LineEnding::Lf);
}

#[test]
fn test_detect_line_ending_crlf() {
    assert_eq!(detect_line_ending("hello\r\nworld\r\n"), LineEnding::Crlf);
}

#[test]
fn test_line_helpers_preserve_crlf_on_move_down() {
    let text = "a\r\nb\r\nc";
    let mut lines = logical_lines(text);
    swap_lines_with_endings(&mut lines, 1, 2);
    fix_line_endings(
        &mut lines,
        has_trailing_newline(text),
        detect_line_ending(text),
    );
    assert_eq!(reassemble_lines(&lines), "a\r\nc\r\nb");
}

#[test]
fn test_line_helpers_preserve_trailing_blank_line() {
    let text = "line one\nline two\n\n";
    let lines = logical_lines(text);
    assert_eq!(lines[2].0, "");
    assert_eq!(lines[2].1, "\n");
    assert_eq!(reassemble_lines(&lines), text);
}

// ── Multi-line indent/outdent ──────────────────────────────────

#[test]
fn test_indent_with_selection() {
    let buf = EditorBuffer::with_text("hello\nworld\nfoo", None);
    buf.move_to(0, 0);
    buf.perform_action(EditorAction::SelectTo { line: 1, col: 0 });
    buf.perform_action(EditorAction::Indent);
    assert_eq!(buf.text(), "\thello\n\tworld\nfoo");
}

#[test]
fn test_unindent_with_selection() {
    let buf = EditorBuffer::with_text("\thello\n\tworld\nfoo", None);
    buf.move_to(0, 0);
    buf.perform_action(EditorAction::SelectTo { line: 1, col: 0 });
    buf.perform_action(EditorAction::Unindent);
    assert_eq!(buf.text(), "hello\nworld\nfoo");
}

// ── Selection normalization ─────────────────────────────────────

#[test]
fn test_shift_left_at_bof_no_selection() {
    let buf = EditorBuffer::with_text("hello", None);
    buf.perform_action(EditorAction::Move {
        direction: CursorMove::Left,
        select: true,
    });
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 0);
    assert_eq!(cursor.column, 0);
    assert!(cursor.selection.is_none());
}

#[test]
fn test_select_to_same_endpoint_preserves_non_empty_selection() {
    let buf = EditorBuffer::with_text("hello", None);
    buf.move_to(0, 2);
    buf.perform_action(EditorAction::SelectTo { line: 0, col: 3 });
    assert!(buf.cursor().selection.is_some());
    // Duplicate SelectTo at the drag endpoint must not collapse the range.
    buf.perform_action(EditorAction::SelectTo { line: 0, col: 3 });
    assert!(buf.cursor().selection.is_some());
    assert_eq!(buf.selection(), Some("l".to_string()));
}

#[test]
fn test_shift_right_then_back_collapses_selection() {
    let buf = EditorBuffer::with_text("hello", None);
    buf.move_to(0, 0);
    buf.perform_action(EditorAction::Move {
        direction: CursorMove::Right,
        select: true,
    });
    buf.perform_action(EditorAction::Move {
        direction: CursorMove::Left,
        select: true,
    });
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 0);
    assert_eq!(cursor.column, 0);
    assert!(cursor.selection.is_none());
}
