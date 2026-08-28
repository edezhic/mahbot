use super::*;

#[test]
fn test_new_empty() {
    let buf = EditorBuffer::with_text("", None);
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
#[expect(clippy::type_complexity)]
fn test_backspace_delete_cases() {
    // (name, action, input, cursor pos, expected text, expected cursor)
    #[rustfmt::skip]
    let cases: &[(&str, EditorAction, &str, Option<(usize, usize)>, &str, Option<(usize, usize)>)] = &[
        ("backspace", EditorAction::Backspace, "hello", Some((0, 5)), "hell", Some((0, 4))),
        ("backspace_at_start", EditorAction::Backspace, "hello", None, "hello", None),
        ("backspace_newline", EditorAction::Backspace, "hello\nworld", Some((1, 0)), "helloworld", Some((0, 5))),
        ("delete", EditorAction::Delete, "hello", None, "ello", None),
        ("delete_at_end", EditorAction::Delete, "hello", Some((0, 5)), "hello", None),
        ("delete_cafe", EditorAction::Delete, "café", Some((0, 3)), "caf", None),
        ("delete_cyrillic", EditorAction::Delete, "привет", Some((0, 0)), "ривет", None),
        // Editor tracks scalar-value columns, not full grapheme clusters.
        ("delete_emoji", EditorAction::Delete, "a🎉b", Some((0, 1)), "ab", None),
    ];
    for &(name, ref action, input, cursor_pos, expected, expected_cursor) in cases {
        let buf = EditorBuffer::with_text(input, None);
        if let Some((line, col)) = cursor_pos {
            buf.move_to(line, col);
        }
        buf.perform_action(action.clone());
        assert_eq!(buf.text(), expected, "case: {name}");
        if let Some((line, col)) = expected_cursor {
            let cursor = buf.cursor();
            assert_eq!(cursor.line, line, "case: {name} (line)");
            assert_eq!(cursor.column, col, "case: {name} (col)");
        }
    }
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
    let buf = EditorBuffer::with_text("", None);
    buf.select_all();
    assert!(buf.selection().is_none());
}

// ── Line comment prefix ───────────────────────────────────────

#[test]
fn test_line_comment_prefix() {
    use super::highlight::HighlightLanguage::*;
    assert_eq!(line_comment_prefix(Some(Rust), None), Some("//"));
    assert_eq!(line_comment_prefix(Some(Python), None), Some("#"));
    assert_eq!(line_comment_prefix(Some(Sql), None), Some("--"));
    assert_eq!(line_comment_prefix(Some(Json), None), None);
    assert_eq!(line_comment_prefix(Some(Html), None), None);
    assert_eq!(line_comment_prefix(Some(Markdown), None), None);
    assert_eq!(line_comment_prefix(None, Some("yaml")), Some("#"));
    assert_eq!(line_comment_prefix(None, Some("yml")), Some("#"));
    assert_eq!(line_comment_prefix(None, Some("rs")), None); // falls to ext-only
    assert_eq!(line_comment_prefix(None, None), None);
    assert_eq!(line_comment_prefix(None, Some("xyz")), None);
}

// ── Toggle line comment ───────────────────────────────────────

#[expect(clippy::too_many_lines)]
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
#[expect(clippy::type_complexity)]
fn test_delete_line_cases() {
    // (name, input, cursor line, selection end, expected text, expected line, expected column)
    #[rustfmt::skip]
    let cases: &[(&str, &str, usize, Option<(usize, usize)>, &str, usize, Option<usize>)] = &[
        // Stayed at index 1 (now "line3")
        ("current_line", "line1\nline2\nline3", 1, None, "line1\nline3", 1, Some(0)),
        ("first_line", "line1\nline2\nline3", 0, None, "line2\nline3", 0, Some(0)),
        ("last_line", "line1\nline2\nline3", 2, None, "line1\nline2", 1, None),
        ("single_line", "hello", 0, None, "", 0, Some(0)),
        ("selected_lines", "a\nb\nc\nd\ne", 1, Some((3, 0)), "a\ne", 1, None),
    ];
    for &(name, input, cursor_line, select_to, expected, expected_line, expected_col) in cases {
        let buf = EditorBuffer::with_text(input, None);
        buf.move_to(cursor_line, 0);
        if let Some((line, col)) = select_to {
            buf.perform_action(EditorAction::SelectTo { line, col });
        }
        buf.perform_action(EditorAction::DeleteLine);
        assert_eq!(buf.text(), expected, "case: {name}");
        let cursor = buf.cursor();
        assert_eq!(cursor.line, expected_line, "case: {name} (line)");
        if let Some(col) = expected_col {
            assert_eq!(cursor.column, col, "case: {name} (col)");
        }
    }
}

// ── Duplicate line ─────────────────────────────────────────────

#[test]
#[expect(clippy::type_complexity)]
fn test_duplicate_line_cases() {
    // (name, input, cursor line, selection end, expected text, expected line, expected column)
    #[rustfmt::skip]
    let cases: &[(&str, &str, usize, Option<(usize, usize)>, &str, usize, Option<usize>)] = &[
        // Cursor on duplicated line
        ("current_line", "hello\nworld", 0, None, "hello\nhello\nworld", 1, Some(0)),
        ("last_line", "hello\nworld", 1, None, "hello\nworld\nworld", 2, None),
        ("selected_lines", "a\nb\nc\nd", 1, Some((2, 0)), "a\nb\nc\nb\nc\nd", 3, None),
    ];
    for &(name, input, cursor_line, select_to, expected, expected_line, expected_col) in cases {
        let buf = EditorBuffer::with_text(input, None);
        buf.move_to(cursor_line, 0);
        if let Some((line, col)) = select_to {
            buf.perform_action(EditorAction::SelectTo { line, col });
        }
        buf.perform_action(EditorAction::DuplicateLine);
        assert_eq!(buf.text(), expected, "case: {name}");
        let cursor = buf.cursor();
        assert_eq!(cursor.line, expected_line, "case: {name} (line)");
        if let Some(col) = expected_col {
            assert_eq!(cursor.column, col, "case: {name} (col)");
        }
    }
}

// ── Move line up/down ──────────────────────────────────────────

#[test]
#[expect(clippy::type_complexity)]
fn test_move_line_cases() {
    // (name, action, input, cursor line, selection end, expected text, expected line, expected column)
    #[rustfmt::skip]
    let cases: &[(&str, EditorAction, &str, usize, Option<(usize, usize)>, &str, Option<usize>, Option<usize>)] = &[
        ("up", EditorAction::MoveLineUp, "a\nb\nc", 1, None, "b\na\nc", Some(0), Some(0)),
        ("down", EditorAction::MoveLineDown, "a\nb\nc", 1, None, "a\nc\nb", Some(2), Some(0)),
        ("up_at_top", EditorAction::MoveLineUp, "a\nb", 0, None, "a\nb", None, None), // No change
        ("down_at_bottom", EditorAction::MoveLineDown, "a\nb", 1, None, "a\nb", None, None), // No change
        // First line of the moved block
        ("selected_down", EditorAction::MoveLineDown, "a\nb\nc\nd", 1, Some((2, 0)), "a\nd\nb\nc", Some(2), Some(0)),
    ];
    for &(name, ref action, input, cursor_line, select_to, expected, expected_line, expected_col) in
        cases
    {
        let buf = EditorBuffer::with_text(input, None);
        buf.move_to(cursor_line, 0);
        if let Some((line, col)) = select_to {
            buf.perform_action(EditorAction::SelectTo { line, col });
        }
        buf.perform_action(action.clone());
        assert_eq!(buf.text(), expected, "case: {name}");
        if let Some(line) = expected_line {
            let cursor = buf.cursor();
            assert_eq!(cursor.line, line, "case: {name} (line)");
            if let Some(col) = expected_col {
                assert_eq!(cursor.column, col, "case: {name} (col)");
            }
        }
    }
}

#[test]
fn test_has_trailing_newline_and_detect_line_ending() {
    assert!(has_trailing_newline("hello\n"));
    assert!(!has_trailing_newline("hello"));
    assert!(!has_trailing_newline(""));
    assert_eq!(detect_line_ending("hello\nworld\n"), LineEnding::Lf);
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
fn test_select_to_duplicate_endpoint_cases() {
    // (name, text, start col, drag endpoint col, expected selection text)
    #[rustfmt::skip]
    let cases: &[(&str, &str, usize, usize, &str)] = &[
        ("duplicate_endpoint", "hello world", 0, 5, "hello"),
        ("same_endpoint", "hello", 2, 3, "l"),
    ];
    for &(name, text, start_col, end_col, expected) in cases {
        let buf = EditorBuffer::with_text(text, None);
        buf.move_to(0, start_col);
        buf.perform_action(EditorAction::SelectTo {
            line: 0,
            col: end_col,
        });
        assert!(
            buf.cursor().selection.is_some(),
            "case: {name} (first SelectTo)"
        );
        // Repeated SelectTo at the drag endpoint (duplicate CursorMoved)
        // must not collapse or clear an existing selection.
        buf.perform_action(EditorAction::SelectTo {
            line: 0,
            col: end_col,
        });
        assert_eq!(
            buf.selection(),
            Some(expected.to_string()),
            "case: {name} (duplicate SelectTo)"
        );
    }
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

#[test]
fn test_set_text_resets_cursor_and_clear() {
    let buf = EditorBuffer::with_text("hello", None);
    buf.move_to(0, 2);
    buf.set_text("world");
    assert_eq!(buf.text(), "world");
    let cursor = buf.cursor();
    assert_eq!(cursor.line, 0);
    assert_eq!(cursor.column, 0);
    assert!(cursor.selection.is_none());
    assert!(!buf.is_empty());
    buf.clear();
    assert!(buf.is_empty());
}

#[test]
fn test_single_line_strips_newlines_and_enter() {
    let buf = EditorBuffer::with_text("a\nb", None);
    buf.set_single_line(true);
    assert_eq!(buf.text(), "ab");
    buf.move_to(0, 1);
    buf.perform_action(EditorAction::Paste("x\ny".to_string()));
    assert_eq!(buf.text(), "axyb");
    assert_eq!(buf.cursor().column, 3);
    buf.perform_action(EditorAction::Enter);
    assert_eq!(buf.text(), "axyb");
    assert_eq!(buf.cursor().line, 0);
}

// ── IME composition (over-the-spot preedit) ──────────────────────

#[test]
fn test_ime_surface_activation() {
    let buf = EditorBuffer::with_text("", None);
    // Masked / password fields never activate IME.
    assert!(
        !EditorWidget::new(&buf)
            .masked(true)
            .is_active_surface(&EditorWidgetState::default())
    );
    // The window must be focused.
    let mut unfocused = EditorWidgetState::default();
    unfocused.is_window_focused = false;
    assert!(!EditorWidget::new(&buf).is_active_surface(&unfocused));
    // A focus-id field activates only while focused.
    let field = EditorWidget::new(&buf).id("field");
    let mut focused = EditorWidgetState::default();
    assert!(!field.is_active_surface(&focused));
    focused.is_focused = true;
    assert!(field.is_active_surface(&focused));
    // The full-page code editor (no focus id, code_mode) is active unless it
    // ignores the keyboard; a focus-id-less *non-code* field is not active.
    assert!(EditorWidget::new(&buf).is_active_surface(&EditorWidgetState::default()));
    assert!(
        !EditorWidget::new(&buf)
            .ignore_keyboard(true)
            .is_active_surface(&EditorWidgetState::default())
    );
    assert!(
        !EditorWidget::new(&buf)
            .code_mode(false)
            .is_active_surface(&EditorWidgetState::default())
    );
}

#[test]
fn test_input_method_preedit_reported() {
    let buf = EditorBuffer::with_text("", None);
    let widget = EditorWidget::new(&buf);
    let mut state = EditorWidgetState::default();
    state.preedit = Some(input_method::Preedit {
        content: "你好".to_string(),
        selection: Some(1..2),
        text_size: None,
    });
    let bounds = Rectangle::new(Point::ORIGIN, Size::new(200.0, 200.0));
    let ime = widget.input_method(&state, bounds);
    assert!(ime.to_owned().is_enabled());
    match ime {
        input_method::InputMethod::Enabled {
            preedit: Some(p), ..
        } => {
            assert_eq!(p.content, "你好");
            assert_eq!(p.selection, Some(1..2));
        }
        other => panic!("expected Enabled with preedit, got {other:?}"),
    }
}

#[test]
fn test_cursor_rect_selection_anchor() {
    let buf = EditorBuffer::with_text("hello\nworld", None);
    // Cursor at (1,4), selection anchor at (0,3) — reversed selection so the
    // IME must anchor at the selection start, not the drawn caret end.
    buf.move_to(0, 3);
    buf.perform_action(EditorAction::SelectTo { line: 1, col: 4 });
    let buffer = with_font_system(|font_sys| {
        let mut b = buf.borrow_buffer_mut();
        reshape_and_shape(&mut b, font_sys, Some(0.0), 0.0, 200.0, 200.0);
        b.clone()
    });
    let geo = TextGeometry {
        clip: Rectangle::new(Point::ORIGIN, Size::new(200.0, 200.0)),
        x: 8.0,
        y: 8.0,
    };
    let rect = ime_caret_rect(&buffer, &geo, &buf);
    let expected = cursor_rect(&buffer, &geo, 0, 3, geo.x);
    assert_eq!(rect, expected);
    // The anchor (line 0) sits above the caret (line 1).
    assert!(rect.y < cursor_rect(&buffer, &geo, 1, 4, geo.x).y);
}

#[test]
fn test_cursor_rect_shaped_position() {
    let buf = EditorBuffer::with_text("hello\nworld", None);
    buf.move_to(1, 2);
    let buffer = with_font_system(|font_sys| {
        let mut b = buf.borrow_buffer_mut();
        reshape_and_shape(&mut b, font_sys, Some(0.0), 0.0, 200.0, 200.0);
        b.clone()
    });
    let geo = TextGeometry {
        clip: Rectangle::new(Point::ORIGIN, Size::new(200.0, 200.0)),
        x: 8.0,
        y: 8.0,
    };
    let rect = cursor_rect(&buffer, &geo, 1, 2, geo.x);
    assert!(rect.y > 8.0);
    // Second line sits below the first.
    let first = cursor_rect(&buffer, &geo, 0, 0, geo.x);
    assert!(rect.y >= first.y);
}

#[test]
fn test_masked_render_buffer_builds_dots_without_reborrow() {
    // Regression test for the Settings-page panic: `layout()` used to call
    // `self.buffer.text()` (which re-borrows the RefCell via `borrow()`) while
    // the `borrow_buffer_mut()` RefMut was still held, panicking on the masked
    // (password) path with "already mutably borrowed". The masked render path
    // must derive the text from the already-held RefMut instead.
    let buf = EditorBuffer::with_text("secret\npassword", None);
    let mbuffer = with_font_system(|font_sys| {
        let buffer = buf.borrow_buffer_mut();
        let mb = build_masked_render_buffer(font_sys, &buffer, false, 0.0, 0.0, 300.0, 100.0);
        // The RefMut is still live and readable after building the masked buffer.
        assert_eq!(buffer_text(&buffer), "secret\npassword");
        mb
    });
    // Rendered text is fully masked, preserving character count and line structure.
    assert_eq!(buffer_text(&mbuffer), "••••••\n••••••••");
}

#[test]
fn test_masked_render_buffer_single_line_does_not_wrap() {
    // Password fields on the Settings page are single-line. The masked dots
    // must not wrap (matching the real single-line buffer layout) — this is
    // the exact widget configuration that triggered the original panic.
    let buf = EditorBuffer::with_text("hunter2", None);
    let mbuffer = with_font_system(|font_sys| {
        let buffer = buf.borrow_buffer_mut();
        build_masked_render_buffer(font_sys, &buffer, true, 0.0, 0.0, 100.0, 30.0)
    });
    assert_eq!(buffer_text(&mbuffer), "•••••••");
}

// ── Single-line height & cursor focus (shared-editor regressions) ──────

#[test]
fn test_single_line_size_hint_is_shrink() {
    let buf = EditorBuffer::with_text("", None);
    // Single-line fields (search / password / rename) must not fill their
    // container vertically — the size hint lets the wrapping container
    // shrink to a single line instead of inheriting `Length::Fill`.
    let size_of = |widget: &EditorWidget<'_>| {
        <EditorWidget<'_> as Widget<EditorAction, iced::Theme, iced::Renderer>>::size(widget)
    };
    let single = EditorWidget::new(&buf).single_line(true);
    assert_eq!(size_of(&single), Size::new(Length::Fill, Length::Shrink));
    // Multi-line prose / code editors keep filling their container.
    let multi = EditorWidget::new(&buf);
    assert_eq!(size_of(&multi), Size::new(Length::Fill, Length::Fill));
}

#[test]
fn test_cursor_follows_focus_state() {
    let buf = EditorBuffer::with_text("", None);
    // A focus-id field draws the caret only while focused.
    let field = EditorWidget::new(&buf).single_line(true).id("field");
    let state = EditorWidgetState::default();
    assert!(!field.should_draw_cursor(&state));
    let focused = EditorWidgetState {
        is_focused: true,
        ..EditorWidgetState::default()
    };
    assert!(field.should_draw_cursor(&focused));
    // Masked / password fields still show the caret when focused (the IME
    // `is_active_surface` predicate would suppress them, but the caret must
    // not).
    let masked = EditorWidget::new(&buf)
        .single_line(true)
        .masked(true)
        .id("pw");
    assert!(masked.should_draw_cursor(&focused));
    // Window blur suppresses the caret even while focused.
    let blurred = EditorWidgetState {
        is_focused: true,
        is_window_focused: false,
        ..EditorWidgetState::default()
    };
    assert!(!field.should_draw_cursor(&blurred));
    // A focus-less field is always active (code editor) while the window is
    // focused and it is not ignoring the keyboard.
    let code = EditorWidget::new(&buf);
    assert!(code.should_draw_cursor(&EditorWidgetState::default()));
    assert!(
        !code
            .ignore_keyboard(true)
            .should_draw_cursor(&EditorWidgetState::default())
    );
    // A focus-less *non-code* field is not an active surface (matching
    // `is_active_surface`), so it must not blink its caret either.
    assert!(
        !EditorWidget::new(&buf)
            .code_mode(false)
            .should_draw_cursor(&EditorWidgetState::default())
    );
}

#[test]
fn test_single_line_height_clamp() {
    let line_height = font_metrics().line_height;
    let padding = 5.0;
    let single_line_h = line_height + 2.0 * padding;
    let close = |a: f32, b: f32| assert!((a - b).abs() < f32::EPSILON, "{a} != {b}");
    // Single-line fields always resolve to exactly one line, never filling
    // the container vertically — but never taller than the container.
    close(
        autosize_height(true, line_height, line_height, padding, None, None, 300.0),
        single_line_h,
    );
    close(
        autosize_height(true, line_height, line_height, padding, None, None, 10.0),
        10.0,
    );
    // Multi-line prose clamps content height to [min, max].
    close(
        autosize_height(
            false,
            line_height,
            50.0,
            padding,
            Some(44.0),
            Some(132.0),
            300.0,
        ),
        60.0,
    );
    // The code editor (no single-line, no min/max) fills the container.
    close(
        autosize_height(false, line_height, 50.0, padding, None, None, 300.0),
        300.0,
    );
}

#[test]
fn mouse_interaction_gated_on_cursor_bounds() {
    // Regression for modal click-outside-to-close: the editor must report
    // `Text` only while the cursor is over its own bounds. Reporting it
    // unconditionally makes iced `Stack` levitate the cursor over a modal's
    // backdrop `mouse_area`, which then bails on the now-Levitating cursor and
    // never delivers the click — so embedding an editor broke click-outside.
    let bounds = Rectangle::new(Point::new(10.0, 10.0), Size::new(100.0, 40.0));

    // Cursor inside the editor bounds → the text-edit affordance.
    let inside = mouse::Cursor::Available(Point::new(50.0, 30.0));
    assert_eq!(
        editor_mouse_interaction(inside, bounds),
        mouse::Interaction::Text
    );

    // Cursor outside the bounds → default (`None`), so the editor never
    // claims the cursor beyond its field and the backdrop stays clickable.
    let outside = mouse::Cursor::Available(Point::new(300.0, 300.0));
    assert_eq!(
        editor_mouse_interaction(outside, bounds),
        mouse::Interaction::default()
    );

    // A levitating cursor (an overlay above already claimed it) is never
    // "over" the editor, so it must not be re-claimed nor re-levitated.
    let levitating = mouse::Cursor::Levitating(Point::new(50.0, 30.0));
    assert_eq!(
        editor_mouse_interaction(levitating, bounds),
        mouse::Interaction::default()
    );

    // No cursor available → default (`None`).
    assert_eq!(
        editor_mouse_interaction(mouse::Cursor::Unavailable, bounds),
        mouse::Interaction::default()
    );
}
