use super::*;

use strum::IntoEnumIterator;

/// Create a Telegram Update JSON with sensible defaults, then apply
/// shallow top-level overrides for test-specific fields.
///
/// Base defaults:
/// ```json
/// { "update_id": 1, "message": { "message_id": 33, "text": "hello",
///   "from": {"id": 555, "username": "alice"},
///   "chat": {"id": -100_200_300} } }
/// ```
///
/// Overrides replace entire top-level keys (shallow merge). To change
/// nested fields, pass the full nested value:
///
/// ```ignore
/// let update = test_update(&[(
///     "message",
///     json!({
///         "message_id": 42, "text": "hi",
///         "from": {"id": 555, "username": "alice"},
///         "chat": {"id": -100_200_300},
///         "message_thread_id": 789
///     }),
/// )]);
/// ```
fn test_update(overrides: &[(&str, serde_json::Value)]) -> serde_json::Value {
    let mut update = serde_json::json!({
        "update_id": 1,
        "message": {
            "message_id": 33,
            "text": "hello",
            "from": { "id": 555, "username": "alice" },
            "chat": { "id": -100_200_300 }
        }
    });
    let obj = update.as_object_mut().unwrap();
    for (key, value) in overrides {
        obj.insert(key.to_string(), value.clone());
    }
    update
}

/// Create a TelegramChannel with a test store initialized.
/// Uses token `"token"` and calls `init_test_store` before returning.
async fn test_channel() -> TelegramChannel {
    crate::users::test_util::init_test_store().await;
    TelegramChannel::new("token".into())
}

#[test]
fn test_parse_recipient() {
    assert_eq!(parse_recipient("12345"), ("12345", None));
    assert_eq!(parse_recipient("12345:678"), ("12345", Some("678")));
    assert_eq!(parse_recipient(""), ("", None));
}

#[test]
fn classify_edit_failure_matches_stable_substrings() {
    use EditMessageFailure::{CannotEdit, NotFound, NotModified, Other};
    // Production feeds the raw error envelope (`read_error_body`); matching is
    // on the embedded description, and envelope fields must not false-match.
    assert!(matches!(
        classify_edit_failure(
            r#"{"ok":false,"error_code":400,"description":"Bad Request: message to edit not found"}"#
        ),
        NotFound
    ));
    assert!(matches!(
        classify_edit_failure(
            r#"{"ok":false,"error_code":400,"description":"Bad Request: message not found"}"#
        ),
        NotFound
    ));
    assert!(matches!(
        classify_edit_failure(
            r#"{"ok":false,"error_code":400,"description":"Bad Request: message is not modified"}"#
        ),
        NotModified
    ));
    assert!(matches!(
        classify_edit_failure(
            r#"{"ok":false,"error_code":400,"description":"Bad Request: message can't be edited"}"#
        ),
        CannotEdit
    ));
    assert!(matches!(
        classify_edit_failure(
            r#"{"ok":false,"error_code":400,"description":"Bad Request: message cant be edited"}"#
        ),
        CannotEdit
    ));
    assert!(matches!(
        classify_edit_failure(
            r#"{"ok":false,"error_code":429,"description":"Too Many Requests: retry after 5"}"#
        ),
        Other
    ));
    assert!(matches!(classify_edit_failure(""), Other));
}

#[test]
fn test_markdown_to_telegram_html() {
    // escapes quotes in link href
    let r = markdown_to_telegram_html("[click](https://example.com?q=\"x\"&a='b')");
    assert_eq!(
        r,
        "<a href=\"https://example.com?q=&quot;x&quot;&amp;a=&#39;b&#39;\">click</a>"
    );
    // escapes quotes/ampersand in plain text
    let r = markdown_to_telegram_html("say \"hi\" & <tag> 'ok'");
    assert_eq!(r, "say &quot;hi&quot; &amp; &lt;tag&gt; &#39;ok&#39;");
    // drops language attribute from code blocks
    let r = markdown_to_telegram_html("```rust\" onclick=\"alert(1)\nlet x = 1;\n```");
    assert_eq!(r, "<pre><code>let x = 1;</code></pre>");
    assert!(!r.contains("language-"));
    assert!(!r.contains("onclick"));

    // Inline formatting inside code blocks is preserved literally
    let r = markdown_to_telegram_html("```\nsome **bold** and `code`\n```");
    assert_eq!(r, "<pre><code>some **bold** and `code`</code></pre>");

    // HTML special characters in code blocks are escaped
    let r = markdown_to_telegram_html("```\n<div> & \"it\" 'works'\n```");
    assert_eq!(
        r,
        "<pre><code>&lt;div&gt; &amp; &quot;it&quot; &#39;works&#39;</code></pre>"
    );

    // Literal </code> in code block must not break the HTML
    let r = markdown_to_telegram_html("```\nuse &lt;/code&gt;\n```");
    assert_eq!(r, "<pre><code>use &amp;lt;/code&amp;gt;</code></pre>");

    // ── Blockquote pass-through ────────────────────────────

    // Opening tag on its own line passes through unchanged
    let r = markdown_to_telegram_html("<blockquote>");
    assert_eq!(r, "<blockquote>");

    // Closing tag on its own line passes through unchanged
    let r = markdown_to_telegram_html("</blockquote>");
    assert_eq!(r, "</blockquote>");

    // Multi-line blockquote: content between tags gets inline formatting
    let r = markdown_to_telegram_html("<blockquote>\nHello **world**\n</blockquote>");
    assert_eq!(r, "<blockquote>\nHello <b>world</b>\n</blockquote>");

    // Malformed tag name: <blockquote123> should NOT pass through
    let r = markdown_to_telegram_html("<blockquote123>");
    assert_eq!(r, "&lt;blockquote123&gt;");

    // Tag with attributes: <blockquote class="x"> should NOT pass through
    let r = markdown_to_telegram_html("<blockquote class=\"x\">");
    assert_eq!(r, "&lt;blockquote class=&quot;x&quot;&gt;");

    // Tag with trailing space inside: <blockquote > should NOT pass through
    let r = markdown_to_telegram_html("<blockquote >");
    assert_eq!(r, "&lt;blockquote &gt;");
}

/// The `/board` listing formats each ticket via [`format_board_line`] and
/// converts the joined text through `markdown_to_telegram_html` line-by-line.
/// A title with markdown-special characters must never corrupt other lines'
/// formatting or leave unbalanced HTML tags (which would make Telegram reject
/// the whole message).
#[test]
fn board_listing_isolates_hostile_titles() {
    // Pin the production format string — if `handle_board_listing`'s helper
    // drifts, this test fails instead of silently covering a stale fixture.
    assert_eq!(
        format_board_line(
            &crate::board::TicketPhase::InDevelopment,
            "mahbot-123",
            "Title",
        ),
        "• **in development** `mahbot-123` Title"
    );

    let state = crate::board::TicketPhase::InDevelopment;
    let lines = vec![
        format_board_line(&state, "mahbot-1", "Fix * unclosed italic"),
        format_board_line(&state, "mahbot-2", "Use `git status` and *pair* ok"),
        format_board_line(&state, "mahbot-3", "<script>alert(1)</script> & tags"),
        format_board_line(&state, "mahbot-4", "Bold **crash** inside title"),
        format_board_line(&state, "mahbot-5", "[link](https://example.com/x?y=1&z=2)"),
        format_board_line(&state, "mahbot-6", "backtick ` unclosed"),
        format_board_line(&state, "mahbot-7", "Normal ticket"),
    ];
    let listing = lines.join("\n");
    let html = markdown_to_telegram_html(&listing);

    // Every line keeps its bold state and monospace ID.
    for line in html.split('\n') {
        assert!(line.starts_with("• <b>"), "state formatting lost: {line:?}");
        assert!(
            line.contains("</b> <code>") && line.contains("</code> "),
            "id formatting lost: {line:?}"
        );
    }
    // No unbalanced tags anywhere — an unclosed marker in a title must be
    // escaped, not emitted as a dangling tag (that would degrade the whole
    // listing to plain text via Telegram's HTML parse failure fallback).
    for (open, close) in [("<b>", "</b>"), ("<i>", "</i>"), ("<code>", "</code>")] {
        assert_eq!(
            html.matches(open).count(),
            html.matches(close).count(),
            "unbalanced {open}/{close} in: {html}"
        );
    }
}

// ── Inline formatting tests ──────────────────────────────────────

#[test]
fn test_inline_formatting() {
    struct Case {
        name: &'static str,
        input: &'static str,
        expected: &'static str,
    }
    let cases = vec![
        Case {
            name: "bold double asterisk",
            input: "**hello** world",
            expected: "<b>hello</b> world",
        },
        Case {
            name: "bold double underscore",
            input: "__hello__ world",
            expected: "<b>hello</b> world",
        },
        Case {
            name: "italic",
            input: "*hello* world",
            expected: "<i>hello</i> world",
        },
        Case {
            name: "inline code",
            input: "use `hello()` in your code",
            expected: "use <code>hello()</code> in your code",
        },
        Case {
            name: "strikethrough",
            input: "this is ~~wrong~~ fixed",
            expected: "this is <s>wrong</s> fixed",
        },
        Case {
            name: "combined",
            input: "**bold** and *italic* and `code` and ~~strike~~",
            expected: "<b>bold</b> and <i>italic</i> and <code>code</code> and <s>strike</s>",
        },
        Case {
            name: "bold inside text",
            input: "before **middle** after",
            expected: "before <b>middle</b> after",
        },
        Case {
            name: "escaping inner HTML",
            input: "**a < b & c > d**",
            expected: "<b>a &lt; b &amp; c &gt; d</b>",
        },
        // `**` without closing should be rendered literally
        Case {
            name: "unmatched double asterisk",
            input: "hello ** world",
            expected: "hello ** world",
        },
        // `*` without closing should be rendered literally
        Case {
            name: "unmatched single asterisk",
            input: "hello * world",
            expected: "hello * world",
        },
        // `***` is not a valid bold or italic construct; rendered literally
        Case {
            name: "triple asterisk",
            input: "***",
            expected: "***",
        },
        // `` ` `` without closing should be rendered literally (the opening ` is pushed as text)
        Case {
            name: "unmatched backtick",
            input: "hello ` world",
            expected: "hello ` world",
        },
        // ` `` ` (two backticks) — the first opens, the second closes (empty content), and since
        // the `end > 0` guard rejects empty matches, both are rendered literally.
        Case {
            name: "double backtick",
            input: "hello `` world",
            expected: "hello `` world",
        },
        // `~` without matching pair should be rendered literally
        Case {
            name: "unmatched tilde",
            input: "hello ~ world",
            expected: "hello ~ world",
        },
        // Bold takes priority over italic for `**`
        Case {
            name: "bold and italic overlap",
            input: "***bold**",
            expected: "<b>*bold</b>",
        },
    ];
    for case in cases {
        let result = markdown_to_telegram_html(case.input);
        assert_eq!(result, case.expected, "case: {}", case.name);
    }
}

#[tokio::test]
async fn parse_update_message_uses_chat_id_as_reply_target() {
    let ch = test_channel().await;
    let update = test_update(&[]);

    let msg = ch
        .parse_update_message(&update)
        .await
        .expect("message should parse");

    assert_eq!(msg.user_name, "alice");
    assert_eq!(msg.reply_target, "-100200300");
    assert_eq!(msg.content, "hello");
}

#[test]
fn parse_attachment_markers_tests() {
    let dir = tempfile::tempdir().unwrap();
    let png = dir.path().join("a.png");
    let ogg = dir.path().join("voice.ogg");
    let vid = dir.path().join("vid.mp4");
    std::fs::write(&png, b"fake-png").unwrap();
    std::fs::write(&ogg, b"fake-ogg").unwrap();
    std::fs::write(&vid, b"fake-mp4").unwrap();

    // Placeholder/inexistent targets stay as literal text (AC 1).
    let (cleaned, att) = parse_attachment_markers("use `[IMAGE:path]` or `[VIDEO:...]`");
    assert_eq!(cleaned, "use `[IMAGE:path]` or `[VIDEO:...]`");
    assert!(att.is_empty());
    // Directory targets are not regular files → prose.
    let (cleaned, att) = parse_attachment_markers(&format!("[IMAGE:{}]", dir.path().display()));
    assert_eq!(cleaned, format!("[IMAGE:{}]", dir.path().display()));
    assert!(att.is_empty());

    // Existing files become attachments (AC 2).
    let (cleaned, att) = parse_attachment_markers(&format!(
        "Here are files [IMAGE:{}] and [AUDIO:{}]",
        png.display(),
        ogg.display()
    ));
    assert_eq!(cleaned, "Here are files  and");
    assert_eq!(att.len(), 2);
    assert_eq!(att[0].kind, TelegramAttachmentKind::Image);
    assert_eq!(att[1].kind, TelegramAttachmentKind::Audio);

    // http(s) URLs become attachments (AC 2).
    let (cleaned, att) = parse_attachment_markers("See [VIDEO:https://example.com/vid.mp4]");
    assert_eq!(cleaned, "See");
    assert_eq!(att.len(), 1);
    assert_eq!(att[0].kind, TelegramAttachmentKind::Video);
    assert_eq!(att[0].target, "https://example.com/vid.mp4");

    // A bad marker doesn't abort valid ones in the same message (AC 3).
    let (cleaned, att) =
        parse_attachment_markers(&format!("[IMAGE:missing.png] ok [VIDEO:{}]", vid.display()));
    assert_eq!(cleaned, "[IMAGE:missing.png] ok");
    assert_eq!(att.len(), 1);
    assert_eq!(att[0].kind, TelegramAttachmentKind::Video);

    // Unknown markers kept as text; case-insensitive matching with a real file.
    let (cleaned, att) = parse_attachment_markers("Report [UNKNOWN:/tmp/a.bin]");
    assert_eq!(cleaned, "Report [UNKNOWN:/tmp/a.bin]");
    assert!(att.is_empty());
    let (cleaned, att) = parse_attachment_markers(&format!("[image:{}]", png.display()));
    assert_eq!(cleaned, "");
    assert_eq!(att.len(), 1);
    assert_eq!(att[0].kind, TelegramAttachmentKind::Image);
}

#[test]
fn parse_path_only_attachment_tests() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("snap.png");
    std::fs::write(&p, b"fake-png").unwrap();
    let parsed = parse_path_only_attachment(p.to_string_lossy().as_ref()).unwrap();
    assert_eq!(parsed.kind, TelegramAttachmentKind::Image);
    assert_eq!(parsed.target, p.to_string_lossy());
    assert!(parse_path_only_attachment("Screenshot saved to /tmp/snap.png").is_none());
}

#[test]
fn infer_attachment_kind_from_target_detects_document_extension() {
    assert_eq!(
        infer_attachment_kind_from_target("https://example.com/files/specs.pdf?download=1"),
        Some(TelegramAttachmentKind::Document)
    );
}

#[tokio::test]
async fn parse_update_message_denies_user_without_username() {
    let ch = test_channel().await;
    let update = test_update(&[(
        "message",
        serde_json::json!({
            "message_id": 9,
            "text": "ping",
            "from": {
                "id": 555
            },
            "chat": {
                "id": 12345
            }
        }),
    )]);

    assert!(
        ch.parse_update_message(&update).await.is_none(),
        "user without username should be denied"
    );
}

#[tokio::test]
async fn parse_update_message_extracts_thread_id_for_forum_topic() {
    let ch = test_channel().await;
    let update = test_update(&[(
        "message",
        serde_json::json!({
            "message_id": 42,
            "text": "hello from topic",
            "from": {
                "id": 555,
                "username": "alice"
            },
            "chat": {
                "id": -100_200_300
            },
            "message_thread_id": 789
        }),
    )]);

    let msg = ch
        .parse_update_message(&update)
        .await
        .expect("message with thread_id should parse");

    assert_eq!(msg.user_name, "alice");
    assert_eq!(msg.reply_target, "-100200300:789");
    assert_eq!(msg.content, "hello from topic");
}

/// Helper: create a callback_query sub-object for testing.
/// Overrides are applied as top-level keys on the callback_query.
fn test_callback_query(overrides: &[(&str, serde_json::Value)]) -> serde_json::Value {
    let mut cq = serde_json::json!({
        "id": "12345",
        "data": "set_model|test-model",
        "from": { "id": 555, "username": "alice" },
        "message": {
            "message_id": 100,
            "chat": { "id": -100_200_300 },
            "date": 1_700_000_000
        }
    });
    let obj = cq.as_object_mut().unwrap();
    for (key, value) in overrides {
        obj.insert(key.to_string(), value.clone());
    }
    cq
}

#[tokio::test]
async fn parse_callback_query_returns_message_with_extracted_fields() {
    let ch = test_channel().await;
    let cq = test_callback_query(&[]);

    let msg = ch
        .parse_callback_query(&cq)
        .await
        .expect("callback query should parse with valid user");

    assert_eq!(msg.user_name, "alice");
    assert_eq!(msg.reply_target, "-100200300");
    assert_eq!(msg.content, "set_model|test-model");
    assert_eq!(msg.channel, "telegram");
    assert_eq!(msg.callback_query_id.as_deref(), Some("12345"));
}

#[tokio::test]
async fn parse_callback_query_rejects_invalid_inputs() {
    let ch = test_channel().await;
    let unknown_user = serde_json::json!({ "id": 999, "username": "unknown_user" });
    let cases = [
        ("no data", [("data", serde_json::Value::Null)]),
        ("no message", [("message", serde_json::Value::Null)]),
        ("unknown user", [("from", unknown_user)]),
    ];
    for (name, overrides) in &cases {
        let cq = test_callback_query(overrides);
        assert!(
            ch.parse_callback_query(&cq).await.is_none(),
            "case {name}: expected rejection"
        );
    }
}

#[tokio::test]
async fn parse_callback_query_accepts_empty_data_and_null_id() {
    // Regression pair: the ? guard in parse_callback_query only rejects
    // absent/null data — NOT an empty string, which yields empty content
    // (distinct from process_updates' unwrap_or("") ACK check). A
    // null/absent callback_query_id likewise stays valid, becoming None
    // rather than Some — unlike missing data which rejects.
    let ch = test_channel().await;
    let cases = [
        (
            "empty-string data",
            [("data", serde_json::json!(""))],
            "",
            Some("12345"),
        ),
        (
            "null id",
            [("id", serde_json::Value::Null)],
            "set_model|test-model",
            None,
        ),
    ];
    for (name, overrides, content, cq_id) in &cases {
        let cq = test_callback_query(overrides);
        let msg = ch
            .parse_callback_query(&cq)
            .await
            .unwrap_or_else(|| panic!("case {name}: expected a valid message"));
        assert_eq!(msg.content, *content, "case {name}");
        assert_eq!(msg.callback_query_id.as_deref(), *cq_id, "case {name}");
    }
}

#[test]
fn telegram_message_splitting() {
    // basic: exact limit → no split
    assert_eq!(
        split_message_for_telegram(&"a".repeat(TELEGRAM_MAX_MESSAGE_LENGTH)).len(),
        1
    );
    assert!(split_message_for_telegram(&"a".repeat(TELEGRAM_MAX_MESSAGE_LENGTH + 1)).len() >= 2);
    let long = "a".repeat(5000);
    let parts = split_message_for_telegram(&long);
    assert!(parts.len() >= 2);
    assert_eq!(parts.join(""), long);
    assert!(split_message_for_telegram("   \n\n\t  ").len() <= 1);

    // edge: code block spanning boundary
    let msg = format!("```python\n{}```\nMore text", "x".repeat(4085));
    for p in &split_message_for_telegram(&msg) {
        assert!(p.len() <= TELEGRAM_MAX_MESSAGE_LENGTH);
    }
    // emoji at boundary
    let msg = format!("{}🎉🎊", "a".repeat(4094));
    for p in &split_message_for_telegram(&msg) {
        assert!(p.chars().count() <= TELEGRAM_MAX_MESSAGE_LENGTH);
    }
}

#[test]
fn newline_split_fallback_prevents_mid_word_break() {
    // Regression: when the only newline is in the first half of the
    // search window and no spaces exist, the old code would hard-split
    // mid-word. The newline fallback (tier 3) prevents this.
    let msg = format!("{}\n{}", "a".repeat(1000), "x".repeat(5000));
    let chunks = split_message_for_telegram(&msg);

    // All chunks must respect the length limit
    for (i, chunk) in chunks.iter().enumerate() {
        assert!(
            chunk.chars().count() <= TELEGRAM_MAX_MESSAGE_LENGTH,
            "chunk {} has {} chars (limit {})",
            i,
            chunk.chars().count(),
            TELEGRAM_MAX_MESSAGE_LENGTH,
        );
    }

    // Concatenation must reconstruct the original message
    assert_eq!(chunks.join(""), msg);

    // The first chunk must end with the newline (not split mid-word)
    assert!(
        chunks[0].ends_with('\n'),
        "first chunk should end with newline, got: {:?}",
        chunks[0].chars().rev().take(10).collect::<String>()
    );
}

#[test]
fn wrapped_chunks_respect_telegram_limit() {
    // Simulate send_text_chunks wrapping to verify no chunk exceeds the
    // 4096-char limit after continuation prefixes are prepended.
    //
    // Middle chunk overhead: "(continued)\n\n...\n\n(continues...)" = 28 chars
    // Last chunk overhead:  "(continued)\n\n..."                   = 13 chars
    // First chunk (multi):  "...\n\n(continues...)"               = 15 chars

    // Build a message designed to force 3+ chunks, then verify wrapping.
    let msg = format!("X{}X", "a".repeat(9000));
    let chunks = split_message_for_telegram(&msg);
    assert!(
        chunks.len() >= 3,
        "expected 3+ chunks to exercise all continuation variants"
    );

    for (i, chunk) in chunks.iter().enumerate() {
        let wrapped = wrap_chunk(chunk, i, chunks.len());
        assert!(
            wrapped.chars().count() <= 4096,
            "chunk {} wrapped length {} exceeds 4096",
            i,
            wrapped.chars().count()
        );
    }

    // Boundary: last chunk exactly at 4066 chars (new limit) — wrapping
    // produces 4066+13=4079 ≤ 4096.  Under the old code, a 4095-char last
    // chunk would wrap to 4108 > 4096.
    //
    // Intentionally uses raw format! rather than wrap_chunk() — this tests
    // the splitter's TELEGRAM_CONTINUATION_OVERHEAD constant, not the
    // wrapping helper. wrap_chunk would return the chunk bare for total==1.
    let boundary = "b".repeat(4066);
    let chunks = split_message_for_telegram(&boundary);
    assert_eq!(chunks.len(), 1, "4066-char message should not split");
    let wrapped = format!("(continued)\n\n{}", chunks[0]);
    assert!(
        wrapped.chars().count() <= 4096,
        "boundary wrapped: {} > 4096",
        wrapped.chars().count()
    );

    // Old bug reproducer: a 4095-char last chunk would wrap to 4108.
    // With the fix, a message that produces a ~4095-char last chunk
    // shouldn't exist; the splitter caps non-first chunks at 4066.
    let near_limit = "c".repeat(4096);
    let chunks = split_message_for_telegram(&near_limit);
    assert_eq!(chunks.len(), 1, "4096-char message should not split");
}

// ─────────────────────────────────────────────────────────────────────
// extract_sender_user_name tests
// ─────────────────────────────────────────────────────────────────────

#[test]
fn test_extract_sender_user_name() {
    let username =
        extract_sender_user_name(&serde_json::json!({"from": {"id": 123, "username": "alice"}}));
    assert_eq!(username, "alice");
    let username = extract_sender_user_name(&serde_json::json!({"from": {"id": 42}}));
    assert_eq!(username, "unknown");
}

// ─────────────────────────────────────────────────────────────────────
// extract_reply_context tests
// ─────────────────────────────────────────────────────────────────────

#[test]
fn test_extract_reply_context() {
    // text message reply
    let msg = serde_json::json!({
        "reply_to_message": {
            "from": { "username": "alice" },
            "text": "Hello world"
        }
    });
    let ctx = TelegramChannel::extract_reply_context(&msg).unwrap();
    assert_eq!(ctx, "> @alice:\n> Hello world");

    // voice message reply
    let msg = serde_json::json!({
        "reply_to_message": {
            "from": { "username": "bob" },
            "voice": { "file_id": "abc", "duration": 5 }
        }
    });
    let ctx = TelegramChannel::extract_reply_context(&msg).unwrap();
    assert_eq!(ctx, "> @bob:\n> [Voice message]");

    // no reply
    let msg = serde_json::json!({
        "text": "just a regular message"
    });
    assert!(TelegramChannel::extract_reply_context(&msg).is_none());

    // no username, uses first_name
    let msg = serde_json::json!({
        "reply_to_message": {
            "from": { "id": 999, "first_name": "Charlie" },
            "text": "Hi there"
        }
    });
    let ctx = TelegramChannel::extract_reply_context(&msg).unwrap();
    assert_eq!(ctx, "> Charlie:\n> Hi there");
}

#[tokio::test]
async fn parse_update_message_includes_reply_context() {
    let ch = test_channel().await;
    let update = test_update(&[(
        "message",
        serde_json::json!({
            "message_id": 10,
            "text": "translate this",
            "from": { "id": 1, "username": "alice" },
            "chat": { "id": 100, "type": "private" },
            "reply_to_message": {
                "from": { "username": "bot" },
                "text": "Bonjour le monde"
            }
        }),
    )]);
    let parsed = ch.parse_update_message(&update).await.unwrap();
    assert!(
        parsed.content.starts_with("> @bot:"),
        "content should start with quote: {}",
        parsed.content
    );
    assert!(
        parsed.content.contains("translate this"),
        "content should contain user text"
    );
    assert!(
        parsed.content.contains("Bonjour le monde"),
        "content should contain quoted text"
    );
}

// ── IncomingAttachment / parse_attachment_metadata tests ─────────

#[test]
fn test_parse_attachment_metadata() {
    // Document with all fields
    let att = TelegramChannel::parse_attachment_metadata(&serde_json::json!({
            "document": {"file_id": "BQ", "file_name": "report.pdf", "file_size": 12345, "mime_type": "application/pdf"}
        }))
        .unwrap();
    assert_eq!(att.kind, IncomingAttachmentKind::Document);
    assert_eq!(att.file_id, "BQ");
    assert_eq!(att.file_name.as_deref(), Some("report.pdf"));
    assert_eq!(att.file_size, Some(12345));
    assert_eq!(att.mime_type.as_deref(), Some("application/pdf"));
    assert!(att.caption.is_none());
    // Photo (picks largest by file_size)
    let att = TelegramChannel::parse_attachment_metadata(&serde_json::json!({
            "photo": [{"file_id": "small_id", "file_size": 100}, {"file_id": "large_id", "file_size": 2000}]
        })).unwrap();
    assert_eq!(att.kind, IncomingAttachmentKind::Photo);
    assert_eq!(att.file_id, "large_id");
    assert_eq!(att.file_size, Some(2000));
    // Caption extraction
    let att = TelegramChannel::parse_attachment_metadata(&serde_json::json!({
        "document": {"file_id": "doc_id", "file_name": "data.csv"}, "caption": "Monthly report"
    }))
    .unwrap();
    assert_eq!(att.caption.as_deref(), Some("Monthly report"));
    let att = TelegramChannel::parse_attachment_metadata(&serde_json::json!({
        "photo": [{"file_id": "photo_id", "file_size": 1000}], "caption": "Look at this"
    }))
    .unwrap();
    assert_eq!(att.caption.as_deref(), Some("Look at this"));
    // Document without optional fields
    let att = TelegramChannel::parse_attachment_metadata(&serde_json::json!({
        "document": {"file_id": "doc_no_name"}
    }))
    .unwrap();
    assert_eq!(att.kind, IncomingAttachmentKind::Document);
    assert_eq!(att.file_id, "doc_no_name");
    assert!(att.file_name.is_none());
    assert!(att.file_size.is_none());
    // Document with mime_type extraction
    let att = TelegramChannel::parse_attachment_metadata(&serde_json::json!({
        "document": {"file_id": "img_doc", "mime_type": "image/png"}
    }))
    .unwrap();
    assert_eq!(att.kind, IncomingAttachmentKind::Document);
    assert_eq!(att.mime_type.as_deref(), Some("image/png"));
    // Voice message
    let att = TelegramChannel::parse_attachment_metadata(
        &serde_json::json!({"voice": {"file_id": "v", "duration": 5}}),
    )
    .unwrap();
    assert_eq!(att.kind, IncomingAttachmentKind::Audio);
    assert_eq!(att.file_id, "v");
    assert!(att.file_name.is_none());
    // Video message
    let att = TelegramChannel::parse_attachment_metadata(&serde_json::json!({
        "video": {"file_id": "vid", "file_name": "clip.mp4", "file_size": 12345, "mime_type": "video/mp4", "duration": 4}
    }))
    .unwrap();
    assert_eq!(att.kind, IncomingAttachmentKind::Video);
    assert_eq!(att.file_id, "vid");
    assert_eq!(att.file_name.as_deref(), Some("clip.mp4"));
    assert_eq!(att.file_size, Some(12345));
    assert_eq!(att.mime_type.as_deref(), Some("video/mp4"));
    // Video note (no file_name/mime_type)
    let att = TelegramChannel::parse_attachment_metadata(
        &serde_json::json!({"video_note": {"file_id": "vn", "duration": 3, "file_size": 999}}),
    )
    .unwrap();
    assert_eq!(att.kind, IncomingAttachmentKind::Video);
    assert_eq!(att.file_id, "vn");
    assert!(att.file_name.is_none());
    assert!(att.mime_type.is_none());
    // Animation (GIF)
    let att = TelegramChannel::parse_attachment_metadata(&serde_json::json!({
        "animation": {"file_id": "anim", "file_name": "sticker.gif", "mime_type": "video/mp4"}
    }))
    .unwrap();
    assert_eq!(att.kind, IncomingAttachmentKind::Video);
    assert_eq!(att.file_id, "anim");
    assert_eq!(att.file_name.as_deref(), Some("sticker.gif"));
    // Audio message
    let att = TelegramChannel::parse_attachment_metadata(
        &serde_json::json!({"audio": {"file_id": "a", "file_name": "song.mp3", "file_size": 999}}),
    )
    .unwrap();
    assert_eq!(att.kind, IncomingAttachmentKind::Audio);
    assert_eq!(att.file_id, "a");
    assert_eq!(att.file_name.as_deref(), Some("song.mp3"));
    assert_eq!(att.file_size, Some(999));
    // No attachment cases
    assert!(
        TelegramChannel::parse_attachment_metadata(&serde_json::json!({"text": "Hello"})).is_none()
    );
    assert!(
        TelegramChannel::parse_attachment_metadata(&serde_json::json!({"photo": []})).is_none()
    );
}

// ── Attachment content format tests ──────────────────────────────

#[test]
#[allow(clippy::too_many_lines)]
fn attachment_content_format_rules() {
    // photo → [IMAGE:]
    let c = format_attachment_content(
        IncomingAttachmentKind::Photo,
        "photo.jpg",
        std::path::Path::new("/tmp/workspace/photo.jpg"),
        None,
    );
    assert_eq!(c, "[IMAGE:/tmp/workspace/photo.jpg]");
    // document → [Document: name] /path
    let c = format_attachment_content(
        IncomingAttachmentKind::Document,
        "report.pdf",
        std::path::Path::new("/tmp/workspace/report.pdf"),
        None,
    );
    assert_eq!(c, "[Document: report.pdf] /tmp/workspace/report.pdf");
    assert!(!c.contains("[IMAGE:"));
    // markdown files never produce [IMAGE:] even when classified as Photo
    let c = format_attachment_content(
        IncomingAttachmentKind::Photo,
        "notes.md",
        std::path::Path::new("/tmp/workspace/notes.md"),
        None,
    );
    assert!(!c.contains("[IMAGE:"));
    assert!(c.starts_with("[Document:"));
    // non-image files classified as Photo fall back to [Document:]
    for (filename, path) in [
        ("file.md", "/tmp/workspace/file.md"),
        ("file.txt", "/tmp/workspace/file.txt"),
        ("file.pdf", "/tmp/workspace/file.pdf"),
        ("file.csv", "/tmp/workspace/file.csv"),
        ("file.json", "/tmp/workspace/file.json"),
        ("file.zip", "/tmp/workspace/file.zip"),
        ("file", "/tmp/workspace/file"),
    ] {
        let c = format_attachment_content(
            IncomingAttachmentKind::Photo,
            filename,
            std::path::Path::new(path),
            None,
        );
        assert!(
            !c.contains("[IMAGE:"),
            "{filename}: should not get [IMAGE:]"
        );
        assert!(
            c.starts_with("[Document:"),
            "{filename}: should use [Document:]"
        );
    }
    // image extensions produce [IMAGE:] (PNG/JPEG/WebP only — gif/bmp are
    // deliberately not routed as images)
    for ext in ["png", "jpg", "jpeg", "webp"] {
        let filename = format!("photo.{ext}");
        let c = format_attachment_content(
            IncomingAttachmentKind::Photo,
            &filename,
            std::path::Path::new(&format!("/tmp/workspace/{filename}")),
            None,
        );
        assert!(c.starts_with("[IMAGE:"), "{ext}: should get [IMAGE:]");
    }
    // Document kind + .jpg extension → [IMAGE:] (not [Document:])
    let c = format_attachment_content(
        IncomingAttachmentKind::Document,
        "image.jpg",
        std::path::Path::new("/tmp/workspace/image.jpg"),
        None,
    );
    assert_eq!(c, "[IMAGE:/tmp/workspace/image.jpg]");
    // Document kind + no extension + jpeg MIME types → [IMAGE:] (mime
    // fallback). Both the canonical image/jpeg and the legacy image/jpg
    // alias are admitted.
    for mime in ["image/jpeg", "image/jpg"] {
        let c = format_attachment_content(
            IncomingAttachmentKind::Document,
            "image_no_ext",
            std::path::Path::new("/tmp/workspace/image_no_ext"),
            Some(mime),
        );
        assert_eq!(c, "[IMAGE:/tmp/workspace/image_no_ext]", "{mime}");
    }
    // gif/bmp MIME types are NOT routed as images (mime fallback whitelist) —
    // they fall through to [Document:]. The legacy image/x-ms-bmp alias is
    // excluded too.
    for mime in ["image/gif", "image/bmp", "image/x-ms-bmp"] {
        let c = format_attachment_content(
            IncomingAttachmentKind::Document,
            "anim_no_ext",
            std::path::Path::new("/tmp/workspace/anim_no_ext"),
            Some(mime),
        );
        assert!(!c.contains("[IMAGE:"), "{mime}: should not get [IMAGE:]");
        assert!(
            c.starts_with("[Document:"),
            "{mime}: should use [Document:]"
        );
    }
    // Audio kind produces [AUDIO:] marker regardless of extension
    let c = format_attachment_content(
        IncomingAttachmentKind::Audio,
        "voice.ogg",
        std::path::Path::new("/tmp/workspace/voice.ogg"),
        None,
    );
    assert_eq!(c, "[AUDIO:/tmp/workspace/voice.ogg]");
    let c = format_attachment_content(
        IncomingAttachmentKind::Audio,
        "song.mp3",
        std::path::Path::new("/tmp/workspace/song.mp3"),
        Some("audio/mpeg"),
    );
    assert_eq!(c, "[AUDIO:/tmp/workspace/song.mp3]");
    // Video kind produces [VIDEO:] marker
    let c = format_attachment_content(
        IncomingAttachmentKind::Video,
        "clip.mp4",
        std::path::Path::new("/tmp/workspace/clip.mp4"),
        None,
    );
    assert_eq!(c, "[VIDEO:/tmp/workspace/clip.mp4]");
    // Document kind + video extension → [VIDEO:] (like image extension routing)
    let c = format_attachment_content(
        IncomingAttachmentKind::Document,
        "clip.mp4",
        std::path::Path::new("/tmp/workspace/clip.mp4"),
        None,
    );
    assert_eq!(c, "[VIDEO:/tmp/workspace/clip.mp4]");
    // Document kind + no extension + mime_type "video/mp4" → [VIDEO:] (mime fallback)
    let c = format_attachment_content(
        IncomingAttachmentKind::Document,
        "clip_no_ext",
        std::path::Path::new("/tmp/workspace/clip_no_ext"),
        Some("video/mp4"),
    );
    assert_eq!(c, "[VIDEO:/tmp/workspace/clip_no_ext]");
    // Video kind with a non-video extension falls back to [Document:]
    let c = format_attachment_content(
        IncomingAttachmentKind::Video,
        "notes.txt",
        std::path::Path::new("/tmp/workspace/notes.txt"),
        None,
    );
    assert_eq!(c, "[Document: notes.txt] /tmp/workspace/notes.txt");
}

#[test]
fn attachment_multimodal_and_helpers() {
    // has_extension over telegram's receive-path IMAGE_EXTENSIONS
    for p in [
        "photo.png",
        "photo.jpg",
        "photo.jpeg",
        "photo.webp",
        "PHOTO.PNG",
    ] {
        assert!(crate::util::has_extension(
            std::path::Path::new(p),
            super::IMAGE_EXTENSIONS
        ));
    }
    // gif/bmp are NOT image extensions anymore (codec trim)
    for p in ["photo.gif", "photo.bmp"] {
        assert!(!crate::util::has_extension(
            std::path::Path::new(p),
            super::IMAGE_EXTENSIONS
        ));
    }
    for p in ["file.md", "file.txt", "file.pdf", "file.csv", "file"] {
        assert!(!crate::util::has_extension(
            std::path::Path::new(p),
            super::IMAGE_EXTENSIONS
        ));
    }
    // is_video_extension
    for p in [
        "clip.mp4",
        "clip.mov",
        "clip.mkv",
        "clip.avi",
        "clip.webm",
        "CLIP.MP4",
    ] {
        assert!(crate::util::is_video_extension(std::path::Path::new(p)));
    }
    for p in ["file.md", "file.png", "file", "clip.mpg"] {
        assert!(!crate::util::is_video_extension(std::path::Path::new(p)));
    }
    // photo with caption
    let content = format!(
        "[IMAGE:{}]\n\nLook at this screenshot",
        std::path::Path::new("/tmp/workspace/photo.jpg").display()
    );
    assert_eq!(
        content,
        "[IMAGE:/tmp/workspace/photo.jpg]\n\nLook at this screenshot"
    );
}

#[test]
fn video_filename_normalization() {
    // GIF-picker animations: ".gif" file_name, H.264 MP4 bytes → mp4 extension
    assert_eq!(
        normalize_video_filename(
            IncomingAttachmentKind::Video,
            "tenor.gif",
            Some("video/mp4")
        ),
        "tenor.mp4"
    );
    // Recognized video extensions pass through untouched
    assert_eq!(
        normalize_video_filename(
            IncomingAttachmentKind::Video,
            "clip.webm",
            Some("video/webm")
        ),
        "clip.webm"
    );
    // No-extension names gain .mp4
    assert_eq!(
        normalize_video_filename(IncomingAttachmentKind::Video, "video_123_45", None),
        "video_123_45.mp4"
    );
    // Documents without a video MIME keep their name
    assert_eq!(
        normalize_video_filename(IncomingAttachmentKind::Document, "tenor.gif", None),
        "tenor.gif"
    );
    // Documents with a video MIME and a non-video extension get a video
    // extension derived from the MIME (webm not mislabeled as mp4)
    assert_eq!(
        normalize_video_filename(
            IncomingAttachmentKind::Document,
            "clip.xyz",
            Some("video/mp4")
        ),
        "clip.mp4"
    );
    assert_eq!(
        normalize_video_filename(
            IncomingAttachmentKind::Document,
            "clip.xyz",
            Some("video/webm")
        ),
        "clip.webm"
    );
}

// ── Forwarded message tests ─────────────────────────────────────

#[tokio::test]
async fn forward_attribution() {
    let ch = test_channel().await;

    // forwarded from user with username
    let update = test_update(&[(
        "message",
        serde_json::json!({
            "message_id": 50,
            "text": "Check this out",
            "from": { "id": 1, "username": "alice" },
            "chat": { "id": 999 },
            "forward_from": {
                "id": 42,
                "first_name": "Bob",
                "username": "bob"
            },
            "forward_date": 1_700_000_000
        }),
    )]);
    let msg = ch.parse_update_message(&update).await.unwrap();
    assert_eq!(msg.content, "[Forwarded from @bob] Check this out");

    // forwarded from channel
    let update = test_update(&[(
        "message",
        serde_json::json!({
            "message_id": 51,
            "text": "Breaking news",
            "from": { "id": 1, "username": "alice" },
            "chat": { "id": 999 },
            "forward_from_chat": {
                "id": -1_001_234_567_890_i64,
                "title": "Daily News",
                "username": "dailynews",
                "type": "channel"
            },
            "forward_date": 1_700_000_000
        }),
    )]);
    let msg = ch.parse_update_message(&update).await.unwrap();
    assert_eq!(
        msg.content,
        "[Forwarded from channel: Daily News] Breaking news"
    );

    // forwarded hidden sender
    let update = test_update(&[(
        "message",
        serde_json::json!({
            "message_id": 52,
            "text": "Secret tip",
            "from": { "id": 1, "username": "alice" },
            "chat": { "id": 999 },
            "forward_sender_name": "Hidden User",
            "forward_date": 1_700_000_000
        }),
    )]);
    let msg = ch.parse_update_message(&update).await.unwrap();
    assert_eq!(msg.content, "[Forwarded from Hidden User] Secret tip");

    // non-forwarded unaffected
    let update = test_update(&[(
        "message",
        serde_json::json!({
            "message_id": 53,
            "text": "Normal message",
            "from": { "id": 1, "username": "alice" },
            "chat": { "id": 999 }
        }),
    )]);
    let msg = ch.parse_update_message(&update).await.unwrap();
    assert_eq!(msg.content, "Normal message");

    // forwarded from user without username
    let update = test_update(&[(
        "message",
        serde_json::json!({
            "message_id": 54,
            "text": "Hello there",
            "from": { "id": 1, "username": "alice" },
            "chat": { "id": 999 },
            "forward_from": {
                "id": 77,
                "first_name": "Charlie"
            },
            "forward_date": 1_700_000_000
        }),
    )]);
    let msg = ch.parse_update_message(&update).await.unwrap();
    assert_eq!(msg.content, "[Forwarded from Charlie] Hello there");

    // forwarded photo with attribution
    let message = serde_json::json!({
        "message_id": 60,
        "from": { "id": 1, "username": "alice" },
        "chat": { "id": 999 },
        "photo": [
            { "file_id": "abc123", "file_unique_id": "u1", "width": 320, "height": 240 }
        ],
        "forward_from": {
            "id": 42,
            "username": "bob"
        },
        "forward_date": 1_700_000_000
    });
    let attr =
        TelegramChannel::format_forward_attribution(&message).expect("should detect forward");
    assert_eq!(attr, "[Forwarded from @bob] ");
    let photo_content = "[IMAGE:/tmp/photo.jpg]".to_string();
    let content = format!("{attr}{photo_content}");
    assert_eq!(content, "[Forwarded from @bob] [IMAGE:/tmp/photo.jpg]");
}

// ── strip_html_tags tests ──────────────────────────────────

#[test]
fn test_strip_html_tags() {
    struct Case {
        name: &'static str,
        input: &'static str,
        expected: &'static str,
    }
    let cases = vec![
        Case {
            name: "empty string",
            input: "",
            expected: "",
        },
        Case {
            name: "plain text",
            input: "hello world",
            expected: "hello world",
        },
        Case {
            name: "simple tag",
            input: "<b>bold</b>",
            expected: "bold",
        },
        Case {
            name: "nested tags",
            input: "<div><span>text</span></div>",
            expected: "text",
        },
        Case {
            name: "self-closing tag",
            input: "before<br/>after",
            expected: "beforeafter",
        },
        // Regression: '<' in tag starts in_tag, then '>' inside attribute
        // value must NOT close the tag.
        Case {
            name: "gt in double-quoted attribute",
            input: "<a title=\"a > b\">link</a>",
            expected: "link",
        },
        Case {
            name: "gt in single-quoted attribute",
            input: "<a title='a > b'>link</a>",
            expected: "link",
        },
        // Double-quoted attribute containing single quotes
        Case {
            name: "mixed quotes - double with single inside",
            input: "<a title=\"he said 'hello'\">text</a>",
            expected: "text",
        },
        // Single-quoted attribute containing double quotes
        Case {
            name: "mixed quotes - single with double inside",
            input: "<a title='he said \"hello\"'>text</a>",
            expected: "text",
        },
        // Multiple attributes, some with '>' inside quoted values
        Case {
            name: "multiple attrs with gt",
            input: "<input type=\"text\" value=\"a > b\" placeholder=\"x > y\">",
            expected: "",
        },
        // '>' that is not inside a tag should be preserved as text
        Case {
            name: "gt outside tag",
            input: "a > b",
            expected: "a > b",
        },
        // '<' that is not part of a tag should start tag mode
        // (pre-existing behavior: bare '<' starts tag stripping)
        Case {
            name: "lt outside tag",
            input: "a < b",
            expected: "a ",
        },
        Case {
            name: "html comment",
            input: "<!-- comment -->visible",
            expected: "visible",
        },
        // Realistic mixed content with tags and text
        Case {
            name: "mixed content",
            input: "Hello <b>world</b>, check <a href=\"https://example.com?q=a > b\">this</a> out!",
            expected: "Hello world, check this out!",
        },
    ];
    for case in cases {
        let result = strip_html_tags(case.input);
        assert_eq!(result, case.expected, "case: {}", case.name);
    }
}

// ── extend_past_open_tag tests ─────────────────────────────

#[allow(clippy::too_many_lines)]
#[test]
fn test_extend_past_open_tag() {
    struct Case {
        name: &'static str,
        input: &'static str,
        pos: usize,
        expected: Option<usize>,
    }
    let cases = vec![
        // No '<' before pos → None
        Case {
            name: "no tag near pos",
            input: "hello world",
            pos: 5,
            expected: None,
        },
        // <b>hello
        // 01234567
        // pos=1 (inside <b>, before '>') → extend past '>' at 2
        Case {
            name: "inside simple tag before gt",
            input: "<b>hello",
            pos: 1,
            expected: Some(3),
        },
        // pos=2 (at the '>' itself) → extend past it
        Case {
            name: "inside simple tag at gt",
            input: "<b>hello",
            pos: 2,
            expected: Some(3),
        },
        // <b>hello
        // 01234567
        // pos=3 (at 'h', past the '>') → None
        Case {
            name: "after simple tag at h",
            input: "<b>hello",
            pos: 3,
            expected: None,
        },
        // pos further into text → None
        Case {
            name: "after simple tag further",
            input: "<b>hello",
            pos: 5,
            expected: None,
        },
        // No '>' exists after the '<' → None (can't extend)
        Case {
            name: "no closing gt",
            input: "<div",
            pos: 3,
            expected: None,
        },
        // Regression: '>' inside double-quoted attribute must not be
        // treated as tag closer. The real '>' is at index 16.
        // <a title="a > b">text
        // 012345678901234567890   (indices)
        //           111111111122
        //      '>' at 12 is inside attribute, real '>' at 16
        // pos=13 (inside tag, past the quoted '>', before real '>') → extend to 17
        Case {
            name: "gt in double-quoted attr before real gt",
            input: "<a title=\"a > b\">text",
            pos: 13,
            expected: Some(17),
        },
        // pos=16 (at the real '>') → extend past it
        Case {
            name: "gt in double-quoted attr at real gt",
            input: "<a title=\"a > b\">text",
            pos: 16,
            expected: Some(17),
        },
        // pos after the real closing '>' → None
        // <a title="a > b">text
        // 012345678901234567890
        //           111111111122
        // real '>' at 16, 't' at 17
        Case {
            name: "after closed tag with gt in attr at 17",
            input: "<a title=\"a > b\">text",
            pos: 17,
            expected: None,
        },
        Case {
            name: "after closed tag with gt in attr at 20",
            input: "<a title=\"a > b\">text",
            pos: 20,
            expected: None,
        },
        // Same scenario with single quotes
        // <a title='a > b'>text
        // real '>' at 16
        Case {
            name: "gt in single-quoted attr",
            input: "<a title='a > b'>text",
            pos: 13,
            expected: Some(17),
        },
        // Double-quoted attr containing single quotes: '>' inside should still be
        // treated as quoted because we're inside double quotes, not single.
        // <a title="he said 'stop'">text
        // real '>' at 25
        // pos=17 (inside the attribute, past the single quotes) → extend past real '>'
        Case {
            name: "mixed quotes",
            input: "<a title=\"he said 'stop'\">text",
            pos: 17,
            expected: Some(26),
        },
        // <div><span>text
        // 012345678901234
        // last '<' at 5 (<span>), its '>' at 10.
        // pos=11 (after both tags are closed) → None
        Case {
            name: "after nested tags at 11",
            input: "<div><span>text",
            pos: 11,
            expected: None,
        },
        // pos=15 (end of string, still after both tags) → None
        Case {
            name: "after nested tags at 15",
            input: "<div><span>text",
            pos: 15,
            expected: None,
        },
        // <div><span>text
        // pos=6 (inside <span>, before its '>') → extend past '>' at 10
        Case {
            name: "inside nested tag",
            input: "<div><span>text",
            pos: 6,
            expected: Some(11),
        },
        // |<b>text → pos=0 is before any '<'
        Case {
            name: "pos at start",
            input: "<b>text",
            pos: 0,
            expected: None,
        },
    ];
    for case in cases {
        let result = extend_past_open_tag(case.input, case.pos);
        assert_eq!(result, case.expected, "case: {}", case.name);
    }
}

// ── decode_action ─────────────────────────────────────────────────────

#[test]
fn test_decode_action() {
    struct Case {
        name: &'static str,
        input: &'static str,
        expected: Option<(&'static str, &'static str)>,
    }

    let cases = [
        Case {
            name: "with payload",
            input: "__act__set_image_model|google/gemini-3.1-flash-image-preview",
            expected: Some(("set_image_model", "google/gemini-3.1-flash-image-preview")),
        },
        Case {
            name: "empty payload pipe",
            input: "__act__clear_session|",
            expected: Some(("clear_session", "")),
        },
        Case {
            name: "no pipe",
            input: "__act__clear_session",
            expected: Some(("clear_session", "")),
        },
        Case {
            name: "rejects non prefix",
            input: "random_text",
            expected: None,
        },
        Case {
            name: "rejects empty",
            input: "",
            expected: None,
        },
    ];

    for case in &cases {
        let result = decode_action(case.input);
        let expected = case
            .expected
            .map(|(action, payload)| (action.to_string(), payload.to_string()));
        assert_eq!(result, expected, "case: {}", case.name);
    }
}

// ── GUI message → Telegram mirror tests ─────────────────────────────
//
// These tests verify that `mirror_gui_message_to_telegram` returns
// early (without sending) for each guard condition, and that
// blockquote-format messages are correctly sent to the user's Telegram
// bindings. They are serialized via [`MIRROR_TEST_LOCK`] because the
// channel registry and store singletons are global.

use crate::util::UnwrapPoison;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

/// Serialization lock for all mirror tests — these tests share the global
/// [`CHANNEL_REGISTRY`] and store singletons, so they must run one at a time.
/// Uses `tokio::sync::Mutex` to avoid blocking worker threads while held
/// across await points.
static MIRROR_TEST_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

async fn acquire_mirror_lock() -> tokio::sync::MutexGuard<'static, ()> {
    MIRROR_TEST_LOCK
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

/// A spy channel that records sent messages in a shared Vec.
struct SpyChannel {
    sent: Arc<Mutex<Vec<SendMessage>>>,
}

#[async_trait]
impl crate::Channel for SpyChannel {
    async fn send(&self, message: &SendMessage) -> anyhow::Result<()> {
        self.sent.lock().unwrap_poison().push(message.clone());
        Ok(())
    }

    async fn listen(&self, _tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "telegram"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Set up the channel registry with a spy Telegram channel and return a
/// shared sent-messages buffer. Idempotent — safe to call from every test.
fn setup_spy_channel() -> &'static Arc<Mutex<Vec<SendMessage>>> {
    static SPY_SENT: OnceLock<Arc<Mutex<Vec<SendMessage>>>> = OnceLock::new();
    SPY_SENT.get_or_init(|| {
        let sent = Arc::new(Mutex::new(Vec::new()));
        let registry = crate::CHANNEL_REGISTRY.get_or_init(crate::ChannelRegistry::default);
        registry.register(Arc::new(SpyChannel {
            sent: Arc::clone(&sent),
        }) as Arc<dyn crate::Channel>);
        sent
    })
}

/// Ensure the user store has a test user with a Telegram binding and
/// reply_target. Idempotent. `ctx` labels setup panics for attribution.
async fn setup_user_with_telegram_binding(user_name: &str, reply_target: &str, ctx: &str) {
    use crate::users::store;
    let store = store();
    let all_roles = crate::Role::iter().collect::<Vec<_>>();
    store
        .add_user(user_name, Some("full"), &all_roles)
        .await
        .unwrap_or_else(|e| panic!("{ctx}: add_user: {e}"));
    store
        .bind_channel(user_name, "telegram", user_name)
        .await
        .unwrap_or_else(|e| panic!("{ctx}: bind_channel: {e}"));
    store
        .update_channel_contact("telegram", user_name, reply_target)
        .await
        .unwrap_or_else(|e| panic!("{ctx}: update_channel_contact: {e}"));
}

/// Three-line preamble shared by all mirror tests: acquire the serialization
/// lock, initialise test stores, and set up the spy channel. Returns the spy
/// sent-messages buffer and the lock guard (kept alive for the test duration).
async fn setup_mirror_test_env() -> (
    &'static Arc<Mutex<Vec<SendMessage>>>,
    tokio::sync::MutexGuard<'static, ()>,
) {
    let lock = acquire_mirror_lock().await;
    crate::util::test::init_test_stores().await;
    let sent = setup_spy_channel();
    (sent, lock)
}

fn gui_msg(user_name: &str, content: &str) -> ChannelMessage {
    ChannelMessage {
        user_name: user_name.to_string(),
        reply_target: String::new(),
        content: content.to_string(),
        channel: "gui".to_string(),
        workspace: "test".to_string(),
        optimistic_id: None,
        callback_query_id: None,
    }
}

fn telegram_msg(user_name: &str, content: &str) -> ChannelMessage {
    ChannelMessage {
        user_name: user_name.to_string(),
        reply_target: "chat:thread".to_string(),
        content: content.to_string(),
        channel: "telegram".to_string(),
        workspace: "test".to_string(),
        optimistic_id: None,
        callback_query_id: None,
    }
}

fn voice_msg(user_name: &str, content: &str) -> ChannelMessage {
    ChannelMessage {
        user_name: user_name.to_string(),
        reply_target: String::new(),
        content: content.to_string(),
        channel: "voice".to_string(),
        workspace: "test".to_string(),
        optimistic_id: None,
        callback_query_id: None,
    }
}

// ── Guard tests: early-return conditions ─────────────────────────────

/// Per-case setup for `assert_mirror_skips`, applied after the shared
/// mirror env is initialised (users::store() panics pre-init).
enum MirrorSkipSetup {
    BoundTo(&'static str, &'static str),
    Unbound(&'static str),
    BoundNoTarget(&'static str),
}

/// Mirror `msg` and assert nothing was sent to the recipient implied by
/// `setup` (reply_target when bound, user name otherwise). Each call
/// acquires the serialization lock; stores and spy are idempotent.
async fn assert_mirror_skips(setup: MirrorSkipSetup, msg: &ChannelMessage, reason: &str) {
    let (sent, _lock) = setup_mirror_test_env().await;
    let (user, filter_recipient) = match setup {
        MirrorSkipSetup::BoundTo(u, t) => {
            setup_user_with_telegram_binding(u, t, &format!("case {reason}")).await;
            (u, t)
        }
        MirrorSkipSetup::Unbound(u) => {
            let s = crate::users::store();
            s.add_user(u, None, &[])
                .await
                .unwrap_or_else(|e| panic!("case {reason}: add_user: {e}"));
            (u, u)
        }
        MirrorSkipSetup::BoundNoTarget(u) => {
            let s = crate::users::store();
            s.add_user(u, None, &[])
                .await
                .unwrap_or_else(|e| panic!("case {reason}: add_user: {e}"));
            s.bind_channel(u, "telegram", u)
                .await
                .unwrap_or_else(|e| panic!("case {reason}: bind_channel: {e}"));
            (u, u)
        }
    };
    // msg.user_name must match the setup user, or the filter matches nothing.
    assert_eq!(msg.user_name, user, "case {reason}");
    super::mirror_gui_message_to_telegram(msg).await;
    let guard = sent.lock().unwrap_poison();
    let our_msgs: Vec<_> = guard
        .iter()
        .filter(|m| m.recipient == filter_recipient)
        .collect();
    assert!(
        our_msgs.is_empty(),
        "case {reason}: got {} message(s)",
        our_msgs.len()
    );
}

#[tokio::test]
async fn mirror_skips_guard_cases() {
    // Each call acquires the serialization lock independently; the global
    // stores (OnceCell) and spy channel (OnceLock) are identical across
    // `setup_mirror_test_env()` calls.
    assert_mirror_skips(
        MirrorSkipSetup::BoundTo("skip_telegram", "target_non_gui"),
        &telegram_msg("skip_telegram", "hello from telegram"),
        "Telegram-originated messages should not send (voice is the only non-GUI source accepted)",
    )
    .await;
    assert_mirror_skips(
        MirrorSkipSetup::BoundTo("skip_ew", "target_empty_ws"),
        &gui_msg("skip_ew", ""),
        "empty content should not send",
    )
    .await;
    assert_mirror_skips(
        MirrorSkipSetup::BoundTo("skip_ew", "target_empty_ws"),
        &gui_msg("skip_ew", "   \t\n  "),
        "whitespace content should not send",
    )
    .await;
    // Create the user but DO NOT bind a Telegram channel; no bindings
    // means no messages for this user at all, so its name is the filter.
    assert_mirror_skips(
        MirrorSkipSetup::Unbound("no_binding"),
        &gui_msg("no_binding", "hello"),
        "user with no bindings should not send",
    )
    .await;
    // Bind a Telegram channel but don't set reply_target (skip
    // update_channel_contact → reply_target stays NULL).
    assert_mirror_skips(
        MirrorSkipSetup::BoundNoTarget("no_target"),
        &gui_msg("no_target", "hello"),
        "binding without reply_target should not send",
    )
    .await;
    // Full binding: exercises the media-only guard (content emptied by
    // marker stripping), not the no-bindings guard.
    assert_mirror_skips(
        MirrorSkipSetup::BoundTo("media_only", "target_media"),
        &gui_msg("media_only", "[IMAGE:/path/to/img.png]"),
        "media-only content should not send",
    )
    .await;
}

// ── Happy path tests ─────────────────────────────────────────────────

#[tokio::test]
async fn sends_blockquote_to_single_binding() {
    let (sent, _lock) = setup_mirror_test_env().await;
    setup_user_with_telegram_binding("single_user", "unique_single", "single_user").await;

    let msg = gui_msg("single_user", "Hello, world!");
    super::mirror_gui_message_to_telegram(&msg).await;

    let guard = sent.lock().unwrap_poison();
    // Filter to our test's messages by recipient.
    let our_msgs: Vec<_> = guard
        .iter()
        .filter(|m| m.recipient == "unique_single")
        .collect();
    assert_eq!(our_msgs.len(), 1, "expected exactly one message");
    assert_eq!(
        our_msgs[0].content,
        "<blockquote>\nHello, world!\n</blockquote>"
    );
    assert!(our_msgs[0].reply_markup.is_none());
}

#[tokio::test]
async fn mirrors_voice_transcript_to_telegram() {
    let (sent, _lock) = setup_mirror_test_env().await;
    setup_user_with_telegram_binding("voice_user", "unique_voice", "voice_user").await;

    // Voice-originated messages (GUI mic-button recording / wake-word
    // command) must mirror exactly like GUI text: same recipient bindings,
    // same blockquote format, no audio payload.
    let msg = voice_msg("voice_user", "Record this voice note");
    super::mirror_gui_message_to_telegram(&msg).await;

    let guard = sent.lock().unwrap_poison();
    let our_msgs: Vec<_> = guard
        .iter()
        .filter(|m| m.recipient == "unique_voice")
        .collect();
    assert_eq!(our_msgs.len(), 1, "expected exactly one message");
    assert_eq!(
        our_msgs[0].content,
        "<blockquote>\nRecord this voice note\n</blockquote>"
    );
    assert!(our_msgs[0].reply_markup.is_none());
}

#[tokio::test]
async fn sends_to_multiple_telegram_bindings() {
    let (sent, _lock) = setup_mirror_test_env().await;
    let store = crate::users::store();
    store
        .add_user("multi_user", None, &[])
        .await
        .expect("add_user");
    // Bind two Telegram accounts with unique recipients.
    store
        .bind_channel("multi_user", "telegram", "multi_user_1")
        .await
        .expect("bind_channel_1");
    store
        .bind_channel("multi_user", "telegram", "multi_user_2")
        .await
        .expect("bind_channel_2");
    store
        .update_channel_contact("telegram", "multi_user_1", "unique_multi_a")
        .await
        .expect("update_channel_contact_1");
    store
        .update_channel_contact("telegram", "multi_user_2", "unique_multi_b")
        .await
        .expect("update_channel_contact_2");

    let msg = gui_msg("multi_user", "Hi both!");
    super::mirror_gui_message_to_telegram(&msg).await;

    let guard = sent.lock().unwrap_poison();
    let our_msgs: Vec<_> = guard
        .iter()
        .filter(|m| m.recipient == "unique_multi_a" || m.recipient == "unique_multi_b")
        .collect();
    assert_eq!(our_msgs.len(), 2, "expected two messages (one per binding)");
    // Both should have the same content.
    for m in &our_msgs {
        assert_eq!(m.content, "<blockquote>\nHi both!\n</blockquote>");
    }
    let recipients: Vec<&str> = our_msgs.iter().map(|m| m.recipient.as_str()).collect();
    assert!(recipients.contains(&"unique_multi_a"));
    assert!(recipients.contains(&"unique_multi_b"));
}

/// Shared helper for media-marker-stripping tests.
///
/// Sets up the mirror test environment, binds `user_name` to `reply_target` in the
/// Telegram channel, mirrors a GUI message with `content`, and asserts that the quoted
/// output equals `expected_quote`.
async fn assert_mirror_strips_markers(
    user_name: &str,
    reply_target: &str,
    content: &str,
    expected_quote: &str,
) {
    let (sent, _lock) = setup_mirror_test_env().await;
    setup_user_with_telegram_binding(user_name, reply_target, user_name).await;

    let msg = gui_msg(user_name, content);
    super::mirror_gui_message_to_telegram(&msg).await;

    let guard = sent.lock().unwrap_poison();
    let our_msgs: Vec<_> = guard
        .iter()
        .filter(|m| m.recipient == reply_target)
        .collect();
    assert_eq!(our_msgs.len(), 1);
    assert_eq!(our_msgs[0].content, expected_quote);
}

#[tokio::test]
async fn strips_media_markers_from_content() {
    assert_mirror_strips_markers(
        "strip_markers",
        "unique_markers",
        "Check this [IMAGE:/tmp/screenshot.png] and my [AUDIO:/tmp/recording.mp3]",
        "<blockquote>\nCheck this  and my\n</blockquote>",
    )
    .await;
}

#[tokio::test]
async fn strips_lowercase_media_markers_from_content() {
    // Regression test: the mirror path must use a case-insensitive regex
    // to strip lowercase markers like [image:...] and [audio:...].
    assert_mirror_strips_markers(
        "lowercase_markers",
        "unique_lowercase",
        "See [image:/tmp/photo.png] and hear [audio:/tmp/sound.mp3]",
        "<blockquote>\nSee  and hear\n</blockquote>",
    )
    .await;
}

#[tokio::test]
async fn preserves_markdown_formatting_in_blockquote() {
    let (sent, _lock) = setup_mirror_test_env().await;
    setup_user_with_telegram_binding("md_user", "unique_md", "md_user").await;

    let msg = gui_msg("md_user", "**bold** and `code` and *italic*");
    super::mirror_gui_message_to_telegram(&msg).await;

    let guard = sent.lock().unwrap_poison();
    let our_msgs: Vec<_> = guard
        .iter()
        .filter(|m| m.recipient == "unique_md")
        .collect();
    assert_eq!(our_msgs.len(), 1);
    // Markdown syntax inside the blockquote passes through — the Telegram
    // channel's markdown_to_telegram_html will handle formatting later.
    assert_eq!(
        our_msgs[0].content,
        "<blockquote>\n**bold** and `code` and *italic*\n</blockquote>"
    );
}

#[tokio::test]
async fn user_command_entries_reflect_role_and_admin() {
    // Serialized with the sibling mirror tests: this test mutates the
    // shared user/workspace stores (alice's workspace + role, menu_ws
    // INSERT), so it must not run concurrently with other store users.
    let _lock = acquire_mirror_lock().await;
    crate::users::test_util::init_test_store().await;
    let store = crate::users::store();

    // Give alice a shared workspace so state-aware admin entries appear.
    crate::util::test::create_test_workspace("/tmp/mahbot_test_ws_menu", "menu_ws").await;
    store
        .update_user(
            "alice",
            crate::users::FieldUpdate::Unchanged,
            crate::users::FieldUpdate::Set("menu_ws"),
            crate::users::FieldUpdate::Unchanged,
        )
        .await
        .unwrap();

    // alice: admin (full permissions), pool = all roles → role commands,
    // Artist commands, and state-aware admin commands.
    let alice = user_command_entries("alice").await;
    let cmds: Vec<&str> = alice.iter().map(|(c, _)| c.as_str()).collect();
    assert!(cmds.contains(&"board"));
    // State-aware pairs reflect the workspace state: not paused →
    // /pause, maintenance disabled → /maintenance_on.
    assert!(cmds.contains(&"pause"));
    assert!(!cmds.contains(&"unpause"));
    assert!(cmds.contains(&"maintenance_on"));
    assert!(!cmds.contains(&"maintenance_off"));
    // Pool roles are direct commands.
    assert!(cmds.contains(&"engineer"));
    assert!(cmds.contains(&"artist"));
    // Artist is in the pool → model commands present.
    assert!(cmds.contains(&"image_models"));
    assert!(cmds.contains(&"video_models"));
    assert_eq!(cmds[0], "manager");
    // Menu order: role commands first, then board/admin, then model
    // commands, with /clear last.
    assert_eq!(cmds.last(), Some(&"clear"));
    let pos = |cmd: &str| cmds.iter().position(|c| *c == cmd).unwrap();
    assert!(pos("manager") < pos("board"));
    assert!(pos("board") < pos("image_models"));
    assert!(pos("image_models") < pos("clear"));

    // The active role's entry is marked. add_user seeds the selection to the
    // first pool role (Manager in Role::iter order).
    let manager_desc = alice
        .iter()
        .find(|(c, _)| c == "manager")
        .map(|(_, d)| d.as_str())
        .unwrap();
    assert!(manager_desc.contains("current"));

    // Switching the active role moves the marker.
    store
        .update_user(
            "alice",
            crate::users::FieldUpdate::Set("artist"),
            crate::users::FieldUpdate::Unchanged,
            crate::users::FieldUpdate::Unchanged,
        )
        .await
        .unwrap();
    let entries = user_command_entries("alice").await;
    let artist_desc = entries
        .iter()
        .find(|(c, _)| c == "artist")
        .map(|(_, d)| d.as_str())
        .unwrap();
    assert!(artist_desc.contains("current"));
    let analyst_desc = entries
        .iter()
        .find(|(c, _)| c == "analyst")
        .map(|(_, d)| d.as_str())
        .unwrap();
    assert!(!analyst_desc.contains("current"));

    // Flipping the workspace state reverses the pairs (the ticket's
    // headline criterion): paused → /unpause, maintenance on →
    // /maintenance_off.
    crate::workspace::store()
        .set_paused("menu_ws", true)
        .await
        .unwrap();
    crate::workspace::store()
        .set_maintenance_enabled("menu_ws", true)
        .await
        .unwrap();
    let flipped = user_command_entries("alice").await;
    let flipped_cmds: Vec<&str> = flipped.iter().map(|(c, _)| c.as_str()).collect();
    assert!(flipped_cmds.contains(&"unpause"));
    assert!(!flipped_cmds.contains(&"pause"));
    assert!(flipped_cmds.contains(&"maintenance_off"));
    assert!(!flipped_cmds.contains(&"maintenance_on"));

    // bob: restricted user — pool commands but no admin commands.
    let bob = user_command_entries("bob").await;
    let cmds: Vec<&str> = bob.iter().map(|(c, _)| c.as_str()).collect();
    assert!(!cmds.contains(&"board"));
    assert!(!cmds.contains(&"pause"));
    assert!(!cmds.contains(&"unpause"));
    assert!(cmds.contains(&"engineer"));
}
