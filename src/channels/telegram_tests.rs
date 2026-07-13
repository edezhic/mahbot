use super::*;

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
fn telegram_api_url() {
    let ch = TelegramChannel::new("123:ABC".into());
    assert_eq!(
        ch.api_url("getMe"),
        "https://api.telegram.org/bot123:ABC/getMe"
    );
}

#[test]
fn parse_recipient_parses_chat_id_only() {
    assert_eq!(parse_recipient("12345"), ("12345", None));
}

#[test]
fn parse_recipient_parses_chat_id_with_thread() {
    assert_eq!(parse_recipient("12345:678"), ("12345", Some("678")));
}

#[test]
fn parse_recipient_handles_empty_string() {
    assert_eq!(parse_recipient(""), ("", None));
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
    let (cleaned, att) =
        parse_attachment_markers("Here are files [IMAGE:/tmp/a.png] and [AUDIO:/tmp/voice.ogg]");
    assert_eq!(cleaned, "Here are files  and");
    assert_eq!(att.len(), 2);
    assert_eq!(att[0].kind, TelegramAttachmentKind::Image);
    assert_eq!(att[1].kind, TelegramAttachmentKind::Audio);
    // invalid markers kept as text
    let (cleaned, att) = parse_attachment_markers("Report [UNKNOWN:/tmp/a.bin]");
    assert_eq!(cleaned, "Report [UNKNOWN:/tmp/a.bin]");
    assert!(att.is_empty());
    // case-insensitive matching
    let (cleaned, att) = parse_attachment_markers("[image:path.png] and [VIDEO:/tmp/vid.mp4]");
    assert_eq!(cleaned, "and");
    assert_eq!(att.len(), 2);
    assert_eq!(att[0].kind, TelegramAttachmentKind::Image);
    assert_eq!(att[1].kind, TelegramAttachmentKind::Video);
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
        "data": "set_model|gpt-4",
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
    assert_eq!(msg.content, "set_model|gpt-4");
    assert_eq!(msg.source_channel, "telegram");
    assert_eq!(msg.callback_query_id.as_deref(), Some("12345"));
}

#[tokio::test]
async fn parse_callback_query_returns_none_when_data_missing() {
    let ch = test_channel().await;
    let cq = test_callback_query(&[("data", serde_json::Value::Null)]);

    assert!(
        ch.parse_callback_query(&cq).await.is_none(),
        "callback query without data should be rejected"
    );
}

#[tokio::test]
async fn parse_callback_query_returns_none_when_message_missing() {
    let ch = test_channel().await;
    let cq = test_callback_query(&[("message", serde_json::Value::Null)]);

    assert!(
        ch.parse_callback_query(&cq).await.is_none(),
        "callback query without message should be rejected"
    );
}

#[tokio::test]
async fn parse_callback_query_returns_none_for_unauthorized_user() {
    let ch = test_channel().await;
    let cq = test_callback_query(&[(
        "from",
        serde_json::json!({ "id": 999, "username": "unknown_user" }),
    )]);

    assert!(
        ch.parse_callback_query(&cq).await.is_none(),
        "callback query from an unknown user should be rejected"
    );
}

#[tokio::test]
async fn parse_callback_query_preserves_data_question_mark_semantics() {
    // The ? guard in parse_callback_query only rejects when data is
    // absent or null — NOT when it's present but empty. This is distinct
    // from process_updates which uses unwrap_or("") for the ACK check.
    // An empty-string data value produces a ChannelMessage with empty content.
    let ch = test_channel().await;
    let cq = test_callback_query(&[("data", serde_json::Value::String(String::new()))]);

    let msg = ch
        .parse_callback_query(&cq)
        .await
        .expect("empty-string data is valid — ? only rejects null/absent");

    assert_eq!(msg.content, "", "empty-string data becomes empty content");
    assert_eq!(msg.callback_query_id.as_deref(), Some("12345"));
}

#[tokio::test]
async fn parse_callback_query_accepts_null_id() {
    // A null/absent callback_query_id should still produce a valid message
    // (the field becomes None rather than Some). This is distinct from
    // missing data which causes rejection.
    let ch = test_channel().await;
    let cq = test_callback_query(&[("id", serde_json::Value::Null)]);

    let msg = ch
        .parse_callback_query(&cq)
        .await
        .expect("null id should not prevent parsing");

    assert!(
        msg.callback_query_id.is_none(),
        "null id → callback_query_id is None"
    );
    assert_eq!(
        msg.content, "set_model|gpt-4",
        "data should still be extracted"
    );
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
    // image extensions produce [IMAGE:]
    for ext in ["png", "jpg", "jpeg", "gif", "webp", "bmp"] {
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
    // Document kind + no extension + mime_type "image/jpeg" → [IMAGE:] (mime fallback)
    let c = format_attachment_content(
        IncomingAttachmentKind::Document,
        "image_no_ext",
        std::path::Path::new("/tmp/workspace/image_no_ext"),
        Some("image/jpeg"),
    );
    assert_eq!(c, "[IMAGE:/tmp/workspace/image_no_ext]");
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
}

#[test]
fn attachment_multimodal_and_helpers() {
    // is_image_extension
    for p in [
        "photo.png",
        "photo.jpg",
        "photo.jpeg",
        "photo.gif",
        "photo.webp",
        "photo.bmp",
        "PHOTO.PNG",
    ] {
        assert!(is_image_extension(std::path::Path::new(p)));
    }
    for p in ["file.md", "file.txt", "file.pdf", "file.csv", "file"] {
        assert!(!is_image_extension(std::path::Path::new(p)));
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

// ── decode_callback ───────────────────────────────────────────────────

#[test]
fn test_decode_callback() {
    struct Case {
        name: &'static str,
        input: &'static str,
        expected: Option<(Option<&'static str>, &'static str)>,
    }

    let cases = [
        Case {
            name: "with ticket id",
            input: "__opt__mahbot-123|Option A",
            expected: Some((Some("mahbot-123"), "Option A")),
        },
        Case {
            name: "empty ticket id",
            input: "__opt__|Label",
            expected: Some((None, "Label")),
        },
        Case {
            name: "no delimiter",
            input: "__opt__BareLabel",
            expected: Some((None, "BareLabel")),
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
        Case {
            name: "label with extra pipes",
            input: "__opt__ticket|A|B|C",
            expected: Some((Some("ticket"), "A|B|C")),
        },
        // Labels containing '|' test a deliberate `split_once` behavior:
        // `split_once('|')` splits on the *first* pipe only, so the label
        // captures everything after it.  Neither ticket_id nor label should
        // contain `|` in practice (per the format contract in the doc comment).
        Case {
            name: "only prefix and pipe",
            input: "__opt__|",
            expected: Some((None, "")),
        },
    ];

    for case in &cases {
        let result = decode_callback(case.input);
        let expected = case
            .expected
            .map(|(tid, lbl)| (tid.map(String::from), lbl.to_string()));
        assert_eq!(result, expected, "case: {}", case.name);
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
/// reply_target. Idempotent.
async fn setup_user_with_telegram_binding(user_name: &str, reply_target: &str) {
    use crate::users::store;
    let store = store();
    store
        .add_user(user_name, Some("full"))
        .await
        .expect("add_user");
    store
        .bind_channel(user_name, "telegram", user_name)
        .await
        .expect("bind_channel");
    store
        .update_channel_contact("telegram", user_name, reply_target)
        .await
        .expect("update_channel_contact");
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
        source_channel: "gui".to_string(),
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
        source_channel: "telegram".to_string(),
        workspace: "test".to_string(),
        optimistic_id: None,
        callback_query_id: None,
    }
}

// ── Guard tests: early-return conditions ─────────────────────────────

#[tokio::test]
async fn skip_non_gui_source() {
    let (sent, _lock) = setup_mirror_test_env().await;
    setup_user_with_telegram_binding("skip_telegram", "target_non_gui").await;

    let msg = telegram_msg("skip_telegram", "hello from telegram");
    super::mirror_gui_message_to_telegram(&msg).await;
    let guard = sent.lock().unwrap_poison();
    let our_msgs: Vec<_> = guard
        .iter()
        .filter(|m| m.recipient == "target_non_gui")
        .collect();
    assert!(our_msgs.is_empty(), "non-GUI source should not send");
}

#[tokio::test]
async fn skip_empty_or_whitespace_content() {
    // Both inputs exercise the same guard — `msg.content.trim().is_empty()`.
    // Each iteration acquires the serialization lock independently; this
    // is safe because the global stores (OnceCell) and the spy channel
    // (OnceLock) are identical across calls to `setup_mirror_test_env()`.
    for content in ["", "   \t\n  "] {
        let (sent, _lock) = setup_mirror_test_env().await;
        setup_user_with_telegram_binding("skip_ew", "target_empty_ws").await;
        let msg = gui_msg("skip_ew", content);
        super::mirror_gui_message_to_telegram(&msg).await;
        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "target_empty_ws")
            .collect();
        assert!(
            our_msgs.is_empty(),
            "content {content:?} should not send, got {} message(s)",
            our_msgs.len()
        );
    }
}

#[tokio::test]
async fn skip_user_with_no_bindings() {
    let (sent, _lock) = setup_mirror_test_env().await;
    // Create user but DO NOT bind a Telegram channel.
    let store = crate::users::store();
    store.add_user("no_binding", None).await.expect("add_user");

    // Use the user's name as the recipient filter — no bindings means
    // no messages should be sent for this user at all.
    let user_name = "no_binding";
    let msg = gui_msg(user_name, "hello");
    super::mirror_gui_message_to_telegram(&msg).await;
    let guard = sent.lock().unwrap_poison();
    let our_msgs: Vec<_> = guard.iter().filter(|m| m.recipient == user_name).collect();
    assert!(our_msgs.is_empty(), "user with no bindings should not send");
}

#[tokio::test]
async fn skip_binding_without_reply_target() {
    let (sent, _lock) = setup_mirror_test_env().await;
    // Bind a Telegram channel but don't set reply_target.
    let store = crate::users::store();
    store.add_user("no_target", None).await.expect("add_user");
    store
        .bind_channel("no_target", "telegram", "no_target")
        .await
        .expect("bind_channel");
    // Note: skip update_channel_contact → reply_target stays NULL.

    let msg = gui_msg("no_target", "hello");
    super::mirror_gui_message_to_telegram(&msg).await;
    let guard = sent.lock().unwrap_poison();
    let our_msgs: Vec<_> = guard
        .iter()
        .filter(|m| m.recipient == "no_target")
        .collect();
    assert!(
        our_msgs.is_empty(),
        "binding without reply_target should not send"
    );
}

#[tokio::test]
async fn skip_media_only_content() {
    let (sent, _lock) = setup_mirror_test_env().await;
    setup_user_with_telegram_binding("media_only", "target_media").await;

    let msg = gui_msg("media_only", "[IMAGE:/path/to/img.png]");
    super::mirror_gui_message_to_telegram(&msg).await;
    let guard = sent.lock().unwrap_poison();
    let our_msgs: Vec<_> = guard
        .iter()
        .filter(|m| m.recipient == "target_media")
        .collect();
    assert!(our_msgs.is_empty(), "media-only content should not send");
}

// ── Happy path tests ─────────────────────────────────────────────────

#[tokio::test]
async fn sends_blockquote_to_single_binding() {
    let (sent, _lock) = setup_mirror_test_env().await;
    setup_user_with_telegram_binding("single_user", "unique_single").await;

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
async fn sends_to_multiple_telegram_bindings() {
    let (sent, _lock) = setup_mirror_test_env().await;
    let store = crate::users::store();
    store.add_user("multi_user", None).await.expect("add_user");
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

#[tokio::test]
async fn strips_media_markers_from_content() {
    let (sent, _lock) = setup_mirror_test_env().await;
    setup_user_with_telegram_binding("strip_markers", "unique_markers").await;

    let msg = gui_msg(
        "strip_markers",
        "Check this [IMAGE:/tmp/screenshot.png] and my [AUDIO:/tmp/recording.mp3]",
    );
    super::mirror_gui_message_to_telegram(&msg).await;

    let guard = sent.lock().unwrap_poison();
    let our_msgs: Vec<_> = guard
        .iter()
        .filter(|m| m.recipient == "unique_markers")
        .collect();
    assert_eq!(our_msgs.len(), 1);
    // Markers should be stripped entirely (trailing whitespace is trimmed).
    assert_eq!(
        our_msgs[0].content,
        "<blockquote>\nCheck this  and my\n</blockquote>"
    );
}

#[tokio::test]
async fn preserves_markdown_formatting_in_blockquote() {
    let (sent, _lock) = setup_mirror_test_env().await;
    setup_user_with_telegram_binding("md_user", "unique_md").await;

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
