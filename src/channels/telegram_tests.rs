//! Telegram transport tests: update and callback decoding, attachment markers, listener framing,
//! intake claims and Bot API HTML rendering.

use super::*;
use crate::channels::telegram_intake::{IN_PROGRESS, Intake, RETENTION_HOURS, TelegramIntakeStore};
use crate::util::test::canonical_without_verbatim_prefix;
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};

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

/// The rendering of each refusal the failure note carries, for the assertions below.
fn refusal_lines(refused: &[crate::RefusedMarker]) -> Vec<String> {
    refused
        .iter()
        .map(|refusal| format!("{}: {}", refusal.marker, refusal.reason))
        .collect()
}

async fn test_channel() -> TelegramChannel {
    crate::users::test_util::init_test_store().await;
    TelegramChannel::new_with("token".into(), CLOUD_API_BASE.to_string(), false)
}

#[test]
fn test_markdown_to_telegram_html() {
    let r = markdown_to_telegram_html("[click](https://example.com?q=\"x\"&a='b')");
    assert_eq!(
        r,
        "<a href=\"https://example.com?q=&quot;x&quot;&amp;a=&#39;b&#39;\">click</a>"
    );
    let r = markdown_to_telegram_html("say \"hi\" & <tag> 'ok'");
    assert_eq!(r, "say &quot;hi&quot; &amp; &lt;tag&gt; &#39;ok&#39;");
    let r = markdown_to_telegram_html("```rust\" onclick=\"alert(1)\nlet x = 1;\n```");
    assert_eq!(r, "<pre><code>let x = 1;</code></pre>");
    assert!(!r.contains("language-"));
    assert!(!r.contains("onclick"));

    let r = markdown_to_telegram_html("```\nsome **bold** and `code`\n```");
    assert_eq!(r, "<pre><code>some **bold** and `code`</code></pre>");

    let r = markdown_to_telegram_html("```\n<div> & \"it\" 'works'\n```");
    assert_eq!(
        r,
        "<pre><code>&lt;div&gt; &amp; &quot;it&quot; &#39;works&#39;</code></pre>"
    );

    let r = markdown_to_telegram_html("```\nuse &lt;/code&gt;\n```");
    assert_eq!(r, "<pre><code>use &amp;lt;/code&amp;gt;</code></pre>");

    let r = markdown_to_telegram_html("<blockquote>");
    assert_eq!(r, "<blockquote>");

    let r = markdown_to_telegram_html("</blockquote>");
    assert_eq!(r, "</blockquote>");

    let r = markdown_to_telegram_html("<blockquote>\nHello **world**\n</blockquote>");
    assert_eq!(r, "<blockquote>\nHello <b>world</b>\n</blockquote>");

    let r = markdown_to_telegram_html("<blockquote123>");
    assert_eq!(r, "&lt;blockquote123&gt;");

    let r = markdown_to_telegram_html("<blockquote class=\"x\">");
    assert_eq!(r, "&lt;blockquote class=&quot;x&quot;&gt;");

    let r = markdown_to_telegram_html("<blockquote >");
    assert_eq!(r, "&lt;blockquote &gt;");

    let r = markdown_to_telegram_html("## [docs](https://example.com)");
    assert_eq!(r, "<a href=\"https://example.com\">docs</a>");
}

/// The `/board` listing converts the joined text line by line: a hostile title must never unbalance
/// another line's HTML, which would make Telegram reject it all.
#[test]
fn board_listing_isolates_hostile_titles() {
    // Pin the production format string — if `handle_board_listing`'s helper drifts, this test fails
    // instead of silently covering a stale fixture.
    assert_eq!(
        format_board_line(&TicketPhase::InDevelopment, "mahbot-123", "Title"),
        "🔨 `mahbot-123` Title"
    );

    let state = TicketPhase::InDevelopment;
    let lines = [
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

    // Every line keeps its phase emoji and monospace ID.
    for line in html.split('\n') {
        assert!(line.starts_with("🔨 "), "phase emoji lost: {line:?}");
        assert!(
            line.contains("<code>") && line.contains("</code> "),
            "id formatting lost: {line:?}"
        );
    }
    // No unbalanced tags anywhere — an unclosed marker in a title must be escaped, not emitted as a
    // dangling tag (that would degrade the whole listing to plain text via Telegram's HTML parse
    // failure fallback).
    for (open, close) in [("<b>", "</b>"), ("<i>", "</i>"), ("<code>", "</code>")] {
        assert_eq!(
            html.matches(open).count(),
            html.matches(close).count(),
            "unbalanced {open}/{close} in: {html}"
        );
    }
}

#[test]
#[expect(clippy::too_many_lines)] // one cohesive table of inline-formatting cases
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
        Case {
            name: "unmatched double asterisk",
            input: "hello ** world",
            expected: "hello ** world",
        },
        Case {
            name: "unmatched single asterisk",
            input: "hello * world",
            expected: "hello * world",
        },
        Case {
            name: "triple asterisk",
            input: "***",
            expected: "***",
        },
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
        Case {
            name: "unmatched tilde",
            input: "hello ~ world",
            expected: "hello ~ world",
        },
        Case {
            name: "bold and italic overlap",
            input: "***bold**",
            expected: "<b>*bold</b>",
        },
        // A formatting span whose content holds a markdown link drops its formatting — the anchor
        // is emitted at the top level, not nested.
        Case {
            name: "bold span with link",
            input: "**see [docs](https://example.com) now**",
            expected: "see <a href=\"https://example.com\">docs</a> now",
        },
        Case {
            name: "italic and strikethrough spans with link",
            input: "*[a](https://a.example)* and ~~[b](https://b.example)~~",
            expected: "<a href=\"https://a.example\">a</a> and <a href=\"https://b.example\">b</a>",
        },
        Case {
            name: "nested spans with link",
            input: "**bold *inner [docs](https://example.com)* tail**",
            expected: "bold inner <a href=\"https://example.com\">docs</a> tail",
        },
        Case {
            name: "bold span without link",
            input: "**hello [not a link]**",
            expected: "<b>hello [not a link]</b>",
        },
        // Link syntax in code renders no anchor, so the bold around it stays (span content is never
        // re-parsed for formatting without a link).
        Case {
            name: "bold span around code with link syntax",
            input: "**bold `[docs](https://example.com)` tail**",
            expected: "<b>bold `[docs](https://example.com)` tail</b>",
        },
        // A media marker is delivery scaffolding, not prose: an anchor labelled with the marker's
        // own text would fabricate a link out of it.
        Case {
            name: "marker label is not a link",
            input: "[IMAGE:data:image/png;base64,AAAA](https://example.com)",
            expected: "[IMAGE:data:image/png;base64,AAAA](https://example.com)",
        },
        Case {
            name: "lowercase marker label is not a link",
            input: "[image:/tmp/x.txt](https://example.com)",
            expected: "[image:/tmp/x.txt](https://example.com)",
        },
        Case {
            name: "uppercase prose label stays a link",
            input: "[NOTE:42](https://example.com)",
            expected: "<a href=\"https://example.com\">NOTE:42</a>",
        },
    ];
    for case in cases {
        let result = markdown_to_telegram_html(case.input);
        assert_eq!(result, case.expected, "case: {}", case.name);
    }
}

/// Write a real 1×1 red PNG raster — the classifier rejects non-raster "images", so fixtures must
/// decode.
fn write_test_png(path: &std::path::Path) {
    let img = image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 0, 0, 255]));
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .expect("test PNG must encode");
    std::fs::write(path, buf).unwrap();
}

#[test]
fn parse_attachment_markers_attach_known_targets_and_refuse_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let png = dir.path().join("a.png");
    let ogg = dir.path().join("voice.ogg");
    let vid = dir.path().join("vid.mp4");
    write_test_png(&png);
    std::fs::write(&ogg, b"fake-ogg").unwrap();
    std::fs::write(&vid, b"fake-mp4").unwrap();
    let roots = vec![dir.path().to_path_buf()];

    let (cleaned, att, refusals) = parse_attachment_markers(
        "use `[IMAGE:path]` or `[VIDEO:...]`",
        &roots,
        TransferLimits::CLOUD,
    );
    assert_eq!(cleaned, "use `[IMAGE:path]` or `[VIDEO:...]`");
    assert!(att.is_empty());
    assert!(refusals.is_empty());

    let (cleaned, att, refusals) = parse_attachment_markers(
        &format!(
            "Here are files [IMAGE:{}] and [AUDIO:{}]",
            png.display(),
            ogg.display()
        ),
        &roots,
        TransferLimits::CLOUD,
    );
    assert_eq!(cleaned, "Here are files  and");
    assert!(refusals.is_empty());
    assert_eq!(att.len(), 2);
    assert_eq!(att[0].kind, TelegramAttachmentKind::Image);
    assert_eq!(att[0].target, png.to_string_lossy());
    assert_eq!(att[1].kind, TelegramAttachmentKind::Audio);
    assert_eq!(att[1].target, ogg.to_string_lossy());

    let (cleaned, att, refusals) = parse_attachment_markers(
        "See [VIDEO:https://example.com/vid.mp4] and [IMAGE:https://example.com/a.png]",
        &roots,
        TransferLimits::CLOUD,
    );
    assert_eq!(cleaned, "See  and");
    assert!(refusals.is_empty());
    assert_eq!(att.len(), 2);
    assert_eq!(att[0].kind, TelegramAttachmentKind::Video);
    assert_eq!(att[0].target, "https://example.com/vid.mp4");
    assert_eq!(att[1].kind, TelegramAttachmentKind::Image);
    assert_eq!(att[1].target, "https://example.com/a.png");

    let (cleaned, att, refusals) = parse_attachment_markers(
        "[IMAGE:https://example.com:bad/x.png]",
        &roots,
        TransferLimits::CLOUD,
    );
    assert_eq!(cleaned, "");
    assert!(att.is_empty());
    assert_eq!(
        refusal_lines(&refusals),
        vec!["[IMAGE:https://example.com:bad/x.png]: it is not a valid address."]
    );

    let (cleaned, att, refusals) =
        parse_attachment_markers("Report [UNKNOWN:/tmp/a.bin]", &roots, TransferLimits::CLOUD);
    assert_eq!(cleaned, "Report [UNKNOWN:/tmp/a.bin]");
    assert!(att.is_empty());
    assert!(refusals.is_empty());

    let (cleaned, att, _) = parse_attachment_markers(
        &format!("[image:{}]", png.display()),
        &roots,
        TransferLimits::CLOUD,
    );
    assert_eq!(cleaned, "");
    assert_eq!(att.len(), 1);
    assert_eq!(att[0].kind, TelegramAttachmentKind::Image);

    // Regression (case-insensitive gate): a lowercase `[image:...]` targeting an existing
    // NON-raster file must not bypass the classifier — it is refused rather than shown to the user
    // as marker text.
    let nonimg = dir.path().join("note.txt");
    std::fs::write(&nonimg, b"not an image").unwrap();
    let (cleaned, att, refusals) = parse_attachment_markers(
        &format!("[image:{}]", nonimg.display()),
        &roots,
        TransferLimits::CLOUD,
    );
    assert_eq!(cleaned, "");
    assert!(att.is_empty(), "a non-raster image marker must not attach");
    assert_eq!(
        refusal_lines(&refusals),
        vec![format!("[IMAGE:{}]: it is not an image.", nonimg.display())]
    );
}

/// A relative media target resolves against the first root that has the file, and is attached from
/// that resolved absolute path — exactly what `[FILE:...]` already does.
#[test]
fn a_relative_media_marker_resolves_against_the_messages_workspace() {
    let dir = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let generated = dir.path().join("generated");
    std::fs::create_dir(&generated).unwrap();
    let png = generated.join("a.png");
    let ogg = dir.path().join("voice.ogg");
    let vid = dir.path().join("clip.mp4");
    write_test_png(&png);
    std::fs::write(&ogg, b"fake-ogg").unwrap();
    std::fs::write(&vid, b"fake-mp4").unwrap();
    let roots = vec![dir.path().to_path_buf()];

    let (cleaned, att, refusals) = parse_attachment_markers(
        "[IMAGE:generated/a.png] [AUDIO:voice.ogg] [VIDEO:clip.mp4]",
        &roots,
        TransferLimits::CLOUD,
    );
    assert_eq!(cleaned, "");
    assert!(refusals.is_empty());
    assert_eq!(att.len(), 3);
    assert_eq!(att[0].kind, TelegramAttachmentKind::Image);
    assert_eq!(att[0].target, png.to_string_lossy());
    assert_eq!(att[1].kind, TelegramAttachmentKind::Audio);
    assert_eq!(att[1].target, ogg.to_string_lossy());
    assert_eq!(att[2].kind, TelegramAttachmentKind::Video);
    assert_eq!(att[2].target, vid.to_string_lossy());

    // Relayed content names a file in the originating workspace: the root that has it wins, as
    // `[FILE:...]` already resolves.
    let relayed = other.path().join("relayed.png");
    write_test_png(&relayed);
    let (_, att, refusals) = parse_attachment_markers(
        "[IMAGE:relayed.png]",
        &[dir.path().to_path_buf(), other.path().to_path_buf()],
        TransferLimits::CLOUD,
    );
    assert!(refusals.is_empty());
    assert_eq!(att.len(), 1);
    assert_eq!(att[0].target, relayed.to_string_lossy());
}

/// A marker whose only defect is a mistyped directory is delivered from the file it meant, a
/// mistyped file name is refused rather than guessed at, and a file the message would deliver more
/// than once goes out once.
#[test]
fn a_mistyped_directory_is_delivered_and_a_mistyped_name_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let generated = root.join("userspaces").join("lyuba").join("generated");
    std::fs::create_dir_all(&generated).unwrap();
    let png = generated.join("shot.png");
    write_test_png(&png);
    let roots = vec![root.to_path_buf()];
    let typo = root
        .join("userseas")
        .join("lyuba")
        .join("generated")
        .join("shot.png");

    let (cleaned, att, refusals) = parse_attachment_markers(
        &format!("Look [IMAGE:{}]", typo.display()),
        &roots,
        TransferLimits::CLOUD,
    );
    assert_eq!(cleaned, "Look");
    assert!(refusals.is_empty());
    assert_eq!(att.len(), 1);
    assert_eq!(att[0].target, png.to_string_lossy());

    // `[FILE:]` is workspace-confined: the repair is judged by the roots like any other path, so
    // it never makes an outside file deliverable.
    let (_, att, refusals) = parse_attachment_markers(
        &format!("[FILE:{}]", typo.display()),
        &roots,
        TransferLimits::CLOUD,
    );
    assert!(refusals.is_empty());
    assert_eq!(att.len(), 1);
    assert_eq!(
        att[0].target,
        canonical_without_verbatim_prefix(&png).to_string_lossy()
    );
    let elsewhere = tempfile::tempdir().unwrap();
    let (_, att, refusals) = parse_attachment_markers(
        &format!("[FILE:{}]", typo.display()),
        &[elsewhere.path().to_path_buf()],
        TransferLimits::CLOUD,
    );
    assert!(att.is_empty());
    assert_eq!(
        refusal_lines(&refusals),
        vec![format!(
            "[FILE:{}]: it is outside your workspace.",
            typo.display()
        )]
    );

    // The repaired marker and the correct one name one file, whichever kind each is written as.
    let (_, att, refusals) = parse_attachment_markers(
        &format!(
            "[IMAGE:{}] [FILE:{}] [IMAGE:{}]",
            typo.display(),
            typo.display(),
            png.display()
        ),
        &roots,
        TransferLimits::CLOUD,
    );
    assert!(refusals.is_empty());
    assert_eq!(att.len(), 1, "one file, one attachment");

    // A mistyped file name is a different file: similar names are the norm here, so nothing is
    // guessed and the refusal stands.
    let name_typo = generated.join("shot.pgn");
    let (_, att, refusals) = parse_attachment_markers(
        &format!("[IMAGE:{}]", name_typo.display()),
        &roots,
        TransferLimits::CLOUD,
    );
    assert!(att.is_empty());
    assert_eq!(
        refusal_lines(&refusals),
        vec![format!("[IMAGE:{}]: file not found.", name_typo.display())]
    );
}

/// One delivered file leaves one repair record, however many kinds or spellings named it, and a
/// repaired marker the kind's gate refuses leaves none: the row counts deliveries, not markers.
#[tokio::test]
#[serial_test::serial(provider)] // the telemetry store seam is process-global
async fn a_delivered_file_leaves_one_repair_record() {
    let (store, _store_tmp) = crate::open_test_store!(crate::logs::LogStore, "log");
    let _seam = crate::util::test::install_test_log_store(store.clone());
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let generated = root.join("userspaces").join("lyuba").join("generated");
    std::fs::create_dir_all(&generated).unwrap();
    let png = generated.join("shot.png");
    write_test_png(&png);
    std::fs::write(generated.join("clip.mp4"), b"fake-mp4").unwrap();
    let roots = vec![root.to_path_buf()];
    let typo = root
        .join("userseas")
        .join("lyuba")
        .join("generated")
        .join("shot.png");
    let other_typo = root
        .join("userspacse")
        .join("lyuba")
        .join("generated")
        .join("shot.png");

    // Two kinds, two spellings, one file: one attachment and one repair record.
    let (_, att, refusals) = parse_attachment_markers(
        &format!(
            "[IMAGE:{}] [FILE:{}] [IMAGE:{}]",
            typo.display(),
            typo.display(),
            other_typo.display()
        ),
        &roots,
        TransferLimits::CLOUD,
    );
    assert!(refusals.is_empty());
    assert_eq!(att.len(), 1, "one file, one attachment");
    crate::telemetry::drain_for_test().await;
    let repair_rows = marker_row_workspaces(&store, MARKER_REPAIRED_MESSAGE, &typo).await;
    assert_eq!(repair_rows.len(), 1, "one file, one repair record");
    assert_eq!(
        repair_rows[0],
        root.file_name().unwrap().to_string_lossy().into_owned(),
        "the row belongs to the workspace holding the file"
    );

    // A correct marker first and a repaired twin after still names one delivered file: one row.
    let (_, att, refusals) = parse_attachment_markers(
        &format!("[IMAGE:{}] [IMAGE:{}]", png.display(), other_typo.display()),
        &roots,
        TransferLimits::CLOUD,
    );
    assert!(refusals.is_empty());
    assert_eq!(att.len(), 1);
    crate::telemetry::drain_for_test().await;
    assert_eq!(
        marker_row_workspaces(&store, MARKER_REPAIRED_MESSAGE, &other_typo)
            .await
            .len(),
        1,
        "a repaired twin of a delivered file is still one repair"
    );

    // Repaired, then refused by the kind's gate: nothing was delivered, so no repair row.
    let clip_typo = root
        .join("userseas")
        .join("lyuba")
        .join("generated")
        .join("clip.mp4");
    let (_, att, refusals) = parse_attachment_markers(
        &format!("[IMAGE:{}]", clip_typo.display()),
        &roots,
        TransferLimits::CLOUD,
    );
    assert!(att.is_empty());
    assert_eq!(refusals.len(), 1);
    crate::telemetry::drain_for_test().await;
    assert!(
        marker_row_workspaces(&store, MARKER_REPAIRED_MESSAGE, &clip_typo)
            .await
            .is_empty(),
        "a repaired marker the gate refused is not a repair"
    );
}

/// A marker record for a path under no workspace carries no workspace at all, rather than an
/// unrelated one: the per-workspace counts stay honest. A repair outside every root is the same.
#[tokio::test]
#[serial_test::serial(provider)] // the telemetry store seam is process-global
async fn a_marker_record_outside_every_workspace_carries_no_workspace() {
    let (store, _store_tmp) = crate::open_test_store!(crate::logs::LogStore, "log");
    let _seam = crate::util::test::install_test_log_store(store.clone());
    let workspace = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let roots = vec![workspace.path().to_path_buf()];

    let missing = outside.path().join("gone.png");
    let (_, att, refusals) = parse_attachment_markers(
        &format!("[IMAGE:{}]", missing.display()),
        &roots,
        TransferLimits::CLOUD,
    );
    assert!(att.is_empty());
    assert_eq!(refusals.len(), 1);

    let pictures = outside.path().join("pictures");
    std::fs::create_dir(&pictures).unwrap();
    write_test_png(&pictures.join("shot.png"));
    let typo = outside.path().join("picturrs").join("shot.png");
    let (_, att, refusals) = parse_attachment_markers(
        &format!("[IMAGE:{}]", typo.display()),
        &roots,
        TransferLimits::CLOUD,
    );
    assert!(refusals.is_empty());
    assert_eq!(att.len(), 1);

    crate::telemetry::drain_for_test().await;
    assert_eq!(
        marker_row_workspaces(&store, MARKER_REFUSED_MESSAGE, &missing).await,
        vec![String::new()],
        "a refusal under no workspace names none"
    );
    assert_eq!(
        marker_row_workspaces(&store, MARKER_REPAIRED_MESSAGE, &typo).await,
        vec![String::new()],
        "a repair under no workspace names none"
    );

    // A `..` cannot smuggle an outside path into a workspace's count.
    let escaped = workspace
        .path()
        .join("..")
        .join(outside.path().file_name().unwrap())
        .join("gone.png");
    let (_, att, refusals) = parse_attachment_markers(
        &format!("[IMAGE:{}]", escaped.display()),
        &roots,
        TransferLimits::CLOUD,
    );
    assert!(att.is_empty());
    assert_eq!(refusals.len(), 1);
    crate::telemetry::drain_for_test().await;
    assert_eq!(
        marker_row_workspaces(&store, MARKER_REFUSED_MESSAGE, &escaped).await,
        vec![String::new()],
        "a `..` does not make an outside path belong to the workspace it is spelled under"
    );
}

/// The workspaces of the marker rows whose message is `message` and whose fields name `path`.
/// Filtered by the test-unique path: the telemetry seam is process-global, so a concurrent test
/// recording a marker of its own must not be counted here.
async fn marker_row_workspaces(
    store: &crate::logs::LogStore,
    message: &str,
    path: &std::path::Path,
) -> Vec<String> {
    let needle = path.to_string_lossy();
    let rows = store
        .conn
        .query(
            "SELECT workspace, fields FROM logs WHERE message = ?1",
            crate::db::params![message],
        )
        .await
        .expect("read the marker rows");
    rows.iter()
        .filter_map(|row| {
            let fields = row.get::<String>(1).ok()?;
            fields
                .contains(needle.as_ref())
                .then(|| row.get::<String>(0).unwrap_or_default())
        })
        .collect()
}

/// One refused marker does not stop the others in the same message: every valid target is attached
/// and only the failing one is reported, in body order.
#[test]
fn one_refused_marker_does_not_cost_the_others() {
    let dir = tempfile::tempdir().unwrap();
    let generated = dir.path().join("generated");
    std::fs::create_dir(&generated).unwrap();
    let png = generated.join("a.png");
    let vid = dir.path().join("clip.mp4");
    write_test_png(&png);
    std::fs::write(&vid, b"fake-mp4").unwrap();

    let (cleaned, att, refusals) = parse_attachment_markers(
        &format!(
            "Two of three [IMAGE:generated/a.png] ok [IMAGE:missing/a.png] [VIDEO:{}]",
            vid.display()
        ),
        &[dir.path().to_path_buf()],
        TransferLimits::CLOUD,
    );
    assert_eq!(cleaned, "Two of three  ok");
    assert_eq!(att.len(), 2);
    assert_eq!(att[0].target, png.to_string_lossy());
    assert_eq!(att[1].kind, TelegramAttachmentKind::Video);
    assert_eq!(
        refusal_lines(&refusals),
        vec!["[IMAGE:missing/a.png]: file not found."]
    );
}

/// `[FILE:...]` is local-only: it resolves against the authorizing workspace roots, carries the
/// CANONICAL path, and reports every refusal with the marker as written.
#[test]
fn parse_file_markers_resolve_within_workspace_roots() {
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let roots = vec![dir.path().to_path_buf()];
    let escape = outside.path().join("secret.txt");
    std::fs::write(&escape, b"secret").unwrap();

    let pdf = dir.path().join("specs.pdf");
    std::fs::write(&pdf, b"fake-pdf").unwrap();
    let canonical = canonical_without_verbatim_prefix(&pdf);
    let (cleaned, att, refusals) = parse_attachment_markers(
        &format!("Here: [FILE:{}]", pdf.display()),
        &roots,
        TransferLimits::CLOUD,
    );
    assert_eq!(cleaned, "Here:");
    assert!(refusals.is_empty());
    assert_eq!(att.len(), 1);
    assert_eq!(att[0].kind, TelegramAttachmentKind::Document);
    assert_eq!(att[0].target, canonical.to_string_lossy());

    let (cleaned, att, refusals) =
        parse_attachment_markers("[FILE:specs.pdf]", &roots, TransferLimits::CLOUD);
    assert_eq!(cleaned, "");
    assert!(refusals.is_empty());
    assert_eq!(att.len(), 1);
    assert_eq!(att[0].target, canonical.to_string_lossy());

    // A relative target no earlier root holds resolves against the first root that does: relayed
    // Manager→Assistant content names a file in the originating workspace.
    let other = tempfile::tempdir().unwrap();
    let relayed = other.path().join("relayed.pdf");
    std::fs::write(&relayed, b"fake-pdf").unwrap();
    let (cleaned, att, refusals) = parse_attachment_markers(
        "[FILE:relayed.pdf]",
        &[dir.path().to_path_buf(), other.path().to_path_buf()],
        TransferLimits::CLOUD,
    );
    assert_eq!(cleaned, "");
    assert!(refusals.is_empty());
    assert_eq!(att.len(), 1);
    assert_eq!(
        att[0].target,
        canonical_without_verbatim_prefix(&relayed).to_string_lossy()
    );

    let (cleaned, att, refusals) = parse_attachment_markers(
        &format!("[FILE:{}]", escape.display()),
        &roots,
        TransferLimits::CLOUD,
    );
    assert_eq!(cleaned, "");
    assert!(att.is_empty());
    let outside_marker = format!("[FILE:{}]: it is outside your workspace.", escape.display());
    assert_eq!(refusal_lines(&refusals), vec![outside_marker]);

    let (cleaned, att, refusals) =
        parse_attachment_markers("[FILE:missing.pdf]", &roots, TransferLimits::CLOUD);
    assert_eq!(cleaned, "");
    assert!(att.is_empty());
    assert_eq!(
        refusal_lines(&refusals),
        vec!["[FILE:missing.pdf]: file not found."]
    );

    let sub = dir.path().join("reports");
    std::fs::create_dir(&sub).unwrap();
    let (_, att, refusals) = parse_attachment_markers(
        &format!("[FILE:{}]", sub.display()),
        &roots,
        TransferLimits::CLOUD,
    );
    assert!(att.is_empty());
    assert_eq!(
        refusal_lines(&refusals),
        vec![format!("[FILE:{}]: it is a directory.", sub.display())]
    );

    let (cleaned, att, refusals) =
        parse_attachment_markers("[FILE:http://x/y.pdf]", &roots, TransferLimits::CLOUD);
    assert_eq!(cleaned, "");
    assert!(att.is_empty());
    assert_eq!(
        refusal_lines(&refusals),
        vec!["[FILE:http://x/y.pdf]: an address is not a local file."]
    );

    // No roots → no workspace to authorize a local file (the plain-text/mirror sends).
    let (cleaned, att, refusals) = parse_attachment_markers(
        &format!("[FILE:{}]", pdf.display()),
        &[],
        TransferLimits::CLOUD,
    );
    assert_eq!(cleaned, "");
    assert!(att.is_empty());
    let unknown = format!(
        "[FILE:{}]: the message's workspace is unknown.",
        pdf.display()
    );
    assert_eq!(refusal_lines(&refusals), vec![unknown]);

    // An inline payload is not a file to hand over, and the marker never echoes it back.
    let blob = format!("[FILE:data:application/pdf;base64,{}]", "A".repeat(500));
    let (cleaned, att, refusals) = parse_attachment_markers(&blob, &roots, TransferLimits::CLOUD);
    assert_eq!(cleaned, "");
    assert!(att.is_empty());
    assert_eq!(
        refusal_lines(&refusals),
        vec!["[FILE]: it carries an inline payload, not a file."]
    );
}

/// A symlink inside the workspace pointing outside it is refused: containment is checked against
/// the CANONICAL target, so the link cannot smuggle a file out.
#[cfg(unix)]
#[test]
fn parse_file_markers_refuse_symlink_out_of_workspace() {
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let escape = outside.path().join("secret.txt");
    std::fs::write(&escape, b"secret").unwrap();
    let link = dir.path().join("link.txt");
    std::os::unix::fs::symlink(&escape, &link).unwrap();

    let (cleaned, att, refusals) = parse_attachment_markers(
        &format!("[FILE:{}]", link.display()),
        &[dir.path().to_path_buf()],
        TransferLimits::CLOUD,
    );
    assert_eq!(cleaned, "");
    assert!(att.is_empty());
    assert_eq!(
        refusal_lines(&refusals),
        vec![format!(
            "[FILE:{}]: it is outside your workspace.",
            link.display()
        )]
    );
}

/// Only the delivery route collapses for a video, not where it comes from: a remote one is fetched
/// and uploaded — never handed over by address, whose document form takes a couple of types — while
/// images and audio keep it, and files upload from disk.
#[test]
fn a_video_route_depends_only_on_its_target() {
    use AttachmentRoute::{ByAddress, FetchThenUpload, LocalFile};

    assert_eq!(
        attachment_route(
            TelegramAttachmentKind::Video,
            "https://example.com/clip.mov"
        ),
        FetchThenUpload
    );
    assert_eq!(
        attachment_route(TelegramAttachmentKind::Video, "/nowhere/clip.mov"),
        LocalFile
    );
    assert_eq!(
        attachment_route(TelegramAttachmentKind::Audio, "https://example.com/a.ogg"),
        ByAddress
    );
    assert_eq!(
        attachment_route(TelegramAttachmentKind::Image, "https://example.com/a.png"),
        ByAddress
    );
}

/// A fetched video is named so a client can pick a player: the address's last segment when it is a
/// video name, else that stem with the declared type's extension.
#[test]
fn a_fetched_video_is_named_for_a_player() {
    let name = |url: &str, content_type: Option<&str>| video_upload_name(url, content_type);
    assert_eq!(
        name("https://example.com/dir/clip.mov?sig=abc#t", None),
        "clip.mov"
    );
    assert_eq!(
        name("https://example.com/clip.mov", Some("video/mp4")),
        "clip.mov"
    );
    assert_eq!(
        name(
            "https://example.com/download?id=7",
            Some("video/quicktime; q=1")
        ),
        "download.mov"
    );
    assert_eq!(
        name("https://example.com/clip.php", Some("video/webm")),
        "clip.webm"
    );
    assert_eq!(name("https://example.com/", Some("video/mp4")), "video.mp4");
    assert_eq!(name("https://example.com", None), "video.mp4");
    assert_eq!(name("https://example.com/a/..", None), "video.mp4");
}

fn assert_document_upload(fake: &FakeTelegram, name: &str, bytes: &str) {
    assert_eq!(fake.count("sendDocument"), 1);
    assert_eq!(fake.count("sendVideo"), 0);
    let body = String::from_utf8_lossy(&fake.bodies("sendDocument")[0]).into_owned();
    assert!(body.contains("name=\"document\""), "{body}");
    assert!(body.contains("disable_content_type_detection"), "{body}");
    assert!(body.contains(name), "the file keeps its name: {body}");
    assert!(
        body.contains(bytes),
        "the file's bytes go out as they are: {body}"
    );
    for geometry in [
        "width",
        "height",
        "duration",
        "thumbnail",
        "supports_streaming",
    ] {
        assert!(
            !body.contains(geometry),
            "no {geometry} may be claimed: {body}"
        );
    }
}

/// What the product puts on the wire for a video: a document upload carrying the file's own bytes,
/// content-type detection off and no geometry claim — the recipient's own player, not the message,
/// describes the clip. A URL-addressed video is the same after the fetch: one upload of the fetched
/// bytes, never a video message, never a handover by address.
#[tokio::test]
async fn a_video_reaches_the_chat_as_one_document_upload() {
    crate::users::test_util::init_test_store().await;
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    let telegram = TelegramChannel::new_with("token-a".into(), fake.base.clone(), false);
    let dir = tempfile::tempdir().unwrap();
    let clip = dir.path().join("clip.mov");
    std::fs::write(&clip, b"fake-mov-bytes").unwrap();

    telegram
        .send(&SendMessage {
            content: format!("[VIDEO:{}]", clip.display()),
            recipient: "42".to_string(),
            reply_markup: None,
            file_roots: Vec::new(),
        })
        .await
        .expect("the video is delivered");

    assert_document_upload(&fake, "clip.mov", "fake-mov-bytes");

    let fake = FakeTelegram::start(&[("token-a", 700)]);
    fake.push_file("clip", b"remote-video-bytes", Duration::ZERO);
    let telegram = TelegramChannel::new_with("token-a".into(), fake.base.clone(), false);
    let url = format!("{}/file/bottoken-a/{}", fake.base, file_path_of("clip"));

    telegram
        .send(&SendMessage {
            content: format!("[VIDEO:{url}]"),
            recipient: "42".to_string(),
            reply_markup: None,
            file_roots: Vec::new(),
        })
        .await
        .expect("the remote video is delivered");

    assert_document_upload(&fake, "clip.mp4", "remote-video-bytes");
}

/// An image addressed by a URL keeps its by-address method: Telegram fetches and serves it as a
/// photo, not as an upload of ours.
#[tokio::test]
async fn an_image_by_address_keeps_the_photo_method() {
    crate::users::test_util::init_test_store().await;
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    let telegram = TelegramChannel::new_with("token-a".into(), fake.base.clone(), false);
    let url = "https://example.com/pictures/sunset.png".to_string();

    telegram
        .send(&SendMessage {
            content: format!("[IMAGE:{url}]"),
            recipient: "42".to_string(),
            reply_markup: None,
            file_roots: Vec::new(),
        })
        .await
        .expect("the image is delivered");

    assert_eq!(fake.count("sendPhoto"), 1);
    assert_eq!(fake.count("sendDocument"), 0);
    let body: serde_json::Value =
        serde_json::from_slice(&fake.bodies("sendPhoto")[0]).expect("the by-address body is JSON");
    assert_eq!(body["photo"], url);
}

/// A fetched copy lives in a temp directory of its own and goes away with the value that carries
/// it, whatever the upload that follows did.
#[tokio::test]
async fn a_fetched_copy_is_removed_with_its_delivery() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    fake.push_file("clip", b"remote-video-bytes", Duration::ZERO);
    let telegram = TelegramChannel::new_with("token-a".into(), fake.base.clone(), false);

    let url = format!("{}/file/bottoken-a/{}", fake.base, file_path_of("clip"));
    let fetched = telegram.fetch_remote_upload(&url).await.expect("fetch");
    let path = fetched.path.clone();
    drop(fetched);
    assert!(
        !path.exists(),
        "the fetched copy must not outlive the delivery"
    );
}

/// An address the model supplied passes the shared screen on every hop — the first one it wrote and
/// each one a redirect names. A metadata target is refused however it is reached, and a hop to the
/// file itself is followed.
#[tokio::test]
async fn an_address_is_screened_on_every_hop() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    fake.push_file("clip.mov", b"remote-video-bytes", Duration::ZERO);
    let telegram = TelegramChannel::new_with("token-a".into(), fake.base.clone(), false);
    let file = format!("{}/file/bottoken-a/{}", fake.base, file_path_of("clip.mov"));

    let direct = telegram
        .fetch_remote_upload("http://169.254.169.254/latest/meta-data")
        .await
        .err()
        .expect("a metadata address is refused on the first hop");
    assert!(format!("{direct:#}").contains("SSRF"), "{direct:#}");

    let followed = telegram
        .fetch_remote_upload(&format!("{}/redirect?to={file}", fake.base))
        .await
        .expect("a redirect to the file itself is followed");
    assert_eq!(
        tokio::fs::read(&followed.path).await.unwrap(),
        b"remote-video-bytes"
    );

    let hopped = telegram
        .fetch_remote_upload(&format!(
            "{}/redirect?to=http://169.254.169.254/latest/meta-data",
            fake.base
        ))
        .await
        .err()
        .expect("a metadata hop is refused");
    assert!(format!("{hopped:#}").contains("SSRF"), "{hopped:#}");
}

/// The shared body stream refuses a declared length over its ceiling before the file is created,
/// and stores a body at the ceiling byte for byte — the inbound/outbound bound.
#[tokio::test]
async fn a_declared_length_over_the_ceiling_is_refused_before_writing() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    fake.push_file("clip", b"twelve bytes", Duration::ZERO);
    let telegram = TelegramChannel::new_with("token-a".into(), fake.base.clone(), false);
    let url = format!("{}/file/bottoken-a/{}", fake.base, file_path_of("clip"));
    let dir = tempfile::tempdir().unwrap();
    let over_limit = |size: u64, max: u64| anyhow::anyhow!("{size} over {max}");

    let over = dir.path().join("over.bin");
    let response = telegram.transfer_client().get(&url).send().await.unwrap();
    let refused = stream_capped(response, &over, Some(4), over_limit)
        .await
        .expect_err("the declared length is over the ceiling");
    assert_eq!(refused.to_string(), "12 over 4");
    assert!(!over.exists(), "a refused body must not be created");

    let exact = dir.path().join("exact.bin");
    let response = telegram.transfer_client().get(&url).send().await.unwrap();
    stream_capped(response, &exact, Some(12), over_limit)
        .await
        .expect("a body at the ceiling is delivered");
    assert_eq!(tokio::fs::read(&exact).await.unwrap(), b"twelve bytes");
}

/// A video address whose bytes cannot be had is reported, never pasted into the reply as a bare
/// link: the user reads why, and nothing is uploaded.
#[tokio::test]
async fn an_unfetchable_video_address_is_reported_not_pasted_as_a_link() {
    crate::users::test_util::init_test_store().await;
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    let telegram = TelegramChannel::new_with("token-a".into(), fake.base.clone(), false);
    let address = "http://127.0.0.1:1/clip.mp4".to_string();

    telegram
        .send(&SendMessage {
            content: format!("[VIDEO:{address}]"),
            recipient: "42".to_string(),
            reply_markup: None,
            file_roots: Vec::new(),
        })
        .await
        .expect_err("a failed delivery is reported to the caller");

    assert_eq!(fake.count("sendDocument"), 0);
    assert_eq!(
        fake.sent_texts(),
        vec!["Could not send &quot;clip.mp4&quot;: it could not be fetched or sent."]
    );
}

/// A marker naming a missing file leaves nothing in the delivered body: the conversation that
/// wrote it is told instead, so the owner gets no broken attachment, and the failure is reported
/// back for that conversation. A message that is nothing but the failed marker sends nothing at
/// all.
#[tokio::test]
async fn a_refused_marker_leaves_no_notice_in_the_delivered_message() {
    crate::users::test_util::init_test_store().await;
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    let telegram = TelegramChannel::new_with("token-a".into(), fake.base.clone(), false);
    let dir = tempfile::tempdir().unwrap();
    let roots = vec![dir.path().to_path_buf()];

    let outcome = telegram
        .send(&SendMessage {
            content: "Here it is [FILE:missing.pdf]".to_string(),
            recipient: "42".to_string(),
            reply_markup: None,
            file_roots: roots.clone(),
        })
        .await
        .expect("the surrounding text is delivered");

    assert_eq!(
        refusal_lines(&outcome.refused),
        vec!["[FILE:missing.pdf]: file not found."]
    );
    assert_eq!(fake.sent_texts(), vec!["Here it is"]);

    let outcome = telegram
        .send(&SendMessage {
            content: "[FILE:missing.pdf]".to_string(),
            recipient: "42".to_string(),
            reply_markup: None,
            file_roots: roots,
        })
        .await
        .expect("nothing to deliver is not an error");

    assert_eq!(outcome.refused.len(), 1);
    assert_eq!(
        fake.sent_texts(),
        vec!["Here it is"],
        "a message that is only the failed marker sends nothing"
    );
}

/// An attachment that fails after a refused marker still returns the report: the text went out,
/// so the failure says the owner received the reply and only the attachment was lost.
#[tokio::test]
async fn a_failed_attachment_still_carries_the_refused_marker_report() {
    crate::users::test_util::init_test_store().await;
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    let telegram = TelegramChannel::new_with("token-a".into(), fake.base.clone(), false);
    let dir = tempfile::tempdir().unwrap();
    let roots = vec![dir.path().to_path_buf()];

    let failure = telegram
        .send(&SendMessage {
            content: "Here it is [FILE:missing.pdf] [VIDEO:http://127.0.0.1:1/clip.mp4]"
                .to_string(),
            recipient: "42".to_string(),
            reply_markup: None,
            file_roots: roots,
        })
        .await
        .expect_err("the attachment failure is reported");

    assert_eq!(
        refusal_lines(&failure.refused),
        vec!["[FILE:missing.pdf]: file not found."]
    );
    assert!(
        failure.delivered,
        "the text went out before the attachment failed"
    );
    assert!(
        fake.sent_texts().iter().any(|text| text == "Here it is"),
        "the chat received the body text"
    );
}

/// The local server hands a file over as a path on this machine instead of serving it: the file is
/// copied from there — left in place, because the server's own record still names it — and no HTTP
/// request is involved.
#[tokio::test]
async fn a_local_server_hands_a_file_over_as_a_path() {
    let root = tempfile::tempdir().unwrap();
    let _files = crate::util::test::set_env_var(
        "MAHBOT_TGAPI_FILES_DIR",
        Some(root.path().to_str().unwrap()),
    );
    let source = root.path().join("clip.bin");
    std::fs::write(&source, vec![7u8; 4096]).unwrap();
    let dest = root.path().join("staged.bin");
    let channel = TelegramChannel::new_with("token".into(), "http://127.0.0.1:9".into(), true);

    channel
        .download_file(&source.to_string_lossy(), &dest, Some(4096))
        .await
        .expect("the file on disk is taken without a transfer");

    assert_eq!(std::fs::read(&dest).unwrap(), vec![7u8; 4096]);
    assert!(
        source.exists(),
        "the server's copy stays where its own record names it"
    );
}

/// A hand-over naming a file outside the directory the product gave the server is refused: nothing
/// is copied, and the named file is not deleted.
#[tokio::test]
async fn a_local_hand_over_outside_the_servers_directory_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let _files = crate::util::test::set_env_var(
        "MAHBOT_TGAPI_FILES_DIR",
        Some(root.path().to_str().unwrap()),
    );
    let elsewhere = tempfile::tempdir().unwrap();
    let source = elsewhere.path().join("owner.txt");
    std::fs::write(&source, b"owner data").unwrap();
    let dest = root.path().join("staged.txt");
    let channel = TelegramChannel::new_with("token".into(), "http://127.0.0.1:9".into(), true);

    let started = std::time::Instant::now();
    let err = channel
        .download_file(&source.to_string_lossy(), &dest, Some(10))
        .await
        .expect_err("a hand-over outside the server's directory must be refused");
    assert!(
        err.to_string().contains(LOCAL_FILE_OUTSIDE_REASON),
        "the refusal must name the hand-over: {err}"
    );
    // Refused as what it is, before the wait: not probed for a minute and then reported as a
    // transfer that never finished.
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "an outside path is not waited for"
    );
    assert!(source.exists(), "the named file must be left untouched");
    assert!(!dest.exists(), "nothing may be copied out of the refusal");
}

/// A hand-over exactly at the declared size is the whole file as soon as it is measured; one that
/// stops changing short of it is a stall, so the wait keeps waiting until the write reaches the
/// declared size. With none declared, a size that settles completes it.
#[tokio::test]
async fn a_hand_over_completes_at_the_declared_size_and_survives_a_stall() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("clip.bin");
    std::fs::write(&source, vec![7u8; 9]).unwrap();

    wait_for_hand_over(&source, Some(9))
        .await
        .expect("a file of the size the attachment declared is the whole file");
    wait_for_hand_over(&source, None)
        .await
        .expect("an undeclared size is complete once it stops changing");

    let stalled = dir.path().join("stalled.bin");
    std::fs::write(&stalled, vec![7u8; 3]).unwrap();
    let (waited, ()) = tokio::join!(wait_for_hand_over(&stalled, Some(9)), async {
        tokio::time::sleep(Duration::from_millis(300)).await;
        std::fs::write(&stalled, vec![7u8; 9]).unwrap();
    });
    waited.expect("a stalled write is not the end of the transfer");
}

/// Each local hand-over failure reaches the owner as what it is — a file from outside the directory
/// the server was given, one that never arrived, one that stopped short — and only a failure
/// nothing recognises is still the transport's own refusal.
#[test]
fn every_local_hand_over_failure_reaches_the_owner_as_itself() {
    let outside: anyhow::Error = LocalHandOver::Outside.into();
    assert_eq!(
        download_failure_reason(&outside, TransferLimits::LOCAL),
        LOCAL_FILE_OUTSIDE_REASON
    );

    let missing: anyhow::Error = LocalHandOver::Missing.into();
    assert_eq!(
        download_failure_reason(&missing, TransferLimits::LOCAL),
        LOCAL_FILE_MISSING_REASON
    );

    let incomplete: anyhow::Error = LocalHandOver::Incomplete {
        got: 3,
        expected: 9,
    }
    .into();
    assert_eq!(
        download_failure_reason(&incomplete, TransferLimits::LOCAL),
        local_file_incomplete_reason(3, 9)
    );

    // Only a failure nothing recognises is still the transport's own refusal, and it hints at the
    // bot-download cap only in the mode that has one.
    let unknown = anyhow::anyhow!("the staging disk went away");
    assert_eq!(
        download_failure_reason(&unknown, TransferLimits::CLOUD),
        "Telegram refused the transfer (it may exceed what the bot can download)"
    );
    assert_eq!(
        download_failure_reason(&unknown, TransferLimits::LOCAL),
        "Telegram refused the transfer"
    );
}

/// The declared-size gate's boundaries: exactly at the cloud limit passes, one byte over trips it,
/// and the local mode — no download cap — never refuses on size at all.
#[test]
fn declared_size_refusal_boundaries() {
    const MIB: u64 = 1024 * 1024;
    let cloud = TransferLimits::CLOUD;
    let limit = telegram_download_limit_reason(TELEGRAM_MAX_FILE_DOWNLOAD_BYTES);
    let cases: &[(u64, Option<String>)] = &[
        (0, None),
        (20 * MIB, None),
        (20 * MIB + 1, Some(limit.clone())),
        (2000 * MIB, Some(limit)),
    ];
    for (size, expected) in cases {
        assert_eq!(
            declared_size_refusal(cloud, *size),
            *expected,
            "size {size} bytes"
        );
    }
    assert_eq!(declared_size_refusal(TransferLimits::LOCAL, u64::MAX), None);
}

#[test]
fn attachment_rejection_content_shapes() {
    let reason = telegram_download_limit_reason(TELEGRAM_MAX_FILE_DOWNLOAD_BYTES);
    assert_eq!(
        attachment_rejection_content(Some("report.pdf"), NOT_RECEIVED, &reason, Some("my report")),
        format!("my report\n\n[File report.pdf: not received — {reason}]")
    );
    assert_eq!(
        attachment_rejection_content(Some("report.pdf"), NOT_STORED, &reason, None),
        format!("[File report.pdf: could not be stored — {reason}]")
    );
    assert_eq!(
        attachment_rejection_content(None, NOT_RECEIVED, &reason, None),
        format!("[File not received — {reason}]")
    );
    for note in [
        attachment_rejection_content(Some("report.pdf"), NOT_RECEIVED, &reason, None),
        attachment_rejection_content(None, NOT_RECEIVED, &reason, None),
    ] {
        assert!(
            !crate::util::MEDIA_MARKER_RE.is_match(&note),
            "a rejection note must not read as a media marker: {note}"
        );
    }
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
        ch.parse_update_message(&update)
            .await
            .expect("parse the update")
            .is_none(),
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
        .expect("parse the update")
        .expect("message with thread_id should parse");

    assert_eq!(msg.user_name, "alice");
    assert_eq!(msg.reply_target, "-100200300:789");
    assert_eq!(msg.content, "hello from topic");
}

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
        .expect("parse the query")
        .expect("callback query should parse with valid user");

    assert_eq!(msg.user_name, "alice");
    assert_eq!(msg.reply_target, "-100200300");
    assert_eq!(msg.content, "set_model|test-model");
    assert_eq!(msg.channel, "telegram");
    assert_eq!(msg.callback_query_id.as_deref(), Some("12345"));
    assert_eq!(msg.chat_id.as_deref(), Some("-100200300"));
    assert_eq!(msg.message_id, Some(100));
}

/// What a callback query narrows on: absent/null data or message, and a sender nothing authorizes,
/// are refused — while an empty-string data stays valid with empty content, and a null
/// callback_query_id stays valid as `None`.
#[tokio::test]
async fn parse_callback_query_narrows_only_on_absent_inputs() {
    let ch = test_channel().await;
    let unknown_user = serde_json::json!({ "id": 999, "username": "unknown_user" });
    let rejected = [
        ("no data", [("data", serde_json::Value::Null)]),
        ("no message", [("message", serde_json::Value::Null)]),
        ("unknown user", [("from", unknown_user)]),
    ];
    for (name, overrides) in &rejected {
        let cq = test_callback_query(overrides);
        assert!(
            ch.parse_callback_query(&cq)
                .await
                .expect("parse the query")
                .is_none(),
            "case {name}: expected rejection"
        );
    }

    // Regression pair: the `?` guard rejects absent/null data but not an empty string (empty
    // content, distinct from `process_updates`' `unwrap_or("")` ACK check), and a null/absent
    // `callback_query_id` stays valid as `None` — unlike missing data.
    let accepted = [
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
    for (name, overrides, content, cq_id) in &accepted {
        let cq = test_callback_query(overrides);
        let msg = ch
            .parse_callback_query(&cq)
            .await
            .expect("parse the query")
            .unwrap_or_else(|| panic!("case {name}: expected a valid message"));
        assert_eq!(msg.content, *content, "case {name}");
        assert_eq!(msg.callback_query_id.as_deref(), *cq_id, "case {name}");
    }
}

/// The chunker never hands Telegram more than it accepts: the limit does not split, one character
/// past it does, the join rebuilds the message, a newline is preferred over a mid-word cut, and a
/// code block or emoji at the boundary stays within the limit.
#[test]
fn splitting_respects_the_telegram_limit() {
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

    let msg = format!("```python\n{}```\nMore text", "x".repeat(4085));
    for p in &split_message_for_telegram(&msg) {
        assert!(p.len() <= TELEGRAM_MAX_MESSAGE_LENGTH);
    }
    let msg = format!("{}🎉🎊", "a".repeat(4094));
    for p in &split_message_for_telegram(&msg) {
        assert!(p.chars().count() <= TELEGRAM_MAX_MESSAGE_LENGTH);
    }

    // Regression: when the only newline is in the first half of the search window and no spaces
    // exist, a hard split would land mid-word; the newline fallback (tier 3) stops it.
    let msg = format!("{}\n{}", "a".repeat(1000), "x".repeat(5000));
    let chunks = split_message_for_telegram(&msg);
    for (i, chunk) in chunks.iter().enumerate() {
        assert!(
            chunk.chars().count() <= TELEGRAM_MAX_MESSAGE_LENGTH,
            "chunk {} has {} chars (limit {})",
            i,
            chunk.chars().count(),
            TELEGRAM_MAX_MESSAGE_LENGTH,
        );
    }
    assert_eq!(chunks.join(""), msg);
    assert!(
        chunks[0].ends_with('\n'),
        "first chunk should end with newline, got: {:?}",
        chunks[0].chars().rev().take(10).collect::<String>()
    );

    // Wrapping: `send_text_chunks` prepends a continuation prefix (29 chars mid-message, 13 on the
    // last), so every raw chunk must leave that room — three chunks cover it.
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

    // The boundary itself is not split: a 4066-char message leaves the room its continuation prefix
    // needs, and a 4096-char one is sent without any prefix at all.
    for (size, wrapped) in [(4066, true), (4096, false)] {
        let boundary = "b".repeat(size);
        let chunks = split_message_for_telegram(&boundary);
        assert_eq!(chunks.len(), 1, "{size}-char message should not split");
        if wrapped {
            let wrapped = format!("(continued)\n\n{}", chunks[0]);
            assert!(
                wrapped.chars().count() <= 4096,
                "{size}-char boundary wrapped: {} > 4096",
                wrapped.chars().count()
            );
        }
    }

    // Regression: a split point inside a long HTML tag whose '>' fell beyond the window could push
    // a chunk past the sendable limit, and Telegram rejected the send (HTML and the plain-text
    // retry both fail), dropping the message. The clamp must target the byte offset of the
    // 4066-char boundary, never a raw byte count: clamping to byte 4066 panics on 3-byte CJK and
    // under-fills 2-byte Cyrillic.
    let tag = format!(
        "<a href=\"https://example.com/{}\">link</a>",
        "y".repeat(120)
    );
    for (name, msg) in [
        ("ascii", format!("{}{}", "x".repeat(4000), tag)),
        ("cjk", format!("{}{}", "界".repeat(4000), tag)),
        ("cyrillic", format!("{}{}", "ы".repeat(4000), tag)),
    ] {
        let chunks = split_message_for_telegram(&msg);
        assert_eq!(chunks.join(""), msg, "{name}: reconstruction");
        assert!(chunks.len() >= 2, "{name}: expected a split");
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.chars().count()
                    <= TELEGRAM_MAX_MESSAGE_LENGTH - TELEGRAM_CONTINUATION_OVERHEAD,
                "{name}: chunk {i} raw {} chars exceeds the 4066 budget",
                chunk.chars().count(),
            );
            let wrapped = wrap_chunk(chunk, i, chunks.len());
            assert!(
                wrapped.chars().count() <= TELEGRAM_MAX_MESSAGE_LENGTH,
                "{name}: chunk {i} wrapped {} chars exceeds 4096",
                wrapped.chars().count(),
            );
        }
        assert!(
            chunks[0].chars().count() >= 4000,
            "{name}: first chunk only {} chars — budget underused",
            chunks[0].chars().count(),
        );
    }
}

#[tokio::test]
async fn usernameless_sender_stays_unauthorized_despite_legacy_binding() {
    let _ch = test_channel().await;
    crate::users::store()
        .bind_channel("alice", "telegram", "unknown")
        .await
        .unwrap();
    // The sender carries no nickname at all and no numeric binding exists, so the legacy "unknown"
    // row must not authorize them.
    for sender in [
        serde_json::json!({"from": {"id": 1}}),
        serde_json::json!({"from": {"id": 1, "username": "unknown"}}),
    ] {
        assert!(
            resolve_authorized_sender(&sender, &serde_json::json!({"chat": {"id": 42}}))
                .await
                .expect("resolve the sender")
                .is_none(),
            "the reserved sentinel must never authorize: {sender}"
        );
    }
    crate::users::store()
        .unbind_channel("alice", "telegram", "unknown")
        .await
        .unwrap();
}

#[tokio::test]
async fn usernameless_sender_is_authorized_by_a_numeric_binding() {
    // The same sender as the sentinel test above, but bound by number: the no-nickname refusal
    // belongs to the nickname way only, so the legacy "unknown" row (still bound to alice here)
    // must not swallow a numeric match. The message itself carries no nickname at all.
    let _ch = test_channel().await;
    let store = crate::users::store();
    store
        .bind_channel("alice", "telegram", "unknown")
        .await
        .unwrap();
    store.add_user("num_user").await.unwrap();
    store.bind_telegram("num_user", "31415926").await.unwrap();
    assert_eq!(
        resolve_authorized_sender(
            &serde_json::json!({"from": {"id": 31_415_926}}),
            &serde_json::json!({"chat": {"id": 42}}),
        )
        .await
        .expect("resolve the sender"),
        Some(("num_user".to_string(), "42".to_string(), "42".to_string())),
    );
    store
        .unbind_channel("num_user", "telegram", "31415926")
        .await
        .unwrap();
    store
        .unbind_channel("alice", "telegram", "unknown")
        .await
        .unwrap();
}

#[tokio::test]
async fn numeric_binding_outranks_a_nickname_bound_to_another_account() {
    let _ch = test_channel().await;
    let store = crate::users::store();
    store.add_user("num_rank").await.unwrap();
    store.add_user("nick_rank").await.unwrap();
    // The nickname's row is visited first (the scan follows the `(channel, identifier)` unique
    // index, and `!` sorts before any digit), so returning the first matching row of either kind
    // would return the nickname's account: the number must win on identity.
    store.bind_telegram("num_rank", "27182818").await.unwrap();
    store.bind_telegram("nick_rank", "!ranknick").await.unwrap();

    // The sender carries both identities, each bound to a different account: the number decides,
    // since it is the one that survives a rename.
    let authorized = resolve_authorized_sender(
        &serde_json::json!({"from": {"id": 27_182_818, "username": "!ranknick"}}),
        &serde_json::json!({"chat": {"id": 42}}),
    )
    .await
    .expect("resolve the sender")
    .expect("the number binds");
    assert_eq!(authorized.0, "num_rank");

    store
        .unbind_channel("num_rank", "telegram", "27182818")
        .await
        .unwrap();
    store
        .unbind_channel("nick_rank", "telegram", "!ranknick")
        .await
        .unwrap();
}

#[tokio::test]
async fn bot_sender_is_refused_even_when_its_nickname_is_bound() {
    // "alice" is bound to alice in the shared fixtures, so only the bot identity can refuse.
    let _ch = test_channel().await;
    assert!(
        resolve_authorized_sender(
            &serde_json::json!({
                "from": {"id": 424_242, "username": "alice", "is_bot": true}
            }),
            &serde_json::json!({"chat": {"id": 42}}),
        )
        .await
        .expect("resolve the sender")
        .is_none()
    );
}

#[tokio::test]
async fn chat_attributed_message_is_refused_even_when_its_sender_is_bound() {
    // A channel post (or the anonymous group administrator) cites a `from` naming the bound
    // nickname, yet the message speaks for the chat, not for that person.
    let _ch = test_channel().await;
    assert!(
        resolve_authorized_sender(
            &serde_json::json!({
                "sender_chat": {"id": -100_200_300, "type": "channel", "title": "Chan"},
                "from": {"id": 424_243, "username": "alice"}
            }),
            &serde_json::json!({"chat": {"id": 42}}),
        )
        .await
        .expect("resolve the sender")
        .is_none()
    );
}

#[tokio::test]
async fn telegram_service_identity_senders_are_refused() {
    let _ch = test_channel().await;
    // Each carries a bound nickname too, so only the identity itself can produce the refusal.
    let cases = [
        ("anonymous group administrator", 1_087_968_824_i64),
        ("message attributed to a channel", 136_817_688),
        ("auto-forwarded channel post", 777_000),
    ];
    for (name, id) in cases {
        assert!(
            resolve_authorized_sender(
                &serde_json::json!({"from": {"id": id, "username": "alice"}}),
                &serde_json::json!({"chat": {"id": 42}}),
            )
            .await
            .expect("resolve the sender")
            .is_none(),
            "case {name}: expected rejection"
        );
    }
}

#[tokio::test]
async fn callback_query_authorizes_the_person_when_its_message_is_chat_attributed() {
    // The clicker is `from` of the callback_query; the keyboard's message can be chat-attributed
    // (`sender_chat`), which must not refuse the person who pressed.
    let ch = test_channel().await;
    let cq = test_callback_query(&[(
        "message",
        serde_json::json!({
            "message_id": 100,
            "sender_chat": {"id": -100_200_300, "type": "channel", "title": "Chan"},
            "chat": {"id": -100_200_300}
        }),
    )]);

    let msg = ch
        .parse_callback_query(&cq)
        .await
        .expect("parse the query")
        .expect("the clicker's own identity authorizes");
    assert_eq!(msg.user_name, "alice");
    assert_eq!(msg.chat_id.as_deref(), Some("-100200300"));
    assert_eq!(msg.message_id, Some(100));
}

/// `build_reply_reference` is the composition around the shared normalization (asserted in
/// `reply_tests`): it names the replied-to author, falls back to the caption when a media message's
/// `text` is empty, and reports nothing when the message is not a reply.
#[test]
fn build_reply_reference_reads_text_or_caption_and_needs_a_reply() {
    let message = serde_json::json!({
        "reply_to_message": {
            "from": { "username": "bob" },
            "photo": [{ "file_id": "x", "width": 1, "height": 1 }],
            "caption": "at the <beach>"
        }
    });
    let reply = build_reply_reference(&message).unwrap();
    assert_eq!(reply.author, "@bob");
    assert_eq!(reply.snippet, "at the beach");

    // No reply: absent and a null reply_to_message (not an object) are nothing to report.
    for message in [
        serde_json::json!({ "text": "no reply here" }),
        serde_json::json!({ "text": "hi", "reply_to_message": null }),
    ] {
        assert!(build_reply_reference(&message).is_none(), "{message}");
    }
}

#[test]
fn replied_to_sender_label_chain() {
    assert_eq!(
        replied_to_sender_label(&serde_json::json!({ "from": { "username": "alice" } })),
        "@alice"
    );
    assert_eq!(
        replied_to_sender_label(&serde_json::json!({ "from": { "id": 1, "first_name": "Bob" } })),
        "Bob"
    );
    assert_eq!(
        replied_to_sender_label(&serde_json::json!({ "from": { "id": 1 } })),
        "user"
    );
    assert_eq!(
        replied_to_sender_label(&serde_json::json!({ "sender_chat": { "title": "Daily News" } })),
        "Daily News"
    );
    assert_eq!(
        replied_to_sender_label(&serde_json::json!({ "sender_chat": { "id": 1 } })),
        "channel"
    );
    assert_eq!(
        replied_to_sender_label(
            &serde_json::json!({ "from": { "is_bot": true, "username": "poll_bot" } })
        ),
        "@poll_bot"
    );
    assert_eq!(
        replied_to_sender_label(&serde_json::json!({ "from": { "is_bot": true } })),
        "bot"
    );
    assert_eq!(
        replied_to_sender_label(&serde_json::json!({ "from": { "first_name": "<Bob>" } })),
        "Bob"
    );
    assert_eq!(replied_to_sender_label(&serde_json::json!({})), "user");
}

#[tokio::test]
async fn parse_update_message_populates_reply_reference() {
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
    let parsed = ch
        .parse_update_message(&update)
        .await
        .expect("parse the update")
        .expect("a message");
    assert_eq!(parsed.content, "translate this");
    let reply = parsed
        .reply_reference
        .expect("parsed message should carry the reply reference");
    assert_eq!(reply.author, "@bot");
    assert_eq!(reply.snippet, "Bonjour le monde");
}

#[tokio::test]
async fn attachment_message_reply_carries_reference() {
    // `try_parse_attachment_message` builds its context through `extract_message_context`, so a
    // reply to a media message must carry the reference too (the no-silent-drop path).
    let ch = test_channel().await;
    let message = serde_json::json!({
        "message_id": 7,
        "from": { "id": 1, "username": "alice" },
        "chat": { "id": 100, "type": "private" },
        "photo": [{ "file_id": "x", "width": 1, "height": 1 }],
        "reply_to_message": {
            "from": { "username": "bob" },
            "photo": [{ "file_id": "y", "width": 1, "height": 1 }]
        }
    });
    let ctx = ch
        .extract_message_context(&message)
        .await
        .expect("resolve the context")
        .expect("valid context");
    let reply = ctx.reply_reference.expect("media reply carries reference");
    assert_eq!(reply.author, "@bob");
    assert_eq!(reply.snippet, "[Photo]");
    assert_eq!(ctx.chat_id, "100");
    assert_eq!(ctx.message_id, 7);
}

#[tokio::test]
async fn reply_that_is_a_command_parses_as_command() {
    let ch = test_channel().await;
    let update = test_update(&[(
        "message",
        serde_json::json!({
            "message_id": 11,
            "text": "/clear",
            "from": { "id": 1, "username": "alice" },
            "chat": { "id": 100, "type": "private" },
            "reply_to_message": {
                "from": { "username": "bot" },
                "text": "some earlier text"
            }
        }),
    )]);
    let parsed = ch
        .parse_update_message(&update)
        .await
        .expect("parse the update")
        .expect("a message");
    assert_eq!(parsed.content, "/clear");
    assert!(crate::parse_bot_command(&parsed.content).is_some());
}

#[tokio::test]
async fn reply_to_command_keeps_reference() {
    let ch = test_channel().await;
    let update = test_update(&[(
        "message",
        serde_json::json!({
            "message_id": 12,
            "text": "why did you clear?",
            "from": { "id": 1, "username": "alice" },
            "chat": { "id": 100, "type": "private" },
            "reply_to_message": {
                "from": { "username": "bot" },
                "text": "/clear"
            }
        }),
    )]);
    let parsed = ch
        .parse_update_message(&update)
        .await
        .expect("parse the update")
        .expect("a message");
    // A non-command reply to a command message keeps its reference and routes to the agent.
    assert_eq!(parsed.content, "why did you clear?");
    assert!(crate::parse_bot_command(&parsed.content).is_none());
    let reply = parsed
        .reply_reference
        .expect("non-command reply keeps its reference");
    assert_eq!(reply.snippet, "/clear");
}

#[test]
#[expect(clippy::too_many_lines)]
fn attachment_content_format_rules() {
    let c = format_attachment_content(
        IncomingAttachmentKind::Photo,
        std::path::Path::new("/tmp/workspace/photo.jpg"),
        None,
    );
    assert_eq!(c, "[IMAGE:/tmp/workspace/photo.jpg]");
    let c = format_attachment_content(
        IncomingAttachmentKind::Document,
        std::path::Path::new("/tmp/workspace/report.pdf"),
        None,
    );
    assert_eq!(c, "[FILE:/tmp/workspace/report.pdf]");
    assert!(!c.contains("[IMAGE:"));
    let dir = tempfile::tempdir().unwrap();
    let nameless = dir.path().join("file_1_2.bin");
    write_test_png(&nameless);
    let c = format_attachment_content(IncomingAttachmentKind::Document, &nameless, None);
    assert_eq!(c, format!("[IMAGE:{}]", nameless.display()));
    let blob = dir.path().join("file_1_3.bin");
    std::fs::write(&blob, b"definitely not a raster").unwrap();
    let c = format_attachment_content(IncomingAttachmentKind::Document, &blob, None);
    assert_eq!(c, format!("[FILE:{}]", blob.display()));
    let c = format_attachment_content(
        IncomingAttachmentKind::Photo,
        std::path::Path::new("/tmp/workspace/notes.md"),
        None,
    );
    assert!(!c.contains("[IMAGE:"));
    assert!(c.starts_with("[FILE:"));
    for path in [
        "/tmp/workspace/file.md",
        "/tmp/workspace/file.txt",
        "/tmp/workspace/file.pdf",
        "/tmp/workspace/file.csv",
        "/tmp/workspace/file.json",
        "/tmp/workspace/file.zip",
        "/tmp/workspace/file",
    ] {
        let c = format_attachment_content(
            IncomingAttachmentKind::Photo,
            std::path::Path::new(path),
            None,
        );
        assert!(!c.contains("[IMAGE:"), "{path}: should not get [IMAGE:]");
        assert!(c.starts_with("[FILE:"), "{path}: should use [FILE:]");
    }
    // image extensions produce [IMAGE:] (PNG/JPEG/WebP only; gif/bmp are not routed)
    for ext in ["png", "jpg", "jpeg", "webp"] {
        let c = format_attachment_content(
            IncomingAttachmentKind::Photo,
            std::path::Path::new(&format!("/tmp/workspace/photo.{ext}")),
            None,
        );
        assert!(c.starts_with("[IMAGE:"), "{ext}: should get [IMAGE:]");
    }
    let c = format_attachment_content(
        IncomingAttachmentKind::Document,
        std::path::Path::new("/tmp/workspace/image.jpg"),
        None,
    );
    assert_eq!(c, "[IMAGE:/tmp/workspace/image.jpg]");
    // Document kind + no extension + jpeg MIME types → [IMAGE:] (mime fallback), admitting both the
    // canonical image/jpeg and the legacy image/jpg alias.
    for mime in ["image/jpeg", "image/jpg"] {
        let c = format_attachment_content(
            IncomingAttachmentKind::Document,
            std::path::Path::new("/tmp/workspace/image_no_ext"),
            Some(mime),
        );
        assert_eq!(c, "[IMAGE:/tmp/workspace/image_no_ext]", "{mime}");
    }
    // gif/bmp MIME types are NOT routed as images (mime fallback whitelist) and fall through to
    // [FILE:]; the legacy image/x-ms-bmp alias is excluded too.
    for mime in ["image/gif", "image/bmp", "image/x-ms-bmp"] {
        let c = format_attachment_content(
            IncomingAttachmentKind::Document,
            std::path::Path::new("/tmp/workspace/anim_no_ext"),
            Some(mime),
        );
        assert!(!c.contains("[IMAGE:"), "{mime}: should not get [IMAGE:]");
        assert!(c.starts_with("[FILE:"), "{mime}: should use [FILE:]");
    }
    // A sound file and a recorded voice message both arrive as an [AUDIO:] marker, which the
    // enrichment pass saves into the workspace and transcribes — the marker follows the kind, not
    // the platform.
    let c = format_attachment_content(
        IncomingAttachmentKind::Audio,
        std::path::Path::new("/tmp/workspace/voice.ogg"),
        None,
    );
    assert_eq!(c, "[AUDIO:/tmp/workspace/voice.ogg]");
    let c = format_attachment_content(
        IncomingAttachmentKind::Voice,
        std::path::Path::new("/tmp/workspace/note.oga"),
        None,
    );
    assert_eq!(c, "[AUDIO:/tmp/workspace/note.oga]");
    let c = format_attachment_content(
        IncomingAttachmentKind::Audio,
        std::path::Path::new("/tmp/workspace/song.mp3"),
        Some("audio/mpeg"),
    );
    assert_eq!(c, "[AUDIO:/tmp/workspace/song.mp3]");
    let c = format_attachment_content(
        IncomingAttachmentKind::Video,
        std::path::Path::new("/tmp/workspace/clip.mp4"),
        None,
    );
    assert_eq!(c, "[VIDEO:/tmp/workspace/clip.mp4]");
    let c = format_attachment_content(
        IncomingAttachmentKind::Document,
        std::path::Path::new("/tmp/workspace/clip.mp4"),
        None,
    );
    assert_eq!(c, "[VIDEO:/tmp/workspace/clip.mp4]");
    let c = format_attachment_content(
        IncomingAttachmentKind::Document,
        std::path::Path::new("/tmp/workspace/clip_no_ext"),
        Some("video/mp4"),
    );
    assert_eq!(c, "[VIDEO:/tmp/workspace/clip_no_ext]");
    let c = format_attachment_content(
        IncomingAttachmentKind::Video,
        std::path::Path::new("/tmp/workspace/notes.txt"),
        None,
    );
    assert_eq!(c, "[FILE:/tmp/workspace/notes.txt]");
}

#[test]
fn sanitize_attachment_filename_neutralizes_traversal() {
    let fallback = "file_1_2.pdf";
    // Only the final component survives, so an absolute path or a `..` climb can never escape the
    // per-message directory.
    assert_eq!(
        sanitize_attachment_filename("../../etc/passwd", fallback),
        "passwd"
    );
    assert_eq!(
        sanitize_attachment_filename("/etc/passwd", fallback),
        "passwd"
    );
    for rejected in ["..", ".", "", "/", "a/.."] {
        assert_eq!(
            sanitize_attachment_filename(rejected, fallback),
            fallback,
            "{rejected:?}"
        );
    }
    // The fallback is cleaned the same way, so the marker invariant does not depend on the caller
    // having pre-sanitized it.
    assert_eq!(
        sanitize_attachment_filename("..", "we]ird[\u{7}name.bin"),
        "we_ird__name.bin"
    );
    assert_eq!(
        sanitize_attachment_filename("C:\\evil\\x.txt", fallback),
        "C:_evil_x.txt"
    );
    assert_eq!(
        sanitize_attachment_filename("a\nb\r\t.txt", fallback),
        "a_b__.txt"
    );
    // Brackets map out too: the emitted `[FILE:<path>]` marker's path group cannot carry a `]`, and
    // an unsanitized one would truncate the marker.
    assert_eq!(
        sanitize_attachment_filename("notes]v2[1].pdf", fallback),
        "notes_v2_1_.pdf"
    );
    assert_eq!(
        sanitize_attachment_filename(".bashrc", fallback),
        "_.bashrc"
    );
    let long = format!("{}.pdf", "x".repeat(500));
    let sanitized = sanitize_attachment_filename(&long, fallback);
    assert_eq!(sanitized, format!("{}.pdf", "x".repeat(176)));
    // A long extension-only name must not become a dotfile either; the `_` counts toward it.
    let long_ext = format!(".{}", "y".repeat(200));
    let dotted = sanitize_attachment_filename(&long_ext, fallback);
    assert!(dotted.starts_with("_."), "{dotted}");
    assert!(
        dotted.len() <= MAX_ATTACHMENT_FILENAME_BYTES,
        "{dotted} is {} bytes",
        dotted.len()
    );
    // The budget is hard: an extension longer than it leaves no stem room, so the name truncates
    // wholesale instead of appending the full extension.
    let huge_ext = format!("call.log.{}", "y".repeat(200));
    let capped = sanitize_attachment_filename(&huge_ext, fallback);
    assert_eq!(capped.len(), MAX_ATTACHMENT_FILENAME_BYTES);
    assert!(capped.starts_with("call.log."), "{capped}");
    // The same budget with a leading character that cannot be cut in two: the stem keeps nothing,
    // and a bare extension would be a dotfile.
    let narrow = format!("🙂.{}", "y".repeat(177));
    let squeezed = sanitize_attachment_filename(&narrow, fallback);
    assert!(!squeezed.starts_with('.'), "{squeezed}");
    assert_eq!(squeezed.len(), MAX_ATTACHMENT_FILENAME_BYTES);
}

#[test]
fn local_attachment_name_uses_sanitized_sender_name_or_fallback() {
    let attachment = IncomingAttachment {
        file_id: "f".to_string(),
        file_name: Some("../../etc/passwd".to_string()),
        file_size: None,
        caption: None,
        kind: IncomingAttachmentKind::Document,
        mime_type: None,
        urgent: false,
    };
    assert_eq!(local_attachment_name(&attachment, "123", 7, None), "passwd");
    let generated = local_attachment_name(
        &IncomingAttachment {
            file_name: None,
            kind: IncomingAttachmentKind::Video,
            ..attachment.clone()
        },
        "123",
        7,
        None,
    );
    assert_eq!(generated, "video_123_7.mp4");
    // A degenerate sender name has no stem to carry an extension swap, so it falls through to the
    // generated name instead of becoming `_...mp4`.
    for degenerate in ["..", ".", ""] {
        assert_eq!(
            local_attachment_name(
                &IncomingAttachment {
                    file_name: Some(degenerate.to_string()),
                    kind: IncomingAttachmentKind::Video,
                    ..attachment.clone()
                },
                "123",
                7,
                Some("mp4"),
            ),
            "video_123_7.mp4",
            "name {degenerate:?}"
        );
    }
    assert_eq!(
        local_attachment_name(
            &IncomingAttachment {
                file_name: None,
                ..attachment
            },
            "123",
            7,
            None
        ),
        "file_123_7.bin"
    );
}

/// A recording is never given an image extension: the name it is saved under is the name the user
/// later plays, so a voice note keeps OGG, a declared container keeps its own extension, and only
/// an unidentifiable sound file stays neutral.
#[test]
fn local_attachment_name_gives_a_recording_a_sound_extension() {
    let attachment = IncomingAttachment {
        file_id: "f".to_string(),
        file_name: None,
        file_size: None,
        caption: None,
        kind: IncomingAttachmentKind::Voice,
        mime_type: Some("audio/ogg".to_string()),
        urgent: false,
    };
    assert_eq!(
        local_attachment_name(&attachment, "123", 7, None),
        "audio_123_7.ogg"
    );
    // Telegram's own file path names the container when it can, and that wins.
    assert_eq!(
        local_attachment_name(&attachment, "123", 7, Some("oga")),
        "audio_123_7.oga"
    );
    // A sound file takes the container its declared type names.
    let song = IncomingAttachment {
        kind: IncomingAttachmentKind::Audio,
        mime_type: Some("audio/mpeg".to_string()),
        ..attachment.clone()
    };
    assert_eq!(
        local_attachment_name(&song, "123", 7, None),
        "audio_123_7.mp3"
    );
    // With no container named anywhere the extension stays neutral, never an image's.
    assert_eq!(
        local_attachment_name(
            &IncomingAttachment {
                mime_type: None,
                ..song.clone()
            },
            "123",
            7,
            None
        ),
        "audio_123_7.bin"
    );
    // A sender's name without an extension gets the sound one; a name the sender wrote keeps it.
    assert_eq!(
        local_attachment_name(
            &IncomingAttachment {
                file_name: Some("song".to_string()),
                ..song.clone()
            },
            "123",
            7,
            None
        ),
        "song.mp3"
    );
    assert_eq!(
        local_attachment_name(
            &IncomingAttachment {
                file_name: Some("song.m4a".to_string()),
                ..song
            },
            "123",
            7,
            None
        ),
        "song.m4a"
    );
}

#[test]
fn local_attachment_name_sanitizes_remote_extension_fallback() {
    let attachment = IncomingAttachment {
        file_id: "f".to_string(),
        file_name: None,
        file_size: None,
        caption: None,
        kind: IncomingAttachmentKind::Document,
        mime_type: None,
        urgent: false,
    };
    // Telegram reports no file_name for many kinds, so the fallback extension comes from `getFile`
    // — remote input that must not survive into the `[FILE:<path>]` marker path.
    let hostile = "..]x/漢\u{7}evil\u{202e}png";
    let name = local_attachment_name(&attachment, "123", 7, Some(hostile));
    assert_eq!(name, "file_123_7.xevilpng");
    let name = local_attachment_name(&attachment, "123", 7, Some("]/漢\u{7}"));
    assert_eq!(name, "file_123_7.bin");
    let long_ext = "a".repeat(500);
    let name = local_attachment_name(&attachment, "123", 7, Some(long_ext.as_str()));
    assert_eq!(name.len(), MAX_ATTACHMENT_FILENAME_BYTES);
    assert!(name.starts_with("file_123_7."), "{name}");
}

/// One polled update of a photo album: only the field the placement plan reads.
fn album_member_update(message_id: i64, group: &str) -> serde_json::Value {
    test_update(&[(
        "message",
        serde_json::json!({ "message_id": message_id, "media_group_id": group }),
    )])
}

fn album_button_press_update(group: &str) -> serde_json::Value {
    test_update(&[(
        "callback_query",
        test_callback_query(&[(
            "message",
            serde_json::json!({ "message_id": 42, "media_group_id": group }),
        )]),
    )])
}

/// The batch placement plan: an album is parsed whole at its first member and its later members are
/// skipped there — a plain message between them stays where it is — and a button press is never
/// placed inside an album. A press carries the message its button sits on, so joining the album
/// would hand the photo over twice or lose the album's other members.
#[test]
fn poll_plan_places_each_album_and_keeps_a_button_press_out_of_it() {
    let updates = vec![
        album_member_update(41, "g1"),
        test_update(&[]),
        album_member_update(42, "g1"),
        album_member_update(43, "g2"),
    ];
    assert_eq!(
        poll_plan(&updates),
        vec![
            PollPlacement::Album(vec![0, 2]),
            PollPlacement::Single,
            PollPlacement::Skip,
            PollPlacement::Album(vec![3]),
        ]
    );

    let press_first = vec![
        album_button_press_update("g1"),
        album_member_update(42, "g1"),
        album_member_update(43, "g1"),
    ];
    assert_eq!(
        poll_plan(&press_first),
        vec![
            PollPlacement::Callback,
            PollPlacement::Album(vec![1, 2]),
            PollPlacement::Skip,
        ]
    );

    let press_between = vec![
        album_member_update(41, "g1"),
        album_button_press_update("g1"),
        album_member_update(43, "g1"),
    ];
    assert_eq!(
        poll_plan(&press_between),
        vec![
            PollPlacement::Album(vec![0, 2]),
            PollPlacement::Callback,
            PollPlacement::Skip,
        ]
    );
}

#[test]
fn album_members_merge_into_one_message() {
    let first_chat = "-100";
    let second_chat = "-200";
    let first = ChannelMessage {
        message_id: Some(41),
        chat_id: Some(first_chat.into()),
        content: "a".into(),
        attachment_dirs: vec![crate::util::telegram_staging_dir_name(first_chat, 41)],
        update_ids: vec![7],
        ..test_msg("alice", "", "telegram", "chat")
    };
    let second = ChannelMessage {
        message_id: Some(42),
        chat_id: Some(second_chat.into()),
        content: "b".into(),
        attachment_dirs: vec![
            crate::util::telegram_staging_dir_name(second_chat, 42),
            crate::util::telegram_staging_dir_name(second_chat, 43),
        ],
        update_ids: vec![8],
        forwarded: true,
        ..test_msg("bob", "", "telegram", "other")
    };

    let album = merge_album_member(None, first.clone());
    assert_eq!(album.content, "a");
    assert_eq!(album.attachment_dirs, first.attachment_dirs);
    assert!(
        !album.forwarded,
        "the first member is not forwarded content"
    );

    let album = merge_album_member(Some(album), second);
    assert_eq!(album.content, "a\nb");
    assert!(
        album.forwarded,
        "a forwarded member makes the album forwarded content"
    );
    assert_eq!(
        album.attachment_dirs,
        vec![
            crate::util::telegram_staging_dir_name(first_chat, 41),
            crate::util::telegram_staging_dir_name(second_chat, 42),
            crate::util::telegram_staging_dir_name(second_chat, 43),
        ]
    );
    // Every member's record entry comes along: an album is one incoming event whose parts are each
    // their own entry, and this merged message is what the hand-over settles — a member left off it
    // would stay held and be taken up a second time by a later run.
    assert_eq!(album.update_ids, vec![7, 8]);
}

#[test]
fn is_control_message_covers_commands_and_callbacks() {
    assert!(is_control_message(&test_msg(
        "alice", "/board", "telegram", "chat"
    )));
    assert!(is_control_message(&test_msg(
        "alice", "/clear", "telegram", "chat"
    )));
    assert!(is_control_message(&test_msg(
        "alice",
        "/image_models",
        "telegram",
        "chat"
    )));
    // The dispatch loop decodes `__act__` payloads from the text alone, so the collector must too —
    // otherwise a burst would fuse the payload into an agent turn while the same text sent alone
    // would be an action. The payload comes from a previous build's keyboard and must stay a
    // control input even though its action is no longer handled.
    assert!(is_control_message(&test_msg(
        "alice",
        "__act__set_workspace|ws",
        "telegram",
        "chat"
    )));
    // The retired workspace, pause and maintenance commands are ordinary chat now — in every
    // spelling — so the collector must not hold them back.
    for text in [
        "/workspace",
        "/pause",
        "/unpause",
        "/maintenance",
        "/maintenance on",
        "/maintenance_on",
        "/MAINTENANCE_OFF",
    ] {
        assert!(
            control_input(&test_msg("alice", text, "telegram", "chat")).is_none(),
            "{text} is no longer a command"
        );
    }

    let mut callback = test_msg("alice", "anything", "telegram", "chat");
    callback.callback_query_id = Some("1".to_string());
    assert!(is_control_message(&callback));
    // The press carries no control input of its own, so it still reaches the agent on the dispatch
    // path — it is only barred from being collected.
    assert!(control_input(&callback).is_none());

    assert!(!is_control_message(&test_msg(
        "alice", "/board", "gui", "chat"
    )));
    assert!(!is_control_message(&test_msg(
        "alice", "hello", "telegram", "chat"
    )));
    assert!(!is_control_message(&test_msg(
        "alice",
        "/etc/hosts",
        "telegram",
        "chat"
    )));

    // A message that opened with the urgent mark is never a command: the mark was consumed, so its
    // content reads as one, but it must reach the agent.
    let mut urgent_command = test_msg("alice", "/clear", "telegram", "chat");
    urgent_command.urgent = true;
    assert!(!is_control_message(&urgent_command));
    assert!(control_input(&urgent_command).is_none());
}

#[tokio::test]
async fn a_leading_mark_is_consumed_from_a_typed_message() {
    let ch = test_channel().await;
    let update = test_update(&[(
        "message",
        serde_json::json!({
            "message_id": 44,
            "text": "  !/clear",
            "from": { "id": 555, "username": "alice" },
            "chat": { "id": -100_200_300 }
        }),
    )]);

    let msg = parsed_message(&ch, &update).await;

    assert_eq!(msg.content, "/clear");
    assert!(msg.urgent);
    assert!(control_input(&msg).is_none());
    assert!(!is_control_message(&msg));
}

/// Only the start of the typed text counts; a mark anywhere else is ordinary text and leaves the
/// message and its urgency as written.
#[tokio::test]
async fn a_mark_is_read_only_at_the_start_of_the_typed_text() {
    let ch = test_channel().await;
    for (typed, content, urgent) in [
        ("!hello", "hello", true),
        ("hello!", "hello!", false),
        ("what? !", "what? !", false),
    ] {
        let update = test_update(&[(
            "message",
            serde_json::json!({
                "message_id": 45,
                "text": typed,
                "from": { "id": 555, "username": "alice" },
                "chat": { "id": -100_200_300 }
            }),
        )]);
        let msg = parsed_message(&ch, &update).await;
        assert_eq!(msg.content, content, "{typed}");
        assert_eq!(msg.urgent, urgent, "{typed}");
    }
}

/// A caption is text the user typed, so it carries the mark like any message does; the album its
/// parts fold into keeps it.
#[test]
fn a_caption_carries_the_mark_and_the_album_keeps_it() {
    let attachment = |caption: &str| {
        TelegramChannel::build_attachment(
            &serde_json::json!({ "file_id": "x" }),
            &serde_json::json!({ "caption": caption }),
            IncomingAttachmentKind::Photo,
        )
        .expect("a photo attachment")
    };

    let marked = attachment("!\tsee this");
    assert_eq!(marked.caption.as_deref(), Some("see this"));
    assert!(marked.urgent);

    let plain = attachment("see this");
    assert_eq!(plain.caption.as_deref(), Some("see this"));
    assert!(!plain.urgent);

    let album = merge_album_member(
        Some(ChannelMessage {
            content: "a".into(),
            ..ChannelMessage::default()
        }),
        ChannelMessage {
            content: "b".into(),
            urgent: true,
            ..ChannelMessage::default()
        },
    );
    assert_eq!(album.content, "a\nb");
    assert!(album.urgent);

    let album = merge_album_member(
        Some(ChannelMessage {
            content: "a".into(),
            ..ChannelMessage::default()
        }),
        ChannelMessage {
            content: "b".into(),
            ..ChannelMessage::default()
        },
    );
    assert_eq!(album.content, "a\nb");
    assert!(!album.urgent);
}

async fn parsed_message(ch: &TelegramChannel, update: &serde_json::Value) -> ChannelMessage {
    ch.parse_update_message(update)
        .await
        .expect("parse the update")
        .expect("a message")
}

/// Every forward attribution: each labelled `forward_origin` renders it, and an origin with no
/// label still counts as forwarded — the fact the burst collector's longer window rests on, the
/// prefix only its rendering. An update with both the modern origin and the legacy fields yields
/// one attribution; an unlabelled origin falls back to the legacy fields, which alone still count.
#[tokio::test]
#[expect(clippy::too_many_lines)] // one cohesive table of forward cases
async fn forward_origin_attribution_covers_every_origin_and_the_legacy_fields() {
    let ch = test_channel().await;

    let with_origin = |origin: serde_json::Value| {
        test_update(&[(
            "message",
            serde_json::json!({
                "message_id": 70,
                "text": "Check this out",
                "from": { "id": 1, "username": "alice" },
                "chat": { "id": 999 },
                "forward_origin": origin,
            }),
        )])
    };

    let cases = [
        (
            serde_json::json!({
                "type": "user", "date": 1_700_000_000,
                "sender_user": { "id": 42, "username": "bob" }
            }),
            "[Forwarded from @bob] Check this out",
        ),
        (
            serde_json::json!({
                "type": "user", "date": 1_700_000_000,
                "sender_user": { "id": 77, "first_name": "Charlie" }
            }),
            "[Forwarded from Charlie] Check this out",
        ),
        (
            serde_json::json!({
                "type": "hidden_user", "date": 1_700_000_000,
                "sender_user_name": "Hidden User"
            }),
            "[Forwarded from Hidden User] Check this out",
        ),
        (
            serde_json::json!({
                "type": "chat", "date": 1_700_000_000,
                "sender_chat": { "id": -100_111_i64, "type": "supergroup", "title": "Group Chat" }
            }),
            "[Forwarded from Group Chat] Check this out",
        ),
        (
            serde_json::json!({
                "type": "channel", "date": 1_700_000_000,
                "sender_chat": { "id": -100_222_i64, "type": "channel", "title": "Daily News" }
            }),
            "[Forwarded from channel: Daily News] Check this out",
        ),
    ];
    for (origin, expected) in cases {
        let msg = parsed_message(&ch, &with_origin(origin)).await;
        assert_eq!(msg.content, expected);
        assert!(msg.forwarded, "a modern forward must count as forwarded");
    }

    // The modern origin wins over the legacy fields: one attribution, not two.
    let both = test_update(&[(
        "message",
        serde_json::json!({
            "message_id": 71,
            "text": "Check this out",
            "from": { "id": 1, "username": "alice" },
            "chat": { "id": 999 },
            "forward_origin": {
                "type": "channel",
                "date": 1_700_000_000,
                "sender_chat": {
                    "id": -100_222_i64,
                    "type": "channel",
                    "title": "Daily News"
                }
            },
            "forward_from": { "id": 42, "username": "bob" },
        }),
    )]);
    let msg = parsed_message(&ch, &both).await;
    assert_eq!(msg.content.matches("[Forwarded from").count(), 1);
    assert!(
        msg.content
            .starts_with("[Forwarded from channel: Daily News] ")
    );

    let unknown = with_origin(serde_json::json!({"type": "something_new", "date": 1_700_000_000}));
    let msg = parsed_message(&ch, &unknown).await;
    assert_eq!(msg.content, "Check this out");
    assert!(msg.forwarded);

    let mut unlabelled = with_origin(serde_json::json!({
        "type": "something_new", "date": 1_700_000_000,
        "sender_chat": { "id": -100_222_i64, "type": "channel", "title": "Ignored" }
    }));
    unlabelled["message"]["forward_from_chat"] =
        serde_json::json!({ "id": -100_333_i64, "title": "Fallback News" });
    let msg = parsed_message(&ch, &unlabelled).await;
    assert_eq!(
        msg.content,
        "[Forwarded from channel: Fallback News] Check this out"
    );

    // They still attribute a message — and still count it as forwarded, the flag the burst
    // collector's longer window rests on, which the rendered prefix alone would not pin.
    let legacy = [
        (
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
            "[Forwarded from @bob] Check this out",
        ),
        (
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
            "[Forwarded from channel: Daily News] Breaking news",
        ),
        (
            serde_json::json!({
                "message_id": 52,
                "text": "Secret tip",
                "from": { "id": 1, "username": "alice" },
                "chat": { "id": 999 },
                "forward_sender_name": "Hidden User",
                "forward_date": 1_700_000_000
            }),
            "[Forwarded from Hidden User] Secret tip",
        ),
        (
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
            "[Forwarded from Charlie] Hello there",
        ),
    ];
    for (message, expected) in legacy {
        let msg = parsed_message(&ch, &test_update(&[("message", message)])).await;
        assert_eq!(msg.content, expected);
        assert!(
            msg.forwarded,
            "a legacy forward field must count as forwarded"
        );
    }

    let plain = test_update(&[(
        "message",
        serde_json::json!({
            "message_id": 53,
            "text": "Normal message",
            "from": { "id": 1, "username": "alice" },
            "chat": { "id": 999 }
        }),
    )]);
    let msg = parsed_message(&ch, &plain).await;
    assert_eq!(msg.content, "Normal message");
    assert!(!msg.forwarded);

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
    let attr = TelegramChannel::forward(&message)
        .prefix
        .expect("should detect forward");
    assert_eq!(attr, "[Forwarded from @bob] ");
    let photo_content = "[IMAGE:/tmp/photo.jpg]".to_string();
    let content = format!("{attr}{photo_content}");
    assert_eq!(content, "[Forwarded from @bob] [IMAGE:/tmp/photo.jpg]");
}

/// A refused forwarded attachment keeps its attribution: a refusal hands on a message of its own,
/// and the `[Forwarded from …]` prefix (with the forwarded fact) must ride on it too. A declared
/// size past Telegram's own download cap is refused before any fetch.
#[tokio::test]
async fn a_refused_forwarded_attachment_keeps_its_attribution() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    let (store, _dir) = test_store().await;
    let ch = channel("token-a", &fake, &store);
    let update = forwarded_photo_update(1, 33, "huge", 30 * 1024 * 1024);

    let msg = ch
        .parse_incoming_update(&update)
        .await
        .expect("parse the update")
        .expect("a message");
    assert!(msg.forwarded);
    assert!(
        msg.content
            .starts_with("[Forwarded from channel: Daily News] [File "),
        "the refusal must keep its attribution: {}",
        msg.content
    );
    assert!(msg.content.contains("not received"));

    until(
        "the user is told the file could not be processed",
        || async {
            fake.sent_texts()
                .iter()
                .any(|text| text.contains("Could not process the file"))
        },
    )
    .await;
}

/// Both readers report offsets in bytes, so multi-byte text around and inside a tag must not shift
/// them.
#[test]
fn test_html_tag_scans_are_byte_offsets() {
    let text = "héllo <b x=\"ü>ö\">täil";
    assert_eq!(strip_html_tags(text), "héllo täil");
    assert_eq!(extend_past_open_tag(text, 16), Some(20));
    assert_eq!(extend_past_open_tag(text, text.find('t').unwrap()), None);

    assert_eq!(strip_html_tags("ä <b x=\"ö>ü"), "ä ");
}

// `mirror_gui_message_to_telegram` returns early for each guard condition, and otherwise sends a
// blockquote-format message to the user's bindings; serialized on MIRROR_TEST_LOCK because the
// channel registry and store singletons are global.

use crate::util::UnwrapPoison;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

/// Serialization lock for all mirror tests: they share the global [`CHANNEL_REGISTRY`] and store
/// singletons. A `tokio::sync::Mutex` so holding it across await points blocks no worker thread.
static MIRROR_TEST_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

async fn acquire_mirror_lock() -> tokio::sync::MutexGuard<'static, ()> {
    MIRROR_TEST_LOCK
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

/// A spy Telegram channel in the registry, and the buffer it records sends into. Idempotent.
fn setup_spy_channel() -> &'static Arc<Mutex<Vec<SendMessage>>> {
    static SPY_SENT: OnceLock<Arc<Mutex<Vec<SendMessage>>>> = OnceLock::new();
    SPY_SENT.get_or_init(|| {
        let (spy, sent) = crate::util::test::SpyChannel::new("telegram");
        let registry = crate::CHANNEL_REGISTRY.get_or_init(crate::ChannelRegistry::default);
        registry.register(Arc::new(spy) as Arc<dyn crate::Channel>);
        sent
    })
}

/// A test user with a Telegram binding and reply_target. Idempotent; `ctx` labels panics.
async fn setup_user_with_telegram_binding(user_name: &str, reply_target: &str, ctx: &str) {
    use crate::users::store;
    let store = store();
    store
        .add_user(user_name)
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

/// Preamble shared by all mirror tests: take the lock, initialise stores, set up the spy. Returns
/// the spy buffer and the lock guard (kept alive for the test's duration).
async fn setup_mirror_test_env() -> (
    &'static Arc<Mutex<Vec<SendMessage>>>,
    tokio::sync::MutexGuard<'static, ()>,
) {
    let lock = acquire_mirror_lock().await;
    crate::util::test::init_test_stores().await;
    let sent = setup_spy_channel();
    (sent, lock)
}

/// Shared `ChannelMessage` constructor core; the `gui_msg`/`telegram_msg`/ `voice_msg` wrappers
/// differ only in `channel` and `reply_target`.
fn test_msg(user_name: &str, content: &str, channel: &str, reply_target: &str) -> ChannelMessage {
    ChannelMessage {
        user_name: user_name.to_string(),
        reply_target: reply_target.to_string(),
        content: content.to_string(),
        display_content: None,
        channel: channel.to_string(),
        workspace: "test".to_string(),
        optimistic_id: None,
        callback_query_id: None,
        reply_reference: None,
        chat_id: None,
        message_id: None,
        attachment_dirs: Vec::new(),
        parts: Vec::new(),
        update_date: None,
        forwarded: false,
        update_ids: Vec::new(),
        dictated: false,
        urgent: false,
    }
}

fn gui_msg(user_name: &str, content: &str) -> ChannelMessage {
    test_msg(user_name, content, "gui", "")
}

fn telegram_msg(user_name: &str, content: &str) -> ChannelMessage {
    test_msg(user_name, content, "telegram", "chat:thread")
}

fn voice_msg(user_name: &str, content: &str) -> ChannelMessage {
    test_msg(user_name, content, "voice", "")
}

async fn mirror_single_to_binding(
    user_name: &str,
    reply_target: &str,
    msg: &ChannelMessage,
) -> SendMessage {
    let (sent, _lock) = setup_mirror_test_env().await;
    setup_user_with_telegram_binding(user_name, reply_target, user_name).await;

    super::mirror_gui_message_to_telegram(msg).await;

    let guard = sent.lock().unwrap_poison();
    let our_msgs: Vec<_> = guard
        .iter()
        .filter(|m| m.recipient == reply_target)
        .collect();
    assert_eq!(our_msgs.len(), 1, "expected exactly one message");
    our_msgs[0].clone()
}

/// Per-case setup for `assert_mirror_skips`, applied after the shared mirror env is initialised
/// (`users::store()` panics pre-init).
enum MirrorSkipSetup {
    BoundTo(&'static str, &'static str),
    Unbound(&'static str),
    BoundNoTarget(&'static str),
}

/// Mirror `msg` and assert nothing was sent to the recipient `setup` implies (reply_target when
/// bound, user name otherwise). Each call takes the lock; stores and spy are idempotent.
async fn assert_mirror_skips(setup: MirrorSkipSetup, msg: &ChannelMessage, reason: &str) {
    let (sent, _lock) = setup_mirror_test_env().await;
    let (user, filter_recipient) = match setup {
        MirrorSkipSetup::BoundTo(u, t) => {
            setup_user_with_telegram_binding(u, t, &format!("case {reason}")).await;
            (u, t)
        }
        MirrorSkipSetup::Unbound(u) => {
            let s = crate::users::store();
            s.add_user(u)
                .await
                .unwrap_or_else(|e| panic!("case {reason}: add_user: {e}"));
            (u, u)
        }
        MirrorSkipSetup::BoundNoTarget(u) => {
            let s = crate::users::store();
            s.add_user(u)
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
    assert_mirror_skips(
        MirrorSkipSetup::BoundTo("skip_telegram", "target_non_gui"),
        &telegram_msg("skip_telegram", "hello from telegram"),
        "Telegram-originated messages should not send (telegram is the only \
         source that can originate a loop)",
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
    assert_mirror_skips(
        MirrorSkipSetup::Unbound("no_binding"),
        &gui_msg("no_binding", "hello"),
        "user with no bindings should not send",
    )
    .await;
    assert_mirror_skips(
        MirrorSkipSetup::BoundNoTarget("no_target"),
        &gui_msg("no_target", "hello"),
        "binding without reply_target should not send",
    )
    .await;
    // Full binding: exercises the media-only guard (content emptied by marker stripping).
    assert_mirror_skips(
        MirrorSkipSetup::BoundTo("media_only", "target_media"),
        &gui_msg("media_only", "[IMAGE:/path/to/img.png]"),
        "media-only content should not send",
    )
    .await;
    // A reply reference must NOT rescue an entirely-empty message: the pre-strip trimmed-empty
    // guard still skips it regardless of a reference.
    let mut empty_ref = gui_msg("skip_ew", "");
    empty_ref.reply_reference = Some(crate::channels::ReplyReference {
        author: "alice".to_string(),
        snippet: "hi".to_string(),
    });
    assert_mirror_skips(
        MirrorSkipSetup::BoundTo("skip_ew", "target_empty_ws"),
        &empty_ref,
        "empty content with a reply reference should not send",
    )
    .await;
}

#[tokio::test]
async fn sends_blockquote_to_single_binding() {
    let mirrored = mirror_single_to_binding(
        "single_user",
        "unique_single",
        &gui_msg("single_user", "Hello, world!"),
    )
    .await;
    assert_eq!(
        mirrored.content,
        "<blockquote>\nHello, world!\n</blockquote>"
    );
    assert!(mirrored.reply_markup.is_none());
}

#[tokio::test]
async fn mirrors_voice_transcript_to_telegram() {
    // Voice-originated messages (mic-button recording / wake word) must mirror exactly like GUI
    // text: same bindings, same blockquote format, no audio payload.
    let mirrored = mirror_single_to_binding(
        "voice_user",
        "unique_voice",
        &voice_msg("voice_user", "Record this voice note"),
    )
    .await;
    assert_eq!(
        mirrored.content,
        "<blockquote>\nRecord this voice note\n</blockquote>"
    );
    assert!(mirrored.reply_markup.is_none());
}

#[tokio::test]
async fn sends_to_multiple_telegram_bindings() {
    let (sent, _lock) = setup_mirror_test_env().await;
    let store = crate::users::store();
    store.add_user("multi_user").await.expect("add_user");
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
    for m in &our_msgs {
        assert_eq!(m.content, "<blockquote>\nHi both!\n</blockquote>");
    }
    let recipients: Vec<&str> = our_msgs.iter().map(|m| m.recipient.as_str()).collect();
    assert!(recipients.contains(&"unique_multi_a"));
    assert!(recipients.contains(&"unique_multi_b"));
}

async fn assert_mirror_strips_markers(
    user_name: &str,
    reply_target: &str,
    content: &str,
    expected_quote: &str,
) {
    let mirrored =
        mirror_single_to_binding(user_name, reply_target, &gui_msg(user_name, content)).await;
    assert_eq!(mirrored.content, expected_quote);
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
    // Regression: the mirror path must use a case-insensitive regex to strip lowercase markers like
    // [image:...] and [audio:...].
    assert_mirror_strips_markers(
        "strip_markers",
        "unique_lowercase",
        "See [image:/tmp/photo.png] and hear [audio:/tmp/sound.mp3]",
        "<blockquote>\nSee  and hear\n</blockquote>",
    )
    .await;
    // A marker-shaped word of an unknown kind is not a marker: only the four known kinds go.
    assert_mirror_strips_markers(
        "strip_markers",
        "unique_unmappable",
        "See [DOCX:/tmp/report.docx] here",
        "<blockquote>\nSee [DOCX:/tmp/report.docx] here\n</blockquote>",
    )
    .await;
}

#[tokio::test]
async fn preserves_markdown_formatting_in_blockquote() {
    let mirrored = mirror_single_to_binding(
        "md_user",
        "unique_md",
        &gui_msg("md_user", "**bold** and `code` and *italic*"),
    )
    .await;
    assert_eq!(
        mirrored.content,
        "<blockquote>\n**bold** and `code` and *italic*\n</blockquote>"
    );
}

#[tokio::test]
async fn mirrors_reply_reference_header() {
    let mut msg = gui_msg("ref_user", "hello");
    msg.reply_reference = Some(crate::channels::ReplyReference {
        author: "alice".to_string(),
        snippet: "hi there".to_string(),
    });
    let mirrored = mirror_single_to_binding("ref_user", "unique_ref", &msg).await;
    assert_eq!(
        mirrored.content,
        "<blockquote>\n↩ alice: hi there\nhello\n</blockquote>"
    );
}

#[tokio::test]
#[serial_test::serial(gui_admin_workspace)] // writes the shared seeded admin row
async fn user_command_entries_reflect_admin_state() {
    // Serialized with the sibling mirror tests: it initialises the shared user store.
    let _lock = acquire_mirror_lock().await;
    crate::users::test_util::init_test_store().await;
    let admin = crate::users::ADMIN_USER_NAME;

    // The admin: no role-switch entry (the Assistant is every account's only role), the two
    // workspace-independent admin commands and the model commands. The SourceTree test env's
    // availability cache seeds `available = true`, so `/update` is present.
    let admin_entries = user_command_entries(admin).await;
    let cmds: Vec<&str> = admin_entries.iter().map(|(c, _)| c.as_str()).collect();
    assert!(cmds.contains(&"board"));
    assert!(cmds.contains(&"archive"));
    assert!(cmds.contains(&"update"));
    for cmd in [
        "workspace",
        "pause",
        "unpause",
        "maintenance",
        "maintenance_on",
        "maintenance_off",
        "agents",
        "manager",
        "assistant",
        "engineer",
    ] {
        assert!(!cmds.contains(&cmd), "{cmd} must be gone from the menu");
    }
    assert!(cmds.contains(&"image_models"));
    assert!(cmds.contains(&"video_models"));
    assert_eq!(cmds.last(), Some(&"clear"));
    let pos = |cmd: &str| cmds.iter().position(|c| *c == cmd).unwrap();
    assert!(pos("board") < pos("archive"));
    assert!(pos("archive") < pos("update"));
    assert!(pos("update") < pos("image_models"));
    assert!(pos("image_models") < pos("video_models"));
    assert!(pos("video_models") < pos("clear"));

    // Hidden branch: no available update → `/update` is absent even for the admin (the cache is a
    // process-local single source of truth). The RAII guard restores the SourceTree default.
    {
        let _guard = crate::self_update::set_update_cache_for_test(false, false);
        let hidden = user_command_entries(admin).await;
        let hidden_cmds: Vec<&str> = hidden.iter().map(|(c, _)| c.as_str()).collect();
        assert!(!hidden_cmds.contains(&"update"));
    }

    // bob: a guest — no admin commands and, the Assistant being the only role, no role-switch entry
    // either. Model commands remain available.
    let bob = user_command_entries("bob").await;
    let cmds: Vec<&str> = bob.iter().map(|(c, _)| c.as_str()).collect();
    for cmd in [
        "agents",
        "board",
        "archive",
        "update",
        "workspace",
        "pause",
        "unpause",
        "maintenance",
        "maintenance_on",
        "maintenance_off",
        "assistant",
        "manager",
    ] {
        assert!(!cmds.contains(&cmd), "{cmd} must not reach a guest's menu");
    }
    assert!(cmds.contains(&"image_models"));
    assert!(cmds.contains(&"video_models"));
    assert_eq!(cmds[0], "image_models");
}

// The intake's guarantee — an event the platform handed over is never lost and is taken up exactly
// once — is checked against a fake Bot API: a listener on a local endpoint
// (`TelegramChannel::for_test`), scripted `getUpdates` batches, and a store of this check's own.
//
// Every check that starts a listener is serialized on the process-global drain flag: the listener
// races it in its poll loop, so a check running while another test winds the daemon down would see
// its listener stop.

/// How long a check waits for state it can observe before failing. Generous on purpose: everything
/// waited for is durable — a record row, a logged request, a held message — so a longer bound only
/// delays failure reporting, while a short one fails a stalled machine.
const OBSERVABLE_TIMEOUT: Duration = Duration::from_secs(5);

/// How long a check waits for a poll that only arrives after the intake's own pause on an
/// unrecordable batch. Deliberately loose: that pause is seconds long by design and its length is
/// not what the check asserts, so the bound only has to be past the pause, not equal to it.
const BACKOFF_OBSERVABLE_TIMEOUT: Duration = Duration::from_secs(15);

/// How long the injected fetch of a forwarded attachment takes: past the group's own ceiling, so a
/// pair reaching the assistant as one turn can only have been held across the fetch.
const FETCH_DELAY: Duration = Duration::from_secs(2);

const AT_MOST_ONE_TURN: Duration = Duration::from_secs(20);

struct FakeTelegram {
    base: String,
    state: Arc<FakeState>,
    stop: Arc<AtomicBool>,
    accept: Option<std::thread::JoinHandle<()>>,
}

struct FakeState {
    accounts: HashMap<String, i64>,
    batches: Mutex<HashMap<String, VecDeque<Vec<serde_json::Value>>>>,
    requests: Mutex<Vec<ServedRequest>>,
    sent_texts: Mutex<Vec<String>>,
    files: Mutex<HashMap<String, FakeFile>>,
}

struct ServedRequest {
    method: String,
    offset: Option<i64>,
    at: std::time::Instant,
    body: Vec<u8>,
}

/// A file the fake serves over a transfer of its own: the bytes, and how long the answer takes. A
/// real fetch of a few megabytes takes seconds, which a slow-fetch check must reproduce.
struct FakeFile {
    bytes: Vec<u8>,
    delay: Duration,
}

struct FakeRequest {
    path: String,
    body: Vec<u8>,
}

struct FakeHead {
    path: String,
    length: usize,
    buffer: Vec<u8>,
    head_end: usize,
}

impl FakeTelegram {
    fn start(accounts: &[(&str, i64)]) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind the fake Bot API");
        listener
            .set_nonblocking(true)
            .expect("the fake Bot API accepts without blocking");
        let port = listener.local_addr().expect("fake Bot API address").port();
        let state = Arc::new(FakeState {
            accounts: accounts
                .iter()
                .map(|(token, id)| ((*token).to_string(), *id))
                .collect(),
            batches: Mutex::new(HashMap::new()),
            requests: Mutex::new(Vec::new()),
            sent_texts: Mutex::new(Vec::new()),
            files: Mutex::new(HashMap::new()),
        });
        let stop = Arc::new(AtomicBool::new(false));
        let accept = std::thread::spawn({
            let state = Arc::clone(&state);
            let stop = Arc::clone(&stop);
            move || {
                while !stop.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            // The accepted socket can inherit the listener's non-blocking mode (it
                            // does on macOS), and a polling handler would then read nothing (or a
                            // half-arrived request) and close the connection before answering. A
                            // handler blocks, so make the stream block.
                            let _ = stream.set_nonblocking(false);
                            let state = Arc::clone(&state);
                            std::thread::spawn(move || serve(stream, &state));
                        }
                        // Polling rather than blocking, so the loop can see `stop`: a blocking
                        // accept would outlive the check that drops the fake.
                        Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(2));
                        }
                        // A transient accept error (a client gone before it was accepted,
                        // `ECONNABORTED`, ordinary on macOS) must not kill the fake: an accept loop
                        // that returns here stops serving every later request.
                        Err(_) => std::thread::sleep(Duration::from_millis(2)),
                    }
                }
            }
        });
        Self {
            base: format!("http://127.0.0.1:{port}"),
            state,
            stop,
            accept: Some(accept),
        }
    }

    fn push_batch(&self, token: &str, updates: Vec<serde_json::Value>) {
        self.state
            .batches
            .lock()
            .expect("fake batches lock")
            .entry(token.to_string())
            .or_default()
            .push_back(updates);
    }

    /// Register a file `file_id` can be fetched as: `getFile` answers the path [`file_path_of`]
    /// builds, and the transfer serves `bytes` after `delay`.
    fn push_file(&self, file_id: &str, bytes: &[u8], delay: Duration) {
        self.state.files.lock().expect("fake files lock").insert(
            file_path_of(file_id),
            FakeFile {
                bytes: bytes.to_vec(),
                delay,
            },
        );
    }

    fn poll_offsets(&self) -> Vec<i64> {
        self.state
            .requests
            .lock()
            .expect("fake request log")
            .iter()
            .filter(|request| request.method == "getUpdates")
            .filter_map(|request| request.offset)
            .collect()
    }

    fn count(&self, method: &str) -> usize {
        self.state
            .requests
            .lock()
            .expect("fake request log")
            .iter()
            .filter(|request| request.method == method)
            .count()
    }

    fn bodies(&self, method: &str) -> Vec<Vec<u8>> {
        self.state
            .requests
            .lock()
            .expect("fake request log")
            .iter()
            .filter(|request| request.method == method)
            .map(|request| request.body.clone())
            .collect()
    }

    fn sent_texts(&self) -> Vec<String> {
        self.state
            .sent_texts
            .lock()
            .expect("fake sendMessage log")
            .clone()
    }
}

impl Drop for FakeTelegram {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(accept) = self.accept.take() {
            let _ = accept.join();
        }
    }
}

fn bot_path(path: &str) -> Option<(&str, &str)> {
    let (token, method) = path.strip_prefix("/bot")?.split_once('/')?;
    (!token.is_empty() && !method.is_empty()).then_some((token, method))
}

fn file_path_of(file_id: &str) -> String {
    format!("photos/{file_id}.jpg")
}

fn not_found() -> serde_json::Value {
    serde_json::json!({ "ok": false, "error_code": 404, "description": "Not Found" })
}

fn serve(mut stream: TcpStream, state: &FakeState) {
    let Some(head) = read_head(&mut stream) else {
        return;
    };
    let Some(request) = read_body(&mut stream, head) else {
        return;
    };
    // A GET carries its parameters in the request line, so the target splits before the method.
    let target = request.path.split('?').next().unwrap_or_default();
    let method = target.rsplit('/').next().unwrap_or_default().to_string();
    // The `offset` a poll asked for, `None` for methods that carry none; read here because the log
    // line below is where it is recorded.
    let offset = if method == "getUpdates" {
        serde_json::from_slice::<serde_json::Value>(&request.body)
            .ok()
            .and_then(|body| body["offset"].as_i64())
    } else {
        None
    };
    // Recorded as the request arrives, before its answer: the log is the durable evidence.
    state
        .requests
        .lock()
        .expect("fake request log")
        .push(ServedRequest {
            method: method.clone(),
            offset,
            at: std::time::Instant::now(),
            body: request.body.clone(),
        });

    if target == "/redirect"
        && let Some(to) = request.path.split("?to=").nth(1)
    {
        let head = format!(
            "HTTP/1.1 302 Found\r\nLocation: {to}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        let _ = stream.write_all(head.as_bytes());
        return;
    }

    // A file transfer is not a Bot API method: `{base}/file/bot<token>/<path>`, body is bytes.
    if let Some(rest) = target.strip_prefix("/file/bot")
        && let Some((token, file_path)) = rest.split_once('/')
    {
        serve_file(&mut stream, state, token, file_path);
        return;
    }
    let Some((token, _)) = bot_path(target) else {
        respond(&mut stream, &not_found());
        return;
    };
    let Some(account) = state.accounts.get(token).copied() else {
        // A listener is only ever started on a token the fake was given, so nothing here models the
        // platform's own per-token authorization.
        respond(&mut stream, &not_found());
        return;
    };

    match method.as_str() {
        "getMe" => respond(
            &mut stream,
            &serde_json::json!({ "ok": true, "result": { "id": account } }),
        ),
        "getUpdates" => serve_get_updates(&mut stream, state, token, &request),
        "sendMessage" => serve_send_message(&mut stream, state, &request),
        "sendDocument" | "sendPhoto" | "sendAudio" => respond(
            &mut stream,
            &serde_json::json!({
                "ok": true,
                "result": { "message_id": 1, "chat": { "id": 5 }, "date": 0 },
            }),
        ),
        "getFile" => serve_get_file(&mut stream, state, &request),
        "setMessageReaction" | "answerCallbackQuery" | "setMyCommands" | "sendChatAction" => {
            respond(
                &mut stream,
                &serde_json::json!({ "ok": true, "result": true }),
            );
        }
        _ => respond(&mut stream, &not_found()),
    }
}

fn serve_get_file(stream: &mut TcpStream, state: &FakeState, request: &FakeRequest) {
    let Some(target) = request.path.split('?').nth(1) else {
        respond(stream, &not_found());
        return;
    };
    let Some((_, file_id)) = target.split_once('=') else {
        respond(stream, &not_found());
        return;
    };
    let path = file_path_of(file_id);
    let Some(size) = state
        .files
        .lock()
        .expect("fake files lock")
        .get(&path)
        .map(|file| file.bytes.len())
    else {
        respond(stream, &not_found());
        return;
    };
    respond(
        stream,
        &serde_json::json!({
            "ok": true,
            "result": { "file_id": file_id, "file_path": path, "file_size": size },
        }),
    );
}

/// The file transfer's answer: the registered bytes after the transfer's own delay. Answered and
/// closed like every other connection, so a slow transfer holds only its handler thread.
fn serve_file(stream: &mut TcpStream, state: &FakeState, token: &str, file_path: &str) {
    if !state.accounts.contains_key(token) {
        respond(stream, &not_found());
        return;
    }
    let registered = state
        .files
        .lock()
        .expect("fake files lock")
        .get(file_path)
        .map(|file| (file.bytes.clone(), file.delay));
    let Some((bytes, delay)) = registered else {
        respond(stream, &not_found());
        return;
    };
    std::thread::sleep(delay);
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        bytes.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&bytes);
}

/// The `getUpdates` answer: the oldest queued batch, an immediate empty result for the handshake's
/// zero-timeout probe, or — as the real long poll does — the batch queued while the client waits;
/// it ends unanswered only when the client gives up.
fn serve_get_updates(
    stream: &mut TcpStream,
    state: &FakeState,
    token: &str,
    request: &FakeRequest,
) {
    if let Some(batch) = pop_batch(state, token) {
        return respond(stream, &serde_json::json!({ "ok": true, "result": batch }));
    }
    let timeout = serde_json::from_slice::<serde_json::Value>(&request.body)
        .ok()
        .and_then(|body| body["timeout"].as_u64())
        .unwrap_or(0);
    if timeout == 0 {
        return respond(stream, &serde_json::json!({ "ok": true, "result": [] }));
    }
    // A long poll with nothing to offer yet: the real API answers the moment an update arrives, so
    // a batch queued while the client waits is served too — which is what lets a check queue a
    // later batch after observing an earlier one. The socket waits non-blocking so the queue can be
    // re-read between reads; `Ok(0)` below is the client giving up, the one case the poll ends
    // unanswered.
    let _ = stream.set_nonblocking(true);
    let mut sink = [0u8; 1024];
    loop {
        if let Some(batch) = pop_batch(state, token) {
            let _ = stream.set_nonblocking(false);
            return respond(stream, &serde_json::json!({ "ok": true, "result": batch }));
        }
        match stream.read(&mut sink) {
            Ok(0) => return,
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(_) => return,
        }
    }
}

fn pop_batch(state: &FakeState, token: &str) -> Option<Vec<serde_json::Value>> {
    state
        .batches
        .lock()
        .expect("fake batches lock")
        .get_mut(token)
        .and_then(VecDeque::pop_front)
}

fn serve_send_message(stream: &mut TcpStream, state: &FakeState, request: &FakeRequest) {
    if let Some(text) = serde_json::from_slice::<serde_json::Value>(&request.body)
        .ok()
        .and_then(|body| body["text"].as_str().map(str::to_string))
    {
        state
            .sent_texts
            .lock()
            .expect("fake sendMessage log")
            .push(text);
    }
    respond(
        stream,
        &serde_json::json!({
            "ok": true,
            "result": { "message_id": 1, "chat": { "id": 5 }, "date": 0 },
        }),
    );
}

fn read_head(stream: &mut TcpStream) -> Option<FakeHead> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 1024];
    let head_end = loop {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(at) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break at + 4;
        }
    };
    let (path, length) = {
        let head = String::from_utf8_lossy(&buffer[..head_end]);
        let path = head
            .lines()
            .next()?
            .split_whitespace()
            .nth(1)
            .unwrap_or_default()
            .to_string();
        let length = head
            .lines()
            .skip(1)
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())?
            })
            .unwrap_or(0);
        (path, length)
    };
    Some(FakeHead {
        path,
        length,
        buffer,
        head_end,
    })
}

fn read_body(stream: &mut TcpStream, head: FakeHead) -> Option<FakeRequest> {
    let FakeHead {
        path,
        length,
        mut buffer,
        head_end,
    } = head;
    let mut chunk = [0u8; 1024];
    while buffer.len() < head_end + length {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    let body = buffer[head_end..(head_end + length).min(buffer.len())].to_vec();
    Some(FakeRequest { path, body })
}

fn respond(stream: &mut TcpStream, body: &serde_json::Value) {
    let text = body.to_string();
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        text.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(text.as_bytes());
}

/// Poll `probe` until it holds, or fail naming `what`: a fixed sleep long enough for the slowest
/// machine is a slow check, and a short one is a flaky check.
async fn until<F, Fut>(what: &str, probe: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    until_within(OBSERVABLE_TIMEOUT, what, probe).await;
}

/// The same wait with the failure bound chosen by the caller: state reached only after the intake's
/// own pause is a pause away, so the cheap bound every other wait uses would fail on it — and only
/// a failing run pays for the loose one.
async fn until_within<F, Fut>(bound: Duration, what: &str, mut probe: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + bound;
    loop {
        if probe().await {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting until {what}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The next message the pipeline receives, or a panic naming `what`: a released burst crosses the
/// collector's quiet window, so its message is a second away, never immediate.
async fn next_message(
    rx: &mut tokio::sync::mpsc::Receiver<ChannelMessage>,
    what: &str,
) -> ChannelMessage {
    next_message_within(OBSERVABLE_TIMEOUT, rx, what).await
}

/// The next message the pipeline receives within `bound`, or a panic naming `what`: a released
/// burst crosses the collector's windows first, so its message is a second (or a fetch) away.
async fn next_message_within(
    bound: Duration,
    rx: &mut tokio::sync::mpsc::Receiver<ChannelMessage>,
    what: &str,
) -> ChannelMessage {
    tokio::time::timeout(bound, rx.recv())
        .await
        .unwrap_or_else(|_| panic!("nothing reached the pipeline: {what}"))
        .unwrap_or_else(|| panic!("the pipeline closed before delivering: {what}"))
}

fn test_text_update(update_id: i64, text: &str) -> serde_json::Value {
    test_update(&[
        ("update_id", serde_json::json!(update_id)),
        (
            "message",
            serde_json::json!({
                "message_id": 33,
                "text": text,
                "from": { "id": 555, "username": "alice" },
                "chat": { "id": -100_200_300 },
            }),
        ),
    ])
}

fn test_album_member(update_id: i64, group: &str, text: &str) -> serde_json::Value {
    test_update(&[
        ("update_id", serde_json::json!(update_id)),
        (
            "message",
            serde_json::json!({
                "message_id": 100 + update_id,
                "text": text,
                "media_group_id": group,
                "from": { "id": 555, "username": "alice" },
                "chat": { "id": -100_200_300 },
            }),
        ),
    ])
}

/// A forwarded photo from alice as Telegram sends one: the attachment is what a slow fetch is
/// injected for.
fn forwarded_photo_update(
    update_id: i64,
    message_id: i64,
    file_id: &str,
    file_size: u64,
) -> serde_json::Value {
    test_update(&[
        ("update_id", serde_json::json!(update_id)),
        (
            "message",
            serde_json::json!({
                "message_id": message_id,
                "date": 1_700_000_000,
                "from": { "id": 555, "username": "alice" },
                "chat": { "id": -100_200_300 },
                "photo": [{
                    "file_id": file_id,
                    "file_unique_id": "u1",
                    "width": 320,
                    "height": 240,
                    "file_size": file_size,
                }],
                "forward_origin": {
                    "type": "channel",
                    "date": 1_699_999_000,
                    "sender_chat": {
                        "id": -100_1234_5678_i64,
                        "type": "channel",
                        "title": "Daily News",
                    },
                },
            }),
        ),
    ])
}

/// A store of this check's own plus the `TempDir` holding its rows, one per check. The store comes
/// back because a restarting check builds a second record over the same durable rows.
async fn test_store() -> (TelegramIntakeStore, tempfile::TempDir) {
    // Alice must be bound to Telegram for a parsed update to authorize her.
    crate::users::test_util::init_test_store().await;
    crate::open_test_store!(TelegramIntakeStore, "telegram_intake")
}

/// A run's own handle onto `store` — a listener's, or a check's own probe. `claim_label` names the
/// run it stands for: a restart, or a row a previous run left, must claim under another id.
fn run(store: &TelegramIntakeStore, claim_label: &str, account: &str) -> Intake {
    Intake::new(store.clone(), Arc::from(claim_label), account)
}

/// A listener pointed at `fake`, recording and claiming through `store`. Its claim is its own, as
/// for a product listener, so a second listener over one store is a restart's other run.
fn channel(token: &str, fake: &FakeTelegram, store: &TelegramIntakeStore) -> TelegramChannel {
    TelegramChannel::for_test(token.into(), fake.base.clone(), store.clone())
}

async fn still_held(store: &TelegramIntakeStore, account: &str) -> Vec<i64> {
    let rows = store
        .conn
        .query(
            "SELECT update_id FROM telegram_intake \
             WHERE account_id = ?1 AND state = ?2",
            crate::db::params![account, IN_PROGRESS],
        )
        .await
        .expect("read the record");
    rows.into_iter()
        .map(|row| row.get::<i64>(0).expect("update_id"))
        .collect()
}

struct BurstRow {
    user_name: String,
    parts: i64,
    released_by: String,
    part_times: Vec<serde_json::Value>,
}

async fn burst_rows(store: &TelegramIntakeStore) -> Vec<BurstRow> {
    let rows = store
        .conn
        .query(
            "SELECT user_name, parts, released_by, part_times FROM telegram_bursts ORDER BY id",
            (),
        )
        .await
        .expect("read the burst record");
    rows.into_iter()
        .map(|row| BurstRow {
            user_name: row.get::<String>(0).expect("user_name"),
            parts: row.get::<i64>(1).expect("parts"),
            released_by: row.get::<String>(2).expect("released_by"),
            part_times: serde_json::from_str(&row.get::<String>(3).expect("part_times"))
                .expect("part_times is JSON"),
        })
        .collect()
}

/// An event recorded but not handed on when the process died is taken up by the next run; otherwise
/// everything that arrived in the intake's last moments would be silently dropped. The restart runs
/// the token rotated for the same account: the record is keyed on the bot account the platform
/// names, not on the token's text, so what the dead run held is its to take.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn restart_takes_up_the_event_a_previous_listener_held() {
    // Two tokens the platform maps to one account: a rotation, not a new bot.
    let fake = FakeTelegram::start(&[("token-a", 700), ("token-b", 700)]);
    fake.push_batch("token-a", vec![test_update(&[])]);
    let (store, _dir) = test_store().await;

    let first = channel("token-a", &fake, &store);

    // The process dies mid-flight: the sink is gone, so the hand-over can never succeed and the
    // event stays held — what a run that died before handing it on leaves behind. Dropping the
    // receiver up front also keeps the wait below off the collector's one-second window.
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    let listener = tokio::spawn(async move {
        let _ = first.listen(tx).await;
    });
    drop(rx);

    until("the record holds the event", || async {
        !still_held(&store, "700").await.is_empty()
    })
    .await;
    listener.abort();
    let _ = listener.await;
    assert!(
        !still_held(&store, "700").await.is_empty(),
        "the event must survive the process that never handed it on"
    );

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let restarted = channel("token-b", &fake, &store);
    let listener = tokio::spawn(async move {
        let _ = restarted.listen(tx).await;
    });

    let msg = next_message(&mut rx, "the event the previous run held").await;
    assert_eq!(msg.content, "hello");
    assert_eq!(msg.channel, "telegram");
    assert!(
        msg.update_ids.is_empty(),
        "the hand-over must settle the entries it carries"
    );

    until("the hand-over settles the recovered event", || async {
        still_held(&store, "700").await.is_empty()
    })
    .await;
    listener.abort();
}

/// An album is one incoming event: its members are recorded, handed over as one message and settled
/// together, never taken up as unrelated messages.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn an_album_is_taken_up_and_settled_as_one_event() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    fake.push_batch(
        "token-a",
        vec![
            test_album_member(1, "group-1", "one"),
            test_album_member(2, "group-1", "two"),
        ],
    );
    let (store, _dir) = test_store().await;
    let telegram = channel("token-a", &fake, &store);

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let listener = tokio::spawn(async move {
        let _ = telegram.listen(tx).await;
    });

    let msg = next_message(&mut rx, "the album, as one message").await;
    assert_eq!(
        msg.content, "one\ntwo",
        "an album's members must reach the pipeline as one message"
    );
    // Both members are settled by that one hand-over: the record keeps no part behind.
    until("the album is settled whole", || async {
        still_held(&store, "700").await.is_empty()
    })
    .await;

    assert!(
        tokio::time::timeout(Duration::from_millis(300), rx.recv())
            .await
            .is_err(),
        "an album's member was handled as an unrelated message"
    );
    listener.abort();
}

#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn a_recovered_album_is_taken_up_as_one_event() {
    // Nothing is queued: the handshake's probe gets an empty result and the long poll then hangs,
    // which keeps the listener quiet while recovery runs.
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    let (store, _dir) = test_store().await;
    run(&store, "previous-run", "700")
        .hold(&[
            &test_album_member(1, "group-1", "one"),
            &test_album_member(2, "group-1", "two"),
        ])
        .await
        .expect("hold the album");

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let telegram = channel("token-a", &fake, &store);
    let listener = tokio::spawn(async move {
        let _ = telegram.listen(tx).await;
    });

    let msg = next_message(&mut rx, "the recovered album, as one message").await;
    assert_eq!(
        msg.content, "one\ntwo",
        "a recovered album's members must reach the pipeline as one message"
    );
    until("the recovered album is settled whole", || async {
        still_held(&store, "700").await.is_empty()
    })
    .await;
    assert!(
        tokio::time::timeout(Duration::from_millis(300), rx.recv())
            .await
            .is_err(),
        "a recovered album's member was handled as an unrelated message"
    );
    listener.abort();
}

/// A token change to another account takes up that account's own backlog, and leaves the previous
/// account's held events alone: an `update_id` is only an identity inside its account, so the new
/// listener must not read the old account's event as a repeat of its own.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn a_token_change_to_another_account_takes_its_own_backlog() {
    let fake = FakeTelegram::start(&[("token-a", 700), ("token-b", 800)]);
    let (store, _dir) = test_store().await;

    run(&store, "previous-run", "700")
        .hold(&[&test_text_update(1, "a-only")])
        .await
        .expect("hold the previous account's event");
    fake.push_batch("token-b", vec![test_text_update(1, "b-only")]);

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let telegram = channel("token-b", &fake, &store);
    let listener = tokio::spawn(async move {
        let _ = telegram.listen(tx).await;
    });

    let msg = next_message(&mut rx, "the new account's own event").await;
    assert_eq!(msg.content, "b-only");

    assert!(
        tokio::time::timeout(Duration::from_millis(300), rx.recv())
            .await
            .is_err(),
        "the previous account's event was taken up under the new account"
    );
    assert!(
        !still_held(&store, "700").await.is_empty(),
        "the previous account's held event must be left alone"
    );
    listener.abort();
}

#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn a_token_change_hands_over_what_the_replaced_listener_held_once() {
    // Two tokens the platform maps to one account: a rotation, not a new bot.
    let fake = FakeTelegram::start(&[("token-a", 700), ("token-b", 700)]);
    fake.push_batch("token-a", vec![test_update(&[])]);
    let (store, _dir) = test_store().await;

    // The listener being replaced cannot hand its burst on — its sink is gone — which is what
    // leaves the event held for the replacement to take up.
    let old: std::sync::Arc<dyn Channel> = std::sync::Arc::new(channel("token-a", &fake, &store));
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    let listener = tokio::spawn({
        let old = std::sync::Arc::clone(&old);
        async move {
            let _ = old.listen(tx).await;
        }
    });
    drop(rx);
    until("the record holds the event", || async {
        !still_held(&store, "700").await.is_empty()
    })
    .await;

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let new: std::sync::Arc<dyn Channel> = std::sync::Arc::new(channel("token-b", &fake, &store));
    let registry = crate::ChannelRegistry::default();
    registry.register(std::sync::Arc::clone(&old));
    let frames = registry.replaced("telegram");
    registry.replace(std::sync::Arc::clone(&new));
    let hand_over = tokio::spawn(hand_over_listener(frames, new, tx));

    let msg = next_message(&mut rx, "the event the replaced listener held").await;
    assert_eq!(msg.content, "hello");
    until("the hand-over settles the recovered event", || async {
        still_held(&store, "700").await.is_empty()
    })
    .await;
    assert!(
        tokio::time::timeout(crate::channels::telegram_group::QUIET_WINDOW * 2, rx.recv())
            .await
            .is_err(),
        "a recovered event must be handed on exactly once"
    );
    listener.abort();
    hand_over.abort();
}

/// A token cleared and then set again must not cost the channel live by then: sends and a later
/// restart reach the listener through the registry, so a teardown removing its channel by name
/// would stop every Telegram send after a quick clear-then-set. Both the teardown and the save's
/// hand-over wait on the one frame the cleared listener runs.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn a_clear_then_set_releases_both_waiters_and_keeps_its_channel() {
    let fake = FakeTelegram::start(&[("token-a", 700), ("token-b", 700)]);
    fake.push_batch("token-a", vec![test_update(&[])]);
    let (store, _dir) = test_store().await;

    let registry = crate::ChannelRegistry::default();

    // The cleared listener holds a burst it cannot hand on: no pipeline room, so its frame is open.
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    tx.send(ChannelMessage::default())
        .await
        .expect("the pipeline accepts the message that fills it");
    let old: std::sync::Arc<dyn Channel> = std::sync::Arc::new(channel("token-a", &fake, &store));
    registry.register(std::sync::Arc::clone(&old));
    let listener = tokio::spawn({
        let old = std::sync::Arc::clone(&old);
        async move {
            let _ = old.listen(tx).await;
        }
    });
    until("the intake takes the event on", || async {
        !still_held(&store, "700").await.is_empty()
    })
    .await;

    let frames = registry.replaced("telegram");
    let teardown = retire_listener(&registry, Some(std::sync::Arc::clone(&old)));
    tokio::pin!(teardown);
    assert!(
        tokio::time::timeout(Duration::from_millis(200), &mut teardown)
            .await
            .is_err(),
        "the clear's teardown must wait for the replaced listener's frame"
    );

    let new: std::sync::Arc<dyn Channel> = std::sync::Arc::new(channel("token-b", &fake, &store));
    registry.replace(std::sync::Arc::clone(&new));
    let mut hand_over = tokio::spawn({
        let frames = std::sync::Arc::clone(&frames);
        let new = std::sync::Arc::clone(&new);
        async move { stop_listeners(&frames, Some(&new)).await }
    });
    assert!(
        tokio::time::timeout(Duration::from_millis(200), &mut hand_over)
            .await
            .is_err(),
        "the hand-over's wait must park on the replaced listener's frame"
    );

    rx.recv()
        .await
        .expect("the message that filled the pipeline");
    let msg = next_message(&mut rx, "the burst the cleared listener held").await;
    assert_eq!(msg.content, "hello");
    tokio::time::timeout(OBSERVABLE_TIMEOUT, teardown)
        .await
        .expect("the clear's teardown must finish once the frame has ended");
    tokio::time::timeout(OBSERVABLE_TIMEOUT, hand_over)
        .await
        .expect("the hand-over waiting on the frame must finish with it")
        .expect("the hand-over task must not panic");

    assert!(
        registry
            .get("telegram")
            .is_some_and(|live| std::sync::Arc::ptr_eq(&live, &new)),
        "the channel that is live after a quick clear-then-set must stay registered"
    );
    listener.abort();
}

/// A hand-over never stops the channel it is about to start, even when that channel is in the
/// frames list under the name — the state a superseded save leaves, its replace having moved past
/// the channel the earlier save built whose hand-over has not run yet. A stop would fire its cancel
/// token, so the listener started right after would return at once.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn a_hand_over_never_stops_the_channel_it_is_about_to_start() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    fake.push_batch("token-a", vec![test_text_update(1, "own")]);
    let (store, _dir) = test_store().await;

    let registry = crate::ChannelRegistry::default();
    let frames = registry.replaced("telegram");
    let new: std::sync::Arc<dyn Channel> = std::sync::Arc::new(channel("token-a", &fake, &store));
    // Two saves in a row, each replacing what the one before built: the second replaces `new`.
    registry.replace(std::sync::Arc::clone(&new));
    registry.replace(std::sync::Arc::new(channel("token-a", &fake, &store)));

    stop_listeners(&frames, Some(&new)).await;

    assert!(
        frames
            .snapshot()
            .iter()
            .any(|held| std::sync::Arc::ptr_eq(held, &new)),
        "a hand-over retired the channel it is about to start"
    );
    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let listener = tokio::spawn({
        let new = std::sync::Arc::clone(&new);
        async move {
            let _ = new.listen(tx).await;
        }
    });
    let msg = next_message(&mut rx, "the batch of the channel a hand-over left alone").await;
    assert_eq!(msg.content, "own");
    listener.abort();
}

/// The other half of a clear: when nothing took the cleared channel's place while its teardown
/// waited, it must not stay registered — sends would reach a listener no longer running.
#[tokio::test(flavor = "multi_thread")]
async fn a_cleared_token_takes_its_channel_out_of_the_registry() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    let (store, _dir) = test_store().await;

    let registry = crate::ChannelRegistry::default();
    let old: std::sync::Arc<dyn Channel> = std::sync::Arc::new(channel("token-a", &fake, &store));
    registry.register(std::sync::Arc::clone(&old));

    retire_listener(&registry, Some(old)).await;
    assert!(
        registry.get("telegram").is_none(),
        "a cleared token must leave no channel behind"
    );
}

/// Saves landing on each other's heels: each is answered with the channel the previous one built,
/// whose listener has not begun, while the frame that really holds events is still open. The wait
/// cannot live on the channels — whichever hand-over walks such a chain first consumes it, and the
/// next save would start a listener over a frame still holding undelivered events, reaching the
/// pipeline twice with the same content. It lives under the channel's name.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn a_save_landing_on_another_waits_out_the_frame_that_holds_events() {
    let fake = FakeTelegram::start(&[
        ("token-a", 700),
        ("token-b", 700),
        ("token-c", 700),
        ("token-d", 700),
    ]);
    fake.push_batch("token-a", vec![test_update(&[])]);
    fake.push_batch("token-b", vec![test_text_update(2, "from-b")]);
    fake.push_batch("token-c", vec![test_text_update(3, "from-c")]);
    fake.push_batch("token-d", vec![test_text_update(4, "from-d")]);
    let (store, _dir) = test_store().await;

    let registry = crate::ChannelRegistry::default();

    // The listener every save replaces holds a burst it cannot hand on (no pipeline room).
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    tx.send(ChannelMessage::default())
        .await
        .expect("the pipeline accepts the message that fills it");
    let old: std::sync::Arc<dyn Channel> = std::sync::Arc::new(channel("token-a", &fake, &store));
    registry.register(std::sync::Arc::clone(&old));
    let listener = tokio::spawn({
        let old = std::sync::Arc::clone(&old);
        async move {
            let _ = old.listen(tx).await;
        }
    });
    until("the intake takes the event on", || async {
        !still_held(&store, "700").await.is_empty()
    })
    .await;

    let first: std::sync::Arc<dyn Channel> = std::sync::Arc::new(channel("token-b", &fake, &store));
    let second: std::sync::Arc<dyn Channel> =
        std::sync::Arc::new(channel("token-c", &fake, &store));
    let third: std::sync::Arc<dyn Channel> = std::sync::Arc::new(channel("token-d", &fake, &store));
    let frames = registry.replaced("telegram");

    let (tx_first, mut rx_first) = tokio::sync::mpsc::channel(8);
    let (tx_second, mut rx_second) = tokio::sync::mpsc::channel(8);
    let (tx_third, mut rx_third) = tokio::sync::mpsc::channel(8);
    registry.replace(std::sync::Arc::clone(&first));
    let hand_over_first = tokio::spawn(hand_over_listener(
        std::sync::Arc::clone(&frames),
        std::sync::Arc::clone(&first),
        tx_first,
    ));
    registry.replace(std::sync::Arc::clone(&second));
    let hand_over_second = tokio::spawn(hand_over_listener(
        std::sync::Arc::clone(&frames),
        std::sync::Arc::clone(&second),
        tx_second,
    ));
    registry.replace(std::sync::Arc::clone(&third));
    let hand_over_third = tokio::spawn(hand_over_listener(
        std::sync::Arc::clone(&frames),
        std::sync::Arc::clone(&third),
        tx_third,
    ));

    // While the frame holding the event stays open, no later save's listener may begin, on pain of
    // taking that event over as it is handed on. The check holds the frame open while it looks, and
    // the negative wait is the shape the property takes: `probe` must not come true.
    let save_listeners = [&first, &second, &third].map(|channel| {
        channel
            .as_any()
            .downcast_ref::<TelegramChannel>()
            .expect("the check's own channel")
    });
    assert!(
        tokio::time::timeout(crate::channels::telegram_group::QUIET_WINDOW * 2, async {
            while !save_listeners.iter().any(|listener| listener.running()) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .is_err(),
        "a save's listener began before the frame that holds events ended"
    );

    // Make room: the replaced frame ends, handing its burst on and settling it, and only then can
    // the hand-overs' listeners begin. Each was parked on that frame, so its end releases all.
    rx.recv()
        .await
        .expect("the message that filled the pipeline");
    let msg = next_message(&mut rx, "the event the replaced listener held").await;
    assert_eq!(msg.content, "hello");
    until("the replaced frame settles the event", || async {
        still_held(&store, "700").await.is_empty()
    })
    .await;
    tokio::time::timeout(OBSERVABLE_TIMEOUT, hand_over_first)
        .await
        .expect("a hand-over waiting on the frame must finish with it")
        .expect("the hand-over task must not panic");
    tokio::time::timeout(OBSERVABLE_TIMEOUT, hand_over_second)
        .await
        .expect("a hand-over waiting on the frame must finish with it")
        .expect("the hand-over task must not panic");

    // Only the last save's listener began: the event is not handed on again, it serves its batch.
    let msg = next_message(
        &mut rx_third,
        "the last save's listener takes up its own poll",
    )
    .await;
    let carried: Vec<&str> = std::iter::once(msg.content.as_str())
        .chain(msg.parts.iter().map(|part| part.content.as_str()))
        .collect();
    assert!(
        carried.contains(&"from-d"),
        "the listener that came last must take up its own batch: {carried:?}"
    );
    assert!(
        !carried.contains(&"hello"),
        "the event the replaced frame handed on must not reach the listener that came last: {carried:?}"
    );
    let handed_on_twice = tokio::time::timeout(
        crate::channels::telegram_group::QUIET_WINDOW * 2,
        rx_third.recv(),
    )
    .await;
    assert!(
        handed_on_twice.is_err(),
        "the event the replaced frame handed on was handled a second time"
    );
    // And the channels the earlier saves built serve their own batches at most, never the event.
    for (rx, what) in [
        (&mut rx_first, "the first save's channel"),
        (&mut rx_second, "the second save's channel"),
    ] {
        while let Ok(Some(msg)) =
            tokio::time::timeout(crate::channels::telegram_group::QUIET_WINDOW * 2, rx.recv()).await
        {
            let carried: Vec<&str> = std::iter::once(msg.content.as_str())
                .chain(msg.parts.iter().map(|part| part.content.as_str()))
                .collect();
            assert!(
                !carried.contains(&"hello"),
                "{what} handed on the event the replaced frame held: {carried:?}"
            );
        }
    }
    assert!(
        registry
            .get("telegram")
            .is_some_and(|live| std::sync::Arc::ptr_eq(&live, &third)),
        "the listener that came last is the one the registry holds"
    );
    listener.abort();
    hand_over_third.abort();
}

/// A row past the retention bound is swept before the first batch is matched: the platform re-uses
/// `update_id` after long silence, so a stale decision would swallow the new event.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn an_event_the_retention_bound_swept_is_not_taken_for_a_repeat() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    fake.push_batch("token-a", vec![test_update(&[])]);
    let (store, _dir) = test_store().await;

    let previous = run(&store, "previous-run", "700");
    previous
        .hold(&[&test_update(&[])])
        .await
        .expect("hold the old event");
    previous.settle(&[1]).await;
    let old = (chrono::Utc::now() - chrono::Duration::hours(RETENTION_HOURS + 1)).to_rfc3339();
    store
        .conn
        .execute(
            "UPDATE telegram_intake SET decided_at = ?1",
            crate::db::params![old],
        )
        .await
        .expect("age the decision past the bound");

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let telegram = channel("token-a", &fake, &store);
    let listener = tokio::spawn(async move {
        let _ = telegram.listen(tx).await;
    });

    let msg = next_message(&mut rx, "the event the platform re-used the id of").await;
    assert_eq!(msg.content, "hello");
    listener.abort();
}

/// A listener's frame is over only once what it held has been handed on: the frame waits for its
/// collector, which hands over and settles the burst. That is what lets a replacement take over
/// only what the cancelled listener really left behind.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn a_cancelled_listener_ends_its_frame_only_once_what_it_held_is_handed_on() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    fake.push_batch("token-a", vec![test_update(&[])]);
    let (store, _dir) = test_store().await;

    // A pipeline with no room in it: the collector cannot hand the burst over until the check takes
    // something out, which is what holds the listener's frame open.
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    tx.send(ChannelMessage::default())
        .await
        .expect("the pipeline accepts the message that fills it");
    let telegram = std::sync::Arc::new(channel("token-a", &fake, &store));
    let listener = tokio::spawn({
        let telegram = std::sync::Arc::clone(&telegram);
        async move {
            let _ = telegram.listen(tx).await;
        }
    });

    until("the intake takes the event on", || async {
        !still_held(&store, "700").await.is_empty()
    })
    .await;

    telegram.cancel_own();
    assert!(
        tokio::time::timeout(Duration::from_millis(500), telegram.wait_stopped())
            .await
            .is_err(),
        "the frame ended while the burst it held was still unhanded"
    );

    // Make room: the hand-over goes through and settles the event; only then does the frame end.
    let filler = rx
        .recv()
        .await
        .expect("the message that filled the pipeline");
    assert!(filler.content.is_empty());
    let msg = next_message(&mut rx, "the event the frame held").await;
    assert_eq!(msg.content, "hello");
    tokio::time::timeout(OBSERVABLE_TIMEOUT, telegram.wait_stopped())
        .await
        .expect("the frame must end once what it held was handed on");
    until("the hand-over settles the event", || async {
        still_held(&store, "700").await.is_empty()
    })
    .await;
    listener.abort();

    // The replacement a cancel-and-wait makes room for: since the frame ended only once what it
    // held was handed on and settled, nothing is left to take up and the event reaches the pipeline
    // once. The wait below is the collector's quiet window with room to spare.
    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let replacement = channel("token-a", &fake, &store);
    let listener = tokio::spawn(async move {
        let _ = replacement.listen(tx).await;
    });
    assert!(
        tokio::time::timeout(crate::channels::telegram_group::QUIET_WINDOW * 2, rx.recv())
            .await
            .is_err(),
        "the replacement must not hand on what the cancelled frame already settled"
    );
    listener.abort();
}

/// The frame is the collector's, not the listening future's: a listener that ends abruptly — task
/// dropped, or a panic — still leaves its frame open while its collector holds an undelivered
/// burst, so a replacement waits the hand-over out instead of starting on top of it. The cancelled
/// listener above ends the way a token change stops one; this end nobody asks for.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn a_dropped_listener_leaves_its_frame_open_until_the_burst_is_handed_on() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    fake.push_batch("token-a", vec![test_update(&[])]);
    let (store, _dir) = test_store().await;

    // A pipeline with no room in it: the collector cannot hand the burst over until the check takes
    // something out, which is what holds the listener's frame open.
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    tx.send(ChannelMessage::default())
        .await
        .expect("the pipeline accepts the message that fills it");
    let telegram = std::sync::Arc::new(channel("token-a", &fake, &store));
    let listener = tokio::spawn({
        let telegram = std::sync::Arc::clone(&telegram);
        async move {
            let _ = telegram.listen(tx).await;
        }
    });

    // The burst is in the collector's hands before the listener is dropped: the poll the intake
    // issues after taking the event on is the sign (a failed hand-over never polls again).
    until("the intake handed the burst to the collector", || async {
        fake.count("getUpdates") >= 2
    })
    .await;
    listener.abort();
    let _ = listener.await;

    assert!(
        telegram.running(),
        "the frame must be the collector's, not the listening future's"
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(500), telegram.wait_stopped())
            .await
            .is_err(),
        "the frame ended while the burst it held was still unhanded"
    );

    // Make room: the collector hands the burst over and settles it, and only then does the frame
    // end — the state a replacement waits for.
    let filler = rx
        .recv()
        .await
        .expect("the message that filled the pipeline");
    assert!(filler.content.is_empty());
    let msg = next_message(&mut rx, "the burst the dropped listener held").await;
    assert_eq!(msg.content, "hello");
    tokio::time::timeout(OBSERVABLE_TIMEOUT, telegram.wait_stopped())
        .await
        .expect("the frame must end once what it held was handed on");
    until("the hand-over settles the event", || async {
        still_held(&store, "700").await.is_empty()
    })
    .await;
}

/// Once the daemon winds down the intake stops fetching — but the event it already holds is still
/// handed over and settled. The pipeline has no room for the burst until the check makes it, so the
/// event sits in the record while the listener winds down, neither dropped nor left unhandled.
/// Serialized on the process-global drain flag, as that check is.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn a_wind_down_stops_fetching_and_keeps_what_it_holds() {
    /// Restores the process-global drain flag even when an assertion fails: a flag left set would
    /// misclassify every later check that consults it.
    struct DrainRestored;
    impl Drop for DrainRestored {
        fn drop(&mut self) {
            crate::shutdown::drain_clear();
        }
    }

    let fake = FakeTelegram::start(&[("token-a", 700)]);
    fake.push_batch("token-a", vec![test_update(&[])]);
    let (store, _dir) = test_store().await;
    let telegram = channel("token-a", &fake, &store);

    // A pipeline with no room: the burst cannot be handed over until the check takes something out,
    // which is what forces the hand-over under the wind-down below.
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    tx.send(ChannelMessage::default())
        .await
        .expect("the pipeline accepts the message that fills it");
    let listener = tokio::spawn(async move {
        let _ = telegram.listen(tx).await;
    });

    // The intake has taken the event on: the acknowledgement its parse sent is the durable sign.
    // The record's hold is not (the collector clears it on hand-over), but with no pipeline room it
    // stays where the assertion below reads it.
    until("the intake takes the event on", || async {
        fake.count("setMessageReaction") >= 1
    })
    .await;
    // The loop's own poll must be in flight before the count is read: a later one asks otherwise.
    until("the intake is polling", || async {
        fake.count("getUpdates") >= 2
    })
    .await;
    let polls_before = fake.count("getUpdates");

    let _restored = DrainRestored;
    crate::shutdown::drain_begin();

    // The listener cannot finish here: its frame is the collector's, still holding the burst, so an
    // await would hang rather than fail. What the check can see while the frame is open is that the
    // event is still in the record — the wind-down dropped nothing already taken on.
    assert_eq!(
        still_held(&store, "700").await,
        vec![1],
        "the wind-down must keep the event the intake already took on"
    );

    // Make room: the collector hands the held burst over, and the event reaches the pipeline as
    // itself, not as the filler the check sent.
    rx.recv()
        .await
        .expect("the message that filled the pipeline");
    let msg = next_message(&mut rx, "the event the wind-down flushed").await;
    assert_eq!(msg.content, "hello");
    until("the wind-down settles the event", || async {
        still_held(&store, "700").await.is_empty()
    })
    .await;

    tokio::time::timeout(OBSERVABLE_TIMEOUT, listener)
        .await
        .expect("the listener must stop when the daemon winds down")
        .expect("the listener task must not panic");

    // Nothing was taken on after the wind-down began. The count is read last because the frame
    // cannot end while the burst is unhanded, so the fetch-stop cannot be asserted earlier.
    assert_eq!(
        fake.count("getUpdates"),
        polls_before,
        "the intake kept fetching while winding down"
    );
}

/// An event the record already holds is never taken up twice: the platform re-offers a batch until
/// its position moves past it, so the same update can legitimately arrive again.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn an_event_the_record_already_holds_is_not_taken_up_again() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    fake.push_batch("token-a", vec![test_update(&[])]);
    fake.push_batch("token-a", vec![test_update(&[])]);
    let (store, _dir) = test_store().await;
    let telegram = channel("token-a", &fake, &store);

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let listener = tokio::spawn(async move {
        let _ = telegram.listen(tx).await;
    });

    let msg = next_message(&mut rx, "the event, exactly once").await;
    assert_eq!(msg.content, "hello");
    assert!(
        tokio::time::timeout(Duration::from_millis(300), rx.recv())
            .await
            .is_err(),
        "an event the record already held was taken up again"
    );

    // The position must move past what the record holds: the platform re-offers a batch until it
    // does, so a stalled position would re-offer the event forever. A stalled position pauses
    // before its next poll, hence the widened bound — the cheap one would time out on timing luck.
    until_within(
        BACKOFF_OBSERVABLE_TIMEOUT,
        "the fake has logged three polls",
        || async { fake.count("getUpdates") >= 3 },
    )
    .await;
    // The handshake's poll is the first, at offset 0; every poll after it must ask for an offset
    // past the event the record holds (the fake serves `update_id` 1), or it is offered forever.
    let offsets = fake.poll_offsets();
    assert!(
        offsets[1..].iter().all(|&offset| offset >= 2),
        "the batch is offered again forever because the position never moved past \
         the event the record already holds: poll offsets {offsets:?}"
    );

    until("the hand-over settles the event", || async {
        still_held(&store, "700").await.is_empty()
    })
    .await;
    listener.abort();
}

/// Control input recovered after a restart is not executed out of context: the user is told to
/// repeat it, so a stale `/clear` cannot wipe a session at a moment the user did not choose.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn recovered_control_input_is_not_executed_out_of_context() {
    // Nothing is queued: the handshake's probe gets an empty result and the long poll then hangs,
    // which keeps the listener quiet while recovery runs.
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    let (store, _dir) = test_store().await;
    run(&store, "previous-run", "700")
        .hold(&[&test_text_update(1, "/clear")])
        .await
        .expect("hold the recovered command");

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let telegram = channel("token-a", &fake, &store);
    let listener = tokio::spawn(async move {
        let _ = telegram.listen(tx).await;
    });

    until("the recovered command is settled", || async {
        still_held(&store, "700").await.is_empty()
    })
    .await;

    assert!(
        fake.sent_texts()
            .iter()
            .any(|text| text == UNACTED_CONTROL_NOTICE),
        "the user must be told the recovered command was not carried out"
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(300), rx.recv())
            .await
            .is_err(),
        "a recovered command must not be executed out of context"
    );
    listener.abort();
}

/// A backlog — what the platform accumulated while the product was down — is input the product
/// never acted on, wherever in it a command sits: the handshake's batch is of unknown age and so is
/// every page behind it, so a command in any of them is refused and its user told to repeat it.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn control_input_anywhere_in_a_backlog_is_not_executed() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    // A full page of what the platform accumulated, with a command at its end — the batch the
    // handshake fetches — and a second page carrying another command, which only a full first page
    // puts behind the backlog's end. The rest of the first page parses to nothing.
    let page = i64::try_from(UPDATES_PAGE).expect("a page of updates fits an update_id");
    let mut first: Vec<serde_json::Value> = (1..page)
        .map(|update_id| serde_json::json!({ "update_id": update_id }))
        .collect();
    first.push(test_text_update(page, "/clear"));
    fake.push_batch("token-a", first);
    fake.push_batch("token-a", vec![test_text_update(page + 1, "/clear")]);
    let (store, _dir) = test_store().await;

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let telegram = channel("token-a", &fake, &store);
    let listener = tokio::spawn(async move {
        let _ = telegram.listen(tx).await;
    });

    until("both backlogged commands are refused", || async {
        fake.sent_texts()
            .iter()
            .filter(|text| *text == UNACTED_CONTROL_NOTICE)
            .count()
            == 2
    })
    .await;
    until("the backlog is settled", || async {
        still_held(&store, "700").await.is_empty()
    })
    .await;

    assert!(
        tokio::time::timeout(Duration::from_millis(300), rx.recv())
            .await
            .is_err(),
        "control input anywhere in a backlog must not be executed"
    );
    listener.abort();
}

/// The complement of the refusals: a command this run's own poll fetched is acted on as it always
/// was — the intake hands it to the pipeline. A classification refusing every control input would
/// leave every other check here green while breaking every command in production.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn control_input_of_this_runs_own_poll_is_dispatched() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    // The handshake's zero-timeout probe takes the empty batch, ending the backlog and leaving the
    // loop's own long poll fresh; the command is the batch after it.
    fake.push_batch("token-a", Vec::new());
    fake.push_batch("token-a", vec![test_text_update(1, "/clear")]);
    let (store, _dir) = test_store().await;

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let telegram = channel("token-a", &fake, &store);
    let listener = tokio::spawn(async move {
        let _ = telegram.listen(tx).await;
    });

    let msg = next_message(&mut rx, "the command this run's own poll fetched").await;
    assert_eq!(msg.content, "/clear");
    assert!(
        !fake
            .sent_texts()
            .iter()
            .any(|text| text == UNACTED_CONTROL_NOTICE),
        "a command of this run's own poll is carried out, not refused"
    );
    until("the dispatched command is settled", || async {
        still_held(&store, "700").await.is_empty()
    })
    .await;
    listener.abort();
}

/// Control input in a batch the intake had to fetch twice is not executed: the record could not be
/// written for the handshake's batch, so the poll collecting it again is looking at events this run
/// already saw, and a command of unknown age must not run.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn control_input_in_a_reoffered_batch_is_not_executed() {
    // The same command three times: the handshake fetches the first copy with the record
    // unwritable, the loop's first poll the second, the poll after that the third.
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    for _ in 0..3 {
        fake.push_batch("token-a", vec![test_text_update(1, "/clear")]);
    }
    let (store, _dir) = test_store().await;
    store
        .conn
        .execute(
            "ALTER TABLE telegram_intake RENAME TO telegram_intake_hidden",
            (),
        )
        .await
        .expect("make the record unwritable");
    let telegram = channel("token-a", &fake, &store);

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let listener = tokio::spawn(async move {
        let _ = telegram.listen(tx).await;
    });

    until_within(BACKOFF_OBSERVABLE_TIMEOUT, "the loop's poll", || async {
        fake.count("getUpdates") >= 2
    })
    .await;
    store
        .conn
        .execute(
            "ALTER TABLE telegram_intake_hidden RENAME TO telegram_intake",
            (),
        )
        .await
        .expect("restore the record");

    until_within(
        BACKOFF_OBSERVABLE_TIMEOUT,
        "the user is told the re-offered command was not carried out",
        || async {
            fake.sent_texts()
                .iter()
                .any(|text| text == UNACTED_CONTROL_NOTICE)
        },
    )
    .await;
    until_within(
        BACKOFF_OBSERVABLE_TIMEOUT,
        "the re-offered command is settled",
        || async { still_held(&store, "700").await.is_empty() },
    )
    .await;
    assert!(
        tokio::time::timeout(Duration::from_millis(300), rx.recv())
            .await
            .is_err(),
        "control input in a batch this run already fetched once must not be executed"
    );
    listener.abort();
}

/// An event whose record cannot be written for is neither acknowledged nor confirmed: nothing is
/// answered before its fate is durable, and the intake backs off instead of re-polling.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn an_event_the_record_cannot_be_written_for_is_neither_acknowledged_nor_confirmed() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    // Two batches: the handshake consumes the first and the loop's first poll the second, which
    // gives a poll an unrecordable event — dropping the table alone is invisible until one arrives.
    fake.push_batch("token-a", vec![test_update(&[])]);
    fake.push_batch("token-a", vec![test_update(&[])]);
    let (store, _dir) = test_store().await;
    store
        .conn
        .execute("DROP TABLE telegram_intake", ())
        .await
        .expect("make the record unwritable");
    let telegram = channel("token-a", &fake, &store);

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let listener = tokio::spawn(async move {
        let _ = telegram.listen(tx).await;
    });

    // The handshake consumed the first batch and the loop's first poll the second, whose record
    // could not be written for. The poll after it is the one the backoff holds off, and it hangs
    // with nothing to serve (still logged on arrival). Waiting needs a bound past a pause — and
    // past two, the handshake pausing on the same batch — hence the widened one.
    until_within(
        BACKOFF_OBSERVABLE_TIMEOUT,
        "the intake retries the unrecordable batch after its backoff",
        || async { fake.count("getUpdates") >= 3 },
    )
    .await;

    // The pause belongs to the intake's own loop, and the handshake's poll is the one asking for
    // offset 0, so the loop's polls are those above 0. The gap between the first two is the real
    // claim: a tight retry would put them milliseconds apart, not a second apart.
    let loop_polls: Vec<std::time::Instant> = {
        let requests = fake.state.requests.lock().expect("fake request log");
        requests
            .iter()
            .filter(|request| {
                request.method == "getUpdates" && request.offset.is_some_and(|offset| offset > 0)
            })
            .map(|request| request.at)
            .collect()
    };
    let gap = loop_polls[1] - loop_polls[0];
    assert!(
        gap >= Duration::from_secs(1),
        "the intake re-polled an unrecordable batch after {gap:?}; it must back off"
    );

    // Nor is the event confirmed: every poll asked for an offset at or below the offered event's
    // own id (`test_update` carries `update_id` 1), because what the record lacks stays on offer.
    let offsets = fake.poll_offsets();
    assert!(
        offsets.iter().all(|&offset| offset <= 1),
        "the intake confirmed an event the record does not hold: poll offsets {offsets:?}"
    );

    assert_eq!(
        fake.count("setMessageReaction"),
        0,
        "an event must not be acknowledged before its fate is durable"
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(300), rx.recv())
            .await
            .is_err(),
        "an event the record could not be written for must not be handed on"
    );
    listener.abort();
}

/// A comment and its slow forwarded attachment reach the assistant as one turn. Both arrive in one
/// poll with the comment first, so the group would be released after its quiet window while the
/// attachment is still fetched: the collector holds it on the preparation instead, past the group's
/// own ceiling since the fetch takes longer.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn a_comment_and_a_slow_forwarded_attachment_reach_the_assistant_as_one_turn() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    fake.push_batch("token-a", Vec::new());
    fake.push_batch(
        "token-a",
        vec![
            test_text_update(1, "look at this"),
            forwarded_photo_update(2, 34, "photo-1", 1024),
        ],
    );
    fake.push_file("photo-1", b"jpeg-bytes", FETCH_DELAY);
    let (store, _dir) = test_store().await;
    let telegram = channel("token-a", &fake, &store);

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let listener = tokio::spawn(async move {
        let _ = telegram.listen(tx).await;
    });

    let msg = next_message_within(
        AT_MOST_ONE_TURN,
        &mut rx,
        "the comment and its forwarded photo, as one turn",
    )
    .await;
    assert_eq!(msg.parts.len(), 2, "the pair must be one turn, not two");
    assert_eq!(msg.parts[0].content, "look at this");
    assert!(
        msg.parts[1]
            .content
            .starts_with("[Forwarded from channel: Daily News] [IMAGE:"),
        "the forwarded photo must keep its attribution: {}",
        msg.parts[1].content
    );
    assert!(
        msg.parts[1].forwarded,
        "the fetched forward must come back as forwarded"
    );

    // The pair is not also answered as its own turn.
    assert!(
        tokio::time::timeout(Duration::from_millis(300), rx.recv())
            .await
            .is_err(),
        "the comment and its forwarded photo must be one turn, not two"
    );

    // The record is the discriminating part: with nothing holding the group the two parts would be
    // a quiet window apart and the attachment a turn of its own. A fetch apart is only possible
    // because the group was held while the fetch ran.
    until_within(AT_MOST_ONE_TURN, "the release is recorded", || async {
        !burst_rows(&store).await.is_empty()
    })
    .await;
    let rows = burst_rows(&store).await;
    assert_eq!(rows.len(), 1, "one burst must be recorded for the one turn");
    assert_eq!(rows[0].user_name, "alice");
    assert_eq!(rows[0].parts, 2);
    assert_eq!(rows[0].released_by, "extended_window");
    assert_eq!(rows[0].part_times.len(), 2);
    let taken_at = |at: usize| {
        chrono::DateTime::parse_from_rfc3339(
            rows[0].part_times[at]["taken_at"]
                .as_str()
                .expect("taken_at is a string"),
        )
        .expect("taken_at is RFC 3339")
    };
    let gap = (taken_at(1) - taken_at(0))
        .to_std()
        .expect("the second part was taken after the first");
    assert!(
        gap >= FETCH_DELAY.saturating_sub(Duration::from_millis(200)),
        "the two parts must be a fetch apart, not a window apart: {gap:?}"
    );

    assert!(rows[0].part_times[0]["sent_at"].is_null());
    assert_eq!(
        rows[0].part_times[1]["sent_at"].as_str(),
        Some(
            chrono::DateTime::from_timestamp(1_700_000_000, 0)
                .expect("a valid timestamp")
                .to_rfc3339()
                .as_str()
        )
    );
    listener.abort();
}

#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn a_slow_forwarded_attachment_and_its_comment_reach_the_assistant_as_one_turn() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    fake.push_batch("token-a", Vec::new());
    fake.push_batch(
        "token-a",
        vec![forwarded_photo_update(1, 33, "photo-2", 1024)],
    );
    fake.push_batch("token-a", vec![test_text_update(2, "my comment")]);
    fake.push_file("photo-2", b"jpeg-bytes", FETCH_DELAY);
    let (store, _dir) = test_store().await;
    let telegram = channel("token-a", &fake, &store);

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let listener = tokio::spawn(async move {
        let _ = telegram.listen(tx).await;
    });

    let msg = next_message_within(
        AT_MOST_ONE_TURN,
        &mut rx,
        "the forwarded photo and its comment, as one turn",
    )
    .await;
    assert_eq!(msg.parts.len(), 2, "the pair must be one turn, not two");
    assert!(
        msg.parts[0].forwarded,
        "the fetched forward must come back as forwarded"
    );
    assert!(
        msg.parts[0]
            .content
            .starts_with("[Forwarded from channel: Daily News] [IMAGE:"),
        "the forwarded photo must keep its attribution: {}",
        msg.parts[0].content
    );
    assert_eq!(msg.parts[1].content, "my comment");

    until_within(AT_MOST_ONE_TURN, "the release is recorded", || async {
        !burst_rows(&store).await.is_empty()
    })
    .await;
    let rows = burst_rows(&store).await;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].parts, 2);
    listener.abort();
}

/// The poll loop keeps polling while a batch is parsed: the fetch runs on the processor, so the
/// next `getUpdates` must arrive while it is in flight — the point of splitting the intake in two.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn the_poll_loop_keeps_polling_while_a_batch_is_parsed() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    // The handshake's zero-timeout probe takes the empty batch; the photo's batch is the first the
    // processor parses, and the third poll must arrive while that fetch still runs.
    fake.push_batch("token-a", Vec::new());
    fake.push_batch(
        "token-a",
        vec![forwarded_photo_update(1, 33, "photo-1", 1024)],
    );
    fake.push_batch("token-a", vec![test_text_update(2, "still here")]);
    fake.push_file("photo-1", b"jpeg-bytes", FETCH_DELAY);
    let (store, _dir) = test_store().await;
    let telegram = channel("token-a", &fake, &store);

    let (tx, _rx) = tokio::sync::mpsc::channel(8);
    let listener = tokio::spawn(async move {
        let _ = telegram.listen(tx).await;
    });

    until_within(
        Duration::from_millis(1500),
        "a third getUpdates arrives while the photo is still being fetched",
        || async { fake.poll_offsets().len() >= 3 },
    )
    .await;
    listener.abort();
}

/// Ordinary behaviour is unchanged: two plain messages in quick succession still merge into one
/// turn, and a message arriving after the group is gone is a turn of its own.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn two_plain_messages_in_quick_succession_still_merge_and_a_later_one_is_its_own_turn() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    // The handshake consumes this batch; the loop's poll then waits for the later one, queued only
    // once the first turn is observed — queued up front it would merge into the group.
    fake.push_batch(
        "token-a",
        vec![test_text_update(1, "first"), test_text_update(2, "second")],
    );
    let (store, _dir) = test_store().await;
    let telegram = channel("token-a", &fake, &store);

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let listener = tokio::spawn(async move {
        let _ = telegram.listen(tx).await;
    });

    let msg = next_message_within(AT_MOST_ONE_TURN, &mut rx, "the two messages, as one turn").await;
    assert_eq!(msg.parts.len(), 2);
    assert_eq!(msg.parts[0].content, "first");
    assert_eq!(msg.parts[1].content, "second");

    fake.push_batch("token-a", vec![test_text_update(3, "later")]);
    let later = next_message(&mut rx, "the later message, as its own turn").await;
    assert!(
        later.parts.is_empty(),
        "a message after the group is gone must be a turn of its own"
    );
    assert_eq!(later.content, "later");

    until("both releases are recorded", || async {
        burst_rows(&store).await.len() == 2
    })
    .await;
    let rows = burst_rows(&store).await;
    assert_eq!(rows[0].parts, 2, "the pair's own row comes first");
    assert_eq!(rows[1].parts, 1, "the lone message is recorded too");
    assert_eq!(rows[1].released_by, "quiet_window");
    listener.abort();
}

/// Control input still leaves the group open ahead of itself: a real bot command in a burst is
/// handed over after the group's own message, never in front of it, and is a turn of its own.
#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial(drain)] // serializes the process-global drain flag
async fn a_command_in_a_burst_leaves_the_group_ahead_of_itself() {
    let fake = FakeTelegram::start(&[("token-a", 700)]);
    // The handshake's zero-timeout probe takes the empty batch, ending the backlog, and the loop's
    // own poll fetches the burst — a command of this run, dispatched rather than refused as stale.
    fake.push_batch("token-a", Vec::new());
    fake.push_batch(
        "token-a",
        vec![
            test_text_update(1, "the group"),
            test_text_update(2, "/clear"),
        ],
    );
    let (store, _dir) = test_store().await;
    let telegram = channel("token-a", &fake, &store);

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let listener = tokio::spawn(async move {
        let _ = telegram.listen(tx).await;
    });

    let group = next_message(&mut rx, "the group ahead of the command").await;
    assert_eq!(group.content, "the group");
    let command = next_message(&mut rx, "the command, as its own turn").await;
    assert!(command.parts.is_empty());
    assert_eq!(command.content, "/clear");

    assert!(
        tokio::time::timeout(Duration::from_millis(300), rx.recv())
            .await
            .is_err(),
        "nothing may follow the group and the command"
    );

    until("the group's release is recorded", || async {
        !burst_rows(&store).await.is_empty()
    })
    .await;
    let rows = burst_rows(&store).await;
    assert_eq!(rows.len(), 1, "only the group's own release is recorded");
    assert_eq!(rows[0].released_by, "control");
    assert_eq!(rows[0].parts, 1);
    listener.abort();
}
