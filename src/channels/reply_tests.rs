use super::*;
use crate::channels::enrich_links;

#[test]
fn truncate_capped_total_never_exceeds_cap() {
    assert_eq!(truncate_capped("hello world", 5), "hell…");
    assert_eq!(truncate_capped("hello world", 5).chars().count(), 5);
    // Degenerate cap: the result must still respect "at most max_chars".
    assert_eq!(truncate_capped("hello", 0), "");
    assert_eq!(truncate_capped("hello", 1), "…");
}

#[test]
fn truncate_capped_multibyte_char_boundary_safe() {
    // Three 4-byte emoji, capped to 2 chars → first emoji + ellipsis, no panic.
    assert_eq!(truncate_capped("😀😀😀", 2), "😀…");
    // Cyrillic chars are single-codepoint Unicode; no mid-char split.
    assert_eq!(truncate_capped("ааааа", 4), "ааа…");
}

#[test]
fn truncate_capped_no_ellipsis_when_within_cap() {
    assert_eq!(truncate_capped("hello", 10), "hello");
    assert_eq!(truncate_capped("", 5), "");
}

#[test]
fn normalize_decodes_entities_before_angle_strip() {
    // `&lt;b&gt;` decodes to `<b>` only after HTML decoding; the trailing
    // angle brackets are then stripped, leaving the inner text.
    assert_eq!(normalize_reply_text("&lt;b&gt;hi"), "bhi");
}

#[test]
fn normalize_strips_blockquote_wrapper() {
    assert_eq!(
        normalize_reply_text("<blockquote>\n↩ alice: hello\n</blockquote>"),
        "↩ alice: hello"
    );
}

#[test]
fn normalize_removes_angle_brackets() {
    assert_eq!(normalize_reply_text("hello <world>"), "hello world");
}

#[test]
fn normalize_collapses_newlines() {
    assert_eq!(normalize_reply_text("a\nb\r\nc"), "a b c");
}

#[test]
fn normalize_maps_image_marker_inline() {
    assert_eq!(
        normalize_reply_text("look at [IMAGE:/p/x.png] now"),
        "look at [Photo] now"
    );
    assert_eq!(
        normalize_reply_text("pic [IMAGE:data:image/png;base64,AAA] ok"),
        "pic [Photo] ok"
    );
}

#[test]
fn normalize_maps_audio_and_video_markers() {
    assert_eq!(
        normalize_reply_text("[AUDIO:/tmp/a.ogg]"),
        "[Voice message]"
    );
    // Case-insensitive marker recognition.
    assert_eq!(normalize_reply_text("[video:/tmp/v.mp4]"), "[Video]");
}

#[test]
fn normalize_caps_at_exactly_snippet_max() {
    let long = "a".repeat(REPLY_SNIPPET_MAX_CHARS + 50);
    let capped = normalize_reply_text(&long);
    assert_eq!(capped.chars().count(), REPLY_SNIPPET_MAX_CHARS);
    assert!(capped.ends_with('…'));
}

#[test]
fn normalize_leaves_exactly_max_unchanged() {
    let exact: String = "a".repeat(REPLY_SNIPPET_MAX_CHARS);
    assert_eq!(normalize_reply_text(&exact), exact);
}

#[test]
fn format_reply_marker_exact_literal() {
    let reply = ReplyReference {
        author: "alice".to_string(),
        snippet: "hi".to_string(),
    };
    assert_eq!(
        format_reply_marker(&reply),
        "<reply-to>alice: hi</reply-to>:\n"
    );
}

#[test]
fn apply_reply_marker_precedes_forwarded_prefix() {
    let reply = ReplyReference {
        author: "alice".to_string(),
        snippet: "hi".to_string(),
    };
    let content = "[Forwarded from @bob] hello";
    assert_eq!(
        apply_reply_marker(content, &reply),
        "<reply-to>alice: hi</reply-to>:\n[Forwarded from @bob] hello"
    );
}

#[tokio::test]
async fn marker_injection_follows_link_enrichment() {
    // Mirrors the bin's process_channel_message ordering: enrich first, then
    // apply the marker. The marker never passes through link enrichment
    // (applied only to the enriched pre-marker content), so it stays at the
    // very front of the routed content — before any URL summary prepended by
    // enrichment and before an embedded forward-attribution prefix.
    let reply = ReplyReference {
        author: "alice".to_string(),
        snippet: "look".to_string(),
    };
    let original = "[Forwarded from @bob] check https://example.com";
    let enriched = enrich_links(original).await.to_string();
    let routed = apply_reply_marker(&enriched, &reply);
    assert!(routed.starts_with("<reply-to>alice: look</reply-to>:\n"));
    assert!(
        routed.find("<reply-to>").unwrap() < routed.find("[Forwarded from @bob]").unwrap(),
        "marker must precede the forward-attribution prefix"
    );
}

#[test]
fn agent_author_label_resolves_display_labels() {
    assert_eq!(agent_author_label(Some("analyst_3")), "Analyst");
    assert_eq!(agent_author_label(Some("engineer")), "Engineer");
}

#[test]
fn agent_author_label_falls_back_to_agent() {
    assert_eq!(agent_author_label(None), "Agent");
    assert_eq!(agent_author_label(Some("stage_comment")), "Agent");
}
