//! Reply-reference tests: normalization, marker injection and author labels
//! for `channels::reply`.

use super::*;
use crate::channels::enrich_links;

/// The normalization runs its rules in one order: entities decode first (so
/// their angle brackets are then stripped), a `<blockquote>` wrapper comes off,
/// angle brackets go, and newlines collapse to spaces.
#[test]
fn normalize_applies_its_rules_in_order() {
    let cases: &[(&str, &str)] = &[
        // `&lt;b&gt;` decodes to `<b>` only after HTML decoding; the trailing
        // angle brackets are then stripped, leaving the inner text.
        ("&lt;b&gt;hi", "bhi"),
        (
            "<blockquote>\n↩ alice: hello\n</blockquote>",
            "↩ alice: hello",
        ),
        ("hello <world>", "hello world"),
        ("a\nb\r\nc", "a b c"),
    ];
    for (raw, expected) in cases {
        assert_eq!(normalize_reply_text(raw), *expected, "raw: {raw:?}");
    }
}

#[test]
fn normalize_maps_every_media_marker() {
    let cases: &[(&str, &str)] = &[
        ("look at [IMAGE:/p/x.png] now", "look at [Photo] now"),
        ("pic [IMAGE:data:image/png;base64,AAA] ok", "pic [Photo] ok"),
        ("[AUDIO:/tmp/a.ogg]", "[Voice message]"),
        // Case-insensitive marker recognition.
        ("[video:/tmp/v.mp4]", "[Video]"),
        ("[FILE:/tmp/report.pdf]", "[File]"),
    ];
    for (raw, expected) in cases {
        assert_eq!(normalize_reply_text(raw), *expected, "raw: {raw:?}");
    }
}

#[test]
fn normalize_caps_at_exactly_snippet_max() {
    let long = "a".repeat(REPLY_SNIPPET_MAX_CHARS + 50);
    let capped = normalize_reply_text(&long);
    assert_eq!(capped.chars().count(), REPLY_SNIPPET_MAX_CHARS);
    assert!(capped.ends_with('…'));

    // A snippet of exactly the cap is not touched at all.
    let exact: String = "a".repeat(REPLY_SNIPPET_MAX_CHARS);
    assert_eq!(normalize_reply_text(&exact), exact);
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

/// The real `enrich_links` call below drives the owner's own Chrome session
/// (it opens a `link-enricher-*` tab and leaves the visit in his history),
/// while the assertions only check marker ordering and hold with or without a
/// browser. Run it explicitly with:
///
/// ```sh
/// cargo test marker_injection_follows_link_enrichment -- --ignored --nocapture
/// ```
#[ignore = "drives the owner's real Chrome session to fetch the link; runs only when explicitly invoked"]
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
fn agent_author_label_resolves_and_falls_back() {
    assert_eq!(agent_author_label(Some("analyst_3")), "Analyst");
    assert_eq!(agent_author_label(Some("engineer")), "Engineer");
    assert_eq!(agent_author_label(None), "Agent");
    assert_eq!(agent_author_label(Some("stage_comment")), "Agent");
}
