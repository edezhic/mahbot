//! Pre-processing for media markers in chat content.
//!
//! Converts raw media markers like `[IMAGE:path]` into forms the markdown
//! renderer can handle gracefully:
//!
//! - `[IMAGE:path]` → proper markdown image syntax `![Image](path)`
//! - `[AUDIO:path]` → 🎵 emoji + filename (text)
//! - `[VIDEO:path]` → 🎬 emoji + placeholder text
//! - `[Audio transcription of ...]: text` → 🔊 emoji + transcribed text
//!
//! The pre-processing is applied **before** `markdown::parse()` so the
//! standard markdown pipeline naturally produces `Item::Image` from the
//! converted image markers. Audio and video markers become plain text
//! with emoji prefixes since inline audio/video playback is not supported.
//!
//! # Canonical marker pattern
//!
//! The `[KIND:path]` format is defined by `MEDIA_MARKER_PATTERN` in
//! `src/util/mod.rs`, which is **the single source of truth** for all marker
//! kinds (`IMAGE`, `AUDIO`, `VIDEO`, and any future additions).  This module
//! uses the shared [`MEDIA_MARKER_RE`] and [`parse_media_marker`] helper to
//! stay in sync — adding a new marker kind there automatically propagates
//! to this module, `enrichment.rs`, `telegram.rs`, `agent.rs`, and
//! `compatible.rs` without needing per-kind regexes here.

use std::sync::LazyLock;

use iced::advanced::{image as advanced_image, text};
use iced::widget::image;
use iced::widget::markdown;
use iced::{ContentFit, Element, Font, Length};

use crate::util::{MEDIA_MARKER_RE, parse_media_marker};

/// Pre-process a content string, converting media markers before markdown parsing.
pub(crate) fn preprocess(content: &str) -> String {
    // Order matters: the audio-transcription annotation contains the word "Audio"
    // which overlaps with the raw `[AUDIO:...]` pattern.  Handle it first.
    let s = replace_audio_transcription(content);
    replace_media_markers(&s)
}

/// `[Audio transcription of {filename}]: {text}` → 🔊 text
///
/// This annotation is produced by `enrich_message` before user messages are
/// broadcast to the GUI, so it now actively reaches the dashboard and is
/// rendered as transcription text.  The handler also covers persisted messages
/// (e.g., replayed from history) where the annotation might appear.
fn replace_audio_transcription(s: &str) -> String {
    static RE: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(r"\[Audio transcription of [^\]]+\]:\s*(.+)")
            .expect("audio transcription regex must compile")
    });
    RE.replace_all(s, "🔊 $1").to_string()
}

/// Convert all `[KIND:path]` markers (IMAGE, AUDIO, VIDEO) to their display form
/// using the canonical [`MEDIA_MARKER_RE`].
///
/// - `[IMAGE:path]` → `![Image](path)` (markdown image syntax — the markdown
///   parser will produce `Item::Image` from this)
/// - `[AUDIO:path]` → 🎵 filename
/// - `[VIDEO:path]` → 🎬 Video: filename
fn replace_media_markers(s: &str) -> String {
    MEDIA_MARKER_RE
        .replace_all(s, |caps: &regex::Captures| {
            let (kind, path) = parse_media_marker(caps);
            match kind {
                "IMAGE" => format!("![Image]({path})"),
                "AUDIO" => format!("🎵 {}", path_filename(path)),
                "VIDEO" => format!("🎬 Video: {}", path_filename(path)),
                // Unreachable for well-formed markers (MEDIA_MARKER_RE only
                // matches IMAGE|AUDIO|VIDEO), but defend against future changes.
                _ => caps.get_match().as_str().to_string(),
            }
        })
        .to_string()
}

/// Extract the file name (last path component) from a path string.
fn path_filename(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .map_or_else(|| path.to_string(), ToString::to_string)
}

// ── Custom markdown Viewer for inline image rendering ──────────────

/// A markdown `Viewer` that renders `![Image](path)` items as actual
/// image widgets.  Audio and video markers are already converted to plain
/// text by [`preprocess`], so only images reach this viewer.
///
/// The viewer uses `Uri` (= `String`) as its message type, identical to
/// the built-in [`DefaultViewer`], so callers can use `.map(…)` to
/// convert link-click events into their own message type.
pub(crate) struct MediaViewer;

/// Singleton instance of [`MediaViewer`] for use with `markdown::view_with()`.
pub(crate) const MEDIA_VIEWER: MediaViewer = MediaViewer;

impl<'a, Theme, Renderer> markdown::Viewer<'a, markdown::Uri, Theme, Renderer> for MediaViewer
where
    Theme: markdown::Catalog + 'a,
    Renderer: text::Renderer<Font = Font> + advanced_image::Renderer<Handle = image::Handle> + 'a,
{
    fn on_link_click(url: markdown::Uri) -> markdown::Uri {
        url
    }

    fn image(
        &self,
        settings: markdown::Settings,
        url: &'a markdown::Uri,
        _title: &'a str,
        _alt: &markdown::Text,
    ) -> Element<'a, markdown::Uri, Theme, Renderer> {
        let path = std::path::Path::new(url.as_str());
        if path.exists() {
            // Render the actual image, constrained to the bubble width.
            // Note: path.exists() is synchronous I/O on the render thread,
            // which is acceptable because file checks are fast (~µs) and
            // images only appear for agent-generated files that were just
            // created.  For sessions with many stale image references a
            // cached-existence check could be added.
            image::Image::new(url.as_str())
                .width(Length::Fill)
                .content_fit(ContentFit::Contain)
                .into()
        } else {
            // File doesn't exist (temp file cleaned up, or path is invalid).
            // Show a fallback with the filename.
            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(url.as_str());
            iced::widget::text(format!("🖼️ {filename}"))
                .size(settings.text_size)
                .into()
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_image_marker() {
        assert_eq!(
            preprocess("Look [IMAGE:/tmp/photo.png] here"),
            "Look ![Image](/tmp/photo.png) here"
        );
    }

    #[test]
    fn replace_image_marker_with_spaces_in_path() {
        assert_eq!(
            preprocess("img [IMAGE:/tmp/my file.png]"),
            "img ![Image](/tmp/my file.png)"
        );
    }

    #[test]
    fn replace_audio_marker() {
        assert_eq!(
            preprocess("Listen [AUDIO:/tmp/recording.ogg]"),
            "Listen 🎵 recording.ogg"
        );
    }

    #[test]
    fn replace_audio_marker_nested_path() {
        assert_eq!(
            preprocess("hear [AUDIO:/dir/subdir/rec.ogg]"),
            "hear 🎵 rec.ogg"
        );
    }

    #[test]
    fn replace_video_marker() {
        assert_eq!(
            preprocess("Watch [VIDEO:/tmp/video.mp4]"),
            "Watch 🎬 Video: video.mp4"
        );
    }

    #[test]
    fn replace_audio_transcription() {
        assert_eq!(
            preprocess("[Audio transcription of recording.ogg]: Hello world"),
            "🔊 Hello world"
        );
    }

    #[test]
    fn replace_audio_transcription_multiline() {
        assert_eq!(
            preprocess("[Audio transcription of voice.ogg]: Line one\nLine two"),
            "🔊 Line one\nLine two"
        );
    }

    #[test]
    fn audio_transcription_prevents_overlap_with_audio_marker() {
        // The audio-transcription format contains "Audio" — the preprocess
        // must handle it before the [AUDIO:...] pattern.
        let result =
            preprocess("[Audio transcription of msg.ogg]: hi there [AUDIO:/tmp/other.ogg]");
        assert_eq!(result, "🔊 hi there 🎵 other.ogg");
    }

    #[test]
    fn multiple_markers_mixed() {
        let input = "![]() [IMAGE:/tmp/a.png] and [AUDIO:/tmp/b.ogg] end [VIDEO:/tmp/c.mp4]";
        let expected = "![]() ![Image](/tmp/a.png) and 🎵 b.ogg end 🎬 Video: c.mp4";
        assert_eq!(preprocess(input), expected);
    }

    #[test]
    fn no_markers_unchanged() {
        assert_eq!(preprocess("Hello world"), "Hello world");
    }

    #[test]
    fn empty_string() {
        assert_eq!(preprocess(""), "");
    }

    #[test]
    fn path_filename_unix() {
        assert_eq!(path_filename("/foo/bar.txt"), "bar.txt");
    }

    #[test]
    fn path_filename_nested() {
        assert_eq!(path_filename("/foo/bar/doc.txt"), "doc.txt");
    }

    #[test]
    fn path_filename_no_dir() {
        assert_eq!(path_filename("bar.txt"), "bar.txt");
    }

    #[test]
    fn path_filename_trailing_slash() {
        // On Unix `file_name()` normalizes the trailing slash.
        assert_eq!(path_filename("/foo/bar/"), "bar");
    }
}
