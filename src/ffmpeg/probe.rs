//! The one `ffprobe` document the product reads.
//!
//! Kept free of process handling and of the module's error type, so every rule
//! here is a pure function of the JSON `ffprobe` prints — which is what the tests
//! feed it, verbatim from the tool.

use std::time::Duration;

use super::ClipInfo;

/// The facts [`ClipInfo`] needs from one `-print_format json` document, or `None`
/// when it describes no usable stream and no duration.
///
/// A corrupt or foreign file makes `ffprobe` print an empty object — sometimes
/// with a zero exit status — so an empty answer is what "cannot be decoded" looks
/// like here; the exit status carries no more.
pub(super) fn parse(json: &[u8]) -> Option<ClipInfo> {
    let document: serde_json::Value = serde_json::from_slice(json).ok()?;
    let no_streams = Vec::new();
    let streams = document["streams"].as_array().unwrap_or(&no_streams);

    let mut has_audio = false;
    let mut dimensions: Option<(u32, u32)> = None;
    let mut streamed: Option<f64> = None;
    for stream in streams {
        match stream["codec_type"].as_str() {
            Some("audio") => has_audio = true,
            // Cover art rides in a video stream of its own; it is not the file's
            // video, and a track with a cover must not read as a video file.
            Some("video") if stream["disposition"]["attached_pic"].as_u64() != Some(1) => {
                if dimensions.is_none()
                    && let (Some(width), Some(height)) =
                        (side(&stream["width"]), side(&stream["height"]))
                {
                    dimensions = Some(upright(width, height, rotation(stream)));
                }
            }
            _ => {}
        }
        if let Some(seconds) = seconds(stream["duration"].as_str()) {
            streamed = Some(streamed.map_or(seconds, |longest| longest.max(seconds)));
        }
    }

    let duration = seconds(document["format"]["duration"].as_str()).or(streamed)?;
    let duration = Duration::try_from_secs_f64(duration).ok()?;
    if !has_audio && dimensions.is_none() {
        return None;
    }
    Some(ClipInfo {
        duration,
        dimensions,
        has_audio,
    })
}

/// A JSON number as a pixel extent.
fn side(value: &serde_json::Value) -> Option<u32> {
    u32::try_from(value.as_u64()?).ok().filter(|side| *side > 0)
}

/// A JSON duration in seconds. `ffprobe` prints it as a string, and as the literal
/// `N/A` where a container does not carry one.
fn seconds(value: Option<&str>) -> Option<f64> {
    let seconds: f64 = value?.parse().ok()?;
    (seconds.is_finite() && seconds >= 0.0).then_some(seconds)
}

/// The rotation a container asks of its video stream, in degrees. The display
/// matrix's own number is what current builds print; older ones only carry the
/// `rotate` tag.
fn rotation(stream: &serde_json::Value) -> i64 {
    let from_matrix = stream["side_data_list"]
        .as_array()
        .into_iter()
        .flatten()
        .find_map(|side_data| side_data["rotation"].as_f64());
    if let Some(degrees) = from_matrix {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a container's rotation is a whole number of degrees"
        )]
        return degrees.round() as i64;
    }
    stream["tags"]["rotate"]
        .as_str()
        .and_then(|tag| tag.parse().ok())
        .unwrap_or(0)
}

/// The dimensions as a viewer sees them: a quarter turn swaps them.
fn upright(width: u32, height: u32, rotation: i64) -> (u32, u32) {
    if rotation.rem_euclid(180) == 90 {
        (height, width)
    } else {
        (width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape `ffprobe` prints for an ordinary phone clip: one video stream and
    /// one audio stream, the duration on the container.
    const CLIP: &str = r#"{
        "streams": [
            {"index": 0, "codec_type": "video", "width": 1280, "height": 720,
             "tags": {}, "disposition": {"attached_pic": 0}},
            {"index": 1, "codec_type": "audio", "tags": {},
             "disposition": {"attached_pic": 0}}
        ],
        "format": {"duration": "2.008000"}
    }"#;

    #[test]
    fn a_clip_reports_its_duration_size_and_audio() {
        let info = parse(CLIP.as_bytes()).expect("a clip is readable");
        assert_eq!(info.duration, Duration::from_millis(2008));
        assert_eq!(info.dimensions, Some((1280, 720)));
        assert!(info.has_audio);

        // An audio file with embedded artwork: the picture rides in a video stream
        // of its own and must not read as the file's video.
        let cover = r#"{
            "streams": [
                {"codec_type": "audio", "duration": "2.000000",
                 "disposition": {"attached_pic": 0}},
                {"codec_type": "video", "width": 200, "height": 200,
                 "duration": "2.000000", "disposition": {"attached_pic": 1}}
            ],
            "format": {"duration": "2.000000"}
        }"#;
        let info = parse(cover.as_bytes()).expect("an audio file is readable");
        assert_eq!(info.dimensions, None, "a cover is not the file's video");
        assert!(info.has_audio);
    }

    #[test]
    fn a_quarter_turn_reports_the_upright_size() {
        for (rotation, expected) in [(90, (720, 1280)), (-90, (720, 1280)), (270, (720, 1280))] {
            let document = format!(
                r#"{{"streams": [{{"codec_type": "video", "width": 1280, "height": 720,
                   "side_data_list": [{{"rotation": {rotation}}}]}}],
                   "format": {{"duration": "2.0"}}}}"#
            );
            let info = parse(document.as_bytes()).expect("a rotated clip is readable");
            assert_eq!(info.dimensions, Some(expected), "rotation {rotation}");
        }
        // Half a turn changes nothing, and the older `rotate` tag is honoured.
        let upside_down = br#"{"streams": [{"codec_type": "video", "width": 1280, "height": 720,
                               "side_data_list": [{"rotation": 180}]}],
                               "format": {"duration": "2.0"}}"#;
        assert_eq!(
            parse(upside_down).expect("readable").dimensions,
            Some((1280, 720))
        );
        let tagged = br#"{"streams": [{"codec_type": "video", "width": 640, "height": 480,
                         "tags": {"rotate": "90"}}], "format": {"duration": "2.0"}}"#;
        assert_eq!(
            parse(tagged).expect("readable").dimensions,
            Some((480, 640))
        );
    }

    #[test]
    fn what_ffprobe_cannot_name_is_undecodable() {
        // A corrupt or foreign file: `ffprobe` prints an empty object, sometimes
        // with a zero exit status.
        assert_eq!(parse(b"{\n\n}"), None);
        assert_eq!(parse(b""), None);
        assert_eq!(parse(b"not json at all"), None);
        // No audio, no video, nothing to work with.
        assert_eq!(parse(br#"{"format": {"duration": "2.0"}}"#), None);
        // A duration a `Duration` cannot hold: crafted media must not take the
        // parse down with it.
        let absurd = br#"{"streams": [{"codec_type": "video", "width": 2, "height": 2}],
                          "format": {"duration": "1e30"}}"#;
        assert_eq!(parse(absurd), None, "a duration no Duration holds");
    }

    #[test]
    fn the_duration_falls_back_to_the_longest_stream() {
        // A container that declares no duration of its own — the audio is all it
        // can be timed by. `N/A` is what `ffprobe` prints where nothing exists.
        let document = br#"{
            "streams": [
                {"codec_type": "audio", "duration": "3.500000"},
                {"codec_type": "audio", "duration": "4.250000"},
                {"codec_type": "video", "width": 640, "height": 480, "duration": "N/A"}
            ],
            "format": {"duration": "N/A"}
        }"#;
        let info = parse(document).expect("stream durations time the file");
        assert_eq!(info.duration, Duration::from_millis(4250));
        assert_eq!(info.dimensions, Some((640, 480)));
    }
}
