//! Media understanding: one path from a video or a recording to what a model reads.
//!
//! A video is never handed over whole. It is probed, its frames are sampled across the clip, its
//! sound track is transcribed with timestamps and labelled, and the three are rendered as one
//! timeline in time order — frames with their marks, speech lines with their marks and text, and a
//! sound label only where there is no speech. A recording takes the same path without frames.
//!
//! The frames reach the model as pictures of four: a picture is a 2×2 grid of frames in reading
//! order — left to right, then the next row — so one image carries four moments. The timeline
//! numbers the frames, and each picture is labelled with the ones it holds, so a moment still has
//! its second.
//!
//! What a file *is* comes from the probe, never from its name: a picture stream makes it a video, a
//! sound track alone makes it a recording. The name only decides whether the probe is asked at all
//! ([`is_media_path`]); what the file holds, and so which of the two it is, is the probe's answer.
//!
//! The sound track is transcribed whole, in five-minute pieces, once: the regions are cached under
//! the product's storage root ([`transcript`]), keyed by the file's path, size and modification
//! time. An overview shows the first ten minutes and says where it stopped; a read window is served
//! from the same cache. A pass that reaches its own bound keeps what it read and the next look
//! resumes from there.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::ffmpeg::{self, ClipInfo, FfmpegError};
use crate::speech::Segment;

mod transcript;

/// How much of a sound track an overview's timeline shows; anything past it is read by window.
const OVERVIEW_SECONDS: f64 = 600.0;

/// An overview's frames: one per second of the clip, never fewer than eight, spread evenly across
/// the whole clip.
const OVERVIEW_FRAMES_MIN: u32 = 8;
const OVERVIEW_FRAME_SIDE: u32 = 512;

/// A window's frames: two per second of the window, spread evenly across it.
const WINDOW_FRAMES_PER_SECOND: f64 = 2.0;
const WINDOW_FRAME_SIDE: u32 = 768;

/// The most frames one read attaches, whichever stretch it covers: six pictures of four.
const FRAMES_MAX: u32 = 24;

/// How many frames one picture carries: four, in reading order — left to right, then the next row.
/// The last picture of a read may hold fewer, its remaining cells blank.
const PICTURE_CELLS: usize = 4;

/// How far short of the stretch it was asked for a pass may stop and still count as having read
/// it. The coverage note and the silence line compare seconds with the same tolerance.
const COVERAGE_SLACK: f64 = 1.0;

/// What a media file is, decided by what the probe finds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Video,
    Audio,
}

/// The stretch of a media file a read covers, in seconds.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Window {
    /// The whole clip at overview density: its frames spread across all of it, its transcript
    /// capped at [`OVERVIEW_SECONDS`].
    Overview,
    /// `start` seconds in, for `length` seconds — to the end of the file when absent.
    Span { start: f64, length: Option<f64> },
}

/// What a media read answers: the file it read, the timeline the model reads, and the frames it
/// names.
pub(crate) struct Reading {
    /// The file the reading came from, so a surface that attaches the pictures never has to be told
    /// again — and cannot be told a different one.
    pub(crate) path: PathBuf,
    pub(crate) text: String,
    /// The extraction the timeline names, removed when the reading is dropped. Absent for a
    /// recording, which has no frames at all, and for a video whose frames could not be taken.
    strip: Option<ffmpeg::Frames>,
}

/// One picture a read attaches: up to four frames laid out two by two, with the frames it holds.
pub(crate) struct Picture {
    pub(crate) data_uri: String,
    pub(crate) size: (u32, u32),
    /// The 1-based numbers of the frames this picture holds, in cell order. A frame that could not
    /// be read leaves its cell blank and is missing here.
    frames: Vec<usize>,
}

impl Picture {
    /// The frames it holds, as the timeline numbers them: `frames 1–4`, or `frames 1–2, 4` when a
    /// cell is blank.
    pub(crate) fn frames_text(&self) -> String {
        frame_numbers(&self.frames)
    }
}

impl Reading {
    /// The frames in timeline order, each with the second it marks.
    fn frames(&self) -> &[(PathBuf, f64)] {
        self.strip.as_ref().map_or(&[], ffmpeg::Frames::frames)
    }

    /// The frames as the pictures a model reads, in timeline order. A picture none of whose frames
    /// could be read is left out — nothing is attached for it, and the log says why.
    pub(crate) async fn pictures(&self) -> Vec<Picture> {
        let mut pictures = Vec::with_capacity(self.frames().len().div_ceil(PICTURE_CELLS));
        for (index, group) in self.frames().chunks(PICTURE_CELLS).enumerate() {
            if let Some(picture) = picture(group, index * PICTURE_CELLS + 1).await {
                pictures.push(picture);
            }
        }
        pictures
    }
}

/// One picture from `group`, whose first frame is numbered `first`: two cells a row in reading
/// order, blank where a group holds fewer than four or a frame cannot be read. `None` when no frame
/// of the group could be read.
async fn picture(group: &[(PathBuf, f64)], first: usize) -> Option<Picture> {
    let mut cells = Vec::with_capacity(group.len());
    for (index, (jpeg, _)) in group.iter().enumerate() {
        match tokio::fs::read(jpeg).await {
            Ok(bytes) => cells.push((index, first + index, bytes)),
            Err(error) => tracing::warn!(
                path = %jpeg.display(),
                %error,
                "A frame could not be read"
            ),
        }
    }
    crate::util::with_block_in_place(|| compose(&cells))
}

/// One picture: the cells laid out two by two and encoded as JPEG. Answers the picture, or `None`
/// when none of its frames could be decoded or the encode failed.
fn compose(cells: &[(usize, usize, Vec<u8>)]) -> Option<Picture> {
    let mut frames = Vec::with_capacity(cells.len());
    for (cell, number, jpeg) in cells {
        match image::load_from_memory(jpeg) {
            Ok(frame) => frames.push((*cell, *number, frame)),
            Err(error) => tracing::warn!(frame = number, %error, "A frame could not be decoded"),
        }
    }
    // One cell size for the whole grid, so the frames keep their aspect ratio and a smaller frame
    // sits at its cell's origin rather than stretching.
    let (width, height) = frames
        .iter()
        .fold((0u32, 0u32), |(width, height), (_, _, frame)| {
            (width.max(frame.width()), height.max(frame.height()))
        });
    if width == 0 || height == 0 {
        return None;
    }
    let mut canvas = image::RgbImage::from_pixel(width * 2, height * 2, image::Rgb([0, 0, 0]));
    for (cell, _, frame) in &frames {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a picture's cell index is 0..4"
        )]
        let (column, row) = ((cell % 2) as u32, (cell / 2) as u32);
        image::imageops::replace(
            &mut canvas,
            &frame.to_rgb8(),
            i64::from(column * width),
            i64::from(row * height),
        );
    }
    let mut jpeg = Vec::new();
    // The frames were cut at this quality, so the grid's own generation loses about as much as the
    // frames already did — a second generation is unavoidable, a heavier one is not.
    if let Err(error) =
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, ffmpeg::FRAME_QUALITY)
            .encode_image(&canvas)
    {
        tracing::warn!(%error, "A picture could not be encoded");
        return None;
    }
    Some(Picture {
        data_uri: crate::util::jpeg_data_uri(&jpeg),
        size: canvas.dimensions(),
        frames: frames.iter().map(|(_, number, _)| *number).collect(),
    })
}

/// Which frames each picture holds, in the order the pictures are attached, and the grid's layout:
/// an arriving message has no tool description to carry it, so the line does.
fn pictures_attached(pictures: &[Picture]) -> String {
    let list = pictures
        .iter()
        .enumerate()
        .map(|(index, picture)| format!("picture {}: {}", index + 1, picture.frames_text()))
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "The frames are attached as {} {} of four, left to right then the next row — {list}.",
        pictures.len(),
        plural(pictures.len(), "picture", "pictures"),
    )
}

/// Frame numbers as one phrase, consecutive runs as ranges: `frames 1–4`, `frames 1–2, 4`.
fn frame_numbers(frames: &[usize]) -> String {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &frame in frames {
        match runs.last_mut() {
            Some((_, last)) if *last + 1 == frame => *last = frame,
            _ => runs.push((frame, frame)),
        }
    }
    let list = runs
        .iter()
        .map(|(first, last)| {
            if first == last {
                first.to_string()
            } else {
                format!("{first}–{last}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{} {list}", plural(frames.len(), "frame", "frames"))
}

/// Why a media read could not answer. The answers are deliberately different: a file that cannot be
/// decoded is not a missing tool, and a missing sound track is not a failed recognition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MediaError {
    /// The product's own half could not do it: its ffmpeg copy is missing or cannot be run, with
    /// the reason.
    Unavailable(String),
    /// The file cannot be read as media at all.
    Undecodable,
    /// The product asked its own media code for something it cannot do, with the reason: a bug on
    /// this side, never a fact about the file.
    Malformed(String),
    /// The file holds no picture and no sound track: it is timed, and there is nothing in it to
    /// read.
    NoTracks,
    /// The sound track is there and could not be recognised, with the engine's reason.
    SoundTrack(String),
    /// Another read of the same file is still transcribing it, and this one would not wait longer.
    Busy,
    /// The product is stopping: no pass will finish.
    Stopping,
    /// A decode ran past its bound.
    TimedOut,
}

impl MediaError {
    /// The failure's own detail, without the sentence [`MediaError::message`] wraps it in: a
    /// timeline line has its own words for it.
    fn detail(&self) -> String {
        match self {
            Self::SoundTrack(reason) => reason.clone(),
            other => other.message(),
        }
    }

    /// What a surface that produced a clip says when the clip could not be read — one sentence, so
    /// the arrival paths and the video tools cannot drift on their wording.
    pub(crate) fn unreadable(&self) -> String {
        format!("The clip could not be read: {}.", self.message())
    }

    /// What happened, in one sentence the model or the owner can act on.
    pub(crate) fn message(&self) -> String {
        match self {
            Self::Unavailable(reason) => {
                format!("the product's own ffmpeg is not usable ({reason})")
            }
            Self::Undecodable => "the file could not be decoded as media".to_string(),
            Self::Malformed(reason) => {
                format!("the product's own media call was malformed ({reason})")
            }
            Self::NoTracks => "the file holds neither a picture nor a sound track".to_string(),
            Self::SoundTrack(reason) => {
                format!("the sound track could not be recognised ({reason})")
            }
            Self::Busy => "another read is still transcribing this file".to_string(),
            // The engine's own words for the same condition, so the two can never drift.
            Self::Stopping => crate::speech::Interrupted.to_string(),
            Self::TimedOut => "the decode ran past its time bound".to_string(),
        }
    }
}

impl From<FfmpegError> for MediaError {
    fn from(error: FfmpegError) -> Self {
        match error {
            FfmpegError::Unavailable(reason) => Self::Unavailable(reason),
            // The probe decides whether there is a picture or a sound track before anything is
            // extracted, so a decode that reports either missing is a decode that failed.
            FfmpegError::NoVideoTrack | FfmpegError::NoAudioTrack | FfmpegError::Undecodable => {
                Self::Undecodable
            }
            // The options are the product's own: a file that decodes is not at fault for them.
            FfmpegError::InvalidOptions(reason) => Self::Malformed(reason.to_string()),
            FfmpegError::TimedOut => Self::TimedOut,
        }
    }
}

/// Whether a path names a container the product reads as media. The probe decides what the file
/// actually holds; this only decides whether asking it is worth a process.
pub(crate) fn is_media_path(path: &Path) -> bool {
    crate::util::is_video_extension(path) || crate::util::is_audio_extension(path)
}

/// Read one media file: sampled, transcribed and rendered for `window`.
///
/// The frames and the sound track are taken at once, and neither failure sinks the other: a clip
/// whose pictures cannot be decoded still answers with its transcript, and a track that cannot be
/// recognised still answers with its frames — each saying what happened.
pub(crate) async fn read(path: &Path, window: Window) -> Result<Reading, MediaError> {
    let info = ffmpeg::probe(path).await?;
    let kind = kind_of(&info)?;
    let duration = info.duration.as_secs_f64();
    let span = window.span(duration);
    // A read that begins past the file's own end has nothing to decode or transcribe: the probe
    // already answered, and the timeline says where the read starts — the second it asked for, not
    // the end it was clamped to.
    if let Some(start) = starts_late(window, duration) {
        return Ok(Reading {
            path: path.to_path_buf(),
            text: past_the_end(path, kind, &info, start),
            strip: None,
        });
    }
    // Boxed: the two passes hold a decode and a transcription apiece, and the read's own future
    // must stay small enough for the tool plumbing around it.
    let frame_pass = Box::pin(async {
        match kind {
            Kind::Video => Some(frames(path, window, info.picture_length().as_secs_f64()).await),
            Kind::Audio => None,
        }
    });
    let track = Box::pin(async {
        if info.has_audio {
            Some(transcript::regions(path, &info).await)
        } else {
            None
        }
    });
    let (frame_pass, track) = tokio::join!(frame_pass, track);
    let (strip, frame_failure) = match frame_pass {
        Some(Ok(strip)) => (strip, None),
        Some(Err(error)) => {
            tracing::warn!(
                path = %path.display(),
                reason = %error.message(),
                "A clip's frames could not be taken"
            );
            (None, Some(error))
        }
        None => (None, None),
    };
    let sound = match track {
        Some(transcribed) => match transcribed.failure {
            // Nothing at all was read: the track's failure is the read's own answer.
            Some(error) if transcribed.reached <= 0.0 => Sound::Failed(error),
            stopped => Sound::Heard {
                regions: transcribed.regions,
                reached: transcribed.reached,
                stopped,
            },
        },
        None => Sound::Absent,
    };
    let text = render(&Timeline {
        path,
        kind,
        info: &info,
        window,
        span,
        frames: strip.as_ref().map_or(&[], ffmpeg::Frames::frames),
        frame_failure: frame_failure.as_ref(),
        sound: &sound,
        // With transcription off the recogniser still runs — the sound labels come from the same
        // pass — and its words are what the timeline leaves out.
        words: !crate::audio::voice::is_transcription_disabled(),
    });
    Ok(Reading {
        path: path.to_path_buf(),
        text,
        strip,
    })
}

/// The arrival form of a reading: the timeline, the line naming which frames each picture holds,
/// then those pictures as the `[IMAGE:]` markers the model reads, in the order they are named. A
/// picture is an ordinary image from here on — the provider turns each marker into an image part
/// after the text, and the conversation keeps it like any other picture. Answers whether it
/// attached any.
pub(crate) async fn inline(reading: &Reading) -> (String, bool) {
    let pictures = reading.pictures().await;
    let mut text = reading.text.clone();
    if !pictures.is_empty() {
        let _ = write!(text, "\n{}", pictures_attached(&pictures));
        for picture in &pictures {
            let _ = write!(text, "\n[IMAGE:{}]", picture.data_uri);
        }
    }
    (text, !pictures.is_empty())
}

fn kind_of(info: &ClipInfo) -> Result<Kind, MediaError> {
    match (info.dimensions.is_some(), info.has_audio) {
        (true, _) => Ok(Kind::Video),
        (false, true) => Ok(Kind::Audio),
        (false, false) => Err(MediaError::NoTracks),
    }
}

/// A stretch of the file, in seconds.
#[derive(Debug, Clone, Copy)]
struct Span {
    start: f64,
    end: f64,
}

impl Span {
    fn length(self) -> f64 {
        (self.end - self.start).max(0.0)
    }
}

impl Window {
    /// The stretch whose transcript this window shows. An overview stops where its cap does.
    fn span(self, duration: f64) -> Span {
        match self {
            Self::Overview => Span {
                start: 0.0,
                end: duration.min(OVERVIEW_SECONDS),
            },
            Self::Span { start, length } => {
                let start = start.clamp(0.0, duration);
                let end = length.map_or(duration, |length| start + length.max(0.0));
                Span {
                    start,
                    end: end.min(duration),
                }
            }
        }
    }

    /// The stretch this window's frames are taken from: an overview spreads them across the whole
    /// clip, never across its capped transcript alone.
    fn frame_span(self, duration: f64) -> Span {
        match self {
            Self::Overview => Span {
                start: 0.0,
                end: duration,
            },
            Self::Span { .. } => self.span(duration),
        }
    }
}

/// How a read samples its frames: the seconds to take them at, and the side they fit.
struct FramePlan {
    marks: Vec<f64>,
    side: u32,
}

impl FramePlan {
    /// `picture` is how long the file's picture stream runs: frames are spread over that, never
    /// over a container that outlasts it.
    fn of(window: Window, picture: f64) -> Self {
        let span = window.frame_span(picture);
        let length = span.length();
        let (side, per_second, min) = match window {
            Window::Overview => (OVERVIEW_FRAME_SIDE, 1.0, OVERVIEW_FRAMES_MIN),
            Window::Span { .. } => (WINDOW_FRAME_SIDE, WINDOW_FRAMES_PER_SECOND, 1),
        };
        // Nothing to spread over — at or past the end of the file, or a length the probe could not
        // give: no window plans a frame there, and an overview plans the one instant the file has
        // rather than its floor of frames all at second zero.
        let empty = length <= 0.0;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a clip's length in whole frames fits u32 by the count's own ceiling"
        )]
        let count = if empty {
            0
        } else {
            ((length * per_second) as u32).clamp(min, FRAMES_MAX)
        };
        // The frames are spread evenly over the stretch, so a mark is the plan's own arithmetic.
        let step = if count == 0 {
            0.0
        } else {
            span.length() / f64::from(count)
        };
        let mut marks: Vec<f64> = (0..count)
            .map(|index| span.start + f64::from(index) * step)
            .collect();
        if marks.is_empty() && matches!(window, Window::Overview) {
            marks.push(span.start);
        }
        Self { marks, side }
    }
}

async fn frames(
    path: &Path,
    window: Window,
    picture: f64,
) -> Result<Option<ffmpeg::Frames>, MediaError> {
    // Frames are planned over the picture stream, never the container: a container that runs past
    // its last picture would otherwise ask for seeks that yield nothing.
    let plan = FramePlan::of(window, picture);
    if plan.marks.is_empty() {
        return Ok(None);
    }
    // A decoder can answer with fewer frames than planned (a mark the file has no picture at
    // writes nothing); what it wrote is what the timeline names, each with its own mark.
    Ok(Some(
        ffmpeg::extract_frames(
            path,
            ffmpeg::FrameOptions {
                marks: &plan.marks,
                max_side: plan.side,
            },
        )
        .await?,
    ))
}

/// What a read has to say about the sound track.
enum Sound {
    /// The recogniser's regions, the second the pass reached, and why it could not finish when it
    /// could not — a piece it could not read, or another read holding the file.
    Heard {
        regions: Vec<Segment>,
        reached: f64,
        stopped: Option<MediaError>,
    },
    /// The file carries no sound track.
    Absent,
    /// The track is there and nothing at all could be read from it, with the reason.
    Failed(MediaError),
}

/// One line of the timeline.
struct Line {
    start: f64,
    end: f64,
    text: String,
}

/// Everything the timeline is rendered from.
struct Timeline<'a> {
    path: &'a Path,
    kind: Kind,
    info: &'a ClipInfo,
    window: Window,
    /// The stretch the transcript covers.
    span: Span,
    /// The frames taken, each with the second it marks, in timeline order.
    frames: &'a [(PathBuf, f64)],
    /// Why the pictures could not be taken, when they could not: the transcript is still shown.
    frame_failure: Option<&'a MediaError>,
    sound: &'a Sound,
    /// Whether the recogniser's words are shown. With transcription off a speech region keeps its
    /// place as a sound label, and the labels fill in around it.
    words: bool,
}

/// The timeline's first line: what the file is, how long, and how many frames this read names.
fn header(path: &Path, kind: Kind, info: &ClipInfo, frames: usize) -> String {
    let size = info
        .dimensions
        .map(|(width, height)| format!(", {width}x{height}"))
        .unwrap_or_default();
    let duration = clock(info.duration.as_secs_f64());
    match kind {
        Kind::Video => format!(
            "Video {} — {duration}{size}, {frames} {}",
            path.display(),
            plural(frames, "frame", "frames"),
        ),
        Kind::Audio => format!("Audio {} — {duration}{size}", path.display()),
    }
}

/// The timeline of a read that begins past the file's own end: the header, then where it starts.
fn past_the_end(path: &Path, kind: Kind, info: &ClipInfo, start: f64) -> String {
    format!(
        "{}\nThe read starts at {}, past the end of the {} file.",
        header(path, kind, info, 0),
        clock(start),
        clock(info.duration.as_secs_f64()),
    )
}

/// The start second a read asks for, when it asks for one past the file's own end.
fn starts_late(window: Window, duration: f64) -> Option<f64> {
    match window {
        Window::Span { start, .. } if start >= duration && duration > 0.0 => Some(start),
        _ => None,
    }
}

/// Render the timeline: one strip in time order, the frames with their marks, the speech lines with
/// their marks and text, and a sound label only where there is no speech.
fn render(timeline: &Timeline<'_>) -> String {
    let mut out = header(
        timeline.path,
        timeline.kind,
        timeline.info,
        timeline.frames.len(),
    );
    if let Some(error) = timeline.frame_failure {
        let _ = write!(out, "\nThe frames could not be taken: {}.", error.detail());
    }
    match timeline.sound {
        Sound::Absent => out.push_str("\nThis video has no sound track."),
        // The recogniser's own failure is worded as one; anything else — a decode, another read
        // holding the file — says what it was instead of blaming the recogniser for it.
        Sound::Failed(error @ MediaError::SoundTrack(_)) => {
            let _ = write!(
                out,
                "\nThe sound track could not be recognised: {}.",
                error.detail()
            );
        }
        Sound::Failed(error) => {
            let _ = write!(
                out,
                "\nThe sound track could not be read: {}.",
                error.message()
            );
        }
        // A piece that could not be read cut this stretch short: the lines read before it are
        // still here, and this says where it stopped — and how much of the track that is, so the
        // coverage note below has nothing left to add.
        Sound::Heard {
            stopped: Some(error),
            ..
        } if !timeline.read_through() => {
            let _ = write!(
                out,
                "\nThe sound track could not be read past {} of {}: {}.",
                clock(timeline.heard()),
                clock(timeline.track()),
                error.detail()
            );
        }
        Sound::Heard { .. } => {}
    }

    let lines = timeline.lines();
    // A stretch the pass never reached is not a stretch that was silent.
    if timeline.sound.is_heard() && lines.is_empty() && timeline.read_through() {
        out.push_str("\nNothing was heard in this stretch.");
    }
    let mut frames = timeline
        .frames
        .iter()
        .map(|(path, at)| (path.as_path(), *at))
        .enumerate()
        .peekable();
    let mut lines = lines.into_iter().peekable();
    while frames.peek().is_some() || lines.peek().is_some() {
        let take_frame = match (frames.peek(), lines.peek()) {
            (Some((_, (_, at))), Some(line)) => *at <= line.start,
            (Some(_), None) => true,
            (None, _) => false,
        };
        if take_frame {
            if let Some((index, (_, at))) = frames.next() {
                let _ = write!(out, "\n{} [frame {}]", mark(at), index + 1);
            }
        } else if let Some(line) = lines.next() {
            // A stretch shorter than the clock's own resolution gets one mark, not an empty range.
            let (start, end) = (clock(line.start), clock(line.end));
            if start == end {
                let _ = write!(out, "\n{start} {}", line.text);
            } else {
                let _ = write!(out, "\n{start}–{end} {}", line.text);
            }
        }
    }
    if let Some(cap) = timeline.coverage_note() {
        let _ = write!(out, "\n{cap}");
    }
    out
}

impl Timeline<'_> {
    /// The second the transcript reaches: the sound track's end when the pass finished, less when
    /// it stopped early. A file nothing was heard from reaches no further than the read's start.
    fn heard(&self) -> f64 {
        match self.sound {
            Sound::Heard { reached, .. } => *reached,
            Sound::Absent | Sound::Failed(_) => self.span.start,
        }
    }

    /// The timeline's speech and sound lines, in time order.
    fn lines(&self) -> Vec<Line> {
        let Sound::Heard { regions, .. } = self.sound else {
            return Vec::new();
        };
        let mut lines: Vec<Line> = Vec::new();
        for region in regions {
            let start = region.start.max(self.span.start);
            let end = region.end.min(self.span.end);
            if end <= start {
                continue;
            }
            let text = match (&region.text, region.label.as_deref()) {
                // The words are read but not shown: the region keeps its place as the sound it
                // is, so a stretch of speech neither looks silent nor claims it went unread.
                (Some(_), _) if !self.words => "(Speech)".to_string(),
                (Some(text), _) => text.trim().to_string(),
                (None, Some(label)) if crate::speech::is_speech_label(label) => {
                    crate::speech::UNTRANSCRIBED.to_string()
                }
                // Silence is not a sound: it says nothing the marks do not already say.
                (None, Some(label)) if crate::speech::is_silence_label(label) => continue,
                (None, Some(label)) => format!("({label})"),
                // The engine labels every region no reading produced, so a region with neither is
                // the one pair its contract rules out — nothing to say about it either way.
                (None, None) => continue,
            };
            // A run the recogniser cut into pieces is one line, so a label never repeats itself.
            match lines.last_mut() {
                Some(last) if last.text == text && last.end >= start => last.end = end,
                _ => lines.push(Line { start, end, text }),
            }
        }
        lines
    }

    /// The note a transcript that stops short of what there is to read carries, so a timeline never
    /// implies the rest is silent: an overview capped at [`OVERVIEW_SECONDS`], or a bounded pass
    /// that ran out of time before the stretch it was asked for ended. The cache covers from the
    /// start, so the range always begins there. Everything is measured against the sound track,
    /// never the container, which can run past it.
    ///
    /// A piece that could not be read is the exception: its own line already says where the pass
    /// stopped and how long the track is, and reading further cannot get past it.
    fn coverage_note(&self) -> Option<String> {
        if !self.sound.is_heard() || self.sound.stopped() {
            return None;
        }
        // What there is to read in this window, and what there is in the whole track.
        let track = self.track();
        let available = self.span.end.min(track);
        let capped =
            matches!(self.window, Window::Overview) && self.span.end < track - COVERAGE_SLACK;
        (!self.read_through() || capped).then(|| {
            format!(
                "The transcript covers 0:00–{} of {} — read further with offset/limit.",
                clock(self.heard().min(available)),
                clock(track)
            )
        })
    }

    /// Whether the pass reached the end of what this window covers: the stretch a timeline is
    /// asked about is only silent when it was actually read.
    fn read_through(&self) -> bool {
        self.heard() >= self.span.end.min(self.track()) - COVERAGE_SLACK
    }

    /// The sound track's own length, which can be less than the container's.
    fn track(&self) -> f64 {
        self.info.sound_length().as_secs_f64()
    }
}

impl Sound {
    fn is_heard(&self) -> bool {
        matches!(self, Self::Heard { .. })
    }

    /// Whether something the pass could not get past cut the track short.
    fn stopped(&self) -> bool {
        matches!(
            self,
            Self::Heard {
                stopped: Some(_),
                ..
            }
        )
    }
}

fn plural(count: usize, one: &'static str, many: &'static str) -> &'static str {
    if count == 1 { one } else { many }
}

/// A second of the file as a clock reads it: `0:07`, `1:02:03`.
fn clock(seconds: f64) -> String {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a clip's length in whole seconds fits u64"
    )]
    spelled(seconds.max(0.0) as u64)
}

/// A frame's mark: [`clock`] with the tenth of a second added where the whole second alone cannot
/// tell two frames apart — a window takes two a second.
fn mark(seconds: f64) -> String {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a clip's length in tenths of a second fits u64"
    )]
    let tenths = (seconds.max(0.0) * 10.0).round() as u64;
    let whole = spelled(tenths / 10);
    match tenths % 10 {
        0 => whole,
        part => format!("{whole}.{part}"),
    }
}

/// `total` seconds as a clock reads them.
fn spelled(total: u64) -> String {
    let (hours, minutes, seconds) = (total / 3600, total % 3600 / 60, total % 60);
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// A clip of `seconds` at 30 fps whose probe the tests hand-build.
    fn info(seconds: u64, dimensions: Option<(u32, u32)>, has_audio: bool) -> ClipInfo {
        ClipInfo {
            duration: Duration::from_secs(seconds),
            dimensions,
            has_audio,
            picture: dimensions.map(|_| Duration::from_secs(seconds)),
            sound: has_audio.then_some(Duration::from_secs(seconds)),
            bytes: 1024,
        }
    }

    #[test]
    fn an_overview_spreads_its_frames_across_the_whole_clip() {
        // One per second of a short clip, its floor at eight.
        let plan = FramePlan::of(Window::Overview, 3.0);
        assert_eq!(plan.marks.len(), OVERVIEW_FRAMES_MIN as usize);
        assert_eq!(plan.side, OVERVIEW_FRAME_SIDE);

        // A long clip keeps the ceiling and spaces the frames over all of it.
        let plan = FramePlan::of(Window::Overview, 2530.0);
        assert_eq!(plan.marks.len(), FRAMES_MAX as usize);
        assert!(plan.marks[0] < 1.0);
        assert!(plan.marks[23] > 2400.0);
    }

    #[test]
    fn a_window_samples_its_own_stretch() {
        let window = Window::Span {
            start: 20.0,
            length: Some(5.0),
        };
        let plan = FramePlan::of(window, 60.0);
        assert_eq!(plan.marks.len(), 10);
        assert_eq!(plan.side, WINDOW_FRAME_SIDE);
        assert!((plan.marks[0] - 20.0).abs() < 1e-9);
        assert!((plan.marks[9] - 24.5).abs() < 1e-9);

        // A window past the last picture plans nothing: the file holds no picture there, which is
        // not a decode that failed.
        let past_the_pictures = FramePlan::of(
            Window::Span {
                start: 7.0,
                length: Some(3.0),
            },
            5.0,
        );
        assert!(past_the_pictures.marks.is_empty());
    }

    #[test]
    fn a_window_longer_than_the_ceiling_still_spreads() {
        let window = Window::Span {
            start: 0.0,
            length: Some(600.0),
        };
        let plan = FramePlan::of(window, 900.0);
        // Two per second would be 1200 frames; the ceiling keeps 24 across the whole window.
        assert_eq!(plan.marks.len(), FRAMES_MAX as usize);
        assert!((plan.marks[1] - 25.0).abs() < 1e-9);
    }

    #[test]
    fn an_overview_transcript_stops_at_ten_minutes_while_its_frames_cover_the_clip() {
        let span = Window::Overview.span(2530.0);
        assert!((span.end - OVERVIEW_SECONDS).abs() < f64::EPSILON);
        // The frames still cover the whole clip.
        assert!((Window::Overview.frame_span(2530.0).end - 2530.0).abs() < f64::EPSILON);
    }

    #[test]
    fn every_failure_says_something_different() {
        let messages = [
            MediaError::Unavailable("not installed".to_string()),
            MediaError::Undecodable,
            MediaError::Malformed("no frames were asked for".to_string()),
            MediaError::NoTracks,
            MediaError::SoundTrack("timed out".to_string()),
            MediaError::Busy,
            MediaError::Stopping,
            MediaError::TimedOut,
        ]
        .map(|error| error.message());
        let distinct: std::collections::HashSet<&String> = messages.iter().collect();
        assert_eq!(distinct.len(), messages.len(), "{messages:?}");
    }

    #[test]
    fn a_decode_failure_is_not_a_failed_recognition() {
        let clip = info(60, Some((640, 360)), true);
        let undecodable = rendered(
            &clip,
            Window::Overview,
            &[],
            &Sound::Failed(MediaError::Undecodable),
            true,
        );
        assert!(
            undecodable.ends_with(
                "The sound track could not be read: the file could not be decoded as media."
            ),
            "{undecodable}"
        );
        let unrecognised = rendered(
            &clip,
            Window::Overview,
            &[],
            &Sound::Failed(MediaError::SoundTrack("timed out".to_string())),
            true,
        );
        assert!(
            unrecognised.ends_with("The sound track could not be recognised: timed out."),
            "{unrecognised}"
        );
    }

    #[test]
    fn a_mark_reads_as_a_clock() {
        assert_eq!(clock(0.0), "0:00");
        assert_eq!(clock(67.9), "1:07");
        assert_eq!(clock(3723.0), "1:02:03");
        // A frame's mark adds the tenth that tells two frames of one second apart, and rounds
        // rather than truncates so the clock never lags the mark.
        assert_eq!(mark(30.0), "0:30");
        assert_eq!(mark(5.5), "0:05.5");
        assert_eq!(mark(59.96), "1:00");
    }

    /// Frames as the renderer sees them: the paths and the marks, with no extraction behind them.
    fn frame_strip(marks: &[f64]) -> Vec<(PathBuf, f64)> {
        marks
            .iter()
            .map(|at| (PathBuf::from(format!("frame-{at}.jpg")), *at))
            .collect()
    }

    fn heard(start: f64, end: f64, text: Option<&str>, label: Option<&str>) -> Segment {
        Segment {
            start,
            end,
            text: text.map(str::to_string),
            label: label.map(str::to_string),
        }
    }

    fn rendered(
        clip: &ClipInfo,
        window: Window,
        strip: &[(PathBuf, f64)],
        sound: &Sound,
        words: bool,
    ) -> String {
        let span = window.span(clip.duration.as_secs_f64());
        render(&Timeline {
            path: Path::new("/clip.mp4"),
            kind: Kind::Video,
            info: clip,
            window,
            span,
            frames: strip,
            frame_failure: None,
            sound,
            words,
        })
    }

    #[test]
    fn the_timeline_is_one_strip_in_time_order() {
        let clip = info(60, Some((640, 360)), true);
        let strip = frame_strip(&[0.0, 30.0]);
        let sound = Sound::Heard {
            regions: vec![
                heard(0.0, 5.0, Some("Hello there."), None),
                heard(5.0, 12.0, None, Some("Music")),
                heard(12.0, 20.0, None, Some("Speech")),
                heard(20.0, 30.0, None, Some("Silence")),
                heard(30.0, 60.0, None, Some("Silence")),
            ],
            stopped: None,
            reached: 60.0,
        };
        assert_eq!(
            rendered(&clip, Window::Overview, &strip, &sound, true),
            "Video /clip.mp4 — 1:00, 640x360, 2 frames\n\
             0:00 [frame 1]\n\
             0:00–0:05 Hello there.\n\
             0:05–0:12 (Music)\n\
             0:12–0:20 [untranscribed speech]\n\
             0:30 [frame 2]"
        );
    }

    #[test]
    fn with_transcription_off_speech_keeps_its_place_and_the_labels_fill_in() {
        // The recogniser read these words; the setting hides them. The stretch must not read as
        // silence, and the labels around it must still show.
        let clip = info(60, Some((640, 360)), true);
        let sound = Sound::Heard {
            regions: vec![
                heard(0.0, 5.0, Some("Hello there."), None),
                heard(5.0, 12.0, None, Some("Music")),
            ],
            stopped: None,
            reached: 60.0,
        };
        let text = rendered(&clip, Window::Overview, &frame_strip(&[0.0]), &sound, false);
        assert!(text.contains("\n0:00 [frame 1]\n"), "{text}");
        assert!(text.contains("\n0:00–0:05 (Speech)\n"), "{text}");
        assert!(text.contains("\n0:05–0:12 (Music)"), "{text}");
        assert!(
            !text.contains("Nothing was heard in this stretch."),
            "{text}"
        );
        assert!(!text.contains("Hello there."), "{text}");
    }

    #[test]
    fn a_capped_overview_says_how_much_it_covered_and_a_silent_clip_says_so() {
        let clip = info(2530, Some((640, 360)), true);
        let strip = frame_strip(&[0.0]);
        let sound = Sound::Heard {
            regions: vec![
                heard(0.0, 10.0, Some("Hi."), None),
                heard(10.0, 2530.0, None, Some("Music")),
            ],
            stopped: None,
            reached: 2530.0,
        };
        let text = rendered(&clip, Window::Overview, &strip, &sound, true);
        assert!(
            text.ends_with(
                "The transcript covers 0:00–10:00 of 42:10 — read further with offset/limit."
            ),
            "{text}"
        );

        let text = rendered(&clip, Window::Overview, &strip, &Sound::Absent, true);
        assert!(text.contains("\nThis video has no sound track."), "{text}");
    }

    #[test]
    fn a_pass_that_stopped_early_says_where_it_stopped() {
        let clip = info(60, Some((640, 360)), true);
        let strip = frame_strip(&[0.0]);
        let sound = Sound::Heard {
            regions: vec![heard(0.0, 10.0, Some("Hi."), None)],
            stopped: None,
            reached: 10.0,
        };
        let text = rendered(&clip, Window::Overview, &strip, &sound, true);
        assert!(
            text.ends_with(
                "The transcript covers 0:00–0:10 of 1:00 — read further with offset/limit."
            ),
            "{text}"
        );
    }

    #[test]
    fn a_stretch_the_pass_never_reached_is_not_called_silent() {
        let clip = info(60, Some((640, 360)), true);
        let stopped_short = Sound::Heard {
            regions: Vec::new(),
            stopped: None,
            reached: 10.0,
        };
        let text = rendered(
            &clip,
            Window::Overview,
            &frame_strip(&[0.0]),
            &stopped_short,
            true,
        );
        assert!(
            !text.contains("Nothing was heard in this stretch."),
            "{text}"
        );

        // Read to the end and holding no sound at all: that is silence, and it says so.
        let silent = Sound::Heard {
            regions: Vec::new(),
            stopped: None,
            reached: 60.0,
        };
        let text = rendered(&clip, Window::Overview, &frame_strip(&[0.0]), &silent, true);
        assert!(
            text.contains("Nothing was heard in this stretch."),
            "{text}"
        );
    }

    #[test]
    fn a_track_that_could_not_be_read_to_the_end_keeps_its_lines_and_says_where() {
        let clip = info(2530, Some((640, 360)), true);
        let sound = Sound::Heard {
            regions: vec![heard(0.0, 300.0, Some("Hi."), None)],
            stopped: Some(MediaError::SoundTrack(
                "the piece could not be decoded".to_string(),
            )),
            reached: 300.0,
        };
        let text = rendered(&clip, Window::Overview, &frame_strip(&[0.0]), &sound, true);
        assert!(text.contains("\n0:00–5:00 Hi."), "{text}");
        // One statement of where the pass stopped and how much of the track that is: reading
        // further cannot get past that piece, so no coverage note advises it.
        assert!(
            text.contains(
                "The sound track could not be read past 5:00 of 42:10: the piece could not be \
                 decoded."
            ),
            "{text}"
        );
        assert!(!text.contains("The transcript covers"), "{text}");
        assert!(
            !text.contains("Nothing was heard in this stretch."),
            "{text}"
        );
    }

    #[test]
    fn a_track_that_stops_before_its_container_needs_no_note() {
        // The container runs a minute, its sound track twenty seconds: reading further cannot
        // reach anything, so the timeline says nothing about coverage.
        let mut clip = info(60, Some((640, 360)), true);
        clip.sound = Some(Duration::from_secs(20));
        let sound = Sound::Heard {
            regions: vec![heard(0.0, 20.0, None, Some("Music"))],
            stopped: None,
            reached: 20.0,
        };
        let text = rendered(&clip, Window::Overview, &frame_strip(&[0.0]), &sound, true);
        assert!(!text.contains("The transcript covers"), "{text}");
    }

    /// One solid-colour frame, JPEG-encoded as ffmpeg writes them.
    fn frame_jpeg(colour: [u8; 3]) -> Vec<u8> {
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, ffmpeg::FRAME_QUALITY)
            .encode_image(&image::RgbImage::from_pixel(4, 2, image::Rgb(colour)))
            .unwrap();
        jpeg
    }

    /// The cells of one picture: each frame's cell index, its number and its bytes.
    fn cells(frames: &[Vec<u8>]) -> Vec<(usize, usize, Vec<u8>)> {
        frames
            .iter()
            .enumerate()
            .map(|(index, jpeg)| (index, index + 1, jpeg.clone()))
            .collect()
    }

    /// The cell a pixel belongs to, by its colour: four frames into one picture in reading order,
    /// the cells of a group holding fewer than four left blank.
    #[test]
    fn four_frames_go_into_one_picture_in_reading_order() {
        let colours = [[255, 0, 0], [0, 255, 0], [0, 0, 255]];
        let picture = compose(&cells(&colours.map(frame_jpeg))).unwrap();
        assert_eq!(picture.size, (8, 4));
        assert_eq!(picture.frames_text(), "frames 1–3");
        let image = image::load_from_memory(&uri_bytes(&picture.data_uri))
            .unwrap()
            .to_rgb8();
        // Reading order: left to right, then the next row; the fourth cell stays blank.
        for (index, colour) in colours.iter().enumerate() {
            let (column, row) = (
                u32::try_from(index % 2).unwrap(),
                u32::try_from(index / 2).unwrap(),
            );
            assert!(
                near(image.get_pixel(column * 4 + 1, row * 2 + 1).0, *colour),
                "cell {index}: {:?}",
                image.get_pixel(column * 4 + 1, row * 2 + 1).0
            );
        }
        assert!(
            near(image.get_pixel(5, 3).0, [0, 0, 0]),
            "{:?}",
            image.get_pixel(5, 3).0
        );
    }

    /// A frame that cannot be decoded leaves its own cell blank rather than dropping the picture's
    /// other moments.
    #[test]
    fn a_frame_that_cannot_be_decoded_leaves_its_cell_blank() {
        let cells = [
            (0, 1, frame_jpeg([255, 0, 0])),
            (1, 2, b"not a JPEG".to_vec()),
            (2, 3, frame_jpeg([0, 255, 0])),
        ];
        let picture = compose(&cells).unwrap();
        assert_eq!(picture.frames_text(), "frames 1, 3");
        let image = image::load_from_memory(&uri_bytes(&picture.data_uri))
            .unwrap()
            .to_rgb8();
        assert!(near(image.get_pixel(1, 1).0, [255, 0, 0]));
        assert!(near(image.get_pixel(5, 1).0, [0, 0, 0]));
        assert!(near(image.get_pixel(1, 3).0, [0, 255, 0]));
    }

    /// Whether two colours survive a JPEG round trip as the same colour: the encoder is lossy and
    /// subsamples chroma, so a solid cell comes back a few levels off.
    fn near(actual: [u8; 3], expected: [u8; 3]) -> bool {
        actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.abs_diff(expected) < 40)
    }

    /// The bytes behind a data URI the composer produced.
    fn uri_bytes(uri: &str) -> Vec<u8> {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD
            .decode(uri.split_once(',').expect("a data URI").1)
            .unwrap()
    }

    #[test]
    fn the_attached_line_names_the_frames_each_picture_holds() {
        let picture = |frames: &[usize]| Picture {
            data_uri: String::new(),
            size: (0, 0),
            frames: frames.to_vec(),
        };
        assert_eq!(
            pictures_attached(&[
                picture(&[1, 2, 3, 4]),
                picture(&[5, 6, 7, 8]),
                picture(&[9, 10])
            ]),
            "The frames are attached as 3 pictures of four, left to right then the next row — \
             picture 1: frames 1–4; picture 2: frames 5–8; picture 3: frames 9–10."
        );
        assert_eq!(
            pictures_attached(&[picture(&[7])]),
            "The frames are attached as 1 picture of four, left to right then the next row — \
             picture 1: frame 7."
        );
    }
}
