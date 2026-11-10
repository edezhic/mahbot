//! The product's own ffmpeg and ffprobe: the copy the product installs itself (`install`) and
//! the operations that copy serves.
//!
//! A copy the owner installed — a package manager's, a vendor download, anything on the search
//! path — is never run: every command here is the absolute path of the product's own build, and
//! no `PATH` lookup is ever made.
//!
//! The operations are exactly four: [`probe`] (duration, upright dimensions, audio presence, the
//! file's own size), [`extract_frames`] (one JPEG at each mark asked for, each fitting a bounded
//! side, written at one quality), [`extract_audio`] (a stretch of the track re-encoded to 16 kHz
//! mono WAV, what the recogniser takes) and [`convert_audio`] (the whole of a recording the product
//! has no decoder for, to the same form). A stretch is asked for as input options, so it is
//! seeked to rather than decoded up to.
//!
//! Each runs under its own [`Duration`], and every output lands in a [`tempfile::TempDir`] under the
//! product's private temp root, never in the owner's workspace, carried by the returned guard, so
//! dropping the result removes the files. A first call installs the pair first, under the download
//! client's own bounds — a copy that cannot be obtained is a missing tool, not a decode that ran
//! out of time. [`convert_audio`] is the exception: it runs only the copy already in place, so a
//! transcription never downloads a toolchain.
//!
//! Failures are typed and deliberately distinct: [`FfmpegError::Unavailable`] is the product's own
//! half not being usable, while [`FfmpegError::Undecodable`], [`FfmpegError::NoVideoTrack`] and
//! [`FfmpegError::NoAudioTrack`] are facts about the file. None of that can be read off the tools'
//! exit statuses, so a failed decode is settled by asking the file again.

mod install;
mod probe;

pub use install::install_on_start;
pub(crate) use install::installed;

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tokio::process::Command;

use install::Binaries;

/// How long reading a file's facts may take.
const PROBE_TIMEOUT: Duration = Duration::from_secs(30);

/// How long the whole frame extraction may take: one seek and one picture per mark.
const FRAMES_TIMEOUT: Duration = Duration::from_secs(120);

/// How long extracting the audio may take: a long recording is decoded in full.
const AUDIO_TIMEOUT: Duration = Duration::from_secs(300);

/// What `ffprobe` reports, and the only entries the parser reads.
const PROBE_ENTRIES: &str = "format=duration:stream=codec_type,width,height,duration\
                            :stream_tags=rotate:stream_side_data=rotation\
                            :stream_disposition=attached_pic";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClipInfo {
    pub(crate) duration: Duration,
    /// The upright size of the first video stream — what a person watching the file sees.
    pub(crate) dimensions: Option<(u32, u32)>,
    pub(crate) has_audio: bool,
    /// The picture stream's own length, when the container declares one: it can stop before its
    /// container does, and a frame asked for past its end is a seek that yields nothing.
    pub(crate) picture: Option<Duration>,
    /// The sound track's own length, when the container declares one: a track can stop before its
    /// container does, and a piece cut past its end holds nothing to read.
    pub(crate) sound: Option<Duration>,
    pub(crate) bytes: u64,
}

impl ClipInfo {
    /// The sound track's own length — never the container's, which can run past the track: a piece
    /// cut past the track's end holds nothing to read, and a transcript cannot claim to reach it.
    #[must_use]
    pub(crate) fn sound_length(&self) -> Duration {
        self.sound.unwrap_or(self.duration)
    }

    /// The picture stream's own length — never the container's, which can run past it: frames are
    /// planned over the pictures, and a mark past the last one costs a seek and yields nothing.
    #[must_use]
    pub(crate) fn picture_length(&self) -> Duration {
        self.picture.unwrap_or(self.duration)
    }
}

/// Why an operation could not answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FfmpegError {
    /// The product's own half could not do it: its copy cannot be run (not installed, or a start
    /// failure), with the reason.
    Unavailable(String),
    NoVideoTrack,
    NoAudioTrack,
    Undecodable,
    TimedOut,
    InvalidOptions(&'static str),
}

/// Frames extracted from one file, each with the second it was taken at, removed when this guard
/// is dropped.
#[derive(Debug)]
pub(crate) struct Frames {
    _dir: tempfile::TempDir,
    frames: Vec<(PathBuf, f64)>,
}

impl Frames {
    /// The frames in playback order, each with its mark. Empty when the file itself carries none.
    #[must_use]
    pub(crate) fn frames(&self) -> &[(PathBuf, f64)] {
        &self.frames
    }
}

/// One file's audio as 16 kHz mono WAV, removed when this guard is dropped.
#[derive(Debug)]
pub(crate) struct Audio {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

impl Audio {
    #[must_use]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FrameOptions<'a> {
    /// The seconds to take a frame at, in playback order.
    pub(crate) marks: &'a [f64],
    /// The longest side of a written frame, in pixels. The aspect ratio is kept.
    pub(crate) max_side: u32,
}

/// The JPEG quality every frame is written at, on ffmpeg's 1–100 scale. A frame is read rather
/// than displayed, which is why 80 is the quality asked for.
pub(crate) const FRAME_QUALITY: u8 = 80;

/// The mjpeg quantiser for [`FRAME_QUALITY`], on ffmpeg's 2 (best) to 31 (worst) scale — inverted,
/// so 100 lands on 2 and 0 on 31.
const FRAME_QUANTISER: u32 = 31 - FRAME_QUALITY as u32 * 29 / 100;

/// The product's own pair, installed on first use — the crate's one entry to them.
pub(crate) async fn binaries() -> Result<Binaries, String> {
    install::ensure_installed().await
}

/// Duration, upright dimensions, audio presence and size of one file.
pub(crate) async fn probe(path: &Path) -> Result<ClipInfo, FfmpegError> {
    let bin = binaries().await.map_err(FfmpegError::Unavailable)?;
    // `probe_with` carries the bound itself, as every tool run does.
    probe_with(&bin.ffprobe, path).await
}

/// One JPEG per mark — the frame at that second, fitting the bounded side — each taken by seeking
/// to its mark, so a long clip costs a seek apiece rather than a decode of everything before it.
pub(crate) async fn extract_frames(
    input: &Path,
    options: FrameOptions<'_>,
) -> Result<Frames, FfmpegError> {
    if options.marks.is_empty() {
        return Err(FfmpegError::InvalidOptions("no frames were asked for"));
    }
    if options.max_side < 2 {
        return Err(FfmpegError::InvalidOptions(
            "the longest side must be at least two pixels",
        ));
    }

    let bin = binaries().await.map_err(FfmpegError::Unavailable)?;
    let dir = scratch()?;
    let deadline = Instant::now() + FRAMES_TIMEOUT;
    let mut frames = Vec::with_capacity(options.marks.len());
    let mut ran_out = false;
    for (index, at) in options.marks.iter().enumerate() {
        let into = dir.path().join(format!("frame-{index:05}.jpg"));
        // The bound is the whole extraction's, shared out. Running out of it keeps the frames
        // already taken: a long clip is worth what it managed, not nothing.
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            ran_out = true;
            break;
        }
        let args = frame_args(input, *at, Some(&into), options.max_side);
        match run_tool(&bin.ffmpeg, &args, left).await {
            Ok(output) if output.success && tokio::fs::try_exists(&into).await.unwrap_or(false) => {
                frames.push((into, *at));
            }
            // A mark the file has no picture at — past the last frame, or a picture stream that
            // stops before its container does — is not a frame, and is not the read's failure.
            Ok(output) => {
                tracing::debug!("ffmpeg wrote no frame at {at}: {}", output.stderr.trim());
            }
            Err(FfmpegError::TimedOut) => {
                ran_out = true;
                break;
            }
            Err(error) => return Err(error),
        }
    }
    // The bound is the product's own limit, not a fact about the file: what it cut short is the
    // answer, and nothing taken at all is the timeout it is.
    if ran_out {
        return if frames.is_empty() {
            Err(FfmpegError::TimedOut)
        } else {
            Ok(Frames { _dir: dir, frames })
        };
    }
    // Nothing at all is settled against the file: a clip that cannot be decoded is not the same
    // answer as one that is merely shorter than its container. The first mark stands for them all
    // — every one was tried and wrote nothing, and the replay is a seek like any other.
    if frames.is_empty() {
        let replay = Decode::Frames {
            at: options.marks[0],
            side: options.max_side,
        };
        return Err(classify(&bin, input, replay).await);
    }
    Ok(Frames { _dir: dir, frames })
}

/// The audio track of `input` — the whole file, or `window` of it — as 16 kHz mono WAV, through the
/// pair installed on first use.
pub(crate) async fn extract_audio(
    input: &Path,
    window: Option<(Duration, Duration)>,
) -> Result<Audio, FfmpegError> {
    let bin = binaries().await.map_err(FfmpegError::Unavailable)?;
    audio_with(&bin, input, window).await
}

/// The whole of `input` as 16 kHz mono WAV, through the copy already in place — never one fetched:
/// asking for a transcription is not a reason to install the toolchain, so a missing copy is
/// [`FfmpegError::Unavailable`] rather than an install.
pub(crate) async fn convert_audio(input: &Path) -> Result<Audio, FfmpegError> {
    let Some(bin) = installed() else {
        return Err(FfmpegError::Unavailable(
            "no copy of the product's media toolchain is installed".to_string(),
        ));
    };
    audio_with(&bin, input, None).await
}

/// The sound track of `input` as 16 kHz mono WAV: the whole file, or `window` of it.
async fn audio_with(
    bin: &Binaries,
    input: &Path,
    window: Option<(Duration, Duration)>,
) -> Result<Audio, FfmpegError> {
    // The bound covers the classification too.
    tokio::time::timeout(AUDIO_TIMEOUT, async {
        let dir = scratch()?;
        let path = dir.path().join("audio.wav");
        let args = audio_args(input, window, Some(&path));
        let output = run_tool(&bin.ffmpeg, &args, AUDIO_TIMEOUT).await?;
        if !output.success {
            tracing::debug!("ffmpeg extracted no audio: {}", output.stderr.trim());
            return Err(classify(bin, input, Decode::Audio(window)).await);
        }
        Ok(Audio { _dir: dir, path })
    })
    .await
    .map_err(|_| FfmpegError::TimedOut)?
}

/// `ffprobe`'s answer for one file, or [`FfmpegError::Undecodable`] when the file cannot be read.
///
/// The file's own size is the one fact `ffprobe` does not report, so it is taken from the file
/// itself: a file that cannot be stat'ed is a file that cannot be read.
async fn probe_with(ffprobe: &Path, path: &Path) -> Result<ClipInfo, FfmpegError> {
    let output = run_tool(ffprobe, &probe_args(path), PROBE_TIMEOUT).await?;
    let Some(mut info) = probe::parse(&output.stdout) else {
        tracing::debug!("ffprobe could not read the file: {}", output.stderr.trim());
        return Err(FfmpegError::Undecodable);
    };
    info.bytes = tokio::fs::metadata(path)
        .await
        .map_err(|_| FfmpegError::Undecodable)?
        .len();
    Ok(info)
}

/// The decode an operation was attempting, so a failure can be replayed as it was.
#[derive(Debug, Clone, Copy)]
enum Decode {
    /// One frame at `at` seconds, fitting `side` pixels.
    Frames { at: f64, side: u32 },
    /// The audio track, over the stretch it was asked for.
    Audio(Option<(Duration, Duration)>),
}

/// Why a decode failed: a file the tools cannot read at all is undecodable, a readable file that
/// simply lacks the track the operation needed says so, and a file that decodes fine means the
/// product's own output was what failed.
///
/// The order matters, and none of it can be read off `ffmpeg`'s exit status. The file is asked
/// instead: its tracks first, then the operation's own decode replayed over the same stretch with
/// nowhere to write — so a file that survives exactly what just failed on it failed on the
/// product's side.
async fn classify(bin: &Binaries, input: &Path, decode: Decode) -> FfmpegError {
    let info = match probe_with(&bin.ffprobe, input).await {
        Err(error) => return error,
        Ok(info) => info,
    };
    let (replay, timeout) = match decode {
        Decode::Frames { .. } if info.dimensions.is_none() => return FfmpegError::NoVideoTrack,
        Decode::Audio(_) if !info.has_audio => return FfmpegError::NoAudioTrack,
        Decode::Frames { at, side } => (frame_args(input, at, None, side), FRAMES_TIMEOUT),
        Decode::Audio(window) => (audio_args(input, window, None), AUDIO_TIMEOUT),
    };
    match run_tool(&bin.ffmpeg, &replay, timeout).await {
        Ok(output) if output.success => {
            FfmpegError::Unavailable("the product's own output could not be written".to_string())
        }
        Ok(_) => FfmpegError::Undecodable,
        Err(error) => error,
    }
}

struct Output {
    success: bool,
    stdout: Vec<u8>,
    stderr: String,
}

/// Run one of the product's own tools to completion or to its bound.
///
/// Nothing is inherited: stdin is null, so a tool can neither read the daemon's terminal nor wait
/// on it, and its output is captured, so a decode can never print into the daemon's streams. A
/// child that outlives the bound is killed by dropping the future (`kill_on_drop`).
///
/// A child killed by a signal says nothing about the file it was reading, so it is not reported
/// as a finished run. Signals are a Unix notion — Windows reports a code for every death — so the
/// check never fires there.
async fn run_tool(
    program: &Path,
    args: &[OsString],
    timeout: Duration,
) -> Result<Output, FfmpegError> {
    let mut command = Command::new(program);
    // A console program started by a console-less parent gets a visible console of
    // its own; the flag is what keeps a probe from flashing a window on Windows.
    #[cfg(target_os = "windows")]
    command.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    command
        .args(args)
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true);
    let finished = tokio::time::timeout(timeout, command.output())
        .await
        .map_err(|_| FfmpegError::TimedOut)?
        .map_err(|e| {
            FfmpegError::Unavailable(format!(
                "the product's own {} could not be started ({})",
                tool_name(program),
                e.kind()
            ))
        })?;
    if finished.status.code().is_none() {
        return Err(FfmpegError::Unavailable(format!(
            "the product's own {} was stopped before it finished",
            tool_name(program)
        )));
    }
    Ok(Output {
        success: finished.status.success(),
        stdout: finished.stdout,
        stderr: String::from_utf8_lossy(&finished.stderr).into_owned(),
    })
}

/// The tool's own name, for a reason a reader can act on: `ffprobe` failing to start is not
/// `ffmpeg` failing to start.
fn tool_name(program: &Path) -> String {
    program
        .file_stem()
        .and_then(std::ffi::OsStr::to_str)
        .map_or_else(|| "tool".to_string(), str::to_owned)
}

/// A directory under the product's private temp root, removed when the guard is dropped.
fn scratch() -> Result<tempfile::TempDir, FfmpegError> {
    tempfile::tempdir().map_err(|e| {
        FfmpegError::Unavailable(format!(
            "the product's own scratch directory could not be created ({})",
            e.kind()
        ))
    })
}

fn probe_args(input: &Path) -> Vec<OsString> {
    vec![
        "-v".into(),
        "error".into(),
        "-print_format".into(),
        "json".into(),
        "-show_entries".into(),
        PROBE_ENTRIES.into(),
        input.into(),
    ]
}

/// The stretch to sample, as input options before `-i`: the decoder starts where it was asked
/// instead of reading everything up to it. Fractional seconds, the form `ffmpeg` parses.
fn window_args(window: Option<(Duration, Duration)>) -> Vec<OsString> {
    window.map_or_else(Vec::new, |(start, length)| {
        vec![
            "-ss".into(),
            format!("{:.3}", start.as_secs_f64()).into(),
            "-t".into(),
            format!("{:.3}", length.as_secs_f64()).into(),
        ]
    })
}

/// One frame at `at` seconds: the seek is an input option, so the decoder starts at the mark and
/// stops after one picture instead of reading everything up to it. One fit into a square box, then
/// the even sides the JPEG encoder needs. With `into` absent the same decode runs with nowhere to
/// write — how [`classify`] replays one that failed.
fn frame_args(input: &Path, at: f64, into: Option<&Path>, side: u32) -> Vec<OsString> {
    let filter = format!(
        "scale={side}:{side}:force_original_aspect_ratio=decrease,\
         scale=trunc(iw/2)*2:trunc(ih/2)*2"
    );
    let mut args: Vec<OsString> = vec!["-v".into(), "error".into(), "-nostdin".into(), "-y".into()];
    args.extend([
        "-ss".into(),
        format!("{at:.3}").into(),
        "-i".into(),
        input.into(),
        "-vf".into(),
        filter.into(),
        "-frames:v".into(),
        "1".into(),
    ]);
    match into {
        Some(path) => args.extend([
            "-q:v".into(),
            FRAME_QUANTISER.to_string().into(),
            "-f".into(),
            "image2".into(),
            path.into(),
        ]),
        None => args.extend(["-f".into(), "null".into(), "-".into()]),
    }
    args
}

/// 16 kHz mono signed 16-bit PCM in a WAV container: what the recogniser takes,
/// whatever the source carried. `into` absent is [`classify`]'s replay.
fn audio_args(
    input: &Path,
    window: Option<(Duration, Duration)>,
    into: Option<&Path>,
) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec!["-v".into(), "error".into(), "-nostdin".into(), "-y".into()];
    args.extend(window_args(window));
    args.extend([
        "-i".into(),
        input.into(),
        "-vn".into(),
        "-sn".into(),
        "-dn".into(),
        "-ac".into(),
        "1".into(),
        "-ar".into(),
        "16000".into(),
        "-c:a".into(),
        "pcm_s16le".into(),
    ]);
    match into {
        Some(output) => args.extend(["-f".into(), "wav".into(), output.into()]),
        None => args.extend(["-f".into(), "null".into(), "-".into()]),
    }
    args
}

/// Test hook: the install root, so a check never writes into the developer's own `~/.mahbot/ffmpeg`.
#[cfg(test)]
pub(crate) use install::ROOT_ENV;

/// Points the ffmpeg release hook ([`install::RELEASES_URL_ENV`]) at `base` for as
/// long as it lives, and puts the previous value back — on a panic too, so no later
/// check in this process inherits it. Holds no lock of its own: the caller keeps the
/// crate's environment lock ([`crate::util::test::env_lock`]) for the check's duration
/// and this guard is dropped before it — a second guard of that kind here would
/// deadlock on it.
#[cfg(test)]
pub(crate) struct ReleasesHook(Option<std::ffi::OsString>);

#[cfg(test)]
impl ReleasesHook {
    pub(crate) fn at(base: &str) -> Self {
        let previous = std::env::var_os(install::RELEASES_URL_ENV);
        // SAFETY: the caller holds the environment lock, and this guard is dropped
        // before it.
        unsafe { std::env::set_var(install::RELEASES_URL_ENV, base) };
        // A test that failed to reach a stream leaves that failure behind process-wide; this
        // hook's own host is a fresh attempt.
        install::forget_failed_attempt();
        Self(previous)
    }
}

#[cfg(test)]
impl Drop for ReleasesHook {
    fn drop(&mut self) {
        // SAFETY: the environment lock outlives this guard.
        unsafe {
            match self.0.take() {
                Some(previous) => std::env::set_var(install::RELEASES_URL_ENV, previous),
                None => std::env::remove_var(install::RELEASES_URL_ENV),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::path::{Path, PathBuf};
    use std::process::Command as Std;

    use super::*;

    /// The replay [`classify`] settles a failure with is the operation's own decode —
    /// the same mark, the same filter and bound — with only the sink swapped for one
    /// that writes nothing. A shorter replay would blame a file that merely fails
    /// further in than the replay looked.
    #[test]
    fn the_replay_is_the_whole_decode_with_nothing_to_write() {
        let input = Path::new("/media/clip.mp4");
        let marks = [2.0, 4.5];
        let options = FrameOptions {
            marks: &marks,
            max_side: 320,
        };
        let at = |args: &[OsString], flag: &str| {
            args.iter()
                .position(|arg| arg.to_str() == Some(flag))
                .unwrap_or_else(|| panic!("{flag} is missing from {args:?}"))
        };

        let frame = frame_args(
            input,
            marks[0],
            Some(Path::new("/scratch/frame-00000.jpg")),
            options.max_side,
        );
        let replay = frame_args(input, marks[0], None, options.max_side);
        // The sinks: `-q:v <quantiser> -f image2 <path>` and `-f null -`.
        assert_eq!(&frame[..frame.len() - 5], &replay[..replay.len() - 3]);
        // The mark is an input option: `-ss` comes before `-i`, so the decoder starts at the
        // second instead of reading everything up to it — and the replay carries it too, or it
        // would blame a file for a stretch it never read.
        for args in [frame.as_slice(), replay.as_slice()] {
            assert!(
                at(args, "-ss") < at(args, "-i"),
                "the seek is an input option"
            );
            assert_eq!(args[at(args, "-ss") + 1].to_str(), Some("2.000"));
            assert_eq!(args[at(args, "-frames:v") + 1].to_str(), Some("1"));
        }

        let audio = audio_args(input, None, Some(Path::new("/scratch/audio.wav")));
        let replay = audio_args(input, None, None);
        // The sinks: `-f wav <path>` and `-f null -`.
        assert_eq!(&audio[..audio.len() - 3], &replay[..replay.len() - 3]);

        // A window is an input option: `-ss` and `-t` come before `-i`.
        let window = Some((Duration::from_secs(2), Duration::from_millis(1500)));
        let audio = audio_args(input, window, Some(Path::new("/scratch/audio.wav")));
        let replay = audio_args(input, window, None);
        for args in [audio.as_slice(), replay.as_slice()] {
            assert!(at(args, "-ss") < at(args, "-i"));
            assert!(at(args, "-t") < at(args, "-i"));
            assert_eq!(args[at(args, "-ss") + 1].to_str(), Some("2.000"));
            assert_eq!(args[at(args, "-t") + 1].to_str(), Some("1.500"));
        }
        assert_eq!(&audio[..at(&audio, "-i")], &replay[..at(&replay, "-i")]);
    }

    /// A throwaway install root, with the release stream pointed at an unreachable host so nothing
    /// is ever fetched into it. `pair` copies the host pair named by `MAHBOT_FFMPEG_LIVE_BIN` in,
    /// which is why a run that uses one must be single-threaded: the root is process-wide.
    struct Install {
        _home: tempfile::TempDir,
        host: Option<PathBuf>,
        root: PathBuf,
        /// Any parsable tag: a copy in place is what the operations use, and the name only says
        /// which build it is.
        bin: PathBuf,
        // Dropped in declaration order: the hook puts the release URL back while the environment
        // lock is still held, so it must be declared before the guard that owns that lock.
        _hook: ReleasesHook,
        _env: crate::util::test::EnvVarGuard,
    }

    impl Install {
        fn new(pair: bool) -> Self {
            let home = tempfile::tempdir().expect("temp home");
            let root = home.path().join(".mahbot/ffmpeg");
            let bin = root.join("ffmpeg-v7.1-mahbot.2").join("bin");
            let env = crate::util::test::set_env_var(install::ROOT_ENV, root.to_str());
            let hook = ReleasesHook::at("http://127.0.0.1:1");
            let host =
                pair.then(|| {
                    PathBuf::from(std::env::var_os("MAHBOT_FFMPEG_LIVE_BIN").expect(
                        "MAHBOT_FFMPEG_LIVE_BIN names a directory holding ffmpeg and ffprobe",
                    ))
                });
            if let Some(host) = &host {
                std::fs::create_dir_all(&bin).expect("install directory");
                for tool in ["ffmpeg", "ffprobe"] {
                    std::fs::copy(host.join(tool), bin.join(tool)).expect("copy the stand-in tool");
                }
            }
            Self {
                _home: home,
                host,
                root,
                bin,
                _hook: hook,
                _env: env,
            }
        }

        /// The host pair the root was filled from.
        fn host(&self) -> &Path {
            self.host.as_deref().expect("the host pair was copied in")
        }
    }

    /// The live exercise of the operations — and of the reading mode built on them —
    /// against a real `ffmpeg` and real files. It is opt-in because the product's own copy comes
    /// from a release stream, which no plain `cargo test` run carries:
    ///
    /// ```text MAHBOT_FFMPEG_LIVE_BIN=/opt/homebrew/bin \ cargo test --lib ffmpeg::tests --
    /// --ignored --test-threads=1 ```
    ///
    /// The named directory's `ffmpeg` and `ffprobe` stand in for the product's own
    /// pair under a throwaway install root, which is why the run must be single-threaded:
    /// the root is process-wide. The clips are made by the same binary: one with audio,
    /// one without, one whose sound outlasts its pictures, one rotated, one that is not a video.
    #[ignore = "requires MAHBOT_FFMPEG_LIVE_BIN: a directory holding a real ffmpeg and ffprobe"]
    #[serial_test::serial(ffmpeg_live)] // relocates the install root, points the release hook
    #[tokio::test]
    async fn the_operations_on_real_files() {
        let install = Install::new(true);
        let host = install.host();
        let found = install::ensure_installed()
            .await
            .expect("the copy in place is used");
        assert_eq!(
            found.ffmpeg,
            install.bin.join("ffmpeg"),
            "by path, not by origin"
        );
        assert_eq!(found.ffprobe, install.bin.join("ffprobe"));

        let media = tempfile::tempdir().expect("media directory");
        let good = media.path().join("good.mp4");
        let silent = media.path().join("silent.mp4");
        let rotated = media.path().join("rotated.mp4");
        let broken = media.path().join("broken.mp4");
        make(
            host,
            &good,
            &[
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=1280x720:rate=30:duration=2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
            ],
        );
        make(
            host,
            &silent,
            &[
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=640x480:rate=25:duration=2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
            ],
        );
        let good_text = good.to_str().expect("utf8").to_string();
        make(
            host,
            &rotated,
            &["-display_rotation", "90", "-i", &good_text, "-c", "copy"],
        );
        std::fs::write(&broken, b"not a video at all").expect("broken file");

        let info = probe(&good).await.expect("a good clip probes");
        assert_eq!(info.duration.as_secs(), 2);
        assert_eq!(info.dimensions, Some((1280, 720)));
        assert!(info.has_audio);

        let no_audio = probe(&silent).await.expect("a silent clip probes");
        assert_eq!(no_audio.dimensions, Some((640, 480)));
        assert!(!no_audio.has_audio);

        let turned = probe(&rotated).await.expect("a rotated clip probes");
        assert_eq!(
            turned.dimensions,
            Some((720, 1280)),
            "a rotated clip reports its upright size"
        );

        assert_eq!(probe(&broken).await.err(), Some(FfmpegError::Undecodable));

        let marks = [0.0, 0.5, 1.0];
        let frames = extract_frames(
            &good,
            FrameOptions {
                marks: &marks,
                max_side: 320,
            },
        )
        .await
        .expect("frames come out");
        assert_eq!(frames.frames().len(), 3, "one frame per mark");
        assert!(
            (frames.frames()[2].1 - 1.0).abs() < f64::EPSILON,
            "each frame carries its mark"
        );
        let first = frames.frames()[0].0.clone();
        for (path, _) in frames.frames() {
            assert_eq!(path.extension(), Some(OsStr::new("jpg")));
            let image = image::load_from_memory(&std::fs::read(path).expect("read a frame"))
                .expect("a JPEG");
            assert!(
                image.width() <= 320 && image.height() <= 320,
                "the side bound"
            );
            assert_eq!(image.width() * 9, image.height() * 16, "the aspect ratio");
        }
        drop(frames);
        assert!(!first.exists(), "dropping the guard removes the frames");

        // A mark the file has no picture at is skipped, not fatal: the frames before it are kept,
        // and the clip is not reported as one the product cannot read.
        let past_the_end = [0.0, 0.5, 1.0, 5.0];
        let frames = extract_frames(
            &good,
            FrameOptions {
                marks: &past_the_end,
                max_side: 320,
            },
        )
        .await
        .expect("the frames before the missing mark come out");
        assert_eq!(frames.frames().len(), 3, "the mark past the end is skipped");

        // A picture stream that stops before its container does says so, which is what keeps
        // frames from being planned past its last picture.
        let soundy = media.path().join("soundy.mp4");
        make(
            host,
            &soundy,
            &[
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=25:duration=2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
            ],
        );
        let uneven = probe(&soundy).await.expect("the uneven clip probes");
        assert_eq!(uneven.duration.as_secs(), 5, "the container's own length");
        assert_eq!(
            uneven.picture.map(|picture| picture.as_secs()),
            Some(2),
            "the picture stream's own length"
        );
        assert_eq!(uneven.sound.map(|sound| sound.as_secs()), Some(5));

        assert_eq!(
            extract_frames(
                &broken,
                FrameOptions {
                    marks: &[0.0],
                    max_side: 320,
                }
            )
            .await
            .err(),
            Some(FfmpegError::Undecodable)
        );

        // A file that decodes is never reported as one that cannot: the answer for a
        // failure on the product's own side is a different one, and the file that
        // proves it here is the clip the frames above came out of.
        let pair = binaries().await.expect("the copy in place is used");
        let options = Decode::Frames { at: 0.0, side: 320 };
        let product_side = classify(&pair, &good, options).await;
        assert!(
            matches!(product_side, FfmpegError::Unavailable(ref reason) if !reason.is_empty()),
            "{product_side:?}"
        );
        assert_eq!(
            classify(&pair, &broken, options).await,
            FfmpegError::Undecodable,
            "the same call still names the file that cannot be read"
        );

        let audio = extract_audio(&good, None).await.expect("audio comes out");
        let wav = std::fs::read(audio.path()).expect("read the WAV");
        // The `fmt ` chunk: mono at byte 22, 16 kHz at byte 24, little-endian.
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 1, "mono");
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            16_000
        );
        assert_eq!(
            probe(audio.path())
                .await
                .expect("the WAV probes")
                .duration
                .as_secs(),
            2
        );

        // The two failure classes stay different: a file without an audio track is
        // not the same answer as a file that cannot be decoded.
        assert_eq!(
            extract_audio(&silent, None).await.err(),
            Some(FfmpegError::NoAudioTrack)
        );
        assert_eq!(
            extract_audio(&broken, None).await.err(),
            Some(FfmpegError::Undecodable)
        );

        // The reading mode over the same clip — the one path no other surface can drive without a
        // provider: a window samples its own seconds at two frames a second, and a window past the
        // last picture plans none instead of blaming a decode.
        let window = |start: f64, length: f64| crate::media::Window::Span {
            start,
            length: Some(length),
        };
        let reading = crate::media::read(&silent, window(0.5, 1.0))
            .await
            .expect("a silent clip reads");
        // The marks tell the window's two frames apart, though both fall inside the same second.
        assert!(
            reading.text.contains("0:00.5 [frame 1]"),
            "{}",
            reading.text
        );
        assert!(reading.text.contains("0:01 [frame 2]"), "{}", reading.text);
        assert!(
            reading.text.contains("This video has no sound track."),
            "{}",
            reading.text
        );
        // Both frames go into one picture: the arrival form says which frames it holds, and the
        // read tool's own form carries the mapping on the picture's annotation instead — the
        // timeline itself never claims a picture.
        let (model, attached) = crate::media::inline(&reading).await;
        assert!(attached);
        assert!(model.contains("picture 1: frames 1–2"), "{model}");
        let (text, payloads) =
            crate::tools::timeline_output(reading, crate::tools::TimelinePictures::Attached).await;
        assert!(!text.contains("picture 1"), "{text}");
        assert_eq!(payloads.len(), 1);
        assert_eq!(
            payloads[0].note.as_deref(),
            Some("picture 1 of 1: frames 1–2")
        );

        let past = crate::media::read(&silent, window(5.0, 1.0))
            .await
            .expect("a window past the end still answers");
        assert!(
            past.pictures().await.is_empty(),
            "there is no picture there"
        );
        // The second the read asked for, not the end it was clamped to.
        assert!(
            past.text
                .contains("The read starts at 0:05, past the end of the 0:02 file."),
            "{}",
            past.text
        );

        // No copy and an unreachable release: the answer says so instead of hanging
        // or panicking, and the clips above are untouched by the attempt.
        std::fs::remove_dir_all(&install.root).expect("remove the copy");
        let unavailable = probe(&good).await.expect_err("no copy, no answer");
        assert!(
            matches!(unavailable, FfmpegError::Unavailable(ref reason) if !reason.is_empty()),
            "{unavailable:?}"
        );
        assert!(good.exists());
    }

    /// Asking for a transcription is not asking for a toolchain: the conversion runs only the copy
    /// already in place, so a recording this process has no decoder for is refused rather than
    /// fetched for. The release stream here answers nothing, so an install attempt would fail — and
    /// the install root is left as empty as it was found.
    #[serial_test::serial(ffmpeg_live, ffmpeg_releases)] // relocates the install root, points the release hook
    #[tokio::test]
    async fn a_conversion_never_installs_the_toolchain() {
        let install = Install::new(false);
        let media = tempfile::tempdir().expect("media directory");
        let recording = media.path().join("note.m4a");
        std::fs::write(&recording, b"\0\0\0\x20ftypM4A \0\0\0\0").expect("a recording");
        let error = crate::speech::decode::samples_of(&recording)
            .await
            .expect_err("no decoder here, and no toolchain to convert it");
        assert!(format!("{error:#}").contains("cannot be read"), "{error:#}");
        assert!(
            installed().is_none(),
            "a transcription must not install the pair"
        );
        assert!(!install.root.exists(), "nothing was installed");
    }

    /// Every container the product reads as a sound reaches the recogniser — a file of each accepted
    /// kind produces audio instead of a refusal. Each is made by the same binary: a one-second sine
    /// in the AAC and ALAC forms of MP4, FLAC, Vorbis in OGG, ADTS AAC, A-law WAV, AIFF and CAF —
    /// every one of them a codec this process hands to the toolchain — plus a plain PCM WAV, which is
    /// the in-process path and proves it is untouched by any of that.
    ///
    /// AMR is not here: the product's ffmpeg carries an AMR decoder and no AMR encoder, so a file of
    /// that kind cannot be produced by the pair the test stands in with. The sniffing matrix for
    /// AMR, WMA and AC-3 is covered by the unit tests in `speech::decode`.
    #[ignore = "requires MAHBOT_FFMPEG_LIVE_BIN: a directory holding a real ffmpeg and ffprobe"]
    #[serial_test::serial(ffmpeg_live)] // relocates the install root, points the release hook
    #[tokio::test]
    async fn every_recording_the_product_reads_reaches_the_recogniser() {
        let install = Install::new(true);
        install::ensure_installed()
            .await
            .expect("the copy in place is used");

        let media = tempfile::tempdir().expect("media directory");
        // `-ac 2 -strict -2` is the native Vorbis encoder: this build carries no libvorbis, and the
        // native one writes two channels only.
        let sine = ["-f", "lavfi", "-i", "sine=frequency=440:duration=1"];
        let cases: [(&str, &[&str]); 9] = [
            ("note.m4a", &["-c:a", "aac"]),
            ("note-alac.m4a", &["-c:a", "alac"]),
            ("note.flac", &["-c:a", "flac"]),
            (
                "note-vorbis.ogg",
                &["-ac", "2", "-c:a", "vorbis", "-strict", "-2"],
            ),
            ("note.aac", &["-f", "adts", "-c:a", "aac"]),
            ("note-alaw.wav", &["-c:a", "pcm_alaw"]),
            ("note.aiff", &["-f", "aiff", "-c:a", "pcm_s16be"]),
            ("note.caf", &["-f", "caf", "-c:a", "pcm_s16le"]),
            ("plain.wav", &["-c:a", "pcm_s16le"]),
        ];
        for (name, codec) in cases {
            let path = media.path().join(name);
            let mut args = sine.to_vec();
            args.extend(codec);
            make(install.host(), &path, &args);
            let (samples, rate) = crate::speech::decode::samples_of(&path)
                .await
                .unwrap_or_else(|error| panic!("{name} does not reach the recogniser: {error:#}"));
            assert!(!samples.is_empty(), "{name} produced no audio");
            assert!(rate > 0, "{name} reported a rate of {rate}");
        }
    }

    /// Run a real `ffmpeg` into `output`.
    fn make(host: &Path, output: &Path, args: &[&str]) {
        let mut command = Std::new(host.join("ffmpeg"));
        command.args(["-v", "error", "-y"]).args(args).arg(output);
        let status = command.status().expect("run the stand-in ffmpeg");
        assert!(
            status.success(),
            "the stand-in ffmpeg failed for {output:?}"
        );
    }
}
