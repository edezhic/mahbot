//! The product's own ffmpeg and ffprobe: the copy the product installs itself
//! (`install`) and the three video operations that copy serves.
//!
//! A copy the owner installed — a package manager's, a vendor download, anything
//! on the search path — is never run: every command here is the absolute path of
//! the product's own build, and no `PATH` lookup is ever made.
//!
//! The operations are exactly three:
//!
//! * [`probe`] — duration, upright dimensions, and whether the file carries an
//!   audio track.
//! * [`extract_frames`] — JPEG frames at a given rate, the longest side bounded,
//!   the count bounded by the caller.
//! * [`extract_audio`] — the audio track re-encoded to 16 kHz mono WAV, which is
//!   what the local transcriber takes.
//!
//! Each runs under its own [`Duration`] — the wait for the copy included — and every
//! output lands in a [`tempfile::TempDir`] under the product's private temp root —
//! never in the owner's workspace or library — carried by the returned guard, so
//! dropping the result removes the files.
//!
//! Failures are typed and deliberately distinct: [`FfmpegError::Unavailable`] is
//! the product's own half not being usable at all (with the reason), while
//! [`FfmpegError::Undecodable`], [`FfmpegError::NoVideoTrack`] and
//! [`FfmpegError::NoAudioTrack`] are facts about the file. None of that can be read
//! off the tools' exit statuses — a decode that finds no audio stream and a file that
//! cannot be decoded end the same way — so a failed decode is settled by asking the
//! file again.

// Each item of the three operations' surface carries `expect(dead_code)` rather than a
// module-wide `allow`: `expect` turns the first caller outside this module into a
// build error that must remove the annotation with it.
mod install;
mod probe;

pub use install::install_on_start;

use std::ffi::OsString;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tokio::process::Command;

use install::Binaries;

/// How long reading a file's facts may take.
const PROBE_TIMEOUT: Duration = Duration::from_secs(30);

/// How long extracting frames may take: a bounded count out of a bounded side, so
/// the work stays short even for a long clip.
const FRAMES_TIMEOUT: Duration = Duration::from_secs(120);

/// How long extracting the audio may take: a long recording is decoded in full.
const AUDIO_TIMEOUT: Duration = Duration::from_secs(300);

/// What `ffprobe` reports, and the only entries the parser reads.
const PROBE_ENTRIES: &str = "format=duration:stream=codec_type,width,height,duration\
                            :stream_tags=rotate:stream_side_data=rotation\
                            :stream_disposition=attached_pic";

/// The video facts the product needs about one file.
#[cfg_attr(not(test), expect(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClipInfo {
    pub(crate) duration: Duration,
    /// The upright size of the first video stream — what a person watching the
    /// file sees, so a rotated phone clip reports its portrait size — or `None`
    /// when the file carries no video stream at all.
    pub(crate) dimensions: Option<(u32, u32)>,
    /// Whether the file carries an audio stream.
    pub(crate) has_audio: bool,
}

/// Why a video operation could not answer.
#[cfg_attr(not(test), expect(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FfmpegError {
    /// The product's own half could not do it: its copy cannot be run (not
    /// installed, not installable, or started and failed), or its scratch area
    /// could not be created or read. The reason names the failing step and carries
    /// no path and no URL.
    Unavailable(String),
    /// The file carries no video stream to read frames from.
    NoVideoTrack,
    /// The file carries no audio stream to extract.
    NoAudioTrack,
    /// The file could not be decoded.
    Undecodable,
    /// The operation outlived its own bound.
    TimedOut,
    /// The caller asked for something the operation cannot be given.
    InvalidOptions(&'static str),
}

/// Frames extracted from one file, removed when this guard is dropped.
#[cfg_attr(not(test), expect(dead_code))]
#[derive(Debug)]
pub(crate) struct Frames {
    /// Holds the directory the frames live in; never read.
    _dir: tempfile::TempDir,
    paths: Vec<PathBuf>,
}

impl Frames {
    /// The frames in playback order. Empty when the file itself carries none.
    #[cfg_attr(not(test), expect(dead_code))]
    #[must_use]
    pub(crate) fn paths(&self) -> &[PathBuf] {
        &self.paths
    }
}

/// One file's audio as 16 kHz mono WAV, removed when this guard is dropped.
#[cfg_attr(not(test), expect(dead_code))]
#[derive(Debug)]
pub(crate) struct Audio {
    /// Holds the directory the WAV lives in; never read.
    _dir: tempfile::TempDir,
    path: PathBuf,
}

impl Audio {
    #[cfg_attr(not(test), expect(dead_code))]
    #[must_use]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

/// The bounds frame extraction is given.
#[cfg_attr(not(test), expect(dead_code))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct FrameOptions {
    /// How many frames of source material per second to keep.
    pub(crate) fps: f64,
    /// The longest side of a written frame, in pixels. The aspect ratio is kept.
    pub(crate) max_side: u32,
    /// The most frames to write.
    pub(crate) max_frames: u32,
}

/// Duration, upright dimensions and audio presence of one file.
///
/// [`FfmpegError::Undecodable`] when the tools cannot read the file at all.
#[cfg_attr(not(test), expect(dead_code))]
pub(crate) async fn probe(path: &Path) -> Result<ClipInfo, FfmpegError> {
    bounded(PROBE_TIMEOUT, async {
        let bin = binaries().await?;
        probe_with(&bin.ffprobe, path).await
    })
    .await
}

/// JPEG frames of `input` at [`FrameOptions::fps`], each fitting
/// [`FrameOptions::max_side`], at most [`FrameOptions::max_frames`] of them.
#[cfg_attr(not(test), expect(dead_code))]
pub(crate) async fn extract_frames(
    input: &Path,
    options: FrameOptions,
) -> Result<Frames, FfmpegError> {
    if !(options.fps.is_finite() && options.fps > 0.0) {
        return Err(FfmpegError::InvalidOptions(
            "the frame rate must be greater than zero",
        ));
    }
    if options.max_side < 2 {
        return Err(FfmpegError::InvalidOptions(
            "the longest side must be at least two pixels",
        ));
    }
    if options.max_frames == 0 {
        return Err(FfmpegError::InvalidOptions("no frames were asked for"));
    }

    bounded(FRAMES_TIMEOUT, async {
        let bin = binaries().await?;
        let dir = scratch()?;
        let args = frames_args(input, Some(dir.path()), &options);
        let output = run_tool(&bin.ffmpeg, &args, FRAMES_TIMEOUT).await?;
        if !output.success {
            tracing::debug!("ffmpeg wrote no frames: {}", output.stderr.trim());
            return Err(classify(&bin, input, Decode::Frames(&options)).await);
        }
        let paths = frames_in(dir.path())?;
        Ok(Frames { _dir: dir, paths })
    })
    .await
}

/// The audio track of `input` as 16 kHz mono WAV.
#[cfg_attr(not(test), expect(dead_code))]
pub(crate) async fn extract_audio(input: &Path) -> Result<Audio, FfmpegError> {
    bounded(AUDIO_TIMEOUT, async {
        let bin = binaries().await?;
        let dir = scratch()?;
        let path = dir.path().join("audio.wav");
        let args = audio_args(input, Some(&path));
        let output = run_tool(&bin.ffmpeg, &args, AUDIO_TIMEOUT).await?;
        if !output.success {
            tracing::debug!("ffmpeg extracted no audio: {}", output.stderr.trim());
            return Err(classify(&bin, input, Decode::Audio).await);
        }
        Ok(Audio { _dir: dir, path })
    })
    .await
}

/// Run one operation under its own bound, the wait for its copy included: a first use
/// that arrives while the release is still being installed waits no longer than the
/// operation itself would have taken.
async fn bounded<T, F>(bound: Duration, operation: F) -> Result<T, FfmpegError>
where
    F: Future<Output = Result<T, FfmpegError>>,
{
    tokio::time::timeout(bound, operation)
        .await
        .map_err(|_| FfmpegError::TimedOut)?
}

/// The product's own pair, installed if it is not there yet.
async fn binaries() -> Result<Binaries, FfmpegError> {
    install::ensure_installed()
        .await
        .map_err(FfmpegError::Unavailable)
}

/// `ffprobe`'s answer for one file, or [`FfmpegError::Undecodable`] when the file
/// yields no streams and no duration — which is what a corrupt or foreign file
/// produces, whatever the tool's exit status.
async fn probe_with(ffprobe: &Path, path: &Path) -> Result<ClipInfo, FfmpegError> {
    let output = run_tool(ffprobe, &probe_args(path), PROBE_TIMEOUT).await?;
    let Some(info) = probe::parse(&output.stdout) else {
        tracing::debug!("ffprobe could not read the file: {}", output.stderr.trim());
        return Err(FfmpegError::Undecodable);
    };
    Ok(info)
}

/// The decode an operation was attempting, so a failure can be replayed as it was.
#[derive(Debug, Clone, Copy)]
enum Decode<'a> {
    Frames(&'a FrameOptions),
    Audio,
}

/// Why a decode failed: a file the tools cannot read at all is undecodable, a readable
/// file that simply lacks the track the operation needed says so, and a file that
/// decodes fine means the product's own output was what failed.
///
/// The order matters, and none of it can be read off `ffmpeg`'s exit status. The file
/// is asked instead: its tracks first, then the operation's own decode replayed over
/// the whole file with nowhere to write — so a file that survives exactly what just
/// failed on it is one whose failure was on the product's own side.
async fn classify(bin: &Binaries, input: &Path, decode: Decode<'_>) -> FfmpegError {
    let info = match probe_with(&bin.ffprobe, input).await {
        Err(error) => return error,
        Ok(info) => info,
    };
    let (replay, timeout) = match decode {
        Decode::Frames(_) if info.dimensions.is_none() => return FfmpegError::NoVideoTrack,
        Decode::Audio if !info.has_audio => return FfmpegError::NoAudioTrack,
        Decode::Frames(options) => (frames_args(input, None, options), FRAMES_TIMEOUT),
        Decode::Audio => (audio_args(input, None), AUDIO_TIMEOUT),
    };
    match run_tool(&bin.ffmpeg, &replay, timeout).await {
        Ok(output) if output.success => {
            FfmpegError::Unavailable("the product's own output could not be written".to_string())
        }
        Ok(_) => FfmpegError::Undecodable,
        Err(error) => error,
    }
}

/// What one finished tool run left behind.
struct Output {
    success: bool,
    stdout: Vec<u8>,
    stderr: String,
}

/// Run one of the product's own tools to completion or to its bound.
///
/// Nothing is inherited: stdin is null, so a tool can neither read the daemon's
/// terminal nor wait on it, and its output is captured, so a decode can never print
/// into the daemon's own streams. A child that outlives the bound is killed by
/// dropping the future, which is why the command is built with `kill_on_drop`.
///
/// A child killed by a signal says nothing about the file it was reading, so it is not
/// reported as a finished run. Signals are a Unix notion — Windows reports a code for
/// every death — so the check simply never fires there.
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

/// The tool's own name, for a reason a reader can act on: `ffprobe` failing to start
/// is not `ffmpeg` failing to start.
fn tool_name(program: &Path) -> String {
    program
        .file_stem()
        .and_then(std::ffi::OsStr::to_str)
        .map_or_else(|| "tool".to_string(), str::to_owned)
}

/// A directory under the product's private temp root, removed when the guard is
/// dropped.
fn scratch() -> Result<tempfile::TempDir, FfmpegError> {
    tempfile::tempdir().map_err(|e| {
        FfmpegError::Unavailable(format!(
            "the product's own scratch directory could not be created ({})",
            e.kind()
        ))
    })
}

/// The frames a successful decode wrote, in playback order.
fn frames_in(dir: &Path) -> Result<Vec<PathBuf>, FfmpegError> {
    let entries = std::fs::read_dir(dir).map_err(|e| {
        FfmpegError::Unavailable(format!(
            "the extracted frames could not be listed ({})",
            e.kind()
        ))
    })?;
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("jpg"))
        })
        .collect();
    // `ffmpeg` numbers the files through the output pattern, so the names sort
    // into playback order.
    paths.sort();
    Ok(paths)
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

/// `fps` first, so the rate is taken from the source timeline, then one fit into
/// a square box, then the even sides the JPEG encoder needs. With `into` absent the
/// same decode runs with nowhere to write — how [`classify`] replays one that failed.
fn frames_args(input: &Path, into: Option<&Path>, options: &FrameOptions) -> Vec<OsString> {
    let side = options.max_side;
    let filter = format!(
        "fps={},scale={side}:{side}:force_original_aspect_ratio=decrease,\
         scale=trunc(iw/2)*2:trunc(ih/2)*2",
        options.fps
    );
    let mut args: Vec<OsString> = vec![
        "-v".into(),
        "error".into(),
        "-nostdin".into(),
        "-y".into(),
        "-i".into(),
        input.into(),
        "-vf".into(),
        filter.into(),
        "-frames:v".into(),
        options.max_frames.to_string().into(),
    ];
    match into {
        Some(dir) => args.extend([
            // `-q:v` runs 2..31 with lower the better: 3 sits near the top, because a
            // frame is read rather than displayed.
            "-q:v".into(),
            "3".into(),
            "-f".into(),
            "image2".into(),
            dir.join("frame-%05d.jpg").into(),
        ]),
        None => args.extend(["-f".into(), "null".into(), "-".into()]),
    }
    args
}

/// 16 kHz mono signed 16-bit PCM in a WAV container: what the local transcriber
/// takes, whatever the source carried. `into` absent is [`classify`]'s replay.
fn audio_args(input: &Path, into: Option<&Path>) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![
        "-v".into(),
        "error".into(),
        "-nostdin".into(),
        "-y".into(),
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
    ];
    match into {
        Some(output) => args.extend(["-f".into(), "wav".into(), output.into()]),
        None => args.extend(["-f".into(), "null".into(), "-".into()]),
    }
    args
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::path::{Path, PathBuf};
    use std::process::Command as Std;

    use super::*;

    /// The replay [`classify`] settles a failure with is the operation's own decode —
    /// the whole file, the same filters and bounds — with only the sink swapped for one
    /// that writes nothing. A shorter replay would blame a file that merely fails
    /// further in than the replay looked.
    #[test]
    fn the_replay_is_the_whole_decode_with_nothing_to_write() {
        let input = Path::new("/media/clip.mp4");
        let options = FrameOptions {
            fps: 2.0,
            max_side: 320,
            max_frames: 5,
        };

        let frames = frames_args(input, Some(Path::new("/scratch")), &options);
        let replay = frames_args(input, None, &options);
        // The sinks: `-q:v 3 -f image2 <pattern>` and `-f null -`.
        let decode = frames.len() - 5;
        assert_eq!(&frames[..decode], &replay[..replay.len() - 3]);

        let audio = audio_args(input, Some(Path::new("/scratch/audio.wav")));
        let replay = audio_args(input, None);
        // The sinks: `-f wav <path>` and `-f null -`.
        assert_eq!(&audio[..audio.len() - 3], &replay[..replay.len() - 3]);
    }

    /// The live exercise of the three operations, against a real `ffmpeg` and real
    /// files. It is opt-in because the product's own copy comes from a release
    /// stream, which no plain `cargo test` run carries:
    ///
    /// ```text
    /// MAHBOT_FFMPEG_LIVE_BIN=/opt/homebrew/bin \
    ///     cargo test --lib ffmpeg::tests -- --ignored --test-threads=1
    /// ```
    ///
    /// The named directory's `ffmpeg` and `ffprobe` stand in for the product's own
    /// pair under a throwaway `HOME`, which is why the run must be single-threaded:
    /// `HOME` is what the install root resolves from. The clips are made by the same
    /// binary: one with audio, one without, one rotated, one that is not a video.
    #[ignore = "requires MAHBOT_FFMPEG_LIVE_BIN: a directory holding a real ffmpeg and ffprobe"]
    #[serial_test::serial(ffmpeg_live)] // relocates HOME, points the release hook
    #[tokio::test]
    async fn the_three_operations_on_real_files() {
        let host = PathBuf::from(
            std::env::var_os("MAHBOT_FFMPEG_LIVE_BIN")
                .expect("MAHBOT_FFMPEG_LIVE_BIN names a directory holding ffmpeg and ffprobe"),
        );
        let home = tempfile::tempdir().expect("temp home");
        let _home = crate::util::test::set_env_var("HOME", home.path().to_str());
        let _hook = ReleasesHook::unreachable();
        let bin = home
            .path()
            .join(".mahbot/ffmpeg")
            .join(install::TAG)
            .join("bin");
        std::fs::create_dir_all(&bin).expect("install directory");
        for tool in ["ffmpeg", "ffprobe"] {
            std::fs::copy(host.join(tool), bin.join(tool)).expect("copy the stand-in tool");
        }
        let found = binaries().await.expect("the copy in place is used");
        assert_eq!(found.ffmpeg, bin.join("ffmpeg"), "by path, not by origin");
        assert_eq!(found.ffprobe, bin.join("ffprobe"));

        let media = tempfile::tempdir().expect("media directory");
        let good = media.path().join("good.mp4");
        let silent = media.path().join("silent.mp4");
        let rotated = media.path().join("rotated.mp4");
        let broken = media.path().join("broken.mp4");
        make(
            &host,
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
            &host,
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
            &host,
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

        let frames = extract_frames(
            &good,
            FrameOptions {
                fps: 2.0,
                max_side: 320,
                max_frames: 3,
            },
        )
        .await
        .expect("frames come out");
        assert_eq!(frames.paths().len(), 3, "the count bound is honoured");
        let first = frames.paths()[0].clone();
        for path in frames.paths() {
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

        assert_eq!(
            extract_frames(
                &broken,
                FrameOptions {
                    fps: 1.0,
                    max_side: 320,
                    max_frames: 3
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
        let options = FrameOptions {
            fps: 1.0,
            max_side: 320,
            max_frames: 3,
        };
        let product_side = classify(&pair, &good, Decode::Frames(&options)).await;
        assert!(
            matches!(product_side, FfmpegError::Unavailable(ref reason) if !reason.is_empty()),
            "{product_side:?}"
        );
        assert_eq!(
            classify(&pair, &broken, Decode::Frames(&options)).await,
            FfmpegError::Undecodable,
            "the same call still names the file that cannot be read"
        );

        let audio = extract_audio(&good).await.expect("audio comes out");
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
            extract_audio(&silent).await.err(),
            Some(FfmpegError::NoAudioTrack)
        );
        assert_eq!(
            extract_audio(&broken).await.err(),
            Some(FfmpegError::Undecodable)
        );

        // No copy and an unreachable release: the answer says so instead of hanging
        // or panicking, and the clips above are untouched by the attempt.
        std::fs::remove_dir_all(home.path().join(".mahbot/ffmpeg")).expect("remove the copy");
        let unavailable = probe(&good).await.expect_err("no copy, no answer");
        assert!(
            matches!(unavailable, FfmpegError::Unavailable(ref reason) if !reason.is_empty()),
            "{unavailable:?}"
        );
        assert!(good.exists());
    }

    /// Points the releases hook at a base that cannot answer, for as long as it lives,
    /// and puts the previous value back — on a panic too, so no later test in this
    /// process inherits it. The `HOME` guard is what holds the environment lock; a
    /// second guard of that kind would deadlock on it.
    struct ReleasesHook(Option<std::ffi::OsString>);

    impl ReleasesHook {
        fn unreachable() -> Self {
            let previous = std::env::var_os(install::RELEASES_URL_ENV);
            // SAFETY: the caller holds the environment lock, and this guard is
            // dropped before it.
            unsafe { std::env::set_var(install::RELEASES_URL_ENV, "http://127.0.0.1:1") };
            Self(previous)
        }
    }

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
