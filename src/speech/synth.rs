//! Spoken replies: text into speech on the inference crate's talker, and the machine's output
//! device.
//!
//! The engine behind the `say` tool ([`crate::tools::SayTool`]) and `mahbot audio speak`: one
//! text in, one waveform out, mono f32 PCM at the engine's own 24 kHz. The whole text is
//! spoken — the product neither splits it nor truncates it nor bounds its length; a text the
//! engine refuses is a refusal, never a shorter answer.
//!
//! The engine carries its one voice, so there is no voice, sample or language to choose, and
//! nothing here reads one.
//!
//! Nothing is spoken by itself: an assistant reply, an automation or a confirmation reaches
//! the engine only through a `say` call. A phrase is then spoken when speech output is enabled
//! ([`is_config_enabled`]), the model set can be loaded, an output device could be opened, and
//! someone is at the computer ([`crate::user_activity`]) — a phrase is not spoken to an empty
//! room. A newer phrase stops the one being spoken.
//!
//! Nothing loads at boot: the model set is fetched the first time a phrase needs it
//! ([`crate::speech::models`], the same machine-wide folder recognition reads), and the engine
//! is released after five idle minutes, so nothing stays resident between phrases. The first
//! phrase after a release waits for the load — the tool call blocks for it — rather than being
//! refused.

use crate::config::CONFIG;
use crate::util::UnwrapPoison;
use anyhow::{Context as _, Result};
use mahbot_inference::loader::{Pair, Role};
use mahbot_inference::synth::Synthesis;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{debug, warn};

use crate::audio::output;
use crate::speech::load::Lazy;
use crate::speech::models::{self, Purpose};
use crate::util::model_state::ModelState;

/// The engine's waveform rate: mono 24 kHz, the sound half's own rate.
const SAMPLE_RATE: u32 = mahbot_inference::synth::SAMPLE_RATE;

/// How often playback checks whether a newer phrase has taken over.
const PLAYBACK_POLL: Duration = Duration::from_millis(50);

/// The longest one phrase's playback is waited on: a sink that stopped draining (the device
/// was lost, or the host stopped consuming) never reports itself empty, and the speaking flag
/// it holds must not latch the microphone shut.
const PLAYBACK_WAIT_LIMIT: Duration = Duration::from_secs(30);

/// The product's one synthesiser: the talker pair from the machine-wide root.
pub(crate) static ENGINE: Lazy<Synthesis> = Lazy::new(Purpose::Synthesis, || {
    Synthesis::new(Arc::new(Pair::of_role(Role::Synthesis)?))
});

/// Bumped by every phrase that reaches the engine: playback of an older one stops as soon as it
/// no longer matches.
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// What became of one [`say_phrase`] call.
pub(crate) enum SayOutcome {
    Speaking,
    SpeechOff,
    NotReady(NotReady),
    Absent,
    /// Nothing was spoken, and this says why in the caller's own words — the engine refused
    /// the text, or the render itself failed.
    Failed(String),
}

/// Why the engine could not speak a phrase right now.
pub(crate) enum NotReady {
    ModelsMissing,
    /// The set could not be loaded: an open failed, or the set is not on this machine. Never the
    /// end — a set on disk is opened again by the next use, and a missing one is fetched again.
    LoadFailed,
    NoAudioDevice,
}

/// Returns `true` if speech output is enabled in config (regardless of model state), so a
/// caller can avoid an unnecessary fetch or device open. Speech output is opt-in:
/// `tts_enabled` must be set to `"true"`.
pub(crate) fn is_config_enabled() -> bool {
    CONFIG.tts_enabled().as_deref() == Some("true")
}

/// The engine's lifecycle state: what the settings row reports.
pub(crate) fn state() -> ModelState {
    ENGINE.state()
}

/// Retry the engine after a `Failed` state, or start it when it has not been asked for yet.
pub(crate) fn retry() {
    ENGINE.retry();
}

/// Test-only: pin the lifecycle state, for a caller exercising the states a real load takes
/// minutes to reach.
#[cfg(test)]
pub(crate) fn test_set_state(state: crate::util::model_state::ModelState) {
    ENGINE.test_set_state(state);
}

/// Obtain the set once, with a notice before a fetch that really begins, and open the engine —
/// the command line's own entry point: one attempt, no retry loop, and a failure is the
/// caller's to report.
pub(crate) async fn ensure_ready_once(announce: impl FnOnce(&Path)) -> Result<()> {
    ENGINE.ensure_ready_once(announce).await
}

/// Speak one short phrase for the admin's Assistant, and return once it has been rendered: the
/// waveform then plays in the background. This is the product's whole speech output.
///
/// A phrase is not spoken to an empty room, and that is decided before the engine is asked for,
/// so an absent owner never pays a cold load. An engine that is not loaded is loaded and waited
/// for — the first phrase after an idle release is spoken after it, and one whose load failed
/// over files on disk is opened again by this same call — while a model set that is absent
/// starts its fetch in the background and is refused. A text the engine refuses is answered as
/// such — nothing is spoken and the reason says why.
pub(crate) async fn say_phrase(text: &str) -> SayOutcome {
    if !is_config_enabled() {
        return SayOutcome::SpeechOff;
    }
    // Before the load: a phrase spoken to an empty room must not pay a cold load.
    if crate::user_activity::is_absent() {
        return SayOutcome::Absent;
    }
    if !output::audio_output_ready() {
        // Detached on purpose: the open outlives this answer, and a blocking task cannot be
        // cancelled.
        drop(tokio::task::spawn_blocking(output::ensure_audio_output));
        return SayOutcome::NotReady(NotReady::NoAudioDevice);
    }
    let Ok(dir) = models::root() else {
        return SayOutcome::NotReady(NotReady::ModelsMissing);
    };
    // A load on disk is waited out, not refused: a refusal is for a set that cannot be loaded.
    let Some(engine) = ENGINE.acquire(&dir).await else {
        return SayOutcome::NotReady(if state() == ModelState::Failed {
            NotReady::LoadFailed
        } else {
            NotReady::ModelsMissing
        });
    };
    // A newer phrase takes over from the one being spoken, and from one still rendering: the
    // playback loop of the older one sees this before its next poll.
    let generation = GENERATION.fetch_add(1, Ordering::AcqRel) + 1;
    match render(engine, text).await {
        Ok(samples) => {
            tokio::spawn(play(samples, generation));
            SayOutcome::Speaking
        }
        // A refusal is the engine declining to speak a text it cannot say whole; anything else
        // out of the same call is a fault. The engine's own type is what tells them apart.
        Err(e)
            if e.downcast_ref::<mahbot_inference::synth::Unspoken>()
                .is_some() =>
        {
            SayOutcome::Failed(format!("the engine did not speak the text — {e:#}"))
        }
        Err(e) => SayOutcome::Failed(format!("the text could not be spoken — {e:#}")),
    }
}

/// Render `text` to the spoken file's bytes: the product's own preparation and one engine call,
/// written as uncompressed 16-bit mono WAV at the engine's own rate.
pub(crate) async fn render_speech_wav(text: &str) -> Result<Vec<u8>> {
    let engine = ENGINE
        .acquire(&models::root()?)
        .await
        .context("the speech model set is not on this machine yet, or it failed to load")?;
    let samples = render(engine, text).await?;
    wav(&samples, SAMPLE_RATE)
}

/// One text through the engine: the product's preparation, then the whole text in one call.
async fn render(engine: Arc<Mutex<Synthesis>>, text: &str) -> Result<Vec<f32>> {
    let processed = preprocess_text(text);

    let pass = tokio::task::spawn_blocking(move || {
        let mut engine = engine.lock().unwrap_poison();
        engine.synthesize(&processed, rand::random())
    });

    let shutdown = crate::shutdown::shutdown_token();
    tokio::select! {
        joined = pass => joined.context("the speech synthesis task panicked")?,
        () = shutdown.cancelled() => anyhow::bail!("Shutdown during speech synthesis"),
    }
}

/// Play one rendered phrase through the shared output device, until it ends or a newer phrase
/// takes over. The "the product is speaking" flag is raised for the playback and the reverb
/// tail, so the microphone never hears the product itself.
async fn play(samples: Vec<f32>, generation: u64) {
    // A newer phrase may have arrived while this one rendered: it is the one to hear.
    if generation != GENERATION.load(Ordering::Acquire) {
        return;
    }
    let Some(mixer) = output::mixer() else {
        return;
    };
    let source = rodio::buffer::SamplesBuffer::new(
        rodio::ChannelCount::MIN,
        rodio::SampleRate::new(SAMPLE_RATE).expect("24 kHz is not zero"),
        samples,
    );
    let speaking = output::Speaking::start();
    let player = rodio::Player::connect_new(&mixer);
    player.append(source);
    debug!("Speech synthesis playback started — the microphone is gated while it plays");

    let deadline = tokio::time::Instant::now() + PLAYBACK_WAIT_LIMIT;
    loop {
        // Over when the sound has played out, or when a newer phrase has taken over.
        if player.empty() || generation != GENERATION.load(Ordering::Acquire) {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            warn!("Speech playback did not drain within {PLAYBACK_WAIT_LIMIT:?}; stopping it");
            break;
        }
        tokio::time::sleep(PLAYBACK_POLL).await;
    }
    player.stop();
    // Keep the flag raised briefly after playback ends, so room acoustics and the output
    // buffers cannot reach the microphone.
    tokio::time::sleep(output::SPEAKING_TAIL).await;
    drop(speaking);
}

/// Render PCM float samples to an in-memory WAV file (16-bit mono PCM), header and data
/// included.
fn wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let channels: u16 = 1;
    let bps: u16 = 16;
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bps / 8);
    let block_align = channels * (bps / 8);
    let data_size = u32::try_from(samples.len() * (bps / 8) as usize)
        .context("WAV data size exceeds u32 range")?;
    let file_size = data_size
        .checked_add(36)
        .context("WAV size exceeds u32 range")?;

    let mut buf = Vec::with_capacity(44 + data_size as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&(16u32).to_le_bytes());
    buf.extend_from_slice(&(1u16).to_le_bytes()); // PCM
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bps.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        #[expect(clippy::cast_possible_truncation)]
        let sample_i16 = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&sample_i16.to_le_bytes());
    }

    Ok(buf)
}

/// Compiled regex patterns used by [`strip_markdown`].
static RE_CODE_BLOCK: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"```[\s\S]*?```").expect("RE_CODE_BLOCK"));
static RE_INLINE_CODE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"`[^`]+`").expect("RE_INLINE_CODE"));
static RE_IMAGE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"!\[[^\]]*\]\([^)]*\)").expect("RE_IMAGE"));
static RE_LINK: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"\[([^\]]*)\]\([^)]*\)").expect("RE_LINK"));
static RE_BOLD_ITALIC: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"\*{1,3}[^*]+\*{1,3}").expect("RE_BOLD_ITALIC"));
static RE_STRIKETHROUGH: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"~~[^~]+~~").expect("RE_STRIKETHROUGH"));
static RE_HEADER: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"(?m)^#{1,6}\s+").expect("RE_HEADER"));
static RE_LIST_DASH: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"(?m)^[\s]*[-*]\s+").expect("RE_LIST_DASH"));
static RE_LIST_NUM: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"(?m)^\s*\d+\.\s+").expect("RE_LIST_NUM"));
static RE_HR: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"(?m)^[-*_]{3,}\s*$").expect("RE_HR"));

/// Preprocess text: NFKD normalize, strip markdown, clean punctuation.
fn preprocess_text(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;

    // Media markers are display/plumbing artifacts, never speech — neither a path nor a
    // base64 data URI is spoken.
    let text = crate::util::MEDIA_MARKER_RE.replace_all(text, " ");
    let mut s: String = text.as_ref().nfkd().collect();
    s = strip_markdown(&s);
    s = remove_emojis(&s);
    s = normalize_symbols(&s);
    s = expand_abbreviations(&s);
    s = fix_punctuation_spacing(&s);
    s = remove_duplicate_quotes(&s);
    s = clean_whitespace(&s);

    if !has_ending_punctuation(&s) {
        s.push('.');
    }

    s
}

/// Strip markdown syntax that would be read verbatim.
fn strip_markdown(text: &str) -> String {
    let mut s = text.to_string();

    s = RE_CODE_BLOCK.replace_all(&s, " ").to_string();
    s = RE_INLINE_CODE.replace_all(&s, " ").to_string();
    s = RE_IMAGE.replace_all(&s, " ").to_string();
    s = RE_LINK.replace_all(&s, "$1").to_string();
    s = RE_BOLD_ITALIC.replace_all(&s, " ").to_string();
    s = RE_STRIKETHROUGH.replace_all(&s, " ").to_string();
    s = RE_HEADER.replace_all(&s, "").to_string();
    s = RE_LIST_DASH.replace_all(&s, "").to_string();
    s = RE_LIST_NUM.replace_all(&s, "").to_string();
    s = RE_HR.replace_all(&s, " ").to_string();

    s
}

fn remove_emojis(text: &str) -> String {
    text.chars()
        .filter(|&c| {
            let code = c as u32;
            !matches!(code,
                0x1F600..=0x1F64F | 0x1F300..=0x1F5FF |
                0x1F680..=0x1F6FF | 0x1F700..=0x1F77F |
                0x1F780..=0x1F7FF | 0x1F800..=0x1F8FF |
                0x1F900..=0x1F9FF | 0x1FA00..=0x1FA6F |
                0x1FA70..=0x1FAFF | 0x1FB00..=0x1FBFF |
                0x2600..=0x26FF | 0x2700..=0x27BF)
        })
        .collect()
}

fn normalize_symbols(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{2013}' | '\u{2011}' | '\u{2014}' => '-',
            '\u{00AF}' | '[' | ']' | '|' | '/' | '#' | '\u{2192}' | '\u{2190}' | '\u{2665}'
            | '\u{2606}' | '\u{2661}' | '\u{00A9}' | '\\' => ' ',
            '\u{201C}' | '\u{201D}' => '"',
            '\u{2018}' | '\u{2019}' | '\u{00B4}' | '`' => '\'',
            _ => c,
        })
        .collect()
}

fn expand_abbreviations(text: &str) -> String {
    text.replace('@', " at ")
        .replace("e.g.,", "for example, ")
        .replace("i.e.,", "that is, ")
}

fn fix_punctuation_spacing(text: &str) -> String {
    text.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" !", "!")
        .replace(" ?", "?")
        .replace(" ;", ";")
        .replace(" :", ":")
        .replace(" '", "'")
}

fn remove_duplicate_quotes(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' || c == '\'' || c == '`' {
            while chars.peek() == Some(&c) {
                chars.next();
            }
            result.push(c);
        } else {
            result.push(c);
        }
    }
    result
}

fn clean_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_space = false;
    for c in text.chars() {
        if c.is_whitespace() {
            if !prev_space {
                result.push(' ');
                prev_space = true;
            }
        } else {
            result.push(c);
            prev_space = false;
        }
    }
    result.trim().to_string()
}

fn has_ending_punctuation(text: &str) -> bool {
    text.chars().last().is_some_and(|c| {
        matches!(
            c,
            '.' | '!' | '?' | '\u{3002}' /* 。 */ | '\u{FF01}' /* ！ */ | '\u{FF1F}' /* ？ */
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_text() {
        let r = preprocess_text("Hello world");
        assert_eq!(
            r, "Hello world.",
            "should normalize and add trailing period"
        );

        // Markdown stripped, emoji removed, symbols normalized, abbreviations expanded.
        // **Hi** is removed entirely — bold syntax carries no text content for speech.
        let r2 = preprocess_text("**Hi** @user, e.g., hello 😊");
        assert_eq!(
            r2, "at user, for example, hello.",
            "should strip markdown, expand abbrevs, remove emoji, add period"
        );

        // A media marker is plumbing, never speech.
        let r3 = preprocess_text("done [IMAGE:/tmp/a.png] here");
        assert_eq!(r3, "done here.", "should drop the media marker");

        let r4 = preprocess_text("Hello, world.");
        assert_eq!(r4, "Hello, world.", "should not double-add period");
    }

    #[test]
    fn test_render_wav_decodes_via_hound() {
        use rodio::Source;
        // WAV decode goes through hound (Apache-2.0 posture): verify it still decodes the
        // written PCM16 mono output with the right rate, channel count and sample values.
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let wav_bytes = wav(&samples, SAMPLE_RATE).expect("the WAV should render");

        let cursor = std::io::Cursor::new(wav_bytes);
        let decoder = rodio::Decoder::new(cursor).expect("WAV should decode via hound");
        assert_eq!(
            decoder.sample_rate().get(),
            SAMPLE_RATE,
            "sample rate mismatch"
        );
        assert_eq!(decoder.channels().get(), 1, "should be mono");
        let decoded_samples: Vec<f32> = decoder.collect();
        assert_eq!(
            decoded_samples.len(),
            samples.len(),
            "sample count mismatch"
        );
        for (got, expected) in decoded_samples.iter().zip(&samples) {
            assert!(
                (got - expected).abs() < 1e-3,
                "sample {got} != expected {expected}"
            );
        }
    }

    // say_phrase
    //
    // What the tool's one entry point answers while speech is off: it reaches neither the
    // engine nor the audio device.
    /// Run `say_phrase` under a known config, restoring it afterwards.
    async fn say_under(config: &str) -> SayOutcome {
        let snapshot = CONFIG.snapshot();
        assert!(
            CONFIG.set_string_field("tts_enabled", config),
            "tts_enabled is a known config key"
        );
        let outcome = say_phrase("A short phrase.").await;
        CONFIG.swap(snapshot);
        outcome
    }

    #[tokio::test]
    #[serial_test::serial(speech)]
    async fn say_refuses_while_speech_is_off() {
        for config in ["", "false"] {
            assert!(
                matches!(say_under(config).await, SayOutcome::SpeechOff),
                "`tts_enabled = '{config}'` must be reported as speech output off"
            );
        }
    }
}
