//! Speech: the product's own engines on the inference crate, both directions.
//!
//! Recognition ([`engine`]) turns a recording into words; synthesis ([`synth`]) turns text into
//! a spoken waveform. Both run on every platform, both read the same machine-wide model folder
//! ([`models`], provisioned by the installer and never created here), and both load lazily
//! through the one lifecycle in [`load`] — nothing at boot, and only the set the operation
//! needs is fetched. The model files are hashed once per start, in the background
//! ([`verify_models_once`]), and no load re-hashes them.
//!
//! Nothing stays resident while idle: [`run_idle_release`] drops an engine that has had no use
//! for five minutes, and once the whole process is cold hands its memory back with the one
//! process-wide clear the inference crate offers. Recognition has two tiers — the listening tier
//! (the pair and the wake word's window listener) and the full engine a transcription needs.
//! With the wake word enabled only the listening tier is kept; with it disabled nothing is.
//!
//! Recognition is lazy on every platform: the surfaces are the composer's microphone, a
//! Telegram voice note or audio attachment, the audio recorded after the wake word,
//! `mahbot audio transcribe`, and a media read — a video's or a recording's sound track, which is
//! transcribed in pieces with timestamps and sound labels. The wake word shares the recognition
//! pair's weights through its own window listener ([`engine::embed_pair`]), so listening never
//! loads a second copy. Synthesis is the admin Assistant's `say` tool and `mahbot audio speak`;
//! nothing is ever spoken on its own.

use tracing::warn;

pub(crate) mod decode;
pub(crate) mod engine;
mod load;
pub(crate) mod models;
pub(crate) mod synth;

pub(crate) use decode::{AUDIO_FORMATS, takes_audio};
pub(crate) use engine::{
    INFERENCE_TIMEOUT, InferenceTimedOut, Interrupted, Segment, UNTRANSCRIBED, Unreadable,
    embed_clip, embed_clips, embed_pair, ensure_ready_once, is_listening_failed,
    is_listening_loaded, is_listening_loading, is_silence_label, is_speech_label,
    is_transcriber_failed, listening_state, retry_listening, retry_transcription,
    spawn_listening_init, spawn_transcriber_init, transcribe_file_async, transcribe_pieces,
    transcribe_samples, transcriber_state,
};

/// Hash every model file that is present, once per process start, off the request path.
pub use models::check_once as verify_models_once;

/// What the wake word and an enrolment need from the listening tier.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ListeningTier {
    /// Nothing needs it: it may be dropped.
    Idle,
    /// Something needs it resident: it is kept, and requested again after the clear.
    Needed,
    /// A workflow is embedding through it: it is never taken, not even for the clear.
    Busy,
}

/// Drop an engine that has had no use for [`load::IDLE_RELEASE`] — the full recognition engine
/// and the synthesiser always, the listening tier only while `demand` says nothing needs it. Each
/// release makes the process dirty; once nothing at all is resident the process-wide clear runs
/// and hands the released weights back, dropping a needed listening tier for the instant it runs
/// unless the tier is busy.
pub async fn run_idle_release(demand: fn() -> ListeningTier) {
    let mut dirty = false;
    loop {
        if !crate::shutdown::sleep_or_shutdown_or_drain(RELEASE_POLL).await {
            return;
        }
        if release_idle(demand).await {
            dirty = true;
        }
        // The clear is what returns the released device weights, and it may only run with nothing
        // resident: it waits for that, and for something to have been released since the last one.
        if dirty && clear_process_memory(demand).await {
            dirty = false;
        }
    }
}

/// How often the idle release looks at the engines.
const RELEASE_POLL: std::time::Duration = std::time::Duration::from_secs(30);

/// One idle-release pass, off the async threads: dropping an engine frees hundreds of megabytes.
/// Returns whether any engine was released.
async fn release_idle(demand: fn() -> ListeningTier) -> bool {
    tokio::task::spawn_blocking(move || {
        let synthesis = synth::ENGINE.release_if_idle();
        let transcriber = engine::TRANSCRIBER.release_if_idle();
        let listening = demand() == ListeningTier::Idle && engine::ENCODER.release_if_idle();
        synthesis || transcriber || listening
    })
    .await
    .unwrap_or_else(|e| {
        warn!("the idle release task panicked: {e}");
        false
    })
}

/// Whether both full engines are cold: nothing resident, nothing loading.
fn both_full_engines_cold() -> bool {
    engine::TRANSCRIBER.is_cold() && synth::ENGINE.is_cold()
}

/// Whether nothing at all is resident or loading — the only state the process-wide clear is safe
/// in, and the state it leaves behind.
fn speech_is_cold() -> bool {
    both_full_engines_cold() && engine::ENCODER.is_cold()
}

/// Hand the released engines' memory back, off the async threads. The clear is the dependency's
/// one and it is process-wide, so every engine has to be cold for it: a resident engine keeps its
/// own buffers, and clearing the pool under it would make its next load upload the same weights a
/// second time. The listening tier is the recognition weights' third holder, so a needed one is
/// dropped for the clear and requested again right after — that reload is the cost of going fully
/// cold, and it happens off the request path. A busy tier is never taken: the clear waits for it.
/// Returns whether the clear ran.
async fn clear_process_memory(demand: fn() -> ListeningTier) -> bool {
    let (cleared, demand) = tokio::task::spawn_blocking(move || {
        let demand = demand();
        let cleared = if demand == ListeningTier::Busy {
            // An enrolment is embedding through the tier: the next tick tries again.
            false
        } else if !both_full_engines_cold() {
            // A full engine is warm, so the clear cannot run: taking the tier for it would only
            // drop and reload it. Checked before the tier is touched, never after.
            false
        } else if demand == ListeningTier::Needed && !engine::ENCODER.release() {
            // A wake-word pass holds it: the next tick tries again.
            false
        } else if !speech_is_cold() {
            // A load raced the release: the process is warm again.
            false
        } else {
            mahbot_inference::release_process_memory();
            true
        };
        (cleared, demand)
    })
    .await
    .unwrap_or_else(|e| {
        warn!("the process-wide memory clear task panicked: {e}");
        (false, ListeningTier::Idle)
    });
    // Whatever became of the clear, a tier the wake word or an enrolment needs is not left cold.
    if demand != ListeningTier::Idle && engine::ENCODER.is_cold() {
        engine::spawn_listening_init();
    }
    cleared
}
