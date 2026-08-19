/// Simple voice pipeline benchmark (three plain metrics) with a
/// single-instance lock and a hung-run timeout.
///
/// This is the fast baseline bench: recognition (X of 40 phrase utterances),
/// false reactions (N on the 113 non-phrase set + a rate per hour on real
/// audio), and data coverage — nothing else.
///
/// # Budget
///
/// The whole WARM run is capped at ~10 minutes (the pinned real-audio subset
/// is sized so the parallel feed fits; the synthetic phases reuse the TTS PCM
/// cache).  The timeout is a fast-fail hang guard only: 15 minutes.
///
/// # Lock mechanism
///
/// Uses a lock file at `~/.mahbot/voice_pipeline_simple.lock` (its own lock
/// file, distinct from the e2e bench's — each bench serializes its own runs;
/// the two benches share the encoder, so operators should not run them
/// simultaneously).  The shared single-instance-lock + timeout scaffolding
/// lives in [`voice_common`].
#[path = "common/voice_common.rs"]
mod voice_common;

fn main() {
    // 15-minute hung-run guard: the warm run is budgeted at ~10 min; 15 min
    // is a fast-fail hang guard with headroom.  Rationale in the module doc.
    voice_common::run_with_timeout(
        "voice_pipeline_simple.lock",
        15,
        mahbot::audio::voice::run_simple_voice_pipeline_benchmark,
    );
}
