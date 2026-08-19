/// Voice pipeline E2E benchmark with single-instance lock and a per-mode timeout.
///
/// The lock prevents concurrent benchmark runs.  The timeout aborts hung runs
/// via [`std::process::exit`] and depends on the run mode: 30 minutes for
/// standard runs (fast-fail hang guard — measured runs complete in ~9 min),
/// 300 minutes when `MAHBOT_FAPH=1` because the env-gated FAPH phase feeds the
/// full 5.99 h real-audio corpus through the encoder at ~6-8× real-time
/// (measured ~47 min FAPH / ~54 min total run on this machine; 300 min is a
/// hung-run abort net with ~5.5× headroom over that measured total).
///
/// # Lock mechanism
///
/// Uses a lock file at `~/.mahbot/voice_pipeline_e2e.lock`.  The shared
/// single-instance-lock + timeout scaffolding lives in [`voice_common`]; the
/// `lock_utils` module behind it is the canonical implementation shared with
/// [`mahbot::self_update`].
#[path = "common/voice_common.rs"]
mod voice_common;

fn main() {
    // Per-mode timeout — 30 min standard / 300 min when MAHBOT_FAPH=1;
    // rationale in the module doc.
    let faph_enabled = std::env::var("MAHBOT_FAPH").as_deref() == Ok("1");
    let timeout_mins: u64 = if faph_enabled { 300 } else { 30 };
    voice_common::run_with_timeout(
        "voice_pipeline_e2e.lock",
        timeout_mins,
        mahbot::audio::voice::run_voice_pipeline_benchmark,
    );
}
