//! Wake-word benchmark (three plain metrics) with a single-instance lock and a
//! hung-run timeout.
//!
//! This target prints the benchmark's three plain metrics: recognition (X of 40
//! phrase utterances), false reactions (N on the 113 non-phrase set plus a rate
//! per hour on real audio) and data coverage.  The second target,
//! `wake_analysis`, is NOT a benchmark: it is an offline analyser over the
//! opt-in measurement capture this bench writes, computing the candidate
//! comparison curves — no models, no stores, no network.
//!
//! An opt-in, bench-only measurement capture (`MAHBOT_WAKE_CAPTURE=baseline` or
//! `measure`, off by default; the default build is unaffected) records the raw
//! encoder token matrices, the product's own per-window statistic, a per-clip
//! detection ledger and the frozen A/B partition, for that analyser to consume.
//!
//! # Budget
//!
//! The whole WARM run is capped at ~10 minutes (the pinned real-audio subset
//! is sized so the parallel feed fits; the synthetic phases reuse the TTS PCM
//! cache).  The timeout is a fast-fail hang guard only: 15 minutes.
//!
//! # TTS caveat
//!
//! Recognition and the synthetic false-reaction set are measured on
//! synthesized (TTS) speech, not real human speech; real audio is used for
//! the false-reaction rate only.
//!
//! # Lock mechanism
//!
//! The single-instance lock and timeout scaffolding live in the shared
//! `benches/common` module (which also documents the lock's release guarantee),
//! where the one `LOCK_NAME` both wake-word targets take is the writer's lock
//! (`~/.mahbot/wake_word.lock`), so the analyser can never read a still-growing
//! capture.  The `lock_utils` module behind it is the canonical implementation
//! shared with [`mahbot::self_update`].
//!
//! # Invocation
//!
//! `cargo bench --no-default-features --features voice-tests --bench wake_word`
//! — `--bench wake_word` is required: the `wake_analysis` target is not a
//! benchmark and hard-fails with exit 1 when no capture exists.

mod common;

fn main() {
    // 15-minute hung-run guard: the warm run is budgeted at ~10 min; 15 min is a
    // fast-fail hang guard with headroom.  Rationale in the module doc.
    common::run_with_timeout(15, mahbot::audio::voice::run_wake_word_benchmark);
}
