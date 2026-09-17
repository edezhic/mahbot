//! Offline analyser over the wake-word measurement capture that
//! `benches/wake_word.rs` writes, under the shared single-instance lock and
//! hung-run timeout scaffolding (`benches/common`).
//!
//! No models, no stores, no network: it reads the capture directory, writes
//! `analysis.json` back into it and prints a one-screen digest to stdout.  The
//! analyser's own entry point exits with status 1 when the capture is missing
//! or inconsistent, so the timeout below is a pure hung-run guard (short — the
//! pass is CPU-only over ~60 MB of tokens).

mod common;

fn main() {
    // Same lock as the bench that writes the capture, so the analyser can never
    // read a capture that is still growing.  10-minute hung-run guard: the pass
    // is CPU-only over the captured token blob.
    common::run_with_timeout(10, mahbot::audio::voice::run_wake_analysis);
}
