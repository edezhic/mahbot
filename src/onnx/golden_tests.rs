//! Golden model-level regression tests for the in-repo ONNX runtime.
//!
//! These run the four pinned Supertonic 3 TTS models (opset 19) through the
//! runtime with fixed deterministic inputs and compare the outputs
//! **bit-exactly** against reference tensors captured from the removed
//! `candle-onnx-mahbot` fork (mahbot-1776).  The fork was the only oracle for
//! TTS audio; these fixtures preserve that oracle in the repo.
//!
//! Feature-gated behind `voice-tests` (like the voice-pipeline e2e bench)
//! because they require the ~383 MB model files on disk under
//! `~/.mahbot/models/supertonic3/onnx`.  When the models are absent the tests
//! skip with a notice instead of failing.
//!
//! Fixture format (`src/onnx/golden/*.bin`): `u32` little-endian dimension
//! count, `u32` little-endian dims, then `f32` little-endian elements.

#![cfg(feature = "voice-tests")]

use super::{read_file, simple_eval};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn build_inputs(inputs: Vec<(&str, Tensor)>) -> HashMap<String, Tensor> {
    inputs
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

/// Deterministic xorshift PRNG — the same recipe used for the fork reference
/// capture (seeds 42 for style_dp/noise, 947 for style_ttl).
struct Xs(u64);
impl Xs {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    fn fill(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let span = (hi - lo) as f32;
        (0..n)
            .map(|_| lo + (self.next_u64() as f64 / u64::MAX as f64) as f32 * span)
            .collect()
    }
}

fn models_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".mahbot/models/supertonic3/onnx"))
}

fn load_fixture(path: &Path) -> (Vec<u32>, Vec<f32>) {
    let buf = std::fs::read(path).expect("read golden fixture");
    let ndims = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let mut dims = Vec::new();
    let mut off = 4;
    for _ in 0..ndims {
        dims.push(u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
        off += 4;
    }
    let mut vals = Vec::new();
    while off + 4 <= buf.len() {
        vals.push(f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
        off += 4;
    }
    (dims, vals)
}

fn assert_matches_golden(label: &str, t: &Tensor, fixture: &str) {
    let golden_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/onnx/golden")
        .join(fixture);
    let (dims, expected) = load_fixture(&golden_path);
    let got_dims: Vec<u32> = t.dims().iter().map(|&d| d as u32).collect();
    assert_eq!(
        got_dims, dims,
        "{label}: shape mismatch {:?} vs golden {:?}",
        got_dims, dims
    );
    let got: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(got.len(), expected.len(), "{label}: element count mismatch");
    let mut max_abs = 0.0f32;
    let mut first: Option<(usize, f32, f32)> = None;
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        let d = (g - e).abs();
        max_abs = max_abs.max(d);
        if d != 0.0 && first.is_none() {
            first = Some((i, *g, *e));
        }
    }
    assert!(
        max_abs == 0.0,
        "{label}: NOT bit-exact vs fork golden (max_abs {max_abs:.3e}{})",
        match first {
            Some((i, g, e)) => format!("; first diff @{i}: got {g}, golden {e}"),
            None => String::new(),
        }
    );
}

fn run_all_models() -> Option<()> {
    let dir = models_dir()?;
    let names = [
        "duration_predictor.onnx",
        "text_encoder.onnx",
        "vector_estimator.onnx",
        "vocoder.onnx",
    ];
    for n in names {
        if !dir.join(n).exists() {
            eprintln!("TTS ONNX golden tests: missing {n} under {dir:?} — skipping");
            return None;
        }
    }
    let dev = &Device::Cpu;

    let dp_model = read_file(dir.join("duration_predictor.onnx")).ok()?;
    let te_model = read_file(dir.join("text_encoder.onnx")).ok()?;
    let ve_model = read_file(dir.join("vector_estimator.onnx")).ok()?;
    let voc_model = read_file(dir.join("vocoder.onnx")).ok()?;

    // Fixture inputs (identical to the fork reference capture):
    // text_ids = [1,2,3,4,5]; style_dp/style_ttl/noise from seeds 42/947.
    let ids: Vec<i64> = vec![1, 2, 3, 4, 5];
    let seq_len = ids.len();
    let text_ids = Tensor::from_slice(&ids, (1, seq_len), dev).ok()?;
    let text_mask = Tensor::from_slice(&vec![1.0f32; seq_len], (1, 1, seq_len), dev).ok()?;
    let mut rng = Xs(42);
    let style_dp = Tensor::from_slice(&rng.fill(8 * 16, -1.0, 1.0), (1, 8, 16), dev).ok()?;
    let mut rng2 = Xs(947);
    let style_ttl = Tensor::from_slice(&rng2.fill(50 * 256, -1.0, 1.0), (1, 50, 256), dev).ok()?;

    // Duration predictor.
    let dp_out = simple_eval(
        &dp_model,
        build_inputs(vec![
            ("text_ids", text_ids.clone()),
            ("style_dp", style_dp.clone()),
            ("text_mask", text_mask.clone()),
        ]),
    )
    .ok()?;
    assert_matches_golden("duration_predictor", dp_out.get("duration")?, "dp_out.bin");

    // Text encoder.
    let te_out = simple_eval(
        &te_model,
        build_inputs(vec![
            ("text_ids", text_ids),
            ("style_ttl", style_ttl.clone()),
            ("text_mask", text_mask.clone()),
        ]),
    )
    .ok()?;
    let text_emb = te_out.get("text_emb")?;
    assert_matches_golden("text_encoder", text_emb, "te_out.bin");

    // Vector estimator at flow-matching steps 0 and 7.
    let latent_len = 8usize;
    let noisy = Tensor::from_slice(
        &rng.fill(144 * latent_len, -1.0, 1.0),
        (1, 144, latent_len),
        dev,
    )
    .ok()?;
    let l_mask = Tensor::from_slice(&vec![1.0f32; latent_len], (1, 1, latent_len), dev).ok()?;
    let total8 = Tensor::new(8.0f32, dev).ok()?.reshape((1,)).ok()?;
    for (step, fixture) in [(0usize, "ve_step0.bin"), (7, "ve_step7.bin")] {
        let step_f = Tensor::new(step as f32, dev).ok()?.reshape((1,)).ok()?;
        let ve_out = simple_eval(
            &ve_model,
            build_inputs(vec![
                ("noisy_latent", noisy.clone()),
                ("text_emb", text_emb.clone()),
                ("style_ttl", style_ttl.clone()),
                ("latent_mask", l_mask.clone()),
                ("text_mask", text_mask.clone()),
                ("current_step", step_f),
                ("total_step", total8.clone()),
            ]),
        )
        .ok()?;
        assert_matches_golden(
            &format!("vector_estimator step {step}"),
            ve_out.get("denoised_latent")?,
            fixture,
        );
    }

    // Vocoder.
    let voc_out = simple_eval(&voc_model, build_inputs(vec![("latent", noisy)])).ok()?;
    assert_matches_golden("vocoder", voc_out.get("wav_tts")?, "voc_out.bin");

    Some(())
}

#[test]
fn golden_models_bit_exact_vs_fork() {
    if run_all_models().is_none() {
        eprintln!("TTS ONNX golden tests skipped (models not on disk)");
    }
}
