//! Wake word: the enrolled phrase and the streaming wiring around the inference crate's
//! decision.
//!
//! The decision itself — the prototype, the negative calibration, the rolling score, the
//! threshold set, the adaptive threshold and the scoring call — is the engine's
//! ([`mahbot_inference::wake`]). This module owns only what the engine does not provide: the
//! persisted enrolment record, the capture of VAD-gated windows, the order in which they are fed
//! (two consecutive windows per encoder pass), the segment clock and the cooldown clock, and the
//! enrolment quality gates.
//!
//! Enrolment embeds each accepted utterance one clip at a time ([`mahbot_inference::listen`]);
//! live detection queues two consecutive windows and embeds them in one pass, which resolves a
//! fire up to one scoring stride (~160 ms) later than a one-window pass. That delay is the
//! accepted price of halving the passes through the encoder.
//!
//! The persisted record is schema 3. A record from the old encoder (schema 2 and earlier) cannot
//! be scored by the engine and is refused, never migrated: the owner enrols again.

use anyhow::Result;
use mahbot_inference::listen;
use mahbot_inference::wake::{
    self, AdaptiveThreshold, Calibration, DetectionGeometry, Enrollment, Thresholds,
};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Re-arm delay after an accepted fire, so a rapid consecutive false trigger cannot follow.
pub(crate) const WAKE_WORD_COOLDOWN: Duration = Duration::from_secs(3);

/// Serialization schema version for [`WakeWordEnrollment`]. Records from the old encoder carry a
/// lower version and are refused at load — there is no migration path.
const ENROLLMENT_SCHEMA_VERSION: u32 = 3;

/// The phrase the enrolment is for. The engine carries it with the record; the product's UI never
/// asks for it.
const WAKE_PHRASE: &str = "mahbot";

/// The persisted enrolment: the prototype, the negative calibration and the anti-prototypes the
/// engine scores with. The threshold set is not stored — the engine's shipped set applies, so a
/// tuning change reaches every existing enrolment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WakeWordEnrollment {
    #[serde(default)]
    pub(crate) schema_version: u32,
    /// L2-normalized mean of the accepted utterance embeddings.
    pub(crate) prototype: Vec<f32>,
    pub(crate) utterance_count: usize,
    #[serde(default)]
    pub(crate) neg_mean: f32,
    #[serde(default)]
    pub(crate) neg_std: f32,
    /// Raw 99th percentile of the negative cosine distribution — diagnostics only.
    #[serde(default)]
    pub(crate) neg_p99: f32,
    #[serde(default)]
    pub(crate) n_negatives: usize,
    /// Up to [`wake::MAX_NEGATIVE_PROTOTYPES`] L2-normalized negative centroids.
    #[serde(default)]
    pub(crate) negative_prototypes: Vec<Vec<f32>>,
    pub(crate) created_at: String,
    pub(crate) trained_at: String,
}

impl WakeWordEnrollment {
    /// Whether the engine can score this record: the current schema and a usable prototype.
    pub(crate) fn usable(&self) -> bool {
        self.schema_version == ENROLLMENT_SCHEMA_VERSION
            && !self.prototype.is_empty()
            && self.prototype.iter().all(|v| v.is_finite())
            && self.neg_mean.is_finite()
            && self.neg_std.is_finite()
            && self
                .negative_prototypes
                .iter()
                .all(|p| p.len() == self.prototype.len())
    }

    /// The engine's own enrolment, with the shipped threshold set.
    pub(crate) fn engine(&self) -> Enrollment {
        Enrollment {
            phrase: WAKE_PHRASE.to_string(),
            prototype: self.prototype.clone(),
            utterance_count: self.utterance_count,
            calibration: Calibration {
                neg_mean: self.neg_mean,
                neg_std: self.neg_std,
                neg_p99: self.neg_p99,
                n_negatives: self.n_negatives,
            },
            negative_prototypes: self.negative_prototypes.clone(),
            thresholds: Thresholds::DEFAULT,
        }
    }

    /// The record for a freshly built engine enrolment.
    pub(crate) fn from_engine(
        enrollment: &Enrollment,
        created_at: String,
        trained_at: String,
    ) -> Self {
        Self {
            schema_version: ENROLLMENT_SCHEMA_VERSION,
            prototype: enrollment.prototype.clone(),
            utterance_count: enrollment.utterance_count,
            neg_mean: enrollment.calibration.neg_mean,
            neg_std: enrollment.calibration.neg_std,
            neg_p99: enrollment.calibration.neg_p99,
            n_negatives: enrollment.calibration.n_negatives,
            negative_prototypes: enrollment.negative_prototypes.clone(),
            created_at,
            trained_at,
        }
    }
}

/// What one resolved group of windows produced: whether the engine heard the phrase at all (a
/// fire the segment clock refused still means a person spoke it), and whether the fire was
/// accepted.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Detection {
    pub(crate) heard: bool,
    pub(crate) fired: bool,
}

/// The streaming wiring around the engine's decision: the VAD-gated window, the windows queued
/// for a two-window encoder pass, the rolling score, the adaptive threshold, the segment clock and
/// the cooldown.
pub(crate) struct WakeDetector {
    enrollment: Enrollment,
    geometry: DetectionGeometry,
    adaptive: AdaptiveThreshold,
    score_window: Vec<f32>,
    speech_window: Vec<f32>,
    pending: Vec<Vec<f32>>,
    stride_samples: usize,
    segment_steps: usize,
    last_fire: Option<Instant>,
}

impl WakeDetector {
    pub(crate) fn new(enrollment: Enrollment) -> Self {
        let geometry = enrollment.geometry();
        let adaptive = AdaptiveThreshold::warmed(&enrollment.thresholds, geometry.static_threshold);
        Self {
            enrollment,
            geometry,
            adaptive,
            score_window: Vec::new(),
            speech_window: Vec::new(),
            pending: Vec::new(),
            stride_samples: 0,
            segment_steps: 0,
            last_fire: None,
        }
    }

    /// Whether the post-fire cooldown is still running.
    pub(crate) fn cooldown_active(&self) -> bool {
        self.last_fire
            .is_some_and(|last| last.elapsed() < WAKE_WORD_COOLDOWN)
    }

    /// Feed one VAD hop: speech hops build the gated window, and every hop advances the stride.
    pub(crate) fn note_hop(&mut self, hop: &[f32], speech: bool) {
        self.stride_samples += hop.len();
        if speech {
            push_capped(&mut self.speech_window, hop, listen::WINDOW_SAMPLES);
        }
    }

    /// Queue the trailing window once a scoring stride has elapsed.
    ///
    /// Silence still counts once the segment has scored: the same trailing window is re-queued
    /// across a VAD gap, which is what lets a short phrase fill the rolling window. Before the
    /// first scored window, a silent stride queues nothing.
    pub(crate) fn take_stride(&mut self, speech_seen: bool) {
        if self.stride_samples < listen::SCORE_STRIDE_SAMPLES {
            return;
        }
        if !speech_seen && self.score_window.is_empty() {
            return;
        }
        self.stride_samples = 0;
        if self.speech_window.is_empty() {
            return;
        }
        let start = self
            .speech_window
            .len()
            .saturating_sub(listen::WINDOW_SAMPLES);
        self.pending.push(self.speech_window[start..].to_vec());
    }

    /// Whether two windows are waiting for one encoder pass.
    pub(crate) fn pair_ready(&self) -> bool {
        self.pending.len() >= listen::PAIR
    }

    /// Score the queued windows in arrival order through the engine's decision. A fire ends the
    /// walk: the windows after it are dropped, exactly as the stream would not have scored them.
    pub(crate) fn resolve(
        &mut self,
        embed: impl FnOnce(&[Vec<f32>]) -> Result<Vec<Vec<f32>>>,
    ) -> Detection {
        if self.pending.is_empty() {
            return Detection::default();
        }
        let windows = std::mem::take(&mut self.pending);
        let embeddings = match embed(&windows) {
            Ok(embeddings) => embeddings,
            Err(e) => {
                warn!("Wake word window encoding failed: {e}");
                return Detection::default();
            }
        };

        let mut detection = Detection::default();
        let thresholds = self.enrollment.thresholds;
        for embedding in &embeddings {
            let total = self.enrollment.soft_score(embedding);
            // Feed only background scores to the adaptive threshold, so it learns the noise floor
            // instead of being contaminated by the scores it is meant to detect.
            let adaptive_value = if total < self.geometry.reset {
                self.adaptive
                    .feed(total, &thresholds, self.geometry.static_threshold)
            } else {
                self.adaptive
                    .peek(&thresholds, self.geometry.static_threshold)
            };
            let threshold = adaptive_value.unwrap_or(self.geometry.static_threshold);
            let (fired, rolling_sum) =
                wake::process_score(total, &mut self.score_window, self.geometry, threshold);
            self.segment_steps += 1;

            debug!(
                "Wake word score: total_score={total:.4} rolling_sum={rolling_sum:.4}/ \
                 threshold={threshold:.2} step={}",
                self.segment_steps,
            );
            #[cfg(feature = "voice-debug")]
            tracing::info!(
                "VOICE_DEBUG: total_score={total:.4} rolling_sum={rolling_sum:.4} \
                 threshold={threshold:.4} fired={fired}"
            );

            if !fired {
                continue;
            }
            detection.heard = true;
            // A fire counts only near the start of its segment; past that position the match is
            // inert until the segment ends.
            if self.segment_steps > wake::WAKE_WORD_MAX_SEGMENT_STEPS {
                continue;
            }
            detection.fired = true;
            self.last_fire = Some(Instant::now());
            self.soft_reset();
            return detection;
        }
        detection
    }

    /// A segment ended: score the lone window it left waiting in the old state, then clear the
    /// per-segment state so nothing accumulates across utterances.
    pub(crate) fn close_segment(
        &mut self,
        embed: impl FnOnce(&[Vec<f32>]) -> Result<Vec<Vec<f32>>>,
    ) -> Detection {
        let detection = self.resolve(embed);
        self.reset_segment();
        detection
    }

    /// The post-fire reset: the gated window, the queued windows and the rolling scores go; the
    /// adaptive state and the cooldown clock stay.
    pub(crate) fn soft_reset(&mut self) {
        self.score_window.clear();
        self.segment_steps = 0;
        self.speech_window.clear();
        self.pending.clear();
        self.stride_samples = 0;
    }

    fn reset_segment(&mut self) {
        self.soft_reset();
        self.adaptive.reset();
    }

    /// The stream reset: everything, the cooldown included.
    pub(crate) fn reset_stream(&mut self) {
        self.reset_segment();
        self.last_fire = None;
    }
}

/// Append `samples` to `buf`, draining the front past `cap`; returns how many were drained so
/// callers can shift their cursors.
pub(crate) fn push_capped(buf: &mut Vec<f32>, samples: &[f32], cap: usize) -> usize {
    buf.extend_from_slice(samples);
    let excess = buf.len().saturating_sub(cap);
    buf.drain(..excess);
    excess
}

/// Build the engine's enrolment from the accepted utterances and the negative pool.
pub(crate) fn build_enrollment(
    utterance_embeddings: &[Vec<f32>],
    calibration: Calibration,
    negative_embeddings: &[Vec<f32>],
) -> Option<Enrollment> {
    Enrollment::build(
        WAKE_PHRASE.to_string(),
        utterance_embeddings,
        calibration,
        negative_embeddings,
    )
}

/// Centroid-consistency gate for the enrolment set: the prototype needs
/// [`wake::MIN_ENROLLMENT_UTTERANCES`] utterances with at least ceil(N ×
/// [`wake::ENROLLMENT_CONSISTENCY_MIN_FRACTION`]) of them at cosine ≥
/// [`wake::ENROLLMENT_CONSISTENCY_MIN_SIMILARITY`] to the centroid. `Ok(centroid)` on pass, else
/// `Err` with a user-facing message.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub(crate) fn consistency_check(utterance_embeddings: &[Vec<f32>]) -> Result<Vec<f32>, String> {
    if utterance_embeddings.len() < wake::MIN_ENROLLMENT_UTTERANCES {
        return Err(format!(
            "Only {} enrollment utterances collected (need ≥{}). \
             Speak clearly and close to the microphone.",
            utterance_embeddings.len(),
            wake::MIN_ENROLLMENT_UTTERANCES,
        ));
    }
    let centroid = wake::normalized_centroid(utterance_embeddings)
        .map_err(|got| format!("An utterance embedding is {got} wide, unlike the first"))?;

    let required = (utterance_embeddings.len() as f32 * wake::ENROLLMENT_CONSISTENCY_MIN_FRACTION)
        .ceil() as usize;
    let passed = utterance_embeddings
        .iter()
        .filter(|emb| wake::cosine(emb, &centroid) >= wake::ENROLLMENT_CONSISTENCY_MIN_SIMILARITY)
        .count();
    if passed < required {
        return Err(format!(
            "Only {passed}/{} enrollment utterances match the prototype \
             (need ≥{required} at cosine ≥ {}). \
             Try re-enrolling with clearer, more consistent speech.",
            utterance_embeddings.len(),
            wake::ENROLLMENT_CONSISTENCY_MIN_SIMILARITY,
        ));
    }
    Ok(centroid)
}

/// Self-test the enrolment: each utterance embedding is fed through the engine's rolling score
/// (three times, the number of windows a phrase spans) with a fresh window and the static
/// threshold. `Ok(())` when enough trigger, else `Err` with a user-facing message.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub(crate) fn self_test(
    utterance_embeddings: &[Vec<f32>],
    enrollment: &Enrollment,
    required_fraction: f32,
) -> Result<(), String> {
    if utterance_embeddings.is_empty() {
        return Err("Self-test skipped: no enrollment samples".to_string());
    }

    let geometry = enrollment.geometry();
    let mut passed = 0usize;
    for embedding in utterance_embeddings {
        let score = enrollment.soft_score(embedding);
        let mut window = Vec::new();
        let mut detected = false;
        for _ in 0..wake::ROLLING_WINDOW_N {
            let (fired, _) =
                wake::process_score(score, &mut window, geometry, geometry.static_threshold);
            if fired {
                detected = true;
                break;
            }
        }
        if detected {
            crate::user_activity::report_activity();
            passed += 1;
        }
    }

    let required = (utterance_embeddings.len() as f32 * required_fraction).ceil() as usize;
    if passed < required {
        Err(format!(
            "Self-test failed: only {passed}/{} utterances triggered detection (need ≥{required}). \
             Try re-enrolling with clearer, more consistent speech.",
            utterance_embeddings.len(),
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enrollment() -> Enrollment {
        let prototype = vec![1.0, 0.0, 0.0, 0.0];
        let utterances = vec![prototype; wake::MIN_ENROLLMENT_UTTERANCES];
        Enrollment::build(
            WAKE_PHRASE.to_string(),
            &utterances,
            Calibration::default(),
            &[],
        )
        .expect("a valid synthetic enrolment")
    }

    fn queue_one_window(detector: &mut WakeDetector) {
        let before = detector.pending.len();
        for _ in 0..listen::SCORE_STRIDE_SAMPLES / 256 {
            detector.note_hop(&[0.0; 256], true);
        }
        detector.take_stride(true);
        assert_eq!(
            detector.pending.len(),
            before + 1,
            "a stride queues one window"
        );
    }

    #[test]
    fn two_windows_are_embedded_in_one_pass() {
        let mut detector = WakeDetector::new(enrollment());
        queue_one_window(&mut detector);
        queue_one_window(&mut detector);
        assert!(detector.pair_ready());

        let mut passes = 0;
        let detection = detector.resolve(|windows| {
            passes += 1;
            assert_eq!(windows.len(), listen::PAIR, "one pass, two windows");
            Ok(vec![vec![1.0, 0.0, 0.0, 0.0]; windows.len()])
        });
        assert_eq!(passes, 1);
        assert!(detection.heard && detection.fired, "the pair fires");
        assert!(!detector.pair_ready(), "the pair is consumed");
        assert!(detector.cooldown_active());
    }

    #[test]
    fn a_lone_window_is_flushed_on_its_own_at_a_segment_close() {
        let mut detector = WakeDetector::new(enrollment());
        queue_one_window(&mut detector);
        assert!(!detector.pair_ready());

        let mut passes = 0;
        let detection = detector.close_segment(|windows| {
            passes += 1;
            assert_eq!(windows.len(), 1, "a lone window is not padded into a pair");
            Ok(vec![vec![1.0, 0.0, 0.0, 0.0]])
        });
        assert_eq!(passes, 1);
        assert!(!detection.fired, "one window can never fire");
        assert!(detector.score_window.is_empty(), "the segment is cleared");
    }

    #[test]
    fn an_old_encoder_record_is_refused() {
        // Schema 2 carried the old encoder's embedding and a nested calibration; the unknown keys
        // are ignored and the version alone refuses it.
        let v2 = r#"{"schema_version":2,"embedding_dim":1024,"prototype":[0.1,0.2],
                     "utterance_count":10,"calibration":{"neg_mean":0.1,"neg_std":0.02,
                     "neg_p99":0.2,"n_negatives":11},"negative_prototypes":[],
                     "created_at":"then","trained_at":"then"}"#;
        let record: WakeWordEnrollment = serde_json::from_str(v2).expect("v2 still parses");
        assert!(!record.usable(), "an old encoder's record is refused");

        let current =
            WakeWordEnrollment::from_engine(&enrollment(), "now".to_string(), "now".to_string());
        let round_trip: WakeWordEnrollment =
            serde_json::from_str(&serde_json::to_string(&current).unwrap()).unwrap();
        assert!(round_trip.usable());
        assert_eq!(round_trip.engine().prototype, current.prototype);
    }
}
