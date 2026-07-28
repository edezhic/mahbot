/// Identity of a speech utterance within the enrollment pipeline.
///
/// Uniqueness is guaranteed only within the same [`Source`] category — two
/// sequences from different sources may have the same [`UtteranceId`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UtteranceId {
    /// Monotonically-increasing index within the source category.
    ///
    /// - Enrollment utterances: increments per `handle_enrollment_sample` call.
    /// - Cache confusable/unrelated: `phrase_index * seeds_per_phrase + seed_variant`.
    /// - Ambient: chunk index within `negative_audio_chunks`.
    /// - Synthetic: always 0 (single combined sequence).
    pub sequence_index: usize,
    /// Variant index: 0=original, 1=speed-down, 2=speed-up,
    /// 3=volume-down, 4=noise. Always 0 for non-augmented sequences.
    pub variant_index: usize,
}

/// Provenance of an embedding sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Original wake-word utterance (unmodified PCM).
    Enrollment,
    /// PCM-augmented variant of an enrollment utterance (speed, volume, noise).
    Augmentation,
    /// Pre-computed confusable near-miss phrase (TTS-synthesised).
    Confusable,
    /// Pre-computed unrelated speech phrase (TTS-synthesised).
    Unrelated,
    /// Synthetic negatives generated from positive statistics.
    Synthetic,
    /// Ambient noise / non-speech environmental audio chunk.
    Ambient,
    /// Owner-general speech collected during Phase 3 enrollment (non-wake-word
    /// speech from the user, used as additional negative examples for both the
    /// classifier and the verifier; mahbot-932).
    Owner,
}

/// Augmentation family for [`Source::Augmentation`].
///
/// Direction is preserved (`SpeedDown` vs `SpeedUp`) so training diagnostics
/// can identify which augmentation type produces the most useful learning
/// signal, without relying on the brittle positional `variant_index` mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AugmentationFamily {
    SpeedDown,
    SpeedUp,
    Volume,
    Noise,
}

/// Label stratum for downstream training / diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelStratum {
    Positive,
    Negative,
}

/// A group of per-frame 96-dim embedding vectors that share a common
/// provenance, augmentation family, and label stratum.
///
/// Windows are formed **within** each sequence only — never across
/// sequences — preventing the cross-utterance window contamination
/// that existed when training operated on flat `&[Vec<f32>]` lists.
#[derive(Debug, Clone)]
pub struct EmbeddingSequence {
    pub id: UtteranceId,
    pub source: Source,
    pub augmentation_family: Option<AugmentationFamily>,
    pub label_stratum: LabelStratum,
    /// Per-frame 96-dim embedding vectors in temporal order.
    pub embeddings: Vec<Vec<f32>>,
}

impl EmbeddingSequence {
    /// Create a positive-label sequence from enrollment or augmentation data.
    pub fn positive(
        id: UtteranceId,
        source: Source,
        augmentation_family: Option<AugmentationFamily>,
        embeddings: Vec<Vec<f32>>,
    ) -> Self {
        Self {
            id,
            source,
            augmentation_family,
            label_stratum: LabelStratum::Positive,
            embeddings,
        }
    }

    /// Create a negative-label sequence from confusable, unrelated, ambient,
    /// or synthetic data.
    pub fn negative(
        id: UtteranceId,
        source: Source,
        augmentation_family: Option<AugmentationFamily>,
        embeddings: Vec<Vec<f32>>,
    ) -> Self {
        Self {
            id,
            source,
            augmentation_family,
            label_stratum: LabelStratum::Negative,
            embeddings,
        }
    }
}
