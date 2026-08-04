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
    pub sequence_index: usize,
    /// Variant index: 0=original, 1=speed-down, 2=speed-up,
    /// 3=volume-down, 4=noise. Always 0 for non-augmented sequences.
    pub variant_index: usize,
}

/// Provenance of an embedding sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    /// Original wake-word utterance (unmodified PCM).
    Enrollment,
    /// PCM-augmented variant of an enrollment utterance (speed, volume, noise).
    Augmentation,
    /// Pre-computed confusable near-miss phrase (TTS-synthesised).
    Confusable,
    /// Pre-computed unrelated speech phrase (TTS-synthesised).
    Unrelated,
    /// Ambient noise / non-speech environmental audio chunk.
    Ambient,
    /// Owner-general speech collected during Phase 3 enrollment (non-wake-word
    /// speech from the user, used as additional negative examples for the
    /// classifier; mahbot-932).
    Owner,
}

/// A group of per-frame 96-dim embedding vectors that share a common
/// provenance.
///
/// Windows are formed **within** each sequence only — never across
/// sequences — preventing the cross-utterance window contamination
/// that existed when training operated on flat `&[Vec<f32>]` lists.
#[derive(Debug, Clone)]
pub struct EmbeddingSequence {
    /// Provenance bookkeeping; read by the voice-tests benchmark.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub id: UtteranceId,
    /// Provenance bookkeeping; read by the voice-tests benchmark.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub source: Source,
    /// Per-frame 96-dim embedding vectors in temporal order.
    pub embeddings: Vec<Vec<f32>>,
}

impl EmbeddingSequence {
    /// Create a sequence with the given provenance and per-frame embeddings.
    pub fn new(id: UtteranceId, source: Source, embeddings: Vec<Vec<f32>>) -> Self {
        Self {
            id,
            source,
            embeddings,
        }
    }
}

/// Test-support helper: wrap flat embeddings into a single sequence with the
/// canonical enrollment provenance used by the wake-word classifier test
/// module (mahbot-1043).
#[cfg(test)]
pub(crate) fn make_test_sequence(embs: Vec<Vec<f32>>) -> EmbeddingSequence {
    EmbeddingSequence {
        id: UtteranceId {
            sequence_index: 0,
            variant_index: 0,
        },
        source: Source::Enrollment,
        embeddings: embs,
    }
}
