//! The recogniser's input: a recording decoded to mono f32 at its own rate, which
//! the engine resamples.
//!
//! Three containers, the same three the product has always taken: WAV, OGG/Opus
//! (a Telegram voice note) and MP3 (an audio attachment). [`AUDIO_FORMATS`] is the
//! one table of that rule — the decoder dispatches through it, [`takes_audio`]
//! accepts a file by it and a caller's refusal lists it — so a format can never be
//! taken by one and refused by another. A file under a name outside the table is
//! taken when its bytes begin with OGG's magic. This is the in-process decoder's table;
//! what the managed ffmpeg reads as a sound track is `crate::util::AUDIO_EXTENSIONS`.

use anyhow::{Context as _, Result, bail};
use std::path::Path;
use tracing::warn;

/// The audio names the recogniser takes and how each is decoded.
pub(crate) const AUDIO_FORMATS: &[(&str, AudioFormat)] = &[
    ("wav", AudioFormat::Wav),
    ("ogg", AudioFormat::Opus),
    ("oga", AudioFormat::Opus),
    ("mp3", AudioFormat::Mp3),
];

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AudioFormat {
    Wav,
    Opus,
    Mp3,
}

/// The format [`AUDIO_FORMATS`] names this file's extension with, if any.
fn format_for(path: &Path) -> Option<AudioFormat> {
    let extension = path.extension()?.to_str()?;
    AUDIO_FORMATS
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(extension))
        .map(|(_, format)| *format)
}

/// Whether the recogniser takes this file — its own rule, and only it: a name
/// [`AUDIO_FORMATS`] holds, or, under a name the decoder does not know, content
/// beginning with Ogg's magic. A known name is settled by the NAME.
pub(crate) fn takes_audio(path: &Path) -> std::io::Result<bool> {
    let mut file = std::fs::File::open(path)?;
    if format_for(path).is_some() {
        return Ok(true);
    }
    let mut magic = [0u8; 4];
    Ok(std::io::Read::read_exact(&mut file, &mut magic).is_ok_and(|()| magic == *b"OggS"))
}

/// Decode any supported file to mono f32 at its own sample rate. A file that declares its format
/// but holds no frames decodes to nothing: silence is not a failure to read it.
pub(crate) fn decode_audio_to_mono_f32(path: &Path) -> Result<(Vec<f32>, u32)> {
    let format = format_for(path);

    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read audio file: {}", path.display()))?;

    if data.len() < 8 {
        bail!("Audio file too small: {}", path.display());
    }

    match format {
        Some(AudioFormat::Wav) => decode_wav(&data, path),
        Some(AudioFormat::Opus) => decode_opus_from_ogg(&data, path),
        Some(AudioFormat::Mp3) => decode_mp3(&data, path),
        // For a name no format is named with, try OGG/Opus: its magic bytes are the only
        // thing that can make such a file audio.
        None if data.starts_with(b"OggS") => decode_opus_from_ogg(&data, path),
        None => bail!(
            "Unsupported audio format '.{ext}' — only WAV, OGG/Opus, and MP3 are supported",
            ext = path.extension().and_then(|e| e.to_str()).unwrap_or("")
        ),
    }
}

/// A RIFF/WAVE file into mono f32 at its declared rate: PCM 8/16/24/32-bit and IEEE
/// float 32, WAVE_FORMAT_EXTENSIBLE included, any channel count averaged to mono.
/// The last `data` chunk is the audio, and a chunk list walked rather than assumed,
/// so a file with metadata chunks before (or after) its audio still reads.
#[expect(clippy::cast_precision_loss)]
fn decode_wav(data: &[u8], path: &Path) -> Result<(Vec<f32>, u32)> {
    const PCM: u16 = 1;
    const FLOAT: u16 = 3;
    const EXTENSIBLE: u16 = 0xFFFE;

    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        bail!("{} is not a RIFF/WAVE file", path.display());
    }
    let mut format = None;
    let mut channels = 0u16;
    let mut rate = 0u32;
    let mut bits = 0u16;
    let mut audio: Option<(usize, usize)> = None;
    let mut at = 12usize;
    while at + 8 <= data.len() {
        let id = &data[at..at + 4];
        let len = u32::from_le_bytes(data[at + 4..at + 8].try_into().unwrap_or_default()) as usize;
        let body = at + 8;
        match id {
            b"fmt " if len >= 16 && body + 16 <= data.len() => {
                format = Some(u16::from_le_bytes([data[body], data[body + 1]]));
                channels = u16::from_le_bytes([data[body + 2], data[body + 3]]);
                rate = u32::from_le_bytes(data[body + 4..body + 8].try_into().unwrap_or_default());
                bits = u16::from_le_bytes([data[body + 14], data[body + 15]]);
                // The extensible form names the real encoding in its sub-format GUID.
                if format == Some(EXTENSIBLE) && len >= 26 && body + 26 <= data.len() {
                    format = Some(u16::from_le_bytes([data[body + 24], data[body + 25]]));
                }
            }
            b"data" => {
                let len = len.min(data.len().saturating_sub(body));
                audio = Some((body, len));
            }
            _ => {}
        }
        at = body.saturating_add(len + (len & 1));
    }

    let (Some(format), Some((from, len))) = (format, audio) else {
        bail!("{} has no format or no audio", path.display());
    };
    if channels == 0 || rate == 0 {
        bail!(
            "{} declares {} channels at {rate} Hz",
            path.display(),
            channels
        );
    }
    let width = usize::from(bits / 8);
    if width == 0 || !matches!((format, bits), (PCM, 8 | 16 | 24 | 32) | (FLOAT, 32)) {
        bail!(
            "{} is not PCM or float audio (format {format}, {bits} bits)",
            path.display()
        );
    }
    let frames = len / (width * usize::from(channels));
    // A WAV whose data chunk holds no frames is a valid file holding no sound — a piece cut past
    // the end of the track is exactly that — so it decodes to nothing rather than failing the
    // recording it was cut from.
    if frames == 0 {
        return Ok((Vec::new(), rate));
    }
    let channels = usize::from(channels);
    let mut samples = Vec::with_capacity(frames);
    for frame in 0..frames {
        let mut sum = 0f32;
        for channel in 0..channels {
            let at = from + (frame * channels + channel) * width;
            let bytes = &data[at..at + width];
            sum += match (format, bits) {
                (PCM, 8) => f32::from(bytes[0]) / 128.0 - 1.0,
                (PCM, 16) => f32::from(i16::from_le_bytes([bytes[0], bytes[1]])) / 32768.0,
                (PCM, 24) => {
                    let value = i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]]) >> 8;
                    value as f32 / 8_388_608.0
                }
                (PCM, 32) => {
                    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32
                        / 2_147_483_648.0
                }
                _ => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            };
        }
        samples.push(sum / channels as f32);
    }
    Ok((samples, rate))
}

/// Decode an OGG/Opus file into mono f32 samples at 16 kHz: `ogg` demuxing and the
/// pure-Rust `opus_decoder` (no C dependencies, no unsafe code). The decoder is built
/// at 16 kHz, which is what the engine reads, so the file is never resampled.
fn decode_opus_from_ogg(data: &[u8], path: &Path) -> Result<(Vec<f32>, u32)> {
    use ogg::reading::PacketReader;
    use std::io::Cursor;

    let cursor = Cursor::new(data);
    let mut reader = PacketReader::new(cursor);
    let sample_rate: u32 = 16_000;
    let mut decoder: Option<opus_decoder::OpusDecoder> = None;
    let mut channels: usize = 1; // Will be set from OpusHead
    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match reader.read_packet() {
            Ok(Some(pkt)) => pkt,
            Ok(None) => break, // End of stream
            Err(e) => {
                warn!(path = %path.display(), error = %e, "OGG demux error, stopping");
                break;
            }
        };

        let packet_data = packet.data;

        if decoder.is_none() {
            if packet_data.starts_with(b"OpusHead") {
                if packet_data.len() < 18 {
                    bail!("Invalid Opus identification header (too short)");
                }
                channels = packet_data[9] as usize;
                match opus_decoder::OpusDecoder::new(sample_rate, channels) {
                    Ok(d) => {
                        decoder = Some(d);
                    }
                    Err(e) => {
                        bail!("Failed to create Opus decoder: {e:?}");
                    }
                }
            } else {
                bail!("OGG file does not contain Opus data");
            }
            continue;
        }

        // Skip the OpusTags comment header — metadata, not audio: feeding it to decode_float
        // produces a spurious decode error.
        if packet_data.starts_with(b"OpusTags") {
            continue;
        }

        if packet_data.is_empty() {
            continue; // Empty packet
        }

        // Decode to f32. Max packet: ~120 ms at 48 kHz stereo = 5760 samples/channel; the buffer
        // has margin for multichannel.
        let max_pcm_len = 5760 * channels.max(2);
        let mut pcm = vec![0.0f32; max_pcm_len];
        let Some(dec) = decoder.as_mut() else {
            bail!("Opus decoder not initialized — missing OpusHead header");
        };
        match dec.decode_float(&packet_data, &mut pcm, false) {
            Ok(n_per_channel) => {
                if channels == 1 {
                    samples.extend_from_slice(&pcm[..n_per_channel]);
                } else {
                    // Stereo → mono by averaging the midpoint: no overflow, and it rounds
                    // identically to (l + r) * 0.5 for normal samples.
                    for i in 0..n_per_channel {
                        samples.push(pcm[i * 2].midpoint(pcm[i * 2 + 1]));
                    }
                }
            }
            Err(e) => {
                warn!(path = %path.display(), error = ?e, "Opus decode error, skipping packet");
            }
        }
    }

    if samples.is_empty() {
        bail!("No audio decoded from {}", path.display());
    }

    Ok((samples, sample_rate))
}

/// Decode an MP3 file into mono f32 samples at the file's sample rate through
/// `minimp3` — a Rust wrapper over the minimp3 C library, so this is a C compiler build
/// dependency.
#[expect(clippy::cast_precision_loss)]
fn decode_mp3(data: &[u8], path: &Path) -> Result<(Vec<f32>, u32)> {
    use minimp3::Decoder as Mp3Decoder;

    let mut decoder = Mp3Decoder::new(data);
    let mut samples: Vec<f32> = Vec::new();
    let mut sample_rate: u32 = 0;

    loop {
        match decoder.next_frame() {
            Ok(frame) => {
                if sample_rate == 0 {
                    sample_rate = u32::try_from(frame.sample_rate).unwrap_or_default();
                }

                let n_ch = frame.channels;
                if n_ch == 1 {
                    for &val in &frame.data {
                        samples.push(f32::from(val) / 32768.0);
                    }
                } else {
                    for chunk in frame.data.chunks(n_ch) {
                        let mono: f32 = chunk.iter().map(|&v| f32::from(v)).sum::<f32>()
                            / n_ch as f32
                            / 32768.0;
                        samples.push(mono);
                    }
                }
            }
            Err(minimp3::Error::Eof) => break,
            Err(e) => {
                warn!(path = %path.display(), error = %e, "MP3 decode error, stopping");
                break;
            }
        }
    }

    if samples.is_empty() {
        bail!("No audio decoded from {}", path.display());
    }
    if sample_rate == 0 {
        bail!("{} declares no sample rate", path.display());
    }

    Ok((samples, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A RIFF/WAVE file of `frames` frames: `bits` per sample, `channels` channels,
    /// every frame carrying the same value.
    fn wav(bits: u16, format: u16, channels: u16, rate: u32, frames: usize) -> Vec<u8> {
        let width = usize::from(bits / 8);
        let data = vec![0u8; frames * width * usize::from(channels)];
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(
            &u32::try_from(36 + data.len())
                .expect("a small wav")
                .to_le_bytes(),
        );
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&format.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * u32::from(channels) * u32::from(bits) / 8).to_le_bytes());
        out.extend_from_slice(&(channels * bits / 8).to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(
            &u32::try_from(data.len())
                .expect("a small wav")
                .to_le_bytes(),
        );
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn a_wav_reads_at_its_own_rate_and_channels() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("a.wav");
        for (bits, format, channels, rate) in [
            (16u16, 1u16, 1u16, 16_000u32),
            (16, 1, 2, 44_100),
            (8, 1, 1, 8_000),
            (24, 1, 1, 48_000),
            (32, 1, 1, 48_000),
            (32, 3, 2, 48_000),
        ] {
            std::fs::write(&path, wav(bits, format, channels, rate, 400)).expect("a wav");
            let (samples, read) = decode_audio_to_mono_f32(&path)
                .unwrap_or_else(|e| panic!("{bits}-bit format {format}: {e:#}"));
            assert_eq!(read, rate, "{bits}-bit format {format}");
            assert_eq!(samples.len(), 400);
        }
    }

    #[test]
    fn a_wav_with_no_frames_decodes_to_nothing() {
        // What ffmpeg writes for a piece cut past the end of the sound: a valid WAV whose data
        // chunk is empty. It holds no sound, which is not a failure to read it.
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("a.wav");
        std::fs::write(&path, wav(16, 1, 1, 16_000, 0)).expect("a wav");
        let (samples, rate) =
            decode_audio_to_mono_f32(&path).expect("an empty wav is not an error");
        assert!(samples.is_empty());
        assert_eq!(rate, 16_000);
    }

    #[test]
    fn a_file_that_is_not_audio_is_refused() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("a.wav");
        std::fs::write(&path, b"not a wav at all").expect("a file");
        assert!(decode_audio_to_mono_f32(&path).is_err());
        let path = dir.path().join("a.xyz");
        std::fs::write(&path, b"not an audio file").expect("a file");
        assert!(decode_audio_to_mono_f32(&path).is_err());
    }

    #[test]
    fn the_accepted_names_are_the_ones_decoded() {
        let dir = tempfile::tempdir().expect("a temp dir");
        for (name, _) in AUDIO_FORMATS {
            let path = dir.path().join(format!("a.{name}"));
            std::fs::write(&path, b"\0\0\0\0\0\0\0\0").expect("a file");
            assert!(takes_audio(&path).expect("a readable file"), "{name}");
        }
        // A name the table does not hold is taken only for Ogg's magic.
        let path = dir.path().join("a.unknown");
        std::fs::write(&path, b"OggS\0\0\0\0").expect("a file");
        assert!(takes_audio(&path).expect("a readable file"));
        std::fs::write(&path, b"RIFF\0\0\0\0").expect("a file");
        assert!(!takes_audio(&path).expect("a readable file"));
        assert!(takes_audio(&dir.path().join("absent.wav")).is_err());
    }
}
