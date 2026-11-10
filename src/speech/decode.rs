//! The recogniser's input: a recording decoded to mono f32 at its own rate, which the engine
//! resamples.
//!
//! What a recording's codec is comes from its bytes, never from its name. Three codecs this process
//! reads itself — WAV of plain PCM or float, OGG/Opus (a voice note arrives as one) and MP3 — and
//! every other sound the product's media handling reads (Vorbis, AAC, ALAC, FLAC, AMR, companded or
//! ADPCM WAV, AIFF, CAF, WMA, AC-3) is converted by the product's own copy of ffmpeg first. That
//! copy is used only where it is already in place: asking for a transcription never installs one.
//!
//! [`crate::util::AUDIO_EXTENSIONS`] is the accept-set — the very names the media path reads as a
//! sound — so a file under one of them is taken even when its bytes name no codec this process
//! knows. The bytes decide the codec, and so whether a recording is decoded here or converted; the
//! name says only that a file is a recording at all, and, when the bytes say nothing, which decoder
//! this process already has for it. A conversion is one at a time, so a burst of attachments
//! converts one after another.

use anyhow::{Context as _, Result, bail};
use std::path::Path;
use tracing::warn;

/// The codecs this process decodes itself.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AudioFormat {
    Wav,
    Opus,
    Mp3,
}

/// What the recogniser makes of a recording.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Sound {
    /// A codec this process decodes itself.
    InProcess(AudioFormat),
    /// A sound this process has no decoder for: the product's media toolchain reads it.
    Managed,
}

/// The RIFF/WAVE encodings this process reads itself: plain PCM of 8, 16, 24 or 32 bits, and
/// 32-bit float.
const PCM: u16 = 1;
const FLOAT: u16 = 3;
/// The extensible form of the `fmt ` chunk, which names the real encoding in its sub-format GUID.
const EXTENSIBLE: u16 = 0xFFFE;

/// ASF's header GUID, which is what a WMA file begins with.
const ASF_MAGIC: [u8; 16] = [
    0x30, 0x26, 0xB2, 0x75, 0x8E, 0x66, 0xCF, 0x11, 0xA6, 0xD9, 0x00, 0xAA, 0x00, 0x62, 0xCE, 0x6C,
];

/// AC-3's sync word.
const AC3_SYNC: [u8; 2] = [0x0B, 0x77];

/// What a recording's bytes are, or `None` when they are not a sound at all.
fn sound_of(bytes: &[u8]) -> Option<Sound> {
    match after_id3(bytes) {
        // A tagged file: the tag says nothing about the codec, so the frames behind it decide — and
        // frames that name nothing are the ordinary tagged MP3, whose tag minimp3 skips.
        Some(frames) => sound_of_frames(frames).or(Some(Sound::InProcess(AudioFormat::Mp3))),
        None => sound_of_frames(bytes),
    }
}

/// The bytes after a leading ID3v2 tag, when the whole tag is there: its own header carries its
/// length, so the frames behind it are found exactly rather than guessed at.
fn after_id3(bytes: &[u8]) -> Option<&[u8]> {
    let size = bytes.get(6..10).filter(|_| bytes.starts_with(b"ID3"))?;
    let length = size.iter().fold(0usize, |length, byte| {
        (length << 7) | usize::from(byte & 0x7F)
    });
    bytes.get(length.saturating_add(10)..)
}

/// What the bytes themselves are, with no tag in front of them.
fn sound_of_frames(bytes: &[u8]) -> Option<Sound> {
    if is_wav(bytes) {
        // The encoding is in the `fmt ` chunk: plain PCM and float are decoded here, and anything
        // else — companded, ADPCM, a codec riding in a WAV — is the toolchain's.
        return Some(match wav_layout(bytes) {
            Some(wav) if wav.in_process() => Sound::InProcess(AudioFormat::Wav),
            _ => Sound::Managed,
        });
    }
    if bytes.starts_with(b"OggS") {
        return Some(ogg_sound(bytes));
    }
    if is_mpeg_sync(bytes) {
        return Some(mpeg_sound(bytes));
    }
    // Every other container the media path reads: the codec inside it is the toolchain's business.
    (bytes.starts_with(b"fLaC")
        || bytes.starts_with(b"caff")
        || bytes.starts_with(&ASF_MAGIC)
        || bytes.starts_with(&AC3_SYNC)
        || is_mp4(bytes)
        || is_aiff(bytes)
        || is_amr(bytes))
    .then_some(Sound::Managed)
}

/// Whether these bytes begin a RIFF/WAVE file.
fn is_wav(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE"
}

/// Whether these bytes begin an AIFF or AIFC file.
fn is_aiff(bytes: &[u8]) -> bool {
    bytes.len() >= 12
        && &bytes[0..4] == b"FORM"
        && (&bytes[8..12] == b"AIFF" || &bytes[8..12] == b"AIFC")
}

/// Whether these bytes begin a 3GPP AMR file.
fn is_amr(bytes: &[u8]) -> bool {
    bytes.starts_with(b"#!AMR\n") || bytes.starts_with(b"#!AMR-WB\n")
}

/// Whether these bytes begin an M4A or M4B recording: an MP4 family file, whose first box is
/// `ftyp` in every file written to the specification.
fn is_mp4(bytes: &[u8]) -> bool {
    matches!(bytes.get(4..8), Some(b"ftyp"))
}

/// Whether these bytes begin an MPEG audio frame: the frame sync, then a header whose version,
/// bitrate and sample rate are ones the format defines. The sync is only eleven bits — a text file
/// beginning with a byte-order mark carries it — so the rest of the header is checked as well.
fn is_mpeg_sync(bytes: &[u8]) -> bool {
    let (Some(first), Some(second), Some(third)) = (bytes.first(), bytes.get(1), bytes.get(2))
    else {
        return false;
    };
    *first == 0xFF
        && second & 0xE0 == 0xE0
        // The version field: 0b01 is reserved.
        && second & 0x18 != 0x08
        // The bitrate index: 0b1111 is not a bitrate.
        && third >> 4 != 0x0F
        // The sample rate index: 0b11 is reserved.
        && (third >> 2) & 0x03 != 0x03
}

/// Raw MPEG audio: layer III is MP3, which minimp3 reads here; the ADTS form of AAC (a zero layer
/// field) and the earlier layers are the toolchain's.
fn mpeg_sound(bytes: &[u8]) -> Sound {
    if bytes[1] & 0x06 == 0x02 {
        Sound::InProcess(AudioFormat::Mp3)
    } else {
        Sound::Managed
    }
}

/// An OGG stream's first packet names its codec: Opus is decoded here (a voice note is one), and
/// everything else an OGG can carry is the toolchain's.
fn ogg_sound(bytes: &[u8]) -> Sound {
    let first = ogg::reading::PacketReader::new(std::io::Cursor::new(bytes))
        .read_packet()
        .ok()
        .flatten();
    match first {
        Some(packet) if packet.data.starts_with(b"OpusHead") => Sound::InProcess(AudioFormat::Opus),
        _ => Sound::Managed,
    }
}

/// How much of a file is read to ask what it holds: every container's identification header sits
/// well inside this.
const HEAD: usize = 64 * 1024;

/// What the recogniser makes of this file: its bytes first, and, for bytes that name no codec, a
/// name the media path already reads as a sound — so a recording that path would read is never
/// refused here. `None` is a file that is not a recording at all.
pub(crate) fn sound_of_file(path: &Path) -> std::io::Result<Option<Sound>> {
    sniff(path).map(|(_, _, sound)| sound)
}

/// The bytes a sniff needed, whether they are the whole file, and what they name.
fn sniff(path: &Path) -> std::io::Result<(Vec<u8>, bool, Option<Sound>)> {
    let (head, whole) = read_head(path)?;
    // A WAV's codec is only in its `fmt ` chunk, which a chunk list running past the head hides:
    // such a file is read whole, so a WAV is never taken for something it is not.
    if !whole && is_wav(&head) && wav_layout(&head).is_none() {
        let bytes = std::fs::read(path)?;
        let sound = sound_of(&bytes).or_else(|| named_sound(path));
        return Ok((bytes, true, sound));
    }
    let sound = sound_of(&head).or_else(|| named_sound(path));
    Ok((head, whole, sound))
}

/// The file's first [`HEAD`] bytes, and whether they are the whole file.
fn read_head(path: &Path) -> std::io::Result<(Vec<u8>, bool)> {
    use std::io::Read as _;
    let mut head = Vec::with_capacity(HEAD);
    std::fs::File::open(path)?
        .take(HEAD as u64)
        .read_to_end(&mut head)?;
    let whole = head.len() < HEAD;
    Ok((head, whole))
}

/// The sound a file's name claims when its bytes name no codec: the decoder this process has for
/// that name, or, for a name only the toolchain reads, the toolchain's. A name says a file is a
/// recording and never which codec it holds — but a name the process already had a decoder for must
/// not come to depend on the toolchain because its bytes are not recognised.
fn named_sound(path: &Path) -> Option<Sound> {
    if !crate::util::is_audio_extension(path) {
        return None;
    }
    let name = path.extension()?.to_str()?.to_ascii_lowercase();
    Some(match name.as_str() {
        "wav" => Sound::InProcess(AudioFormat::Wav),
        "ogg" | "oga" | "opus" => Sound::InProcess(AudioFormat::Opus),
        "mp3" => Sound::InProcess(AudioFormat::Mp3),
        _ => Sound::Managed,
    })
}

/// One conversion at a time: a burst of recordings must not put several decodes on the machine.
static CONVERTING: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// A recording as the engine's input, or the marker that its codec is the toolchain's.
enum Read {
    /// Decoded here: the samples, and the rate they were taken at.
    Ready(Vec<f32>, u32),
    /// A sound this process has no decoder for.
    Convert,
}

/// The whole recording as the engine's input: mono f32 at its own rate, decoded here when the codec
/// is one this process reads and converted first when it is not.
pub(crate) async fn samples_of(path: &Path) -> Result<(Vec<f32>, u32)> {
    let owned = path.to_owned();
    match tokio::task::spawn_blocking(move || read(&owned))
        .await
        .context("Audio decode task panicked")??
    {
        Read::Ready(samples, rate) => Ok((samples, rate)),
        Read::Convert => convert(path).await,
    }
}

/// A recording read in one pass: the codec from its head, then the sound itself decoded here when
/// the codec is one this process reads.
fn read(path: &Path) -> Result<Read> {
    let (bytes, whole, sound) =
        sniff(path).with_context(|| format!("Failed to read audio file: {}", path.display()))?;
    match sound {
        None => bail!("{} is not a recording", path.display()),
        Some(Sound::Managed) => Ok(Read::Convert),
        Some(Sound::InProcess(format)) => {
            let bytes = if whole {
                bytes
            } else {
                std::fs::read(path)
                    .with_context(|| format!("Failed to read audio file: {}", path.display()))?
            };
            decode_in_process(&bytes, format, path)
                .map(|(samples, rate)| Read::Ready(samples, rate))
        }
    }
}

/// A file this process reads itself, decoded — a piece of a media read's sound track, which the
/// toolchain has already written as a WAV.
pub(crate) fn decode_file_in_process(path: &Path) -> Result<(Vec<f32>, u32)> {
    match read(path)? {
        Read::Ready(samples, rate) => Ok((samples, rate)),
        Read::Convert => bail!(
            "{} is not a sound this process reads itself",
            path.display()
        ),
    }
}

/// A sound this process has no decoder for, as the engine's input: the product's own copy of ffmpeg
/// converts it to the 16 kHz mono WAV the WAV decoder takes. The copy already in place is used, and
/// never one fetched for it, so a transcription downloads nothing; one conversion runs at a time.
/// The conversion carries ffmpeg's own bound — the engine's bound is the recognition pass's, as it
/// has always been.
async fn convert(path: &Path) -> Result<(Vec<f32>, u32)> {
    use crate::ffmpeg::FfmpegError;

    let audio = {
        let _one = CONVERTING.lock().await;
        crate::ffmpeg::convert_audio(path).await.map_err(|error| {
            let reason = match error {
                FfmpegError::Unavailable(reason) => reason,
                FfmpegError::NoAudioTrack | FfmpegError::NoVideoTrack => {
                    "it holds no sound to read".to_string()
                }
                FfmpegError::Undecodable => "it holds no sound the product can decode".to_string(),
                FfmpegError::TimedOut => {
                    "it could not be converted within the time a conversion is given".to_string()
                }
                FfmpegError::InvalidOptions(reason) => reason.to_string(),
            };
            anyhow::anyhow!("{} cannot be read: {reason}", path.display())
        })?
    };
    let converted = audio.path().to_owned();
    let owned = path.to_owned();
    tokio::task::spawn_blocking(move || {
        let bytes = std::fs::read(&converted).with_context(|| {
            format!("Failed to read the converted sound of {}", owned.display())
        })?;
        decode_in_process(&bytes, AudioFormat::Wav, &owned)
    })
    .await
    .context("Audio decode task panicked")?
}

/// Decode bytes this process reads itself to mono f32 at their own rate. A file that declares its
/// format but holds no frames decodes to nothing: silence is not a failure to read it.
fn decode_in_process(bytes: &[u8], format: AudioFormat, path: &Path) -> Result<(Vec<f32>, u32)> {
    if bytes.len() < 8 {
        bail!("Audio file too small: {}", path.display());
    }
    match format {
        AudioFormat::Wav => decode_wav(bytes, path),
        AudioFormat::Opus => decode_opus_from_ogg(bytes, path),
        AudioFormat::Mp3 => decode_mp3(bytes, path),
    }
}

/// A RIFF/WAVE file's chunks: the encoding its `fmt ` chunk declares and where its sound sits.
struct Wav {
    format: u16,
    channels: u16,
    rate: u32,
    bits: u16,
    /// The `data` chunk: where the sound starts and how long it is.
    sound: (usize, usize),
}

impl Wav {
    /// Whether this process decodes the encoding itself: plain PCM of 8, 16, 24 or 32 bits, or
    /// 32-bit float. Anything else — companded, ADPCM, a codec riding in a WAV — is converted.
    fn in_process(&self) -> bool {
        matches!(
            (self.format, self.bits),
            (PCM, 8 | 16 | 24 | 32) | (FLOAT, 32)
        )
    }
}

/// Walk a RIFF/WAVE file's chunk list rather than assuming it, so a file with metadata chunks
/// before (or after) its sound still reads. The last `data` chunk is the sound. `None` when the
/// bytes are not RIFF/WAVE, or declare no format or no sound.
fn wav_layout(bytes: &[u8]) -> Option<Wav> {
    if !is_wav(bytes) {
        return None;
    }
    let mut format = None;
    let mut channels = 0u16;
    let mut rate = 0u32;
    let mut bits = 0u16;
    let mut sound = None;
    let mut at = 12usize;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap_or_default()) as usize;
        let body = at + 8;
        match id {
            b"fmt " if len >= 16 && body + 16 <= bytes.len() => {
                format = Some(u16::from_le_bytes([bytes[body], bytes[body + 1]]));
                channels = u16::from_le_bytes([bytes[body + 2], bytes[body + 3]]);
                rate = u32::from_le_bytes(bytes[body + 4..body + 8].try_into().unwrap_or_default());
                bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]);
                // The extensible form names the real encoding in its sub-format GUID.
                if format == Some(EXTENSIBLE) && len >= 26 && body + 26 <= bytes.len() {
                    format = Some(u16::from_le_bytes([bytes[body + 24], bytes[body + 25]]));
                }
            }
            b"data" => {
                let len = len.min(bytes.len().saturating_sub(body));
                sound = Some((body, len));
            }
            _ => {}
        }
        at = body.saturating_add(len + (len & 1));
    }
    Some(Wav {
        format: format?,
        channels,
        rate,
        bits,
        sound: sound?,
    })
}

/// A RIFF/WAVE file into mono f32 at its declared rate: PCM 8/16/24/32-bit and IEEE float 32,
/// WAVE_FORMAT_EXTENSIBLE included, any channel count averaged to mono.
#[expect(clippy::cast_precision_loss)]
fn decode_wav(data: &[u8], path: &Path) -> Result<(Vec<f32>, u32)> {
    let Some(wav) = wav_layout(data) else {
        bail!("{} is not a RIFF/WAVE file holding a sound", path.display());
    };
    if wav.channels == 0 || wav.rate == 0 {
        bail!(
            "{} declares {} channels at {} Hz",
            path.display(),
            wav.channels,
            wav.rate
        );
    }
    let Wav {
        format,
        channels,
        rate,
        bits,
        sound: (from, len),
    } = wav;
    // The encoding is the whole of the decoder: the router sends a WAV here only when it holds one
    // of these, and anything else is refused rather than read as samples it is not.
    let sample: fn(&[u8]) -> f32 = match (format, bits) {
        (PCM, 8) => |bytes| f32::from(bytes[0]) / 128.0 - 1.0,
        (PCM, 16) => |bytes| f32::from(i16::from_le_bytes([bytes[0], bytes[1]])) / 32768.0,
        (PCM, 24) => |bytes| {
            (i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]]) >> 8) as f32 / 8_388_608.0
        },
        (PCM, 32) => |bytes| {
            i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32 / 2_147_483_648.0
        },
        (FLOAT, 32) => |bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        _ => bail!(
            "{} is not PCM or float audio (format {}, {} bits)",
            path.display(),
            format,
            bits
        ),
    };
    let width = usize::from(bits / 8);
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
            sum += sample(&data[at..at + width]);
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

    /// An OGG stream whose first packet is `packet`, written by the crate's own muxer so the page
    /// is one a reader really accepts.
    fn ogg(packet: &[u8]) -> Vec<u8> {
        use ogg::writing::{PacketWriteEndInfo, PacketWriter};
        let mut out = Vec::new();
        let mut writer = PacketWriter::new(std::io::Cursor::new(&mut out));
        writer
            .write_packet(packet, 0, PacketWriteEndInfo::EndPage, 0)
            .expect("an ogg page");
        out
    }

    /// The bytes of a container whose codec this process has no decoder for: only the header is
    /// needed, because the codec is all that is asked of it.
    fn mp4() -> Vec<u8> {
        let mut out = vec![0, 0, 0, 0x20];
        out.extend_from_slice(b"ftypM4A ");
        out
    }

    /// `frames` behind an ID3v2 tag of `tag` bytes, as a tagged file carries them.
    fn tagged(tag: usize, frames: &[u8]) -> Vec<u8> {
        let mut out = b"ID3\x04\x00\x00".to_vec();
        let size = u32::try_from(tag).expect("a small tag");
        // The tag's length, seven bits to a byte, most significant first.
        let byte = |shift: u32| u8::try_from((size >> shift) & 0x7F).expect("seven bits");
        out.extend_from_slice(&[byte(21), byte(14), byte(7), byte(0)]);
        out.resize(out.len() + tag, 0);
        out.extend_from_slice(frames);
        out
    }

    #[test]
    fn a_wav_reads_at_its_own_rate_and_channels() {
        let path = Path::new("a.wav");
        for (bits, format, channels, rate) in [
            (16u16, 1u16, 1u16, 16_000u32),
            (16, 1, 2, 44_100),
            (8, 1, 1, 8_000),
            (24, 1, 1, 48_000),
            (32, 1, 1, 48_000),
            (32, 3, 2, 48_000),
        ] {
            let bytes = wav(bits, format, channels, rate, 400);
            assert_eq!(sound_of(&bytes), Some(Sound::InProcess(AudioFormat::Wav)));
            let (samples, read) = decode_in_process(&bytes, AudioFormat::Wav, path)
                .unwrap_or_else(|e| panic!("{bits}-bit format {format}: {e:#}"));
            assert_eq!(read, rate, "{bits}-bit format {format}");
            assert_eq!(samples.len(), 400);
        }
    }

    #[test]
    fn a_wav_with_no_frames_decodes_to_nothing() {
        // What ffmpeg writes for a piece cut past the end of the sound: a valid WAV whose data
        // chunk is empty. It holds no sound, which is not a failure to read it.
        let bytes = wav(16, 1, 1, 16_000, 0);
        let (samples, rate) = decode_in_process(&bytes, AudioFormat::Wav, Path::new("a.wav"))
            .expect("an empty wav is not an error");
        assert!(samples.is_empty());
        assert_eq!(rate, 16_000);
    }

    /// A tag in front of the frames says nothing about the codec: the frames behind it decide, and
    /// frames that name nothing are the ordinary tagged MP3.
    #[test]
    fn a_tag_never_decides_the_codec() {
        let adts = [0xFF, 0xF1, 0x50, 0x80, 0, 0, 0, 0];
        assert_eq!(sound_of(&tagged(64, &adts)), Some(Sound::Managed));
        let mp3 = [0xFF, 0xFB, 0x90, 0x00, 0, 0, 0, 0];
        assert_eq!(
            sound_of(&tagged(64, &mp3)),
            Some(Sound::InProcess(AudioFormat::Mp3))
        );
        // A tag whose frames name nothing at all is the ordinary tagged MP3: minimp3 skips the tag
        // and reads the frames behind it.
        assert_eq!(
            sound_of(&tagged(64, b"\0\0\0\0")),
            Some(Sound::InProcess(AudioFormat::Mp3))
        );
        // A tag that claims more bytes than the file holds hides its frames: the bytes name no
        // codec, and the name the file carries decides.
        assert_eq!(sound_of(b"ID3\x04\x00\x00\x7F\x7F\x7F\x7F"), None);
    }

    /// A WAV the process does not decode itself is a WAV the toolchain reads: companded, ADPCM, a
    /// PCM width the decoder does not take, and the extensible form naming one of those.
    #[test]
    fn a_wav_that_is_not_plain_pcm_is_converted() {
        for (format, bits) in [
            (6u16, 8u16), // A-law
            (7, 8),       // µ-law
            (2, 4),       // MS ADPCM
            (0x11, 4),    // IMA ADPCM
            (1, 20),      // PCM packed 20 bits to three bytes
            (0x55, 0),    // MP3 riding in a WAV
            (0xFFFE, 16), // extensible with no sub-format the walker can read
        ] {
            let bytes = wav(bits, format, 1, 8_000, 4);
            assert_eq!(
                sound_of(&bytes),
                Some(Sound::Managed),
                "format {format}, {bits} bits"
            );
        }
    }

    /// An OGG stream's first packet names its codec: Opus is decoded here (a voice note is one),
    /// and Vorbis is the toolchain's.
    #[test]
    fn the_codec_inside_decides_not_the_name() {
        assert_eq!(
            sound_of(&ogg(
                b"OpusHead\x01\x01\x00\x00\x80\xbb\x00\x00\x00\x00\x00"
            )),
            Some(Sound::InProcess(AudioFormat::Opus))
        );
        assert_eq!(sound_of(&ogg(b"\x01vorbis\0\0\0\0")), Some(Sound::Managed));
    }

    /// Every container the media path reads as a sound is a recording here too, whatever its name
    /// and whatever its codec: the accept-set is the media path's own table.
    #[test]
    fn every_name_the_media_path_reads_as_audio_is_a_recording() {
        let opus = ogg(b"OpusHead\x01\x01\x00\x00\x80\xbb\x00\x00\x00\x00\x00");
        let pcm = wav(16, PCM, 1, 16_000, 8);
        let m4a = mp4();
        let headers: [(&str, &[u8]); 15] = [
            ("wav", &pcm),
            ("mp3", &[0xFF, 0xFB, 0x90, 0x00, 0, 0, 0, 0]),
            ("ogg", &opus),
            ("oga", &opus),
            ("opus", &opus),
            ("m4a", &m4a),
            ("m4b", &m4a),
            ("aac", &[0xFF, 0xF1, 0x50, 0x80, 0, 0, 0, 0]),
            ("flac", b"fLaC\0\0\0\x22"),
            ("wma", &ASF_MAGIC),
            ("aiff", b"FORM\0\0\0\0AIFF"),
            ("aif", b"FORM\0\0\0\0AIFC"),
            ("amr", b"#!AMR\n"),
            ("caf", b"caff\0\x01\0\0"),
            ("ac3", &AC3_SYNC),
        ];
        // A name added to the accept-set must come with a header here, or this says so.
        let mut named: Vec<&str> = headers.iter().map(|(name, _)| *name).collect();
        named.sort_unstable();
        let mut accept = crate::util::AUDIO_EXTENSIONS.to_vec();
        accept.sort_unstable();
        assert_eq!(named, accept);
        for (name, header) in headers {
            assert!(sound_of(header).is_some(), "{name} must be a recording");
        }
        // The names this process has always decoded are still its own: an unrecognised file under
        // one of them must never come to depend on the toolchain.
        for name in ["wav", "ogg", "oga", "opus", "mp3"] {
            let path = Path::new("a").with_extension(name);
            assert!(
                matches!(named_sound(&path), Some(Sound::InProcess(_))),
                "{name} is decoded here, never converted"
            );
        }
        // A name the media path does not read is not a recording's name at all.
        assert_eq!(named_sound(Path::new("a.txt")), None);
    }

    /// A recording this process has no decoder for is converted by the copy already in place and
    /// then read: the stand-in pair here writes the 16 kHz mono WAV the real one writes, so the whole
    /// route — sniffed by its bytes, converted, decoded — is exercised without a real toolchain and
    /// without the speech model. The release stream answers nothing, so a conversion that reached for
    /// an install instead of the copy in place would fail here.
    #[cfg(unix)]
    #[serial_test::serial(ffmpeg_live, ffmpeg_releases)]
    #[tokio::test]
    async fn a_sound_with_no_decoder_here_is_converted_by_the_copy_in_place() {
        let home = tempfile::tempdir().expect("temp home");
        let root = home.path().join("ffmpeg");
        let _root = crate::util::test::set_env_var(crate::ffmpeg::ROOT_ENV, root.to_str());
        let _hook = crate::ffmpeg::ReleasesHook::at("http://127.0.0.1:1");

        // What the real pair writes for a recording: 16 kHz mono PCM WAV.
        let fixture = home.path().join("converted.wav");
        std::fs::write(&fixture, wav(16, PCM, 1, 16_000, 400)).expect("a wav");
        let bin = root.join("ffmpeg-v7.1-mahbot.2").join("bin");
        std::fs::create_dir_all(&bin).expect("install directory");
        // The last argument is where the conversion writes; the fixture is what it writes there.
        let stand_in = format!(
            "#!/bin/sh\nfor argument in \"$@\"; do into=$argument; done\ncat '{}' > \"$into\"\n",
            fixture.display()
        );
        for tool in ["ffmpeg", "ffprobe"] {
            let path = bin.join(tool);
            std::fs::write(&path, &stand_in).expect("a stand-in tool");
            crate::util::test::make_executable(&path);
        }

        let recording = home.path().join("note.m4a");
        std::fs::write(&recording, mp4()).expect("a recording");
        let (samples, rate) = samples_of(&recording)
            .await
            .expect("the copy in place converts it");
        assert_eq!(rate, 16_000);
        assert_eq!(samples.len(), 400);
    }

    #[test]
    fn a_file_that_is_not_a_recording_is_refused() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, b"not an audio file").expect("a file");
        assert_eq!(sound_of_file(&path).expect("a readable file"), None);
        // The bytes name the codec wherever they are, whatever the file is called.
        let path = dir.path().join("recording.bin");
        std::fs::write(&path, mp4()).expect("a file");
        assert_eq!(
            sound_of_file(&path).expect("a readable file"),
            Some(Sound::Managed)
        );
        assert!(sound_of_file(&dir.path().join("absent.wav")).is_err());
    }

    /// A file under a name the media path reads is a recording even when its bytes name no codec
    /// this process knows — and the bytes still decide the codec, so a name never overrides one
    /// they name.
    #[test]
    fn a_files_name_says_what_it_is_but_never_what_it_holds() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let file = |name: &str, bytes: &[u8]| {
            let path = dir.path().join(name);
            std::fs::write(&path, bytes).expect("a file");
            path
        };
        // Bytes that name no codec: the name says what the file is, and which decoder it gets.
        assert_eq!(
            sound_of_file(&file("a.m4a", b"\0\0\0\0\0\0\0\0")).expect("a file"),
            Some(Sound::Managed)
        );
        assert_eq!(
            sound_of_file(&file("a.wav", b"\0\0\0\0\0\0\0\0")).expect("a file"),
            Some(Sound::InProcess(AudioFormat::Wav))
        );
        // Bytes that name one decide it, whatever the name says: Vorbis in an .ogg, A-law in a .wav.
        assert_eq!(
            sound_of_file(&file("a.ogg", &ogg(b"\x01vorbis\0\0\0\0"))).expect("a file"),
            Some(Sound::Managed)
        );
        assert_eq!(
            sound_of_file(&file("a.wav", &wav(8, 6, 1, 8_000, 8))).expect("a file"),
            Some(Sound::Managed)
        );
    }
}
