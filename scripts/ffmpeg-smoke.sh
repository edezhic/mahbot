#!/bin/sh
# Prove what the product downloads, on the pair that will be published. Invoked
# by `.github/workflows/ffmpeg-release.yml` on every leg of the release.
#
#   ffmpeg-smoke.sh <bin-dir> <scratch-dir> <https-url>
#
# <bin-dir> holds the freshly built `ffmpeg` and `ffprobe`, and everything below is
# written into `<scratch-dir>/ffmpeg-smoke`, a directory of this script's own.
# <https-url> is a file this leg fetches over https and turns into output: a build
# that cannot open the protocol, or whose TLS backend cannot complete a handshake,
# fails here.
#
# Every capability the product promises is proven by producing output with it,
# never by a name appearing in a list: H.264, HEVC and VP9 encoding, Opus
# encoding, PNG encoding and decoding, the zscale filter, an https fetch whose peer
# is verified against the machine's own store, and the three jobs the product itself
# runs (probe, JPEG frames, 16 kHz mono WAV).
set -eu

if [ $# -lt 3 ]; then
    echo "usage: ffmpeg-smoke.sh <bin-dir> <scratch-dir> <https-url>" >&2
    exit 1
fi

bin=$1
scratch=$2
url=$3

case $(uname -s) in
*_NT-*) exe=.exe ;;
*) exe= ;;
esac
ffmpeg=$bin/ffmpeg$exe
ffprobe=$bin/ffprobe$exe

fail() {
    echo "::error::$*" >&2
    exit 1
}

# Everything below is written into a directory of this script's own inside the caller's
# scratch path, so a re-run never empties a directory it was merely pointed at.
[ -n "$scratch" ] || fail "a scratch directory is required"
scratch=$scratch/ffmpeg-smoke
rm -rf "$scratch"
mkdir -p "$scratch"

# ── What the pair says about itself ─────────────────────────────────

banner=$("$ffmpeg" -version 2>&1 | tr -d '\r') || fail "ffmpeg -version does not run"
for option in --enable-gpl --enable-version3 --enable-zlib --enable-libx264 \
    --enable-libx265 --enable-libvpx --enable-libopus --enable-libzimg; do
    printf '%s\n' "$banner" | grep -q -- "$option" ||
        fail "ffmpeg -version carries no $option, so its configuration line is not this build's"
done
if printf '%s\n' "$banner" | grep -q -- '--enable-nonfree'; then
    fail "ffmpeg was configured with --enable-nonfree"
fi
if ! printf '%s\n' "$banner" | grep -Eq -- '--enable-(openssl|schannel)'; then
    fail "ffmpeg -version names no TLS backend, so https cannot be there"
fi

# ── What the pair needs from the machine it runs on ─────────────────

case $(uname -s) in
Darwin*)
    libs=$(otool -L "$ffmpeg" | tail -n +2 | awk '{print $1}' | sort -u)
    foreign=$(printf '%s\n' "$libs" | grep -vE '^(/usr/lib/|/System/Library/)' || true)
    ;;
Linux*)
    libs=$(ldd "$ffmpeg" 2>&1 | grep -oE '/[^ ]+' | sort -u || true)
    foreign=$(printf '%s\n' "$libs" | grep -vE '^/(lib|lib64|usr/lib)(/|$)' || true)
    ;;
*)
    # The mingw runtime is what `-static` was for: a libwinpthread or a
    # libgcc/libstdc++/libc++ DLL in the imports would mean an exe that only
    # runs where the toolchain is installed.
    libs=$(objdump -p "$ffmpeg" | sed -nE 's/^[[:space:]]*DLL Name: (.*)$/\1/p' | sort -u)
    foreign=$(printf '%s\n' "$libs" | grep -iE \
        '^(libwinpthread|libgcc_s|libunwind|libstdc\+\+|libc\+\+|libssp|zlib1|libbz2|libiconv|libssl|libcrypto)' || true)
    ;;
esac
[ -z "$foreign" ] || fail "ffmpeg needs libraries that are not the system's:
$foreign"
echo "libraries, all the system's own: $(printf '%s' "$libs" | tr '\n' ' ')"

# ── Every format the product accepts, and the two containers it used to skip

"$ffprobe" -hide_banner -demuxers | tr ' ,' '\n\n' >"$scratch/demuxers"
for demuxer in mov matroska avi asf mpeg ogg wav mp3 flac; do
    grep -qx "$demuxer" "$scratch/demuxers" ||
        fail "the pair cannot read $demuxer"
done
"$ffprobe" -hide_banner -decoders | tr ' ,' '\n\n' >"$scratch/decoders"
for decoder in h264 hevc vp8 vp9 av1 mpeg4 aac vorbis opus flac pcm_s16le mjpeg; do
    grep -qx "$decoder" "$scratch/decoders" ||
        fail "the pair cannot decode $decoder"
done
# The MP3 decoder is `mp3float` wherever the float one is the only one.
grep -qxE 'mp3|mp3float' "$scratch/decoders" || fail "the pair cannot decode mp3"

# ── The clip every check below is made from ─────────────────────────

"$ffmpeg" -v error -y \
    -f lavfi -i "testsrc=size=320x240:rate=25:duration=1" \
    -f lavfi -i "sine=frequency=440:duration=1" \
    -c:v mpeg4 -c:a pcm_s16le -shortest "$scratch/clip.mov"

codec_of() {
    "$ffprobe" -v error -select_streams "$1" -show_entries stream=codec_name \
        -of csv=p=0 "$2" | tr -d '\r'
}

# expect_video <what> <codec> <file> — the codec of the file's first video stream.
expect_video() {
    got=$(codec_of v:0 "$3")
    [ "$got" = "$2" ] || fail "$1: the output reports '$got', not '$2'"
    echo "$1: $got"
}

# size_of <file> — the pixel size of the file's first video stream, `w,h`.
size_of() {
    "$ffprobe" -v error -select_streams v:0 -show_entries stream=width,height \
        -of csv=p=0 "$1" | tr -d '\r'
}
# expect_size <what> <file> <w,h> — the pixel size the product's own output must have.
expect_size() {
    got=$(size_of "$2")
    [ "$got" = "$3" ] || fail "$1: the output is $got, not $3"
    echo "$1: $got"
}
# expect_picture <what> <file> — the file decodes to a picture of non-zero size, whatever
# that size is: the fetched file's dimensions are not this script's to decide. Only
# ffprobe's own `width,height` shape is accepted, both numbers whole and above zero.
expect_picture() {
    got=$(size_of "$2")
    case $got in
    *,*) ;;
    *) fail "$1: the output did not decode to a picture ($got)" ;;
    esac
    width=${got%%,*}
    height=${got#*,}
    case $width in
    '' | *[!0-9]*) fail "$1: the output did not decode to a picture ($got)" ;;
    esac
    case $height in
    '' | *[!0-9]*) fail "$1: the output did not decode to a picture ($got)" ;;
    esac
    { [ "$width" -gt 0 ] && [ "$height" -gt 0 ]; } ||
        fail "$1: the output did not decode to a picture of non-zero size ($got)"
    echo "$1: $got"
}

# ── Encoding, each one proven by the stream it wrote ────────────────

"$ffmpeg" -v error -y -i "$scratch/clip.mov" \
    -c:v libx264 -preset ultrafast "$scratch/h264.mp4"
expect_video "H.264 encoding" h264 "$scratch/h264.mp4"

"$ffmpeg" -v error -y -i "$scratch/clip.mov" \
    -c:v libx265 -preset ultrafast -x265-params log-level=none "$scratch/hevc.mp4"
expect_video "HEVC encoding" hevc "$scratch/hevc.mp4"

"$ffmpeg" -v error -y -i "$scratch/clip.mov" \
    -c:v libvpx-vp9 -deadline realtime -cpu-used 8 -b:v 200k "$scratch/vp9.webm"
expect_video "VP9 encoding" vp9 "$scratch/vp9.webm"

"$ffmpeg" -v error -y -f lavfi -i "sine=frequency=440:duration=1" \
    -c:a libopus -b:a 32k "$scratch/opus.ogg"
got=$(codec_of a:0 "$scratch/opus.ogg")
[ "$got" = opus ] || fail "Opus encoding: the output reports '$got', not 'opus'"
echo "Opus encoding: $got"

# ── PNG, written and read back ──────────────────────────────────────

"$ffmpeg" -v error -y -i "$scratch/clip.mov" -frames:v 1 "$scratch/frame.png"
expect_video "PNG encoding" png "$scratch/frame.png"
"$ffmpeg" -v error -y -i "$scratch/frame.png" -frames:v 1 "$scratch/frame-copy.png"
expect_size "PNG decoding" "$scratch/frame-copy.png" 320,240

# ── zscale ──────────────────────────────────────────────────────────

"$ffmpeg" -v error -y -i "$scratch/clip.mov" \
    -vf "zscale=w=160:h=120" -frames:v 1 "$scratch/zscale.png"
expect_size "zscale" "$scratch/zscale.png" 160,120

# ── https, with a file this leg turns into output ───────────────────

# No TLS variable is taken from the shell — a bundle the runner happens to export would
# prove the environment, not the binary — and every leg verifies the peer, which no TLS
# backend of this FFmpeg does unless asked (`tls_verify` is off by default). Windows
# verifies through SChannel's own system store; the unix legs carry OpenSSL built with
# `--openssldir=/etc/ssl`, and this FFmpeg reads CAs only from an explicit `ca_file`
# (falling back to the compiled-in store came upstream after n7.1.3), so the machine's
# own bundle is named and the handshake is verified against it.

fetch() {
    env -u SSL_CERT_FILE -u SSL_CERT_DIR -u CURL_CA_BUNDLE -u REQUESTS_CA_BUNDLE "$@"
}

case $(uname -s) in
*_NT-*)
    fetch "$ffmpeg" -v error -y -tls_verify 1 -i "$url" \
        -frames:v 1 "$scratch/https.png" ||
        fail "https: $url could not be read and verified through the system store"
    ;;
*)
    bundle=""
    for candidate in /etc/ssl/certs/ca-certificates.crt /etc/ssl/cert.pem; do
        if [ -r "$candidate" ]; then
            bundle=$candidate
            break
        fi
    done
    [ -n "$bundle" ] || fail "https: this host has no CA bundle to verify against"
    fetch "$ffmpeg" -v error -y -tls_verify 1 -ca_file "$bundle" -i "$url" \
        -frames:v 1 "$scratch/https.png" ||
        fail "https: $url could not be read and verified against $bundle"
    ;;
esac
expect_picture "https" "$scratch/https.png"

# ── The three jobs the product itself runs ──────────────────────────

"$ffprobe" -v error -print_format json \
    -show_entries "stream=codec_type,width,height:format=duration" \
    "$scratch/clip.mov" >"$scratch/probe.json"
grep -q '"video"' "$scratch/probe.json" && grep -q '"audio"' "$scratch/probe.json" ||
    fail "the probe did not report the video and audio streams of the sample"

"$ffmpeg" -v error -nostdin -y -i "$scratch/clip.mov" \
    -vf "fps=1,scale=160:160:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    -frames:v 2 -q:v 3 -f image2 "$scratch/frame-%05d.jpg"
test -f "$scratch/frame-00001.jpg" || fail "the frame extraction wrote no JPEG"

"$ffmpeg" -v error -nostdin -y -i "$scratch/clip.mov" \
    -vn -ac 1 -ar 16000 -c:a pcm_s16le -f wav "$scratch/audio.wav"
"$ffprobe" -v error -print_format json \
    -show_entries "stream=codec_name,sample_rate,channels" \
    "$scratch/audio.wav" >"$scratch/wav.json"
grep -q '"pcm_s16le"' "$scratch/wav.json" &&
    grep -qE '"sample_rate"[[:space:]]*:[[:space:]]*"?16000"?' "$scratch/wav.json" &&
    grep -qE '"channels"[[:space:]]*:[[:space:]]*1' "$scratch/wav.json" ||
    fail "the extracted audio is not 16 kHz mono PCM"

echo "every capability and the three jobs work on $(uname -sm)"
