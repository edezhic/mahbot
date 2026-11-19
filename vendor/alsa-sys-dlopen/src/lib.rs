//! The ALSA bindings `alsa` 0.11 compiles against, resolved through `dlopen` instead of linked
//! at load.
//!
//! A bare Linux server has no `libasound.so.2`, and a binary that names it as a loader
//! dependency refuses to start before any of the product's own code runs. This crate keeps the
//! same package name and version the `alsa` crate asks for, so it is a drop-in replacement, but
//! every `snd_*` function is a wrapper that looks its symbol up on first call. With no library
//! the wrappers return a failure value; with the library present they call the real function, so
//! audio behaves exactly as it did when it was linked.
//!
//! Whether the library is there is the product's question, not this crate's: it probes the same
//! soname itself (`mahbot::audio::library_available`) because the published package replaces this
//! shim with the registry crate, which has no such function.
//!
//! `src/generated.rs` holds the bindings themselves, committed rather than generated: they are the
//! published `alsa-sys` 0.4.0 bindings — the surface the last release that linked the library
//! shipped — with each `extern "C"` declaration replaced by its `dlopen` wrapper. Freezing them is
//! what makes this crate compile the same on every machine: a build that reads
//! `<alsa/asoundlib.h>` compiles whatever vintage that header is, and the release image's is older
//! than the `alsa` crate expects. `regenerate.rs` is the transform, run by hand through
//! `regenerate.sh` when the audio crate moves; it is not part of the build.
// The bindings are bindgen's own spelling, and they are generated code: the naming lints and the
// clippy families that a generated surface cannot satisfy are allowed rather than rewritten.
#![allow(
    non_camel_case_types,
    non_snake_case,
    clippy::all,
    clippy::pedantic
)]

use libc::{pid_t, pollfd, timespec, timeval, FILE};

include!("generated.rs");

// Header macros: the published bindings carry functions and types, not the `#define`s the `alsa`
// crate names. The values are the header's own, kept here because a macro has no symbol to look up.
pub const SND_PCM_NONBLOCK: core::ffi::c_int = 0x1;
pub const SND_PCM_ASYNC: core::ffi::c_int = 0x2;

pub const SND_SEQ_OPEN_OUTPUT: i32 = 1;
pub const SND_SEQ_OPEN_INPUT: i32 = 2;
pub const SND_SEQ_OPEN_DUPLEX: i32 = SND_SEQ_OPEN_OUTPUT | SND_SEQ_OPEN_INPUT;
pub const SND_SEQ_NONBLOCK: i32 = 0x0001;
pub const SND_SEQ_ADDRESS_BROADCAST: u8 = 255;
pub const SND_SEQ_ADDRESS_SUBSCRIBERS: u8 = 254;
pub const SND_SEQ_ADDRESS_UNKNOWN: u8 = 253;
pub const SND_SEQ_QUEUE_DIRECT: u8 = 253;
pub const SND_SEQ_TIME_MODE_MASK: u8 = 1 << 1;
pub const SND_SEQ_TIME_STAMP_MASK: u8 = 1 << 0;
pub const SND_SEQ_TIME_MODE_REL: u8 = 1 << 1;
pub const SND_SEQ_TIME_STAMP_REAL: u8 = 1 << 0;
pub const SND_SEQ_TIME_STAMP_TICK: u8 = 0 << 0;
pub const SND_SEQ_TIME_MODE_ABS: u8 = 0 << 1;
pub const SND_SEQ_CLIENT_SYSTEM: u8 = 0;
pub const SND_SEQ_PORT_SYSTEM_TIMER: u8 = 0;
pub const SND_SEQ_PORT_SYSTEM_ANNOUNCE: u8 = 1;
pub const SND_SEQ_PRIORITY_HIGH: u8 = 1 << 4;
pub const SND_SEQ_EVENT_LENGTH_FIXED: u8 = 0 << 2;
pub const SND_SEQ_EVENT_LENGTH_MASK: u8 = 3 << 2;
pub const SND_SEQ_EVENT_LENGTH_VARIABLE: u8 = 1 << 2;
pub const SND_SEQ_EVENT_LENGTH_VARUSR: u8 = 2 << 2;
