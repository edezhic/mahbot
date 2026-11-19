//! Required, and empty: cargo asks a crate that declares a system link name (`links = "alsa"`) for
//! a build script, and this crate has nothing to build. The bindings are committed in
//! `src/generated.rs`, so no C header is read and no binding generator runs; nothing is linked, and
//! that absence is the point — every `snd_*` call is resolved with `dlopen` at first use, which is
//! what keeps `libasound.so.2` out of the shipped file's loader dependencies.
fn main() {}
