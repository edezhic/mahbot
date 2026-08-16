# candle-onnx-mahbot

Fork of [`candle-onnx`](https://github.com/huggingface/candle) 0.11.0 (ONNX
support for [Candle](https://github.com/huggingface/candle)) maintained by the
MahBot project. The canonical source lives in the
[`mahbot`](https://github.com/edezhic/mahbot) repository under
`vendor/candle-onnx/`; fork releases are cut from that directory. Upstream
candle-onnx 0.11.0 cannot run MahBot's ONNX voice-pipeline models
(mel-spectrogram, embeddings, wake word, TTS); this fork carries the patches
that fix those root causes.

## Patch set (delta vs upstream candle-onnx 0.11.0)

- **Implicit dtype promotion** for binary ops (`Add`/`Sub`/`Mul`/`Div`,
  `MatMul`, `Gemm`, `Where`, `Conv` bias, `ConstantOfShape`, `Concat`,
  `Gather` index arithmetic): mixed-dtype graphs are upcast to a common dtype
  (int → float, wider type wins) instead of failing on dtype mismatch.
- **`Max` op** — variadic `broadcast_maximum`.
- **`Softplus` op** — numerically stable (`x > 20 → x`, else
  `ln(exp(x) + 1)`).
- **`Reciprocal` op**.
- **`LayerNormalization` op** — population variance, optional `beta`,
  `epsilon` default `1e-5`, negative `axis` supported.
- **`Pad` opset 18+** — accepts the optional 3rd `constant_value` input
  (including the empty-string "not provided" form) and supports `edge` and
  `constant` modes; negative padding is rejected explicitly.
- **`PReLU` scalar-slope support** — a single-element slope is applied to all
  channels (`is_scalar = true`).
- **`Conv` dtype promotion** — input/weight (and bias) coerced to a common
  dtype before `conv1d`/`conv2d`.
- **`Pow` negative-base fix** — non-float base upcast to F32 so `powf` (which
  handles negative bases) is used.
- **Float initializer normalization** to F32, and **relaxed input dtype
  validation** for graph inputs that are overridden by initializers.
- **Crate-split import fixes** — upstream references the removed `candle`
  facade alias; the fork imports `candle_core` directly.

Unit tests cover `Reciprocal`, `Softplus`, and `LayerNormalization`.

## Behavior differences from upstream

- Binary-op dtype promotion, `Concat` majority-dtype promotion, and F32
  initializer normalization change results for mixed-dtype graphs. This
  matches common inference-engine behavior (e.g. F64 constants become F32),
  but models that relied on upstream's strict dtype errors will behave
  differently.
- **Known latent bug**: `Pad`'s opset-18+ `constant_value` input is accepted
  but silently discarded — padding always uses zeros (explicit TODO in
  `eval.rs`). Fine for MahBot's fixed model set; models that request non-zero
  constant padding produce wrong results.

## Build requirement: protoc

`prost-build` no longer bundles protoc binaries, so compiling this crate (or
anything that depends on it, e.g. `cargo install mahbot`) requires `protoc` on
`PATH`:

```
error: failed to run custom build command for `candle-onnx-mahbot`
Caused by: // (...)
  Could not find `protoc` installation and this build crate cannot proceed without this knowledge.
```

Install protoc (https://grpc.io/docs/protoc-installation/) and make it
available on `PATH`, e.g. `brew install protobuf` on macOS.

## Publishing

Fork releases are cut from `vendor/candle-onnx/` in the mahbot repository.
`cargo publish` requires a crates.io API token for the crate owner
(`cargo login`).
