//! Minimal in-repo ONNX inference runtime.
//!
//! This module replaces the vendored `candle-onnx-mahbot` fork (removed in
//! mahbot-1776).  It exists for exactly one consumer: the Supertonic 3 TTS
//! pipeline ([`crate::audio::tts`]), which executes four pinned ONNX models
//! (duration_predictor, text_encoder, vector_estimator, vocoder — all opset
//! 19, a closed subset of 39 ops).  Nothing else in the codebase consumes
//! ONNX today, so the runtime deliberately implements only what those models
//! need:
//!
//! * **Proto layer** — a hand-rolled protobuf wire-format reader for the
//!   ONNX subset (no prost, no build-time protoc).  Tensor payloads are
//!   handled in `raw_data` form; the models contain no `external_data` /
//!   typed-array payloads.
//! * **Evaluator** — the 39 ops the four models use, replicating the fork's
//!   load-bearing semantics exactly (implicit dtype promotion, Pad `edge`
//!   mode, LayerNormalization with explicit epsilon, negative Slice steps,
//!   Concat majority-dtype promotion, Unsqueeze negative-axis handling, ...).
//! * **Load-time preparation** — initializers and `Constant` tensor
//!   attributes are converted to CPU tensors once at load, so repeated
//!   `simple_eval` calls (8 flow-matching steps × chunks) do not re-materialize
//!   the ~256 MB vector_estimator weights on every call as the fork did.

// The ONNX wire format and dims are inherently i64/varint while candle
// dims are usize; the reader's single-letter locals and the op dispatcher's
// size are idiomatic for this domain — allow the corresponding pedantic
// lints at module level rather than scattering per-function attributes.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

pub mod eval;

use candle_core::{DType, Device, Result, Tensor, bail};
use std::path::Path;

pub use eval::simple_eval;

// ── Parsed model representation ───────────────────────────────────────

/// A parsed ONNX model.  Initializers and `Constant` tensor attributes are
/// pre-converted to CPU tensors at parse time.
#[derive(Debug, Clone)]
pub struct Model {
    pub graph: Graph,
}

impl Model {
    /// Parse a model from raw ONNX protobuf bytes.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        parse_model(buf)
    }

    /// Name of the graph's first output (the pipeline's `extract_output`
    /// helper resolves the tensor this way).  Falls back to `"output"` when
    /// the graph declares no outputs.
    #[must_use]
    pub fn output_name(&self) -> &str {
        self.graph
            .outputs
            .first()
            .map_or("output", |o| o.name.as_str())
    }
}

/// Read and parse an ONNX model file.
pub fn read_file<P: AsRef<Path>>(p: P) -> Result<Model> {
    let buf = std::fs::read(p)?;
    Model::from_bytes(&buf)
}

#[derive(Debug, Clone)]
pub struct Graph {
    /// Graph nodes in topological order.
    pub nodes: Vec<Node>,
    /// Pre-converted initializers (float dtypes normalized to F32, matching
    /// the fork), in graph order.
    pub initializers: Vec<(String, Tensor)>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub op_type: String,
    /// May be empty (unnamed nodes).
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<Attribute>,
}

#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub kind: AttrKind,
}

#[derive(Debug, Clone)]
pub enum AttrKind {
    Int(i64),
    Float(f32),
    /// Raw UTF-8 string payload (e.g. Pad `mode`).
    Bytes(Vec<u8>),
    Ints(Vec<i64>),
    /// `repeated float` attribute payload.  Decoded for completeness; no
    /// consumer in the pinned TTS models uses it (the fork's `Max`/`Selu`
    /// era is gone), so callers treat it as unsupported.
    FloatsUnused,
    /// Tensor attribute (e.g. `Constant` value, `ConstantOfShape` value),
    /// pre-converted to a CPU tensor at load time.
    Tensor(Tensor),
}

#[derive(Debug, Clone)]
pub struct ValueInfo {
    pub name: String,
    /// Declared element type mapped to a candle dtype; `None` when the type
    /// is absent or uses an unmapped ONNX data type (validation skipped).
    pub elem_type: Option<DType>,
}

// ── Minimal protobuf wire-format reader ───────────────────────────────
//
// Only the ONNX message subset used by the four TTS models is decoded.
// Field numbers follow onnx.proto3 (the OLD AttributeProto numbering:
// ints=8, floats=7, strings=9, tensors=10, type=20).

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WireType {
    Varint,
    Fixed64,
    LenDelimited,
    Fixed32,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    fn eof(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn read_u8(&mut self) -> Result<u8> {
        if self.pos >= self.buf.len() {
            bail!("truncated protobuf message");
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_varint(&mut self) -> Result<u64> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        loop {
            let b = self.read_u8()?;
            result |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
            if shift >= 64 {
                bail!("malformed protobuf varint (too long)");
            }
        }
    }

    fn read_tag(&mut self) -> Result<(u32, WireType)> {
        let tag = self.read_varint()?;
        let field = (tag >> 3) as u32;
        let wire = match tag & 0x7 {
            0 => WireType::Varint,
            1 => WireType::Fixed64,
            2 => WireType::LenDelimited,
            5 => WireType::Fixed32,
            other => bail!("unsupported protobuf wire type {other}"),
        };
        Ok((field, wire))
    }

    fn read_len_delimited(&mut self) -> Result<&'a [u8]> {
        let len = self.read_varint()? as usize;
        if self.pos + len > self.buf.len() {
            bail!("truncated length-delimited protobuf field");
        }
        let start = self.pos;
        self.pos += len;
        Ok(&self.buf[start..start + len])
    }

    fn read_fixed32(&mut self) -> Result<u32> {
        let mut bytes = [0u8; 4];
        for b in &mut bytes {
            *b = self.read_u8()?;
        }
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_fixed64(&mut self) -> Result<u64> {
        let mut bytes = [0u8; 8];
        for b in &mut bytes {
            *b = self.read_u8()?;
        }
        Ok(u64::from_le_bytes(bytes))
    }

    fn skip(&mut self, wire: WireType) -> Result<()> {
        match wire {
            WireType::Varint => {
                let _ = self.read_varint()?;
            }
            WireType::Fixed64 => {
                let _ = self.read_fixed64()?;
            }
            WireType::LenDelimited => {
                let _ = self.read_len_delimited()?;
            }
            WireType::Fixed32 => {
                let _ = self.read_fixed32()?;
            }
        }
        Ok(())
    }

    /// Read a `repeated int64` field that may be encoded packed
    /// (length-delimited varint list) or unpacked (repeated varint fields).
    fn read_varints(&mut self, wire: WireType) -> Result<Vec<i64>> {
        match wire {
            WireType::Varint => Ok(vec![self.read_varint()? as i64]),
            WireType::LenDelimited => {
                let sub = self.read_len_delimited()?;
                let mut r = Reader::new(sub);
                let mut out = Vec::new();
                while !r.eof() {
                    out.push(r.read_varint()? as i64);
                }
                Ok(out)
            }
            other => bail!("unexpected wire type {other:?} for packed varint field"),
        }
    }

    /// Read a `repeated float` field (packed fixed32 list or single).
    fn read_floats(&mut self, wire: WireType) -> Result<Vec<f32>> {
        match wire {
            WireType::Fixed32 => Ok(vec![f32::from_bits(self.read_fixed32()?)]),
            WireType::LenDelimited => {
                let sub = self.read_len_delimited()?;
                if sub.len() % 4 != 0 {
                    bail!("malformed packed float field (length {})", sub.len());
                }
                Ok(sub
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
            other => bail!("unexpected wire type {other:?} for packed float field"),
        }
    }

    /// Read a `repeated double` field (packed fixed64 list or single).
    fn read_doubles(&mut self, wire: WireType) -> Result<Vec<f64>> {
        match wire {
            WireType::Fixed64 => Ok(vec![f64::from_bits(self.read_fixed64()?)]),
            WireType::LenDelimited => {
                let sub = self.read_len_delimited()?;
                if sub.len() % 8 != 0 {
                    bail!("malformed packed double field (length {})", sub.len());
                }
                Ok(sub
                    .chunks_exact(8)
                    .map(|c| {
                        f64::from_bits(u64::from_le_bytes([
                            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                        ]))
                    })
                    .collect())
            }
            other => bail!("unexpected wire type {other:?} for packed double field"),
        }
    }
}

fn parse_model(buf: &[u8]) -> Result<Model> {
    let mut r = Reader::new(buf);
    let mut graph = None;
    while !r.eof() {
        let (field, wire) = r.read_tag()?;
        match (field, wire) {
            (7, WireType::LenDelimited) => {
                let sub = r.read_len_delimited()?;
                graph = Some(parse_graph(sub)?);
            }
            _ => r.skip(wire)?,
        }
    }
    let graph = graph.ok_or_else(|| candle_core::Error::Msg("model has no graph".to_string()))?;
    Ok(Model { graph })
}

fn parse_graph(buf: &[u8]) -> Result<Graph> {
    let mut r = Reader::new(buf);
    let mut nodes = Vec::new();
    let mut initializers = Vec::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    while !r.eof() {
        let (field, wire) = r.read_tag()?;
        match (field, wire) {
            (1, WireType::LenDelimited) => {
                let sub = r.read_len_delimited()?;
                nodes.push(parse_node(sub)?);
            }
            (5, WireType::LenDelimited) => {
                let sub = r.read_len_delimited()?;
                let raw = parse_tensor_proto(sub)?;
                let mut tensor = raw_to_tensor(&raw)?;
                // Normalize float initializers to F32 (fork parity): many
                // ONNX exports store constants as F64, downstream ops expect
                // F32.  The pinned TTS models contain no F64 tensors, so this
                // is defensive only.
                if tensor.dtype().is_float() && tensor.dtype() != DType::F32 {
                    tensor = tensor.to_dtype(DType::F32)?;
                }
                initializers.push((raw.name, tensor));
            }
            (11, WireType::LenDelimited) => {
                let sub = r.read_len_delimited()?;
                inputs.push(parse_value_info(sub)?);
            }
            (12, WireType::LenDelimited) => {
                let sub = r.read_len_delimited()?;
                outputs.push(parse_value_info(sub)?);
            }
            _ => r.skip(wire)?,
        }
    }
    Ok(Graph {
        nodes,
        initializers,
        inputs,
        outputs,
    })
}

fn parse_node(buf: &[u8]) -> Result<Node> {
    let mut r = Reader::new(buf);
    let mut node = Node {
        op_type: String::new(),
        name: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attributes: Vec::new(),
    };
    while !r.eof() {
        let (field, wire) = r.read_tag()?;
        match (field, wire) {
            (1, WireType::LenDelimited) => {
                node.inputs
                    .push(String::from_utf8_lossy(r.read_len_delimited()?).into_owned());
            }
            (2, WireType::LenDelimited) => {
                node.outputs
                    .push(String::from_utf8_lossy(r.read_len_delimited()?).into_owned());
            }
            (3, WireType::LenDelimited) => {
                node.name = String::from_utf8_lossy(r.read_len_delimited()?).into_owned();
            }
            (4, WireType::LenDelimited) => {
                node.op_type = String::from_utf8_lossy(r.read_len_delimited()?).into_owned();
            }
            (5, WireType::LenDelimited) => {
                let sub = r.read_len_delimited()?;
                node.attributes.push(parse_attribute(sub)?);
            }
            _ => r.skip(wire)?,
        }
    }
    Ok(node)
}

fn parse_attribute(buf: &[u8]) -> Result<Attribute> {
    let mut r = Reader::new(buf);
    let mut name = String::new();
    let mut f = None;
    let mut i = None;
    let mut s = None;
    let mut t = None;
    let mut ints: Option<Vec<i64>> = None;
    let mut floats: Option<Vec<f32>> = None;
    let mut type_disc = None;
    while !r.eof() {
        let (field, wire) = r.read_tag()?;
        match (field, wire) {
            (1, WireType::LenDelimited) => {
                name = String::from_utf8_lossy(r.read_len_delimited()?).into_owned();
            }
            (2, WireType::Fixed32) => f = Some(f32::from_bits(r.read_fixed32()?)),
            (3, WireType::Varint) => i = Some(r.read_varint()? as i64),
            (4, WireType::LenDelimited) => s = Some(r.read_len_delimited()?.to_vec()),
            (5, WireType::LenDelimited) => {
                let sub = r.read_len_delimited()?;
                t = Some(parse_tensor_proto(sub)?);
            }
            // Repeated scalar fields may be encoded packed (one length-
            // delimited list) or unpacked (one field per element) — the four
            // TTS models use the unpacked form.  Accumulate both.
            (7, _) => {
                let vals = r.read_floats(wire)?;
                floats.get_or_insert_with(Vec::new).extend(vals);
            }
            (8, _) => {
                let vals = r.read_varints(wire)?;
                ints.get_or_insert_with(Vec::new).extend(vals);
            }
            (20, WireType::Varint) => type_disc = Some(r.read_varint()? as i32),
            _ => r.skip(wire)?,
        }
    }
    // AttributeType discriminators (onnx.proto3): FLOAT=1, INT=2, STRING=3,
    // TENSOR=4, FLOATS=6, INTS=7, STRINGS=8, TENSORS=9, ...
    let kind = match type_disc {
        Some(1) => AttrKind::Float(f.unwrap_or(0.0)),
        Some(2) => AttrKind::Int(i.unwrap_or(0)),
        Some(3) => AttrKind::Bytes(s.unwrap_or_default()),
        Some(4) => {
            let raw = t.ok_or_else(|| {
                candle_core::Error::Msg(format!("attribute {name:?} is TENSOR without a value"))
            })?;
            AttrKind::Tensor(raw_to_tensor(&raw)?)
        }
        Some(6) => AttrKind::FloatsUnused,
        Some(7) => AttrKind::Ints(ints.unwrap_or_default()),
        Some(8 | 9) => {
            bail!("attribute {name:?} uses an unsupported list form (strings/tensors)")
        }
        Some(other) => bail!("attribute {name:?} uses unsupported type {other}"),
        // No discriminator: fall back to the present value field.
        None => {
            if let Some(v) = f {
                AttrKind::Float(v)
            } else if let Some(v) = i {
                AttrKind::Int(v)
            } else if let Some(v) = s {
                AttrKind::Bytes(v)
            } else if let Some(v) = ints {
                AttrKind::Ints(v)
            } else if floats.is_some() {
                AttrKind::FloatsUnused
            } else if let Some(raw) = t {
                AttrKind::Tensor(raw_to_tensor(&raw)?)
            } else {
                bail!("attribute {name:?} has no value");
            }
        }
    };
    Ok(Attribute { name, kind })
}

struct RawTensor {
    name: String,
    dims: Vec<usize>,
    data_type: i32,
    data: TensorPayload,
}

enum TensorPayload {
    Raw(Vec<u8>),
    Float(Vec<f32>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Double(Vec<f64>),
}

fn parse_tensor_proto(buf: &[u8]) -> Result<RawTensor> {
    let mut r = Reader::new(buf);
    let mut name = String::new();
    let mut dims = Vec::new();
    let mut data_type = 0i32;
    let mut floats: Option<Vec<f32>> = None;
    let mut int32s: Option<Vec<i32>> = None;
    let mut int64s: Option<Vec<i64>> = None;
    let mut doubles: Option<Vec<f64>> = None;
    let mut raw: Option<Vec<u8>> = None;
    while !r.eof() {
        let (field, wire) = r.read_tag()?;
        match (field, wire) {
            (1, _) => {
                let vals = r.read_varints(wire)?;
                for d in vals {
                    if d < 0 {
                        bail!("negative tensor dimension {d}");
                    }
                    dims.push(d as usize);
                }
            }
            (2, WireType::Varint) => data_type = r.read_varint()? as i32,
            // Repeated scalar payloads may be packed or unpacked — accumulate.
            (4, _) => {
                let vals = r.read_floats(wire)?;
                floats.get_or_insert_with(Vec::new).extend(vals);
            }
            (5, _) => {
                let vals = r.read_varints(wire)?;
                int32s
                    .get_or_insert_with(Vec::new)
                    .extend(vals.into_iter().map(|v| v as i32));
            }
            (7, _) => {
                let vals = r.read_varints(wire)?;
                int64s.get_or_insert_with(Vec::new).extend(vals);
            }
            (8, WireType::LenDelimited) => {
                name = String::from_utf8_lossy(r.read_len_delimited()?).into_owned();
            }
            (9, WireType::LenDelimited) => {
                raw = Some(r.read_len_delimited()?.to_vec());
            }
            (10, _) => {
                let vals = r.read_doubles(wire)?;
                doubles.get_or_insert_with(Vec::new).extend(vals);
            }
            _ => r.skip(wire)?,
        }
    }
    // Typed arrays take precedence over raw_data (fork get_tensor parity);
    // zero-element tensors may omit all payload fields.
    let data = if let Some(v) = floats {
        TensorPayload::Float(v)
    } else if let Some(v) = int32s {
        TensorPayload::Int32(v)
    } else if let Some(v) = int64s {
        TensorPayload::Int64(v)
    } else if let Some(v) = doubles {
        TensorPayload::Double(v)
    } else if let Some(v) = raw {
        TensorPayload::Raw(v)
    } else {
        TensorPayload::Raw(Vec::new())
    };
    Ok(RawTensor {
        name,
        dims,
        data_type,
        data,
    })
}

fn parse_value_info(buf: &[u8]) -> Result<ValueInfo> {
    let mut r = Reader::new(buf);
    let mut name = String::new();
    let mut type_buf: Option<&[u8]> = None;
    while !r.eof() {
        let (field, wire) = r.read_tag()?;
        match (field, wire) {
            (1, WireType::LenDelimited) => {
                name = String::from_utf8_lossy(r.read_len_delimited()?).into_owned();
            }
            (2, WireType::LenDelimited) => type_buf = Some(r.read_len_delimited()?),
            _ => r.skip(wire)?,
        }
    }
    let elem_type = match type_buf {
        Some(sub) => parse_type_elem_type(sub)?,
        None => None,
    };
    Ok(ValueInfo { name, elem_type })
}

/// Extract `TypeProto.Tensor.elem_type` (mapped to a candle dtype) from a
/// `TypeProto` payload.  `None` when the type is absent, not a tensor type,
/// or uses an unmapped ONNX data type.
fn parse_type_elem_type(buf: &[u8]) -> Result<Option<DType>> {
    let mut r = Reader::new(buf);
    while !r.eof() {
        let (field, wire) = r.read_tag()?;
        match (field, wire) {
            (1, WireType::LenDelimited) => {
                // tensor_type: TypeProto.Tensor { int32 elem_type = 1; ... }
                let sub = r.read_len_delimited()?;
                let mut tr = Reader::new(sub);
                while !tr.eof() {
                    let (tf, tw) = tr.read_tag()?;
                    match (tf, tw) {
                        (1, WireType::Varint) => {
                            return Ok(dtype(tr.read_varint()? as i32));
                        }
                        _ => tr.skip(tw)?,
                    }
                }
                return Ok(None);
            }
            _ => r.skip(wire)?,
        }
    }
    Ok(None)
}

/// Map an ONNX `TensorProto.DataType` value to a candle dtype (fork parity).
pub(crate) fn dtype(dt: i32) -> Option<DType> {
    match dt {
        2 | 9 => Some(DType::U8), // UINT8, BOOL (candle stores bools as U8)
        12 => Some(DType::U32),   // UINT32
        7 => Some(DType::I64),    // INT64
        10 => Some(DType::F16),   // FLOAT16
        1 => Some(DType::F32),    // FLOAT
        11 => Some(DType::F64),   // DOUBLE
        _ => None,
    }
}

/// Convert a parsed tensor payload into a CPU candle tensor, mirroring the
/// fork's `get_tensor` semantics: INT32 payloads are widened to I64, typed
/// arrays take precedence over `raw_data`, everything else uses
/// `from_raw_buffer`.
fn raw_to_tensor(t: &RawTensor) -> Result<Tensor> {
    let dev = &Device::Cpu;
    if t.data_type == 6 {
        // INT32 → I64
        let data: Vec<i64> = match &t.data {
            TensorPayload::Int32(v) => v.iter().map(|&x| i64::from(x)).collect(),
            TensorPayload::Raw(b) => b
                .chunks_exact(4)
                .map(|c| i64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect(),
            _ => bail!("INT32 tensor '{}' has an incompatible payload", t.name),
        };
        let len = data.len();
        return Tensor::from_vec(data, len, dev);
    }
    let Some(dt) = dtype(t.data_type) else {
        bail!(
            "unsupported tensor data type {} for '{}'",
            t.data_type,
            t.name
        );
    };
    let dims = t.dims.as_slice();
    let tensor = match (&t.data, dt) {
        (TensorPayload::Float(v), DType::F32) => Tensor::from_slice(v, dims, dev)?,
        (TensorPayload::Double(v), DType::F64) => Tensor::from_slice(v, dims, dev)?,
        (TensorPayload::Int64(v), DType::I64) => Tensor::from_slice(v, dims, dev)?,
        (TensorPayload::Raw(b), _) => Tensor::from_raw_buffer(b, dt, dims, dev)?,
        (payload, _) => {
            bail!(
                "tensor '{}' (data type {}) has an incompatible payload {:?}",
                t.name,
                t.data_type,
                payload_kind(payload)
            )
        }
    };
    Ok(tensor)
}

fn payload_kind(p: &TensorPayload) -> &'static str {
    match p {
        TensorPayload::Raw(_) => "raw_data",
        TensorPayload::Float(_) => "float_data",
        TensorPayload::Int32(_) => "int32_data",
        TensorPayload::Int64(_) => "int64_data",
        TensorPayload::Double(_) => "double_data",
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "voice-tests"))]
mod golden_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip a tiny wire-encoded attribute with the OLD field numbering
    /// (ints=8, floats=7, strings=9, type=20) to guard the reader against the
    /// protobuf-version trap called out in the fork-removal analysis.
    #[test]
    fn test_parse_attribute_old_numbering() {
        // name = "mode" (field 1), s = "constant" (field 4),
        // type = STRING (field 20, varint 3).
        let mut wire = Vec::new();
        wire.extend_from_slice(&[0x0a, 4, b'm', b'o', b'd', b'e']); // name
        wire.extend_from_slice(&[0x22, 8, b'c', b'o', b'n', b's', b't', b'a', b'n', b't']); // s
        wire.extend_from_slice(&[0xa0, 0x01, 3]); // type = 3 (STRING)
        let attr = parse_attribute(&wire).unwrap();
        assert_eq!(attr.name, "mode");
        match attr.kind {
            AttrKind::Bytes(b) => assert_eq!(b, b"constant"),
            other => panic!("expected Bytes, got {other:?}"),
        }

        // ints (field 8, packed varint), type = INTS (field 20, varint 7).
        let mut wire = Vec::new();
        wire.extend_from_slice(&[0x0a, 5, b'a', b'x', b'i', b's', b'_']); // name
        wire.extend_from_slice(&[0x42, 3, 1, 2, 3]); // field 8 packed [1,2,3]
        wire.extend_from_slice(&[0xa0, 0x01, 7]); // type = 7 (INTS)
        let attr = parse_attribute(&wire).unwrap();
        match attr.kind {
            AttrKind::Ints(v) => assert_eq!(v, vec![1, 2, 3]),
            other => panic!("expected Ints, got {other:?}"),
        }
    }

    /// Verify the wire reader decodes a packed int64 tensor payload (the form
    /// all four TTS model initializers use).
    #[test]
    fn test_parse_tensor_proto_raw() {
        // dims = [2] (field 1, packed varint), data_type = 7 (field 2),
        // raw_data = 16 bytes (field 9).
        let mut wire = Vec::new();
        wire.extend_from_slice(&[0x0a, 1, 2]); // dims packed [2]
        wire.extend_from_slice(&[0x10, 7]); // data_type = INT64
        let mut raw = vec![0u8; 16];
        raw[0] = 1;
        raw[8] = 2;
        wire.push(0x4a); // field 9, len-delimited
        wire.push(16);
        wire.extend_from_slice(&raw);
        let t = parse_tensor_proto(&wire).unwrap();
        assert_eq!(t.dims, vec![2]);
        assert_eq!(t.data_type, 7);
        let tensor = raw_to_tensor(&t).unwrap();
        assert_eq!(tensor.to_vec1::<i64>().unwrap(), vec![1, 2]);
    }

    #[test]
    fn test_dtype_mapping() {
        assert_eq!(dtype(1), Some(DType::F32)); // FLOAT
        assert_eq!(dtype(7), Some(DType::I64)); // INT64
        assert_eq!(dtype(9), Some(DType::U8)); // BOOL
        assert_eq!(dtype(6), None); // INT32 handled by raw_to_tensor, not dtype()
        assert_eq!(dtype(3), None); // INT8 unmapped
    }
}
