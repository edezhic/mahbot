//! Direct graph evaluator for the pinned Supertonic 3 TTS models.
//!
//! This is a focused re-implementation of the removed `candle-onnx-mahbot`
//! fork's `simple_eval`, keeping only the 39 ops the four models use and
//! preserving the fork's load-bearing semantics exactly:
//!
//! * implicit dtype promotion for mixed I64/F32 arithmetic
//!   ([`promote_types`] — `duration_predictor`/`text_encoder` contain many
//!   mixed-dtype nodes);
//! * `Pad` `edge` mode (used by all convnext `dwconv` pads) and constant
//!   mode with an empty-string opset-18 third input treated as "not
//!   provided" (zero padding); a *provided* non-zero pad value is honored
//!   (the fork silently discarded it);
//! * `LayerNormalization` with the explicit `epsilon` attribute (1e-6 in all
//!   71 nodes, not the 1e-5 default), population variance via
//!   mean → square → mean;
//! * negative `Slice` steps (60 nodes use `step = -1`), 4- and 5-input
//!   forms, and the fork's clamp rules;
//! * `Concat` majority-dtype promotion plus the trailing-singleton-squeeze
//!   rank hack;
//! * `Unsqueeze` negative-axis off-by-one math;
//! * `Gather` negative-index normalization (axis 0);
//! * `Pow` with negative base via `powf` and I64→F32 base conversion;
//! * `Softplus` stable branch (`x > 20 ? x : ln(exp(x) + 1)`);
//! * `Gemm` `transB = 1`; `Reshape` 0 → input dim; `Cast` BOOL→U8 and
//!   INT32→I64;
//! * input dtype validation that skips initializer-overridden graph inputs.
//!
//! The evaluator runs on CPU (the TTS pipeline's device).  Initializers and
//! `Constant` tensor attributes are pre-converted at load time, so each
//! `simple_eval` call performs no weight re-materialization.

use crate::onnx::{AttrKind, Model, Node};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, bail};
use candle_nn::activation::PReLU;
use std::collections::HashMap;

type Value = Tensor;

// ── Attribute access ──────────────────────────────────────────────────

fn get_attr<'a>(node: &'a Node, name: &str) -> Result<&'a AttrKind> {
    node.attributes
        .iter()
        .find(|a| a.name == name)
        .map(|a| &a.kind)
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "cannot find the '{name}' attribute in '{}' for {}",
                node.op_type, node.name
            ))
        })
}

fn get_attr_opt<'a>(node: &'a Node, name: &str) -> Option<&'a AttrKind> {
    node.attributes
        .iter()
        .find(|a| a.name == name)
        .map(|a| &a.kind)
}

fn attr_i(attr: &AttrKind, node: &Node, name: &str) -> Result<i64> {
    match attr {
        AttrKind::Int(v) => Ok(*v),
        other => bail!(
            "attribute '{name}' of op '{}' ({}) is not an int: {other:?}",
            node.op_type,
            node.name
        ),
    }
}

fn attr_f(attr: &AttrKind, node: &Node, name: &str) -> Result<f32> {
    match attr {
        AttrKind::Float(v) => Ok(*v),
        other => bail!(
            "attribute '{name}' of op '{}' ({}) is not a float: {other:?}",
            node.op_type,
            node.name
        ),
    }
}

fn attr_ints<'a>(attr: &'a AttrKind, node: &Node, name: &str) -> Result<&'a [i64]> {
    match attr {
        AttrKind::Ints(v) => Ok(v),
        other => bail!(
            "attribute '{name}' of op '{}' ({}) is not an int list: {other:?}",
            node.op_type,
            node.name
        ),
    }
}

fn attr_bytes<'a>(attr: &'a AttrKind, node: &Node, name: &str) -> Result<&'a [u8]> {
    match attr {
        AttrKind::Bytes(v) => Ok(v),
        other => bail!(
            "attribute '{name}' of op '{}' ({}) is not a string: {other:?}",
            node.op_type,
            node.name
        ),
    }
}

fn attr_tensor<'a>(attr: &'a AttrKind, node: &Node, name: &str) -> Result<&'a Tensor> {
    match attr {
        AttrKind::Tensor(t) => Ok(t),
        other => bail!(
            "attribute '{name}' of op '{}' ({}) is not a tensor: {other:?}",
            node.op_type,
            node.name
        ),
    }
}

// ── Scalar extraction helpers ─────────────────────────────────────────

/// Extract a scalar from tensors that may be wrapped in extra dimensions
/// (some ONNX exports use shape `[1]`/`[1,1]` where scalars are expected).
/// Only accepts single-element tensors.
fn to_scalar_flexible<T: candle_core::WithDType>(t: &Tensor) -> Result<T> {
    if t.rank() > 0 && t.elem_count() == 1 {
        t.flatten_all()?.i(0)?.to_scalar::<T>()
    } else {
        t.to_scalar::<T>()
    }
}

// ── Dtype promotion helpers (fork parity) ─────────────────────────────

/// Ensure two tensors have the same dtype by upcasting the smaller type.
/// ONNX allows implicit type promotion in binary ops; candle-core's
/// `broadcast_*` ops require matching dtypes.
///
/// Promotes within type families (float→float, int→int) and promotes
/// integers to floats when mixed with float types.  Mixed float pairs prefer
/// F32 over F64 (inference-friendly), matching the fork.
fn promote_types(a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor)> {
    if a.dtype() == b.dtype() {
        return Ok((a.clone(), b.clone()));
    }
    let a_f = a.dtype().is_float();
    let b_f = b.dtype().is_float();
    // Mixed float+int: promote int to the float type.
    if a_f != b_f {
        if a_f {
            return Ok((a.clone(), b.to_dtype(a.dtype())?));
        }
        return Ok((a.to_dtype(b.dtype())?, b.clone()));
    }
    let target = if a_f {
        match (a.dtype(), b.dtype()) {
            (DType::F32, _) | (_, DType::F32) => DType::F32,
            (DType::F64, _) | (_, DType::F64) => DType::F64,
            (DType::F16, _) | (_, DType::F16) => DType::F16,
            (DType::BF16, _) | (_, DType::BF16) => DType::BF16,
            _ => return Ok((a.clone(), b.clone())),
        }
    } else {
        match (a.dtype(), b.dtype()) {
            (DType::I64, _) | (_, DType::I64) => DType::I64,
            (DType::I32, _) | (_, DType::I32) => DType::I32,
            (DType::U32, _) | (_, DType::U32) => DType::U32,
            (DType::U8, _) | (_, DType::U8) => DType::U8,
            _ => return Ok((a.clone(), b.clone())),
        }
    };
    Ok((a.to_dtype(target)?, b.to_dtype(target)?))
}

fn safe_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (a, b) = promote_types(a, b)?;
    a.broadcast_add(&b)
}

fn safe_sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (a, b) = promote_types(a, b)?;
    a.broadcast_sub(&b)
}

fn safe_mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (a, b) = promote_types(a, b)?;
    a.broadcast_mul(&b)
}

fn safe_div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (a, b) = promote_types(a, b)?;
    a.broadcast_div(&b)
}

fn safe_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (a, b) = promote_types(a, b)?;
    a.broadcast_matmul(&b)
}

/// `where_cond` with automatic dtype promotion for the two value branches.
/// The condition tensor is not promoted.
fn safe_where(cond: &Tensor, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (a, b) = promote_types(a, b)?;
    cond.where_cond(&a, &b)
}

// ── Broadcasting helpers ──────────────────────────────────────────────

fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Result<Vec<usize>> {
    let (longest, shortest) = if shape_a.len() > shape_b.len() {
        (shape_a, shape_b)
    } else {
        (shape_b, shape_a)
    };
    let diff = longest.len() - shortest.len();
    let mut target_shape = longest[0..diff].to_vec();
    for (dim1, dim2) in longest[diff..].iter().zip(shortest.iter()) {
        if *dim1 == *dim2 || *dim2 == 1 || *dim1 == 1 {
            target_shape.push(usize::max(*dim1, *dim2));
        } else {
            bail!(
                "Expand: incompatible shapes for broadcast, {:?} and {:?}",
                shape_a,
                shape_b
            );
        }
    }
    Ok(target_shape)
}

fn broadcast_shape_from_many(shapes: &[&[usize]]) -> Result<Vec<usize>> {
    if shapes.is_empty() {
        return Ok(Vec::new());
    }
    let mut shape_out = shapes[0].to_vec();
    for shape in &shapes[1..] {
        shape_out = broadcast_shape(&shape_out, shape)?;
    }
    Ok(shape_out)
}

// ── Evaluator ─────────────────────────────────────────────────────────

/// Evaluate the model graph with the provided inputs, returning a map from
/// graph output name to tensor.  The fork-compatible entry point; the TTS
/// pipeline calls this once per model per chunk/flow step.
pub fn simple_eval(
    model: &Model,
    mut values: HashMap<String, Value>,
) -> Result<HashMap<String, Value>> {
    let graph = &model.graph;

    // Initializers override caller-provided inputs of the same name
    // (initializer-overridden graph inputs).  Pre-converted at load time.
    for (name, tensor) in &graph.initializers {
        values.insert(name.clone(), tensor.clone());
    }

    // Validate provided graph inputs against their declared dtypes, skipping
    // initializer-overridden inputs (their dtypes may differ from the graph
    // type declaration after initializer normalization).
    for input in &graph.inputs {
        let Some(expected) = input.elem_type else {
            continue;
        };
        let Some(tensor) = values.get(&input.name) else {
            bail!("missing input {}", input.name);
        };
        if tensor.dtype() != expected {
            let is_initializer = graph.initializers.iter().any(|(n, _)| n == &input.name);
            if !is_initializer {
                bail!(
                    "unexpected dtype for {}, got {:?}, expected {expected:?}",
                    input.name,
                    tensor.dtype()
                );
            }
        }
    }

    // The nodes are topologically sorted, so process them in order.
    for node in &graph.nodes {
        let get = |input_name: &str| match values.get(input_name) {
            Some(value) => Ok(value),
            None => bail!("cannot find {input_name} for op '{}'", node.name),
        };
        let get_opt = |i: usize| {
            node.inputs
                .get(i)
                .filter(|s: &&String| !s.is_empty())
                .map(|s| get(s))
        };

        match node.op_type.as_str() {
            "Add" => {
                let input0 = get(&node.inputs[0])?;
                let input1 = get(&node.inputs[1])?;
                let output = safe_add(input0, input1)?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Sub" => {
                let input0 = get(&node.inputs[0])?;
                let input1 = get(&node.inputs[1])?;
                let output = safe_sub(input0, input1)?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Mul" => {
                let input0 = get(&node.inputs[0])?;
                let input1 = get(&node.inputs[1])?;
                let output = safe_mul(input0, input1)?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Div" => {
                let input0 = get(&node.inputs[0])?;
                let input1 = get(&node.inputs[1])?;
                let output = safe_div(input0, input1)?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Pow" => {
                let input0 = get(&node.inputs[0])?;
                let input1 = get(&node.inputs[1])?;
                // broadcast_pow cannot handle negative bases; use powf which
                // handles them correctly.  powf is float-only, so cast to F32
                // when needed (e.g. Pow(I64, I64) in the models).
                if let Ok(exp) = to_scalar_flexible::<f64>(&input1.to_dtype(DType::F64)?) {
                    let base = if input0.dtype().is_float() {
                        input0.clone()
                    } else {
                        input0.to_dtype(DType::F32)?
                    };
                    let output = base.powf(exp)?;
                    values.insert(node.outputs[0].clone(), output);
                } else {
                    let output = input0.broadcast_pow(input1)?;
                    values.insert(node.outputs[0].clone(), output);
                }
            }
            "Exp" => {
                let xs = get(&node.inputs[0])?;
                let output = xs.exp()?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Equal" => {
                let input0 = get(&node.inputs[0])?;
                let input1 = get(&node.inputs[1])?;
                let output = input0.broadcast_eq(input1)?;
                values.insert(node.outputs[0].clone(), output);
            }
            "MatMul" => {
                let input0 = get(&node.inputs[0])?;
                let input1 = get(&node.inputs[1])?;
                let output = safe_matmul(input0, input1)?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Reshape" => {
                let input0 = get(&node.inputs[0])?;
                let input1 = get(&node.inputs[1])?.to_vec1::<i64>()?;
                // At most a single -1 or 0 is expected (0 keeps the input
                // dimension at that index).
                let mut other_than_minus1 = 1usize;
                for &v in &input1 {
                    if v != -1 && v != 0 {
                        other_than_minus1 *= v as usize;
                    }
                }
                let input1 = input1
                    .iter()
                    .enumerate()
                    .map(|(idx, &v)| match v {
                        -1 => Ok(input0.elem_count() / other_than_minus1),
                        0 => input0.dim(idx),
                        _ => Ok(v as usize),
                    })
                    .collect::<Result<Vec<usize>>>()?;
                let output = input0.reshape(input1)?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Softmax" => {
                let input = get(&node.inputs[0])?;
                let output = match get_attr_opt(node, "axis") {
                    None => candle_nn::ops::softmax_last_dim(input)?,
                    Some(attr) => {
                        let axis = attr_i(attr, node, "axis")?;
                        let axis = input.normalize_axis(axis)?;
                        candle_nn::ops::softmax(input, axis)?
                    }
                };
                values.insert(node.outputs[0].clone(), output);
            }
            "Softplus" => {
                let input = get(&node.inputs[0])?;
                // Numerically stable softplus: x > 20 ? x : ln(exp(x) + 1)
                let mask = input.gt(20.0f64)?;
                let ones = Tensor::ones(input.dims(), input.dtype(), input.device())?;
                let exp_add_one = safe_add(&input.exp()?, &ones)?;
                let stable = exp_add_one.log()?;
                let output = safe_where(&mask, input, &stable)?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Transpose" => {
                let input = get(&node.inputs[0])?;
                let output = match get_attr_opt(node, "perm") {
                    None => input.t()?,
                    Some(attr) => {
                        let perm = attr_ints(attr, node, "perm")?
                            .iter()
                            .map(|&v| v as usize)
                            .collect::<Vec<_>>();
                        input.permute(perm)?.contiguous()?
                    }
                };
                values.insert(node.outputs[0].clone(), output);
            }
            "BatchNormalization" => {
                if attr_i(get_attr(node, "training_mode")?, node, "training_mode")? != 0 {
                    bail!("training mode is not supported for BatchNorm");
                }
                let eps = get_attr_opt(node, "epsilon")
                    .map(|a| attr_f(a, node, "epsilon"))
                    .transpose()?
                    .unwrap_or(1e-5);
                let xs = get(&node.inputs[0])?;
                let weight = get(&node.inputs[1])?;
                let bias = get(&node.inputs[2])?;
                let running_mean = get(&node.inputs[3])?;
                let running_var = get(&node.inputs[4])?;
                let target_shape: Vec<usize> = xs
                    .dims()
                    .iter()
                    .enumerate()
                    .map(|(idx, v)| if idx == 1 { *v } else { 1 })
                    .collect();
                let target_shape = target_shape.as_slice();
                let mean = running_mean.reshape(target_shape)?;
                let xs = safe_sub(xs, &mean)?;
                let var = (running_var.reshape(target_shape)? + f64::from(eps))?.sqrt()?;
                let xs = safe_div(&xs, &var)?;
                let weight = weight.reshape(target_shape)?;
                let bias = bias.reshape(target_shape)?;
                let xs = safe_mul(&xs, &weight)?;
                let xs = safe_add(&xs, &bias)?;
                values.insert(node.outputs[0].clone(), xs);
            }
            "Squeeze" => {
                let xs = get(&node.inputs[0])?;
                let mut axes = if node.inputs.len() <= 1 {
                    // Contract all the dimensions with size 1 except the
                    // batch dim.
                    xs.dims()
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, &s)| (s == 1 && idx > 0).then_some(idx))
                        .collect()
                } else {
                    get(&node.inputs[1])?
                        .to_vec1::<i64>()?
                        .iter()
                        .map(|&i| xs.normalize_axis(i))
                        .collect::<Result<Vec<_>>>()?
                };
                axes.sort_unstable();
                let mut xs = xs.clone();
                for &axis in axes.iter().rev() {
                    xs = xs.squeeze(axis)?;
                }
                values.insert(node.outputs[0].clone(), xs);
            }
            "ConstantOfShape" => {
                let input = get(&node.inputs[0])?;
                let value = match get_attr_opt(node, "value") {
                    Some(attr) => attr_tensor(attr, node, "value")?.clone(),
                    None => Tensor::zeros((), DType::F32, &Device::Cpu)?,
                };
                let shape_vec: Vec<usize> = input
                    .to_vec1::<i64>()?
                    .iter()
                    .map(|&x| x as usize)
                    .collect();
                let ones = Tensor::ones(shape_vec, value.dtype(), input.device())?;
                let xs = safe_mul(&ones, &value)?;
                values.insert(node.outputs[0].clone(), xs);
            }
            "Unsqueeze" => {
                let xs = get(&node.inputs[0])?;
                let axes = match get_attr_opt(node, "axes") {
                    Some(attr) => attr_ints(attr, node, "axes")?.to_vec(),
                    None => get(&node.inputs[1])?.to_vec1::<i64>()?,
                };
                let mut axes = axes
                    .iter()
                    .map(|&i| {
                        if i == xs.rank() as i64 {
                            Ok(xs.rank())
                        } else if i < 0 {
                            // normalize_axis doesn't work here because we want
                            // normalization relative to the FINAL size, not the
                            // current one (off by one).
                            Ok(xs.rank() - (-i as usize) + 1)
                        } else {
                            xs.normalize_axis(i)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                axes.sort_unstable();
                let mut xs = xs.clone();
                for &axis in axes.iter().rev() {
                    xs = xs.unsqueeze(axis)?;
                }
                values.insert(node.outputs[0].clone(), xs);
            }
            "Clip" => {
                let xs = get(&node.inputs[0])?;
                let xs = if let Some(mins) = get_opt(1) {
                    xs.broadcast_maximum(mins?)?
                } else {
                    xs.clone()
                };
                let xs = if let Some(maxs) = get_opt(2) {
                    xs.broadcast_minimum(maxs?)?
                } else {
                    xs.clone()
                };
                values.insert(node.outputs[0].clone(), xs);
            }
            "Gather" => {
                let xs = get(&node.inputs[0])?;
                let indices = get(&node.inputs[1])?;
                let axis = match get_attr_opt(node, "axis") {
                    Some(attr) => attr_i(attr, node, "axis")?,
                    None => 0,
                };
                let axis = xs.normalize_axis(axis)?;

                // index_select does not support negative indices, so normalize
                // them to positive via mask arithmetic.
                let indices = &{
                    let zeros = Tensor::zeros(indices.shape(), indices.dtype(), indices.device())?;
                    let max = Tensor::new(xs.dims()[axis] as i64, indices.device())?
                        .to_dtype(indices.dtype())?;
                    let mask = indices.lt(&zeros)?;
                    let mask_f = mask.to_dtype(indices.dtype())?;
                    safe_mul(&mask_f, &max)?.add(indices)?
                };

                // candle does not support tensor indexing, so the fork's
                // workarounds are replicated: scalar, 1-D, and 2-D indices.
                let xs = match indices.dims() {
                    [] => {
                        let index = indices.to_vec0::<i64>()? as usize;
                        xs.narrow(axis, index, 1)?.squeeze(axis)?
                    }
                    [_] => xs.index_select(indices, axis)?,
                    [first, _] => {
                        let mut v = Vec::with_capacity(*first);
                        for i in 0..*first {
                            v.push(xs.index_select(&indices.get(i)?, axis)?);
                        }
                        Tensor::stack(&v, axis)?
                    }
                    _ => bail!(
                        "Gather with indices rank > 2 is unsupported (op '{}')",
                        node.name
                    ),
                };
                values.insert(node.outputs[0].clone(), xs);
            }
            "Shape" => {
                let xs = get(&node.inputs[0])?;
                let start = get_attr_opt(node, "start")
                    .map(|a| attr_i(a, node, "start"))
                    .transpose()?
                    .unwrap_or(0);
                let end = get_attr_opt(node, "end")
                    .map(|a| attr_i(a, node, "end"))
                    .transpose()?
                    .unwrap_or(-1);
                let start = xs.normalize_axis(start)?;
                let end = xs.normalize_axis(end)?;
                let mut dims = vec![];
                for idx in start..=end {
                    dims.push(xs.dim(idx)? as i64);
                }
                let dims = Tensor::from_vec(dims, xs.rank(), xs.device())?;
                values.insert(node.outputs[0].clone(), dims);
            }
            "Concat" => {
                let inputs = node
                    .inputs
                    .iter()
                    .map(|n| Ok(get(n.as_str())?.clone()))
                    .collect::<Result<Vec<Value>>>()?;
                let axis = attr_i(get_attr(node, "axis")?, node, "axis")?;
                if inputs.is_empty() {
                    bail!("empty concat");
                }
                // Find minimum rank among inputs and squeeze trailing
                // singleton dims to match (fork rank hack).
                let min_rank = inputs.iter().map(Tensor::rank).min().unwrap();
                let inputs: Vec<_> = inputs
                    .into_iter()
                    .map(|t| {
                        let mut t = t;
                        while t.rank() > min_rank {
                            let last_dim = t.rank() - 1;
                            if t.dims()[last_dim] == 1 {
                                t = t.squeeze(last_dim).unwrap_or(t);
                            } else {
                                break;
                            }
                        }
                        t
                    })
                    .collect();
                let axis = inputs[0].normalize_axis(axis)?;
                // Promote all inputs to a common dtype before concatenating:
                // majority rule (most-represented dtype) to minimize precision
                // loss while avoiding cascading dtype errors from a single
                // odd-one-out input (e.g. an I64 index tensor among F32
                // values).  Ties are harmless.
                let mut dtype_counts = HashMap::new();
                for t in &inputs {
                    *dtype_counts.entry(t.dtype()).or_insert(0usize) += 1;
                }
                let target_dtype = dtype_counts
                    .into_iter()
                    .max_by_key(|&(_, count)| count)
                    .map(|(dt, _)| dt)
                    .expect("at least one input exists — guaranteed by the empty check above");
                let inputs: Vec<Value> = inputs
                    .into_iter()
                    .map(|t| {
                        if t.dtype() == target_dtype {
                            Ok(t)
                        } else {
                            t.to_dtype(target_dtype).map_err(|e| {
                                candle_core::Error::Msg(format!(
                                    "Concat dtype promotion failed for node '{}': {e}",
                                    node.name
                                ))
                            })
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                let output = Tensor::cat(&inputs, axis).map_err(|e| {
                    let shapes: Vec<_> = inputs.iter().map(|t| format!("{:?}", t.dims())).collect();
                    candle_core::Error::Msg(format!(
                        "Concat failed for node '{}': {e} (input shapes: {shapes:?})",
                        node.name
                    ))
                })?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Cos" => {
                let input = get(&node.inputs[0])?;
                let output = input.cos()?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Sin" => {
                let input = get(&node.inputs[0])?;
                let output = input.sin()?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Erf" => {
                let input = get(&node.inputs[0])?;
                let output = input.erf()?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Tanh" => {
                let input = get(&node.inputs[0])?;
                let output = input.tanh()?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Reciprocal" => {
                let input = get(&node.inputs[0])?;
                let output = input.recip()?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Relu" => {
                let input = get(&node.inputs[0])?;
                let output = input.relu()?;
                values.insert(node.outputs[0].clone(), output);
            }
            "PRelu" => {
                let input = get(&node.inputs[0])?;
                let slope = get(&node.inputs[1])?;
                // ONNX PReLU allows a single scalar slope applied to all
                // channels; set is_scalar=true when the slope has 1 element.
                let is_scalar = slope.elem_count() == 1;
                let output = PReLU::new(slope.clone(), is_scalar).forward(input)?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Constant" => {
                let value = get_attr(node, "value")?;
                let output = attr_tensor(value, node, "value")?.clone();
                values.insert(node.outputs[0].clone(), output);
            }
            "Cast" => {
                let input = get(&node.inputs[0])?;
                let dt = attr_i(get_attr(node, "to")?, node, "to")?;
                let dtype = match dt {
                    6 => DType::I64, // INT32 → I64 (fork parity)
                    dt => crate::onnx::dtype(dt as i32).ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "unsupported 'to' value {dt} for cast {}",
                            node.name
                        ))
                    })?,
                };
                let output = input.to_dtype(dtype)?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Pad" => {
                let mode = match get_attr_opt(node, "mode") {
                    Some(attr) => {
                        String::from_utf8_lossy(attr_bytes(attr, node, "mode")?).into_owned()
                    }
                    None => "constant".to_string(),
                };
                let data = get(&node.inputs[0])?;
                let pads = get(&node.inputs[1])?;
                // ONNX opset 18+ allows an optional 3rd input:
                // constant_value.  It arrives as an empty string when not
                // provided — treat that as not provided.
                let constant_value = if node.inputs.len() >= 3 && !node.inputs[2].is_empty() {
                    if mode == "constant" {
                        Some(get(&node.inputs[2])?.clone())
                    } else {
                        None
                    }
                } else {
                    None
                };
                if node.inputs.len() > 3 {
                    bail!(
                        "unsupported number of inputs {} for Pad node {:?}, expected 2 or 3",
                        node.inputs.len(),
                        node.name
                    );
                }
                if pads.rank() != 1 {
                    bail!("Pad expects 'pads' input to be 1D vector: {pads:?}");
                }
                if pads.dim(0)? != 2 * data.rank() {
                    bail!(
                        "Pad expects 'pads' input len to be 2 * rank of 'data' input: pads: {pads}, data rank: {}",
                        data.rank()
                    );
                }

                let pads = pads.to_vec1::<i64>()?;
                let (pads_pre, pads_post) = pads.split_at(pads.len() / 2);

                match mode.as_str() {
                    "edge" => {
                        let mut out = data.clone();
                        for (i, (&pre, &post)) in pads_pre.iter().zip(pads_post.iter()).enumerate()
                        {
                            if pre < 0 || post < 0 {
                                bail!(
                                    "Pad edge mode does not support negative padding in {:?}",
                                    node.name
                                );
                            }
                            let pre = pre as usize;
                            let post = post as usize;
                            if pre == 0 && post == 0 {
                                continue;
                            }
                            out = out.pad_with_same(i, pre, post)?;
                        }
                        values.insert(node.outputs[0].clone(), out);
                    }
                    "constant" => {
                        let mut out = data.clone();
                        for (i, (&pre, &post)) in pads_pre.iter().zip(pads_post.iter()).enumerate()
                        {
                            if pre < 0 || post < 0 {
                                bail!(
                                    "Pad constant mode does not support negative padding in {:?}",
                                    node.name
                                );
                            }
                            let pre = pre as usize;
                            let post = post as usize;
                            if pre == 0 && post == 0 {
                                continue;
                            }
                            let value = constant_value.as_ref();
                            out = pad_constant(&out, i, pre, post, value)?;
                        }
                        values.insert(node.outputs[0].clone(), out);
                    }
                    other => bail!(
                        "unsupported 'mode' value {other:?} for Pad node {:?}",
                        node.name
                    ),
                }
            }
            "Slice" => {
                let data = get(&node.inputs[0])?;
                let starts = get(&node.inputs[1])?;
                let ends = get(&node.inputs[2])?;
                let default_axes;
                let default_steps;
                let axes: &Tensor;
                let steps: &Tensor;
                // If axes are omitted they are [0, ..., r-1]; if steps are
                // omitted they are all 1.
                match node.inputs.len() {
                    3 => {
                        let len = starts.dims()[0];
                        default_axes = Some(Tensor::arange(0, len as i64, starts.device())?);
                        axes = default_axes.as_ref().unwrap();
                        default_steps = Some(Tensor::ones((len,), DType::I64, starts.device())?);
                        steps = default_steps.as_ref().unwrap();
                    }
                    4 => {
                        let len = starts.dims()[0];
                        axes = get(&node.inputs[3])?;
                        default_steps = Some(Tensor::ones((len,), DType::I64, starts.device())?);
                        steps = default_steps.as_ref().unwrap();
                    }
                    5 => {
                        steps = get(&node.inputs[4])?;
                        axes = get(&node.inputs[3])?;
                    }
                    _ => bail!(
                        "Slice node is invalid, expected 3-5 inputs, got {}: {:?}",
                        node.inputs.len(),
                        node
                    ),
                }

                let mut out = data.clone();
                for (i, ax) in axes.to_vec1::<i64>()?.into_iter().enumerate() {
                    // Negative axes are made non-negative by adding r.
                    let norm_axis = if ax < 0 { ax + data.rank() as i64 } else { ax } as usize;

                    let data_dim = data.dims()[norm_axis] as i64;
                    let mut s = to_scalar_flexible::<i64>(&starts.get(i)?)?;
                    let mut e = to_scalar_flexible::<i64>(&ends.get(i)?)?;
                    // Negative starts/ends get dims[axes[i]] added.
                    if s < 0 {
                        s += data_dim;
                    }
                    if e < 0 {
                        e += data_dim;
                    }

                    let p = to_scalar_flexible::<i64>(&steps.get(i)?)?;
                    // Clamp rules differ by step sign:
                    //  positive: starts in [0, dim], ends in [0, dim]
                    //  negative: starts in [0, dim-1], ends in [-1, dim-1]
                    if p >= 0 {
                        s = s.clamp(0, data_dim);
                        e = e.clamp(0, data_dim);
                    } else {
                        s = s.clamp(0, data_dim - 1);
                        e = e.clamp(-1, data_dim - 1);
                    }

                    let indexes = Tensor::arange_step(s, e, p, data.device())?;
                    out = out.contiguous()?.index_select(&indexes, norm_axis)?;
                }
                values.insert(node.outputs[0].clone(), out);
            }
            "ReduceSum" => {
                let input = get(&node.inputs[0])?;
                let axes = get_opt(1);
                let keepdims = get_attr_opt(node, "keepdims")
                    .map(|a| attr_i(a, node, "keepdims"))
                    .transpose()?
                    .unwrap_or(1);
                let noop_with_empty_axes = get_attr_opt(node, "noop_with_empty_axes")
                    .map(|a| attr_i(a, node, "noop_with_empty_axes"))
                    .transpose()?
                    .unwrap_or(0);

                let axes: Vec<usize> = match axes {
                    Some(Ok(axes)) => axes
                        .to_vec1::<i64>()?
                        .into_iter()
                        .map(|x| x as usize)
                        .collect(),
                    Some(Err(_)) | None => {
                        if noop_with_empty_axes == 1 {
                            vec![]
                        } else {
                            (0..input.rank()).collect()
                        }
                    }
                };

                let output = if keepdims == 1 {
                    input.sum_keepdim(axes)?
                } else {
                    input.sum(axes)?
                };
                values.insert(node.outputs[0].clone(), output);
            }
            "Split" => {
                let input_tensor = get(&node.inputs[0])?;
                let axis = get_attr_opt(node, "axis")
                    .map(|a| attr_i(a, node, "axis"))
                    .transpose()?
                    .unwrap_or(0);
                let axis = input_tensor.normalize_axis(axis)?;

                // Determine split sizes: from the split input when provided,
                // else equal division with the remainder added to the last.
                let splits = if node.inputs.len() > 1 {
                    let split_tensor = get(&node.inputs[1])?.to_vec1::<i64>()?;
                    split_tensor.iter().map(|&x| x as usize).collect::<Vec<_>>()
                } else {
                    let num_outputs = if let Some(attr) = get_attr_opt(node, "num_outputs") {
                        attr_i(attr, node, "num_outputs")? as usize
                    } else {
                        node.outputs.len()
                    };
                    let input_dim = input_tensor.dim(axis)?;
                    let mut split_sizes = vec![input_dim / num_outputs; num_outputs];
                    let remainder = input_dim % num_outputs;
                    if remainder > 0 {
                        split_sizes[num_outputs - 1] += remainder;
                    }
                    split_sizes
                };

                let mut outputs = vec![];
                let mut start = 0;
                for &size in &splits {
                    let end = start + size;
                    let slice = input_tensor.narrow(axis, start, size)?;
                    outputs.push(slice);
                    start = end;
                }

                for (output, slice) in node.outputs.iter().zip(outputs) {
                    values.insert(output.clone(), slice);
                }
            }
            "Expand" => {
                let input_tensor = get(&node.inputs[0])?;
                let input_shape = get(&node.inputs[1])?;
                if input_shape.rank() != 1 {
                    bail!("Expand expects 'shape' input to be 1D tensor: {input_shape:?}");
                }
                let input_tensor_dims = input_tensor.dims();
                let input_shape_dims = input_shape
                    .to_vec1::<i64>()?
                    .into_iter()
                    .map(|x| x as usize)
                    .collect::<Vec<_>>();
                let target_shape = broadcast_shape(input_tensor_dims, input_shape_dims.as_slice())?;
                let expanded_tensor = input_tensor.broadcast_as(target_shape)?;
                values.insert(node.outputs[0].clone(), expanded_tensor);
            }
            "Tile" => {
                let input = get(&node.inputs[0])?;
                let repeats = get(&node.inputs[1])?.to_vec1::<i64>()?;
                let mut result = input.clone();
                for (dim, &repeat) in repeats.iter().enumerate() {
                    if repeat > 1 {
                        let repeat = repeat as usize;
                        let tensors: Vec<_> = (0..repeat).map(|_| result.clone()).collect();
                        result = Tensor::cat(&tensors, dim)?;
                    }
                }
                values.insert(node.outputs[0].clone(), result);
            }
            "LayerNormalization" => {
                let input = get(&node.inputs[0])?;
                let gamma = get(&node.inputs[1])?;
                let beta = if node.inputs.len() > 2 {
                    Some(get(&node.inputs[2])?)
                } else {
                    None
                };
                let axis = get_attr_opt(node, "axis")
                    .map(|a| attr_i(a, node, "axis"))
                    .transpose()?
                    .unwrap_or(-1);
                let epsilon = get_attr_opt(node, "epsilon")
                    .map(|a| attr_f(a, node, "epsilon"))
                    .transpose()?
                    .unwrap_or(1e-5);

                let n_dims = input.dims().len();
                let normal_axis: usize = if axis < 0 {
                    (n_dims as i64 + axis) as usize
                } else {
                    axis as usize
                };

                // Compute mean and population variance along the normalized
                // axis (fork op order: mean → center → square → mean).
                let mean = input.mean_keepdim(vec![normal_axis])?;
                let centered = safe_sub(input, &mean)?;
                let pop_var = centered.sqr()?.mean_keepdim(vec![normal_axis])?;

                // Normalize: (x - mean) / sqrt(var + epsilon).  The epsilon
                // scalar is created as F64 (candle `Tensor::new(f64)`) and
                // safe_add promotes the pair to F32 — fork parity.
                let eps_t = Tensor::new(f64::from(epsilon), input.device())?;
                let pop_var_plus_eps = safe_add(&pop_var, &eps_t)?;
                let denom = pop_var_plus_eps.sqrt()?;
                let normalized = safe_div(&centered, &denom)?;

                // Scale and shift: gamma * normalized + beta
                let output = if let Some(beta) = beta {
                    safe_add(&safe_mul(&normalized, gamma)?, beta)?
                } else {
                    safe_mul(&normalized, gamma)?
                };
                values.insert(node.outputs[0].clone(), output);
            }
            "Gemm" => {
                let a = get(&node.inputs[0])?;
                let b = get(&node.inputs[1])?;
                let c = get(&node.inputs[2])?;

                let alpha = get_attr_opt(node, "alpha")
                    .map(|attr| attr_f(attr, node, "alpha"))
                    .transpose()?
                    .unwrap_or(1.0);
                let beta = get_attr_opt(node, "beta")
                    .map(|attr| attr_f(attr, node, "beta"))
                    .transpose()?
                    .unwrap_or(1.0);

                let alpha = Tensor::full(alpha, a.shape(), &Device::Cpu)?;
                let beta = Tensor::full(beta, c.shape(), &Device::Cpu)?;

                let trans_a = get_attr_opt(node, "transA")
                    .map(|attr| attr_i(attr, node, "transA"))
                    .transpose()?
                    .unwrap_or(0);
                let trans_b = get_attr_opt(node, "transB")
                    .map(|attr| attr_i(attr, node, "transB"))
                    .transpose()?
                    .unwrap_or(0);

                let a = if trans_a == 0 { a.clone() } else { a.t()? };
                let b = if trans_b == 0 { b.clone() } else { b.t()? };

                let a_mul = safe_mul(&a, &alpha)?;
                let c_mul = safe_mul(c, &beta)?;
                let output = safe_add(&safe_matmul(&a_mul, &b)?, &c_mul)?;
                values.insert(node.outputs[0].clone(), output);
            }
            "Conv" => {
                let dilations = get_attr_opt(node, "dilations");
                let groups = get_attr_opt(node, "group")
                    .map(|a| attr_i(a, node, "group"))
                    .transpose()?
                    .unwrap_or(1);
                let pads = get_attr_opt(node, "pads");
                let strides = get_attr_opt(node, "strides");
                if let Some(auto_pad) = get_attr_opt(node, "auto_pad") {
                    let s = String::from_utf8_lossy(attr_bytes(auto_pad, node, "auto_pad")?);
                    if s != "NOTSET" {
                        bail!("unsupported auto_pad {s}");
                    }
                }
                let xs = get(&node.inputs[0])?;
                let ws = get(&node.inputs[1])?;
                // Ensure input and weight have the same dtype.
                let (xs, ws) = promote_types(xs, ws)?;
                let ys = match ws.rank() {
                    // 1-D convolution only (all four TTS models).
                    3 => {
                        let (pads, xs) = match pads {
                            None => (0, xs.clone()),
                            Some(attr) => {
                                let p = attr_ints(attr, node, "pads")?;
                                match p {
                                    [p] => (*p as usize, xs.clone()),
                                    [p1, p2] => {
                                        if p1 == p2 {
                                            (*p1 as usize, xs.clone())
                                        } else {
                                            (
                                                0usize,
                                                xs.pad_with_zeros(2, *p1 as usize, *p2 as usize)?,
                                            )
                                        }
                                    }
                                    _ => bail!(
                                        "more pads than expected in conv1d {p:?} {}",
                                        node.name
                                    ),
                                }
                            }
                        };
                        let strides = match strides {
                            None => 1,
                            Some(attr) => {
                                let s = attr_ints(attr, node, "strides")?;
                                match s {
                                    [p] => *p as usize,
                                    _ => bail!(
                                        "more strides than expected in conv1d {s:?} {}",
                                        node.name
                                    ),
                                }
                            }
                        };
                        let dilations = match dilations {
                            None => 1,
                            Some(attr) => {
                                let d = attr_ints(attr, node, "dilations")?;
                                match d {
                                    [p] => *p as usize,
                                    _ => bail!(
                                        "more dilations than expected in conv1d {d:?} {}",
                                        node.name
                                    ),
                                }
                            }
                        };
                        xs.conv1d(&ws, pads, strides, dilations, groups as usize)?
                    }
                    rank => bail!(
                        "unsupported rank for weight matrix {rank} in conv {} (1-D only)",
                        node.name
                    ),
                };
                let ys = if node.inputs.len() > 2 {
                    let bs = get(&node.inputs[2])?;
                    let mut bs_shape = vec![1; ys.rank()];
                    bs_shape[1] = bs.elem_count();
                    safe_add(&ys, &bs.reshape(bs_shape)?)?
                } else {
                    ys
                };
                values.insert(node.outputs[0].clone(), ys);
            }
            "Where" => {
                let cond = get(&node.inputs[0])?;
                let a = get(&node.inputs[1])?;
                let b = get(&node.inputs[2])?;

                // where_cond requires all inputs the same shape; the ONNX
                // Where op only requires broadcastability — broadcast first.
                let shape = broadcast_shape_from_many(&[cond.dims(), a.dims(), b.dims()])?;
                let cond = cond.broadcast_as(shape.clone())?;
                let a = a.broadcast_as(shape.clone())?;
                let b = b.broadcast_as(shape)?;
                let output = safe_where(&cond, &a, &b)?;
                values.insert(node.outputs[0].clone(), output);
            }
            op_type => bail!("unsupported op_type {op_type} for op {node:?}"),
        }
    }

    graph
        .outputs
        .iter()
        .map(|output| match values.remove(&output.name) {
            None => bail!("cannot find output {}", output.name),
            Some(value) => Ok((output.name.clone(), value)),
        })
        .collect()
}

/// Pad a single dimension of `t` with `left`/`right` elements.  A zero pad
/// value uses candle's `pad_with_zeros` (the fork's exact path — bit-identical
/// and fast); a non-zero value is honored via explicit concatenation (the
/// fork silently dropped it).
fn pad_constant(
    t: &Tensor,
    dim: usize,
    left: usize,
    right: usize,
    value: Option<&Tensor>,
) -> Result<Tensor> {
    match value {
        None => t.pad_with_zeros(dim, left, right),
        Some(v) if scalar_is_zero(v)? => t.pad_with_zeros(dim, left, right),
        Some(v) => {
            // Explicit constant-value padding: build full tensors of the pad
            // value and concatenate along the padded dimension.
            let mut parts: Vec<Tensor> = Vec::new();
            if left > 0 {
                let mut l_shape = t.dims().to_vec();
                l_shape[dim] = left;
                parts.push(full_with(v, l_shape)?);
            }
            parts.push(t.clone());
            if right > 0 {
                let mut r_shape = t.dims().to_vec();
                r_shape[dim] = right;
                parts.push(full_with(v, r_shape)?);
            }
            Tensor::cat(&parts, dim)
        }
    }
}

fn scalar_is_zero(t: &Tensor) -> Result<bool> {
    let v = match t.dtype() {
        DType::F32 => f64::from(to_scalar_flexible::<f32>(t)?),
        DType::F64 => to_scalar_flexible::<f64>(t)?,
        DType::I64 => to_scalar_flexible::<i64>(t)? as f64,
        DType::U8 => f64::from(to_scalar_flexible::<u8>(t)?),
        DType::U32 => f64::from(to_scalar_flexible::<u32>(t)?),
        other => bail!("unsupported pad value dtype {other:?}"),
    };
    Ok(v == 0.0)
}

/// Build a tensor filled with the (single-element) value of `value`.
fn full_with(value: &Tensor, shape: Vec<usize>) -> Result<Tensor> {
    let dev = value.device();
    let t = match value.dtype() {
        DType::F32 => Tensor::full(to_scalar_flexible::<f32>(value)?, shape, dev)?,
        DType::F64 => Tensor::full(to_scalar_flexible::<f64>(value)?, shape, dev)?,
        DType::I64 => Tensor::full(to_scalar_flexible::<i64>(value)?, shape, dev)?,
        DType::U8 => Tensor::full(to_scalar_flexible::<u8>(value)?, shape, dev)?,
        DType::U32 => Tensor::full(to_scalar_flexible::<u32>(value)?, shape, dev)?,
        other => bail!("unsupported pad value dtype {other:?}"),
    };
    Ok(t)
}

#[cfg(test)]
#[path = "eval_tests.rs"]
mod eval_tests;
