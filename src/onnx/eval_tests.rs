//! Operator-level tests for the in-repo ONNX runtime.
//!
//! Ported from the removed candle-onnx-mahbot fork's tests
//! (`graph_patch_ops.rs` + inline `eval.rs` tests) and extended with the
//! mixed-dtype / Pad / Slice / Unsqueeze / Gemm / Concat patterns that the
//! four pinned TTS models actually exercise.  Models are built in memory from
//! the runtime's native types — no .onnx files, no protobuf encoding needed.

use super::{safe_add, safe_where, simple_eval};
use crate::onnx::{AttrKind, Attribute, Graph, Model, Node, ValueInfo};
use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op_type: op.to_string(),
        name: String::new(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attributes: vec![],
    }
}

fn node_attrs(op: &str, inputs: &[&str], outputs: &[&str], attributes: Vec<Attribute>) -> Node {
    Node {
        op_type: op.to_string(),
        name: String::new(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attributes,
    }
}

fn attr(name: &str, kind: AttrKind) -> Attribute {
    Attribute {
        name: name.to_string(),
        kind,
    }
}

fn graph(nodes: Vec<Node>, outputs: &[&str]) -> Graph {
    Graph {
        nodes,
        initializers: vec![],
        inputs: vec![],
        outputs: outputs
            .iter()
            .map(|n| ValueInfo {
                name: n.to_string(),
                elem_type: None,
            })
            .collect(),
    }
}

fn model(graph: Graph) -> Model {
    Model { graph }
}

fn run(model: &Model, inputs: Vec<(&str, Tensor)>) -> Result<HashMap<String, Tensor>> {
    let map = inputs
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    simple_eval(model, map)
}

fn get<'a>(m: &'a HashMap<String, Tensor>, name: &str) -> &'a Tensor {
    m.get(name).expect("output not found")
}

/// Implicit dtype promotion: `Add(F32, I64)` upcasts the integer input to F32
/// (fork `test_dtype_promotion_add`).
#[test]
fn test_dtype_promotion_add() -> Result<()> {
    let manual_graph = model(graph(vec![node("Add", &["x", "y"], &["z"])], &["z"]));
    let eval = run(
        &manual_graph,
        vec![
            ("x", Tensor::new(&[2.0f32, 4.0f32], &Device::Cpu)?),
            ("y", Tensor::new(&[1i64, 2i64], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    assert_eq!(z.dtype(), DType::F32, "mixed add must promote to F32");
    assert_eq!(z.to_vec1::<f32>()?, vec![3.0f32, 6.0]);
    Ok(())
}

/// Mixed-dtype `Mul(F32, I64)` — the actual model usage pattern in
/// duration_predictor / text_encoder.
#[test]
fn test_dtype_promotion_mul() -> Result<()> {
    let manual_graph = model(graph(vec![node("Mul", &["x", "y"], &["z"])], &["z"]));
    let eval = run(
        &manual_graph,
        vec![
            ("x", Tensor::new(&[2.0f32, 4.0f32], &Device::Cpu)?),
            ("y", Tensor::new(&[3i64, 5i64], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    assert_eq!(z.dtype(), DType::F32);
    assert_eq!(z.to_vec1::<f32>()?, vec![6.0f32, 20.0]);
    Ok(())
}

/// Softplus → Reciprocal chain (fork's graph test minus the dropped Max op).
#[test]
fn test_softplus_reciprocal_chain() -> Result<()> {
    let manual_graph = model(graph(
        vec![
            node("Softplus", &["x"], &["s"]),
            node("Reciprocal", &["s"], &["r"]),
        ],
        &["r"],
    ));
    let eval = run(
        &manual_graph,
        vec![("x", Tensor::new(&[1.0f32, 0.0f32], &Device::Cpu)?)],
    )?;
    let z = get(&eval, "r");
    let vals = z.to_vec1::<f32>()?;
    // softplus(1) = ln(1 + e) ≈ 1.313262, reciprocal ≈ 0.761467
    // softplus(0) = ln(2), reciprocal = 1/ln(2) = LOG2_E exactly
    let expected = [0.761_467_f32, std::f32::consts::LOG2_E];
    for (got, want) in vals.iter().zip(expected) {
        assert!((got - want).abs() < 1e-4, "got {got}, expected {want}");
    }
    Ok(())
}

/// LayerNormalization with explicit gamma/beta, default axis -1 / epsilon 1e-5.
/// Input [1, 2, 3]: mean 2, population variance 2/3, output
/// (x - 2) / sqrt(2/3 + 1e-5) ≈ [-1.224744, 0, 1.224744] (fork graph test).
#[test]
fn test_layer_normalization_graph() -> Result<()> {
    let manual_graph = model(graph(
        vec![node("LayerNormalization", &["x", "gamma", "beta"], &["z"])],
        &["z"],
    ));
    let eval = run(
        &manual_graph,
        vec![
            ("x", Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?),
            ("gamma", Tensor::new(&[1.0f32, 1.0, 1.0], &Device::Cpu)?),
            ("beta", Tensor::new(&[0.0f32, 0.0, 0.0], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    let vals = z.to_vec1::<f32>()?;
    let expected = [-1.224_744_f32, 0.0, 1.224_744];
    for (got, want) in vals.iter().zip(expected) {
        assert!((got - want).abs() < 1e-4, "got {got}, expected {want}");
    }
    Ok(())
}

/// LayerNormalization honors the explicit epsilon attribute — all 71 model
/// nodes use 1e-6, not the 1e-5 default.
#[test]
fn test_layer_normalization_explicit_epsilon() -> Result<()> {
    let ln = node_attrs(
        "LayerNormalization",
        &["x", "gamma", "beta"],
        &["z"],
        vec![
            attr("axis", AttrKind::Int(-1)),
            attr("epsilon", AttrKind::Float(1e-6)),
        ],
    );
    let manual_graph = model(graph(vec![ln], &["z"]));
    // Input all zeros → output all zeros regardless of epsilon.
    let eval = run(
        &manual_graph,
        vec![
            ("x", Tensor::new(&[0.0f32, 0.0, 0.0], &Device::Cpu)?),
            ("gamma", Tensor::new(&[1.0f32, 1.0, 1.0], &Device::Cpu)?),
            ("beta", Tensor::new(&[0.0f32, 0.0, 0.0], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    let vals = z.to_vec1::<f32>()?;
    assert!(vals.iter().all(|v| v.abs() < 1e-6), "got {vals:?}");
    // A tiny-variance input makes the epsilon term measurable.  x = [0, 0,
    // 1e-3]: mean m = 1e-3/3, centered = [-m, -m, 2m], population variance
    // = (m² + m² + 4m²)/3 = 2m².  With epsilon=1e-6 the output[2] element is
    // 2m / sqrt(2m² + 1e-6); with the default 1e-5 it would differ
    // measurably, so this guards the explicit-epsilon attribute.
    let eval = run(
        &manual_graph,
        vec![
            ("x", Tensor::new(&[0.0f32, 0.0, 1e-3], &Device::Cpu)?),
            ("gamma", Tensor::new(&[1.0f32, 1.0, 1.0], &Device::Cpu)?),
            ("beta", Tensor::new(&[0.0f32, 0.0, 0.0], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    let vals = z.to_vec1::<f32>()?;
    let m = 1e-3_f64 / 3.0;
    let var = 2.0 * m * m;
    let expected = 2.0 * m / (var + 1e-6).sqrt();
    assert!(
        (f64::from(vals[2]) - expected).abs() < 1e-6,
        "epsilon=1e-6 not honored: got {}, expected {expected}",
        vals[2]
    );
    Ok(())
}

/// Pad with mode="edge" on a 2D tensor, pads [1, 0, 1, 0] (fork graph test).
#[test]
fn test_pad_edge_mode_graph() -> Result<()> {
    let pad_node = node_attrs(
        "Pad",
        &["data", "pads"],
        &["z"],
        vec![attr("mode", AttrKind::Bytes(b"edge".to_vec()))],
    );
    let manual_graph = model(graph(vec![pad_node], &["z"]));
    let eval = run(
        &manual_graph,
        vec![
            (
                "data",
                Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu)?,
            ),
            ("pads", Tensor::new(&[1i64, 0, 1, 0], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    let vals = z.to_vec2::<f32>()?;
    let expected = vec![
        vec![1.0f32, 2.0],
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![3.0, 4.0],
    ];
    assert_eq!(vals, expected);
    Ok(())
}

/// Pad constant mode with the opset-18 3-input form whose constant_value
/// arrives as an EMPTY STRING (the actual model usage): treated as not
/// provided → zero padding.
#[test]
fn test_pad_constant_empty_third_input() -> Result<()> {
    let pad_node = node_attrs(
        "Pad",
        &["data", "pads", ""],
        &["z"],
        vec![attr("mode", AttrKind::Bytes(b"constant".to_vec()))],
    );
    let manual_graph = model(graph(vec![pad_node], &["z"]));
    let eval = run(
        &manual_graph,
        vec![
            (
                "data",
                Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu)?,
            ),
            ("pads", Tensor::new(&[1i64, 1, 0, 0], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    let vals = z.to_vec2::<f32>()?;
    // pads=[1,1,0,0] → dim0: pre=1, post=0; dim1: pre=1, post=0.
    let expected = vec![
        vec![0.0f32, 0.0, 0.0],
        vec![0.0, 1.0, 2.0],
        vec![0.0, 3.0, 4.0],
    ];
    assert_eq!(vals, expected);
    Ok(())
}

/// Pad constant mode with a PROVIDED non-zero pad value: honored (the fork
/// silently dropped it; the ticket requires it not be dropped).
#[test]
fn test_pad_constant_provided_value() -> Result<()> {
    let pad_node = node_attrs(
        "Pad",
        &["data", "pads", "value"],
        &["z"],
        vec![attr("mode", AttrKind::Bytes(b"constant".to_vec()))],
    );
    let manual_graph = model(graph(vec![pad_node], &["z"]));
    let eval = run(
        &manual_graph,
        vec![
            ("data", Tensor::new(&[1.0f32, 2.0], &Device::Cpu)?),
            ("pads", Tensor::new(&[1i64, 1], &Device::Cpu)?),
            ("value", Tensor::new(&[7.0f32], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    let vals = z.to_vec1::<f32>()?;
    assert_eq!(vals, vec![7.0f32, 1.0, 2.0, 7.0]);
    Ok(())
}

/// PReLU with a single-element slope applied to all channels (fork graph
/// test; upstream only handled per-channel slopes).
#[test]
fn test_prelu_scalar_slope_graph() -> Result<()> {
    let manual_graph = model(graph(vec![node("PRelu", &["x", "slope"], &["z"])], &["z"]));
    let eval = run(
        &manual_graph,
        vec![
            (
                "x",
                Tensor::new(&[[-1.0f32, 2.0], [3.0, -4.0]], &Device::Cpu)?,
            ),
            ("slope", Tensor::new(&[0.25f32], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    let vals = z.to_vec2::<f32>()?;
    let expected = vec![vec![-0.25f32, 2.0], vec![3.0, -1.0]];
    assert_eq!(vals, expected);
    Ok(())
}

/// Reciprocal: 1/x with the fork's inline test values.
#[test]
fn test_operator_reciprocal() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::new(&[2.0f32, 4.0, 0.5, -1.0], dev)?;
    let result = input.recip()?;
    let vals: Vec<f32> = result.to_vec1()?;
    assert!(
        (vals[0] - 0.5).abs() < 1e-6,
        "1/2 should be 0.5, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 0.25).abs() < 1e-6,
        "1/4 should be 0.25, got {}",
        vals[1]
    );
    assert!(
        (vals[2] - 2.0).abs() < 1e-6,
        "1/0.5 should be 2.0, got {}",
        vals[2]
    );
    assert!(
        (vals[3] - (-1.0)).abs() < 1e-6,
        "1/-1 should be -1.0, got {}",
        vals[3]
    );
    let zero = Tensor::new(&[0.0f32], dev)?;
    let inf_result = zero.recip()?;
    let inf_vals: Vec<f32> = inf_result.to_vec1()?;
    assert!(
        inf_vals[0].is_infinite(),
        "1/0 should be inf, got {}",
        inf_vals[0]
    );
    Ok(())
}

/// Softplus numerical stability (both branches), fork inline test.
#[test]
fn test_operator_softplus() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::new(&[-100.0f32, -10.0, 0.0, 5.0, 20.0, 25.0, 100.0], dev)?;
    // Stable softplus: x > 20 ? x : ln(exp(x) + 1)
    let mask = input.gt(20.0f64)?;
    let ones = Tensor::ones(input.dims(), input.dtype(), input.device())?;
    let exp_add_one = safe_add(&input.exp()?, &ones)?;
    let stable = exp_add_one.log()?;
    let output = safe_where(&mask, &input, &stable)?;
    let vals: Vec<f32> = output.to_vec1()?;
    assert!(
        vals[0] < 1e-40,
        "softplus(-100) should be ~0, got {}",
        vals[0]
    );
    assert!(
        (vals[2] - 0.693147).abs() < 1e-5,
        "softplus(0) should be ~0.693147, got {}",
        vals[2]
    );
    assert!(
        (vals[3] - 5.0067).abs() < 1e-3,
        "softplus(5) should be ~5.0067, got {}",
        vals[3]
    );
    assert!(
        (vals[5] - 25.0).abs() < 1e-5,
        "softplus(25) should be 25.0 (stable branch), got {}",
        vals[5]
    );
    assert!(
        (vals[6] - 100.0).abs() < 1e-5,
        "softplus(100) should be 100.0 (stable branch), got {}",
        vals[6]
    );
    Ok(())
}

/// Slice with negative steps (step = -1; 60 nodes across the models use it).
#[test]
fn test_slice_negative_step() -> Result<()> {
    let manual_graph = model(graph(
        vec![node("Slice", &["x", "s", "e", "a", "st"], &["z"])],
        &["z"],
    ));
    let eval = run(
        &manual_graph,
        vec![
            (
                "x",
                Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &Device::Cpu)?,
            ),
            ("s", Tensor::new(&[-1i64], &Device::Cpu)?),
            ("e", Tensor::new(&[-4i64], &Device::Cpu)?),
            ("a", Tensor::new(&[1i64], &Device::Cpu)?),
            ("st", Tensor::new(&[-1i64], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    // Axis 1, start=-1→2, end=-4→-1 (clamped to -1), step=-1 → indices [2,1,0].
    let vals = z.to_vec2::<f32>()?;
    assert_eq!(vals, vec![vec![3.0f32, 2.0, 1.0], vec![6.0, 5.0, 4.0]]);
    Ok(())
}

/// Slice 4-input form (axes provided, steps default to 1).
#[test]
fn test_slice_4_input_form() -> Result<()> {
    let manual_graph = model(graph(
        vec![node("Slice", &["x", "s", "e", "a"], &["z"])],
        &["z"],
    ));
    let eval = run(
        &manual_graph,
        vec![
            (
                "x",
                Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &Device::Cpu)?,
            ),
            ("s", Tensor::new(&[1i64], &Device::Cpu)?),
            ("e", Tensor::new(&[2i64], &Device::Cpu)?),
            ("a", Tensor::new(&[1i64], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    let vals = z.to_vec2::<f32>()?;
    assert_eq!(vals, vec![vec![2.0f32], vec![5.0]]);
    Ok(())
}

/// Gemm with transB=1 (the model usage).
#[test]
fn test_gemm_transb() -> Result<()> {
    let gemm = node_attrs(
        "Gemm",
        &["a", "b", "c"],
        &["z"],
        vec![
            attr("alpha", AttrKind::Float(1.0)),
            attr("beta", AttrKind::Float(1.0)),
            attr("transB", AttrKind::Int(1)),
        ],
    );
    let manual_graph = model(graph(vec![gemm], &["z"]));
    let eval = run(
        &manual_graph,
        vec![
            ("a", Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu)?),
            (
                "b",
                Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu)?,
            ),
            ("c", Tensor::new(&[0.5f32, 0.5], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    // a @ b^T + c = [1,2] @ [[1,0],[0,1]] + [0.5,0.5] = [1.5, 2.5]
    let vals = z.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(vals, vec![1.5f32, 2.5]);
    Ok(())
}

/// Unsqueeze with a negative axis (the fork's off-by-one handling).
#[test]
fn test_unsqueeze_negative_axis() -> Result<()> {
    let manual_graph = model(graph(
        vec![node("Unsqueeze", &["x", "axes"], &["z"])],
        &["z"],
    ));
    let eval = run(
        &manual_graph,
        vec![
            ("x", Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?),
            ("axes", Tensor::new(&[-1i64], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    // rank 1 → -1 → 1 - 1 + 1 = 1 → shape [3, 1]
    assert_eq!(z.dims(), &[3, 1]);

    // Also verify the positive in-bounds axis path.
    let eval = run(
        &manual_graph,
        vec![
            ("x", Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?),
            ("axes", Tensor::new(&[0i64], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    assert_eq!(z.dims(), &[1, 3]);
    Ok(())
}

/// Concat with mixed dtypes promotes by majority rule (the model pattern: one
/// I64 index tensor among many F32 values).
#[test]
fn test_concat_majority_dtype_promotion() -> Result<()> {
    let cat = node_attrs(
        "Concat",
        &["a", "b", "c"],
        &["z"],
        vec![attr("axis", AttrKind::Int(0))],
    );
    let manual_graph = model(graph(vec![cat], &["z"]));
    let eval = run(
        &manual_graph,
        vec![
            ("a", Tensor::new(&[[1.0f32]], &Device::Cpu)?),
            ("b", Tensor::new(&[[2i64]], &Device::Cpu)?),
            ("c", Tensor::new(&[[3.0f32]], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    assert_eq!(z.dtype(), DType::F32, "majority F32 must win");
    assert_eq!(
        z.to_vec2::<f32>()?,
        vec![vec![1.0f32], vec![2.0], vec![3.0]]
    );
    Ok(())
}

/// Concat squeezes trailing singleton dims to the minimum input rank (the
/// fork's rank hack used by the models).
#[test]
fn test_concat_trailing_singleton_squeeze() -> Result<()> {
    let cat = node_attrs(
        "Concat",
        &["a", "b"],
        &["z"],
        vec![attr("axis", AttrKind::Int(0))],
    );
    let manual_graph = model(graph(vec![cat], &["z"]));
    // a: [2, 1] (rank 2), b: [2] (rank 1) → a squeezed to [2]
    let eval = run(
        &manual_graph,
        vec![
            ("a", Tensor::new(&[[1.0f32], [2.0]], &Device::Cpu)?),
            ("b", Tensor::new(&[3.0f32, 4.0], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    assert_eq!(z.dims(), &[4]);
    assert_eq!(z.to_vec1::<f32>()?, vec![1.0f32, 2.0, 3.0, 4.0]);
    Ok(())
}

/// Pow with a negative F32 base and I64 scalar exponent — the actual model
/// pattern (`Pow(F32, I64)` on attention scores; the scalar-exponent path
/// reads the I64 exponent via a cast to F64 and applies `powf`).
#[test]
fn test_pow_negative_base_f32_i64_exp() -> Result<()> {
    let manual_graph = model(graph(
        vec![
            node("Pow", &["x", "exp2"], &["z2"]),
            node("Pow", &["x", "exp3"], &["z3"]),
        ],
        &["z2", "z3"],
    ));
    let eval = run(
        &manual_graph,
        vec![
            ("x", Tensor::new(&[-2.0f32, -3.0], &Device::Cpu)?),
            ("exp2", Tensor::new(&[2i64], &Device::Cpu)?),
            ("exp3", Tensor::new(&[3i64], &Device::Cpu)?),
        ],
    )?;
    // powf path: (-2)² = 4, (-3)² = 9; (-2)³ = -8, (-3)³ = -27.
    assert_eq!(get(&eval, "z2").to_vec1::<f32>()?, vec![4.0f32, 9.0]);
    assert_eq!(get(&eval, "z3").to_vec1::<f32>()?, vec![-8.0f32, -27.0]);
    Ok(())
}

/// Cast BOOL→U8 (candle stores bools as U8).
#[test]
fn test_cast_bool_to_u8() -> Result<()> {
    let cast = node_attrs("Cast", &["x"], &["z"], vec![attr("to", AttrKind::Int(9))]);
    let manual_graph = model(graph(vec![cast], &["z"]));
    let eval = run(
        &manual_graph,
        vec![("x", Tensor::new(&[0u8, 1u8, 1u8], &Device::Cpu)?)],
    )?;
    let z = get(&eval, "z");
    assert_eq!(z.dtype(), DType::U8);
    assert_eq!(z.to_vec1::<u8>()?, vec![0, 1, 1]);
    Ok(())
}

/// Cast INT32→I64 (fork parity).
#[test]
fn test_cast_int32_to_i64() -> Result<()> {
    let cast = node_attrs("Cast", &["x"], &["z"], vec![attr("to", AttrKind::Int(6))]);
    let manual_graph = model(graph(vec![cast], &["z"]));
    let eval = run(
        &manual_graph,
        vec![("x", Tensor::new(&[1i64, 2i64], &Device::Cpu)?)],
    )?;
    let z = get(&eval, "z");
    assert_eq!(z.dtype(), DType::I64);
    Ok(())
}

/// Reshape: 0 keeps the input dimension at that index (fork parity).
#[test]
fn test_reshape_zero_keeps_input_dim() -> Result<()> {
    let manual_graph = model(graph(
        vec![node("Reshape", &["x", "shape"], &["z"])],
        &["z"],
    ));
    let eval = run(
        &manual_graph,
        vec![
            (
                "x",
                Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu)?,
            ),
            ("shape", Tensor::new(&[0i64, 2], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    // [0, 2] → keep dim 0 (2), explicit 2 → [2, 2]
    assert_eq!(z.dims(), &[2, 2]);
    assert_eq!(z.to_vec2::<f32>()?, vec![vec![1.0f32, 2.0], vec![3.0, 4.0]]);
    Ok(())
}

/// Reshape: -1 infers the dimension (fork parity).
#[test]
fn test_reshape_minus_one_infers() -> Result<()> {
    let manual_graph = model(graph(
        vec![node("Reshape", &["x", "shape"], &["z"])],
        &["z"],
    ));
    let eval = run(
        &manual_graph,
        vec![
            (
                "x",
                Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu)?,
            ),
            ("shape", Tensor::new(&[-1i64, 2], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    // [-1, 2] → 4/2 = 2 → [2, 2]
    assert_eq!(z.dims(), &[2, 2]);
    assert_eq!(z.to_vec2::<f32>()?, vec![vec![1.0f32, 2.0], vec![3.0, 4.0]]);
    Ok(())
}

/// Where with explicit broadcast (cond/a/b all broadcastable, not equal).
#[test]
fn test_where_broadcast() -> Result<()> {
    let manual_graph = model(graph(
        vec![node("Where", &["cond", "a", "b"], &["z"])],
        &["z"],
    ));
    let eval = run(
        &manual_graph,
        vec![
            ("cond", Tensor::new(&[[1u8], [0u8]], &Device::Cpu)?),
            ("a", Tensor::new(&[10.0f32, 11.0, 12.0], &Device::Cpu)?),
            ("b", Tensor::new(&[20.0f32, 21.0, 22.0], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    let vals = z.to_vec2::<f32>()?;
    assert_eq!(
        vals,
        vec![vec![10.0f32, 11.0, 12.0], vec![20.0, 21.0, 22.0]]
    );
    Ok(())
}

/// Gather with negative indices on axis 0 (the mask-normalization path).
#[test]
fn test_gather_negative_indices() -> Result<()> {
    let gather = node_attrs(
        "Gather",
        &["x", "idx"],
        &["z"],
        vec![attr("axis", AttrKind::Int(0))],
    );
    let manual_graph = model(graph(vec![gather], &["z"]));
    let eval = run(
        &manual_graph,
        vec![
            (
                "x",
                Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &Device::Cpu)?,
            ),
            ("idx", Tensor::new(&[0i64, -1], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    let vals = z.to_vec2::<f32>()?;
    assert_eq!(vals, vec![vec![1.0f32, 2.0], vec![5.0, 6.0]]);
    Ok(())
}

/// Conv 1-D, grouped, with bias (the model's dwconv/pwconv pattern).
#[test]
fn test_conv1d_grouped_with_bias() -> Result<()> {
    // Depthwise conv: group = in_channels = 4, kernel 3, in/out channels 4.
    let conv = node_attrs(
        "Conv",
        &["x", "w", "b"],
        &["z"],
        vec![
            attr("group", AttrKind::Int(4)),
            attr("kernel_shape", AttrKind::Ints(vec![3])),
            attr("pads", AttrKind::Ints(vec![0, 0])),
            attr("strides", AttrKind::Ints(vec![1])),
            attr("dilations", AttrKind::Ints(vec![1])),
        ],
    );
    let manual_graph = model(graph(vec![conv], &["z"]));
    let w = vec![
        1.0f32, 0.0, 0.0, // group 0, channel 0
        0.0, 1.0, 0.0, // group 0, channel 1
        0.0, 0.0, 1.0, // group 1, channel 2
        1.0, 1.0, 0.0, // group 1, channel 3
    ];
    let eval = run(
        &manual_graph,
        vec![
            (
                "x",
                Tensor::new(
                    &[[
                        [1.0f32, 2.0, 3.0, 4.0, 5.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0],
                    ]],
                    &Device::Cpu,
                )?,
            ),
            ("w", Tensor::from_vec(w, (4, 1, 3), &Device::Cpu)?),
            ("b", Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    let vals = z.to_vec3::<f32>()?;
    // Channel 0: conv [1,0,0] → [1,2,3]; channel 1: [0,1,0] → [2,3,4];
    // channel 2: [0,0,1] → [3,4,5]; channel 3: [1,1,0] → [3,5,7].
    assert_eq!(
        vals,
        vec![vec![
            vec![1.0f32, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
            vec![3.0, 5.0, 7.0]
        ]]
    );
    Ok(())
}

/// ConstantOfShape with a provided value tensor.
#[test]
fn test_constant_of_shape_with_value() -> Result<()> {
    let cos = node_attrs(
        "ConstantOfShape",
        &["shape"],
        &["z"],
        vec![attr(
            "value",
            AttrKind::Tensor(Tensor::new(&[2.0f32], &Device::Cpu).unwrap()),
        )],
    );
    let manual_graph = model(graph(vec![cos], &["z"]));
    let eval = run(
        &manual_graph,
        vec![("shape", Tensor::new(&[2i64, 3], &Device::Cpu)?)],
    )?;
    let z = get(&eval, "z");
    assert_eq!(z.dims(), &[2, 3]);
    let vals = z.to_vec2::<f32>()?;
    assert_eq!(vals, vec![vec![2.0f32; 3], vec![2.0; 3]]);
    Ok(())
}

/// Shape with explicit start/end attributes spanning the full rank (the fork
/// builds the output with shape `(rank,)`, so the slice must cover all dims —
/// the models use the defaults).
#[test]
fn test_shape_start_end() -> Result<()> {
    let shape = node_attrs(
        "Shape",
        &["x"],
        &["z"],
        vec![
            attr("start", AttrKind::Int(0)),
            attr("end", AttrKind::Int(-1)),
        ],
    );
    let manual_graph = model(graph(vec![shape], &["z"]));
    let eval = run(
        &manual_graph,
        vec![(
            "x",
            Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu)?,
        )],
    )?;
    let z = get(&eval, "z");
    assert_eq!(z.to_vec1::<i64>()?, vec![2, 2]);
    Ok(())
}

/// Split with equal division and remainder-to-last.
#[test]
fn test_split_remainder_to_last() -> Result<()> {
    let split = node_attrs(
        "Split",
        &["x"],
        &["a", "b"],
        vec![attr("axis", AttrKind::Int(0))],
    );
    let manual_graph = model(graph(vec![split], &["a", "b"]));
    let eval = run(
        &manual_graph,
        vec![("x", Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?)],
    )?;
    assert_eq!(get(&eval, "a").to_vec1::<f32>()?, vec![1.0f32]);
    assert_eq!(get(&eval, "b").to_vec1::<f32>()?, vec![2.0f32, 3.0]);
    Ok(())
}

/// Equal produces a bool tensor (stored as U8).
#[test]
fn test_equal_bool_output() -> Result<()> {
    let manual_graph = model(graph(vec![node("Equal", &["x", "y"], &["z"])], &["z"]));
    let eval = run(
        &manual_graph,
        vec![
            ("x", Tensor::new(&[1i64, 2, 3], &Device::Cpu)?),
            ("y", Tensor::new(&[1i64, 0, 3], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    assert_eq!(z.dtype(), DType::U8);
    assert_eq!(z.to_vec1::<u8>()?, vec![1, 0, 1]);
    Ok(())
}

/// Softmax with an explicit axis attribute.
#[test]
fn test_softmax_axis() -> Result<()> {
    let softmax = node_attrs(
        "Softmax",
        &["x"],
        &["z"],
        vec![attr("axis", AttrKind::Int(-1))],
    );
    let manual_graph = model(graph(vec![softmax], &["z"]));
    let eval = run(
        &manual_graph,
        vec![("x", Tensor::new(&[[1.0f32, 1.0, 1.0]], &Device::Cpu)?)],
    )?;
    let z = get(&eval, "z");
    let vals = z.to_vec2::<f32>()?;
    let expected = 1.0f32 / 3.0;
    for v in &vals[0] {
        assert!((v - expected).abs() < 1e-6, "got {v}, expected {expected}");
    }
    Ok(())
}

/// ReduceSum with an axes input and keepdims=0.
#[test]
fn test_reduce_sum_axes_keepdims() -> Result<()> {
    let rs = node_attrs(
        "ReduceSum",
        &["x", "axes"],
        &["z"],
        vec![attr("keepdims", AttrKind::Int(0))],
    );
    let manual_graph = model(graph(vec![rs], &["z"]));
    let eval = run(
        &manual_graph,
        vec![
            ("x", Tensor::new(&[[1.0f32, 2.0, 3.0]], &Device::Cpu)?),
            ("axes", Tensor::new(&[1i64], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    assert_eq!(z.dims(), &[1]);
    assert_eq!(z.to_vec1::<f32>()?, vec![6.0f32]);
    Ok(())
}

/// Transpose with a perm attribute.
#[test]
fn test_transpose_perm() -> Result<()> {
    let tr = node_attrs(
        "Transpose",
        &["x"],
        &["z"],
        vec![attr("perm", AttrKind::Ints(vec![0, 2, 1]))],
    );
    let manual_graph = model(graph(vec![tr], &["z"]));
    let eval = run(
        &manual_graph,
        vec![(
            "x",
            Tensor::new(&[[[1.0f32, 2.0], [3.0, 4.0]]], &Device::Cpu)?,
        )],
    )?;
    let z = get(&eval, "z");
    assert_eq!(z.dims(), &[1, 2, 2]);
    let vals = z.to_vec3::<f32>()?;
    assert_eq!(vals, vec![vec![vec![1.0f32, 3.0], vec![2.0, 4.0]]]);
    Ok(())
}

/// Tile with repeats.
#[test]
fn test_tile() -> Result<()> {
    let manual_graph = model(graph(vec![node("Tile", &["x", "repeats"], &["z"])], &["z"]));
    let eval = run(
        &manual_graph,
        vec![
            ("x", Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu)?),
            ("repeats", Tensor::new(&[2i64, 1], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    let vals = z.to_vec2::<f32>()?;
    assert_eq!(vals, vec![vec![1.0f32, 2.0], vec![1.0, 2.0]]);
    Ok(())
}

/// Expand to a larger shape.
#[test]
fn test_expand() -> Result<()> {
    let manual_graph = model(graph(vec![node("Expand", &["x", "shape"], &["z"])], &["z"]));
    let eval = run(
        &manual_graph,
        vec![
            ("x", Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu)?),
            ("shape", Tensor::new(&[3i64, 2], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    let vals = z.to_vec2::<f32>()?;
    assert_eq!(
        vals,
        vec![vec![1.0f32, 2.0], vec![1.0, 2.0], vec![1.0, 2.0]]
    );
    Ok(())
}

/// Clip with min/max inputs.
#[test]
fn test_clip() -> Result<()> {
    let manual_graph = model(graph(
        vec![node("Clip", &["x", "min", "max"], &["z"])],
        &["z"],
    ));
    let eval = run(
        &manual_graph,
        vec![
            ("x", Tensor::new(&[-2.0f32, 0.5, 3.0], &Device::Cpu)?),
            ("min", Tensor::new(&[-1.0f32], &Device::Cpu)?),
            ("max", Tensor::new(&[1.0f32], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    assert_eq!(z.to_vec1::<f32>()?, vec![-1.0f32, 0.5, 1.0]);
    Ok(())
}

/// BatchNormalization in inference mode (5 inputs, epsilon attribute).
#[test]
fn test_batch_normalization() -> Result<()> {
    let bn = node_attrs(
        "BatchNormalization",
        &["x", "w", "b", "mean", "var"],
        &["z"],
        vec![
            attr("epsilon", AttrKind::Float(1e-5)),
            attr("training_mode", AttrKind::Int(0)),
        ],
    );
    let manual_graph = model(graph(vec![bn], &["z"]));
    let eval = run(
        &manual_graph,
        vec![
            (
                "x",
                Tensor::new(&[[[1.0f32, 2.0], [3.0, 4.0]]], &Device::Cpu)?,
            ),
            ("w", Tensor::new(&[1.0f32, 1.0], &Device::Cpu)?),
            ("b", Tensor::new(&[0.0f32, 0.0], &Device::Cpu)?),
            ("mean", Tensor::new(&[2.0f32, 3.5], &Device::Cpu)?),
            ("var", Tensor::new(&[1.0f32, 1.0], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    let vals = z.to_vec3::<f32>()?;
    // (x - mean) / sqrt(var + eps): channel 0 → [-1, 0], channel 1 → [-0.5, 0.5]
    assert!((vals[0][0][0] - (-1.0)).abs() < 1e-4);
    assert!(vals[0][0][1].abs() < 1e-6);
    assert!((vals[0][1][0] + 0.5).abs() < 1e-4);
    assert!((vals[0][1][1] - 0.5).abs() < 1e-4);
    Ok(())
}

/// Cos/Sin/Erf/Tanh/Exp elementary ops.
#[test]
fn test_elementary_ops() -> Result<()> {
    let manual_graph = model(graph(
        vec![
            node("Cos", &["x"], &["c"]),
            node("Sin", &["x"], &["s"]),
            node("Tanh", &["x"], &["t"]),
            node("Exp", &["x"], &["e"]),
            node("Erf", &["x"], &["r"]),
        ],
        &["c", "s", "t", "e", "r"],
    ));
    let eval = run(
        &manual_graph,
        vec![("x", Tensor::new(&[0.0f32, 1.0], &Device::Cpu)?)],
    )?;
    let c = get(&eval, "c").to_vec1::<f32>()?;
    let s = get(&eval, "s").to_vec1::<f32>()?;
    let t = get(&eval, "t").to_vec1::<f32>()?;
    let e = get(&eval, "e").to_vec1::<f32>()?;
    let r = get(&eval, "r").to_vec1::<f32>()?;
    assert!((c[0] - 1.0).abs() < 1e-6);
    assert!(s[0].abs() < 1e-6);
    assert!(t[0].abs() < 1e-6);
    assert!((e[0] - 1.0).abs() < 1e-6);
    assert!(r[0].abs() < 1e-6);
    assert!((c[1] - 0.5403).abs() < 1e-4);
    assert!((s[1] - 0.8414).abs() < 1e-4);
    assert!((t[1] - 0.7615).abs() < 1e-4);
    assert!((e[1] - std::f32::consts::E).abs() < 1e-6);
    assert!((r[1] - 0.8427).abs() < 1e-4);
    Ok(())
}

/// Div/Sub with mixed dtypes (model patterns).
#[test]
fn test_div_sub_mixed_dtype() -> Result<()> {
    let manual_graph = model(graph(
        vec![
            node("Div", &["a", "b"], &["d"]),
            node("Sub", &["a", "c"], &["s"]),
        ],
        &["d", "s"],
    ));
    let eval = run(
        &manual_graph,
        vec![
            ("a", Tensor::new(&[4.0f32, 9.0], &Device::Cpu)?),
            ("b", Tensor::new(&[2i64, 3i64], &Device::Cpu)?),
            ("c", Tensor::new(&[1i64, 1i64], &Device::Cpu)?),
        ],
    )?;
    assert_eq!(get(&eval, "d").to_vec1::<f32>()?, vec![2.0f32, 3.0]);
    assert_eq!(get(&eval, "s").to_vec1::<f32>()?, vec![3.0f32, 8.0]);
    Ok(())
}

/// Squeeze with an axes input.
#[test]
fn test_squeeze_axes_input() -> Result<()> {
    let manual_graph = model(graph(vec![node("Squeeze", &["x", "axes"], &["z"])], &["z"]));
    let eval = run(
        &manual_graph,
        vec![
            ("x", Tensor::new(&[[[1.0f32], [2.0]]], &Device::Cpu)?),
            ("axes", Tensor::new(&[2i64], &Device::Cpu)?),
        ],
    )?;
    let z = get(&eval, "z");
    assert_eq!(z.dims(), &[1, 2]);
    Ok(())
}

/// Relu.
#[test]
fn test_relu() -> Result<()> {
    let manual_graph = model(graph(vec![node("Relu", &["x"], &["z"])], &["z"]));
    let eval = run(
        &manual_graph,
        vec![("x", Tensor::new(&[-1.0f32, 0.0, 2.0], &Device::Cpu)?)],
    )?;
    assert_eq!(get(&eval, "z").to_vec1::<f32>()?, vec![0.0f32, 0.0, 2.0]);
    Ok(())
}

/// MatMul (2-D).
#[test]
fn test_matmul() -> Result<()> {
    let manual_graph = model(graph(vec![node("MatMul", &["a", "b"], &["z"])], &["z"]));
    let eval = run(
        &manual_graph,
        vec![
            (
                "a",
                Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu)?,
            ),
            (
                "b",
                Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu)?,
            ),
        ],
    )?;
    let z = get(&eval, "z");
    let vals = z.to_vec2::<f32>()?;
    assert_eq!(vals, vec![vec![1.0f32, 2.0], vec![3.0, 4.0]]);
    Ok(())
}

/// Constant nodes resolve from the pre-converted tensor attribute.
#[test]
fn test_constant() -> Result<()> {
    let c = node_attrs(
        "Constant",
        &[],
        &["z"],
        vec![attr(
            "value",
            AttrKind::Tensor(Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu).unwrap()),
        )],
    );
    let manual_graph = model(graph(vec![c], &["z"]));
    let eval = run(&manual_graph, vec![])?;
    let z = get(&eval, "z");
    assert_eq!(z.to_vec2::<f32>()?, vec![vec![1.0f32, 2.0]]);
    Ok(())
}

/// The reader's dtype mapping used by Cast (INT64=7, BOOL=9).
#[test]
fn test_cast_to_int64() -> Result<()> {
    let cast = node_attrs("Cast", &["x"], &["z"], vec![attr("to", AttrKind::Int(7))]);
    let manual_graph = model(graph(vec![cast], &["z"]));
    let eval = run(
        &manual_graph,
        vec![("x", Tensor::new(&[1u8, 0u8], &Device::Cpu)?)],
    )?;
    let z = get(&eval, "z");
    assert_eq!(z.dtype(), DType::I64);
    assert_eq!(z.to_vec1::<i64>()?, vec![1, 0]);
    Ok(())
}
