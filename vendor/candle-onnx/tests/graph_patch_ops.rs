//! End-to-end graph-level smoke tests for the patch surface (delta vs upstream
//! candle-onnx 0.10.2): implicit dtype promotion, Softplus/Reciprocal/Max,
//! LayerNormalization, Pad `edge` mode, and PReLU scalar-slope support — each
//! exercised through [`candle_onnx::simple_eval`], the real ONNX graph
//! evaluator that the MahBot voice pipeline runs.
//!
//! Unlike upstream's `tests/ops.rs`, these tests are self-contained: the model
//! protos are built in memory, so no external model downloads are required.

use candle_core::{Device, Result, Tensor};
use candle_onnx::onnx::attribute_proto::AttributeType;
use candle_onnx::onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};
use candle_onnx::simple_eval;
use std::collections::HashMap;

fn model(graph: GraphProto) -> ModelProto {
    ModelProto {
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        ir_version: 0,
        opset_import: vec![],
        producer_name: String::new(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 0,
        doc_string: String::new(),
        graph: Some(graph),
    }
}

fn graph(nodes: Vec<NodeProto>, outputs: &[&str]) -> GraphProto {
    GraphProto {
        node: nodes,
        name: String::new(),
        initializer: vec![],
        input: vec![],
        output: outputs
            .iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                doc_string: String::new(),
                r#type: None,
            })
            .collect(),
        value_info: vec![],
        doc_string: String::new(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }
}

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    NodeProto {
        op_type: op.to_string(),
        domain: String::new(),
        attribute: vec![],
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        name: String::new(),
        doc_string: String::new(),
    }
}

/// Implicit dtype promotion: `Add(F32, I64)` upcasts the integer input to F32.
/// Upstream fails this with a dtype mismatch on `broadcast_add`; the fork's
/// `promote_types` makes it succeed.
#[test]
fn test_dtype_promotion_add() -> Result<()> {
    let manual_graph = model(graph(vec![node("Add", &["x", "y"], &["z"])], &["z"]));

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::new(&[2.0f32, 4.0f32], &Device::Cpu)?,
    );
    inputs.insert("y".to_string(), Tensor::new(&[1i64, 2i64], &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;
    let z = eval.get("z").expect("output 'z' not found");
    assert_eq!(
        z.dtype(),
        candle_core::DType::F32,
        "mixed add must promote to F32"
    );
    assert_eq!(z.to_vec1::<f32>()?, vec![3.0f32, 6.0]);
    Ok(())
}

/// Softplus → Reciprocal → Max chain (three ops added by the fork), evaluated
/// as one graph.
#[test]
fn test_softplus_reciprocal_max_chain() -> Result<()> {
    let manual_graph = model(graph(
        vec![
            node("Softplus", &["x"], &["s"]),
            node("Reciprocal", &["s"], &["r"]),
            node("Max", &["r", "y"], &["z"]),
        ],
        &["z"],
    ));

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::new(&[1.0f32, 0.0f32], &Device::Cpu)?,
    );
    inputs.insert(
        "y".to_string(),
        Tensor::new(&[0.5f32, 0.25f32], &Device::Cpu)?,
    );

    let eval = simple_eval(&manual_graph, inputs)?;
    let z = eval.get("z").expect("output 'z' not found");
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
/// Input [1, 2, 3]: mean 2, population variance 2/3, so the output is
/// (x - 2) / sqrt(2/3 + 1e-5) ≈ [-1.224744, 0, 1.224744].
#[test]
fn test_layer_normalization_graph() -> Result<()> {
    let manual_graph = model(graph(
        vec![node("LayerNormalization", &["x", "gamma", "beta"], &["z"])],
        &["z"],
    ));

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?,
    );
    inputs.insert(
        "gamma".to_string(),
        Tensor::new(&[1.0f32, 1.0, 1.0], &Device::Cpu)?,
    );
    inputs.insert(
        "beta".to_string(),
        Tensor::new(&[0.0f32, 0.0, 0.0], &Device::Cpu)?,
    );

    let eval = simple_eval(&manual_graph, inputs)?;
    let z = eval.get("z").expect("output 'z' not found");
    let vals = z.to_vec1::<f32>()?;
    let expected = [-1.224_744_f32, 0.0, 1.224_744];
    for (got, want) in vals.iter().zip(expected) {
        assert!((got - want).abs() < 1e-4, "got {got}, expected {want}");
    }
    Ok(())
}

/// Pad with mode="edge" on a 2D tensor, pads [1, 0, 1, 0] (rank 2 → 4 values).
#[test]
fn test_pad_edge_mode_graph() -> Result<()> {
    let mut pad_node = node("Pad", &["data", "pads"], &["z"]);
    pad_node.attribute = vec![AttributeProto {
        name: "mode".to_string(),
        r#type: AttributeType::String as i32,
        s: b"edge".to_vec(),
        ..Default::default()
    }];
    let manual_graph = model(graph(vec![pad_node], &["z"]));

    let mut inputs = HashMap::new();
    inputs.insert(
        "data".to_string(),
        Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu)?,
    );
    inputs.insert(
        "pads".to_string(),
        Tensor::new(&[1i64, 0, 1, 0], &Device::Cpu)?,
    );

    let eval = simple_eval(&manual_graph, inputs)?;
    let z = eval.get("z").expect("output 'z' not found");
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

/// PReLU with a single-element slope (ONNX allows one scalar slope applied to
/// all channels); upstream only handled per-channel slopes.
#[test]
fn test_prelu_scalar_slope_graph() -> Result<()> {
    let manual_graph = model(graph(vec![node("PRelu", &["x", "slope"], &["z"])], &["z"]));

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::new(&[[-1.0f32, 2.0], [3.0, -4.0]], &Device::Cpu)?,
    );
    inputs.insert("slope".to_string(), Tensor::new(&[0.25f32], &Device::Cpu)?);

    let eval = simple_eval(&manual_graph, inputs)?;
    let z = eval.get("z").expect("output 'z' not found");
    let vals = z.to_vec2::<f32>()?;
    let expected = vec![vec![-0.25f32, 2.0], vec![3.0, -1.0]];
    assert_eq!(vals, expected);
    Ok(())
}
