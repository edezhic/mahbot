//! JSON helper functions for extracting typed values from `serde_json::Value`.
//!
//! All functions operate on `&serde_json::Value` and a string key, providing
//! convenient access to commonly-needed extraction patterns used throughout
//! the codebase — particularly in tool argument parsing.

use serde_json::Value;

/// Extract a required string field from JSON args, returning an error if missing.
pub(crate) fn get_str<'a>(val: &'a Value, key: &str) -> anyhow::Result<&'a str> {
    val.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("Missing required field: {key}"))
}

/// Extract an optional string field from JSON args.
pub(crate) fn get_opt_str<'a>(val: &'a Value, key: &str) -> Option<&'a str> {
    val.get(key).and_then(Value::as_str)
}

/// Extract a boolean field with default value.
pub(crate) fn get_bool(val: &Value, key: &str, default: bool) -> bool {
    val.get(key).and_then(Value::as_bool).unwrap_or(default)
}

/// Extract an optional i64 field.
pub(crate) fn get_opt_i64(val: &Value, key: &str) -> Option<i64> {
    val.get(key).and_then(Value::as_i64)
}

/// Extract an optional u64 field.
pub(crate) fn get_opt_u64(val: &Value, key: &str) -> Option<u64> {
    val.get(key).and_then(Value::as_u64)
}

/// Extract a usize field with default value.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn get_usize(val: &Value, key: &str, default: usize) -> usize {
    val.get(key)
        .and_then(Value::as_u64)
        .map_or(default, |v| v as usize)
}

/// Extract a string array field as `Vec<String>`.
pub(crate) fn get_str_array(val: &Value, key: &str) -> Vec<String> {
    val.get(key)
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Extract an optional bool field.
pub(crate) fn get_opt_bool(val: &Value, key: &str) -> Option<bool> {
    val.get(key).and_then(Value::as_bool)
}
