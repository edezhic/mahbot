# Deduplicate test permission-setting blocks in self_update.rs

## Finding

Two `#[cfg(unix)]` test blocks in `src/self_update.rs` set file executable permissions on a source binary with identical 5-line code:

**Location 1: `test_copy_to_cargo_bin_success` (lines 1101–1105)**
```rust
#[cfg(unix)]
{
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&source, PermissionsExt::from_mode(0o755)).unwrap();
}
```

**Location 2: `test_resolve_spawn_path_copy_success` (lines 1249–1253)**
```rust
#[cfg(unix)]
{
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&source, PermissionsExt::from_mode(0o755)).unwrap();
}
```

## Proposal

Extract a shared helper function in the `#[cfg(test)] mod tests` block, placed right after `use super::*;` (line 866):

```rust
fn make_executable(_path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(_path, PermissionsExt::from_mode(0o755)).unwrap();
    }
}
```

Note: `#[cfg(unix)]` is placed **inside** the function body, not on the function itself. This ensures the function is always defined (compiles on all platforms) but is a no-op on non-Unix. The call sites therefore need no cfg guard at all — just `make_executable(&source);` unconditionally — exactly matching the behavior of the original code (where the `#[cfg(unix)]` block compiled away to nothing on non-Unix).

The parameter is prefixed `_path` (not `path`) to suppress an `unused_variables` warning on non-Unix platforms where the function body compiles away to nothing.

Replace both inline blocks with:
```rust
        make_executable(&source);
```

### Updated locations

**`test_copy_to_cargo_bin_success`** (lines 1093–1119): the 5-line inline block becomes a single function call.

**`test_resolve_spawn_path_copy_success`** (lines 1241–1266): same replacement.

## Scope note

A third `PermissionsExt::from_mode(0o755)` call exists in `test_is_executable_on_unix` (line 1054). That test function is deliberately **not** included in this refactor because:
- The function is already `#[cfg(unix)]`-gated (no extra `#[cfg(unix)]` block needed)
- It has a function-level `use std::os::unix::fs::PermissionsExt;` import shared with two other `set_permissions` calls using different modes (`0o644` and `0o100`)
- Applying `make_executable` there would save only one call site while leaving the function-level `use` import intact for the other modes, creating an inconsistent pattern
- The deduplication benefit is limited to the two standalone `#[cfg(unix)]` blocks

## Impact

- **Total LoC change**: net **+1 line** (git diff: 11 insertions, 10 deletions; 8 lines for helper + 1 new blank line separator between helper and first test, minus 5×2 = 10 from the two replaced blocks)
- **Readability**: Intent is clearer — "make this executable" vs "set permissions to 0o755"
- **Risk**: Trivial change, test-only code, no functional impact
- **Files affected**: `src/self_update.rs` only

## References

- `src/self_update.rs` lines 1101–1105: `#[cfg(unix)]` block in `test_copy_to_cargo_bin_success`
- `src/self_update.rs` lines 1249–1253: `#[cfg(unix)]` block in `test_resolve_spawn_path_copy_success`
- `src/self_update.rs` line 1038: function-level `use std::os::unix::fs::PermissionsExt` in `test_is_executable_on_unix` (excluded from refactor — see Scope note)