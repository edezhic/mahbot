Based on your workspace analysis above, output the discovered dev tooling commands as a JSON object only.

```json
{"format": "command" | null, "format_check": "command" | null, "lint": "command" | null, "lint_fix": "command" | null, "type_check": "command" | null, "build": "command" | null, "unit_test": "command" | null}
```

Each field is a shell command string (the minimal invocation that works from the workspace root) or `null` if no such tooling exists.

For multi-language projects, each command must be a compound command chained with `&&` covering all languages.

Lint commands must keep the auto-fix pass and the lint gate in SEPARATE fields: `lint_fix` is a fix-only command (e.g. `cargo clippy --fix --allow-dirty`) and `lint` is a gate-only command (e.g. `cargo clippy -- -D warnings`). NEVER combine `--fix` with a `-D`/`--deny` gate in a single invocation — clippy's fix driver silently disables the lint gate in that form (rust-clippy#17444) and exits 0 with unfixable warnings remaining.

Output ONLY the JSON object. Do NOT call any tools.
