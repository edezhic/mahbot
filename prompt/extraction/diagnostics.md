Based on your workspace analysis above, output the discovered dev tooling commands as a JSON object only.

```json
{"format": "command" | null, "format_check": "command" | null, "lint": "command" | null, "lint_fix": "command" | null, "type_check": "command" | null, "build": "command" | null, "unit_test": "command" | null}
```

Each field is a shell command string (the minimal invocation that works from the workspace root) or `null` if no such tooling exists.

For multi-language projects, each command must be a compound command chained with `&&` covering all languages.

Output ONLY the JSON object. Do NOT call any tools.
