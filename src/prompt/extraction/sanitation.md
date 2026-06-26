Based on your inspection of the untracked files, provide your sanitation verdict as a JSON object only:

```json
{"pass": true/false, "garbage_files": ["path/to/garbage1", ...], "rationale": "<explanation>"}
```

Where:
- pass: `true` if all untracked files are legitimate project files (no garbage), `false` if any garbage files were detected
- garbage_files: list of file paths that are garbage artifacts (empty if `pass` is `true`)
- rationale: brief explanation of your decision, mentioning key files inspected and garbage indicators found

Output ONLY the JSON object. Do NOT call any tools.
