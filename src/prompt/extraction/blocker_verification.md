Based on your verification above, provide your verdicts as a JSON object only:

```json
{"verdicts": [{"index": 0, "verdict": "confirmed", "reasoning": "<evidence-backed justification>", "sharpened_text": "<only when verdict is sharpened>"}, ...]}
```

Where:
- index: the 0-based index of the blocker in the injected list.
- verdict: one of "confirmed", "refuted", "sharpened".
- reasoning: concise evidence-backed justification (required).
- sharpened_text: required only when verdict == "sharpened"; the precise rewritten blocker.

You must provide exactly one entry per blocker in the injected list.

Output ONLY the JSON object. Do NOT call any tools.
