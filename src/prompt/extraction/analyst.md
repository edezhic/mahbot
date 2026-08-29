Based on your research analysis above, provide your verdict on the ticket's readiness for implementation as a JSON object only:

```json
{"issues": [{"text": "<issue 1>", "grade": "minor|major|blocker"}, ...]}
```

Where:
- issues: list of specific concerns about the ticket's assumptions, missing context, unclear requirements, or potential blockers (empty if none).
  - text: the concern, written as one specific sentence.
  - grade: the concern's severity — "minor" (a small gap / non-blocking note), "major" (a material problem), or "blocker" (prevents implementing the ticket as proposed).

Output ONLY the JSON object. Do NOT call any tools.
