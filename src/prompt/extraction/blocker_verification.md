Based on your verification above, provide your outcomes as a JSON object only:

```json
{"verdicts": [{"index": 0, "kind": "main_path_blocker", "severity": "high", "impact": "<what it affects / how it constrains the ticket>", "reasoning": "<evidence-backed justification>"}, ...]}
```

Where:
- index: the 0-based index of the blocker in the injected list.
- kind: one of "main_path_blocker" (blocks the main implementation path) or "risk_edge_case" (a real but non-blocking risk/edge case).
- severity: one of "low", "medium", "high", "critical".
- impact: required, non-empty; what the blocker affects / how it constrains the proposed ticket.
- reasoning: required, non-empty; evidence-backed justification.

You must provide exactly one entry per blocker in the injected list.

Output ONLY the JSON object. Do NOT call any tools.
