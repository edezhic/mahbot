Based on your analysis above, provide your link confirmations as a JSON object only:

{"links": [{"new_id": 0, "verdict": "confirm|reject"}]}

Rules:
- new_id: 0-based index of the new claim within the New Claims list, exactly as in the annotation pass. Every link above must be judged exactly once.
- confirm = the duplicate/contradiction relation is real.
- reject = the relation is uncertain or wrong — the link becomes weak/unconfirmed, never merged or dropped.
- When uncertain, output reject.

Output ONLY the JSON object. Do NOT call any tools.