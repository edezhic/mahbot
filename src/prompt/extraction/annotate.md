Based on your analysis above, provide your claim annotations as a JSON object only:

{"annotations": [{"new_id": 0, "verdict": "novel|duplicate|contradicts", "existing_id": 0, "contradiction": "<contradiction note>"}]}

Rules:
- new_id: 0-based index of the new claim within the New Claims list. Every new claim must be annotated exactly once.
- novel = a new fact not restated by any existing claim. existing_id and contradiction are omitted.
- duplicate = the same fact restated by an existing claim. existing_id is required.
- contradicts = directly contradicts an existing claim. existing_id is required, plus a contradiction note stating how the new claim conflicts.
- The contradiction note is present exactly when verdict is "contradicts".

Output ONLY the JSON object. Do NOT call any tools.