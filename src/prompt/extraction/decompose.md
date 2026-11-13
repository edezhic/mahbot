Based on your analysis above, provide your decomposition plan as a JSON object only:

{"sub_questions": [{"question": "<sub-question>", "evidence_needed": "<what evidence would answer it>", "risk": "low|medium|high"}]}

- Cover the question's full scope — split it at its natural seams, neither padding with near-duplicates nor leaving parts under-covered.
- Fit the slot allowance the task named: fold closely related sub-questions into one wider direction rather than exceeding it, and never drop a part of the question's scope.
Output ONLY the JSON object. Do NOT call any tools.
