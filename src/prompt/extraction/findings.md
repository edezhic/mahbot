Based on your research analysis above, provide your findings as a JSON object only:

{"claims": [{"claim": "<claim statement>", "source": "<primary source URL or file path>", "confidence": "low|medium|high", "contradictions": ["<contradicting evidence or alternative view>", ...]}], "unanswered": ["<aspect you could not answer>", ...]}

Rules:
- claims: the concrete load-bearing factual claims you can support — one entry per distinct claim, not every observation.
- source: the strongest source for the claim (URL, file path, or "analyst reasoning" for inferred claims). Required for every claim.
- confidence: low = uncertain/inferred; medium = multiple sources or strong secondary; high = primary/authoritative source.
- contradictions: explicitly list any evidence or views that contradict the claim; empty list when none.
- unanswered: what remains unanswered — be honest, do not fabricate answers.

Output ONLY the JSON object. Do NOT call any tools.