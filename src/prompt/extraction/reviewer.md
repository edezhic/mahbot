Based on the analysis above, provide your code review verdict as a JSON object only:

```json
{"score": <1-10>, "issues": ["<issue 1>", ...]}
```

Where:
- score: 1-10 rating
- issues: list of specific problems (empty if none)

Do not turn a point the ticket's scope already settles — whether the scope accepts it as it is or excludes it explicitly — into an issue, and do not let it lower the score, even if the answer mentions it. Anything the scope does not settle is still an issue as usual.

Output ONLY the JSON object. Do NOT call any tools.
