Distill the conversation to its essential context while preserving:
- The user's original question or request
- Key findings from any research (web search or analyst delegation)
- Async sub-agent job ids: the `(job <id>)` in dispatch acknowledgements and the `Job: <id>` first line of delivered result envelopes (verbatim)
- Important conclusions or decisions
- Any follow-up questions or requests from the user
- Media work: original uploads and reference images (paths, markers), generation attempts (prompts used, tool calls, output paths `IMAGE:path` / `VIDEO:path`), user feedback on each iteration, and the rule of a single generation attempt before user review

Maintain a natural question-and-answer flow. Omit any tool output, error messages, or intermediate reasoning that is no longer relevant.
