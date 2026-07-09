Summarize the conversation so far into concise context for the REVIEWER role.

PRESERVE exactly:
- Changed-code context: files, symbols, and architectural boundaries touched
- Review findings by severity and why each matters
- Architectural or invariant concerns
- Missing tests or verification gaps
- Confirmed issues vs speculative concerns
- Final review posture (approved, changes requested, blocking)
- All identifiers (UUIDs, hashes, file paths, URLs, tokens, IPs)

OMIT:
- Verbose tool output (keep only key results)
- System prompt text or rule-file boilerplate
- Tool schemas, tool catalogs, or skill listings
- Runtime metadata such as current time, host, model, cwd

Be thorough. DO NOT USE ANY TOOLS. ONLY RESPOND WITH THE SUMMARY.
