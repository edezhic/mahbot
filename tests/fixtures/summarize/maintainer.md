Summarize the conversation so far into concise context for the MAINTAINER role.

PRESERVE exactly:
- Cleanup and refactoring opportunities discovered
- Evidence supporting each opportunity (duplication, dead code, drift, complexity)
- Affected files, modules, and safe refactor boundaries
- Safety and value rationale for each suggestion
- Rejected ideas and why they were skipped
- Constraints: no macros, no module-directory splits, LoC impact considered
- Tickets created (IDs, titles) and areas not yet investigated
- Sub-agent (`ask`) findings from deeper investigations
- All identifiers (UUIDs, hashes, file paths, URLs, tokens, IPs)

OMIT:
- Verbose tool output (keep only key results)
- System prompt text or rule-file boilerplate
- Tool schemas, tool catalogs, or skill listings
- Runtime metadata such as current time, host, model, cwd

Be thorough. DO NOT USE ANY TOOLS. ONLY RESPOND WITH THE SUMMARY.
