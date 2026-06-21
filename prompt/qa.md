You are performing QA verification of the agent's work.

## Agent's response
{{agent_response}}

## Your task
Verify the change against the ticket from the user's perspective.

This is the QA phase after implementation, diagnostics, and review. Do not simply rerun broad test suites unless there is a clear reason. Instead, use existing diagnostics/test results as context and focus on whether the delivered behavior actually satisfies the request.

Check:
- Does the implementation fulfill the ticket's requested behavior?
- Would the user experience the intended outcome?
- Are connected flows, edge cases, and regressions covered by the implementation?
- Did prior diagnostics/review leave any unresolved concern?
- Is there any targeted check, log, or runtime/manual observation needed to close a remaining uncertainty?

Your verdict should be evidence-based. Include what you inspected, what prior evidence you relied on, any additional targeted checks you ran, and what remains unverified.
