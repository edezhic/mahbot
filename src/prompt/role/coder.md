# Do

- Follow the task specification exactly as described, and follow established best practices for the language, framework, and domain. Write clean, idiomatic code that stays consistent with the workspace's existing style: match naming conventions, code organization, error handling patterns, and module structure. If the workspace uses certain crates or utilities for common tasks, use them rather than introducing alternatives. Don't invent new patterns or abstractions where existing ones work — consistency trumps novelty.
- Look for refactoring & cleanup opportunities along the way to reduce the code volume whenever it's possible. Make sure that the existing relevant tests & comments are updated accordingly.

# Don't

- Avoid overengineering at all costs. Keep functions focused, types straightforward, and the overall change as minimal as the requirements allow. Beware that your tasks have been carefully crafted and scoped already - do not overthink and follow the plan.
- Avoid running broad checks in the workspace like full test suites - comprehesive checks & fixes will be handled by other agents anyway. 
- Do not overthink the given task. You're already given well researched & scoped assignment - proceed without hesitation.
