Correctness angle: focus your review on correctness — logic errors, edge cases and boundary conditions, invalid inputs, concurrency, and regressions. Trace the changed code paths for these failure modes while keeping the full review bar.

---

Maintainability angle: focus your review on maintainability — clarity, naming, duplication, and adherence to the existing codebase patterns and conventions. Judge how well the change fits the codebase while keeping the full review bar.

---

Robustness angle: focus your review on robustness — error paths, resource leaks, panics, auth and injection concerns, and retry/backoff behavior. Probe the failure paths while keeping the full review bar.

---

Integration angle: focus your review on integration — contract compatibility, caller impact, user-visible behavior, configuration, and migration or rollout concerns. Verify the change against its callers and surrounding systems while keeping the full review bar.
