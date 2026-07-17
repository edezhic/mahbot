# Resolution: Ticket Priority Field — mahbot-717

**Status:** Resolved  
**Date:** 2026-07-17  
**Author:** Egor Dezic  

---

This document records the design decisions for adding a `priority` field to tickets. Each issue is presented with the agreed decision and rationale.

---

## 1. Supersede Priority Semantics

**Decision:** Do **not** inherit priority on supersede. Default to `1` (the column default `DEFAULT 1`). The Manager must explicitly set `priority` when calling supersede. This is implemented by having `params.priority` default to `1` in `build_params`; the supersede path uses that value without any override or inheritance logic.

**Rationale:** Consistency with other fields — title, description, and other metadata are not inherited during supersede either. Keeping the default at `1` avoids surprises where a superseded ticket accidentally inherits a low-priority (high-number) value from its parent.

---

## 2. Agent-Facing Display: `short_display()` vs `detailed_display()`

**Decision:**

- **`short_display()`** — do **not** include priority. The format remains intentionally compact: `"  [reporter] [phase] id: title"`. Adding priority would bloat every line with no clear parsing benefit at a glance.
- **`detailed_display()`** — **do** include priority. Add the line `Priority: P{i64}\n` after the `Updated: ...` line. This gives the Manager full visibility when calling `get_ticket`.

Update doc comments for both methods to reflect this behaviour.

**Rationale:** The short display is used in ticket lists and context windows where real estate is scarce. The detailed display is the canonical "inspect a ticket" view and should surface all relevant fields.

---

## 3. Archived Search Display

**Decision:** Do **not** include priority in archived search results. The format `"  [phase] id: title"` stays compact.

**Rationale:** Same as short display — archives are browsed for quick scanning. The Manager can use `get_ticket` for full details on any archived ticket of interest.

---

## 4. Sidebar Sort Order (`list_all_tickets`)

**Decision:** Change the `list_all_tickets` query from:

```sql
ORDER BY created_at DESC
```

to:

```sql
ORDER BY priority ASC, created_at DESC
```

Higher priority (lower number) appears first; within the same priority level, newest tickets appear first.

**Rationale:** The sidebar is the primary navigation for active tickets. Sorting by priority first ensures urgent work is always visible at the top. Creation time remains the secondary sort for chronological ordering within each tier.

---

## 5. Maintainer Exclusion Mechanism

**Decision:** Add a `priority_visible: bool` field to `CreateTicketTool`. Set to `true` for the Manager instance, `false` for the Maintainer instance. The `parameters_schema()` method checks this flag to include or exclude the `priority` parameter entirely from the JSON schema — no separate struct required.

**Rationale:** The schema-level approach is the cleanest way to control tool visibility per role. It avoids branching in execute logic, keeps the code DRY (single struct), and prevents the Maintainer from even being offered the parameter by the LLM.

---

## 6. Maintainer Hardcoded Priority of 3

**Decision:** Apply the hardcoded value of `3` to **all** `execute()` paths when `self.reporter == "maintainer"` — both the regular create path and the supersede path. The value is applied in `build_params` (or directly in `execute`), ensuring the Maintainer can never accidentally create urgent tickets.

**Rationale:** Maintainer-created tickets are always non-critical (bug reports, feature suggestions). Hardcoding priority `3` (low urgency) prevents the Maintainer from bypassing the Manager's triage process. Applying it to supersede as well is defensive — a supersede path should not escalate priority.

---

## 7. Prompt File (`create_ticket.md`)

**Decision:** Add `priority` parameter documentation to the embedded prompt file. The documentation should state:

- Priority is an integer where lower numbers = higher urgency.
- Default is `1`.
- The Maintainer's tickets are always created at priority `3`.

**Note:** The schema (`parameters_schema()`) controls per-role visibility — the prompt file is documentation only and is shared across roles.

**Rationale:** The prompt file informs the LLM about the parameter's semantics. Schema-level visibility ensures the parameter is only offered when appropriate, but the prompt still explains the concept for the Manager's context.

---

## 8. Rust Type

**Decision:** Use `i64` for the priority field to match existing conventions (`lines_added`, `lines_removed` are both `i64`).

**Rationale:** Consistency with the rest of the codebase. `i64` is the standard integer type used for database-backed numeric fields across the project.

---

## 9. Chip Colors

**Decision:** Use Flexoki-compatible muted colours derived from the existing `ticket_phase_color` palette:

| Priority | Background (RGB) | Text (RGB) | Analogous Phase |
|----------|------------------|------------|-----------------|
| P0       | (0.380, 0.114, 0.114) | (0.878, 0.753, 0.753) | Failed |
| P1       | (0.380, 0.216, 0.078) | (0.941, 0.878, 0.784) | InDevelopment |
| P2       | (0.310, 0.224, 0.102) | (0.902, 0.863, 0.784) | InDiagnostics |
| P3       | (0.176, 0.310, 0.208) | (0.784, 0.902, 0.816) | QaPassed |
| P4+      | (0.114, 0.176, 0.114) | (0.753, 0.816, 0.753) | Done |

Any priority ≥ 4 uses the P4+ (green) scheme. Values < 0 are clamped to P0.

**Rationale:** Muted colours reduce visual noise in the GUI. Mapping to existing phase colours provides a familiar visual language — red=urgent, green=done/low-urgency.

---

## 10. Schema Migration

**Decision:** Bump `PRAGMA user_version` to `2`. At startup, check if the `priority` column exists via `PRAGMA table_info(tickets)`; if absent, add it with:

```sql
ALTER TABLE tickets ADD COLUMN priority INTEGER NOT NULL DEFAULT 1;
```

Follow the exact same pattern as the existing version 1 migration (e.g., `PRAGMA user_version = 1` for the initial schema).

**Rationale:** The existing version 1 migration pattern is proven and handles idempotency (check-then-migrate). Using the column default `1` ensures existing records without the column are retroactively assigned the standard priority.

---

## 11. `format_ticket_block` (System Prompt — Current Ticket)

**Decision:** Add the priority line to `format_ticket_block`. The block is part of the system prompt presented to the agent (Engineer, etc.) showing their currently assigned ticket. The priority value should be displayed so the agent is aware of the urgency.

**Rationale:** The current-ticket block is the agent's primary context for their assigned work. Showing priority ensures the agent (and by extension any downstream tooling) understands the urgency of the ticket.

---

## 12. Dispatch Query (`claim_ticket_in_workspace`)

**Decision:** Change the subquery's `ORDER BY` from:

```sql
ORDER BY t1.pipeline_reservation DESC, t1.created_at ASC
```

to:

```sql
ORDER BY t1.pipeline_reservation DESC, t1.priority ASC, t1.created_at ASC
```

This ensures that within the same reservation tier, tickets with higher priority (lower number) are claimed first.

**Rationale:** The dispatch system selects the next ticket for an agent to work on. Priority should be the primary ordering factor after reservation status — higher-urgency tickets should be picked before lower-urgency ones, regardless of age.

---

## Summary of Changes

| Area | Change |
|------|--------|
| **DB schema** | Add `priority INTEGER NOT NULL DEFAULT 1`, bump `PRAGMA user_version` to 2 |
| **Rust type** | `Ticket { priority: i64 }`, default `1` |
| **Supersede** | No inheritance; defaults to `1` |
| **`short_display()`** | Unchanged (no priority) |
| **`detailed_display()`** | Add `Priority: P{n}\n` after `Updated:` |
| **Archived display** | Unchanged (no priority) |
| **Sidebar sort** | `ORDER BY priority ASC, created_at DESC` |
| **Dispatch sort** | `ORDER BY pipeline_reservation DESC, priority ASC, created_at ASC` |
| **CreateTicketTool** | `priority_visible: bool` controls schema inclusion |
| **Maintainer override** | Hardcode `priority = 3` in `build_params` / `execute` |
| **Prompt file** | Document priority semantics |
| **GUI chips** | Flexoki colours per priority tier |
| **`format_ticket_block`** | Add priority line |
