<available-custom-roles>
Custom roles the administrator has defined — each with its behaviour (how it is addressed when no identifier is given: `continuous` continues one common branch, `one_off` runs fresh each time), the tools it may use, and the opening of the role's own prompt:
{{roles}}
Point-in-time snapshot taken at session start, refreshed only on compaction; `mahbot_config`'s `list_custom_roles` is the source of truth for the current set.
</available-custom-roles>