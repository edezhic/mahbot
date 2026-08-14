

=== REPAIR ROUND {{round}} ===
Only the latest REPAIR ROUND section applies — earlier sections are historical and superseded.
{{framing}}
Accepted groups are FROZEN and must NEVER be re-proposed; repair ONLY the remaining items. Reference every member by its item id from the input list — never by text.
{{rejections_section}}
Frozen groups (skeletons — do NOT re-propose their members; references to a group flagged contradiction:true are ignored):
{{frozen_groups_lines}}
Remaining items (place EVERY un-pinned one in a proposed group or the ungrouped list; pinned items must NOT be placed in a group). Reference each item by its id:
{{remainder_lines}}
Respond with ONLY a JSON object matching this REPAIR-DELTA schema (no extra fields):
{
  "summary": "optional — ONLY if no summary was accepted yet",
  "groups": [
    {
      "heading": "short thematic heading",
      "contradiction": false,
      "members": [
        {"id": 0}
      ]
    }
  ],
  "ungrouped": [
    {"id": 2}
  ],
  "references": [
    {"group": 0, "member": {"id": 2}}
  ]
}

Note: every item listed in `references` must ALSO appear in `groups` or `ungrouped` — a reference alone does not place the item.
