- Agreement counts and the DISPUTED marker are rendered automatically by the system next to
  the member text — never state them in the summary or in group headings. Member text is a
  verbatim copy and may contain numbers (property values, line numbers, ranges) — copy them
  exactly, never strip or alter them.
- Set "contradiction": true on a group ONLY when agents genuinely disagree about the
  same property. Genuine disagreements include, but are not limited to:
  * negation and polarity flips: one agent affirms a claim and another denies it or
    states the opposite (e.g. "safe" vs "not safe", "improves" vs "worsens",
    "succeeds" vs "does not succeed") — even when the rest of the sentence is identical;
  * property-number disagreements: agents give different values for the same measured
    property (e.g. "6-19x" vs "12-19x" speedup, "retry count 3" vs "retry count 7").
  When agents disagree, put BOTH sides' verbatim issues in the SAME group and mark
  contradiction: true — never merge them silently into a single agreed claim.
- Do NOT set "contradiction": true when the issues are the same claim with different
  locators or evidence details (line numbers, file paths, commit hashes, code snippets,
  test names). Those are the same issue, not a disagreement.