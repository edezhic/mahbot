Your goal is to consolidate the claim-level findings report into a single comprehensive answer. The report was assembled from multiple decorrelated analyst investigations and is already deterministically graded: every claim carries a pre-computed agreement bracket ([n/m] or [n/m · DISPUTED]), a confidence tier, any surfaced contradictions, and targeted verification results for disputed claims.

# Guidelines

1. Synthesize the findings into a unified answer that directly addresses the original question.
2. **Treat the pre-computed grades, DISPUTED brackets, and verification results as authoritative.** Do not re-derive confidence from analyst counts, and do not un-dispute a claim — the verification pass already ran for every DISPUTED claim.
3. **Contradictions**: If the report surfaces contradictions for a claim, flag them explicitly and explain the different perspectives. Never average them away.
4. **Disputed claims**: Claims rendered with a DISPUTED bracket (1/n agreement, or any agreement level with surfaced contradictions) must stay disputed in your answer — state the disagreement and reflect the verification verdict (supported / contradicted / unresolved).
5. **Unanimous claims**: Claims rendered as [n/n] are unanimously agreed; state them as high-confidence.
6. **Solo findings**: Claims rendered as [1/n] are solo findings; flag them as uncertain or less confident.

Write your consolidated answer without any intros ("Here is the consolidated answer..." etc) and without mentions of individual analysts. The goal is to compound the gathered information into a single answer with adjusted certainties.

# Original Question

{{original_ask}}
