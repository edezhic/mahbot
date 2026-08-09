//! Shared LLM grouping/synthesis core for pipeline joint-verdict and ask
//! consolidation.
//!
//! One LLM pass groups id-referenced items (pipeline issues; ask claims)
//! into thematic/semantic groups. Every item gets a stable numeric id,
//! assigned once per operation across the whole input (global flat numbering);
//! the LLM refers to items by id only, and the system resolves id → (agent,
//! text) via [`ItemTable`] for rendering. The LLM never produces counts:
//! brackets `[n/N]` and the DISPUTED marker are computed here from distinct
//! cited agent ids — pure arithmetic.
//!
//! Validation is strictly structural: ids in range, no duplicate placement,
//! set completeness (every item appears exactly once across placed ∪ pinned ∪
//! remainder), and a contradiction guardrail (contradiction:true requires ≥2
//! distinct cited agents). Text-based membership checks are gone everywhere —
//! a wrong-but-valid id is accepted by design (semantic judgment is the LLM's;
//! the gate only catches structural lies).
//!
//! The single retry mode is the progress-preserving repair
//! ([`run_grouping_repair`]): accepted groups freeze, repair rounds propose
//! deltas for the remainder only, and termination deterministically places
//! every remaining item in the ungrouped section.

use serde::Deserialize;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::time::Instant;

use crate::retry::{FailureClass, RetryFailureRecord, RetryLoop, RetryPolicy};
use crate::{ChatRequest, ChatRequestMeta, Workspace};

// ── Schema (strict — unknown fields rejected) ──────────────────────────

/// One member of a group: a flat global item id (matches the `{id}: {text}`
/// input list numbering). The source (agent) and text are resolved by the
/// system via the operation's [`ItemTable`] — never by the model.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupingMember {
    /// Flat global item id, 0-based across all sources combined.
    pub id: usize,
}

/// A group of related items plus the LLM's contradiction judgment.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupingGroup {
    pub heading: String,
    /// Set by the LLM only for genuine contradictions between agents.
    #[serde(default)]
    pub contradiction: bool,
    pub members: Vec<GroupingMember>,
}

/// The full grouping output: summary prose + groups + an explicit ungrouped
/// list (a valid, visible destination — silent dropout is rejected).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupingOutput {
    pub summary: String,
    pub groups: Vec<GroupingGroup>,
    #[serde(default)]
    pub ungrouped: Vec<GroupingMember>,
}

/// A repair-round proposal: only new groups for REMAINING items, an explicit
/// ungrouped list (satisfies the round's completeness declaration — never
/// final), optional contradiction references from remainder items to frozen
/// groups, and an optional summary (accepted only when none was yet).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupingDelta {
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub groups: Vec<GroupingGroup>,
    #[serde(default)]
    pub ungrouped: Vec<GroupingMember>,
    #[serde(default)]
    pub references: Vec<GroupingReference>,
}

/// A remainder item that contradicts a frozen group: renders with DISPUTED
/// and a code-computed cross-reference to that group — the frozen group
/// itself never changes. The target group index is stable because frozen
/// groups are append-only.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupingReference {
    /// Index into the frozen-groups list (0-based, matches the prompt
    /// skeletons).
    pub group: usize,
    /// The remainder item flagging the contradiction.
    pub member: GroupingMember,
}

/// Outcome of the repair-mode grouping pass (pipeline synthesis + ask
/// consolidation).
pub(crate) enum RepairOutcome {
    /// Frozen groups + first-accepted summary + a code-computed ungrouped
    /// remainder. `references` attach DISPUTED cross-references from
    /// remainder items to frozen groups (code-side dedupe already applied).
    Repaired {
        output: GroupingOutput,
        references: Vec<GroupingReference>,
    },
    Fallback,
}

// ── Item id table (global flat numbering) ──────────────────────────────

/// Global flat id → (agent, item index) table for one grouping operation.
/// Ids are assigned once per operation across the whole input: every item
/// across all sources gets exactly one id, in (agent, item) order. The model
/// refers to items by id only; the system resolves id → (agent, text) here
/// for rendering and severity lookups.
///
/// A wrong-but-valid id is accepted by design: validation is structural only
/// (range, duplicates, completeness, contradiction ≥2 agents), and the LLM's
/// semantic judgment is the only correctness gate. Never re-add text-equality
/// membership checks — they only create a false sense of safety.
pub(crate) struct ItemTable<'a> {
    /// id → (agent index, item index within that agent's list).
    rows: Vec<(usize, usize)>,
    items_by_agent: &'a [Vec<String>],
}

impl<'a> ItemTable<'a> {
    #[must_use]
    pub(crate) fn new(items_by_agent: &'a [Vec<String>]) -> Self {
        let mut rows = Vec::new();
        for (agent, items) in items_by_agent.iter().enumerate() {
            for i in 0..items.len() {
                rows.push((agent, i));
            }
        }
        Self {
            rows,
            items_by_agent,
        }
    }

    /// Total item count across all sources.
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    /// Resolve an id to its (agent, text) — `None` when out of range.
    #[must_use]
    pub(crate) fn resolve(&self, id: usize) -> Option<(usize, &str)> {
        let (agent, item) = *self.rows.get(id)?;
        let text = self.items_by_agent.get(agent)?.get(item)?;
        Some((agent, text))
    }

    /// Resolve an id to its (agent, per-agent item index) — `None` when out
    /// of range.
    #[must_use]
    pub(crate) fn resolve_index(&self, id: usize) -> Option<(usize, usize)> {
        self.rows.get(id).copied()
    }

    /// Agent index of an id — `None` when out of range.
    #[must_use]
    pub(crate) fn agent(&self, id: usize) -> Option<usize> {
        self.rows.get(id).map(|r| r.0)
    }
}

// ── Bracket arithmetic (pure code — never LLM-written counts) ───────────

/// Distinct cited agent ids of a group, sorted (resolved via the item table;
/// out-of-range ids are ignored — validation rejects them before rendering).
#[must_use]
pub(crate) fn distinct_agents(group: &GroupingGroup, table: &ItemTable<'_>) -> Vec<usize> {
    let mut agents: Vec<usize> = group
        .members
        .iter()
        .filter_map(|m| table.agent(m.id))
        .collect();
    agents.sort_unstable();
    agents.dedup();
    agents
}

/// Render the bracket for a group: `[n/N]`, with ` · DISPUTED` appended only
/// when the group carries a genuine contradiction (contradiction:true — which
/// validation guarantees implies ≥2 distinct cited agents). Solo `[1/N]`
/// renders without DISPUTED.
#[must_use]
pub(crate) fn bracket_label(n: usize, n_valid: usize, disputed: bool) -> String {
    if disputed {
        format!("[{n}/{n_valid} · DISPUTED]")
    } else {
        format!("[{n}/{n_valid}]")
    }
}

// ── Rendering helpers (shared) ─────────────────────────────────────────

/// Render one member line attributed to its source: `Agent N: text`
/// (resolved via the item table).
#[must_use]
pub(crate) fn render_member_line(member: &GroupingMember, table: &ItemTable<'_>) -> String {
    match table.resolve(member.id) {
        Some((agent, text)) => format!("Agent {agent}: {text}"),
        None => format!("Agent ?: <unknown item id {}>", member.id),
    }
}

/// Render `Agent N:\n- id: text` input material for the grouping pass —
/// global flat ids across all agents (id = (agent, item) order), matching the
/// schema's `id` field. Empty agent slots are skipped.
#[must_use]
pub(crate) fn numbered_items_material(items_by_agent: &[Vec<String>]) -> String {
    let mut material = String::new();
    let mut id = 0usize;
    for (agent_idx, items) in items_by_agent.iter().enumerate() {
        if items.is_empty() {
            continue;
        }
        let _ = writeln!(material, "Agent {agent_idx}:");
        for item in items {
            let _ = writeln!(material, "- {id}: {item}");
            id += 1;
        }
    }
    material
}

// ── Repair-mode grouping (pipeline synthesis + ask consolidation) ───────

/// Mutable state of the repair protocol across rounds.
pub(crate) struct RepairState<'a> {
    /// Global flat id table for this operation (id → agent, text).
    table: ItemTable<'a>,
    /// Accepted (frozen) groups in order — indices are stable references.
    frozen_groups: Vec<GroupingGroup>,
    /// Ids of frozen-group members — placement bookkeeping.
    placed: HashSet<usize>,
    /// Accepted contradiction references (remainder item id → frozen group idx).
    references: Vec<GroupingReference>,
    /// Ids of items pinned to the remainder by an accepted reference — they
    /// can never be placed in a group, so the DISPUTED render stays visible
    /// and `accepted_refs` never goes stale.
    pinned: HashSet<usize>,
    /// Frozen groups already flagged contradiction — do-not-re-flag targets.
    flagged: HashSet<usize>,
    /// First accepted summary (never revised).
    summary: Option<String>,
}

impl<'a> RepairState<'a> {
    pub(crate) fn new(items_by_agent: &'a [Vec<String>]) -> Self {
        Self {
            table: ItemTable::new(items_by_agent),
            frozen_groups: Vec::new(),
            placed: HashSet::new(),
            references: Vec::new(),
            pinned: HashSet::new(),
            flagged: HashSet::new(),
            summary: None,
        }
    }

    /// Deterministic remainder: every non-placed item id in global order.
    #[must_use]
    pub(crate) fn remainder(&self) -> Vec<usize> {
        (0..self.table.len())
            .filter(|id| !self.placed.contains(id))
            .collect()
    }

    /// True when every non-placed item is pinned by an accepted contradiction
    /// reference — nothing left to place (pinned items render in the ungrouped
    /// section), so the run terminates without a redundant empty round.
    #[must_use]
    pub(crate) fn complete(&self) -> bool {
        (0..self.table.len()).all(|id| self.placed.contains(&id) || self.pinned.contains(&id))
    }
}

/// One round's proposal, parsed from the full schema (round 1) or the
/// repair-delta schema (repair rounds).
pub(crate) struct RoundInput {
    pub summary: Option<String>,
    pub groups: Vec<GroupingGroup>,
    pub ungrouped: Vec<GroupingMember>,
    pub references: Vec<GroupingReference>,
}

/// Per-round acceptance result.
#[derive(Default)]
pub(crate) struct RoundOutcome {
    /// Round-level rejection (incomplete coverage) — nothing froze this round.
    pub round_rejection: Option<String>,
    /// Per-group/per-entry rejections fed to the next round, each with its
    /// typed cause (no substring classification of message wording).
    pub rejections: Vec<(FailureClass, String)>,
    /// Newly frozen groups.
    pub froze: usize,
    /// Newly accepted contradiction references.
    pub accepted_refs: usize,
    /// A summary was accepted this round.
    pub accepted_summary: bool,
}

/// Id-range check for one member (no placement check). Text membership is
/// deliberately NOT checked — ids are the only references (see [`ItemTable`]).
fn member_range_error(member: &GroupingMember, table: &ItemTable<'_>) -> Option<String> {
    (member.id >= table.len()).then(|| format!("unknown item id {}", member.id))
}

/// Push a typed rejection onto the outcome.
fn reject(outcome: &mut RoundOutcome, cause: FailureClass, msg: String) {
    outcome.rejections.push((cause, msg));
}

/// Process one round's proposal: global checks first (completeness on the raw
/// proposal — silent dropout rejects the round), then the summary (first
/// accepted wins), then per-group acceptance (id range, no already-placed
/// member), then ungrouped entries and contradiction references. Rejected
/// groups are dropped with per-group reasons; accepted groups freeze
/// immediately. Validation is strictly structural — ids in range, no
/// duplicate placement, set completeness, contradiction ≥2 distinct agents
/// (see [`ItemTable`] for the accepted wrong-but-valid-id risk).
#[allow(clippy::too_many_lines)]
pub(crate) fn process_round(input: RoundInput, state: &mut RepairState<'_>) -> RoundOutcome {
    let RoundInput {
        summary,
        groups,
        ungrouped,
        references,
    } = input;
    let mut outcome = RoundOutcome::default();

    // Round-level completeness on the RAW proposal: every unfrozen item id
    // must appear in a proposed group or the ungrouped list.
    let mut raw: HashSet<usize> = HashSet::new();
    for g in &groups {
        raw.extend(g.members.iter().map(|m| m.id));
    }
    raw.extend(ungrouped.iter().map(|m| m.id));
    for id in 0..state.table.len() {
        // Placed (frozen-group) and pinned (accepted reference → committed
        // to the remainder) items need not be re-proposed this round.
        if state.placed.contains(&id) || state.pinned.contains(&id) {
            continue;
        }
        if !raw.contains(&id) {
            outcome.round_rejection = Some(format!(
                "item {id} missing from every proposed group and the ungrouped list"
            ));
            return outcome;
        }
    }

    // Summary: first accepted is final; repair rounds may supply one only when
    // none was accepted yet.
    if state.summary.is_none() {
        if let Some(s) = summary.as_deref().filter(|s| !s.trim().is_empty()) {
            state.summary = Some(s.trim().to_string());
            outcome.accepted_summary = true;
        } else {
            reject(
                &mut outcome,
                FailureClass::ValidationOther,
                "summary is empty or missing — supply a non-empty summary".to_string(),
            );
        }
    }

    // Per-group acceptance (in order; first placement wins within the round).
    // The telemetry cause is first-wins: the first detected defect determines
    // the class, so a multi-defect group never loses its primary cause.
    for (i, group) in groups.into_iter().enumerate() {
        let mut reasons: Vec<String> = Vec::new();
        let mut cause: Option<FailureClass> = None;
        if group.members.is_empty() {
            reasons.push("group has no members".to_string());
            cause.get_or_insert(FailureClass::Membership);
        }
        if group.heading.trim().is_empty() {
            reasons.push("empty heading".to_string());
            cause.get_or_insert(FailureClass::ValidationOther);
        }
        if group.contradiction && distinct_agents(&group, &state.table).len() < 2 {
            reasons.push("contradiction without ≥2 distinct cited agents".to_string());
            cause.get_or_insert(FailureClass::ContradictionAgents);
        }
        // Intra-group duplicate ids are rejected (a member may be placed only
        // once per round; already-placed/pinned members are rejected below).
        let mut seen: HashSet<usize> = HashSet::new();
        for member in &group.members {
            if !seen.insert(member.id) {
                reasons.push("duplicate member within the group".to_string());
            } else if state.pinned.contains(&member.id) {
                reasons.push(
                    "member has an accepted contradiction reference and must remain ungrouped"
                        .to_string(),
                );
            } else if let Some(msg) = member_range_error(member, &state.table) {
                reasons.push(msg);
            } else if state.placed.contains(&member.id) {
                reasons.push("member already placed in a frozen group".to_string());
            }
            cause.get_or_insert(FailureClass::Membership);
        }
        if reasons.is_empty() {
            for member in &group.members {
                state.placed.insert(member.id);
            }
            if group.contradiction {
                state.flagged.insert(state.frozen_groups.len());
            }
            state.frozen_groups.push(group);
            outcome.froze += 1;
        } else {
            reject(
                &mut outcome,
                // Every reason-pushing branch above sets a cause, so this is
                // never None.
                cause.unwrap(),
                format!("group {i}: {}", reasons.join("; ")),
            );
        }
    }

    // Ungrouped entries: per-member id-range validation; invalid entries are
    // dropped with reasons (the raw proposal already satisfied completeness).
    // Already-placed entries are dropped silently — the placement wins.
    for (i, member) in ungrouped.into_iter().enumerate() {
        if let Some(msg) = member_range_error(&member, &state.table) {
            reject(
                &mut outcome,
                FailureClass::Membership,
                format!("ungrouped entry {i}: {msg}"),
            );
        }
    }

    // Contradiction references to frozen groups.
    for (i, reference) in references.into_iter().enumerate() {
        if reference.group >= state.frozen_groups.len() {
            reject(
                &mut outcome,
                FailureClass::ValidationOther,
                format!(
                    "reference {i}: unknown frozen group index {}",
                    reference.group
                ),
            );
            continue;
        }
        if state.flagged.contains(&reference.group) {
            continue; // cosmetic duplicate — never a rejection
        }
        if state
            .references
            .iter()
            .any(|r| r.group == reference.group && r.member.id == reference.member.id)
        {
            continue; // already accepted — cosmetic duplicate
        }
        if let Some(msg) = member_range_error(&reference.member, &state.table) {
            reject(
                &mut outcome,
                FailureClass::Membership,
                format!("reference {i}: {msg}"),
            );
            continue;
        }
        if state.placed.contains(&reference.member.id) {
            reject(
                &mut outcome,
                FailureClass::Membership,
                format!("reference {i}: member already placed in a frozen group"),
            );
            continue;
        }
        let mut agents = distinct_agents(&state.frozen_groups[reference.group], &state.table);
        if let Some(agent) = state.table.agent(reference.member.id)
            && !agents.contains(&agent)
        {
            agents.push(agent);
        }
        if agents.len() < 2 {
            reject(
                &mut outcome,
                FailureClass::ContradictionAgents,
                format!("reference {i}: contradiction without ≥2 distinct cited agents"),
            );
            continue;
        }
        state.pinned.insert(reference.member.id);
        state.references.push(reference);
        outcome.accepted_refs += 1;
    }
    outcome
}

/// How the previous round ended — frames the next repair section honestly:
/// a transport failure or parse error never claims the response was rejected;
/// a fully-accepted round never claims a rejection; only mixed rounds are
/// presented as partially accepted.
#[derive(Clone, Copy)]
enum PrevRound {
    /// Previous call did not complete (transport failure).
    Transport,
    /// Previous response could not be parsed.
    Parse,
    /// Previous round parsed and processed.
    Processed {
        /// Whether the round accepted anything (froze groups, accepted a
        /// reference, or accepted the first summary) — distinguishes
        /// "partially accepted" from "rejected".
        accepted: bool,
    },
}

/// Append the repair-round instructions (delta schema, frozen skeletons,
/// remainder list, previous-round feedback) to the last user message.
/// Append-only: the message prefix stays byte-identical across rounds, so the
/// KV prefix is preserved. The framing is honest about the previous round's
/// disposition: a transport failure (no response at all) is not presented as
/// a rejected response.
#[allow(clippy::too_many_lines)]
fn append_repair_instructions(
    request: &mut ChatRequest,
    round: u32,
    state: &RepairState<'_>,
    rejections: &[String],
    prev: PrevRound,
) {
    let mut section = String::new();
    let framing = match prev {
        PrevRound::Transport => {
            "The previous call did not complete (transport failure) — please \
             respond now with the repair delta."
        }
        PrevRound::Parse => {
            "Your previous response could not be parsed — please respond now \
             with the repair delta."
        }
        PrevRound::Processed { accepted: false } => {
            "Your previous response was rejected — the rejected proposals are \
             listed below."
        }
        PrevRound::Processed { accepted: true } if rejections.is_empty() => {
            "Your previous response was accepted; the remaining items below \
             still need placement."
        }
        PrevRound::Processed { accepted: true } => {
            "Your previous response was partially accepted; rejected proposals \
             are listed below."
        }
    };
    let _ = write!(
        section,
        "\n\n=== REPAIR ROUND {round} ===\n\
         Only the latest REPAIR ROUND section applies — earlier sections are \
         historical and superseded.\n\
         {framing}\n\
         Accepted groups are FROZEN and must NEVER be re-proposed; repair \
         ONLY the remaining items. Reference every member by its item id from \
         the input list — never by text.\n"
    );
    if !rejections.is_empty() {
        section.push_str("\nRejected proposals still outstanding (fix these):\n");
        for r in rejections {
            let _ = writeln!(section, "- {r}");
        }
    }
    section.push_str(
        "\nFrozen groups (skeletons — do NOT re-propose their members; references \
         to a group flagged contradiction:true are ignored):\n",
    );
    if state.frozen_groups.is_empty() {
        section.push_str("- none\n");
    } else {
        for (i, g) in state.frozen_groups.iter().enumerate() {
            let agents = distinct_agents(g, &state.table)
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let representative = g.members.first().map_or_else(
                || "—".to_string(),
                |m| format!("{} (id {})", render_member_line(m, &state.table), m.id),
            );
            let _ = writeln!(
                section,
                "- {i}: {:?} [agents {agents}, contradiction: {}] — e.g. {representative}",
                g.heading, g.contradiction,
            );
        }
    }
    section.push_str(
        "\nRemaining items (place EVERY un-pinned one in a proposed group or the \
         ungrouped list; pinned items must NOT be placed in a group). Reference \
         each item by its id:\n",
    );
    for id in state.remainder() {
        let line = render_member_line(&GroupingMember { id }, &state.table);
        if state.pinned.contains(&id) {
            let _ = writeln!(
                section,
                "- {id}: {line} [pinned — accepted contradiction reference; do NOT place in a group]"
            );
        } else {
            let _ = writeln!(section, "- {id}: {line}");
        }
    }
    section.push_str(
        r#"
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
"#,
    );
    section.push_str(
        "\nNote: every item listed in `references` must ALSO appear in `groups` \
         or `ungrouped` — a reference alone does not place the item.\n",
    );
    if let Some(last) = request.messages.last_mut() {
        last.content.push_str(&section);
    }
}

/// Run the repair-mode grouping pass (pipeline synthesis + ask
/// consolidation): 1 full synthesis call + up to N-1 repair rounds (N =
/// hardcoded `DEFAULT_SYNTHESIS_MAX_ATTEMPTS`, 3 = the lower edge of the
/// approved 3–5 band). Accepted groups freeze; repair rounds propose deltas for the
/// remainder only. Termination — zero-progress (a repair round freezing no new
/// groups) or budget exhaustion — places every remaining item
/// deterministically in the ungrouped section. Fail-open fires only on
/// transport/parse/non-retryable exhaustion or an exhaustion with zero groups
/// ever frozen.
///
/// Validation and parse rejections consume a round immediately (no backoff);
/// transport failures keep the synthesis backoff schedule (indexed by the
/// transport-failure count, not the round number, so a validation-rejected
/// round never advances the backoff slot). The 600 s operation cap stays
/// binding (the synthesis policy hardcodes a 10-minute `operation_timeout`,
/// independent of the general hardcoded `DEFAULT_OPERATION_TIMEOUT`).
///
/// A round-1 parse failure permanently converts the run to the repair-delta
/// schema — the model never gets a chance to emit a corrected full output.
/// Consistent with the total-calls budget reinterpretation.
///
/// Rejection feedback is one-shot for fully-rejected rounds: a repair round
/// that fails completeness (silent dropout) freezes zero groups and therefore
/// triggers the zero-progress break before the fed-back reasons can be acted
/// on. Only rounds that freeze at least one group consume feedback across
/// rounds.
///
/// Telemetry: a round that accepts anything (froze groups, accepted a
/// reference, or accepted the first summary) is recorded as a SUCCESS through
/// the existing per-call recording path (per-round usage, the calibration
/// signal); every rejection persists its typed cause as its own
/// `retry_failures` row, from mixed and failed rounds alike. A parsed round
/// that accepts nothing is a FAILURE recorded with a typed granular cause —
/// including clean zero-progress rounds (valid ungrouped-only deltas), the
/// most common terminal shape — so no round is ever invisible in the tables
/// and the failure trail is never empty on Fallback (no spurious
/// `transport`/`retry_attempts=0` classification). Per-round exclusivity
/// holds (a round is never both a success and a failure row), but a Fallback
/// operation may still carry both an earlier summary-only round's success row
/// and the operation-level failure row — the accepted summary is discarded by
/// the fallback render. Row counts: a failed round with N rejections persists
/// N per-rejection rows PLUS one round-level failure row (the operation-trail
/// record that drives `final_class`), while a mixed round persists only the N
/// per-rejection rows plus its success row — calibration consumers should
/// count causes per-rejection, not per-row.
#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
pub(crate) async fn run_grouping_repair(
    ws: &Workspace,
    purpose: &'static str,
    mut request: ChatRequest,
    items_by_agent: &[Vec<String>],
) -> RepairOutcome {
    let _call = crate::call_registry::NON_AGENT_CALLS.register(purpose, &ws.name);
    crate::prompt::prepend_general_context(&mut request.messages, ws).await;
    let policy = RetryPolicy::synthesis();
    let mut loop_state = RetryLoop::new(&policy);
    let operation_started = Instant::now();
    let mut state = RepairState::new(items_by_agent);
    // Rejection reasons fed to the next repair round; cleared only once a
    // round responds, so feedback survives a mid-repair transport failure.
    let mut rejections: Vec<String> = Vec::new();
    // How the previous round ended — frames the next repair section honestly.
    let mut prev_round = PrevRound::Processed { accepted: false };
    // Transport-failure count — indexes the backoff schedule independently of
    // the round number (validation rejections consume a round without a sleep).
    let mut transport_failures: u32 = 0;

    for round in 1..=policy.max_attempts {
        if loop_state.expired() {
            break;
        }
        if round > 1 {
            append_repair_instructions(&mut request, round, &state, &rejections, prev_round);
        }
        let attempt_started = Instant::now();
        let resp = match crate::providers::chat_scoped(
            request.clone(),
            policy.idle_timeout,
            loop_state.deadline(),
        )
        .await
        {
            Ok(resp) => resp,
            Err(err) => {
                let non_retryable = !err.class.is_retryable();
                loop_state.record(round, err.record).await;
                if non_retryable {
                    break;
                }
                transport_failures += 1;
                if let Err(FailureClass::Shutdown) =
                    loop_state.sleep_between(transport_failures).await
                {
                    break;
                }
                prev_round = PrevRound::Transport;
                continue;
            }
        };
        let raw = resp.text_or_empty();
        let parsed: Result<RoundInput, _> = if round == 1 {
            crate::util::json::parse_fenced_json::<GroupingOutput>(raw).map(|o| RoundInput {
                summary: Some(o.summary),
                groups: o.groups,
                ungrouped: o.ungrouped,
                references: Vec::new(),
            })
        } else {
            crate::util::json::parse_fenced_json::<GroupingDelta>(raw).map(|d| RoundInput {
                summary: d.summary,
                groups: d.groups,
                ungrouped: d.ungrouped,
                references: d.references,
            })
        };
        let outcome = match parsed {
            Ok(input) => {
                let outcome = process_round(input, &mut state);
                // Deliver the previous round's rejection feedback (appended
                // above), then replace it with this round's rejections. A
                // transport or parse failure never clears it, so feedback
                // survives a mid-repair outage and the next section re-lists it.
                rejections.clear();
                if let Some(msg) = &outcome.round_rejection {
                    rejections.push(msg.clone());
                }
                rejections.extend(outcome.rejections.iter().map(|(_, msg)| msg.clone()));
                outcome
            }
            Err(e) => {
                let rec = RetryFailureRecord::new_simple(
                    round,
                    FailureClass::Parse,
                    &e,
                    attempt_started.elapsed().as_millis() as u64,
                    None,
                );
                loop_state.record(round, rec).await;
                prev_round = PrevRound::Parse;
                continue;
            }
        };
        // A round that accepted anything is a success — recorded per-round
        // through the existing per-call recording path (usage tokens per
        // round, the calibration signal). A round is never both a success row
        // and a failure row in llm_requests; its rejections still feed the
        // next round and are persisted below as granular retry_failures rows.
        let accepted = outcome.froze > 0 || outcome.accepted_refs > 0 || outcome.accepted_summary;
        prev_round = PrevRound::Processed { accepted };
        if accepted {
            crate::stats::record_llm_success(&request, attempt_started, round, &resp).await;
        }
        // Typed granular cause for telemetry: completeness (round-level
        // rejection) or the first per-group/per-entry cause.
        let cause = if outcome.round_rejection.is_some() {
            FailureClass::Completeness
        } else {
            outcome
                .rejections
                .first()
                .map_or(FailureClass::ValidationOther, |(c, _)| *c)
        };
        tracing::info!(
            purpose,
            workspace = %ws.name,
            round,
            accepted,
            froze = outcome.froze,
            refs = outcome.accepted_refs,
            summary = outcome.accepted_summary,
            rejections = outcome.rejections.len(),
            cause = cause.label(),
            "Grouping synthesis repair round",
        );
        // Every rejection persists its typed cause as its own retry_failures
        // row — from mixed rounds (accepted content AND rejected proposals)
        // and failed rounds alike — so the granular-cause calibration surface
        // never under-reports the most common round outcome.
        if !outcome.rejections.is_empty() {
            for (class, msg) in &outcome.rejections {
                let err = anyhow::anyhow!("{msg}");
                let rec = RetryFailureRecord::new_simple(
                    round,
                    *class,
                    &err,
                    attempt_started.elapsed().as_millis() as u64,
                    None,
                );
                crate::retry::record_retry_failure(&rec).await;
            }
        }
        // A parsed-and-processed round that accepted NOTHING is a failure with
        // the typed cause — including clean zero-progress rounds (a valid
        // ungrouped-only delta): they are the most common terminal shape and
        // must be visible in the calibration surface. Recording them also
        // keeps the failure trail non-empty, so a subsequent Fallback never
        // mislabels the operation as `transport`/`retry_attempts=0`.
        if !accepted {
            let detail = if rejections.is_empty() {
                "round accepted nothing (zero-progress; no validation rejections)".to_string()
            } else {
                rejections.join("; ")
            };
            let err = anyhow::anyhow!("grouping validation failed: {detail}");
            let rec = RetryFailureRecord::new_simple(
                round,
                cause,
                &err,
                attempt_started.elapsed().as_millis() as u64,
                None,
            );
            loop_state.record(round, rec).await;
        }
        if state.complete() {
            break; // every item is frozen or pinned — nothing left to place
        }
        if round > 1 && outcome.froze == 0 {
            break; // zero-progress repair round — stop immediately
        }
    }

    if state.frozen_groups.is_empty() {
        tracing::warn!(
            purpose,
            workspace = %ws.name,
            attempts = policy.max_attempts,
            "Grouping synthesis exhausted — writing deterministic fail-open output",
        );
        // A Fallback with an empty failure trail (e.g. a summary-only round at
        // max_attempts=1, or a deadline hit before any attempt) would be
        // classified `transport`/`retry_attempts=0` — spurious. Record the
        // real cause so the operation-level row reflects what happened.
        if !loop_state.has_failures() {
            let detail = if state.summary.is_some() {
                "operation exhausted after accepting only a summary — no groups ever frozen"
            } else {
                "operation exhausted with no accepted output"
            };
            let rec = RetryFailureRecord::new_simple(
                policy.max_attempts,
                FailureClass::ValidationOther,
                &anyhow::anyhow!("{detail}"),
                0,
                None,
            );
            loop_state.record(policy.max_attempts, rec).await;
        }
        let final_class = loop_state.final_class();
        let exhausted = crate::retry::RetryExhausted::with_last_raw(
            loop_state.into_failures(),
            final_class,
            None,
        );
        crate::stats::record_llm_failure(&request, operation_started, &exhausted).await;
        return RepairOutcome::Fallback;
    }
    let ungrouped: Vec<GroupingMember> = state
        .remainder()
        .into_iter()
        .map(|id| GroupingMember { id })
        .collect();
    tracing::info!(
        purpose,
        workspace = %ws.name,
        groups = state.frozen_groups.len(),
        ungrouped = ungrouped.len(),
        "Grouping synthesis repaired",
    );
    RepairOutcome::Repaired {
        output: GroupingOutput {
            summary: state.summary.unwrap_or_default(),
            groups: state.frozen_groups,
            ungrouped,
        },
        references: state.references,
    }
}

/// Build a grouping request with the given system prompt, user material, and
/// consumer-supplied model/effort/routing/max_tokens. The `meta` purpose and
/// agent id are filled in here so every consumer shares the same telemetry
/// shape (the registry entry uses the same purpose string).
#[allow(clippy::too_many_arguments)]
pub(crate) fn grouping_request(
    ws: &Workspace,
    purpose: &'static str,
    system: &str,
    user: &str,
    model: String,
    reasoning_effort: Option<String>,
    provider_order: Option<String>,
    provider_allow_fallbacks: Option<bool>,
    max_tokens: Option<u32>,
) -> ChatRequest {
    ChatRequest {
        messages: vec![
            crate::ChatMessage::system(system),
            crate::ChatMessage::user(user),
        ],
        tools: None,
        model,
        allow_image_parts: false,
        max_tokens,
        reasoning_effort,
        provider_order,
        provider_allow_fallbacks,
        response_format_json_object: true,
        meta: Some(ChatRequestMeta {
            purpose,
            agent_id: format!("grouping_{}_{}", ws.name, crate::generate_suffix()),
            role: "grouping".to_string(),
            workspace: ws.name.clone(),
            ticket_id: None,
        }),
    }
}
