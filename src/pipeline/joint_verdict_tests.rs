use super::*;
use crate::Verdict;
use crate::consensus::{
    GroupingGroup, GroupingMember, GroupingOutput, ItemTable, RepairOutcome, RepairState,
    RoundInput, bracket_label, distinct_agents, process_round,
};

fn verdict(score: u8, issues: &[&str]) -> Verdict {
    Verdict {
        score,
        issues_detected: issues.iter().map(|&s| s.to_string()).collect(),
    }
}

fn round(
    stage: &'static str,
    verdicts: Vec<(usize, Verdict)>,
    failures: Vec<(usize, &'static str)>,
    header: &'static str,
) -> JointRound<'static> {
    JointRound {
        stage,
        dispatched: verdicts.len() + failures.len(),
        verdicts: verdicts
            .into_iter()
            .map(|(agent_index, verdict)| JointVerdict {
                agent_index,
                verdict: Box::leak(Box::new(verdict)),
            })
            .collect(),
        failures: failures
            .into_iter()
            .map(|(agent_index, dump)| JointFailure {
                agent_index,
                dump: dump.to_string(),
            })
            .collect(),
        header: header.to_string(),
        threshold: 9,
    }
}

fn member(id: usize) -> GroupingMember {
    GroupingMember { id }
}

fn group(heading: &str, contradiction: bool, members: Vec<GroupingMember>) -> GroupingGroup {
    GroupingGroup {
        heading: heading.to_string(),
        contradiction,
        members,
    }
}

fn output(
    summary: &str,
    groups: Vec<GroupingGroup>,
    ungrouped: Vec<GroupingMember>,
) -> GroupingOutput {
    GroupingOutput {
        summary: summary.to_string(),
        groups,
        ungrouped,
    }
}

/// Repair-mode terminal with no contradiction references (renderer tests).
fn repaired(out: GroupingOutput) -> RepairOutcome {
    RepairOutcome::Repaired {
        output: out,
        references: vec![],
    }
}

/// Round-1 acceptance helper: run the repair-path validator over the output
/// with an empty state (the round-1 surface — the sole structural validator).
fn validate_round1(output: &GroupingOutput, items: &[Vec<String>]) -> Result<(), String> {
    let mut state = RepairState::new(items);
    let input = RoundInput {
        summary: Some(output.summary.clone()),
        groups: output.groups.clone(),
        ungrouped: output.ungrouped.clone(),
        references: Vec::new(),
    };
    let outcome = process_round(input, &mut state);
    outcome
        .round_rejection
        .or_else(|| outcome.rejections.first().map(|(_, m)| m.clone()))
        .map_or(Ok(()), Err)
}

// ── Shared-core validator invariants (structural, id-based) ─────────────

#[test]
fn validator_accepts_id_based_grouping() {
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check", "Missing retry"])),
            (1, verdict(9, &["Naming nit"])),
        ],
        vec![],
        "",
    );
    let items = issues_by_agent(&r);
    // Global flat ids: 0 = agent0 "No timeout check", 1 = agent0 "Missing
    // retry", 2 = agent1 "Naming nit".
    let out = output(
        "The review found timeout handling gaps and a naming nit.",
        vec![group("Timeouts", false, vec![member(0)])],
        vec![member(1), member(2)],
    );
    assert_eq!(validate_round1(&out, &items), Ok(()));
}

#[test]
fn validator_rejects_out_of_range_id() {
    // The old fabricated-text rejection is now the out-of-range-id rejection:
    // the model can only cite existing ids, so an id beyond the table is the
    // only way to reference something that does not exist.
    let r = round(
        "Review",
        vec![(0, verdict(4, &["No timeout check"]))],
        vec![],
        "",
    );
    let items = issues_by_agent(&r);
    for bad in [1usize, 99] {
        let out = output(
            "summary",
            vec![group("Timeouts", false, vec![member(bad)])],
            vec![member(0)],
        );
        let err = validate_round1(&out, &items).expect_err("must reject");
        assert!(err.contains("unknown item id"), "err: {err}");
    }
}

#[test]
fn validator_rejects_duplicate_placement() {
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check"])),
            (1, verdict(3, &["Missing retry"])),
        ],
        vec![],
        "",
    );
    let items = issues_by_agent(&r);
    // id 0 is placed in two groups — the second placement is rejected.
    let out = output(
        "summary",
        vec![
            group("A", false, vec![member(0)]),
            group("B", false, vec![member(0)]),
        ],
        vec![member(1)],
    );
    let err = validate_round1(&out, &items).expect_err("must reject");
    assert!(err.contains("already placed"), "err: {err}");
}

#[test]
fn validator_rejects_contradiction_without_two_agents() {
    let r = round(
        "Review",
        vec![(0, verdict(4, &["No timeout check"]))],
        vec![],
        "",
    );
    let items = issues_by_agent(&r);
    let out = output("summary", vec![group("A", true, vec![member(0)])], vec![]);
    let err = validate_round1(&out, &items).expect_err("must reject");
    assert!(
        err.contains("without ≥2 distinct cited agents"),
        "contradiction guardrail: {err}"
    );
}

#[test]
fn validator_rejects_incomplete_coverage() {
    let r = round(
        "Review",
        vec![(0, verdict(4, &["No timeout check", "Missing retry"]))],
        vec![],
        "",
    );
    let items = issues_by_agent(&r);
    // id 1 ("Missing retry") is silently dropped from groups AND ungrouped.
    let out = output("summary", vec![group("A", false, vec![member(0)])], vec![]);
    let err = validate_round1(&out, &items).expect_err("must reject");
    assert!(
        err.contains("missing from every proposed group"),
        "err: {err}"
    );
}

#[test]
fn duplicate_issue_within_agent_gets_distinct_ids() {
    // Per-agent dedup is removed: the same issue twice from one agent is TWO
    // distinct ids, and the model places each exactly once. A single citation
    // of the item is now incomplete coverage.
    let r = round(
        "Review",
        vec![(0, verdict(4, &["No timeout check", "No timeout check"]))],
        vec![],
        "",
    );
    let items = issues_by_agent(&r);
    assert_eq!(
        items[0],
        vec![
            "No timeout check".to_string(),
            "No timeout check".to_string()
        ],
        "identical texts stay distinct ids"
    );
    let out = output("summary", vec![group("A", false, vec![member(0)])], vec![]);
    let err = validate_round1(&out, &items).expect_err("one citation is incomplete");
    assert!(
        err.contains("missing from every proposed group"),
        "err: {err}"
    );
    let out = output(
        "summary",
        vec![group("A", false, vec![member(0)])],
        vec![member(1)],
    );
    assert_eq!(validate_round1(&out, &items), Ok(()));
}

#[test]
fn validator_uses_original_agent_indices_with_mid_round_failure() {
    // Agent 1 failed mid-round: its slot stays empty. Ids follow the ORIGINAL
    // dispatch order — id 1 belongs to the failed agent's (empty) slot, so a
    // faithful LLM citing agent 2's issue uses id 1 and passes.
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check"])),
            (2, verdict(3, &["Missing retry"])),
        ],
        vec![(1, "agent crashed")],
        "",
    );
    let items = issues_by_agent(&r);
    assert_eq!(
        items[1],
        Vec::<String>::new(),
        "failed agent slot stays empty"
    );
    let out = output(
        "The review found a timeout gap and a missing retry.",
        vec![group("Robustness", false, vec![member(0), member(1)])],
        vec![],
    );
    assert_eq!(
        validate_round1(&out, &items),
        Ok(()),
        "original dispatch order must pass validation"
    );
    let text = render_joint_comment(&r, &repaired(out), &ItemTable::new(&issues_by_agent(&r)));
    assert!(
        text.contains("Missing retry"),
        "member of agent 2 must render via the id table: {text}"
    );
    assert!(
        text.contains("No timeout check"),
        "member of agent 0 must render via the id table: {text}"
    );
    assert!(
        !text.contains("Agent 0: No timeout check") && !text.contains("Agent 2: Missing retry"),
        "issue lines carry no agent prefixes: {text}"
    );
    assert!(
        text.contains("Agent 2: agent crashed"),
        "the failure appendix keeps its agent label: {text}"
    );
    assert!(!text.contains("missing issue"), "no silent drops: {text}");
}

// ── Bracket arithmetic ─────────────────────────────────────────────────

#[test]
fn brackets_computed_from_distinct_cited_agents() {
    let items = vec![
        vec!["x".to_string()],
        vec![],
        vec!["x".to_string(), "y".to_string()],
    ];
    let table = ItemTable::new(&items);
    let g = group("A", false, vec![member(0), member(1), member(2)]);
    assert_eq!(distinct_agents(&g, &table), vec![0, 2]);
    // Solo [1/N] renders without DISPUTED.
    assert_eq!(bracket_label(1, 3, false), "[1/3]");
    // Contradiction renders [n/N · DISPUTED].
    assert_eq!(bracket_label(2, 3, true), "[2/3 · DISPUTED]");
    // Multi-agent consensus without contradiction stays plain.
    assert_eq!(bracket_label(3, 3, false), "[3/3]");
    // Single-valid round: [1/1] (no DISPUTED).
    assert_eq!(bracket_label(1, 1, false), "[1/1]");
}

// ── Rendering ───────────────────────────────────────────────────────────

#[test]
fn renderer_with_partial_failures() {
    let r = round(
        "Review",
        vec![(0, verdict(4, &["No timeout check"])), (1, verdict(9, &[]))],
        vec![(2, "agent produced no response — crashed")],
        "",
    );
    let text = render_joint_comment(
        &r,
        &RepairOutcome::Fallback,
        &ItemTable::new(&issues_by_agent(&r)),
    );
    assert!(
        !text.contains("valid verdicts"),
        "round headline and verdict counts are noise: {text}"
    );
    assert!(
        text.contains("- No timeout check"),
        "fallback renders the raw per-agent member dump: {text}"
    );
    assert!(text.contains("### Agent failures"));
    assert!(text.contains("Agent 3: agent produced no response — crashed"));
    assert!(
        text.contains("LLM grouping unavailable"),
        "fallback marker: {text}"
    );
}

#[test]
fn renderer_no_issues_summary_respects_threshold() {
    // Fully clean round: all verdicts clear the threshold.
    let r = round(
        "Review",
        vec![(0, verdict(10, &[])), (1, verdict(9, &[]))],
        vec![],
        "",
    );
    let text = render_joint_comment(
        &r,
        &RepairOutcome::Fallback,
        &ItemTable::new(&issues_by_agent(&r)),
    );
    assert!(
        !text.contains("valid verdicts"),
        "round headline and verdict counts are noise: {text}"
    );
    assert!(
        text.contains("all 2 agents passed clean"),
        "clean-round summary must state the pass outcome: {text}"
    );
    assert!(
        !text.contains("LLM grouping unavailable"),
        "clean rounds must not imply the grouping step failed: {text}"
    );

    // Bounced round: a sub-threshold verdict with an empty issues list
    // (reachable — extraction only validates the score range) must not
    // claim a clean pass.
    let bounced = round(
        "Review",
        vec![(0, verdict(8, &[])), (1, verdict(10, &[]))],
        vec![],
        "",
    );
    let text = render_joint_comment(
        &bounced,
        &RepairOutcome::Fallback,
        &ItemTable::new(&issues_by_agent(&bounced)),
    );
    assert!(
        !text.contains("passed clean"),
        "bounced round must not claim a clean pass: {text}"
    );
    assert!(
        text.contains("No issues found by the responding agents."),
        "bounced round keeps the neutral no-issues wording: {text}"
    );

    // All-failed round: zero valid verdicts — the summary must not imply
    // that responding agents exist.
    let all_failed = round("Review", vec![], vec![(0, "crashed"), (1, "crashed")], "");
    let text = render_joint_comment(
        &all_failed,
        &RepairOutcome::Fallback,
        &ItemTable::new(&issues_by_agent(&all_failed)),
    );
    assert!(
        text.contains("no agents produced a verdict"),
        "all-failed summary must not imply responders: {text}"
    );
}

#[test]
fn renderer_with_synthesis_groups_and_contradiction() {
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check"])),
            (1, verdict(3, &["Missing error handling"])),
        ],
        vec![],
        "",
    );
    let items = issues_by_agent(&r);
    let table = ItemTable::new(&items);
    let out = output(
        "The review found two distinct issues.",
        vec![
            group("Robustness", false, vec![member(0)]),
            group("Safety", true, vec![member(0), member(1)]),
        ],
        vec![],
    );
    let text = render_joint_comment(&r, &repaired(out), &table);
    assert!(
        text.contains("**Robustness**\n- No timeout check"),
        "solo group renders without brackets or agent attribution: {text}"
    );
    assert!(
        text.contains("**Safety** — DISPUTED\n- No timeout check\n- Missing error handling"),
        "contradiction group keeps its heading, both member issues, and the DISPUTED marker: {text}"
    );
    assert!(
        !text.contains("Agent 0:") && !text.contains('[') && !text.contains(']'),
        "per-agent attribution and brackets are stripped: {text}"
    );
}

#[test]
fn renderer_with_ungrouped_trailing_section() {
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check"])),
            (1, verdict(9, &["Naming nit"])),
        ],
        vec![],
        "",
    );
    let items = issues_by_agent(&r);
    let table = ItemTable::new(&items);
    let out = output(
        "One issue grouped, one left ungrouped.",
        vec![group("Timeouts", false, vec![member(0)])],
        vec![member(1)],
    );
    let text = render_joint_comment(&r, &repaired(out), &table);
    assert!(
        text.contains("**Ungrouped**"),
        "ungrouped section renders: {text}"
    );
    assert!(
        text.contains("- Naming nit"),
        "ungrouped member renders deterministically: {text}"
    );
}

#[test]
fn renderer_strips_blocker_prefixes() {
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check"])),
            (1, verdict(9, &["Naming nit"])),
        ],
        vec![],
        "",
    );
    let items = issues_by_agent(&r);
    let table = ItemTable::new(&items);
    let out = output(
        "summary",
        vec![
            group("A", false, vec![member(0)]),
            group("B", false, vec![member(1)]),
        ],
        vec![],
    );
    let text = render_joint_comment(&r, &repaired(out), &table);
    assert!(
        text.contains("- No timeout check"),
        "below-threshold agent's issue text is kept: {text}"
    );
    assert!(
        !text.contains("[blocker]"),
        "blocker severity prefixes are stripped: {text}"
    );
}

// ── Prompt↔validator contract (global flat item ids) ───────────────────

#[test]
fn synthesis_request_uses_global_flat_item_ids() {
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check"])),
            (1, verdict(3, &["Missing retry"])),
        ],
        vec![],
        "",
    );
    let ws = crate::workspace::test_ws_named("/tmp/test_ws", "joint_verdict_synth_request");
    let request = synthesis_request(&r, Role::Reviewer, &ws);
    let system = &request.messages[0].content;
    assert!(
        system.contains("id"),
        "prompt must state the id-reference contract: {system}"
    );
    let user = &request.messages[1].content;
    assert!(
        user.contains("0: No timeout check") && user.contains("1: Missing retry"),
        "input must number items with global flat ids: {user}"
    );
    assert!(
        user.contains("Agent 0:") && user.contains("Agent 1:"),
        "agent labels remain visible: {user}"
    );
    assert!(
        !user.contains("Agent 0 (score") && !user.contains("(score"),
        "scores are not part of the synthesis input: {user}"
    );
    assert!(
        !user.contains("Agent 2"),
        "no agent index beyond the dispatch count: {user}"
    );
}

#[tokio::test]
#[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
#[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global seams
async fn run_synthesis_end_to_end_zero_based_contract() {
    let _lock = crate::util::test::retry_tests_lock();
    let _policy = crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());

    // A faithful LLM copies the input labels (0-based) straight into the
    // schema — this output must be accepted end-to-end (round 1 completes:
    // every item is in a frozen group).
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check"])),
            (1, verdict(3, &["Missing retry"])),
        ],
        vec![],
        "",
    );
    let provider = crate::util::test::FakeProvider::new().ok(
        r#"{"summary":"Two distinct issues.","groups":[{"heading":"Robustness","contradiction":false,"members":[{"id":0},{"id":1}]}],"ungrouped":[]}"#,
    );
    let _provider = crate::util::test::install_fake_provider(std::sync::Arc::new(provider));
    let ws = crate::workspace::test_ws_named("/tmp/test_ws", "joint_verdict_zero_based");
    let outcome = run_synthesis(&r, Role::Reviewer, &ws, "test_ticket", "Test ticket").await;
    match outcome {
        RepairOutcome::Repaired { output, .. } => assert_eq!(output.groups.len(), 1),
        RepairOutcome::Fallback => panic!("valid id-based output must not fall back"),
    }

    // An out-of-range id (99 in a 2-item round) must be rejected and
    // eventually fall back. Round 1 is completeness-rejected (the raw
    // proposal omits real items); rounds 2–3 hit the unscripted-default
    // parse failure (the script holds one response) — exhaustion is
    // parse-failure + budget, not zero-progress.
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check"])),
            (1, verdict(3, &["Missing retry"])),
        ],
        vec![],
        "",
    );
    let fake = std::sync::Arc::new(crate::util::test::FakeProvider::new().ok(
        r#"{"summary":"Two distinct issues.","groups":[{"heading":"Robustness","contradiction":false,"members":[{"id":99}]}],"ungrouped":[]}"#,
    ));
    let _provider = crate::util::test::install_fake_provider(fake.clone());
    let outcome = run_synthesis(&r, Role::Reviewer, &ws, "test_ticket", "Test ticket").await;
    assert!(
        matches!(outcome, RepairOutcome::Fallback),
        "out-of-range id must exhaust synthesis into the fallback"
    );
    // The rejection must be fed back into the next round so the LLM can
    // self-correct (never a byte-identical resend after a validation error).
    let fingerprints = fake.request_fingerprints.lock().unwrap().clone();
    assert!(
        fingerprints.len() >= 2 && fingerprints[1].contains("previous response was rejected"),
        "validation rejection must be fed back: {fingerprints:?}"
    );
}

#[tokio::test]
#[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
#[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global seams
async fn repair_rounds_freeze_groups_and_converge() {
    let _lock = crate::util::test::retry_tests_lock();
    let _policy = crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());

    // Round 1: one valid group (freezes) + one out-of-range-id group (dropped
    // with a per-group reason) + ungrouped entries covering the remainder.
    // Round 2: delta groups the remaining item. Round 3: delta leaves the
    // last item ungrouped → zero-progress stops the loop; the remainder
    // renders deterministically in the ungrouped section.
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check", "Missing retry"])),
            (1, verdict(9, &["Naming nit"])),
        ],
        vec![],
        "",
    );
    // ids: 0 = agent0 "No timeout check", 1 = agent0 "Missing retry",
    // 2 = agent1 "Naming nit".
    let fake = crate::util::test::FakeProvider::new()
        .ok(
            r#"{"summary":"Two issues grouped, one left.","groups":[{"heading":"Robustness","contradiction":false,"members":[{"id":0}]},{"heading":"Fabricated","contradiction":false,"members":[{"id":99}]}],"ungrouped":[{"id":1},{"id":2}]}"#,
        )
        .ok(
            r#"{"groups":[{"heading":"Retry","contradiction":false,"members":[{"id":1}]}],"ungrouped":[{"id":2}]}"#,
        )
        .ok(
            r#"{"groups":[],"ungrouped":[{"id":2}]}"#,
        );
    let fake = std::sync::Arc::new(fake);
    let _provider = crate::util::test::install_fake_provider(fake.clone());
    let ws = crate::workspace::test_ws_named("/tmp/test_ws", "joint_verdict_freeze");
    let outcome = run_synthesis(&r, Role::Reviewer, &ws, "test_ticket", "Test ticket").await;
    let RepairOutcome::Repaired { output, references } = outcome else {
        panic!("repair must converge, got fallback");
    };
    assert_eq!(
        output.groups.len(),
        2,
        "frozen groups preserved across rounds"
    );
    assert_eq!(
        output.ungrouped,
        vec![member(2)],
        "deterministic remainder placement"
    );
    assert_eq!(references.len(), 0);
    assert!(
        output.summary.contains("Two issues grouped"),
        "first-accepted summary is final: {}",
        output.summary
    );
    let fingerprints = fake.request_fingerprints.lock().unwrap().clone();
    assert_eq!(fingerprints.len(), 3, "1 full + 2 repair rounds");
    assert!(
        fingerprints[1].contains("REPAIR ROUND 2") && fingerprints[1].contains("unknown item id"),
        "repair instructions + per-group rejection must reach the model: {}",
        fingerprints[1]
    );
}

#[tokio::test]
#[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
#[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global seams
async fn repair_zero_progress_with_no_frozen_groups_falls_back() {
    let _lock = crate::util::test::retry_tests_lock();
    let _policy = crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());

    // Round 1 freezes nothing (fabricated group dropped) — proceeds to a
    // repair round (round-1 zero-frozen never terminates the loop). Round 2
    // freezes nothing again → zero-progress with zero groups ever frozen →
    // deterministic fail-open.
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check"])),
            (1, verdict(3, &["Missing retry"])),
        ],
        vec![],
        "",
    );
    // ids: 0 = agent0 "No timeout check", 1 = agent1 "Missing retry".
    let fake = crate::util::test::FakeProvider::new()
        .ok(
            r#"{"summary":"s","groups":[{"heading":"Bad","contradiction":false,"members":[{"id":99}]}],"ungrouped":[{"id":0},{"id":1}]}"#,
        )
        .ok(
            r#"{"groups":[{"heading":"Bad","contradiction":false,"members":[{"id":98}]}],"ungrouped":[{"id":0},{"id":1}]}"#,
        );
    let fake = std::sync::Arc::new(fake);
    let _provider = crate::util::test::install_fake_provider(fake);
    let ws = crate::workspace::test_ws_named("/tmp/test_ws", "joint_verdict_zero_progress");
    let outcome = run_synthesis(&r, Role::Reviewer, &ws, "test_ticket", "Test ticket").await;
    assert!(
        matches!(outcome, RepairOutcome::Fallback),
        "zero-progress with zero groups ever frozen must fall back"
    );
}

#[tokio::test]
#[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
#[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global seams
async fn repair_contradiction_reference_renders_disputed() {
    let _lock = crate::util::test::retry_tests_lock();
    let _policy = crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());

    // Round 1 freezes a consensus group (contradiction:false). Round 2's
    // delta references it from a remainder item → accepted (≥2 distinct
    // agents: frozen group agent 0 + item agent 1) and rendered in the
    // ungrouped section with DISPUTED + a cross-reference to the frozen group.
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["Safe"])),
            (1, verdict(3, &["Actually unsafe"])),
        ],
        vec![],
        "",
    );
    // ids: 0 = agent0 "Safe", 1 = agent1 "Actually unsafe".
    let fake = crate::util::test::FakeProvider::new()
        .ok(
            r#"{"summary":"One consensus, one dispute.","groups":[{"heading":"Safety","contradiction":false,"members":[{"id":0}]}],"ungrouped":[{"id":1}]}"#,
        )
        .ok(
            r#"{"groups":[],"ungrouped":[{"id":1}],"references":[{"group":0,"member":{"id":1}}]}"#,
        );
    let fake = std::sync::Arc::new(fake);
    let _provider = crate::util::test::install_fake_provider(fake);
    let ws = crate::workspace::test_ws_named("/tmp/test_ws", "joint_verdict_disputed");
    let outcome = run_synthesis(&r, Role::Reviewer, &ws, "test_ticket", "Test ticket").await;
    let RepairOutcome::Repaired { output, references } = outcome else {
        panic!("reference round must converge, got fallback");
    };
    assert_eq!(references.len(), 1, "contradiction reference accepted");
    let text = render_joint_comment(
        &r,
        &RepairOutcome::Repaired { output, references },
        &ItemTable::new(&issues_by_agent(&r)),
    );
    assert!(
        text.contains("Actually unsafe [DISPUTED — contradicts group 0 \"Safety\"]"),
        "reference must render with DISPUTED + cross-ref: {text}"
    );
}

#[tokio::test]
#[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
#[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global seams
async fn repair_rejects_empty_member_group() {
    let _lock = crate::util::test::retry_tests_lock();
    let _policy = crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());

    // Round 1 proposes an empty-member group (must NOT freeze as progress —
    // it places nothing and would render an empty group heading) alongside a
    // valid group; the rejection reaches the round-2 repair prompt. Round 2
    // leaves the remainder ungrouped → zero-progress stop.
    let r = round(
        "Review",
        vec![(0, verdict(4, &["Safe"])), (1, verdict(3, &["Unsafe"]))],
        vec![],
        "",
    );
    let fake = crate::util::test::FakeProvider::new()
        .ok(
            r#"{"summary":"s","groups":[{"heading":"Empty","contradiction":false,"members":[]},{"heading":"Safety","contradiction":false,"members":[{"id":0}]}],"ungrouped":[{"id":1}]}"#,
        )
        .ok(
            r#"{"groups":[],"ungrouped":[{"id":1}]}"#,
        );
    let fake = std::sync::Arc::new(fake);
    let _provider = crate::util::test::install_fake_provider(fake.clone());
    let ws = crate::workspace::test_ws_named("/tmp/test_ws", "joint_verdict_empty_member");
    let outcome = run_synthesis(&r, Role::Reviewer, &ws, "test_ticket", "Test ticket").await;
    let RepairOutcome::Repaired { output, .. } = outcome else {
        panic!("empty-member rejection must not force a fallback");
    };
    assert_eq!(
        output.groups.len(),
        1,
        "empty-member group must not freeze as progress"
    );
    let fingerprints = fake.request_fingerprints.lock().unwrap().clone();
    assert!(
        fingerprints[1].contains("group has no members"),
        "empty-member rejection must reach the repair prompt: {}",
        fingerprints[1]
    );
}

// ── Calibrated dynamic agent counts ─────────────────────────────────────

#[test]
fn review_base_from_signals_thresholds() {
    // Arguments reference the calibration constants so threshold changes
    // cannot silently drift the test away from the shipped ladder.
    let low = DEFAULT_REVIEW_COUNT_LOW_CHURN;
    let high = DEFAULT_REVIEW_COUNT_HIGH_CHURN;
    // Low churn (≤ low) → 2.
    assert_eq!(review_base_from_signals(10, low, high), 2);
    assert_eq!(
        review_base_from_signals(low, low, high),
        2,
        "boundary {low} is inclusive → 2"
    );
    // High churn (> high) → 4.
    assert_eq!(
        review_base_from_signals(high + 1, low, high),
        4,
        "boundary {high} is exclusive → 4 requires > {high}"
    );
    assert_eq!(review_base_from_signals(high * 2, low, high), 4);
    // Middle → 3, including exactly `high`.
    assert_eq!(review_base_from_signals(low + 1, low, high), 3);
    assert_eq!(review_base_from_signals(high - 1, low, high), 3);
    assert_eq!(
        review_base_from_signals(high, low, high),
        3,
        "exactly {high} stays 3 (strict > boundary)"
    );
}

#[test]
fn review_agent_count_adjustments() {
    assert_eq!(review_agent_count(2, 1), 2, "normal ticket keeps base");
    assert_eq!(review_agent_count(3, 1), 3, "no bounce adjustment");
    assert_eq!(review_agent_count(4, 1), 4, "capped at 4");
    assert_eq!(review_agent_count(2, 0), 3, "P0 never gets 2");
    assert_eq!(review_agent_count(3, 0), 3, "P0 floor 3");
    assert_eq!(review_agent_count(4, 0), 4, "P0 with base 4");
}
