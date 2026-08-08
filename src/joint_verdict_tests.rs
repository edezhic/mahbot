use super::*;
use crate::Verdict;
use crate::consensus::{
    GroupingGroup, GroupingMember, GroupingOutput, RepairOutcome, bracket_label, distinct_agents,
    normalize_item, validate_grouping,
};

fn verdict(score: u8, issues: &[&str]) -> Verdict {
    Verdict {
        score,
        critique: None,
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

fn member(agent: usize, text: &str) -> GroupingMember {
    GroupingMember {
        agent,
        text: text.to_string(),
    }
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

// ── Shared-core validator invariants ────────────────────────────────────

#[test]
fn validator_accepts_verbatim_grouping() {
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
    let out = output(
        "The review found timeout handling gaps and a naming nit.",
        vec![group(
            "Timeouts",
            false,
            vec![member(0, "No timeout check")],
        )],
        vec![member(0, "Missing retry"), member(1, "Naming nit")],
    );
    assert_eq!(validate_grouping(&out, &items), Ok(()));
}

#[test]
fn validator_rejects_fabricated_consensus() {
    let r = round(
        "Review",
        vec![(0, verdict(4, &["No timeout check"]))],
        vec![],
        "",
    );
    let items = issues_by_agent(&r);

    // Fabricated text (not in agent 0's list) is rejected.
    let out = output(
        "summary",
        vec![group(
            "Timeouts",
            false,
            vec![member(0, "No timeout check AND missing retry")],
        )],
        vec![],
    );
    let err = validate_grouping(&out, &items).expect_err("must reject");
    assert!(err.contains("not found verbatim"), "err: {err}");

    // Unknown agent index is rejected.
    let out = output(
        "summary",
        vec![group(
            "Timeouts",
            false,
            vec![member(7, "No timeout check")],
        )],
        vec![],
    );
    let err = validate_grouping(&out, &items).expect_err("must reject");
    assert!(err.contains("unknown agent index"), "err: {err}");
}

#[test]
fn validator_rejects_double_counting() {
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
    let out = output(
        "summary",
        vec![
            group("A", false, vec![member(0, "No timeout check")]),
            group("B", false, vec![member(0, "No timeout check")]),
        ],
        vec![],
    );
    let err = validate_grouping(&out, &items).expect_err("must reject");
    assert!(err.contains("double-counts"), "err: {err}");
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
    let out = output(
        "summary",
        vec![group("A", true, vec![member(0, "No timeout check")])],
        vec![],
    );
    let err = validate_grouping(&out, &items).expect_err("must reject");
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
    // "Missing retry" is silently dropped from groups AND ungrouped.
    let out = output(
        "summary",
        vec![group("A", false, vec![member(0, "No timeout check")])],
        vec![],
    );
    let err = validate_grouping(&out, &items).expect_err("must reject");
    assert!(err.contains("missing from every group"), "err: {err}");
}

#[test]
fn duplicate_issue_within_agent_is_deduped() {
    // One agent reporting the same issue twice must not make the
    // double-count/completeness pair unsatisfiable: the per-agent input list
    // is deduped, so a single citation of the item validates and covers it.
    let r = round(
        "Review",
        vec![(0, verdict(4, &["No timeout check", "No timeout check"]))],
        vec![],
        "",
    );
    let items = issues_by_agent(&r);
    assert_eq!(items[0], vec!["No timeout check".to_string()]);
    let out = output(
        "summary",
        vec![group("A", false, vec![member(0, "No timeout check")])],
        vec![],
    );
    assert_eq!(validate_grouping(&out, &items), Ok(()));
}

#[test]
fn validator_uses_original_agent_indices_with_mid_round_failure() {
    // Agent 1 failed mid-round: the LLM sees "Agent 0" and "Agent 2" (original
    // dispatch indices) in the input, never a compacted 0..n-1 space. A
    // faithful LLM emitting agent:2 must pass validation AND render.
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check"])),
            (2, verdict(3, &["Missing retry"])),
        ],
        vec![(1, "agent crashed")],
        "",
    );
    let out = output(
        "The review found a timeout gap and a missing retry.",
        vec![group(
            "Robustness",
            false,
            vec![member(0, "No timeout check"), member(2, "Missing retry")],
        )],
        vec![],
    );
    assert_eq!(
        validate_grouping(&out, &issues_by_agent(&r)),
        Ok(()),
        "original dispatch indices must pass validation"
    );
    let text = render_joint_comment(&r, &repaired(out));
    assert!(
        text.contains("Missing retry"),
        "member of agent 2 must render: {text}"
    );
    assert!(
        text.contains("No timeout check"),
        "member of agent 0 must render: {text}"
    );
    assert!(!text.contains("missing issue"), "no silent drops: {text}");
}

#[test]
fn normalize_item_collapses_whitespace() {
    assert_eq!(
        normalize_item("  No   timeout\ncheck  "),
        "No timeout check"
    );
    assert_eq!(normalize_item("a\tb c"), "a b c");
}

// ── Bracket arithmetic ─────────────────────────────────────────────────

#[test]
fn brackets_computed_from_distinct_cited_agents() {
    let g = group(
        "A",
        false,
        vec![member(0, "x"), member(2, "x"), member(2, "y")],
    );
    assert_eq!(distinct_agents(&g), vec![0, 2]);
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
        "1 of 2 valid verdicts failed (threshold 9/10).",
    );
    let text = render_joint_comment(&r, &RepairOutcome::Fallback);
    assert!(text.contains("## Review round — 2/3 valid verdicts"));
    assert!(
        text.contains("- Agent 0: No timeout check"),
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
    let out = output(
        "The review found two distinct issues.",
        vec![
            group("Robustness", false, vec![member(0, "No timeout check")]),
            group(
                "Safety",
                true,
                vec![
                    member(0, "No timeout check"),
                    member(1, "Missing error handling"),
                ],
            ),
        ],
        vec![],
    );
    let text = render_joint_comment(&r, &repaired(out));
    assert!(
        text.contains("**Robustness** [1/2]"),
        "solo group renders [1/2] without DISPUTED: {text}"
    );
    assert!(
        text.contains("**Safety** [2/2 · DISPUTED]"),
        "contradiction group renders [2/2 · DISPUTED]: {text}"
    );
    assert!(
        text.contains("Agent 0: No timeout check"),
        "member lines are attributed per source: {text}"
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
    let out = output(
        "One issue grouped, one left ungrouped.",
        vec![group(
            "Timeouts",
            false,
            vec![member(0, "No timeout check")],
        )],
        vec![member(1, "Naming nit")],
    );
    let text = render_joint_comment(&r, &repaired(out));
    assert!(
        text.contains("**Ungrouped**"),
        "ungrouped section renders: {text}"
    );
    assert!(
        text.contains("- Agent 1: Naming nit"),
        "ungrouped member renders deterministically: {text}"
    );
}

#[test]
fn renderer_marks_blocker_from_cited_agent_score() {
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check"])),
            (1, verdict(9, &["Naming nit"])),
        ],
        vec![],
        "",
    );
    let out = output(
        "summary",
        vec![
            group("A", false, vec![member(0, "No timeout check")]),
            group("B", false, vec![member(1, "Naming nit")]),
        ],
        vec![],
    );
    let text = render_joint_comment(&r, &repaired(out));
    assert!(
        text.contains("[blocker] Agent 0: No timeout check"),
        "below-threshold agent's member renders as blocker: {text}"
    );
    assert!(
        !text.contains("[blocker] Agent 1"),
        "passing agent's member is not a blocker: {text}"
    );
}

// ── Prompt↔validator contract (zero-based agent labels) ────────────────

#[test]
fn synthesis_request_uses_zero_based_agent_labels() {
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check"])),
            (1, verdict(3, &["Missing retry"])),
        ],
        vec![],
        "",
    );
    let ws = crate::workspace::test_ws("/tmp/test_ws");
    let request = synthesis_request(&r, Role::Reviewer, &ws);
    let system = &request.messages[0].content;
    assert!(
        system.contains("ZERO-BASED"),
        "prompt must state the zero-based label contract: {system}"
    );
    let user = &request.messages[1].content;
    assert!(
        user.contains("Agent 0:"),
        "input must label the first agent 'Agent 0': {user}"
    );
    assert!(
        user.contains("Agent 1:"),
        "input must label the second agent 'Agent 1': {user}"
    );
    assert!(
        !user.contains("Agent 0 (score") && !user.contains("(score"),
        "scores are not part of the synthesis input (the renderer adds them): {user}"
    );
    assert!(
        !user.contains("Agent 2"),
        "no agent index beyond the dispatch count: {user}"
    );
}

#[tokio::test]
#[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global seams
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
        r#"{"summary":"Two distinct issues.","groups":[{"heading":"Robustness","contradiction":false,"members":[{"agent":0,"text":"No timeout check"},{"agent":1,"text":"Missing retry"}]}],"ungrouped":[]}"#,
    );
    let _provider = crate::util::test::install_fake_provider(std::sync::Arc::new(provider));
    let ws = crate::workspace::test_ws("/tmp/test_ws");
    let outcome = run_synthesis(&r, Role::Reviewer, &ws).await;
    match outcome {
        RepairOutcome::Repaired { output, .. } => assert_eq!(output.groups.len(), 1),
        RepairOutcome::Fallback => panic!("valid zero-based output must not fall back"),
    }

    // A 1-based index (agent 2 in a 2-agent round — the pre-fix prompt
    // contract) must be rejected and eventually fall back. Round 1 is
    // completeness-rejected (the raw proposal omits real items); rounds 2–3
    // hit the unscripted-default parse failure (the script holds one
    // response) — exhaustion is parse-failure + budget, not zero-progress.
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
        r#"{"summary":"Two distinct issues.","groups":[{"heading":"Robustness","contradiction":false,"members":[{"agent":2,"text":"No timeout check"}]}],"ungrouped":[]}"#,
    ));
    let _provider = crate::util::test::install_fake_provider(fake.clone());
    let outcome = run_synthesis(&r, Role::Reviewer, &ws).await;
    assert!(
        matches!(outcome, RepairOutcome::Fallback),
        "out-of-range 1-based index must exhaust synthesis into the fallback"
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
#[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global seams
async fn repair_rounds_freeze_groups_and_converge() {
    let _lock = crate::util::test::retry_tests_lock();
    let _policy = crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());

    // Round 1: one valid group (freezes) + one fabricated group (dropped with
    // a per-group reason) + ungrouped entries covering the remainder.
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
    let fake = crate::util::test::FakeProvider::new()
        .ok(
            r#"{"summary":"Two issues grouped, one left.","groups":[{"heading":"Robustness","contradiction":false,"members":[{"agent":0,"text":"No timeout check"}]},{"heading":"Fabricated","contradiction":false,"members":[{"agent":1,"text":"Invented text"}]}],"ungrouped":[{"agent":0,"text":"Missing retry"},{"agent":1,"text":"Naming nit"}]}"#,
        )
        .ok(
            r#"{"groups":[{"heading":"Retry","contradiction":false,"members":[{"agent":0,"text":"Missing retry"}]}],"ungrouped":[{"agent":1,"text":"Naming nit"}]}"#,
        )
        .ok(
            r#"{"groups":[],"ungrouped":[{"agent":1,"text":"Naming nit"}]}"#,
        );
    let fake = std::sync::Arc::new(fake);
    let _provider = crate::util::test::install_fake_provider(fake.clone());
    let ws = crate::workspace::test_ws("/tmp/test_ws");
    let outcome = run_synthesis(&r, Role::Reviewer, &ws).await;
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
        vec![member(1, "Naming nit")],
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
        fingerprints[1].contains("REPAIR ROUND 2")
            && fingerprints[1].contains("not found verbatim"),
        "repair instructions + per-group rejection must reach the model: {}",
        fingerprints[1]
    );
}

#[tokio::test]
#[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global seams
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
    let fake = crate::util::test::FakeProvider::new()
        .ok(
            r#"{"summary":"s","groups":[{"heading":"Bad","contradiction":false,"members":[{"agent":0,"text":"Invented"}]}],"ungrouped":[{"agent":0,"text":"No timeout check"},{"agent":1,"text":"Missing retry"}]}"#,
        )
        .ok(
            r#"{"groups":[{"heading":"Bad","contradiction":false,"members":[{"agent":1,"text":"Invented"}]}],"ungrouped":[{"agent":0,"text":"No timeout check"},{"agent":1,"text":"Missing retry"}]}"#,
        );
    let fake = std::sync::Arc::new(fake);
    let _provider = crate::util::test::install_fake_provider(fake);
    let ws = crate::workspace::test_ws("/tmp/test_ws");
    let outcome = run_synthesis(&r, Role::Reviewer, &ws).await;
    assert!(
        matches!(outcome, RepairOutcome::Fallback),
        "zero-progress with zero groups ever frozen must fall back"
    );
}

#[tokio::test]
#[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global seams
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
    let fake = crate::util::test::FakeProvider::new()
        .ok(
            r#"{"summary":"One consensus, one dispute.","groups":[{"heading":"Safety","contradiction":false,"members":[{"agent":0,"text":"Safe"}]}],"ungrouped":[{"agent":1,"text":"Actually unsafe"}]}"#,
        )
        .ok(
            r#"{"groups":[],"ungrouped":[{"agent":1,"text":"Actually unsafe"}],"references":[{"group":0,"member":{"agent":1,"text":"Actually unsafe"}}]}"#,
        );
    let fake = std::sync::Arc::new(fake);
    let _provider = crate::util::test::install_fake_provider(fake);
    let ws = crate::workspace::test_ws("/tmp/test_ws");
    let outcome = run_synthesis(&r, Role::Reviewer, &ws).await;
    let RepairOutcome::Repaired { output, references } = outcome else {
        panic!("reference round must converge, got fallback");
    };
    assert_eq!(references.len(), 1, "contradiction reference accepted");
    let text = render_joint_comment(&r, &RepairOutcome::Repaired { output, references });
    assert!(
        text.contains("Agent 1: Actually unsafe [DISPUTED — contradicts group 0 \"Safety\"]"),
        "reference must render with DISPUTED + cross-ref: {text}"
    );
}

#[tokio::test]
#[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global seams
async fn repair_rejects_empty_member_group() {
    let _lock = crate::util::test::retry_tests_lock();
    let _policy = crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());

    // Round 1 proposes an empty-member group (must NOT freeze as progress —
    // it places nothing and would render a bogus [0/N] bracket) alongside a
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
            r#"{"summary":"s","groups":[{"heading":"Empty","contradiction":false,"members":[]},{"heading":"Safety","contradiction":false,"members":[{"agent":0,"text":"Safe"}]}],"ungrouped":[{"agent":1,"text":"Unsafe"}]}"#,
        )
        .ok(
            r#"{"groups":[],"ungrouped":[{"agent":1,"text":"Unsafe"}]}"#,
        );
    let fake = std::sync::Arc::new(fake);
    let _provider = crate::util::test::install_fake_provider(fake.clone());
    let ws = crate::workspace::test_ws("/tmp/test_ws");
    let outcome = run_synthesis(&r, Role::Reviewer, &ws).await;
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
    // Low churn, no added files → 2.
    assert_eq!(review_base_from_signals(10, 5, 0, 50, 400), 2);
    // High churn → 4.
    assert_eq!(review_base_from_signals(500, 10, 0, 50, 400), 4);
    // Added files → 4 regardless of churn.
    assert_eq!(review_base_from_signals(10, 5, 1, 50, 400), 4);
    // Middle → 3.
    assert_eq!(review_base_from_signals(60, 10, 0, 50, 400), 3);
}

#[test]
fn review_agent_count_adjustments() {
    assert_eq!(review_agent_count(2, 1, false), 2, "first round, no bounce");
    assert_eq!(review_agent_count(2, 1, true), 3, "bounced before gets +1");
    assert_eq!(review_agent_count(3, 1, true), 4, "bounce +1 from 3");
    assert_eq!(review_agent_count(4, 1, true), 4, "capped at 4");
    assert_eq!(review_agent_count(2, 0, false), 3, "P0 never gets 2");
    assert_eq!(
        review_agent_count(2, 0, true),
        3,
        "P0 floor 3 even with bounce"
    );
    assert_eq!(review_agent_count(3, 0, true), 4, "P0 with bounce from 3");
}

#[test]
fn analysis_escalation_trigger() {
    use crate::management::ParallelVerdict;
    let v = |score: u8| {
        ParallelVerdict::Verdict(crate::Verdict {
            score,
            critique: None,
            issues_detected: vec![],
        })
    };

    // All dispatched analysts flagged blockers → escalate.
    assert!(analysis_escalation_needed(&[v(3), v(5), v(6)], 3));
    // Any analyst passing → no escalation.
    assert!(!analysis_escalation_needed(&[v(3), v(7), v(6)], 3));
    assert!(!analysis_escalation_needed(&[v(3), v(10), v(6)], 3));
    // A no-response analyst breaks unanimity → no escalation.
    assert!(!analysis_escalation_needed(
        &[
            v(3),
            ParallelVerdict::NoResponse("no response".into()),
            v(5),
        ],
        3,
    ));
    // Empty / single rounds never escalate (only the base dispatch can).
    assert!(!analysis_escalation_needed(&[], 3));
    assert!(!analysis_escalation_needed(&[v(3)], 3));
    // Dispatched count drives the comparison (not a hard-coded 3).
    assert!(!analysis_escalation_needed(&[v(3), v(5), v(6)], 5));
    assert!(analysis_escalation_needed(
        &[v(3), v(5), v(6), v(4), v(2)],
        5
    ));
}
