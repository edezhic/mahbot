use super::*;
use crate::Verdict;

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

fn issues_by_agent(round: &JointRound<'_>) -> Vec<Vec<String>> {
    // Mirrors run_synthesis: keyed by ORIGINAL dispatch index (empty slots for
    // failed agents), never a compacted 0..n_valid-1 space.
    let mut by_agent: Vec<Vec<String>> = vec![Vec::new(); round.dispatched];
    for v in &round.verdicts {
        by_agent[v.agent_index] = v
            .verdict
            .issues_detected
            .iter()
            .map(|i| normalize_issue_text(i))
            .collect();
    }
    by_agent
}

// ── Deterministic merge + bracket semantics ─────────────────────────────

#[test]
fn merge_and_brackets_follow_spec_semantics() {
    // Three valid verdicts; one issue raised by all three, one by two,
    // one by a single agent.
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check", "Missing retry"])),
            (1, verdict(3, &["No timeout check", "Lock ordering"])),
            (2, verdict(9, &["No timeout check"])),
        ],
        vec![],
        "1 of 3 valid verdicts failed.",
    );
    let issues = merge_issues(&r);
    assert_eq!(issues.len(), 3);
    let bracket = |text: &str| -> String {
        let issue = issues
            .iter()
            .find(|m| normalize_issue_text(&m.text) == text)
            .expect("issue present");
        bracket_label(issue, false)
    };
    assert_eq!(
        bracket("No timeout check"),
        "[3/3]",
        "all valid agents agree"
    );
    assert_eq!(
        bracket("Lock ordering"),
        "DISPUTED",
        "exactly one agent raised it"
    );
    assert_eq!(
        bracket("Missing retry"),
        "DISPUTED",
        "exactly one agent raised it"
    );

    // Two agents raised the same issue.
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["Shared issue"])),
            (1, verdict(3, &["Shared issue"])),
            (2, verdict(9, &["Other"])),
        ],
        vec![],
        "",
    );
    let issues = merge_issues(&r);
    let shared = issues
        .iter()
        .find(|m| m.text == "Shared issue")
        .expect("shared issue");
    assert_eq!(bracket_label(shared, false), "[2/3]");

    // Single-agent round renders [1/1] with the unverified note.
    let r = round("Review", vec![(0, verdict(4, &["Solo issue"]))], vec![], "");
    let issues = merge_issues(&r);
    assert!(
        bracket_label(&issues[0], false).contains("[1/1]"),
        "single-agent bracket"
    );
    assert!(
        bracket_label(&issues[0], false).contains("not cross-checked"),
        "single-agent note"
    );

    // The synthesis contradiction flag forces DISPUTED even for multi-agent
    // agreement.
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["Shared issue"])),
            (1, verdict(3, &["Shared issue"])),
        ],
        vec![],
        "",
    );
    let issues = merge_issues(&r);
    assert_eq!(bracket_label(&issues[0], true), "DISPUTED");
}

#[test]
fn merge_dedupes_within_an_agent() {
    // The same agent listing the same issue twice contributes one vote.
    let r = round(
        "Review",
        vec![(0, verdict(4, &["No timeout check", "No timeout check"]))],
        vec![],
        "",
    );
    let issues = merge_issues(&r);
    assert_eq!(issues.len(), 1, "duplicate issue within an agent collapses");
    assert_eq!(issues[0].agents, vec![0]);
}

// ── Numeric-semantics classifier ────────────────────────────────────────

#[test]
fn numeric_semantics_classifier() {
    // Locators / evidence differences are NOT contradictions.
    assert!(issues_differ_only_in_numeric_details(
        "line 3281",
        "line 6684–6715"
    ));
    assert!(issues_differ_only_in_numeric_details(
        "8-9 lines",
        "8-11 lines"
    ));
    assert!(issues_differ_only_in_numeric_details(
        "config.0.timeout",
        "config.1.timeout"
    ));
    // Identical texts are vacuously numeric-only (never a contradiction).
    assert!(issues_differ_only_in_numeric_details(
        "No timeout check",
        "No timeout check"
    ));
    // Genuine property differences are NOT numeric-only.
    assert!(!issues_differ_only_in_numeric_details(
        "Missing error handling",
        "Missing timeout handling"
    ));
    assert!(!issues_differ_only_in_numeric_details(
        "Race in shutdown path",
        "Deadlock in checkpoint path"
    ));
}

// ── Validator invariants ────────────────────────────────────────────────

fn output(summary: &str, groups: Vec<SynthesisGroup>) -> SynthesisOutput {
    SynthesisOutput {
        summary: summary.to_string(),
        groups,
    }
}

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
    let out = output(
        "The review found timeout handling gaps and a naming nit.",
        vec![SynthesisGroup {
            heading: "Timeouts".into(),
            contradiction: false,
            members: vec![SynthesisMember {
                agent: 0,
                text: "No timeout check".into(),
            }],
        }],
    );
    assert_eq!(
        validate_synthesis_output(&out, &issues_by_agent(&r)),
        Ok(())
    );
}

#[test]
fn validator_rejects_fabricated_consensus() {
    // The LLM claims agent 1 reported an issue it never wrote — the
    // anti-fabrication core: this must never reach a ticket comment.
    let r = round(
        "Review",
        vec![(0, verdict(4, &["No timeout check"]))],
        vec![],
        "",
    );
    let out = output(
        "Both agents agree on the timeout issue.",
        vec![SynthesisGroup {
            heading: "Timeouts".into(),
            contradiction: false,
            members: vec![SynthesisMember {
                agent: 0,
                text: "No timeout check".into(),
            }],
        }],
    );
    // Agent 0's issue is verbatim — accepted even though the summary
    // overstates consensus (the pipeline renders its own brackets).
    assert_eq!(
        validate_synthesis_output(&out, &issues_by_agent(&r)),
        Ok(())
    );

    // Fabricated text (not in agent 0's list) is rejected.
    let out = output(
        "summary",
        vec![SynthesisGroup {
            heading: "Timeouts".into(),
            contradiction: false,
            members: vec![SynthesisMember {
                agent: 0,
                text: "No timeout check AND missing retry".into(),
            }],
        }],
    );
    let err = validate_synthesis_output(&out, &issues_by_agent(&r)).expect_err("must reject");
    assert!(err.contains("not found verbatim"), "err: {err}");

    // Unknown agent index is rejected.
    let out = output(
        "summary",
        vec![SynthesisGroup {
            heading: "Timeouts".into(),
            contradiction: false,
            members: vec![SynthesisMember {
                agent: 7,
                text: "No timeout check".into(),
            }],
        }],
    );
    let err = validate_synthesis_output(&out, &issues_by_agent(&r)).expect_err("must reject");
    assert!(err.contains("unknown agent index"), "err: {err}");
}

#[test]
fn validator_rejects_double_counted_issue() {
    let r = round(
        "Review",
        vec![(0, verdict(4, &["No timeout check"]))],
        vec![],
        "",
    );
    let out = output(
        "summary",
        vec![
            SynthesisGroup {
                heading: "A".into(),
                contradiction: false,
                members: vec![SynthesisMember {
                    agent: 0,
                    text: "No timeout check".into(),
                }],
            },
            SynthesisGroup {
                heading: "B".into(),
                contradiction: false,
                members: vec![SynthesisMember {
                    agent: 0,
                    text: "No timeout check".into(),
                }],
            },
        ],
    );
    let err = validate_synthesis_output(&out, &issues_by_agent(&r)).expect_err("must reject");
    assert!(err.contains("double-counts"), "err: {err}");

    // The same merged issue cited via DIFFERENT agents' copies in two groups
    // must also be rejected — the double-count check is issue-level (keyed on
    // normalized text), not per (agent, text) pair.
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check"])),
            (1, verdict(3, &["No timeout check"])),
        ],
        vec![],
        "",
    );
    let out = output(
        "summary",
        vec![
            SynthesisGroup {
                heading: "A".into(),
                contradiction: false,
                members: vec![SynthesisMember {
                    agent: 0,
                    text: "No timeout check".into(),
                }],
            },
            SynthesisGroup {
                heading: "B".into(),
                contradiction: false,
                members: vec![SynthesisMember {
                    agent: 1,
                    text: "No timeout check".into(),
                }],
            },
        ],
    );
    let err = validate_synthesis_output(&out, &issues_by_agent(&r)).expect_err("must reject");
    assert!(err.contains("double-counts"), "err: {err}");
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
        vec![SynthesisGroup {
            heading: "Robustness".into(),
            contradiction: false,
            members: vec![
                SynthesisMember {
                    agent: 0,
                    text: "No timeout check".into(),
                },
                SynthesisMember {
                    agent: 2,
                    text: "Missing retry".into(),
                },
            ],
        }],
    );
    assert_eq!(
        validate_synthesis_output(&out, &issues_by_agent(&r)),
        Ok(()),
        "original dispatch indices must pass validation"
    );
    let text = render_joint_comment(&r, &SynthesisOutcome::Grouped(out));
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
fn validator_rejects_count_literals() {
    let r = round(
        "Review",
        vec![(0, verdict(4, &["[2/3] tests fail", "No timeout check"]))],
        vec![],
        "",
    );
    // A bracket count inside a MEMBER's text is exempt: it is a verbatim copy
    // of the agent's issue (the agent wrote "[2/3] tests fail") — the
    // membership check is authoritative for member text, not the prose rule.
    let out = output(
        "summary",
        vec![SynthesisGroup {
            heading: "A".into(),
            contradiction: false,
            members: vec![SynthesisMember {
                agent: 0,
                text: "[2/3] tests fail".into(),
            }],
        }],
    );
    assert_eq!(
        validate_synthesis_output(&out, &issues_by_agent(&r)),
        Ok(()),
        "verbatim member text containing a bracket count must pass"
    );

    // A bracket count in the LLM's own heading is a count position.
    let out = output(
        "summary",
        vec![SynthesisGroup {
            heading: "2/3 tests fail".into(),
            contradiction: false,
            members: vec![SynthesisMember {
                agent: 0,
                text: "No timeout check".into(),
            }],
        }],
    );
    let err = validate_synthesis_output(&out, &issues_by_agent(&r)).expect_err("must reject");
    assert!(err.contains("contains a number"), "err: {err}");

    // Standalone prose numbers in the summary ("2 of 3", "4/10", "3 files",
    // "score 8") are rejected — the prompt forbids ALL numbers there.
    for bad_summary in [
        "2 of 3 agents flagged it",
        "4/10 score",
        "40% of issues",
        "3 files affected",
        "score 8",
    ] {
        let out = output(
            bad_summary,
            vec![SynthesisGroup {
                heading: "A".into(),
                contradiction: false,
                members: vec![SynthesisMember {
                    agent: 0,
                    text: "No timeout check".into(),
                }],
            }],
        );
        let err = validate_synthesis_output(&out, &issues_by_agent(&r))
            .expect_err("must reject prose number");
        assert!(err.contains("contains a number"), "err: {err}");
    }

    // The summary check runs even when NO groups were produced.
    let out = output("All 3 of 3 agents passed.", vec![]);
    let err = validate_synthesis_output(&out, &issues_by_agent(&r)).expect_err("must reject");
    assert!(err.contains("contains a number"), "err: {err}");
}

#[test]
fn normalize_issue_text_collapses_whitespace() {
    assert_eq!(
        normalize_issue_text("  No   timeout\ncheck  "),
        "No timeout check"
    );
    assert_eq!(normalize_issue_text("a\tb c"), "a b c");
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
    let text = render_joint_comment(&r, &SynthesisOutcome::Fallback);
    assert!(text.contains("## Review round — 2/3 valid verdicts"));
    assert!(
        text.contains("DISPUTED [blocker] No timeout check"),
        "single-agent issue renders DISPUTED with blocker severity: {text}"
    );
    assert!(text.contains("### Agent failures"));
    assert!(text.contains("Agent 3: agent produced no response — crashed"));
    assert!(text.contains("LLM grouping unavailable"), "fallback marker");
    assert!(
        text.contains("[blocker]"),
        "failing-source issue renders as blocker"
    );
}

#[test]
fn renderer_with_synthesis_groups_and_contradiction_veto() {
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
        vec![SynthesisGroup {
            heading: "Robustness".into(),
            contradiction: true,
            members: vec![
                SynthesisMember {
                    agent: 0,
                    text: "No timeout check".into(),
                },
                SynthesisMember {
                    agent: 1,
                    text: "Missing error handling".into(),
                },
            ],
        }],
    );
    let text = render_joint_comment(&r, &SynthesisOutcome::Grouped(out));
    assert!(text.contains("**Robustness**"));
    assert!(
        text.contains("DISPUTED [blocker] No timeout check"),
        "contradiction flag forces DISPUTED: {text}"
    );
    assert!(text.contains("DISPUTED [blocker] Missing error handling"));
    assert!(text.contains("### Summary"));
    assert!(text.contains("The review found two distinct issues."));

    // Numeric-only differences veto the contradiction flag: two agents wrote
    // the SAME locator (merged to [2/3]) and one wrote a different locator.
    // Because the difference is purely numeric (locators/evidence), the
    // group's contradiction flag must NOT demote the [2/3] agreement.
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["Bug at line 3281"])),
            (1, verdict(3, &["Bug at line 3281"])),
            (2, verdict(5, &["Bug at line 6684–6715"])),
        ],
        vec![],
        "",
    );
    let out = output(
        "The same bug appears at different line ranges.",
        vec![SynthesisGroup {
            heading: "Locator".into(),
            contradiction: true,
            members: vec![
                SynthesisMember {
                    agent: 0,
                    text: "Bug at line 3281".into(),
                },
                SynthesisMember {
                    agent: 2,
                    text: "Bug at line 6684–6715".into(),
                },
            ],
        }],
    );
    let text = render_joint_comment(&r, &SynthesisOutcome::Grouped(out));
    assert!(
        text.contains("[2/3] [blocker] Bug at line 3281"),
        "numeric-only difference must NOT demote the [2/3] agreement: {text}"
    );
    assert!(
        text.contains("Bug at line 6684–6715"),
        "both members render"
    );
}

// ── Calibrated dynamic agent counts ─────────────────────────────────────

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

#[test]
fn review_count_formula() {
    let low = DEFAULT_REVIEW_COUNT_LOW_CHURN as i64;
    let high = DEFAULT_REVIEW_COUNT_HIGH_CHURN as i64;

    // 2 reviewers for low churn with zero added files.
    assert_eq!(
        review_base_from_signals(10, 10, 0, low, high),
        2,
        "low total churn"
    );
    assert_eq!(
        review_base_from_signals(49, 49, 0, low, high),
        2,
        "at the low boundary"
    );
    // 4 for high churn OR any added file.
    assert_eq!(
        review_base_from_signals(400, 10, 0, low, high),
        4,
        "total at high"
    );
    assert_eq!(
        review_base_from_signals(10, 400, 0, low, high),
        4,
        "per-file at high"
    );
    assert_eq!(
        review_base_from_signals(10, 10, 1, low, high),
        4,
        "any added file"
    );
    // 3 otherwise.
    assert_eq!(
        review_base_from_signals(100, 100, 0, low, high),
        3,
        "middle band"
    );
    assert_eq!(
        review_base_from_signals(50, 50, 0, low, high),
        3,
        "at the low boundary is not <"
    );

    // Adjustments: bounce +1 capped at 4; P0 floor 3; never 1.
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
    let request = synthesis_request(&r, Role::Reviewer);
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
        "scores are not part of the synthesis input (the LLM never produces numbers): {user}"
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
    // schema — this output must be accepted end-to-end.
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
        r#"{"summary":"Two distinct issues.","groups":[{"heading":"Robustness","contradiction":false,"members":[{"agent":0,"text":"No timeout check"},{"agent":1,"text":"Missing retry"}]}]}"#,
    );
    let _provider = crate::util::test::install_fake_provider(std::sync::Arc::new(provider));
    let outcome = run_synthesis(&r, Role::Reviewer).await;
    match outcome {
        SynthesisOutcome::Grouped(out) => assert_eq!(out.groups.len(), 1),
        SynthesisOutcome::Fallback => panic!("valid zero-based output must not fall back"),
    }

    // A 1-based index (agent 2 in a 2-agent round — the pre-fix prompt
    // contract) must be rejected and eventually fall back.
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
        r#"{"summary":"Two distinct issues.","groups":[{"heading":"Robustness","contradiction":false,"members":[{"agent":2,"text":"No timeout check"}]}]}"#,
    ));
    let _provider = crate::util::test::install_fake_provider(fake.clone());
    let outcome = run_synthesis(&r, Role::Reviewer).await;
    assert!(
        matches!(outcome, SynthesisOutcome::Fallback),
        "out-of-range 1-based index must exhaust synthesis into the fallback"
    );
    // The rejection must be fed back into the next attempt so the LLM can
    // self-correct (never a byte-identical resend after a validation error).
    let fingerprints = fake.request_fingerprints.lock().unwrap().clone();
    assert!(
        fingerprints.len() >= 2 && fingerprints[1].contains("previous response was rejected"),
        "validation rejection must be fed back into the retry: {fingerprints:?}"
    );
}

#[tokio::test]
#[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global seams
async fn run_synthesis_mid_round_failure_keeps_original_indices() {
    let _lock = crate::util::test::retry_tests_lock();
    let _policy = crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());

    // Agent 1 failed mid-round. The input labels the survivors by their
    // ORIGINAL dispatch positions ("Agent 0"/"Agent 2") — a faithful LLM
    // copying those labels must be accepted end-to-end and render both.
    let r = round(
        "Review",
        vec![
            (0, verdict(4, &["No timeout check"])),
            (2, verdict(3, &["Missing retry"])),
        ],
        vec![(1, "agent crashed")],
        "",
    );
    let provider = crate::util::test::FakeProvider::new().ok(
        r#"{"summary":"Two distinct issues.","groups":[{"heading":"Robustness","contradiction":false,"members":[{"agent":0,"text":"No timeout check"},{"agent":2,"text":"Missing retry"}]}]}"#,
    );
    let _provider = crate::util::test::install_fake_provider(std::sync::Arc::new(provider));
    let outcome = run_synthesis(&r, Role::Reviewer).await;
    match outcome {
        SynthesisOutcome::Grouped(out) => assert_eq!(out.groups.len(), 1),
        SynthesisOutcome::Fallback => {
            panic!("mid-round-failure output with original indices must not fall back")
        }
    }
}
