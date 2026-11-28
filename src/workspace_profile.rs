//! The per-workspace profile: who staffs each stage, the round's pass threshold and whether every
//! project check must pass.
//!
//! The profile is a workspace setting, stored in the workspace's own row: `workspaces.profile`
//! holds only what the owner customised and `workspaces.profile_confirmed` records that the owner
//! accepted the workspace's settings. An absent field follows the shipped default, so a later
//! change to that default reaches every workspace that did not pin the field; editing a field pins
//! that field alone. The check list is not stored here: it keeps its own column
//! (`workspaces.diagnostics`) and its own writers.
//!
//! The stage vocabulary is the core's ([`TicketPhase`]): only the five pipeline stages have a
//! profile, and the order of the stage graph, the bounce cap, the reviewer-count calibration, the
//! analysis escalation and the sanitation taxonomy stay in the stage code. A profile that names an
//! unknown participant, a participant outside its stage, the same participant twice in one stage,
//! or leaves a stage with nobody to do its work is invalid. An unconfirmed profile stops the
//! pipeline; a confirmed one this build cannot read never does — the workspace runs on the default
//! and the owner is told.

use crate::Workspace;
use crate::pipeline::board::TicketPhase;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// Verification's passing score when the profile does not pin one.
const DEFAULT_PASS_THRESHOLD: u8 = 9;

/// Whether every project command must pass when the profile does not pin it.
const DEFAULT_REQUIRE_ALL_CHECKS: bool = true;

/// The most a stored profile may occupy. The poll round reads it per workspace every tick, and a
/// profile is a handful of names, so anything this large is not one.
const MAX_STORED_PROFILE_BYTES: usize = 4_096;

/// The five profiled stages in pipeline order: the stage, who may staff it, and who it cannot run
/// without. The stage graph and the conditions under which a group joins stay in the stage code.
const STAGES: [(TicketPhase, &[Participant], &[Participant]); 5] = [
    (
        TicketPhase::Analysis,
        &[Participant::Analyst],
        &[Participant::Analyst],
    ),
    (
        TicketPhase::Planning,
        &[Participant::Manager],
        &[Participant::Manager],
    ),
    (
        TicketPhase::InDevelopment,
        &[Participant::Engineer],
        &[Participant::Engineer],
    ),
    (
        TicketPhase::Verification,
        &[Participant::Reviewer, Participant::Tester],
        &[Participant::Tester],
    ),
    (
        TicketPhase::InSanitation,
        &[Participant::Sanitation],
        &[Participant::Sanitation],
    ),
];

/// One kind of participant a stage can be staffed with. A profile names these; a stage's own logic
/// decides what the name means for it (the verification round reads [`Participant::Reviewer`] and
/// [`Participant::Tester`], and the other stages carry their single participant as the invariant
/// their work cannot do without).
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::Display, strum::EnumString)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum Participant {
    Analyst,
    Manager,
    Engineer,
    Reviewer,
    Tester,
    Sanitation,
}

impl Participant {
    #[must_use]
    fn parse(name: &str) -> Option<Self> {
        name.parse().ok()
    }
}

/// Why a stored profile cannot be used: it is not the shape this build reads, or its roster or
/// pass conditions are not legal. Either way a confirmed workspace keeps running on the default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProfileError(String);

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// What the owner customised, as stored. Every field is optional: an absent field follows the
/// default, which is what lets a later shipped default reach an unpinned workspace.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub(crate) struct StoredProfile {
    /// Per-stage participant lists, keyed by stage token. An absent stage follows the default.
    pub stages: BTreeMap<String, Vec<String>>,
    /// Verification's passing score, 0-10.
    pub pass_threshold: Option<u8>,
    /// Whether every project command must pass.
    pub require_all_checks: Option<bool>,
}

impl StoredProfile {
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.stages.is_empty() && self.pass_threshold.is_none() && self.require_all_checks.is_none()
    }

    pub(crate) fn parse(json: &str) -> Result<Self, ProfileError> {
        if json.len() > MAX_STORED_PROFILE_BYTES {
            return Err(ProfileError(format!(
                "a profile may occupy {MAX_STORED_PROFILE_BYTES} bytes, this one is {}",
                json.len()
            )));
        }
        serde_json::from_str(json).map_err(|e| ProfileError(e.to_string()))
    }

    #[must_use]
    pub(crate) fn to_json(&self) -> String {
        serde_json::to_string(self).expect("a stored profile is serializable")
    }

    /// Check the profile against this build's participant set and the stages' invariants.
    pub(crate) fn validate(&self) -> Result<(), ProfileError> {
        self.rosters().map(|_| ())
    }

    /// The stored rosters, parsed and checked in ONE pass — `None` for a stage the profile does
    /// not pin, so the shipped default stands for it. This is the single place a stored name is
    /// read: an unknown participant, one outside its stage, the same participant twice in one
    /// stage, or a stage left with nobody to do its work is refused here.
    fn rosters(&self) -> Result<[Option<Vec<Participant>>; STAGES.len()], ProfileError> {
        if let Some(threshold) = self.pass_threshold
            && threshold > 10
        {
            return Err(ProfileError(format!(
                "the pass threshold {threshold} is outside 0-10"
            )));
        }
        let mut rosters: [Option<Vec<Participant>>; STAGES.len()] = Default::default();
        for (slot, (stage, allowed, required)) in STAGES.into_iter().enumerate() {
            let Some(names) = self.stages.get(stage.as_ref()) else {
                continue;
            };
            if names.is_empty() {
                return Err(ProfileError(format!(
                    "the {} stage has nobody to do its work",
                    stage.as_ref()
                )));
            }
            let mut participants = Vec::with_capacity(names.len());
            for name in names {
                let participant = Participant::parse(name).ok_or_else(|| {
                    ProfileError(format!("'{name}' is not a participant this build knows"))
                })?;
                if !allowed.contains(&participant) {
                    return Err(ProfileError(format!(
                        "'{name}' does not do the {} stage's work",
                        stage.as_ref()
                    )));
                }
                // A roster names people to start: the same one twice would start the same work
                // twice and read as two participants wherever the profile is rendered.
                if participants.contains(&participant) {
                    return Err(ProfileError(format!(
                        "the {} stage names the {participant} twice",
                        stage.as_ref()
                    )));
                }
                participants.push(participant);
            }
            for participant in required {
                if !participants.contains(participant) {
                    return Err(ProfileError(format!(
                        "the {} stage cannot run without the {participant}",
                        stage.as_ref()
                    )));
                }
            }
            rosters[slot] = Some(participants);
        }
        if let Some(stage) = self.stages.keys().find(|name| {
            !STAGES
                .iter()
                .any(|(stage, ..)| stage.as_ref() == name.as_str())
        }) {
            return Err(ProfileError(format!(
                "'{stage}' is not a stage this build knows"
            )));
        }
        Ok(rosters)
    }
}

/// The effective profile a stage round reads: every field resolved, defaults included. It is
/// serialized through [`StoredProfile`]'s one shape — stages keyed by their token — so a round's
/// record and a workspace's own setting are the same document, and reordering [`STAGES`] in a later
/// build cannot silently move a roster onto another stage (see [`crate::jobs::set_round_profile`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Profile {
    /// One participant list per [`STAGES`] slot, in that order.
    stages: [Vec<Participant>; STAGES.len()],
    pub pass_threshold: u8,
    pub require_all_checks: bool,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            stages: STAGES.map(|(_, allowed, _)| allowed.to_vec()),
            pass_threshold: DEFAULT_PASS_THRESHOLD,
            require_all_checks: DEFAULT_REQUIRE_ALL_CHECKS,
        }
    }
}

impl Profile {
    /// Resolve a stored profile over the default. `None` (or an empty stored value) is the default
    /// profile; an unreadable or invalid one is the error the caller reports and runs past.
    pub(crate) fn resolve(stored: Option<&str>) -> Result<Self, ProfileError> {
        let Some(stored) = stored.map(str::trim).filter(|json| !json.is_empty()) else {
            return Ok(Self::default());
        };
        Self::over(&StoredProfile::parse(stored)?)
    }

    /// The default profile with the stored profile's own fields pinned — one resolution, shared by
    /// a workspace's setting and a round's record.
    fn over(stored: &StoredProfile) -> Result<Self, ProfileError> {
        let mut profile = Self::default();
        for (slot, roster) in stored.rosters()?.into_iter().enumerate() {
            if let Some(participants) = roster {
                profile.stages[slot] = participants;
            }
        }
        if let Some(threshold) = stored.pass_threshold {
            profile.pass_threshold = threshold;
        }
        if let Some(require_all) = stored.require_all_checks {
            profile.require_all_checks = require_all;
        }
        Ok(profile)
    }

    /// The profile as a stored document: every field pinned, so reading it back needs no defaults
    /// and cannot follow a shipped default that changed mid-round.
    fn stored(&self) -> StoredProfile {
        StoredProfile {
            stages: STAGES
                .iter()
                .enumerate()
                .map(|(slot, (stage, ..))| {
                    (
                        stage.as_ref().to_string(),
                        self.stages[slot].iter().map(ToString::to_string).collect(),
                    )
                })
                .collect(),
            pass_threshold: Some(self.pass_threshold),
            require_all_checks: Some(self.require_all_checks),
        }
    }

    #[must_use]
    pub(crate) fn participants(&self, stage: TicketPhase) -> &[Participant] {
        // Only the five profiled stages have a roster; a phase outside them (Backlog, Queued, the
        // terminal ones) has nobody to dispatch, and an empty list says exactly that.
        let Some(slot) = STAGES
            .iter()
            .position(|(candidate, ..)| *candidate == stage)
        else {
            return &[];
        };
        &self.stages[slot]
    }

    #[must_use]
    pub(crate) fn includes(&self, stage: TicketPhase, participant: Participant) -> bool {
        self.participants(stage).contains(&participant)
    }

    /// The profile as a round records it — see [`Self::stored`].
    #[must_use]
    pub(crate) fn to_json(&self) -> String {
        self.stored().to_json()
    }

    /// A profile a round recorded, read back through the same shape a workspace setting uses.
    pub(crate) fn from_json(json: &str) -> Result<Self, ProfileError> {
        Self::over(&StoredProfile::parse(json)?)
    }
}

/// The profiled stage a token names, when it is one this build knows.
#[must_use]
pub(crate) fn stage_named(token: &str) -> Option<TicketPhase> {
    STAGES
        .iter()
        .map(|(stage, ..)| *stage)
        .find(|stage| stage.as_ref() == token.trim())
}

/// What a workspace's profile row says. The pipeline waits on [`Self::Unconfirmed`] and runs past
/// [`Self::Invalid`] on the default profile. The state is a verdict only: the profile in force is
/// [`ProfileRead::profile`], read once, so a state never carries a second copy of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ProfileState {
    /// Not confirmed yet: the profile is what confirmation would put in force.
    Unconfirmed,
    Invalid(ProfileError),
    Confirmed,
}

/// A workspace's profile row, read once: the profile in force and the state that qualifies it.
/// Callers that need both take one read instead of parsing the stored JSON per question.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProfileRead {
    pub profile: Profile,
    pub state: ProfileState,
}

#[must_use]
pub(crate) fn read(ws: &Workspace) -> ProfileRead {
    if !ws.profile_confirmed {
        let profile = Profile::resolve(ws.profile.as_deref()).unwrap_or_default();
        return ProfileRead {
            state: ProfileState::Unconfirmed,
            profile,
        };
    }
    match Profile::resolve(ws.profile.as_deref()) {
        Ok(profile) => ProfileRead {
            state: ProfileState::Confirmed,
            profile,
        },
        Err(error) => ProfileRead {
            state: ProfileState::Invalid(error),
            profile: Profile::default(),
        },
    }
}

#[must_use]
pub(crate) fn state(ws: &Workspace) -> ProfileState {
    read(ws).state
}

/// Render the effective profile for a read-back: one line per stage, then the pass conditions.
#[must_use]
pub(crate) fn render(profile: &Profile) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for (stage, ..) in STAGES {
        let participants = profile
            .participants(stage)
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "- {}: {participants}", stage.as_ref());
    }
    let _ = write!(
        out,
        "- pass threshold: {}/10\n- every check must pass: {}",
        profile.pass_threshold,
        if profile.require_all_checks {
            "yes"
        } else {
            "no"
        }
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws(profile: Option<&str>, confirmed: bool) -> Workspace {
        Workspace {
            profile: profile.map(str::to_string),
            profile_confirmed: confirmed,
            ..Workspace::default()
        }
    }

    #[test]
    fn the_default_profile_is_todays_software_settings() {
        let profile = Profile::default();
        assert_eq!(
            profile.participants(TicketPhase::Analysis),
            [Participant::Analyst]
        );
        assert_eq!(
            profile.participants(TicketPhase::Verification),
            [Participant::Reviewer, Participant::Tester]
        );
        assert_eq!(profile.pass_threshold, DEFAULT_PASS_THRESHOLD);
        assert!(profile.require_all_checks);
    }

    #[test]
    fn an_unedited_field_follows_the_default_while_an_edited_one_is_pinned() {
        let profile = Profile::resolve(Some(r#"{"pass_threshold":7}"#)).expect("a legal profile");
        assert_eq!(profile.pass_threshold, 7);
        assert!(profile.require_all_checks);
        assert_eq!(
            profile.participants(TicketPhase::InDevelopment),
            [Participant::Engineer]
        );
    }

    /// A round records the profile it started on and reads it back unchanged: every field is
    /// resolved in the record, so the round cannot follow a default that moved mid-round.
    #[test]
    fn a_recorded_profile_reads_back_as_the_round_started_with_it() {
        let profile = Profile::resolve(Some(
            r#"{"stages":{"verification":["tester"]},"pass_threshold":6}"#,
        ))
        .expect("a legal profile");
        let recorded = profile.to_json();
        assert_eq!(
            Profile::from_json(&recorded).expect("the round's own record reads back"),
            profile
        );
        let oversized = " ".repeat(MAX_STORED_PROFILE_BYTES + 1);
        assert!(
            Profile::from_json(&oversized).is_err(),
            "a record larger than a profile may be is refused, not parsed"
        );
    }

    #[test]
    fn verification_may_drop_the_reviewers_but_never_the_tester() {
        let ok = Profile::resolve(Some(r#"{"stages":{"verification":["tester"]}}"#))
            .expect("tester-only verification");
        assert!(!ok.includes(TicketPhase::Verification, Participant::Reviewer));

        assert!(
            Profile::resolve(Some(r#"{"stages":{"verification":["reviewer"]}}"#)).is_err(),
            "a round with no tester is invalid"
        );
    }

    #[test]
    fn unknown_names_stages_and_out_of_range_thresholds_are_invalid() {
        for stored in [
            r#"{"stages":{"analysis":["reviewer"]}}"#,
            r#"{"stages":{"deployment":["engineer"]}}"#,
            r#"{"stages":{"analysis":["wizard"]}}"#,
            r#"{"stages":{"analysis":[]}}"#,
            r#"{"pass_threshold":11}"#,
            r#"{"surprise":true}"#,
        ] {
            assert!(
                Profile::resolve(Some(stored)).is_err(),
                "{stored} must be refused"
            );
        }
    }

    #[test]
    fn a_participant_named_twice_in_one_stage_is_invalid() {
        let error = Profile::resolve(Some(r#"{"stages":{"verification":["tester","tester"]}}"#))
            .expect_err("a stage naming the tester twice is invalid");
        assert!(error.to_string().contains("twice"), "got: {error}");
    }

    #[test]
    fn a_stored_profile_too_large_to_be_one_is_refused() {
        let oversized = format!(
            r#"{{"stages":{{}}{}}}"#,
            " ".repeat(MAX_STORED_PROFILE_BYTES)
        );
        assert!(Profile::resolve(Some(&oversized)).is_err());
    }

    #[test]
    fn a_participant_round_trips_through_its_stored_token() {
        for participant in [
            Participant::Analyst,
            Participant::Manager,
            Participant::Engineer,
            Participant::Reviewer,
            Participant::Tester,
            Participant::Sanitation,
        ] {
            assert_eq!(
                Participant::parse(&participant.to_string()),
                Some(participant)
            );
        }
        assert_eq!(Participant::parse("wizard"), None);
    }

    #[test]
    fn an_unconfirmed_workspace_waits_and_an_invalid_one_runs_on_the_default() {
        assert!(matches!(state(&ws(None, false)), ProfileState::Unconfirmed));

        let invalid = read(&ws(Some("not json"), true));
        assert!(matches!(invalid.state, ProfileState::Invalid(_)));
        assert_eq!(invalid.profile, Profile::default());

        let confirmed = state(&ws(None, true));
        assert!(matches!(confirmed, ProfileState::Confirmed));
    }

    #[test]
    fn an_unconfirmed_workspace_shows_what_confirmation_would_put_in_force() {
        let read = read(&ws(Some(r#"{"pass_threshold":6}"#), false));
        assert!(matches!(read.state, ProfileState::Unconfirmed));
        assert_eq!(read.profile.pass_threshold, 6);
    }
}
