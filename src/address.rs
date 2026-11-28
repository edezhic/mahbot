//! One addressing: which agent the product is talking to, spelled once.
//!
//! Every agent id the product produces — a direct chat, the owner's Assistant, a workspace
//! Manager, a ticket round and its parallel roster slots, the Maintainer, an analyze round and its
//! analyst and verification slots, a research run, its orchestrator and its verification analysts, a
//! discovery or cleanup round, a custom role's continuable branch or one-off run, a worker branch —
//! is an [`AgentAddress`] rendered by [`AgentAddress::id`]. Agent ids are session primary keys, so
//! the spellings here are frozen: a changed one orphans every stored conversation.
//!
//! What this module is not: a registry of live agents. It answers *which* agent, never whether one
//! is running ([`crate::agent::registry`]) or what it is doing. [`AgentAddress::parse`] is the
//! inverse only where a stored id decides its parts exactly — a transient round's generated suffix
//! is ambiguous with the segments around it, so a parsed transient address answers the *kind*,
//! never the parts, and must never be rendered back. A direct id is split the same way: its leading
//! user segment and its workspace are both free-form, so the split between them is exact only when
//! the user name carries no `_`, and no caller reads those parts back either.
//!
//! The invariant: the prefix families the SQL layers match (`LIKE 'prefix_%'` — the transient TTL
//! purge, the dead-session filter, the user-name escaper) are projections of this enum's kinds, so
//! a kind cannot be transient here and permanent there.

use std::borrow::Cow;

/// A direct chat's address: `{user}_{ws_name}_{role}`, channel-agnostic — one conversation per
/// (user + workspace + role) whatever the originating channel. Role last for consistent
/// identification in logs and debugging, and immune to underscores in the other segments; the
/// router no longer parses these strings (the role is carried by
/// [`AgentJob`](crate::agent::message_router::AgentJob)). Stable across messages, so the
/// conversation accumulates in a single session.
///
/// The user's OWN personal workspace (`personal:{user}`) drops the duplicated user segment,
/// giving `{user}_personal:{role}` — keyed strictly on the `personal:` prefix AND the embedded
/// user matching the routed raw user, so a project named `personal_work` or another user's
/// personal workspace keeps the full form.
///
/// A real user whose name collides with a reserved agent-id namespace is escaped with a `user_`
/// prefix ([`safe_user_segment`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AgentAddress {
    Direct {
        user: String,
        role: String,
        workspace: String,
    },
    /// The owner's Assistant session — [`Self::Direct`] in the user's own personal workspace,
    /// named as its own kind because automation arming and notices address it without knowing
    /// either segment.
    Assistant { user: String },
    /// A workspace Manager: `manager_{ws_name}`, workspace-scoped and user-facing.
    Manager { workspace: String },
    /// Ticket-driven agent work: `ticket_{ticket_id}_{role}`. A parallel round's roster slot folds
    /// its `_{index}_{suffix}` into the ticket key — `ticket_{ticket_id}_{index}_{suffix}_{role}`,
    /// exactly the key [`Self::parse`] yields for such an id.
    Ticket { ticket: String, role: String },
    /// The Maintainer of a workspace: `maintainer_{ws_name}`, deterministic so a run interrupted
    /// by a restart resumes instead of starting over.
    Maintainer { workspace: String },
    /// A sub-agent analyze round: `analyze_{ws_name}_{suffix}_{role}`, role last. A parallel
    /// round's analyst slot folds its `_{index}` into the suffix —
    /// `analyze_{ws_name}_{suffix}_{index}_{role}`.
    Analyze {
        workspace: String,
        suffix: String,
        role: String,
    },
    /// A deep-research sub-agent (decomposers, round-1 researchers, gap-round analysts):
    /// `research_{ws_name}_{suffix}_{label}`.
    Research {
        workspace: String,
        suffix: String,
        label: String,
    },
    /// An analyze round's verification analyst: `analyze_{ws_name}_verify_{suffix}_{index}`. Its
    /// own family because the round's `_verify` marker is part of the key, not a segment a parse
    /// could mistake for a workspace.
    AnalyzeVerifier { workspace: String, suffix: String },
    /// A research run's verification analyst: `research_{ws_name}_verify_{suffix}_{index}`.
    ResearchVerifier { workspace: String, suffix: String },
    /// The research run's orchestrator, which speaks for the run rather than for one member:
    /// `research_{ws_name}_orchestrator` — two segments, not [`Self::Research`]'s three.
    ResearchOrchestrator { workspace: String },
    /// Workspace role discovery: `discovery_{ws_name}_{suffix}_{role}`, role last.
    Discovery {
        workspace: String,
        suffix: String,
        role: String,
    },
    /// A sanitation cleanup round: `cleanup_{job_id}`. Shared by the research run's cleanup and the
    /// periodic temp cleaner (`crate::temp`), so the transient-prefix invariant covers both.
    Cleanup { job_id: String },
    /// A custom role's continuable branch: `role_{name}`, or `role_{name}:{identifier}` for a named
    /// one ([`CUSTOM_ROLE_AGENT_ID_PREFIX`]). Deterministic, so the next message on the branch
    /// resumes the same session, which persists like a direct conversation.
    CustomRole {
        name: String,
        identifier: Option<String>,
    },
    /// A custom role's ONE-OFF run: `rolerun_{name}:{suffix}` — fresh per launch, transient
    /// ([`CUSTOM_ROLE_RUN_AGENT_ID_PREFIX`]).
    CustomRoleRun { name: String, suffix: String },
    /// A worker branch: `worker_{branch}` — deterministic, so the next message on the branch
    /// resumes the same session ([`WORKER_BRANCH_AGENT_ID_PREFIX`]).
    WorkerBranch { branch: String },
}

impl AgentAddress {
    /// The owner's Assistant in `user`'s personal workspace.
    #[must_use]
    pub(crate) fn assistant(user: impl Into<String>) -> Self {
        Self::Assistant { user: user.into() }
    }

    /// A workspace's Manager.
    #[must_use]
    pub(crate) fn manager(workspace: impl Into<String>) -> Self {
        Self::Manager {
            workspace: workspace.into(),
        }
    }

    /// A direct chat: `user` in `role` at `workspace` — the one path to a [`Self::Direct`].
    #[must_use]
    pub(crate) fn direct(
        user: impl Into<String>,
        role: impl Into<String>,
        workspace: impl Into<String>,
    ) -> Self {
        Self::Direct {
            user: user.into(),
            role: role.into(),
            workspace: workspace.into(),
        }
    }

    /// Ticket-driven agent work: `ticket`'s `role` round.
    #[must_use]
    pub(crate) fn ticket(ticket: impl Into<String>, role: impl Into<String>) -> Self {
        Self::Ticket {
            ticket: ticket.into(),
            role: role.into(),
        }
    }

    /// The Maintainer of a workspace.
    #[must_use]
    pub(crate) fn maintainer(workspace: impl Into<String>) -> Self {
        Self::Maintainer {
            workspace: workspace.into(),
        }
    }

    /// A sanitation cleanup round.
    #[must_use]
    pub(crate) fn cleanup(job_id: impl Into<String>) -> Self {
        Self::Cleanup {
            job_id: job_id.into(),
        }
    }

    /// A sub-agent analyze round. The suffix is generated here, at construction: a transient
    /// address is a fresh one per run, never resumed.
    #[must_use]
    pub(crate) fn analyze(workspace: &str, role: &str) -> Self {
        Self::Analyze {
            workspace: workspace.to_string(),
            suffix: crate::generate_suffix(),
            role: role.to_string(),
        }
    }

    /// A parallel analyze round's analyst slot: `analyze_{ws_name}_{suffix}_{index}_{role}`. The
    /// round's own suffix arrives from the caller (a resumed round keeps it) with the slot index
    /// folded in, so [`Self::Analyze`] renders the slot's frozen spelling unchanged.
    #[must_use]
    pub(crate) fn analyze_slot(workspace: &str, suffix: &str, index: usize, role: &str) -> Self {
        Self::Analyze {
            workspace: workspace.to_string(),
            suffix: format!("{suffix}_{index}"),
            role: role.to_string(),
        }
    }

    /// A deep-research sub-agent round, suffixed the same way.
    #[must_use]
    pub(crate) fn research(workspace: &str, label: &str) -> Self {
        Self::Research {
            workspace: workspace.to_string(),
            suffix: crate::generate_suffix(),
            label: label.to_string(),
        }
    }

    /// An analyze round's verification analyst: the round's own suffix arrives from the caller with
    /// the slot index folded in, exactly as [`Self::analyze_slot`] folds its own.
    #[must_use]
    pub(crate) fn analyze_verifier(workspace: &str, suffix: &str, index: usize) -> Self {
        Self::AnalyzeVerifier {
            workspace: workspace.to_string(),
            suffix: format!("{suffix}_{index}"),
        }
    }

    /// A research run's verification analyst, spelled the same way under the run's own prefix.
    #[must_use]
    pub(crate) fn research_verifier(workspace: &str, suffix: &str, index: usize) -> Self {
        Self::ResearchVerifier {
            workspace: workspace.to_string(),
            suffix: format!("{suffix}_{index}"),
        }
    }

    /// The research run's orchestrator: the coordination calls' address, unsuffixed because it
    /// speaks for the run rather than for one member of it.
    #[must_use]
    pub(crate) fn research_orchestrator(workspace: &str) -> Self {
        Self::ResearchOrchestrator {
            workspace: workspace.to_string(),
        }
    }

    /// A workspace role-discovery round, suffixed the same way.
    #[must_use]
    pub(crate) fn discovery(workspace: &str, role: &str) -> Self {
        Self::Discovery {
            workspace: workspace.to_string(),
            suffix: crate::generate_suffix(),
            role: role.to_string(),
        }
    }

    /// A custom role's one-off run, fresh per launch.
    #[must_use]
    pub(crate) fn custom_role_run(name: &str) -> Self {
        Self::CustomRoleRun {
            name: name.to_string(),
            suffix: crate::generate_suffix(),
        }
    }

    /// A custom role's continuable branch: `role_{name}`, or `role_{name}:{identifier}` for a named
    /// one. Deterministic, so the next message on the branch resumes the same session.
    #[must_use]
    pub(crate) fn custom_role(name: &str, identifier: Option<&str>) -> Self {
        Self::CustomRole {
            name: name.to_string(),
            identifier: identifier.map(str::to_string),
        }
    }

    /// A worker branch: `branch`'s continuable conversation.
    #[must_use]
    pub(crate) fn worker_branch(branch: impl Into<String>) -> Self {
        Self::WorkerBranch {
            branch: branch.into(),
        }
    }

    /// The agent a routed message from `user_name` in `role` at `workspace_name` reaches: the
    /// workspace's Manager for the manager role, a direct chat for every other.
    #[must_use]
    pub(crate) fn routed(user_name: &str, role: &str, workspace_name: &str) -> Self {
        if role == crate::Role::Manager.as_str() {
            Self::manager(workspace_name)
        } else {
            Self::direct(user_name, role, workspace_name)
        }
    }

    /// THE one place an agent id is spelled. Every id in the product is rendered here, so the
    /// frozen spellings — session primary keys, all of them — have one home.
    #[must_use]
    pub(crate) fn id(&self) -> String {
        match self {
            Self::Direct {
                user,
                role,
                workspace,
            } => {
                // Invariant: the routed user identity is never empty
                // ([`crate::session::normalize_user_name`]). Guard the ID itself so no bare
                // "_ws_role" key can be produced — the single chokepoint for every direct ID.
                let user = crate::session::normalize_user_name(user, "AgentAddress::Direct");
                // The leading segment stays escaped, comparing the embedded user raw so reserved
                // names like `manager` yield `user_manager_personal:{role}`.
                if crate::users::personal_user_name(workspace) == Some(user) {
                    return format!("{}_personal:{role}", safe_user_segment(user));
                }
                format!("{}_{}_{}", safe_user_segment(user), workspace, role)
            }
            Self::Assistant { user } => format!(
                "{}_personal:{}",
                safe_user_segment(crate::session::normalize_user_name(
                    user,
                    "AgentAddress::Assistant",
                )),
                crate::Role::Assistant.as_str()
            ),
            Self::Manager { workspace } => format!("manager_{workspace}"),
            Self::Ticket { ticket, role } => format!("ticket_{ticket}_{role}"),
            Self::Maintainer { workspace } => format!("maintainer_{workspace}"),
            Self::Analyze {
                workspace,
                suffix,
                role,
            } => format!("analyze_{workspace}_{suffix}_{role}"),
            Self::Research {
                workspace,
                suffix,
                label,
            } => format!("research_{workspace}_{suffix}_{label}"),
            Self::AnalyzeVerifier { workspace, suffix } => {
                format!("analyze_{workspace}_verify_{suffix}")
            }
            Self::ResearchVerifier { workspace, suffix } => {
                format!("research_{workspace}_verify_{suffix}")
            }
            Self::ResearchOrchestrator { workspace } => {
                format!("research_{workspace}_orchestrator")
            }
            Self::Discovery {
                workspace,
                suffix,
                role,
            } => format!("discovery_{workspace}_{suffix}_{role}"),
            Self::Cleanup { job_id } => format!("cleanup_{job_id}"),
            Self::CustomRole { name, identifier } => match identifier {
                Some(identifier) => format!("{CUSTOM_ROLE_AGENT_ID_PREFIX}{name}:{identifier}"),
                None => format!("{CUSTOM_ROLE_AGENT_ID_PREFIX}{name}"),
            },
            Self::CustomRoleRun { name, suffix } => {
                format!("{CUSTOM_ROLE_RUN_AGENT_ID_PREFIX}{name}:{suffix}")
            }
            Self::WorkerBranch { branch } => format!("{WORKER_BRANCH_AGENT_ID_PREFIX}{branch}"),
        }
    }

    /// The kind a stored id names, where the id decides it. Exact for the prefixed families —
    /// manager, maintainer, worker branch, custom role, ticket, cleanup — where the prefix alone
    /// separates the parts. The round families and the direct (segment) form are answered by a
    /// best-effort split: a transient round's generated suffix is ambiguous with the segments
    /// around it, and a direct id's leading user segment and its workspace are both free-form, so
    /// neither yields parts that can be rendered back. The suffixed round families are told from
    /// their verifier forms by a shape test ([`verifier_split`]), which is narrower than exactness:
    /// a generated suffix that itself contains the separator, or a research label that ends in
    /// digits behind a suffix-shaped token, is answered as the other family. No caller reads a part
    /// it did not already know — neither a round's parts nor a direct id's.
    #[must_use]
    pub(crate) fn parse(id: &str) -> Option<Self> {
        if let Some(workspace) = id.strip_prefix("manager_") {
            return Some(Self::Manager {
                workspace: workspace.to_string(),
            });
        }
        if let Some(workspace) = id.strip_prefix("maintainer_") {
            return Some(Self::Maintainer {
                workspace: workspace.to_string(),
            });
        }
        if let Some(branch) = id.strip_prefix(WORKER_BRANCH_AGENT_ID_PREFIX) {
            return Some(Self::WorkerBranch {
                branch: branch.to_string(),
            });
        }
        if let Some(rest) = id.strip_prefix(CUSTOM_ROLE_RUN_AGENT_ID_PREFIX) {
            let (name, suffix) = rest.split_once(':').unwrap_or((rest, ""));
            return Some(Self::CustomRoleRun {
                name: name.to_string(),
                suffix: suffix.to_string(),
            });
        }
        if let Some(rest) = id.strip_prefix(CUSTOM_ROLE_AGENT_ID_PREFIX) {
            let (name, identifier) = match rest.split_once(':') {
                Some((name, identifier)) => (name, Some(identifier.to_string())),
                None => (rest, None),
            };
            return Some(Self::CustomRole {
                name: name.to_string(),
                identifier,
            });
        }
        if let Some(rest) = id.strip_prefix("ticket_") {
            let (ticket, role) = rest.rsplit_once('_').unwrap_or((rest, ""));
            return Some(Self::Ticket {
                ticket: ticket.to_string(),
                role: role.to_string(),
            });
        }
        if let Some(job_id) = id.strip_prefix("cleanup_") {
            return Some(Self::Cleanup {
                job_id: job_id.to_string(),
            });
        }
        // The suffixed round families, then the two segment forms: the split is a best effort
        // (see the method's own note), the kind is not. A verifier's `_verify` marker is read from
        // the right — its id ends in the slot index, which a round id never does (a round ends in
        // its role) — so a workspace that itself contains `_verify_` is still named whole.
        if let Some(rest) = id.strip_prefix("analyze_") {
            if let Some((workspace, suffix)) = verifier_split(rest) {
                return Some(Self::AnalyzeVerifier {
                    workspace: workspace.to_string(),
                    suffix: suffix.to_string(),
                });
            }
            let (workspace, suffix, role) = split_round(rest);
            return Some(Self::Analyze {
                workspace,
                suffix,
                role,
            });
        }
        if let Some(rest) = id.strip_prefix("research_") {
            if let Some((workspace, suffix)) = verifier_split(rest) {
                return Some(Self::ResearchVerifier {
                    workspace: workspace.to_string(),
                    suffix: suffix.to_string(),
                });
            }
            let (workspace, suffix, label) = split_round(rest);
            return Some(Self::Research {
                workspace,
                suffix,
                label,
            });
        }
        if let Some(rest) = id.strip_prefix("discovery_") {
            let (workspace, suffix, role) = split_round(rest);
            return Some(Self::Discovery {
                workspace,
                suffix,
                role,
            });
        }
        if let Some((user, role)) = id.rsplit_once("_personal:") {
            return Some(parse_personal(user, role));
        }
        let (rest, role) = id.rsplit_once('_')?;
        let (user, workspace) = rest.split_once('_')?;
        Some(Self::Direct {
            user: user.to_string(),
            role: role.to_string(),
            workspace: workspace.to_string(),
        })
    }

    /// Whether this address is transient — background-only and TTL-purged
    /// ([`crate::session::cleanup_old_transient_sessions`]): a ticket round and its parallel roster
    /// slots, an analyze round and its analyst and verification slots, a research run, its
    /// orchestrator and its verification analysts, a discovery round, a cleanup round, the
    /// Maintainer, a custom role's one-off run. The user-facing addresses persist indefinitely: a
    /// direct chat, the Assistant, a workspace Manager (which carries chat and notification
    /// context), a custom role's continuable branch and a worker branch.
    ///
    /// Answered from the variant, not by rendering the id and prefix-matching it: the kind is what
    /// decides, and [`TRANSIENT_AGENT_ID_PREFIXES`] is its SQL projection — a test pins the two
    /// together, so a kind cannot be transient here and permanent in the purge.
    #[must_use]
    pub(crate) fn is_transient(&self) -> bool {
        matches!(
            self,
            Self::Ticket { .. }
                | Self::Maintainer { .. }
                | Self::Analyze { .. }
                | Self::AnalyzeVerifier { .. }
                | Self::Research { .. }
                | Self::ResearchVerifier { .. }
                | Self::ResearchOrchestrator { .. }
                | Self::Discovery { .. }
                | Self::Cleanup { .. }
                | Self::CustomRoleRun { .. }
        )
    }

    /// Whether this address names a custom role's session: a continuable branch or a one-off run.
    #[must_use]
    pub(crate) fn is_custom_role(&self) -> bool {
        matches!(self, Self::CustomRole { .. } | Self::CustomRoleRun { .. })
    }

    /// The branch name inside a worker branch's address, when the id is one.
    #[must_use]
    pub(crate) fn branch_name(&self) -> Option<&str> {
        match self {
            Self::WorkerBranch { branch } => Some(branch),
            _ => None,
        }
    }

    /// Whether a stored id names an agent a user never talks to directly — a workspace Manager or
    /// any transient round. [`non_user_facing_agent_id_prefixes`] is the SQL projection of this.
    #[must_use]
    pub(crate) fn is_non_user_facing(&self) -> bool {
        matches!(self, Self::Manager { .. }) || self.is_transient()
    }
}

/// The address a `{user}_personal:{role}` id spells: the Assistant when the role is its own, the
/// user's direct chat in their personal workspace otherwise. The leading segment is un-escaped and
/// the workspace derived from the raw name, so a reserved-name user round-trips.
fn parse_personal(user: &str, role: &str) -> AgentAddress {
    let user = raw_user_segment(user);
    if role == crate::Role::Assistant.as_str() {
        AgentAddress::Assistant {
            user: user.to_string(),
        }
    } else {
        AgentAddress::Direct {
            user: user.to_string(),
            role: role.to_string(),
            workspace: crate::users::personal_workspace_name(user),
        }
    }
}

/// Split `{workspace}_verify_{suffix}_{index}` of a verifier id, read from the right. What a
/// verifier id guarantees is a shape, not an inverse: `…_verify_<generated suffix>_<digits>`,
/// where the suffix is [`crate::is_generated_suffix`]'s and the index is a slot number. A round id
/// ends in its role instead, so the shape is what tells the two apart — including when the
/// workspace's own name contains `_verify_`. It is deliberately not exact: a generated suffix may
/// itself contain the separator (and is then read as a round), and a research round whose label
/// ends in digits behind a suffix-shaped token is read as a verifier. `None` for an id that is not
/// the verifier form, which then falls through to [`split_round`].
fn verifier_split(rest: &str) -> Option<(&str, &str)> {
    let (workspace, suffix) = rest.rsplit_once("_verify_")?;
    let (generated, index) = suffix.rsplit_once('_')?;
    let numeric = !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit());
    (numeric && crate::is_generated_suffix(generated)).then_some((workspace, suffix))
}

/// Split `{workspace}_{suffix}_{label}` of a suffixed round family — best effort, the suffix's own
/// alphabet includes `_`.
fn split_round(rest: &str) -> (String, String, String) {
    let (head, label) = rest.rsplit_once('_').unwrap_or((rest, ""));
    let (workspace, suffix) = head.rsplit_once('_').unwrap_or((head, ""));
    (workspace.to_string(), suffix.to_string(), label.to_string())
}

/// Agent-id prefixes for transient (background-only, non-user-facing) sessions — auto-created
/// analysts, engineers, maintainers, discovery rounds, cleanup rounds — cleaned up by
/// [`crate::session::cleanup_old_transient_sessions`].
///
/// User-facing agents persist indefinitely and are deliberately excluded: direct chat,
/// `manager_{ws_name}` (which carries chat conversation and notification context), a custom role's
/// continuable branch, and a worker branch. A new kind that can talk to users directly must be
/// excluded here too.
pub(crate) const TRANSIENT_AGENT_ID_PREFIXES: &[&str] = &[
    "ticket_",
    "analyze_",
    "research_",
    "cleanup_",
    "maintainer_",
    "discovery_",
    "rolerun_",
];

/// Agent-id prefix of a custom role's continuable branch — `role_{name}` (the record's default
/// branch) or `role_{name}:{identifier}`. The `:` separator keeps the two forms injective: names
/// and identifiers may both contain `_`, so `_` would alias `(a_b, None)` with `(a, Some(b))` —
/// and `personal:{user}` already establishes `:` in agent ids.
///
/// Deliberately NOT in [`TRANSIENT_AGENT_ID_PREFIXES`]: it is a user-facing conversation, never
/// TTL-purged and eligible for dead-session recovery. It is in the wider
/// [`reserved_agent_id_prefixes`] union, so a user literally named `role` becomes `user_role`.
pub(crate) const CUSTOM_ROLE_AGENT_ID_PREFIX: &str = "role_";

/// Agent-id prefix of a custom role's ONE-OFF run — `rolerun_{name}:{suffix}`. Same `:` injectivity
/// rule as [`CUSTOM_ROLE_AGENT_ID_PREFIX`]: the [`crate::generate_suffix`] NanoID may itself
/// contain `_`. Transient, so it expires with the ordinary session lifetime.
pub(crate) const CUSTOM_ROLE_RUN_AGENT_ID_PREFIX: &str = "rolerun_";

/// Agent-id prefix of a worker branch — `worker_{branch}`, a continuable conversation like a custom
/// role's branch: deliberately NOT in [`TRANSIENT_AGENT_ID_PREFIXES`] (never TTL-purged), IS
/// recovered by the dead-session poller, and IS in [`reserved_agent_id_prefixes`] so no direct user
/// id can shadow it. Not in [`non_user_facing_agent_id_prefixes`]: the branch is user-facing.
pub(crate) const WORKER_BRANCH_AGENT_ID_PREFIX: &str = "worker_";

/// The prefixes of every non-user-facing agent id: `manager_` plus every
/// [`TRANSIENT_AGENT_ID_PREFIXES`] entry.
///
/// These identify session keys a user never talks to directly. The dead-session poller excludes
/// this whole union, while [`crate::session::cleanup_old_transient_sessions`] purges only the
/// transient entries (`manager_` is intentionally excluded there). Both match via SQL
/// `LIKE 'prefix_%'`, which Turso applies case-insensitively for ASCII with `_` as any one char —
/// so every prefix here must be at least as long as the shortest id it must not swallow.
pub(crate) fn non_user_facing_agent_id_prefixes() -> impl Iterator<Item = &'static str> {
    std::iter::once("manager_").chain(TRANSIENT_AGENT_ID_PREFIXES.iter().copied())
}

/// Every prefix a direct user id must not shadow: the
/// non-user-facing set ([`non_user_facing_agent_id_prefixes`]) plus
/// [`CUSTOM_ROLE_AGENT_ID_PREFIX`] and [`WORKER_BRANCH_AGENT_ID_PREFIX`], which a
/// user-visible session still owns.
pub(crate) fn reserved_agent_id_prefixes() -> impl Iterator<Item = &'static str> {
    non_user_facing_agent_id_prefixes()
        .chain([CUSTOM_ROLE_AGENT_ID_PREFIX, WORKER_BRANCH_AGENT_ID_PREFIX])
}

/// Does a user name collide with a reserved agent-ID prefix when it becomes the leading segment of
/// a direct agent ID? Such a name would otherwise be purged by the SQL exclusion `LIKE 'prefix_%'`
/// (case-insensitive for ASCII, `_` = any one char), so `ticket`, `ticketx`, and `TicketBob` all
/// collide. Names already starting with `user_` are also treated as colliding to keep
/// [`safe_user_segment`] injective.
#[must_use]
fn user_segment_collides(user_name: &str) -> bool {
    let touches_reserved = reserved_agent_id_prefixes()
        .any(|prefix| starts_with_ignore_ascii_case(user_name, prefix.trim_end_matches('_')));
    touches_reserved || starts_with_ignore_ascii_case(user_name, "user_")
}

/// Escape a user name for the leading segment of a direct agent ID by prefixing colliding names
/// ([`user_segment_collides`]) with `user_`, so their session is never mistaken for a
/// transient/background one. The escape is injective: a name already starting with `user_` is ALSO
/// escaped (`user_ticket` → `user_user_ticket`), so `ticket` → `user_ticket` can never collide with
/// a distinct real user named `user_ticket`.
#[must_use]
fn safe_user_segment(user_name: &str) -> Cow<'_, str> {
    if user_segment_collides(user_name) {
        Cow::Owned(format!("user_{user_name}"))
    } else {
        Cow::Borrowed(user_name)
    }
}

/// The raw user name a `personal:` segment spells — [`safe_user_segment`]'s inverse. Every segment
/// starting with `user_` is escaped (the escape is injective, see [`safe_user_segment`]), so
/// stripping that prefix is the exact inverse of a whole segment. Only a `_personal:` id carries the
/// name whole; a direct id is split at its first separator, so a user name containing `_` comes back
/// as its head and the rest joins the workspace.
#[must_use]
fn raw_user_segment(segment: &str) -> &str {
    segment.strip_prefix("user_").unwrap_or(segment)
}

/// ASCII-case-insensitive prefix test — how Turso's `LIKE 'prefix%'` compares, so the Rust-side
/// mirrors of those predicates agree with the SQL.
#[must_use]
pub(crate) fn starts_with_ignore_ascii_case(s: &str, prefix: &str) -> bool {
    s.get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every transient kind's id starts with the prefix [`TRANSIENT_AGENT_ID_PREFIXES`] lists, or
    /// [`crate::session::cleanup_old_transient_sessions`] never purges it (a leak); and no
    /// user-facing kind's id starts with one, or the purge silently deletes user history — a
    /// colliding user name is escaped by [`safe_user_segment`].
    #[test]
    fn the_transient_kinds_are_exactly_the_prefixes_the_purge_matches() {
        let transient = [
            AgentAddress::Ticket {
                ticket: "abc123".to_string(),
                role: "analyst".to_string(),
            },
            AgentAddress::analyze("ws", "worker"),
            AgentAddress::analyze_slot("ws", "nano", 1, "analyst"),
            AgentAddress::analyze_verifier("ws", "nano", 2),
            AgentAddress::research("ws", "decomposer"),
            AgentAddress::research_verifier("ws", "nano", 0),
            AgentAddress::research_orchestrator("ws"),
            AgentAddress::Cleanup {
                job_id: "run_abc123".to_string(),
            },
            AgentAddress::Maintainer {
                workspace: "ws".to_string(),
            },
            AgentAddress::discovery("ws", "analyst"),
            AgentAddress::custom_role_run("ws"),
        ];
        for address in &transient {
            let id = address.id();
            assert!(
                address.is_transient(),
                "{id} is transient but is_transient says otherwise"
            );
            let prefix = TRANSIENT_AGENT_ID_PREFIXES
                .iter()
                .find(|prefix| id.starts_with(*prefix))
                .unwrap_or_else(|| {
                    panic!(
                        "{id} starts with no registered transient prefix — its sessions would \
                         never be cleaned up.\nFix: add its prefix to TRANSIENT_AGENT_ID_PREFIXES."
                    )
                });
            assert!(
                prefix.trim_end_matches('_').len() > 1,
                "{prefix} is too short to be a namespace"
            );
        }

        let user_facing = [
            AgentAddress::Direct {
                user: "alice".to_string(),
                role: "analyst".to_string(),
                workspace: "test-ws".to_string(),
            },
            AgentAddress::assistant("admin"),
            AgentAddress::manager("test-ws"),
            AgentAddress::custom_role("briefer", None),
            AgentAddress::custom_role("briefer", Some("hourly")),
            AgentAddress::WorkerBranch {
                branch: "drafting".to_string(),
            },
        ];
        for address in &user_facing {
            let id = address.id();
            assert!(
                !address.is_transient(),
                "{id} is user-facing but is_transient says otherwise"
            );
            for prefix in TRANSIENT_AGENT_ID_PREFIXES {
                assert!(
                    !id.starts_with(prefix),
                    "USER-FACING COLLISION: prefix='{prefix}' matches id='{id}'"
                );
            }
        }
    }

    /// The `_` is part of an address, not the SQL `LIKE` wildcard: a user name that merely begins
    /// a reserved namespace word is escaped, so the produced id never starts (case-insensitively)
    /// with the bare word.
    #[test]
    fn a_user_name_never_shadows_a_reserved_namespace() {
        let reserved_prefixes: Vec<&str> = reserved_agent_id_prefixes().collect();
        for prefix in &reserved_prefixes {
            let bare_word = prefix.trim_end_matches('_');
            // Capitalized variant (uppercase only the first char) covers the ASCII
            // case-insensitivity of Turso's SQLite-compatible `LIKE`, e.g. `Ticket`, `Manager`.
            let capitalized = {
                let mut chars = bare_word.chars();
                match chars.next() {
                    Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                    None => bare_word.to_string(),
                }
            };
            let user_names = [
                bare_word.to_string(),
                format!("{bare_word}_bob"),
                format!("{bare_word}x"),
                capitalized.clone(),
                format!("{capitalized}Bob"),
            ];
            for user_name in &user_names {
                let id = direct(user_name, "analyst", "test-ws");
                assert!(
                    !starts_with_ignore_ascii_case(&id, bare_word),
                    "DIRECT AGENT ID COLLISION: bare-word='{bare_word}' (case-insensitive) \
                     matches id='{id}' (user='{user_name}').\nFix: safe_user_segment must escape \
                     any user name starting (case-insensitively) with '{bare_word}'.",
                );
            }

            // The escape is injective: a real user whose name already begins with `user_` is
            // escaped AGAIN, so the escape of the bare word can never collide with a distinct
            // real user literally named `user_{bare_word}`.
            let user_word = format!("user_{bare_word}");
            let escaped = direct(&user_word, "analyst", "test-ws");
            assert!(
                escaped.starts_with("user_user_"),
                "real user '{user_word}' should be double-escaped to start with 'user_user_', \
                 got '{escaped}'",
            );
            assert_ne!(
                escaped,
                direct(bare_word, "analyst", "test-ws"),
                "escape of '{bare_word}' collides with escape of '{user_word}'",
            );
        }

        // A normal user is NOT escaped — the id keeps its raw user segment.
        assert_eq!(
            direct("alice", "analyst", "test-ws"),
            "alice_test-ws_analyst"
        );
    }

    fn direct(user: &str, role: &str, workspace: &str) -> String {
        AgentAddress::Direct {
            user: user.to_string(),
            role: role.to_string(),
            workspace: workspace.to_string(),
        }
        .id()
    }

    /// A one-off run's address must keep its role name recoverable: with a `_` separator, a role
    /// named `a_b` and the NanoID suffix's own `_` would make role `a_b` + suffix `c` and role `a`
    /// + suffix `b_c` one identical id.
    #[test]
    fn one_off_run_address_separates_the_name_from_its_suffix() {
        let id = AgentAddress::custom_role_run("a_b").id();
        let (name, suffix) = id
            .strip_prefix(CUSTOM_ROLE_RUN_AGENT_ID_PREFIX)
            .expect("the one-off prefix")
            .split_once(':')
            .expect("a `:` between the name and the generated suffix");
        assert_eq!(name, "a_b");
        assert!(!suffix.is_empty(), "the suffix must survive the separator");
    }

    /// Dispatch is on the role string, matched exactly as [`crate::Role`] spells it: lowercase
    /// `manager` addresses the workspace's Manager, while every other role — a capital-M `Manager`
    /// included — is a direct, channel-agnostic id with the role as its LAST segment.
    #[test]
    fn routing_dispatches_on_the_lowercase_manager_role() {
        let cases = [
            ("alice", "manager", "my-workspace", "manager_my-workspace"),
            (
                "bob",
                "engineer",
                "my-workspace",
                "bob_my-workspace_engineer",
            ),
            ("carol", "Manager", "ws", "carol_ws_Manager"),
        ];
        for (user, role, workspace, expected) in cases {
            assert_eq!(AgentAddress::routed(user, role, workspace).id(), expected);
        }
        assert_ne!(
            AgentAddress::routed("carol", "Manager", "ws").id(),
            "manager_ws",
            "capital-M 'Manager' must not address the Manager"
        );
    }

    /// A parallel round's roster slot folds into the singular kind's own segments — the frozen
    /// spelling is `analyze_{ws_name}_{suffix}_{index}_{role}` for an analyze round and
    /// `ticket_{ticket_id}_{index}_{suffix}_{role}` for a ticket round. Both are session primary
    /// keys: a changed one orphans every stored round.
    #[test]
    fn a_parallel_round_slot_folds_into_the_singular_spelling() {
        assert_eq!(
            AgentAddress::analyze_slot("ws", "nano", 1, "analyst").id(),
            "analyze_ws_nano_1_analyst",
        );
        assert_eq!(
            AgentAddress::Ticket {
                ticket: "abc123_0_nano".to_string(),
                role: "analyst".to_string(),
            }
            .id(),
            "ticket_abc123_0_nano_analyst",
        );
    }

    /// A user's OWN personal workspace dedups the duplicated user segment, so the role is
    /// `:`-delimited after the marker; another user's, or a project merely named `personal_work`,
    /// keeps the full form.
    #[test]
    fn a_direct_address_dedups_the_users_own_personal_workspace() {
        assert_eq!(
            direct("alice", "assistant", "personal:alice"),
            "alice_personal:assistant",
        );
        assert_eq!(
            direct("bob", "assistant", "personal:admin"),
            "bob_personal:admin_assistant",
        );
        assert_eq!(
            direct("alice", "analyst", "personal_work"),
            "alice_personal_work_analyst",
        );
        // A reserved-name user still gets the `user_` escape on the leading segment while the
        // embedded user is compared raw — collision-safe yet deduped.
        assert_eq!(
            direct("manager", "engineer", "personal:manager"),
            "user_manager_personal:engineer",
        );
        // A user already starting with `user_` is injectively double-escaped.
        assert_eq!(
            direct("user_ticket", "analyst", "personal:user_ticket"),
            "user_user_ticket_personal:analyst",
        );
    }

    /// The Assistant address is the direct one in the user's own personal workspace — the two must
    /// never drift, because automation arming addresses one and chat resumes the other.
    #[test]
    fn the_assistant_address_is_the_direct_one_in_the_users_own_workspace() {
        for user in ["admin", "manager", "user_ticket"] {
            assert_eq!(
                AgentAddress::assistant(user).id(),
                direct(
                    user,
                    crate::Role::Assistant.as_str(),
                    &crate::users::personal_workspace_name(user),
                ),
            );
        }
    }

    /// `parse` is the inverse for every kind a stored id decides exactly — the ones the product
    /// asks a kind question about.
    #[test]
    fn a_deterministic_address_parses_back_to_itself() {
        let addresses = [
            AgentAddress::assistant("admin"),
            AgentAddress::manager("qa_ws"),
            AgentAddress::Ticket {
                ticket: "abc123".to_string(),
                role: "engineer".to_string(),
            },
            AgentAddress::Maintainer {
                workspace: "qa_ws".to_string(),
            },
            AgentAddress::Cleanup {
                job_id: "run_abc123".to_string(),
            },
            AgentAddress::custom_role("a_b", None),
            AgentAddress::custom_role("a", Some("b")),
            AgentAddress::WorkerBranch {
                branch: "drafting".to_string(),
            },
            // A reserved-name user is escaped in the leading segment; the parse un-escapes it and
            // derives the personal workspace from the raw name.
            AgentAddress::assistant("manager"),
            AgentAddress::direct("ticket", "engineer", "personal:ticket"),
            AgentAddress::direct("user_ticket", "analyst", "personal:user_ticket"),
            // A non-personal direct id splits exactly while its user segment carries no `_`; the
            // workspace may carry one, since it takes the rest of the id before the role.
            AgentAddress::direct("alice", "engineer", "my_workspace"),
        ];
        for address in addresses {
            let id = address.id();
            assert_eq!(
                AgentAddress::parse(&id),
                Some(address),
                "round trip of {id}"
            );
        }
        // An id with no separator at all names no kind of ours.
        assert_eq!(AgentAddress::parse("nounderscores"), None);
    }

    /// A direct id's user segment and its workspace are both free-form, so the boundary between
    /// them is the id's first separator and nothing in the id says where the user name ends. The
    /// kind is still exact; the parts are not.
    #[test]
    fn a_direct_id_whose_user_name_contains_a_separator_splits_at_its_first_one() {
        let id = AgentAddress::direct("user_bob", "engineer", "myws").id();
        assert_eq!(id, "user_user_bob_myws_engineer");
        // The split is the id's first separator, so the user comes back short and the workspace
        // long: both parts are free-form, and the id alone cannot say where the name ends.
        assert_eq!(
            AgentAddress::parse(&id),
            Some(AgentAddress::Direct {
                user: "user".to_string(),
                role: "engineer".to_string(),
                workspace: "user_bob_myws".to_string(),
            }),
        );
    }

    /// A verification analyst's id parses back to the address that produced it, under both round
    /// families — the `_verify` marker is what tells it apart from an ordinary round's slot. A
    /// workspace whose own name contains `_verify_` must not be mistaken for the marker: the
    /// verifier form is read from the right-hand `_verify_<suffix>_<index>` tail, so both a round
    /// and a verifier round-trip with such a workspace.
    #[test]
    fn a_verification_analyst_parses_back_to_the_address_that_produced_it() {
        for address in [
            AgentAddress::analyze_verifier("ws", "abCD12", 2),
            AgentAddress::research_verifier("ws", "abCD12", 0),
            AgentAddress::analyze_verifier("my_verify_api", "abCD12", 2),
            AgentAddress::research_verifier("my_verify_api", "abCD12", 0),
            AgentAddress::Analyze {
                workspace: "my_verify_api".to_string(),
                suffix: "abCD12".to_string(),
                role: "analyst".to_string(),
            },
        ] {
            let id = address.id();
            assert!(address.is_transient(), "{id} is not a conversation");
            assert_eq!(
                AgentAddress::parse(&id),
                Some(address),
                "round trip of {id}"
            );
        }
        // A research run's label is free-form, so its id is answered by kind alone — the label a
        // caller emits may end in digits and lose its tail to the best-effort split.
        let research = AgentAddress::Research {
            workspace: "my_verify_api".to_string(),
            suffix: "abCD12".to_string(),
            label: "r1_0".to_string(),
        }
        .id();
        assert!(
            matches!(
                AgentAddress::parse(&research),
                Some(AgentAddress::Research { .. })
            ),
            "the kind of {research} is not a guess",
        );
        assert_eq!(
            AgentAddress::analyze_verifier("ws", "abCD12", 2).id(),
            "analyze_ws_verify_abCD12_2",
        );
        assert_eq!(
            AgentAddress::research_verifier("ws", "abCD12", 0).id(),
            "research_ws_verify_abCD12_0",
        );
        // The shape test is what decides: a verifier-shaped tail with no generated suffix behind
        // it is a round, not a verifier.
        assert!(matches!(
            AgentAddress::parse("research_ws_verify_nano_0"),
            Some(AgentAddress::Research { .. })
        ));
    }

    /// A stored id answers the kind questions the session layer asks of it: a custom role's
    /// branch and one-off run, a worker branch, and what a user never talks to directly.
    #[test]
    fn a_stored_id_answers_which_kind_it_names() {
        let branch = AgentAddress::custom_role("briefer", Some("hourly")).id();
        assert!(
            AgentAddress::parse(&branch).is_some_and(|address| address.is_custom_role()),
            "{branch}"
        );
        let run = AgentAddress::custom_role_run("briefer").id();
        assert!(
            AgentAddress::parse(&run).is_some_and(|address| address.is_custom_role()),
            "{run}"
        );
        assert!(
            !AgentAddress::parse(&AgentAddress::assistant("admin").id())
                .is_some_and(|address| address.is_custom_role()),
        );

        let worker = AgentAddress::WorkerBranch {
            branch: "drafting".to_string(),
        }
        .id();
        assert_eq!(
            AgentAddress::parse(&worker)
                .and_then(|address| address.branch_name().map(str::to_string)),
            Some("drafting".to_string()),
        );

        for id in [
            AgentAddress::manager("qa_ws").id(),
            AgentAddress::analyze("qa_ws", "analyst").id(),
            AgentAddress::Cleanup {
                job_id: "run_1".to_string(),
            }
            .id(),
        ] {
            assert!(
                AgentAddress::parse(&id).is_some_and(|address| address.is_non_user_facing()),
                "{id} is background-only"
            );
        }
        for id in [
            AgentAddress::assistant("admin").id(),
            AgentAddress::custom_role("briefer", None).id(),
            AgentAddress::WorkerBranch {
                branch: "drafting".to_string(),
            }
            .id(),
        ] {
            assert!(
                !AgentAddress::parse(&id).is_some_and(|address| address.is_non_user_facing()),
                "{id} is a conversation the user talks to"
            );
        }
    }
}
