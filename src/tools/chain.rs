//! A `send` chain: one call whose answer travels through more addressees before coming back.
//!
//! The `for` argument belongs to the owner's Assistant alone (and its command-line twin): it names
//! where a step's answer goes next, in the same shape as the call minus `message`, and may carry
//! its own `for`, so a single call describes the whole chain. The call itself is step one; a chain
//! of ten is the outer call plus up to nine nested `for`. The records themselves live beside the
//! launch that carries them ([`crate::launch`]).
//!
//! A step's durable record is one of the records a hop already has: a worker round's `jobs` row, a
//! branch launch's queued envelope, or — for a role or worker-branch step, which has no round of
//! its own — the conversation's `session_metadata.chain`. That last one is what makes a role step
//! survive a restart: the boot replay reclaims its queued envelope as already appended, and its
//! conversation is what still says it is a chain step.
//!
//! The hand-off is the durability boundary, and it is one rule: a step that has not answered
//! continues from where it was (its conversation resumed, its launch rebuilt from its row), and a
//! step that answered but whose hand-off did not commit is replayed. Every step's durable record is
//! named [`record_id`], which is what keeps a replay from forking the chain: the replayed step
//! finds the next step's record already there and leaves it alone. The window past that — the next
//! step having finished too — is the same at-least-once window the product already accepts for
//! delegations.
//!
//! A step that fails, answers nothing or cannot reach its folder ends the chain; the caller — the
//! owner's Assistant, the one conversation a chain can be started from — gets one step-numbered
//! failure in the envelope that step's addressee already uses. Only the final answer comes back.
//! A chain is one stoppable group in Running Agents for its whole life. Ending a chain is loud in
//! every case but one: a chain the owner stopped delivers nothing, and any other way its records
//! disappear is a failure the caller is told about.

use anyhow::{Context as _, Result, bail};

use crate::Workspace;
use crate::agent::message_router::AnsweredFor;
use crate::agent::registry::ParentKey;
use crate::launch::{BusyFolder, Chain, MAX_STEPS, Step, Stored, record_id};
use crate::tools::send::{Addressee, Argument, FOR_ARG};
use crate::util::UnwrapPoison as _;

/// The chain's id, derived from the call itself: the task and every address it names. A re-issued
/// call computes the same id, so while the chain it started is still in flight it resumes that
/// chain instead of starting a second one.
#[must_use]
pub(crate) fn id_of(task: &str, outer: &Step, next: &[Step]) -> String {
    use sha2::{Digest as _, Sha256};
    use std::fmt::Write as _;
    let mut hasher = Sha256::new();
    hasher.update(task.as_bytes());
    // A step's schema is part of what makes this chain this chain, and two schemas differing only
    // in key order or whitespace are the same one. It joins the identity only when a step carries
    // one, so a call without a schema produces exactly the identity it produced before.
    let schemas = std::iter::once(outer)
        .chain(next)
        .any(|step| step.extract.is_some());
    for step in std::iter::once(outer).chain(next) {
        // Every step is the same length-prefixed fields, so no two calls can hash alike.
        for part in [
            Some(step.addressee.as_str()),
            step.place.as_deref(),
            step.branch.as_deref(),
            step.name.as_deref(),
        ] {
            match part {
                Some(part) => {
                    hasher.update((part.len() as u64).to_le_bytes());
                    hasher.update(part.as_bytes());
                }
                None => hasher.update(u64::MAX.to_le_bytes()),
            }
        }
        if schemas {
            match step.extract.as_ref() {
                Some(extract) => {
                    let canonical = extract.canonical();
                    hasher.update((canonical.len() as u64).to_le_bytes());
                    hasher.update(canonical.as_bytes());
                }
                None => hasher.update(u64::MAX.to_le_bytes()),
            }
        }
    }
    let mut id = String::with_capacity(16);
    for byte in hasher.finalize().iter().take(8) {
        // Writing into a `String` cannot fail.
        let _ = write!(id, "{byte:02x}");
    }
    id
}

/// Read the `for` value into the steps that follow the call. The call itself is step one, so the
/// returned list holds steps two onwards.
pub(crate) fn parse(value: &serde_json::Value) -> Result<Vec<Step>> {
    let mut steps = Vec::new();
    let mut level = value.clone();
    loop {
        let object = as_object(&level)?;
        steps.push(step_from(&object)?);
        // The call itself is step one, so `for` may nest at most nine deep.
        if steps.len() >= MAX_STEPS {
            bail!(
                "usage: a chain is at most {MAX_STEPS} steps — the call itself is step one, so `for` \
                 may nest at most {} deep",
                MAX_STEPS - 1
            );
        }
        match object.get(FOR_ARG) {
            None | Some(serde_json::Value::Null) => return Ok(steps),
            Some(next) => level = next.clone(),
        }
    }
}

/// The one refusal a malformed `for` value gets, wherever its shape is wrong.
const SHAPE: &str = "usage: `for` must be a JSON object with an `addressee` and, when that \
                     addressee needs them, `place`, `branch` or `name`";

/// Every key one step may carry, `for` included.
const STEP_KEYS: [&str; 6] = [
    "addressee",
    "place",
    "branch",
    "name",
    FOR_ARG,
    crate::extract::ARG,
];

/// The `for` value's own shape, whichever form it was given in.
fn as_object(value: &serde_json::Value) -> Result<serde_json::Map<String, serde_json::Value>> {
    let object = match value {
        serde_json::Value::Object(map) => Some(map.clone()),
        // The command line carries the value as one flag: JSON text is the same value.
        serde_json::Value::String(text) => match serde_json::from_str(text) {
            Ok(serde_json::Value::Object(map)) => Some(map),
            _ => None,
        },
        _ => None,
    };
    object.ok_or_else(|| anyhow::anyhow!(SHAPE))
}

fn step_from(object: &serde_json::Map<String, serde_json::Value>) -> Result<Step> {
    for key in object.keys() {
        if !STEP_KEYS.contains(&key.as_str()) {
            bail!(
                "usage: `for` has no `{key}` — a step names an `addressee` and, when that addressee \
                 needs them, its `place`, `branch` or `name`, plus its own `for` and `extract`"
            );
        }
    }
    let raw = object
        .get("addressee")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "usage: every `for` step needs an `addressee` — `worker`, `worker_branch` or `role`"
            )
        })?;
    let addressee = Addressee::parse(raw).ok_or_else(|| {
        anyhow::anyhow!(
            "usage: `{raw}` is not an address — a chain step is `worker`, `worker_branch` or `role`"
        )
    })?;
    // The three addresses a chain cannot use, each for a reason of its own.
    match addressee {
        Addressee::Worker | Addressee::WorkerBranch | Addressee::Role => {}
        Addressee::Guest => bail!(
            "usage: a `guest` address cannot be a chain step — nothing comes back from it, so the \
             chain could not continue"
        ),
        Addressee::Assistant => bail!(
            "usage: `assistant` cannot be a chain step — it is the conversation the chain answers \
             to, not an agent a step hands its answer to"
        ),
        Addressee::Manager => bail!(
            "usage: `manager` cannot be a chain step — a Manager turn may end with no answer at \
             all, which would stall the chain forever"
        ),
    }
    let text = |key: &str| -> Result<Option<String>> {
        match object.get(key) {
            None | Some(serde_json::Value::Null) => Ok(None),
            // An empty value is an absent one, exactly as an empty flag on the command line is.
            Some(serde_json::Value::String(value)) if value.trim().is_empty() => Ok(None),
            Some(serde_json::Value::String(value)) => Ok(Some(value.trim().to_string())),
            Some(_) => bail!("usage: `{key}` of a chain step must be a string"),
        }
    };
    let step = Step {
        addressee,
        place: text("place")?,
        branch: text("branch")?,
        name: text("name")?,
        // The schema the step asks its answer to match: an object or its JSON text, validated
        // whole before anything runs.
        extract: match object.get(crate::extract::ARG) {
            None | Some(serde_json::Value::Null) => None,
            Some(value) => Some(crate::extract::Extract::from_arg(value)?),
        },
    };
    // The same per-address argument rules the command line applies: an argument the step's own
    // address does not take is refused rather than half-understood.
    let given = |arg: Argument| match arg {
        Argument::Place => step.place.is_some(),
        Argument::Branch => step.branch.is_some(),
        Argument::Name => step.name.is_some(),
    };
    for arg in Argument::ALL {
        if given(arg) && !addressee.takes(arg) {
            bail!(
                "usage: `{}` is not an argument of the `{}` address",
                arg.as_str(),
                addressee.as_str()
            );
        }
    }
    for required in addressee.requires() {
        if !given(required) {
            bail!(
                "usage: a `{}` chain step needs `{}`",
                addressee.as_str(),
                required.as_str()
            );
        }
    }
    Ok(step)
}

/// Refuse a step that cannot run, before anything starts: an unknown role, an invalid branch, an
/// unusable place. What only becomes false later — a role deleted, a folder gone — ends the chain
/// at that step instead (see [`dispatch`]).
pub(crate) async fn precheck(step: &Step) -> Result<()> {
    match step.addressee {
        Addressee::Worker => {}
        Addressee::WorkerBranch => {
            crate::custom_role::validate_worker_branch(step.branch.as_deref().unwrap_or_default())?;
        }
        Addressee::Role => {
            let name = step.name.as_deref().unwrap_or_default();
            anyhow::ensure!(
                crate::custom_role::get(name).await?.is_some(),
                "usage: no custom role named '{name}' — a chain step must name an existing role"
            );
            if let Some(identifier) = step.branch.as_deref() {
                crate::custom_role::validate_identifier(identifier)?;
            }
        }
        other => bail!("usage: `{}` cannot be a chain step", other.as_str()),
    }
    place_of(step).await?;
    Ok(())
}

/// The folder a step runs in, resolved exactly as the same delegation resolves it today: a worker's
/// `place` by the worker rule, a role's by the launch rule. `None` for a worker step that names no
/// place — it runs in the chain's own workspace.
async fn place_of(step: &Step) -> Result<Option<Workspace>> {
    match step.addressee {
        Addressee::Worker | Addressee::WorkerBranch => match step.place.as_deref() {
            None => Ok(None),
            Some(raw) => {
                crate::tools::delegation::resolve(
                    crate::tools::delegation::DelegationKind::Worker,
                    &serde_json::json!({ "place": raw }),
                )
                .await
            }
        },
        Addressee::Role => crate::launch::launch_workspace(step.place.as_deref())
            .await
            .map(Some),
        other => bail!("usage: `{}` cannot be a chain step", other.as_str()),
    }
}

/// Start a chain: dispatch step one with its own message and answer with what will happen.
pub(crate) async fn start(
    chain: Chain,
    origin: &Workspace,
    answered_for: AnsweredFor,
) -> Result<String> {
    // The id is derived from the call, so a re-issued call finds the chain it already started —
    // any of its steps' records still standing — and dispatches nothing again, saying so rather
    // than claiming to have started it.
    match crate::jobs::chain_has_records(&chain.id, 1..=chain.total()).await {
        Some(false) => {}
        Some(true) => {
            return Ok(format!(
                "The chain of {} steps this call describes is already in flight; its final answer \
                 will arrive as the last step's own message.",
                chain.total()
            ));
        }
        // The read failed: dispatching a second copy of a chain is worse than sending nothing, and
        // claiming it started would be a lie about work that never began.
        None => bail!(
            "the chain could not be checked for one already in flight, so nothing was sent — try \
             again"
        ),
    }
    // A re-issued call after a stop is a NEW chain: the stop's mark belongs to the chain that was
    // running, and this call asks for another.
    STOPPED.lock().unwrap_poison().remove(&chain.id);
    let message = first_message(&chain);
    if let Err(e) = dispatch(&chain, &message, origin, answered_for).await {
        forget(&chain.id);
        return Err(e);
    }
    Ok(format!(
        "Chain of {} steps started; only the final answer will arrive, as the last step's own \
         message.",
        chain.total()
    ))
}

/// The message step one receives: the task as it was sent, plus where in the chain it is and that
/// its answer continues to another agent.
fn first_message(chain: &Chain) -> String {
    crate::prompt::substitute(
        &crate::prompt::load_prompt("chain_first.md"),
        &[
            ("{{step}}", &chain.step.to_string()),
            ("{{total}}", &chain.total().to_string()),
            ("{{task}}", &chain.task),
            (
                "{{next}}",
                &chain.next.first().map_or_else(String::new, Step::label),
            ),
        ],
    )
}

/// The message a later step receives: the original task, the previous step's whole answer, and —
/// for every step that still carries a `for` — where it is, who produced the text, and that its
/// answer continues to another agent.
fn later_message(chain: &Chain, producer: &str, answer: &str) -> String {
    let asset = if chain.next.is_empty() {
        "chain_last.md"
    } else {
        "chain_next.md"
    };
    crate::prompt::substitute(
        &crate::prompt::load_prompt(asset),
        &[
            ("{{step}}", &chain.step.to_string()),
            ("{{total}}", &chain.total().to_string()),
            ("{{task}}", &chain.task),
            ("{{producer}}", producer),
            ("{{answer}}", answer),
            (
                "{{next}}",
                &chain.next.first().map_or_else(String::new, Step::label),
            ),
        ],
    )
}

/// Dispatch one step's address: a worker round, a worker branch, or a custom role — the launch
/// authority each of them already has, handed the chain so the step's record carries it.
pub(crate) async fn dispatch(
    chain: &Chain,
    message: &str,
    origin: &Workspace,
    answered_for: AnsweredFor,
) -> Result<()> {
    // Every step re-asserts the chain's run-lifetime marker, so the group stays visible across a
    // restart and while a step waits in a conversation's queue.
    keep_visible(chain);
    let step = &chain.current;
    let place = place_of(step).await?;
    match step.addressee {
        Addressee::Worker => {
            // A single worker run in an occupied folder fails the chain: two agents in one folder
            // overwrite each other's edits.
            BusyFolder::Chain(chain).check(place.as_ref().unwrap_or(origin), None)?;
            crate::tools::implement::hand_over_chain_round(origin, place.as_ref(), message, chain)
                .await
        }
        Addressee::WorkerBranch => {
            // A step waits its turn in the conversation it names; another agent in the folder still
            // fails it, and the launch owns that rule.
            crate::launch::worker_branch(
                origin,
                step.branch.as_deref().unwrap_or_default(),
                message,
                place.as_ref(),
                None,
                BusyFolder::Chain(chain),
                // The chain record owns the step's schema (`settle` reads it from there), so the
                // launch carries none: two carriers could disagree, and the launch's copy would
                // outlive the step it describes.
                None,
                answered_for,
            )
            .await
            .map(|_| ())
        }
        Addressee::Role => {
            // `place_of` always resolves a role's folder (`launch_workspace` answers the owner's
            // personal folder when the step names none), so the chain's own workspace is never
            // reached here.
            let workspace = place.as_ref().unwrap_or(origin);
            // The launch owns the folder rule: only it knows which conversation the address names,
            // so only it can tell the step's own turn from another agent's.
            crate::launch::custom_role(
                step.name.as_deref().unwrap_or_default(),
                message,
                step.branch.as_deref(),
                workspace,
                None,
                BusyFolder::Chain(chain),
                // The chain record owns the step's schema — see the worker-branch arm.
                None,
                answered_for,
            )
            .await
            .map(|_| ())
        }
        other => bail!("usage: `{}` cannot be a chain step", other.as_str()),
    }
}

/// Hand one step's answer on: dispatch the next step, or deliver the final answer to the chain's
/// caller. An ended chain says what became of it — silence only for a deliberate stop.
async fn hand_on(chain: &Chain, answer: &str) {
    if stopped(chain) {
        // Stopped: nothing is delivered, and the group went with the stop.
        return;
    }
    // Every other way the chain's records can disappear ends with the failure said out loud; a
    // read failure is not one of them, because delivering is the safe side.
    if crate::jobs::chain_has_records(&chain.id, 1..=chain.total()).await == Some(false) {
        fail(
            chain,
            "the chain's records are gone — there is nothing left to hand the answer to",
        )
        .await;
        return;
    }
    let Some(next) = chain.next.first() else {
        forget(&chain.id);
        deliver(chain, answer).await;
        consume(chain).await;
        return;
    };
    // The hand-off is idempotent: a replayed step whose answer was already handed on finds the
    // next step's record — its round, its queued envelope, or its conversation — and leaves it
    // alone.
    if crate::jobs::chain_has_records(&chain.id, [chain.step + 1]).await == Some(true) {
        consume(chain).await;
        return;
    }
    let producer = chain.current.label();
    let mut next_chain = chain.clone();
    next_chain.current = next.clone();
    next_chain.step += 1;
    next_chain.next = chain.next[1..].to_vec();
    let message = later_message(&next_chain, &producer, answer);
    let origin = match origin(&next_chain).await {
        Ok(origin) => origin,
        Err(e) => {
            fail(&next_chain, &e.to_string()).await;
            return;
        }
    };
    // `AnsweredFor::Caller`: a step is dispatched from outside any agent turn, so a hand-off that
    // could not be recorded is reported to the caller rather than swallowed as an agent's own
    // best-effort delivery would be.
    if let Err(e) = dispatch(&next_chain, &message, &origin, AnsweredFor::Caller).await {
        fail(&next_chain, &e.to_string()).await;
        return;
    }
    // The hand-off is one transition: the next step's record is durable before this one is
    // consumed, so a replay finds the next step instead of starting it twice.
    consume(chain).await;
}

/// What a step's turn came to: its answer, or none at all plus why. The one place the "hand the
/// answer on, or end the chain" rule lives — whichever path ran the step: a worker round, a branch
/// turn or a role turn. `conversation` is the session that produced the answer, whose recorded
/// model and provider the transformation follows. A turn with no answer is a failure inside a chain,
/// and `reason` is where the caller says why, so a generic sentence is never the only thing the
/// owner learns.
pub(crate) async fn settle(
    chain: &Chain,
    conversation: Option<&str>,
    answer: Option<&str>,
    reason: &str,
) {
    match answer {
        Some(answer) if !answer.trim().is_empty() => {
            // A step that asked for JSON hands JSON on; a transformation that fails forwards the
            // whole answer, and a stop landing meanwhile is caught by `hand_on`.
            let answer = crate::extract::answer_as_json(
                chain.current.extract.as_ref(),
                &crate::extract::StepContext {
                    role: chain.current.role(),
                    workspace: &chain.workspace,
                    conversation,
                    parent: Some(ParentKey::Chain(chain.id.clone())),
                },
                answer,
            )
            .await;
            hand_on(chain, &answer).await;
        }
        _ => fail(chain, reason).await,
    }
}

/// The step is over: its durable record goes, and its conversation stops being a chain step, so
/// nothing replays it and a later turn there is an ordinary turn again.
async fn consume(chain: &Chain) {
    if let Err(e) = crate::jobs::delete_chain_step(&record_id(&chain.id, chain.step)).await {
        tracing::warn!(
            chain = %chain.id,
            step = chain.step,
            error = %e,
            "Failed to consume a chain step's record",
        );
    }
    if let Err(e) = crate::jobs::clear_chain_step(&chain.id, chain.step).await {
        tracing::warn!(
            chain = %chain.id,
            step = chain.step,
            error = %e,
            "Failed to clear a chain step's conversation mark",
        );
    }
}

/// Deliver one message to the chain's caller, in the envelope the step running now already uses:
/// a `<worker-report>` for a worker or a branch, a `<custom-role-message>` for a role. A step that
/// never ran names the folder it would have worked in.
async fn deliver(chain: &Chain, message: &str) {
    let (folder, root) = match origin(chain).await {
        Ok(origin) => step_place(&chain.current, Some(&origin)).await,
        // The chain's own folder is gone: name the durable address rather than nothing.
        Err(_) => (chain.workspace.clone(), chain.workspace.clone()),
    };
    crate::agent::message_router::deliver_chain_message(&chain.current, message, &folder, &root)
        .await;
}

/// The folder an envelope for `step` names and the workspace it belongs to: the step's own `place`
/// when it names one, else `origin` — the chain's own workspace, where the same delegation from the
/// owner's Assistant runs. A folder that no longer resolves names the durable address instead.
async fn step_place(step: &Step, origin: Option<&Workspace>) -> (String, String) {
    match place_of(step).await.ok().flatten() {
        Some(ws) => (ws.path, ws.name),
        None => origin.map_or_else(
            || (String::new(), String::new()),
            |ws| (ws.path.clone(), ws.name.clone()),
        ),
    }
}

/// The one failure sentence a chain ends with: where in the chain it stopped, and why.
fn failure_message(position: &str, reason: &str) -> String {
    crate::prompt::substitute(
        &crate::prompt::load_prompt("chain_failure.md"),
        &[("{{position}}", position), ("{{reason}}", reason)],
    )
}

/// End a chain with one step-numbered failure to its caller, in the envelope that step's own
/// addressee uses today. A deliberate stop is the one ending that says nothing.
pub(crate) async fn fail(chain: &Chain, reason: &str) {
    if stopped(chain) {
        return;
    }
    let message = failure_message(&format!("Step {} of {}", chain.step, chain.total()), reason);
    deliver(chain, &message).await;
    end_quietly(&chain.id).await;
}

/// Abandon a chain step whose stored record cannot be read: nothing is ever handed on, and the
/// caller gets the one failure wording — step-numbered when the record id still names the step.
/// `step` is the address the step ran as, which decides the envelope; `record` is the id the value
/// was read under, when it is known. Returns `false` for a value that is no unreadable chain step,
/// so a caller can fall through to its ordinary path.
pub(crate) async fn abandon_unreadable(
    stored: &Stored,
    step: &Step,
    workspace: &str,
    record: Option<&str>,
) -> bool {
    if !stored.is_unreadable() {
        return false;
    }
    // A round's row names the chain AND the step; a conversation's corrupt pointer names only the
    // chain, and only when the value still says which.
    let named = record.and_then(crate::launch::record_parts);
    let chain_id = named
        .map(|(chain_id, _)| chain_id)
        .or(stored.unreadable_chain());
    tracing::error!(
        chain = ?chain_id,
        record = ?record,
        "a chain step's stored record is unreadable — the chain is abandoned",
    );
    let position = named.map_or_else(
        || "A chain step".to_string(),
        |(_, step)| format!("Step {step}"),
    );
    let message = failure_message(
        &position,
        "its stored record could not be read, so the chain stopped",
    );
    let (folder, root) = match crate::users::resolve_workspace(workspace)
        .await
        .ok()
        .flatten()
    {
        Some(origin) => step_place(step, Some(&origin)).await,
        None => (workspace.to_string(), workspace.to_string()),
    };
    crate::agent::message_router::deliver_chain_message(step, &message, &folder, &root).await;
    if let Some(chain_id) = chain_id {
        end_quietly(chain_id).await;
    }
    true
}

/// The one way a chain is torn down: its Running Agents group goes, every conversation of it is
/// retired — so none resumes as an ordinary delegation whose answer would reach the owner — and its
/// durable rows are dropped, so no step replays. Both halves run even when the first fails, and the
/// first failure is the one returned. A chain that finishes normally is not torn down: its last step
/// is consumed instead.
async fn end(chain_id: &str) -> Result<()> {
    forget(chain_id);
    let retired = crate::jobs::retire_chain_sessions(chain_id)
        .await
        .context("retire an ended chain's conversations");
    let dropped = crate::jobs::delete_chain_rows(chain_id)
        .await
        .context("drop an ended chain's rows");
    retired.and(dropped)
}

/// [`end`] from a path with nobody to report to: the failure is logged, and the chain is over
/// either way.
async fn end_quietly(chain_id: &str) {
    if let Err(e) = end(chain_id).await {
        tracing::warn!(chain = %chain_id, error = %e, "Failed to end a chain");
    }
}

async fn origin(chain: &Chain) -> Result<Workspace> {
    crate::users::resolve_workspace(&chain.workspace)
        .await?
        .ok_or_else(|| anyhow::anyhow!("the chain's folder '{}' no longer exists", chain.workspace))
}

/// Keep the chain visible in Running Agents for its whole life, including while a step waits in a
/// conversation's queue: a run-lifetime marker on the chain's own group. It lives in a process
/// global so it outlives the dispatch call, and is dropped when the chain ends or is stopped. The
/// stop lock is held across the registration, so a dispatch racing a stop cannot re-create a group
/// whose chain is already over.
pub(crate) fn keep_visible(chain: &Chain) {
    let marks = STOPPED.lock().unwrap_poison();
    if marks.contains(&chain.id) {
        return;
    }
    let guard = crate::agent::registry::NON_AGENT_CALLS.register(
        "chain",
        &chain.workspace,
        Some(crate::agent::registry::ParentKey::Chain(chain.id.clone())),
        true,
        Some(chain.task.clone()),
    );
    // The marker belongs to the chain, not to this call: hold it until the chain ends.
    CHAIN_VISIBLE
        .lock()
        .unwrap_poison()
        .insert(chain.id.clone(), guard);
}

/// The run-lifetime markers of the chains in flight, so the group stays visible between steps.
static CHAIN_VISIBLE: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, crate::agent::registry::NonAgentCallGuard>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// The chains the owner stopped in this process. A stop drops the chain's records and retires its
/// conversations, so a restart needs no mark; what still needs one is an envelope already sitting in
/// a conversation's queue, which would otherwise run its whole turn and then report the chain's
/// records as gone. A stop must stay stopped, so the mark has no deadline: it lives as long as the
/// process, and a re-issued call for the same chain clears it.
static STOPPED: std::sync::LazyLock<std::sync::Mutex<std::collections::HashSet<String>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashSet::new()));

/// Whether the owner stopped this chain. The one ending that says nothing.
pub(crate) fn stopped(chain: &Chain) -> bool {
    STOPPED.lock().unwrap_poison().contains(&chain.id)
}

/// Record a deliberate stop, so a step still queued in a conversation is dropped when it comes up
/// and the chain's group is never re-created. The mark has no deadline: a stop must stay stopped.
fn mark_stopped(chain_id: &str) {
    STOPPED.lock().unwrap_poison().insert(chain_id.to_string());
}

/// Drop the chain's run-lifetime marker: the chain is over or was stopped.
fn forget(chain_id: &str) {
    CHAIN_VISIBLE.lock().unwrap_poison().remove(chain_id);
}

/// Stop one chain: cancel the step running now, retire every conversation of it, drop its durable
/// rows so no step replays, and deliver nothing further. A step still waiting in a conversation's
/// queue is dropped when it comes up (see [`stopped`]). Another chain's step queued in the same
/// conversation is untouched — the rows are found by the chain's own id, never by the conversation.
pub(crate) async fn stop(chain_id: &str) -> Result<()> {
    // The mark keeps a step already queued from running, and keeps its group from coming back; the
    // step running now is what still needs cancelling.
    mark_stopped(chain_id);
    crate::agent::registry::AGENT_REGISTRY.cancel_by_parent_key(
        &crate::agent::registry::ParentKey::Chain(chain_id.to_string()),
    );
    end(chain_id).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn branch(branch: &str) -> Step {
        Step {
            addressee: Addressee::WorkerBranch,
            place: None,
            branch: Some(branch.to_string()),
            name: None,
            extract: None,
        }
    }

    fn role(name: &str) -> Step {
        Step {
            addressee: Addressee::Role,
            place: None,
            branch: None,
            name: Some(name.to_string()),
            extract: None,
        }
    }

    fn worker() -> Step {
        Step {
            addressee: Addressee::Worker,
            place: None,
            branch: None,
            name: None,
            extract: None,
        }
    }

    fn chain(
        id: &str,
        workspace: &str,
        step: usize,
        total: usize,
        current: Step,
        next: Vec<Step>,
    ) -> Chain {
        assert_eq!(
            step + next.len(),
            total,
            "a chain's total is the step running now plus the steps after it"
        );
        Chain {
            id: id.to_string(),
            task: "draft it".to_string(),
            user: crate::users::ADMIN_USER_NAME.to_string(),
            channel: "gui".to_string(),
            workspace: workspace.to_string(),
            current,
            step,
            next,
        }
    }

    /// Seed the durable record a running step owns, so the chain's liveness check sees it.
    async fn seed_step(chain: &Chain, ws: &crate::Workspace) {
        crate::jobs::spawn_job(
            &crate::session::store().conn,
            &record_id(&chain.id, chain.step),
            "the step's task",
            &ws.name,
            &chain.user,
            &chain.channel,
            crate::Role::Worker,
            &[],
            &crate::jobs::SpawnChild::Implement,
            None,
            None,
            None,
        )
        .await
        .expect("seed the running step's row");
    }

    #[test]
    fn a_malformed_for_is_refused_whole() {
        for bad in [
            json!("not json at all"),
            json!({"addressee": "worker", "message": "x"}),
            json!({"addressee": "worker", "unknown": "x"}),
            json!({"addressee": "worker", "branch": "b"}),
            json!({"addressee": "worker_branch"}),
            json!({"addressee": "role"}),
            json!({"addressee": "manager", "place": "p"}),
            json!({"addressee": "guest", "name": "g"}),
            json!({"addressee": "assistant"}),
            json!({"addressee": "nobody"}),
            json!({"place": "p"}),
        ] {
            assert!(parse(&bad).is_err(), "must refuse {bad}");
        }
        assert_eq!(
            parse(&json!({
                "addressee": "role",
                "name": "editor",
                "for": {"addressee": "worker_branch", "branch": "publish"},
            }))
            .expect("a well-formed chain"),
            vec![role("editor"), branch("publish"),],
        );
        // An empty value is an absent one, exactly as an empty flag on the command line is.
        assert_eq!(
            parse(&json!({"addressee": "worker", "place": "  "})).expect("an empty place is none"),
            vec![worker()],
        );
    }

    /// A step's schema is read from the step that names it, in either form a call can carry it —
    /// the object an agent passes and the JSON text the command line passes.
    #[test]
    fn a_step_reads_the_schema_it_names() {
        let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let expected = crate::extract::Extract::from_arg(&schema).expect("the schema is accepted");
        let parsed: Vec<Option<crate::extract::Extract>> =
            [schema.clone(), json!(schema.to_string())]
                .into_iter()
                .map(|given| {
                    let steps = parse(&json!({"addressee": "worker", "extract": given}))
                        .expect("a well-formed step");
                    steps[0].extract.clone()
                })
                .collect();
        assert_eq!(
            parsed,
            vec![Some(expected.clone()), Some(expected)],
            "a step carries the schema it names, in either form"
        );
        // A schema a step cannot use is refused whole, before anything runs.
        for bad in [
            json!({"addressee": "worker", "extract": "not json"}),
            json!({"addressee": "worker", "extract": {"type": "object", "allOf": []}}),
        ] {
            assert!(parse(&bad).is_err(), "must refuse {bad}");
        }
    }

    #[test]
    fn a_chain_is_at_most_ten_steps() {
        // The `for` value holds steps two onwards: nine of them plus the call is ten.
        let mut nested = json!({"addressee": "worker"});
        for _ in 0..(MAX_STEPS - 2) {
            nested = json!({"addressee": "worker", "for": nested});
        }
        assert_eq!(
            parse(&nested).expect("nine steps after the call").len(),
            MAX_STEPS - 1
        );
        let too_deep = json!({"addressee": "worker", "for": nested});
        assert!(
            parse(&too_deep).is_err(),
            "ten steps after the call is one too many"
        );
    }

    #[test]
    fn the_chain_id_is_the_call_so_a_replay_resumes_it() {
        let outer = branch("one");
        let next = vec![role("editor")];
        assert_eq!(
            id_of("draft it", &outer, &next),
            id_of("draft it", &outer, &next)
        );
        assert_ne!(
            id_of("draft it", &outer, &next),
            id_of("edit it", &outer, &next)
        );
        assert_ne!(
            id_of("draft it", &outer, &next),
            id_of("draft it", &branch("two"), &next)
        );
    }

    /// A step's schema is part of the chain's identity, and two schemas differing only in key order
    /// or whitespace are one schema — so a re-issued call finds its chain either way. A call that
    /// carries no schema keeps exactly the identity it had before the argument existed.
    #[test]
    fn a_steps_schema_is_part_of_the_chains_identity() {
        let plain = id_of("draft it", &branch("one"), &[role("editor")]);
        assert_eq!(
            plain,
            id_of("draft it", &branch("one"), &[role("editor")]),
            "two schema-less calls are one chain"
        );
        // Exactly the identity the call had before the argument existed, so a chain started by an
        // older build is still found by the same call.
        assert_eq!(plain, "aa0f6e84e78c572f");

        let with_schema = |text: &str| Step {
            extract: Some(
                crate::extract::Extract::from_text(text).expect("the schema is accepted"),
            ),
            ..branch("one")
        };
        let first = with_schema(r#"{"type":"object","properties":{"a":{"type":"string"}}}"#);
        let reordered =
            with_schema(r#"{ "properties": { "a": { "type": "string" } }, "type": "object" }"#);
        assert_ne!(
            id_of("draft it", &first, &[role("editor")]),
            plain,
            "a schema changes the chain's identity"
        );
        assert_eq!(
            id_of("draft it", &first, &[role("editor")]),
            id_of("draft it", &reordered, &[role("editor")]),
            "key order and whitespace are not identity"
        );
        assert_ne!(
            id_of("draft it", &first, &[role("editor")]),
            id_of(
                "draft it",
                &with_schema(r#"{"type":"string"}"#),
                &[role("editor")]
            ),
            "a different schema is a different chain"
        );
        // A schema on a step AFTER the call is the chain's identity too.
        let later = Step {
            extract: Some(crate::extract::Extract::from_text(r#"{"type":"string"}"#).expect("ok")),
            ..role("editor")
        };
        assert_ne!(
            id_of("draft it", &branch("one"), &[later]),
            plain,
            "a schema on a later step changes the identity as well"
        );
    }

    /// A field a stored record carries and this build does not know makes the whole record
    /// unreadable, never silently reduced: a step that carried a schema must not come back as an
    /// ordinary delegation returning prose.
    #[test]
    fn an_unknown_field_makes_a_stored_record_unreadable() {
        let base = json!({
            "id": "drifted",
            "task": "draft it",
            "user": crate::users::ADMIN_USER_NAME,
            "channel": "gui",
            "workspace": "ws",
            "current": {"addressee": "worker"},
            "step": 1,
        });
        assert!(
            Stored::parse(Some(base.to_string())).step().is_some(),
            "the shape this build knows reads as a step"
        );
        let mut drifted = base.clone();
        drifted["current"]["unknown"] = json!(1);
        assert!(
            Stored::parse(Some(drifted.to_string())).is_unreadable(),
            "an unknown field of a step"
        );
        let mut drifted = base;
        drifted["unknown"] = json!(1);
        assert!(
            Stored::parse(Some(drifted.to_string())).is_unreadable(),
            "an unknown field of the chain itself"
        );

        // The envelopes that carry a schema fail to decode whole for the same reason.
        let launch = crate::Launch {
            role: crate::Role::Worker,
            name: String::new(),
            identifier: None,
            prompt: String::new(),
            tools: Vec::new(),
            agent_id: crate::session::worker_branch_agent_id("drifted"),
            reply_to: None,
            chain: None,
            extract: None,
        };
        let mut value = serde_json::to_value(&launch).expect("a launch serializes");
        value["unknown"] = json!(1);
        assert!(
            serde_json::from_value::<crate::Launch>(value).is_err(),
            "a launch carrying an unknown field is refused whole"
        );
        let job = crate::agent::message_router::AgentJob {
            content: "draft it".to_string(),
            workspace_name: "ws".to_string(),
            user_name: crate::users::ADMIN_USER_NAME.to_string(),
            channel: "gui".to_string(),
            kind: crate::agent::message_router::MessageKind::UserMessage,
            role: crate::Role::Worker,
            reply_target: None,
            pending_job_id: None,
            originating_workspace: None,
            custom_role: None,
            reply_to_custom: None,
            urgent: false,
            extract: None,
        };
        let mut value = serde_json::to_value(&job).expect("an envelope serializes");
        value["unknown"] = json!(1);
        assert!(
            serde_json::from_value::<crate::agent::message_router::AgentJob>(value).is_err(),
            "an envelope carrying an unknown field is refused whole"
        );
    }

    /// A stored chain value has three states, and "unreadable" is never "no chain": a step whose
    /// record is corrupt must be abandoned, not delivered as an ordinary delegation.
    #[test]
    fn an_unreadable_chain_record_is_never_an_ordinary_round() {
        assert!(matches!(Stored::parse(None), Stored::Ordinary));
        let unreadable = Stored::parse(Some("{not json".to_string()));
        assert!(unreadable.is_unreadable());
        assert!(unreadable.step().is_none());
        assert_eq!(
            unreadable.unreadable_chain(),
            None,
            "a value nothing about which is legible names no chain"
        );
        // A record this build cannot read as a chain — an older or newer shape — still names its
        // chain, so the ending can retire that chain's conversations.
        let drifted = Stored::parse(Some(
            json!({"id": "drifted", "task": "draft it", "steps": []}).to_string(),
        ));
        assert!(drifted.is_unreadable());
        assert_eq!(drifted.unreadable_chain(), Some("drifted"));
    }

    /// A schema that can be filled replaces the step's answer with the JSON document: the caller
    /// receives JSON as the chain's final answer, and the prose is gone.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn a_schema_that_can_be_filled_replaces_the_answer() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("chain_filled")
                .to_string_lossy(),
            "chain_filled_ws",
        )
        .await;
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = std::sync::Arc::new(crate::util::test::FakeProvider::new().ok(r#"{"a":"one"}"#));
        let _provider_guard = crate::util::test::install_fake_provider(fake.clone());
        let mut current = role("writer");
        current.extract = Some(
            crate::extract::Extract::from_text(
                r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}"#,
            )
            .expect("the schema is accepted"),
        );
        let filled = chain("filled", "chain_filled_ws", 1, 1, current, Vec::new());
        seed_step(&filled, &ws).await;
        let admin = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);
        let mut admin_rx = crate::agent::message_router::register_agent(&admin);

        settle(&filled, None, Some("the whole answer"), "no answer").await;

        let job = tokio::time::timeout(std::time::Duration::from_secs(2), admin_rx.recv())
            .await
            .expect("the final answer is delivered")
            .expect("the final envelope");
        assert!(
            job.content.contains(r#"{"a":"one"}"#),
            "the JSON document is the answer: {}",
            job.content
        );
        assert!(
            !job.content.contains("the whole answer"),
            "the prose is gone: {}",
            job.content
        );
        assert_eq!(
            fake.sent.lock().unwrap().len(),
            1,
            "one conforming reply is enough"
        );
        crate::agent::message_router::unregister_agent(&admin);
    }

    /// A schema that cannot be filled forwards the step's whole answer unchanged: the step's work is
    /// never lost, and the chain is never failed because of it.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn a_schema_that_cannot_be_filled_forwards_the_whole_answer() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("chain_failopen")
                .to_string_lossy(),
            "chain_failopen_ws",
        )
        .await;
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = std::sync::Arc::new(
            crate::util::test::FakeProvider::new()
                .ok("not json at all")
                .ok("not json at all")
                .ok("not json at all"),
        );
        let _provider_guard = crate::util::test::install_fake_provider(fake.clone());
        let admin = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);
        let mut admin_rx = crate::agent::message_router::register_agent(&admin);

        let mut current = role("writer");
        current.extract = Some(
            crate::extract::Extract::from_text(
                r#"{"type":"object","properties":{"a":{"type":"string"}}}"#,
            )
            .expect("the schema is accepted"),
        );
        let failopen = chain("failopen", "chain_failopen_ws", 1, 1, current, Vec::new());
        seed_step(&failopen, &ws).await;
        settle(&failopen, None, Some("the whole answer"), "no answer").await;
        let job = tokio::time::timeout(std::time::Duration::from_secs(2), admin_rx.recv())
            .await
            .expect("the final answer is delivered")
            .expect("the final envelope");
        assert!(
            job.content.contains("the whole answer"),
            "an answer that cannot be turned into JSON is forwarded whole: {}",
            job.content
        );
        assert_eq!(
            fake.sent.lock().unwrap().len(),
            3,
            "the transformation spends its own attempt budget, then gives up"
        );

        // A step that asked for nothing is handed on untouched, with no call at all.
        let plain = chain(
            "plain",
            "chain_failopen_ws",
            1,
            1,
            role("writer"),
            Vec::new(),
        );
        seed_step(&plain, &ws).await;
        settle(&plain, None, Some("the whole answer"), "no answer").await;
        let job = tokio::time::timeout(std::time::Duration::from_secs(2), admin_rx.recv())
            .await
            .expect("the final answer is delivered")
            .expect("the final envelope");
        assert!(job.content.contains("the whole answer"));
        assert_eq!(
            fake.sent.lock().unwrap().len(),
            3,
            "a step with no schema makes no call"
        );
        crate::agent::message_router::unregister_agent(&admin);
    }

    /// A worker step's `jobs` row carries the chain and the chain's own identity, so a restart
    /// resumes the same chain instead of delivering the step's answer to the caller.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn a_worker_step_row_carries_the_chain() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("chain_row")
                .to_string_lossy(),
            "chain_row_ws",
        )
        .await;
        let chain = chain(
            "rowstep",
            "chain_row_ws",
            2,
            3,
            worker(),
            vec![branch("three")],
        );
        let job_id = record_id(&chain.id, 2);
        let args = crate::tools::CoreJobArgs {
            job_id: &job_id,
            caller_role: crate::Role::Assistant,
            user_name: &chain.user,
            channel: &chain.channel,
            spawn: Some(crate::tools::SpawnRecord {
                delivery_name: &ws.name,
                exec_dir: None,
                custom_launch: None,
            }),
            caller_agent_id: None,
            fail_on_checkpoint_error: false,
            chain: Some(&chain),
            extract: None,
        };
        crate::tools::implement::spawn_implement_rows(&ws, "draft it", &args)
            .await
            .expect("the step's row is written");

        let row = crate::session::store()
            .conn
            .query(
                "SELECT chain, user_name, channel FROM jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .expect("read the step's row");
        assert_eq!(row.len(), 1, "the step has its own durable row");
        let stored = Stored::parse(row[0].get::<Option<String>>(0).expect("a chain value"));
        assert_eq!(
            stored.step().map(|c| (c.step, c.total())),
            Some((2, 3)),
            "the row carries the chain and where in it the step is"
        );
        assert_eq!(row[0].get::<String>(1).unwrap(), chain.user);
        assert_eq!(row[0].get::<String>(2).unwrap(), chain.channel);

        crate::jobs::delete_chain_rows(&chain.id)
            .await
            .expect("clean up this test's chain");
    }

    /// A worker step runs from its own row and hands its answer on: the next step receives the
    /// original task and the whole answer, and the finished step's record is consumed.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn a_worker_step_hands_its_answer_on() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let _provider_guard = crate::util::test::install_fake_provider(std::sync::Arc::new(
            crate::util::test::FakeProvider::new().ok("the worker's answer"),
        ));
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("chain_worker")
                .to_string_lossy(),
            "chain_worker_ws",
        )
        .await;
        let chain = chain(
            "workerstep",
            "chain_worker_ws",
            1,
            2,
            worker(),
            vec![branch("two")],
        );
        let mut next_rx = crate::agent::message_router::register_agent(
            &crate::session::worker_branch_agent_id("two"),
        );

        dispatch(&chain, "draft it", &ws, AnsweredFor::Caller)
            .await
            .expect("the worker step dispatches");

        let job = tokio::time::timeout(std::time::Duration::from_secs(10), next_rx.recv())
            .await
            .expect("the worker's answer is handed on")
            .expect("the next step's envelope");
        assert!(
            job.content.contains("the worker's answer"),
            "the next step receives the whole answer: {}",
            job.content
        );
        assert!(
            job.content.contains("draft it"),
            "and the original task: {}",
            job.content
        );
        assert_eq!(
            job.custom_role
                .as_ref()
                .and_then(|launch| launch.chain.as_ref())
                .map(|c| c.step),
            Some(2),
            "the next step knows where it is"
        );
        assert_eq!(
            crate::jobs::chain_has_records(&chain.id, [1]).await,
            Some(false),
            "the finished step's record is consumed once its answer was handed on"
        );

        crate::agent::message_router::unregister_agent(&crate::session::worker_branch_agent_id(
            "two",
        ));
        crate::jobs::delete_chain_rows(&chain.id)
            .await
            .expect("clean up this test's chain");
    }

    /// A step's answer goes straight on to the next step — never to the conversation that started
    /// the chain — and the next step's durable record is written before it runs.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn an_answer_goes_to_the_next_step_not_the_caller() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("chain_next")
                .to_string_lossy(),
            "chain_next_ws",
        )
        .await;
        let chain = chain(
            "nextstep",
            "chain_next_ws",
            1,
            2,
            branch("one"),
            vec![branch("two")],
        );
        seed_step(&chain, &ws).await;

        let mut next_rx = crate::agent::message_router::register_agent(
            &crate::session::worker_branch_agent_id("two"),
        );
        let admin = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);
        let mut admin_rx = crate::agent::message_router::register_agent(&admin);

        hand_on(&chain, "the drafted text").await;

        let job = tokio::time::timeout(std::time::Duration::from_secs(2), next_rx.recv())
            .await
            .expect("the next step is dispatched")
            .expect("the next step's envelope");
        assert!(
            job.content.contains("draft it"),
            "the task travels unchanged: {}",
            job.content
        );
        assert!(
            job.content.contains("the drafted text"),
            "the previous step's whole answer travels: {}",
            job.content
        );
        assert_eq!(
            job.custom_role
                .as_ref()
                .and_then(|launch| launch.chain.as_ref())
                .map(|c| c.step),
            Some(2),
            "the next step knows where it is"
        );
        assert!(
            admin_rx.try_recv().is_err(),
            "an intermediate answer never reaches the conversation that started the chain"
        );
        assert_eq!(
            crate::jobs::chain_has_records(&chain.id, [2]).await,
            Some(true),
            "the next step's record is durable before it runs"
        );
        crate::agent::message_router::unregister_agent(&crate::session::worker_branch_agent_id(
            "two",
        ));
        crate::agent::message_router::unregister_agent(&admin);
    }

    /// The last step's answer comes back to the conversation the chain started in, in the envelope
    /// that addressee already uses.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn the_last_answer_comes_back_to_the_conversation_that_started_it() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("chain_last")
                .to_string_lossy(),
            "chain_last_ws",
        )
        .await;
        let chain = chain(
            "laststep",
            "chain_last_ws",
            2,
            2,
            role("writer"),
            Vec::new(),
        );
        seed_step(&chain, &ws).await;

        let admin = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);
        let mut admin_rx = crate::agent::message_router::register_agent(&admin);

        hand_on(&chain, "the final text").await;

        let job = tokio::time::timeout(std::time::Duration::from_secs(2), admin_rx.recv())
            .await
            .expect("the final answer is delivered")
            .expect("the final envelope");
        assert!(job.content.contains("<custom-role-message"));
        assert!(job.content.contains("the final text"));
        crate::agent::message_router::unregister_agent(&admin);
    }

    /// A next step that cannot be dispatched (a role deleted since the chain was sent) fails the
    /// chain, naming the step that could not run.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn a_step_that_cannot_be_dispatched_fails_the_chain() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("chain_dispatch_fail")
                .to_string_lossy(),
            "chain_dispatch_fail_ws",
        )
        .await;
        let chain = chain(
            "dispatch_fail",
            "chain_dispatch_fail_ws",
            1,
            2,
            branch("one"),
            vec![role("deleted_since")],
        );
        seed_step(&chain, &ws).await;

        let admin = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);
        let mut admin_rx = crate::agent::message_router::register_agent(&admin);

        hand_on(&chain, "the drafted text").await;

        let job = tokio::time::timeout(std::time::Duration::from_secs(2), admin_rx.recv())
            .await
            .expect("the chain's failure is delivered")
            .expect("the failure envelope");
        assert!(
            job.content.contains("Step 2 of 2 failed"),
            "{}",
            job.content
        );
        assert!(job.content.contains("deleted_since"), "{}", job.content);
        crate::agent::message_router::unregister_agent(&admin);
    }

    /// A failing step stops the chain with one message naming the step and the reason.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn a_failed_step_names_its_number_and_reason() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("chain_fail")
                .to_string_lossy(),
            "chain_fail_ws",
        )
        .await;
        let chain = chain(
            "failing",
            "chain_fail_ws",
            2,
            3,
            branch("two"),
            vec![branch("three")],
        );
        seed_step(&chain, &ws).await;

        let admin = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);
        let mut admin_rx = crate::agent::message_router::register_agent(&admin);

        fail(&chain, "the folder is gone").await;

        let job = tokio::time::timeout(std::time::Duration::from_secs(2), admin_rx.recv())
            .await
            .expect("the failure is delivered")
            .expect("the failure envelope");
        assert!(
            job.content
                .contains("Step 2 of 3 failed: the folder is gone"),
            "{}",
            job.content
        );
        assert!(
            job.content.contains("<worker-report"),
            "the step's own envelope"
        );
        crate::agent::message_router::unregister_agent(&admin);
    }

    /// Stopping a chain drops only its own durable records: another chain's step waiting in the
    /// same conversation still runs.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn stopping_a_chain_drops_only_its_own_rows() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("chain_stop")
                .to_string_lossy(),
            "chain_stop_ws",
        )
        .await;
        let mine = chain(
            "mine",
            "chain_stop_ws",
            1,
            2,
            branch("shared"),
            vec![branch("next")],
        );
        let theirs = chain(
            "theirs",
            "chain_stop_ws",
            1,
            2,
            branch("shared"),
            vec![branch("next")],
        );
        seed_step(&mine, &ws).await;
        seed_step(&theirs, &ws).await;
        // A step still waiting in another conversation's queue has no `jobs` row yet — only its
        // durable envelope. Stopping the chain must drop that too, and nothing of the other chain.
        for chain in [&mine, &theirs] {
            let mut queued = chain.clone();
            queued.current = chain.next[0].clone();
            queued.step = 2;
            queued.next = Vec::new();
            let job = crate::agent::message_router::AgentJob {
                content: "queued".to_string(),
                workspace_name: ws.name.clone(),
                user_name: chain.user.clone(),
                channel: "gui".to_string(),
                kind: crate::agent::message_router::MessageKind::UserMessage,
                role: crate::Role::Worker,
                reply_target: None,
                pending_job_id: None,
                originating_workspace: None,
                custom_role: Some(crate::Launch {
                    role: crate::Role::Worker,
                    name: String::new(),
                    identifier: Some("shared".to_string()),
                    prompt: String::new(),
                    tools: Vec::new(),
                    agent_id: crate::session::worker_branch_agent_id("shared"),
                    reply_to: None,
                    chain: Some(Box::new(queued)),
                    extract: None,
                }),
                reply_to_custom: None,
                urgent: false,
                extract: None,
            };
            crate::agent::message_router::persist_pending(&job, record_id(&chain.id, 2))
                .await
                .expect("queue the step");
        }

        stop(&mine.id).await.expect("stop the chain");

        assert_eq!(
            crate::jobs::chain_has_records(&mine.id, [1]).await,
            Some(false),
            "the stopped chain's running step is dropped"
        );
        assert_eq!(
            crate::jobs::chain_has_records(&mine.id, [2]).await,
            Some(false),
            "and its step still waiting in a conversation's queue"
        );
        assert_eq!(
            crate::jobs::chain_has_records(&theirs.id, [1]).await,
            Some(true),
            "another chain's step in the same conversation is untouched"
        );
        assert_eq!(
            crate::jobs::chain_has_records(&theirs.id, [2]).await,
            Some(true),
            "nor is its step still waiting in that conversation"
        );
    }

    /// Ending a chain is loud in every case but a stop: a chain whose records are gone delivers the
    /// one step-numbered failure, so it never ends with nothing said.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn a_chain_whose_records_are_gone_says_so() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("chain_vanished")
                .to_string_lossy(),
            "chain_vanished_ws",
        )
        .await;
        let chain = chain("vanished", &ws.name, 2, 2, branch("two"), Vec::new());
        let admin = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);
        let mut admin_rx = crate::agent::message_router::register_agent(&admin);

        hand_on(&chain, "the drafted text").await;

        let job = tokio::time::timeout(std::time::Duration::from_secs(2), admin_rx.recv())
            .await
            .expect("the chain's failure is delivered")
            .expect("the failure envelope");
        assert!(
            job.content.contains("Step 2 of 2 failed"),
            "{}",
            job.content
        );
        assert!(job.content.contains("records are gone"), "{}", job.content);
        assert!(
            !job.content.contains("the drafted text"),
            "the answer of a chain that cannot continue is dropped, not delivered: {}",
            job.content
        );
        crate::agent::message_router::unregister_agent(&admin);
    }

    /// Stopping a chain retires the conversation of the step it was mid-turn in: the poller never
    /// resumes it as an ordinary branch, and the chain reports itself stopped rather than vanished.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn stopping_a_chain_retires_its_step_conversation() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("chain_stop_retire")
                .to_string_lossy(),
            "chain_stop_retire_ws",
        )
        .await;
        let chain = chain(
            "retired",
            "chain_stop_retire_ws",
            1,
            2,
            branch("editor"),
            vec![branch("next")],
        );
        seed_step(&chain, &ws).await;
        let conversation = crate::session::worker_branch_agent_id("editor");
        crate::session::store()
            .set_branch(
                &conversation,
                None,
                Some("editor"),
                None,
                Some(&chain),
                None,
            )
            .await
            .expect("the step's conversation records its chain");

        stop(&chain.id).await.expect("stop the chain");

        // The stopped chain delivers nothing further, whatever a step of it finishes with.
        assert!(stopped(&chain));
        assert!(
            crate::session::store()
                .get_launch_stopped(&conversation)
                .await,
            "the stopped step's conversation is never restarted"
        );
        let recorded = crate::session::store()
            .conn
            .query_optional(
                "SELECT chain FROM session_metadata WHERE agent_id = ?1",
                crate::db::params![conversation.clone()],
                |row| row.get::<Option<String>>(0),
            )
            .await
            .expect("read the step's conversation")
            .flatten();
        assert_eq!(recorded, None, "and it is an ordinary branch again");

        // A re-issued call starts a fresh chain: the stop's mark belonged to the chain that was
        // running, and this call asks for another.
        let mut rx = crate::agent::message_router::register_agent(&conversation);
        start(chain.clone(), &ws, AnsweredFor::Caller)
            .await
            .expect("the re-issued chain starts");
        assert!(!stopped(&chain), "the re-issued chain is not born stopped");
        let job = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("the fresh chain's step one is dispatched")
            .expect("the step's envelope");
        assert!(
            job.custom_role
                .as_ref()
                .and_then(|launch| launch.chain.as_ref())
                .is_some(),
            "and it carries the chain it starts"
        );
        crate::agent::message_router::unregister_agent(&conversation);
        crate::jobs::delete_chain_rows(&chain.id)
            .await
            .expect("clean up the re-issued chain");
    }

    /// A stop is not undone by a dispatch that races it: the chain's group never comes back, so the
    /// owner is not offered a Stop for a chain that is already over.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn a_stopped_chain_does_not_get_its_group_back() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let chain = chain(
            "stopped_group",
            &crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME),
            1,
            2,
            worker(),
            vec![worker()],
        );
        assert!(!visible(&chain), "nothing is running yet");
        keep_visible(&chain);
        assert!(visible(&chain), "a chain in flight is one group");

        stop(&chain.id).await.expect("stop the chain");

        assert!(!visible(&chain), "the stop drops the group");
        keep_visible(&chain);
        assert!(
            !visible(&chain),
            "a dispatch that raced the stop cannot re-create it"
        );
    }

    /// Whether the chain's Running Agents group exists.
    fn visible(chain: &Chain) -> bool {
        crate::agent::registry::NON_AGENT_CALLS
            .list()
            .iter()
            .any(|call| {
                call.parent_key == Some(crate::agent::registry::ParentKey::Chain(chain.id.clone()))
            })
    }

    /// A step that produced no answer ends the chain with the agent's own reason — the one failure
    /// the owner sees is the only place that reason can come from.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn a_step_without_an_answer_names_the_agents_reason() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("chain_reason")
                .to_string_lossy(),
            "chain_reason_ws",
        )
        .await;
        let chain = chain("reasoned", &ws.name, 1, 1, branch("editor"), Vec::new());
        let admin = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);
        let mut admin_rx = crate::agent::message_router::register_agent(&admin);

        settle(&chain, None, None, "the provider refused the request (401)").await;

        let job = tokio::time::timeout(std::time::Duration::from_secs(2), admin_rx.recv())
            .await
            .expect("the chain's failure is delivered")
            .expect("the failure envelope");
        assert!(
            job.content.contains("Step 1 of 1 failed"),
            "{}",
            job.content
        );
        assert!(
            job.content
                .contains("the provider refused the request (401)"),
            "the reason the step failed is what the owner is told: {}",
            job.content
        );
        crate::agent::message_router::unregister_agent(&admin);
    }

    /// A queued envelope is a step's record too: a chain whose only trace is the envelope waiting in
    /// a conversation's queue is still in flight, and a role or branch step's conversation is a
    /// record on its own.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn a_queued_envelope_or_a_conversation_keeps_a_chain_in_flight() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("chain_records")
                .to_string_lossy(),
            "chain_records_ws",
        )
        .await;
        let chain = chain(
            "records",
            &ws.name,
            1,
            2,
            branch("one"),
            vec![branch("two")],
        );
        assert_eq!(
            crate::jobs::chain_has_records(&chain.id, 1..=chain.total()).await,
            Some(false),
            "a chain nothing was written for is not in flight"
        );
        // Only the queued envelope of step one exists.
        let job = crate::agent::message_router::AgentJob {
            content: "queued".to_string(),
            workspace_name: ws.name.clone(),
            user_name: chain.user.clone(),
            channel: chain.channel.clone(),
            kind: crate::agent::message_router::MessageKind::UserMessage,
            role: crate::Role::Worker,
            reply_target: None,
            pending_job_id: None,
            originating_workspace: None,
            custom_role: None,
            reply_to_custom: None,
            urgent: false,
            extract: None,
        };
        crate::agent::message_router::persist_pending(&job, record_id(&chain.id, 1))
            .await
            .expect("queue step one");
        assert_eq!(
            crate::jobs::chain_has_records(&chain.id, 1..=chain.total()).await,
            Some(true),
            "a queued envelope is a step's durable record"
        );
        crate::jobs::delete_chain_step(&record_id(&chain.id, 1))
            .await
            .expect("consume the envelope");
        // Only the conversation of step two exists — a role or worker-branch step has no round.
        let conversation = crate::session::worker_branch_agent_id("two");
        let second = Chain {
            step: 2,
            next: Vec::new(),
            ..chain.clone()
        };
        crate::session::store()
            .set_branch(&conversation, None, Some("two"), None, Some(&second), None)
            .await
            .expect("the step's conversation records its chain");
        assert_eq!(
            crate::jobs::chain_has_records(&chain.id, 1..=chain.total()).await,
            Some(true),
            "a step's conversation is its record when it has no round of its own"
        );
        // And it is what makes a replayed hand-off idempotent: the next step already has a record,
        // so it is not dispatched a second time.
        assert_eq!(
            crate::jobs::chain_has_records(&chain.id, [2]).await,
            Some(true),
            "the hand-off's idempotency check sees a step that only a conversation records"
        );
        crate::jobs::retire_chain_session(&conversation)
            .await
            .expect("retire the test's conversation");
    }

    /// An unreadable chain ends once: the caller gets the one step-numbered failure, and every
    /// conversation of the chain is retired so a sibling step's death cannot report the ending a
    /// second time.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn an_unreadable_chain_ends_once_and_retires_its_conversations() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("chain_unreadable")
                .to_string_lossy(),
            "chain_unreadable_ws",
        )
        .await;
        let chain = chain(
            "corrupt",
            &ws.name,
            2,
            3,
            branch("two"),
            vec![branch("three")],
        );
        let sibling = crate::session::worker_branch_agent_id("one");
        crate::session::store()
            .set_branch(&sibling, None, Some("one"), None, Some(&chain), None)
            .await
            .expect("an earlier step's conversation records the chain");
        let admin = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);
        let mut admin_rx = crate::agent::message_router::register_agent(&admin);

        assert!(
            abandon_unreadable(
                &Stored::Unreadable(None),
                &branch("two"),
                &ws.name,
                Some(&record_id(&chain.id, 2)),
            )
            .await,
            "an unreadable step is abandoned"
        );

        let job = tokio::time::timeout(std::time::Duration::from_secs(2), admin_rx.recv())
            .await
            .expect("the chain's failure is delivered")
            .expect("the failure envelope");
        assert!(job.content.contains("Step 2 failed"), "{}", job.content);
        assert!(
            job.content.contains("stored record could not be read"),
            "{}",
            job.content
        );
        assert!(
            crate::session::store().get_launch_stopped(&sibling).await,
            "a sibling step's conversation is retired with the chain"
        );
        assert!(
            crate::jobs::chain_sessions(&chain.id)
                .await
                .expect("read the chain's conversations")
                .is_empty(),
            "so no step of it can report the ending again"
        );
        assert_eq!(
            crate::jobs::chain_has_records(&chain.id, 1..=chain.total()).await,
            Some(false),
            "and its rows are gone"
        );
        crate::agent::message_router::unregister_agent(&admin);
    }

    /// A conversation's corrupt chain pointer names no step, but the value still names its chain:
    /// the ending retires that chain's other conversations and drops its rows, so a sibling step
    /// cannot report the ending a second time and the group does not linger.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn an_unreadable_pointer_still_ends_its_chain() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("chain_pointer")
                .to_string_lossy(),
            "chain_pointer_ws",
        )
        .await;
        let chain = chain(
            "pointer",
            &ws.name,
            2,
            3,
            branch("two"),
            vec![branch("three")],
        );
        keep_visible(&chain);
        let sibling = crate::session::worker_branch_agent_id("one");
        crate::session::store()
            .set_branch(&sibling, None, Some("one"), None, Some(&chain), None)
            .await
            .expect("an earlier step's conversation records the chain");
        let admin = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);
        let mut admin_rx = crate::agent::message_router::register_agent(&admin);
        // A record this build cannot read as a chain, but which still says which chain it is.
        let corrupt = Stored::parse(Some(
            json!({"id": chain.id, "task": "draft it", "steps": []}).to_string(),
        ));

        assert!(
            abandon_unreadable(&corrupt, &branch("two"), &ws.name, None).await,
            "an unreadable pointer is abandoned"
        );

        let job = tokio::time::timeout(std::time::Duration::from_secs(2), admin_rx.recv())
            .await
            .expect("the chain's failure is delivered")
            .expect("the failure envelope");
        assert!(
            job.content.contains("A chain step failed"),
            "{}",
            job.content
        );
        assert!(
            crate::session::store().get_launch_stopped(&sibling).await,
            "a sibling step's conversation is retired with the chain"
        );
        assert_eq!(
            crate::jobs::chain_has_records(&chain.id, 1..=chain.total()).await,
            Some(false),
            "and its rows are gone"
        );
        assert!(!visible(&chain), "nor does its Running Agents group linger");
        crate::agent::message_router::unregister_agent(&admin);
    }
}
