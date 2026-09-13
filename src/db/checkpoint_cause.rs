//! The engine's own reason for a failed WAL checkpoint, kept for the product's
//! failure record.
//!
//! The engine reports a failed checkpoint as a *successful* statement whose row
//! is always the same three values (`busy=1`, `NULL`, `NULL`), so
//! [`crate::db::Connection::run_checkpoint`] fails with the same constant text
//! for a corrupt store and a healthy one. The engine's real reason is emitted
//! only by a debug-level event inside `turso_core`'s VDBE execution
//! ([`ENGINE_TARGET`]), which the product's logging filter drops. This module
//! keeps that one event — and nothing else — via a narrow capture layer, armed
//! per engine call by a [`CauseSink`] so a cause is never stale and never belongs
//! to another store's call, and attaches it to the checkpoint error instead of
//! emitting it as a log record of its own.
//!
//! That reason carries two duties: the failure record shows it to the operator
//! ([`crate::db::failure_record`]), and [`is_blocked_checkpoint`] uses it to tell a
//! blocked attempt, which never counts towards a stop, from a genuine pager error,
//! which a persistent failure window turns into the stop
//! ([`crate::db::checkpoint`]).
//!
//! Accepted cost: enabling DEBUG on that engine module enables its other debug
//! callsites too (notably its once-per-transaction-end event), whose arguments
//! are evaluated and reach this layer's cheap "no sink armed" check whether or
//! not a checkpoint is running; and enabling a target at DEBUG raises the
//! subscriber's `max_level_hint`, so `LevelFilter::current()` is DEBUG
//! process-wide and every `debug!` callsite gets past the static level check to
//! its own callsite interest rather than short-circuiting at INFO (this layer
//! interests no other target, so the added work is that lookup).
//!
//! What is verified where: the site ([`ENGINE_TARGET`], [`CAUSE_PREFIX`]) is
//! checked against the pinned engine's own source (turso_core 0.7.2,
//! `vdbe/execute.rs`), the capture and its attachment are exercised through the
//! production seam — including under the real layer stack
//! ([`crate::logs::log_layers`]), which is what installs this layer in the
//! daemon — and the product's own failure path is driven from the
//! checkpoint round's attempt seam. An end-to-end engine failure stays
//! unobserved — that limit is stated with the record format it affects, in
//! [`crate::db::failure_record`].

use std::future::Future;
use std::sync::{Arc, Mutex};

use crate::util::UnwrapPoison;

/// The engine's event site (`turso_core/vdbe/execute.rs`).
const ENGINE_TARGET: &str = "turso_core::vdbe::execute";
/// The message prefix of the engine's checkpoint-failure event.
const CAUSE_PREFIX: &str = "PRAGMA wal_checkpoint failed";
/// The engine's reason for a checkpoint it could not run because another
/// operation held the store back: `LimboError::Busy`, as the captured sentence
/// renders it (the engine formats its error with `{:?}`, and `Busy` is a unit
/// variant). A blocked checkpoint shares the engine's failure row and sentence
/// with every genuine pager error, so this reason is the only thing that tells
/// them apart.
pub(crate) const BLOCKED_REASON: &str = "PRAGMA wal_checkpoint failed: Busy";

tokio::task_local! {
    /// The sink armed for the polls of the engine call currently running, if
    /// any. Set only by [`CauseSink::scoped`].
    static ARMED: Arc<Mutex<Option<String>>>;
}

/// True when the engine's own reason in `e` says the checkpoint was blocked by
/// another operation rather than genuinely failing. Matched up to a word
/// boundary, never as a bare `Busy` substring, so the engine's `BusySnapshot`
/// is not read as a blocked checkpoint. This capture is the only thing that tells a
/// blocked attempt from a genuine failure, so a reason that never arrived keeps its
/// round failing: it opens or extends the failure window and counts towards the stop.
/// A blocked attempt never does: it opens no window and is no failure of its own.
pub(crate) fn is_blocked_checkpoint(e: &anyhow::Error) -> bool {
    format!("{e:#}")
        .split_once(BLOCKED_REASON)
        .is_some_and(|(_, rest)| !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_'))
}

/// The capture layer's filter: this one engine target at DEBUG, nothing else.
pub(crate) fn filter() -> tracing_subscriber::filter::Targets {
    tracing_subscriber::filter::Targets::new()
        .with_target(ENGINE_TARGET, tracing::level_filters::LevelFilter::DEBUG)
}

/// A per-call sink for the engine's own reason for a failed checkpoint: only
/// events emitted inside [`Self::scoped`]'s future reach this sink.
#[derive(Clone)]
pub(crate) struct CauseSink(Arc<Mutex<Option<String>>>);

impl CauseSink {
    /// An empty sink; nothing is captured until it arms a future through
    /// [`Self::scoped`].
    pub(crate) fn new() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }

    /// Arm this sink for `fut`'s polls: an engine event fired inside them lands
    /// in this sink and nowhere else. The future owns its clone of the sink, so
    /// it carries no borrow of `self` (precise capturing keeps `'_` out of the
    /// opaque type; callers spawn the result).
    pub(crate) fn scoped<T, F>(&self, fut: F) -> impl Future<Output = T> + use<T, F>
    where
        F: Future<Output = T>,
    {
        ARMED.scope(Arc::clone(&self.0), fut)
    }

    /// Attach the captured engine reason to `e`, when the engine reported one:
    /// its own chain link rendered as `engine cause: <engine text>` so the
    /// record line stays greppable.
    pub(crate) fn attach(&self, e: anyhow::Error) -> anyhow::Error {
        match self.0.lock().unwrap_poison().take() {
            Some(cause) => e.context(format!("engine cause: {cause}")),
            None => e,
        }
    }
}

/// Keeps the engine's own reason for a failed checkpoint — the one event this
/// process enables on the engine's behalf.
pub(crate) struct CauseCaptureLayer;

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CauseCaptureLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        // No sink armed for this task (or one already holds a reason) → the
        // event can be dropped before any work: the engine's unrelated debug
        // events on this target cost one lock check, no `String` allocation
        // and no `Debug` formatting.
        let capturing = ARMED
            .try_with(|sink| sink.lock().unwrap_poison().is_none())
            .unwrap_or(false);
        if !capturing {
            return;
        }
        let mut visitor = MessageVisitor(None);
        event.record(&mut visitor);
        let Some(message) = visitor.0 else {
            return;
        };
        if message.starts_with(CAUSE_PREFIX) {
            // Still armed and still empty: `on_event` runs synchronously on the
            // task that armed the sink, so nothing can change in between.
            let _ = ARMED.try_with(|sink| *sink.lock().unwrap_poison() = Some(message));
        }
    }
}

/// Keeps the event's `message` field, and only that field.
struct MessageVisitor(Option<String>);

impl tracing::field::Visit for MessageVisitor {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.0 = Some(value.to_string());
        }
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            // A `format_args!` message renders its interpolated arguments
            // through `Debug`, so this keeps the engine's full sentence.
            self.0 = Some(format!("{value:?}"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::registry::Registry;

    /// A subscriber holding only the bare capture layer (no filter).
    fn capturing_subscriber() -> impl tracing::Subscriber + Send + Sync {
        Registry::default().with(CauseCaptureLayer)
    }

    /// A subscriber with the production filter applied (see [`filter`]).
    fn filtered_subscriber() -> impl tracing::Subscriber + Send + Sync {
        Registry::default().with(CauseCaptureLayer.with_filter(filter()))
    }

    /// The engine's blocked checkpoint and its genuine pager errors arrive as the
    /// same failure row, so the captured reason is the only thing that tells them
    /// apart: the engine's own busy sentence — captured for real here — is the
    /// blocked attempt, and every other reason, including the engine's similarly
    /// spelled `BusySnapshot`, is a genuine failure. The sentence is written out
    /// instead of read from [`BLOCKED_REASON`], so the classifier's constant meets
    /// an independently spelled copy; the engine's own rendering is only checkable
    /// against the pinned source (see the module header).
    #[tokio::test]
    async fn only_the_engines_busy_sentence_reads_as_blocked() {
        let _guard = tracing::subscriber::set_default(filtered_subscriber());
        let sink = CauseSink::new();
        sink.scoped(async {
            tracing::debug!(target: ENGINE_TARGET, "PRAGMA wal_checkpoint failed: Busy");
        })
        .await;
        let blocked = sink.attach(anyhow::anyhow!(
            "Unexpected result from PRAGMA wal_checkpoint"
        ));
        assert!(
            blocked
                .to_string()
                .contains("engine cause: PRAGMA wal_checkpoint failed: Busy"),
            "the captured sentence must reach the error: {blocked:#}"
        );
        assert!(
            is_blocked_checkpoint(&blocked),
            "the engine's busy reason must read as a blocked checkpoint: {blocked:#}"
        );

        for genuine in [
            "PRAGMA wal_checkpoint failed: BusySnapshot",
            "PRAGMA wal_checkpoint failed: Corrupt(\"page 3\")",
            "no engine reason was captured",
        ] {
            let e = anyhow::anyhow!("Unexpected result from PRAGMA wal_checkpoint")
                .context(format!("engine cause: {genuine}"));
            assert!(
                !is_blocked_checkpoint(&e),
                "{genuine:?} must not read as a blocked checkpoint: {e:#}"
            );
        }
    }

    /// A sink is armed only for its own calls: each concurrent scope keeps its
    /// own reason, and an engine event outside any scope reaches no sink.
    #[tokio::test]
    async fn a_sink_keeps_only_its_own_calls_reason() {
        let _guard = tracing::subscriber::set_default(capturing_subscriber());
        let first = CauseSink::new();
        let second = CauseSink::new();
        futures_util::join!(
            first.scoped(async {
                tracing::debug!(target: ENGINE_TARGET, "PRAGMA wal_checkpoint failed: first");
            }),
            second.scoped(async {
                tracing::debug!(target: ENGINE_TARGET, "PRAGMA wal_checkpoint failed: second");
            }),
        );

        let first_text = format!("{:#}", first.attach(anyhow::anyhow!("first call failed")));
        assert!(
            first_text.contains("PRAGMA wal_checkpoint failed: first"),
            "the first sink must keep its own call's reason: {first_text:?}"
        );
        assert!(
            !first_text.contains("second"),
            "the first sink must not see the other call's reason: {first_text:?}"
        );

        tracing::debug!(target: ENGINE_TARGET, "PRAGMA wal_checkpoint failed: unarmed");
        let second_text = format!("{:#}", second.attach(anyhow::anyhow!("second call failed")));
        assert!(
            second_text.contains("PRAGMA wal_checkpoint failed: second"),
            "the second sink must keep its own call's reason: {second_text:?}"
        );
        assert!(
            !second_text.contains("unarmed"),
            "an engine event outside any scope must reach no sink: {second_text:?}"
        );
    }

    /// Under the production filter only the checkpoint-failure sentence on the
    /// engine's own target is kept, whole — prefix and interpolated diagnostics —
    /// and it reaches the error chain through the production seam: the same
    /// sentence on another target never reaches the layer, and a different
    /// message on the engine target is dropped by the message check.
    #[tokio::test]
    async fn only_the_engines_checkpoint_sentence_is_kept() {
        let _guard = tracing::subscriber::set_default(filtered_subscriber());
        let sink = CauseSink::new();
        sink.scoped(async {
            tracing::debug!(target: ENGINE_TARGET, "a different message");
            tracing::debug!(
                target: "turso_core::storage::pager",
                "PRAGMA wal_checkpoint failed: another target"
            );
            tracing::debug!(target: ENGINE_TARGET, "PRAGMA wal_checkpoint failed: literal");
        })
        .await;

        let rendered = format!("{:#}", sink.attach(anyhow::anyhow!("no reason captured")));
        assert!(
            rendered.contains("engine cause: PRAGMA wal_checkpoint failed: literal"),
            "the engine's own checkpoint sentence at DEBUG must be captured: {rendered:?}"
        );
        assert!(
            !rendered.contains("a different message"),
            "a different message on the engine target must be dropped: {rendered:?}"
        );
        assert!(
            !rendered.contains("another target"),
            "the engine's sentence on another target must be filtered out: {rendered:?}"
        );

        // The engine's real event interpolates its diagnostics, so a formatted
        // message must be kept whole as well as a literal one — and the product's
        // own text stays a separate chain link.
        let interpolated = CauseSink::new();
        interpolated
            .scoped(async {
                tracing::debug!(
                    target: ENGINE_TARGET,
                    "PRAGMA wal_checkpoint failed: {:?}",
                    "boom"
                );
            })
            .await;
        let rendered = format!(
            "{:#}",
            interpolated.attach(anyhow::anyhow!(
                "Unexpected result from PRAGMA wal_checkpoint"
            ))
        );
        assert!(
            rendered.contains("engine cause: PRAGMA wal_checkpoint failed:"),
            "the captured cause must keep the engine's sentence: {rendered:?}"
        );
        assert!(
            rendered.contains("boom"),
            "the captured cause must keep the interpolated diagnostics: {rendered:?}"
        );
        assert!(
            rendered.contains("Unexpected result from PRAGMA wal_checkpoint"),
            "the product's own text must survive: {rendered:?}"
        );
    }

    /// The production layer stack — the real one, built by
    /// [`crate::logs::log_layers`], not a copy of it: the `EnvFilter` must stay a
    /// LAYER filter on the JSON log layer, because installed globally it would
    /// drop the engine's DEBUG event and the record would silently fall back to
    /// the product's constant text with every other test still green.
    #[tokio::test]
    async fn the_production_filter_stack_still_keeps_the_engine_reason() {
        use tracing_subscriber::EnvFilter;

        let _guard = tracing::subscriber::set_default(crate::logs::log_layers(
            std::io::sink,
            EnvFilter::new(crate::logs::DEFAULT_LOG_FILTER),
        ));
        let sink = CauseSink::new();
        sink.scoped(async {
            tracing::debug!(
                target: ENGINE_TARGET,
                "PRAGMA wal_checkpoint failed: the production stack"
            );
        })
        .await;

        let rendered = format!("{:#}", sink.attach(anyhow::anyhow!("no reason captured")));
        assert!(
            rendered.contains("the production stack"),
            "the capture layer must keep the engine's reason under the production filter stack: \
             {rendered:?}"
        );
    }
}
