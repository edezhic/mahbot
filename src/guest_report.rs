//! Guest capability reports: a guest's Assistant tells the owner that its user needed something
//! the guest toolset does not offer — no mail, no calendar, no shell — and nothing in the
//! product lets a guest create a custom tool instead.
//!
//! A report is one-way. It reaches the owner's Assistant as an incoming notice through the
//! durable path ([`crate::agent::message_router::deliver_guest_report`]), written down before it
//! is routed so a restart cannot lose it, and nothing comes back to the guest: its own
//! conversation holds only the tool's answer. No tool is granted, implied or promised — what an
//! account may use stays the owner's decision.
//!
//! The ledger (`guest_reports` in `core.db`, reached through the account store — it is
//! per-account bookkeeping) is not an inbox: one row per report holding the account, the local
//! day and a digest of the text — never the text itself — and every write prunes the days
//! before, so it never grows past one day and nothing in it can be read back as a report.
//!
//! Two limits ride on it: a day's cap per guest, and a refusal of a report identical to one the
//! same guest already filed that day (which does not spend the day). Both are stated in the
//! tool's own description so the guest's model can comply, and a refusal names the limit it hit.

use anyhow::Result;
use sha2::{Digest as _, Sha256};
use tracing::warn;

use crate::db;

/// The shortest report that says anything: what the user needed and what was missing.
pub(crate) const MIN_REPORT_CHARS: usize = 15;

/// The longest that stays a report rather than a document.
pub(crate) const MAX_REPORT_CHARS: usize = 1200;

/// Reports one guest may file per local day.
pub(crate) const DAILY_REPORT_CAP: i64 = 5;

/// What the guest is told about its report: the only outcome that earns "passed on" is a
/// durable hand-over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Filed {
    /// Written down and handed to the owner's Assistant.
    Passed,
    /// This account already filed the same report today; nothing was sent.
    Duplicate,
    /// The account has filed its day's reports; nothing was sent.
    OverCap,
    /// The report could not be written down, so it was not handed over.
    NotDelivered,
}

/// File one report from `sender` (the turn's own account, never a tool argument). Every
/// failure to write it down — the ledger or the hand-over — is [`Filed::NotDelivered`]: the
/// guest is never told a delivery happened that did not.
pub(crate) async fn file(sender: &str, text: &str) -> Filed {
    file_with(
        sender,
        text,
        crate::agent::message_router::deliver_guest_report,
    )
    .await
}

/// [`file`] with the hand-over as a parameter, so a test can watch a failed delivery give the
/// day's slot back.
async fn file_with<F, Fut>(sender: &str, text: &str, deliver: F) -> Filed
where
    F: FnOnce(String) -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
{
    let day = chrono::Local::now().format("%Y-%m-%d").to_string();
    let digest = digest(text);
    match record(sender, &day, &digest).await {
        Ok(Recorded::Slot) => {}
        Ok(Recorded::Duplicate) => return Filed::Duplicate,
        Ok(Recorded::OverCap) => return Filed::OverCap,
        Err(e) => {
            warn!(user = %sender, error = %e, "Guest report: could not record the report");
            return Filed::NotDelivered;
        }
    }
    if let Err(e) = deliver(render(sender, text)).await {
        warn!(
            user = %sender,
            error = %e,
            "Guest report: could not be made durable — not delivered",
        );
        // Nothing reached the owner's Assistant, so the report must not spend the account's day.
        if let Err(e) = release(sender, &day, &digest).await {
            warn!(user = %sender, error = %e, "Guest report: could not release the day's slot");
        }
        return Filed::NotDelivered;
    }
    Filed::Passed
}

/// The ledger's verdict on one report.
enum Recorded {
    /// A slot was taken for it.
    Slot,
    /// The same account already filed this exact report today.
    Duplicate,
    /// The account's day is full.
    OverCap,
}

// Take the report's slot in one transaction: prune the days before, read today's tally and
// whether this exact report is already in it, then insert. The transaction is what makes the
// check and the insert one step — the writing connection is held for the whole of it, so no
// other writer can slip a report in between them.
async fn record(sender: &str, day: &str, digest: &str) -> Result<Recorded> {
    let tx = crate::users::store().conn.begin_tx().await?;
    tx.execute(
        "DELETE FROM guest_reports WHERE day <> ?1",
        db::params![day],
    )
    .await?;
    let (filed, duplicate) = tx
        .execute_one(
            "SELECT COUNT(*), COALESCE(SUM(CASE WHEN digest = ?3 THEN 1 ELSE 0 END), 0) \
             FROM guest_reports WHERE user_name = ?1 AND day = ?2",
            db::params![sender, day, digest],
            |row| Ok::<_, turso::Error>((row.get::<i64>(0)?, row.get::<i64>(1)?)),
        )
        .await?;
    let recorded = if duplicate > 0 {
        Recorded::Duplicate
    } else if filed >= DAILY_REPORT_CAP {
        Recorded::OverCap
    } else {
        tx.execute(
            "INSERT INTO guest_reports (user_name, day, digest) VALUES (?1, ?2, ?3)",
            db::params![sender, day, digest],
        )
        .await?;
        Recorded::Slot
    };
    tx.commit().await?;
    Ok(recorded)
}

// Give the day's slot back when the report never reached the owner's Assistant.
async fn release(sender: &str, day: &str, digest: &str) -> Result<()> {
    crate::users::store()
        .conn
        .execute(
            "DELETE FROM guest_reports WHERE user_name = ?1 AND day = ?2 AND digest = ?3",
            db::params![sender, day, digest],
        )
        .await?;
    Ok(())
}

// A report's identity for the duplicate check: surrounding space dropped and every run of
// whitespace collapsed, so a re-sent copy is caught and a genuinely different report is not.
fn digest(text: &str) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    crate::util::hex_string(&Sha256::digest(normalized.as_bytes()))
}

// The notice the owner's Assistant receives. The report is a guest's own words, so it is
// escaped into the envelope: whatever it contains, it cannot close, reopen or imitate the
// framing around it.
fn render(sender: &str, text: &str) -> String {
    crate::prompt::substitute(
        &crate::prompt::load_prompt("guest_report.md"),
        &[
            ("{{sender}}", &crate::tools::envelope_attr(sender)),
            ("{{report}}", &escape_markup(text)),
        ],
    )
}

/// `text` with the characters that carry meaning to this product — `&`, `<`, `>` and `[` —
/// neutralized, so a report can never forge the envelope that carries it nor smuggle a media
/// marker (which the agent loop would turn into an attachment) into the owner's turn. Quotes
/// stay: they matter only inside an attribute, and in prose they are just prose.
fn escape_markup(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('[', "&#91;")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A report that never reached the owner's Assistant is not "passed on", and it does not
    /// spend the account's day: the slot goes back, so the same report can be filed again.
    #[tokio::test]
    async fn an_undelivered_report_spends_no_day() {
        crate::util::test::init_test_stores().await;
        let sender = "guest_report_undelivered";
        let text = "I need a calendar tool to schedule meetings with my team.";
        let fail = |_: String| async { anyhow::bail!("the router is unavailable") };

        assert_eq!(file_with(sender, text, fail).await, Filed::NotDelivered);
        // Nothing was left behind: the same report is neither a duplicate nor refused by a cap
        // it never reached.
        assert_eq!(file_with(sender, text, fail).await, Filed::NotDelivered);

        let filed = crate::users::store()
            .conn
            .execute_one(
                "SELECT COUNT(*) FROM guest_reports WHERE user_name = ?1",
                db::params![sender],
                |row| row.get::<i64>(0),
            )
            .await
            .expect("read the ledger");
        assert_eq!(filed, 0, "the day's slot was given back");
    }
}
