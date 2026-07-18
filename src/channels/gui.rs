//! GUI channel — bridges the Iced dashboard into the MahBot channel system.
//!
//! Unlike Telegram which has its own async listener loop, the GUI channel uses
//! an internal mpsc pair: the Iced UI pushes `ChannelMessage` values into
//! `GUI_MESSAGE_TX`, and `listen()` reads them from the paired receiver,
//! forwarding each one into the shared pipeline `tx`.
//!
//! Outgoing agent responses are broadcast to the GUI dashboard and persisted
//! to chat_history centrally via [`super::broadcast_and_persist_agent_response`].
//! `GuiChannel::send()` is pure transport (no-op) — all broadcast+persist
//! happens in the canonical function so every path gets consistent treatment.

use crate::Channel;
use crate::util::UnwrapPoison;
use crate::{ChannelMessage, SendMessage};
use async_trait::async_trait;
use tokio::sync::mpsc;

/// The GUI channel — always registered, even in dashboard-only mode.
pub struct GuiChannel {
    /// The internal receiver, stored so `listen()` can consume it.
    gui_rx: std::sync::Mutex<Option<mpsc::UnboundedReceiver<ChannelMessage>>>,
}

impl GuiChannel {
    /// Create a new GuiChannel with an internal mpsc pair.
    ///
    /// Returns `(Self, gui_tx)`. The caller must:
    /// 1. Store `gui_tx` in `GUI_MESSAGE_TX` globally
    /// 2. Register this channel in the channel registry
    /// 3. Call `listen(tx)` to start consuming from the internal receiver
    #[must_use]
    pub fn new() -> (Self, mpsc::UnboundedSender<ChannelMessage>) {
        let (gui_tx, gui_rx) = mpsc::unbounded_channel::<ChannelMessage>();
        let channel = Self {
            gui_rx: std::sync::Mutex::new(Some(gui_rx)),
        };
        (channel, gui_tx)
    }
}

#[async_trait]
impl Channel for GuiChannel {
    /// Pure transport — broadcast and persistence are handled centrally
    /// by [`super::broadcast_and_persist_agent_response`].
    async fn send(&self, _message: &SendMessage) -> anyhow::Result<()> {
        Ok(())
    }

    async fn listen(&self, tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()> {
        // Take the receiver from the internal mutex. After this, we own it
        // and can drop the &self reference.
        let mut gui_rx = self
            .gui_rx
            .lock()
            .unwrap_poison()
            .take()
            .expect("GuiChannel::listen() called twice");
        // Mutex guard is dropped here.

        // Forward each GUI-originated message into the shared pipeline.
        // Broadcast+persist is handled centrally in process_channel_message().
        while let Some(msg) = gui_rx.recv().await {
            if tx.send(msg).await.is_err() {
                tracing::info!("GuiChannel: pipeline closed — shutting down listener");
                break;
            }
        }

        tracing::info!("GuiChannel: listener stopped");
        Ok(())
    }

    fn name(&self) -> &'static str {
        "gui"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn start_typing(&self, recipient: &str) -> anyhow::Result<()> {
        use crate::ChatEvent;

        if let Some(tx) = crate::CHAT_BROADCAST.get() {
            // KNOWN LIMITATION: This path lacks workspace context because the
            // Channel trait's `start_typing` does not carry workspace info.
            // The broadcast_typing path (per-agent consumer loop) covers ~95% of typing
            // events with correct workspace scoping. An empty workspace string
            // means this typing indicator will be filtered out by the GUI handler's
            // workspace check — safe but invisible. Deferred: add workspace to
            // Channel::start_typing signature.
            let _ = tx.send(ChatEvent::Typing {
                user_name: recipient.to_string(),
                is_typing: true,
                workspace: String::new(),
            });
        }
        Ok(())
    }

    /// GUI users are addressed by sender name.
    fn resolve_recipient(&self, user_name: &str, _reply_target: &str) -> Option<String> {
        Some(user_name.to_string())
    }
}
