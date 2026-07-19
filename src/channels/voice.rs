//! Voice channel — registered in the channel registry so the message
//! routing system recognises voice as a valid message source.
//!
//! There is no outbound voice transport — agent responses are broadcast
//! to the GUI and persisted to chat_history by the standard response
//! delivery path. The `send()` method is a no-op, matching the pattern
//! used by [`GuiChannel`](super::gui::GuiChannel).
//!
//! The voice pipeline runs its own mic-capture loop independently;
//! `listen()` is a no-op because incoming voice commands flow through
//! [`crate::voice::route_to_agent`], not through a channel listener.

use crate::Channel;
use crate::{ChannelMessage, SendMessage};
use async_trait::async_trait;
use std::sync::Arc;

/// The voice channel — registered so the message routing system can
/// resolve the `"voice"` channel name when delivering agent responses.
pub struct VoiceChannel;

#[async_trait]
impl Channel for VoiceChannel {
    /// No-op — voice has no outbound transport. Agent responses are
    /// broadcast to the GUI and persisted to chat_history by
    /// [`crate::channels::broadcast_and_persist_agent_response`].
    async fn send(&self, _message: &SendMessage) -> anyhow::Result<()> {
        Ok(())
    }

    /// No-op — the voice pipeline manages its own mic-capture loop.
    async fn listen(&self, _tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "voice"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Register the voice channel in the global channel registry.
///
/// Called during bootstrap from `init_message_pipeline`. The channel
/// registry must already be initialised — callers should ensure
/// [`crate::CHANNEL_REGISTRY`] has been set before invoking this.
///
/// The `VoiceChannel` has a no-op `send()` (agent responses are
/// delivered via broadcast+persist independently of the registry) and
/// a no-op `listen()` (the voice pipeline runs its own mic-capture
/// loop). Registration resolves the `"voice"` channel name so the
/// message routing system can look it up when constructing replies.
pub fn register_global() {
    let channel: Arc<dyn Channel> = Arc::new(VoiceChannel);
    crate::channel_registry().register(channel);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CHANNEL_REGISTRY, ChannelRegistry, channel_registry};

    /// Verify that [`register_global`] correctly registers the voice
    /// channel — i.e. the same function called by the production
    /// bootstrap path in `init_message_pipeline`.
    #[test]
    fn test_voice_channel_registration() {
        // Initialise the registry (idempotent — reuses if already set).
        CHANNEL_REGISTRY.get_or_init(ChannelRegistry::default);

        // Call the same registration function used by production.
        register_global();

        let found = channel_registry().get("voice");
        assert!(
            found.is_some(),
            "VoiceChannel should be findable by 'voice' name"
        );
    }
}
