//! The `say` tool: one short phrase spoken aloud by the admin's Assistant.
//!
//! The product speaks nothing by itself — a reply, an automation or a confirmation
//! reaches the engine only when the Assistant asks for it here. The phrase is
//! handed to [`crate::audio::tts::say_phrase`], which renders and plays it in the
//! background: the call returns at once, and a newer phrase replaces one still
//! being spoken.
//!
//! A phrase is one short sentence, not a reading: anything over
//! [`MAX_PHRASE_CHARS`] is refused rather than spoken. Every refusal — too long,
//! speech output switched off, engine not up, nobody at the computer — is this
//! tool's ordinary answer, not a failure: the Assistant did nothing wrong, and
//! each one says what to do instead. The engine's own bring-up (download, device
//! open) is kicked on the side and never waited for.
//!
//! The refusal for an absent user is the tool's one use of the machine-wide
//! user-activity state ([`crate::user_activity`]), and it comes last in the
//! engine's gates so the answers for speech that cannot be produced at all keep
//! precedence.
//!
//! macOS only, where the speech engine lives — the module is `cfg`-gated in
//! `tools/mod.rs`, and the admin's Assistant is its only holder.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use crate::Workspace;
use crate::audio::tts::{self, NotReady, SayOutcome};
use crate::tools::Tool;

/// The longest phrase spoken: a spoken note is one short sentence, counted as
/// the characters a person sees (after trimming), never bytes.
const MAX_PHRASE_CHARS: usize = 100;

pub(crate) struct SayTool;

#[async_trait]
impl Tool for SayTool {
    fn name(&self) -> &'static str {
        "say"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "text": {
                    "type": "string",
                    "description": format!(
                        "The phrase to speak, in the language the user speaks — at most \
                         {MAX_PHRASE_CHARS} characters."
                    ),
                }
            }),
            &["text"],
        )
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let Some(text) = args.get("text").and_then(|text| text.as_str()) else {
            return Ok(
                "Nothing was spoken: `text` is required and names the phrase to say.".into(),
            );
        };
        // Counted as the model wrote it, minus the whitespace around it, and as
        // the characters a person sees rather than bytes.
        let text = text.trim();
        if text.is_empty() {
            return Ok("Nothing was spoken: the phrase is empty.".into());
        }
        let chars = text.chars().count();
        if chars > MAX_PHRASE_CHARS {
            return Ok(format!(
                "Nothing was spoken: the phrase is {chars} characters long. Say it in one short \
                 phrase of at most {MAX_PHRASE_CHARS} characters — long text is not read aloud."
            ));
        }
        Ok(match tts::say_phrase(text) {
            SayOutcome::Speaking => "Spoken: the phrase plays now.".into(),
            SayOutcome::SpeechOff => {
                "Nothing was spoken: the user has speech output turned off right now.".into()
            }
            SayOutcome::Absent => {
                "Nothing was spoken: the user is not at the computer right now.".into()
            }
            SayOutcome::NotReady(reason) => {
                format!("Nothing was spoken: {}.", not_ready_reason(&reason))
            }
        })
    }
}

/// The reason a not-ready engine gives, worded as the model reads it: what is
/// missing, and the bring-up the call left running in the background.
fn not_ready_reason(reason: &NotReady) -> &'static str {
    match reason {
        NotReady::ModelsMissing => {
            "the speech models are not loaded yet, and they are being prepared in the background"
        }
        NotReady::DownloadFailed => {
            "the speech models could not be downloaded — a retry is running in the background"
        }
        NotReady::NoAudioDevice => "this machine has no audio output device",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run `args` through the tool with speech output switched OFF for the
    /// duration, restoring the config afterwards: every answer below then comes
    /// from the length gate or from the engine's own refusal, never from a
    /// device.
    async fn say(args: serde_json::Value) -> String {
        let snapshot = crate::config::CONFIG.snapshot();
        assert!(
            crate::config::CONFIG.set_string_field("tts_enabled", ""),
            "tts_enabled is a known config key"
        );
        let answer = SayTool
            .execute(&crate::workspace::test_ws("test"), args)
            .await
            .unwrap();
        crate::config::CONFIG.swap(snapshot);
        answer
    }

    /// The description is where the model learns the limit and both refusals, so
    /// it must state them — and the rule that sets the spoken language.
    #[test]
    fn the_description_states_the_limit_and_the_refusals() {
        let description = SayTool.description();
        assert!(
            description.contains(&MAX_PHRASE_CHARS.to_string()),
            "the description must state the {MAX_PHRASE_CHARS}-character limit: {description}"
        );
        for expected in [
            "not read aloud",
            "speech output turned off",
            "language the user speaks",
        ] {
            assert!(
                description.contains(expected),
                "the description must say '{expected}': {description}"
            );
        }
    }

    #[tokio::test]
    #[serial_test::serial(tts)]
    async fn a_phrase_over_the_limit_is_refused_by_visible_characters() {
        let too_long = say(json!({ "text": "a".repeat(101) })).await;
        assert!(
            too_long.contains("101") && too_long.contains("100"),
            "the refusal must name the limit and the length, got: {too_long}"
        );

        // 100 characters, whatever their byte length, and whatever whitespace
        // surrounds them: within the limit, so the engine sees the phrase.
        for text in ["é".repeat(100), format!("  {}  ", "é".repeat(100))] {
            let answer = say(json!({ "text": text })).await;
            assert!(
                answer.contains("speech output turned off"),
                "a phrase within the limit reaches the engine, which refuses it while speech is \
                 off — got: {answer}"
            );
        }
    }

    #[tokio::test]
    #[serial_test::serial(tts)]
    async fn a_call_without_text_says_so() {
        assert!(
            say(json!({})).await.contains("`text` is required"),
            "a missing phrase must be answered plainly"
        );
    }
}
