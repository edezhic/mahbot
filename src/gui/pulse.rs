//! The pulse dot: a filled dot whose opacity pulses, marking work in progress —
//! an agent session that is working, a tool call whose result is still to come,
//! a busy Telegram switch, a voice recording.
//!
//! Phase is read from a monotonic clock at draw time, so the widget keeps no
//! per-frame state: each frame self-arms a [`window::RedrawRequest::At`] redraw
//! at its own step, mirroring the editor's cursor blink. Presence in the tree is
//! the running signal — out of the tree, it stops re-arming redraws.

use std::sync::LazyLock;
use std::time::{Duration, Instant};

use iced::advanced::layout::{self, Layout};
use iced::advanced::mouse;
use iced::advanced::renderer;
use iced::advanced::widget::{Tree, Widget};
use iced::advanced::{Clipboard, Renderer, Shell};
use iced::border::Radius;
use iced::window;
use iced::{Border, Color, Element, Event, Length, Rectangle, Shadow, Size};

use super::theme;

/// Diameter of the activity dot in a text row. The ring it replaces was a
/// hairline, and a solid fill of that full text-height box reads markedly
/// heavier, so the dot is smaller than the text beside it.
const ACTIVITY_DOT: f32 = 8.0;

/// Full fade cycle: out over half of it, back in over the other half.
const PERIOD_MS: u64 = 1000;

/// Redraw step of the fade — smooth for an opacity ramp.
const STEP_MS: u64 = 50;

/// The faded end of the range: never fully off, so it reads as a pulse, not a blink.
const DIM: f32 = 0.15;

/// Boot-anchored monotonic clock, so NTP or clock adjustments can't jitter the animation.
static BOOT: LazyLock<Instant> = LazyLock::new(Instant::now);

/// A filled dot whose opacity fades out and back in at [`PERIOD_MS`]. Passive —
/// it takes no press and shows no tooltip.
pub(crate) struct PulseDot {
    diameter: f32,
    color: Color,
}

#[must_use]
pub(crate) fn pulse_dot(diameter: f32, color: Color) -> PulseDot {
    PulseDot { diameter, color }
}

/// The activity indicator: the pulse dot in the theme accent, sized to sit in a text row.
#[must_use]
pub(crate) fn activity_dot() -> PulseDot {
    pulse_dot(ACTIVITY_DOT, theme::ACCENT)
}

/// Monotonic milliseconds since boot. Drives both the fade phase and the next
/// redraw, so the two stay aligned.
#[must_use]
fn phase_ms() -> u64 {
    u64::try_from(BOOT.elapsed().as_millis()).unwrap_or(0)
}

/// Fully on at the cycle's start, dimmest at its midpoint, fully on at its end.
#[must_use]
#[expect(clippy::cast_precision_loss)]
fn pulse_alpha(elapsed_ms: u64) -> f32 {
    let phase = (elapsed_ms % PERIOD_MS) as f32 / PERIOD_MS as f32 * std::f32::consts::TAU;
    DIM + (1.0 - DIM) * (phase.cos() * 0.5 + 0.5)
}

/// Milliseconds until the next [`STEP_MS`] boundary; exactly on a boundary it
/// returns a full step, so the scheduled delay is never ~0.
#[must_use]
fn next_step_in(elapsed_ms: u64) -> u64 {
    let rem = elapsed_ms % STEP_MS;
    if rem == 0 { STEP_MS } else { STEP_MS - rem }
}

/// Self-arm the next redraw at the next [`STEP_MS`] boundary — this is what keeps
/// a clock-driven widget animating with no subscription and no messages. A hidden
/// window draws nothing, so nothing is armed for one.
fn arm_redraw<Message>(event: &Event, shell: &mut Shell<'_, Message>) {
    if super::app_window::hidden() {
        return;
    }
    if let Event::Window(window::Event::RedrawRequested(_)) = event {
        shell.request_redraw_at(window::RedrawRequest::At(
            Instant::now() + Duration::from_millis(next_step_in(phase_ms()) + 1),
        ));
    }
}

impl<Message> Widget<Message, iced::Theme, iced::Renderer> for PulseDot {
    fn size(&self) -> Size<Length> {
        Size::new(Length::Fixed(self.diameter), Length::Fixed(self.diameter))
    }

    fn layout(
        &mut self,
        _tree: &mut Tree,
        _renderer: &iced::Renderer,
        _limits: &layout::Limits,
    ) -> layout::Node {
        layout::Node::new(Size::new(self.diameter, self.diameter))
    }

    fn draw(
        &self,
        _tree: &Tree,
        renderer: &mut iced::Renderer,
        _theme: &iced::Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        renderer.fill_quad(
            renderer::Quad {
                bounds: layout.bounds(),
                border: Border {
                    radius: Radius::from(self.diameter / 2.0),
                    width: 0.0,
                    color: Color::TRANSPARENT,
                },
                shadow: Shadow::default(),
                snap: false,
            },
            self.color.scale_alpha(pulse_alpha(phase_ms())),
        );
    }

    fn update(
        &mut self,
        _tree: &mut Tree,
        event: &Event,
        _layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _renderer: &iced::Renderer,
        _clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        _viewport: &Rectangle,
    ) {
        arm_redraw(event, shell);
    }
}

impl<'a, Message> From<PulseDot> for Element<'a, Message, iced::Theme, iced::Renderer>
where
    Message: 'a,
{
    fn from(dot: PulseDot) -> Self {
        Self::new(dot)
    }
}

#[cfg(test)]
mod tests {
    use super::{STEP_MS, next_step_in};

    #[test]
    fn next_step_in_boundaries() {
        assert_eq!(next_step_in(0), STEP_MS);
        assert_eq!(next_step_in(STEP_MS - 1), 1);
        assert_eq!(next_step_in(STEP_MS), STEP_MS);
        assert_eq!(next_step_in(1), STEP_MS - 1);
        assert_eq!(next_step_in(STEP_MS + 1), STEP_MS - 1);
        assert_eq!(next_step_in(2 * STEP_MS), STEP_MS);
    }
}
