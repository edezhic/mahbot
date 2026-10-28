//! Clock-driven animation widgets: the indeterminate spinner and the recording
//! dot's pulse.
//!
//! Animation phase is read from a monotonic clock at draw time, so the widgets
//! keep no per-frame state; each frame self-arms a scheduled
//! [`window::RedrawRequest::At`] redraw at the widget's own step, mirroring the
//! editor's cursor blink — no subscription, no messages. A widget is only ever
//! placed in the tree while its work item is running, so its presence in the
//! tree is the running signal: when it leaves the tree it stops re-arming
//! redraws and costs nothing.

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

// ── Constants ───────────────────────────────────────────────────────

/// Number of dots drawn around the orbit. The head dot (`i == 0`) leads the
/// ring, the remaining [`DOTS`] - 1 trail behind it with decreasing opacity.
const DOTS: usize = 8;

/// Full rotation period in milliseconds — the spinner completes one turn per
/// second, matching a conventional indeterminate spinner.
const PERIOD_MS: u64 = 1000;

/// Upper bound of redraws per second. The spinner self-arms a scheduled redraw
/// at least [`PERIOD_MS`] / [`SPINNER_FPS`] apart so a busy page never pays for
/// a per-frame draw; idle pages don't host the widget at all.
const SPINNER_FPS: u64 = 24;

/// Redraw step of the spinner: [`PERIOD_MS`] / [`SPINNER_FPS`].
const STEP_MS: u64 = PERIOD_MS / SPINNER_FPS;

/// Boot-anchored monotonic clock. [`phase_ms`] is the elapsed time since this
/// instant; using a monotonic clock (rather than wall-clock epoch millis) means
/// NTP or clock adjustments can't jitter the animation.
static BOOT: LazyLock<Instant> = LazyLock::new(Instant::now);

/// An indeterminate spinner rendered as a ring of dots.
///
/// The trailing dots fade toward the head, giving a directional rotation.
/// Phase is taken from a monotonic clock at draw time, so the widget is
/// stateless between frames. The widget is only ever placed in the tree while
/// the target work item is running, so its presence in the tree is the running
/// signal: when it leaves the tree it stops re-arming redraws.
pub(crate) struct Spinner {
    /// The bounding diameter of the spinner glyph.
    diameter: f32,
}

/// Create a spinning (constantly animating) [`Spinner`] of the given diameter
/// in the theme accent colour.
#[must_use]
pub(crate) fn spinner(diameter: f32) -> Spinner {
    Spinner { diameter }
}

/// Monotonic milliseconds since boot (truncated to `u64`). Used both to derive
/// the rotation phase at draw time and to schedule the next redraw, so the two
/// stay aligned.
#[must_use]
fn phase_ms() -> u64 {
    u64::try_from(BOOT.elapsed().as_millis()).unwrap_or(0)
}

/// The opacity of dot `i` in the ring: the head dot (`i == 0`) is fully
/// visible and each successive trailing dot fades linearly down to 0.15 at the
/// last trailing dot.
#[must_use]
#[expect(clippy::cast_precision_loss)]
fn dot_opacity(i: usize) -> f32 {
    1.0 - i as f32 / (DOTS as f32 - 1.0) * 0.85
}

/// Milliseconds remaining until the next `step_ms` boundary. When exactly on a
/// boundary returns `step_ms`, so the scheduled delay is never ~0 (callers add
/// +1 ms).
#[must_use]
fn next_step_in(elapsed_ms: u64, step_ms: u64) -> u64 {
    let rem = elapsed_ms % step_ms;
    if rem == 0 { step_ms } else { step_ms - rem }
}

/// Self-arm the next redraw at the next `step_ms` boundary: this is what keeps a
/// clock-driven widget animating with no subscription and no messages.
///
/// A hidden window draws nothing, so nothing is armed for one: the animation resumes with
/// the next frame the window does draw, when it comes back.
fn arm_redraw<Message>(event: &Event, shell: &mut Shell<'_, Message>, step_ms: u64) {
    if super::app_window::hidden() {
        return;
    }
    if let Event::Window(window::Event::RedrawRequested(_)) = event {
        shell.request_redraw_at(window::RedrawRequest::At(
            Instant::now() + Duration::from_millis(next_step_in(phase_ms(), step_ms) + 1),
        ));
    }
}

impl<Message> Widget<Message, iced::Theme, iced::Renderer> for Spinner {
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

    #[expect(clippy::cast_precision_loss)]
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
        let center = layout.bounds().center();
        let dot_radius = self.diameter * 0.15;
        let orbit_radius = self.diameter / 2.0 - dot_radius;
        let ms = phase_ms() % PERIOD_MS;
        let head_angle = ms as f32 / PERIOD_MS as f32 * std::f32::consts::TAU;

        for i in 0..DOTS {
            let angle = head_angle + i as f32 / DOTS as f32 * std::f32::consts::TAU;
            let x = center.x + angle.cos() * orbit_radius;
            let y = center.y + angle.sin() * orbit_radius;
            let color = theme::ACCENT.scale_alpha(dot_opacity(i));
            renderer.fill_quad(
                renderer::Quad {
                    bounds: Rectangle {
                        x: x - dot_radius,
                        y: y - dot_radius,
                        width: dot_radius * 2.0,
                        height: dot_radius * 2.0,
                    },
                    border: Border {
                        radius: Radius::from(dot_radius),
                        width: 0.0,
                        color: Color::TRANSPARENT,
                    },
                    shadow: Shadow::default(),
                    snap: false,
                },
                color,
            );
        }
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
        arm_redraw(event, shell, STEP_MS);
    }
}

impl<'a, Message> From<Spinner> for Element<'a, Message, iced::Theme, iced::Renderer>
where
    Message: 'a,
{
    fn from(spinner: Spinner) -> Self {
        Self::new(spinner)
    }
}

// ── Recording dot ───────────────────────────────────────────────────

/// Full fade cycle of the recording dot: out over half of it, back in over the
/// other half.
#[cfg(target_os = "macos")]
const DOT_PERIOD_MS: u64 = 1000;

/// Redraw step of the recording dot's fade — smooth for an opacity ramp.
#[cfg(target_os = "macos")]
const DOT_STEP_MS: u64 = 50;

/// The faded end of the recording dot's range: never fully off, so the
/// indicator reads as a pulse rather than a blink.
#[cfg(target_os = "macos")]
const DOT_DIM: f32 = 0.15;

/// A recording indicator: a dot whose opacity fades out and back in at
/// [`DOT_PERIOD_MS`], in the colour that marks a running recording. Passive — it
/// takes no press and shows no tooltip.
#[cfg(target_os = "macos")]
pub(crate) struct PulseDot {
    diameter: f32,
}

/// Create a pulsing dot of the given diameter.
#[cfg(target_os = "macos")]
#[must_use]
pub(crate) fn pulse_dot(diameter: f32) -> PulseDot {
    PulseDot { diameter }
}

/// Opacity of the dot `elapsed_ms` into its fade: fully on at the cycle's
/// start, dimmest at its midpoint, fully on again at its end.
#[cfg(target_os = "macos")]
#[must_use]
#[expect(clippy::cast_precision_loss)]
fn pulse_alpha(elapsed_ms: u64) -> f32 {
    let phase = (elapsed_ms % DOT_PERIOD_MS) as f32 / DOT_PERIOD_MS as f32 * std::f32::consts::TAU;
    DOT_DIM + (1.0 - DOT_DIM) * (phase.cos() * 0.5 + 0.5)
}

#[cfg(target_os = "macos")]
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
            theme::STATUS_ERROR.scale_alpha(pulse_alpha(phase_ms())),
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
        arm_redraw(event, shell, DOT_STEP_MS);
    }
}

#[cfg(target_os = "macos")]
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
    use super::{DOTS, STEP_MS, dot_opacity, next_step_in};

    #[test]
    fn next_step_in_boundaries() {
        assert_eq!(next_step_in(0, STEP_MS), STEP_MS);
        assert_eq!(next_step_in(STEP_MS - 1, STEP_MS), 1);
        // Exactly on a boundary still asks for a full step, never ~0.
        assert_eq!(next_step_in(STEP_MS, STEP_MS), STEP_MS);
        // Just past a boundary has one millisecond less to go, and so on.
        assert_eq!(next_step_in(1, STEP_MS), STEP_MS - 1);
        assert_eq!(next_step_in(STEP_MS + 1, STEP_MS), STEP_MS - 1);
        assert_eq!(next_step_in(2 * STEP_MS, STEP_MS), STEP_MS);
    }

    #[test]
    fn dot_opacity_fades_to_head_and_foot() {
        assert!((dot_opacity(0) - 1.0).abs() < 1e-6);
        // Strictly decreasing across the ring.
        for i in 0..DOTS - 1 {
            assert!(dot_opacity(i) > dot_opacity(i + 1));
        }
        // Last trailing dot lands at the target floor.
        assert!((dot_opacity(DOTS - 1) - 0.15).abs() < 1e-6);
    }
}
