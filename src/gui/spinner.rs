//! A lightweight indeterminate spinner widget.
//!
//! Rotation phase is read from a monotonic clock at draw time, so the widget
//! keeps no per-frame state; each frame self-arms a scheduled
//! [`window::RedrawRequest::At`] redraw capped at [`SPINNER_FPS`], mirroring the
//! editor's cursor blink — no subscription, no messages. The widget is only ever
//! placed in the tree while the target work item is running, so its presence in
//! the tree is the running signal: when it leaves the tree it stops re-arming
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
    /// The dot colour; each dot's alpha is scaled per the trailing fade.
    color: Color,
}

/// Create a spinning (constantly animating) [`Spinner`] of the given diameter
/// in the theme accent colour.
#[must_use]
pub(crate) fn spinner(diameter: f32) -> Spinner {
    Spinner {
        diameter,
        color: theme::ACCENT,
    }
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

/// Milliseconds remaining until the next rotation step boundary. Step is
/// [`PERIOD_MS`] / [`SPINNER_FPS`]; when exactly on a boundary returns
/// [`PERIOD_MS`] / [`SPINNER_FPS`] so the scheduled delay is never ~0 (callers
/// add +1 ms).
#[must_use]
fn next_step_in(elapsed_ms: u64) -> u64 {
    let step = PERIOD_MS / SPINNER_FPS;
    let rem = elapsed_ms % step;
    if rem == 0 { step } else { step - rem }
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
            let color = self.color.scale_alpha(dot_opacity(i));
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
        if let Event::Window(window::Event::RedrawRequested(_)) = event {
            let ms = phase_ms();
            let now = Instant::now();
            shell.request_redraw_at(window::RedrawRequest::At(
                now + Duration::from_millis(next_step_in(ms) + 1),
            ));
        }
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

#[cfg(test)]
mod tests {
    use super::{DOTS, PERIOD_MS, SPINNER_FPS, dot_opacity, next_step_in};

    #[test]
    fn next_step_in_boundaries() {
        let step = PERIOD_MS / SPINNER_FPS;
        assert_eq!(next_step_in(0), step);
        assert_eq!(next_step_in(step - 1), 1);
        assert_eq!(next_step_in(step), step);
        // Just above a boundary: one step past the boundary has `step - 1` to go.
        assert_eq!(next_step_in(step + 1), step - 1);
        // A full period later is again exactly on a boundary.
        assert_eq!(next_step_in(2 * step), step);
        // Well inside a step.
        assert_eq!(next_step_in(1), step - 1);
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
