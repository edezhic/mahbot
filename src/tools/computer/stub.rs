//! Stub [`Backend`] for unsupported platforms (neither macOS nor Linux).
//! Permanent — the tool is never advertised here.

use super::core::{
    self, AppInfo, Backend, Capture, ElementAct, Locator, Observation, RawInput, SurfaceGeometry,
    TargetSpec, WindowInfo,
};

pub(crate) static STUB_BACKEND: StubBackend = StubBackend;

pub(crate) struct StubBackend;

#[async_trait::async_trait]
impl Backend for StubBackend {
    fn accessibility_available(&self) -> bool {
        false
    }

    async fn capture_available(&self) -> bool {
        false
    }

    fn capture_unavailable_error(&self) -> anyhow::Error {
        core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            "the computer tool is not supported on this platform",
        )
    }

    async fn list_apps(&self) -> anyhow::Result<Vec<AppInfo>> {
        Err(Self::unsupported("list_apps"))
    }

    async fn list_windows(&self, _app: Option<&AppInfo>) -> anyhow::Result<Vec<WindowInfo>> {
        Err(Self::unsupported("list_windows"))
    }

    async fn focused_window(&self) -> anyhow::Result<WindowInfo> {
        Err(Self::unsupported("focused_window"))
    }

    async fn surface_geometry(&self, _target: &TargetSpec) -> anyhow::Result<SurfaceGeometry> {
        Err(Self::unsupported("surface_geometry"))
    }

    async fn observe(&self, _target: &TargetSpec) -> anyhow::Result<Observation> {
        Err(Self::unsupported("observe"))
    }

    async fn act_on_element(
        &self,
        _target: &TargetSpec,
        _locator: &Locator,
        _act: ElementAct,
    ) -> anyhow::Result<()> {
        Err(Self::unsupported("act_on_element"))
    }

    async fn raw_input(&self, _target: &TargetSpec, _input: RawInput) -> anyhow::Result<()> {
        Err(Self::unsupported("raw_input"))
    }

    async fn cursor_position(&self) -> anyhow::Result<(f64, f64)> {
        Err(Self::unsupported("cursor_position"))
    }

    async fn capture(&self, _target: &TargetSpec) -> anyhow::Result<Capture> {
        Err(Self::unsupported("capture"))
    }
}

impl StubBackend {
    fn unsupported(op: &str) -> anyhow::Error {
        core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            format!("{op}: the computer tool is not supported on this platform"),
        )
    }
}
