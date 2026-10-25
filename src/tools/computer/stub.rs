//! Stub [`Backend`] for platforms with no computer-use implementation — every
//! platform except Linux, macOS and Windows. Permanent: the tool is never
//! advertised here, because the capability itself is absent.

use super::core::{
    self, Actuation, AppInfo, AppListing, Backend, Capture, CaptureBudget, ElementAct, Impact,
    Locator, Observation, RawInput, SurfaceGeometry, TargetSpec, WindowInfo, WindowListing,
};

pub(crate) static STUB_BACKEND: StubBackend = StubBackend;

pub(crate) struct StubBackend;

#[async_trait::async_trait]
impl Backend for StubBackend {
    fn is_supported(&self) -> bool {
        false
    }

    /// Never reached: nothing here is advertised, so no call gets far enough to ask.
    async fn desktop_unavailable(&self) -> Option<String> {
        None
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

    async fn list_apps(&self) -> anyhow::Result<AppListing> {
        Err(Self::unsupported("list_apps"))
    }

    async fn list_windows(&self, _app: Option<&AppInfo>) -> anyhow::Result<WindowListing> {
        Err(Self::unsupported("list_windows"))
    }

    async fn surface_window(&self, _target: &TargetSpec) -> anyhow::Result<WindowInfo> {
        Err(Self::unsupported("surface_window"))
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
    ) -> anyhow::Result<Impact> {
        Err(Self::unsupported("act_on_element"))
    }

    async fn raw_input(&self, _target: &TargetSpec, _input: RawInput) -> anyhow::Result<Actuation> {
        Err(Self::unsupported("raw_input"))
    }

    async fn cursor_position(&self) -> anyhow::Result<(f64, f64)> {
        Err(Self::unsupported("cursor_position"))
    }

    async fn capture(
        &self,
        _target: &TargetSpec,
        _budget: &CaptureBudget,
    ) -> anyhow::Result<Capture> {
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
