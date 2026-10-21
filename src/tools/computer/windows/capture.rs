//! Stills for the Windows backend: Windows.Graphics.Capture first, then the GDI
//! routes that still work when it cannot serve a frame.
//!
//! Routes, in order, per target:
//!
//! * a window — a `GraphicsCaptureItem` for the window (its own content), then a
//!   desktop `BitBlt` over the window's rect (what a person sees, occluders
//!   included);
//! * the screen — one capture item per display stitched into the virtual-desktop
//!   rect, then a single desktop `BitBlt`.
//!
//! The capture facility needs Windows 10 1903; older builds fall straight through
//! to the GDI route, which is also why observation degrades instead of failing.
//!
//! A route that yields nothing usable falls through to the next one; what "usable"
//! means is [`take_route`]'s rule. When every route comes back unusable the caller
//! gets an `unsupported` error naming each one and its honest cause.
//!
//! Everything here runs on the apartment thread. The D3D device is created lazily and
//! dropped when a route fails, so a lost device (a driver reset, a GPU switch) is
//! rebuilt by the next capture instead of leaving this route dead for the thread's
//! life.

use super::super::core::{self, Capture, SurfaceGeometry, TargetSpec};
use super::window;
use anyhow::anyhow;
use std::time::{Duration, Instant};
use windows::Graphics::Capture::{
    Direct3D11CaptureFrame, Direct3D11CaptureFramePool, GraphicsCaptureItem,
};
use windows::Graphics::DirectX::Direct3D11::IDirect3DDevice;
use windows::Graphics::DirectX::DirectXPixelFormat;
use windows::Win32::Foundation::{HMODULE, HWND, LPARAM, RECT};
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP, D3D_FEATURE_LEVEL_10_0, D3D_FEATURE_LEVEL_10_1,
    D3D_FEATURE_LEVEL_11_0,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_MAP_READ,
    D3D11_MAPPED_SUBRESOURCE, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING,
    D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{IDXGIAdapter, IDXGIDevice};
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BitBlt, CAPTUREBLT, CreateCompatibleDC, CreateDIBSection,
    DIB_RGB_COLORS, DeleteDC, DeleteObject, EnumDisplayMonitors, GdiFlush, GetDC, GetMonitorInfoW,
    HBITMAP, HDC, HGDIOBJ, HMONITOR, MONITORINFO, ReleaseDC, SRCCOPY, SelectObject,
};
use windows::Win32::System::WinRT::Direct3D11::{
    CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess,
};
use windows::Win32::System::WinRT::Graphics::Capture::IGraphicsCaptureItemInterop;
use windows::Win32::System::WinRT::RoGetActivationFactory;
use windows::core::{BOOL, HSTRING, IInspectable, Interface as _};

/// How long one route waits for a frame before the next route runs.
const FRAME_WAIT: Duration = Duration::from_millis(2_500);
/// The whole multi-display stitch, so a host with many displays cannot spend
/// [`super::CAPTURE_TIMEOUT`] waiting one display at a time.
const SCREEN_BUDGET: Duration = Duration::from_secs(12);
/// Poll interval while waiting — the free-threaded pool has no event we could await.
const FRAME_POLL: Duration = Duration::from_millis(10);
/// A pixel at or below this on every channel is "black" for the frame check.
const BLACK_LEVEL: u8 = 8;
/// How far a frame may differ from the surface it shows: a compositor rounds by a
/// pixel or two at a display or DWM seam, which is not a resize.
const EXTENT_TOLERANCE: u32 = 2;
/// Refuse an absurd allocation rather than trying it.
const MAX_CAPTURE_EDGE: f64 = 16_384.0;
const BLANK_CAUSE: &str = "an all-black frame (a minimized window, protected content, or a surface \
     that refuses readback)";

/// The capture side of the apartment thread: the reusable D3D device.
#[derive(Default)]
pub(super) struct CaptureContext {
    device: Option<Gpu>,
}

impl CaptureContext {
    pub(super) fn capture(&mut self, target: &TargetSpec) -> anyhow::Result<Capture> {
        match target {
            TargetSpec::Screen => self.screen(),
            other => {
                let resolved = window::resolve(other)?;
                self.window_frame(&resolved)
            }
        }
    }

    fn screen(&mut self) -> anyhow::Result<Capture> {
        let geometry = window::screen_geometry()?;
        let mut causes = Vec::new();
        let graphics = self.with_device(|gpu| screen_graphics(gpu, &geometry));
        if let Some(frame) = take_route(graphics, "Windows.Graphics.Capture", &mut causes) {
            return Ok(frame);
        }
        if let Some(frame) = take_route(blit_screen(&geometry), "BitBlt", &mut causes) {
            return Ok(frame);
        }
        Err(no_frame_error("the screen", &causes))
    }

    fn window_frame(&mut self, resolved: &window::TargetWindow) -> anyhow::Result<Capture> {
        let mut causes = Vec::new();
        let graphics = self.with_device(|gpu| window_graphics(gpu, resolved));
        if let Some(frame) = take_route(graphics, "Windows.Graphics.Capture", &mut causes) {
            return Ok(frame);
        }
        if let Some(frame) = take_route(blit_window(&resolved.surface), "BitBlt", &mut causes) {
            return Ok(frame);
        }
        Err(no_frame_error(
            &format!("window \"{}\"", resolved.title),
            &causes,
        ))
    }

    /// Run one capture-facility route on the cached device. A failure drops the device
    /// so the next capture builds a fresh one: a device loss (a driver reset, a GPU
    /// switch) must not leave this route dead for the apartment thread's life. The
    /// route is not retried — a determinate failure (no capture area, an unexpected
    /// pixel format, an extent the surface cannot explain) is not repaired by a fresh
    /// device, and a retry would double the route's cost past [`super::CAPTURE_TIMEOUT`].
    fn with_device(
        &mut self,
        route: impl Fn(&Gpu) -> anyhow::Result<Capture>,
    ) -> anyhow::Result<Capture> {
        let result = route(self.gpu()?);
        if result.is_err() {
            self.device = None;
        }
        result
    }

    /// The D3D device, created on first use and reused until a route fails (then
    /// dropped, see [`with_device`]).
    fn gpu(&mut self) -> anyhow::Result<&Gpu> {
        let gpu = match self.device.take() {
            Some(gpu) => gpu,
            None => Gpu::create()?,
        };
        Ok(self.device.insert(gpu))
    }
}

/// One window item's frame, aligned with the surface the model's coordinates refer
/// to (see [`align_to_surface`]).
fn window_graphics(gpu: &Gpu, resolved: &window::TargetWindow) -> anyhow::Result<Capture> {
    let item: GraphicsCaptureItem =
        // SAFETY: the interop factory is a live WinRT object on this apartment thread
        // and `resolved.hwnd` is an OS handle nothing here owns or closes; the item
        // comes back owned by this call.
        unsafe { capture_interop()?.CreateForWindow(resolved.hwnd) }?;
    let frame = grab(gpu, &item, FRAME_WAIT)?;
    align_to_surface(frame, resolved.hwnd, &resolved.surface)
}

/// The virtual desktop stitched from one capture item per display, within
/// [`SCREEN_BUDGET`].
fn screen_graphics(gpu: &Gpu, geometry: &SurfaceGeometry) -> anyhow::Result<Capture> {
    let displays = monitors()?;
    let width = pixels(geometry.width)?;
    let height = pixels(geometry.height)?;
    // Opaque black: an L-shaped desktop leaves gaps no display covers, and a
    // transparent hole would read as "nothing there" in the PNG.
    let mut buffer = opaque_canvas(width, height);
    let deadline = Instant::now() + SCREEN_BUDGET;
    for (monitor, rect) in displays {
        let item: GraphicsCaptureItem =
            // SAFETY: `monitor` is a handle `EnumDisplayMonitors` reported moments ago,
            // and the call is on the thread that owns both the factory and the device;
            // the item comes back owned here.
            unsafe { capture_interop()?.CreateForMonitor(monitor) }?;
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(anyhow!("the displays did not produce frames in time"));
        }
        let frame = grab(gpu, &item, remaining.min(FRAME_WAIT))?;
        let extent = SurfaceGeometry {
            x: f64::from(rect.left),
            y: f64::from(rect.top),
            width: f64::from(rect.right - rect.left),
            height: f64::from(rect.bottom - rect.top),
        };
        let (expected_width, expected_height) = edge(&extent)?;
        if !matches_extent(&frame, expected_width, expected_height) {
            return Err(anyhow!(
                "a display's frame is {}x{} but the display is {expected_width}x{expected_height} — \
                 it changed during the capture",
                frame.width,
                frame.height
            ));
        }
        let dx = extent.x - geometry.x;
        let dy = extent.y - geometry.y;
        if dx < 0.0 || dy < 0.0 {
            return Err(anyhow!("a display lies outside the virtual desktop"));
        }
        core::blit_rgba(
            &mut buffer,
            width as usize,
            height as usize,
            &frame,
            dx as usize,
            dy as usize,
        );
    }
    Ok(Capture {
        width,
        height,
        rgba: buffer,
    })
}

/// A frame from the capture facility, or the reason there was none.
fn grab(gpu: &Gpu, item: &GraphicsCaptureItem, wait: Duration) -> anyhow::Result<Capture> {
    let size = item.Size()?;
    if size.Width <= 0 || size.Height <= 0 {
        return Err(anyhow!(
            "the target has no capture area ({}x{}) — it was minimized, resized or closed \
             during the capture",
            size.Width,
            size.Height
        ));
    }
    // The free-threaded pool needs no dispatcher queue, which is exactly the shape
    // of this thread; the older `Create` is the fallback for an older build.
    let pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
        &gpu.winrt,
        DirectXPixelFormat::B8G8R8A8UIntNormalized,
        2,
        size,
    )
    .or_else(|_| {
        Direct3D11CaptureFramePool::Create(
            &gpu.winrt,
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,
            size,
        )
    })?;
    let session = pool.CreateCaptureSession(item)?;
    // Cursor inclusion is best-effort (Windows 10 2004+); the capture indicator the
    // platform draws is not touched at all.
    let _ = session.SetIsCursorCaptureEnabled(true);
    session.StartCapture()?;
    let frame = match wait_for_frame(&pool, wait) {
        Ok(frame) => frame,
        Err(e) => {
            let _ = session.Close();
            let _ = pool.Close();
            return Err(e);
        }
    };
    // Split out so the `?` inside the read cannot skip the closes below: the session
    // and the pool are released whichever way it ends.
    let pixels = read_frame(gpu, &frame);
    let _ = frame.Close();
    let _ = session.Close();
    let _ = pool.Close();
    pixels
}

/// The frame's pixels, through the D3D texture its surface exposes.
fn read_frame(gpu: &Gpu, frame: &Direct3D11CaptureFrame) -> anyhow::Result<Capture> {
    let surface = frame.Surface()?;
    let access: IDirect3DDxgiInterfaceAccess = surface.cast()?;
    // SAFETY: `access` is the DXGI access interface of the frame's live surface; the
    // texture it yields is an owned, refcounted `ID3D11Texture2D`.
    let texture: ID3D11Texture2D = unsafe { access.GetInterface()? };
    read_texture(gpu, &texture)
}

fn wait_for_frame(
    pool: &Direct3D11CaptureFramePool,
    wait: Duration,
) -> anyhow::Result<Direct3D11CaptureFrame> {
    let deadline = Instant::now() + wait;
    loop {
        match pool.TryGetNextFrame() {
            Ok(frame) => return Ok(frame),
            Err(_) if Instant::now() >= deadline => {
                return Err(anyhow!("no frame arrived within {}ms", wait.as_millis()));
            }
            Err(_) => std::thread::sleep(FRAME_POLL),
        }
    }
}

/// Copy the captured texture into a CPU-readable staging texture and flatten it to
/// opaque RGBA. The frame's own alpha is dropped: a still is what the screen shows,
/// and a transparent pixel in the PNG would read as "nothing there".
fn read_texture(gpu: &Gpu, texture: &ID3D11Texture2D) -> anyhow::Result<Capture> {
    let mut source = D3D11_TEXTURE2D_DESC::default();
    // SAFETY: `texture` is a live owned interface and `source` is a live local of
    // exactly the type `GetDesc` writes.
    unsafe { texture.GetDesc(&raw mut source) };
    if source.Format != DXGI_FORMAT_B8G8R8A8_UNORM {
        return Err(anyhow!(
            "the capture facility returned an unexpected pixel format ({}) — an HDR or \
             wide-gamut surface; use observe instead",
            source.Format.0
        ));
    }
    let staging = D3D11_TEXTURE2D_DESC {
        Width: source.Width,
        Height: source.Height,
        MipLevels: 1,
        ArraySize: 1,
        Format: source.Format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_STAGING,
        BindFlags: 0,
        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0.cast_unsigned(),
        MiscFlags: 0,
    };
    let mut staging_texture = None;
    // SAFETY: `staging` is a live desc built from the source texture's own fields, and
    // `staging_texture` is a local out-slot read only after this succeeds.
    unsafe {
        gpu.device
            .CreateTexture2D(&raw const staging, None, Some(&raw mut staging_texture))
    }?;
    let staging_texture =
        staging_texture.ok_or_else(|| anyhow!("the staging texture could not be created"))?;
    // SAFETY: `gpu.context` is the immediate context of the device that created the
    // staging texture and backs the frame pool the source came from; the staging
    // texture repeats the source's own size and format, which `CopyResource` requires.
    unsafe { gpu.context.CopyResource(&staging_texture, texture) };
    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    // SAFETY: the staging texture was created CPU-readable and is not bound for any
    // other use here; `mapped` is a live local out-slot for the mapped row pitch.
    unsafe {
        gpu.context.Map(
            &staging_texture,
            0,
            D3D11_MAP_READ,
            0,
            Some(&raw mut mapped),
        )
    }?;
    let width = source.Width;
    let height = source.Height;
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    for row in 0..height {
        let start = (mapped.pData as usize) + row as usize * mapped.RowPitch as usize;
        // SAFETY: `pData` is the mapping made above and stays valid until the `Unmap`
        // below; D3D11 guarantees `RowPitch >= width * 4` for this format, so row
        // `row`'s `width * 4` bytes lie inside the mapped image, `u8`-aligned.
        let line = unsafe { std::slice::from_raw_parts(start as *const u8, width as usize * 4) };
        for pixel in line.as_chunks::<4>().0 {
            push_pixel(&mut rgba, pixel);
        }
    }
    // SAFETY: it releases exactly the mapping `Map` made above, after the last read
    // of `pData`; no `line` slice is alive past this point.
    unsafe { gpu.context.Unmap(&staging_texture, 0) };
    Ok(Capture {
        width,
        height,
        rgba,
    })
}

/// The reusable D3D11 device, also exposed as the WinRT device the frame pool takes.
struct Gpu {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    winrt: IDirect3DDevice,
}

impl Gpu {
    fn create() -> anyhow::Result<Self> {
        let levels = [
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0,
        ];
        let mut device = None;
        let mut context = None;
        // Hardware first; WARP (the software rasterizer) keeps capture working on a
        // machine whose GPU driver refuses a device.
        // SAFETY: no adapter object is passed, `levels` is a three-entry array whose
        // slice length the binding passes as the feature-level count, and `device` /
        // `context` are local out-slots read only after this succeeds.
        let mut created = unsafe {
            D3D11CreateDevice(
                None::<&IDXGIAdapter>,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                Some(&levels),
                D3D11_SDK_VERSION,
                Some(&raw mut device),
                None,
                Some(&raw mut context),
            )
        };
        if created.is_err() {
            // SAFETY: as above — the same three-entry level array and the same local
            // out-slots, which a failed first attempt left nothing of use in.
            created = unsafe {
                D3D11CreateDevice(
                    None::<&IDXGIAdapter>,
                    D3D_DRIVER_TYPE_WARP,
                    HMODULE::default(),
                    D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                    Some(&levels),
                    D3D11_SDK_VERSION,
                    Some(&raw mut device),
                    None,
                    Some(&raw mut context),
                )
            };
        }
        created?;
        let device = device.ok_or_else(|| anyhow!("no D3D11 device was returned"))?;
        let context = context.ok_or_else(|| anyhow!("no D3D11 context was returned"))?;
        let dxgi: IDXGIDevice = device.cast()?;
        // SAFETY: `dxgi` is a live interface of the device created just above, and the
        // call runs on the initialized apartment thread; the inspectable is owned here.
        let inspectable: IInspectable = unsafe { CreateDirect3D11DeviceFromDXGIDevice(&dxgi) }?;
        let winrt: IDirect3DDevice = inspectable.cast()?;
        Ok(Self {
            device,
            context,
            winrt,
        })
    }
}

/// The activation factory that turns an `HWND`/`HMONITOR` into a capture item.
fn capture_interop() -> anyhow::Result<IGraphicsCaptureItemInterop> {
    let factory = HSTRING::from("Windows.Graphics.Capture.GraphicsCaptureItem");
    // SAFETY: `factory` is a live `HSTRING`; WinRT activation needs an initialized
    // apartment, which `thread::Ctx::open` did on this very thread before any capture.
    Ok(unsafe { RoGetActivationFactory(&factory) }?)
}

fn monitors() -> anyhow::Result<Vec<(HMONITOR, RECT)>> {
    let mut out: Vec<(HMONITOR, RECT)> = Vec::new();
    // SAFETY: `out` is a live local that outlives this synchronous enumeration and is
    // handed to `collect_monitor` by address through the `LPARAM`.
    let enumerated = unsafe {
        EnumDisplayMonitors(
            None,
            None,
            Some(collect_monitor),
            LPARAM(std::ptr::from_mut(&mut out).cast::<std::ffi::c_void>() as isize),
        )
    };
    if !enumerated.as_bool() {
        return Err(anyhow!("the displays could not be enumerated"));
    }
    if out.is_empty() {
        return Err(anyhow!("no display was reported"));
    }
    Ok(out)
}

// SAFETY: only ever installed as `EnumDisplayMonitors`' callback, with `dwdata` set
// to the address of `monitors`' own `Vec`; `monitor` is the OS's handle for this
// iteration (not owned here), and returning TRUE keeps the enumeration going.
unsafe extern "system" fn collect_monitor(
    monitor: HMONITOR,
    _dc: HDC,
    _rect: *mut RECT,
    data: LPARAM,
) -> BOOL {
    let mut info = MONITORINFO {
        cbSize: std::mem::size_of::<MONITORINFO>() as u32,
        ..Default::default()
    };
    // SAFETY: `monitor` came from the enumeration and `info` is a live local whose
    // `cbSize` is its own size, which is what the call checks before writing.
    if unsafe { GetMonitorInfoW(monitor, &raw mut info) }.as_bool() {
        // SAFETY: `data` is that `&mut Vec` from `monitors`, still live and only
        // reachable here, so it is aligned and pushable.
        let out = unsafe { &mut *(data.0 as *mut Vec<(HMONITOR, RECT)>) };
        out.push((monitor, info.rcMonitor));
    }
    BOOL(1)
}

// ── GDI routes ──────────────────────────────────────────────────────────

/// A 32-bit top-down DIB section selected into a memory DC, freed on drop.
struct Dib {
    dc: HDC,
    bitmap: HBITMAP,
    previous: HGDIOBJ,
    bits: *mut std::ffi::c_void,
    width: u32,
    height: u32,
}

impl Dib {
    fn create(width: u32, height: u32) -> anyhow::Result<Self> {
        let info = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: i32::try_from(width).unwrap_or(i32::MAX),
                // Negative: a top-down bitmap, so row 0 is the top row.
                biHeight: -i32::try_from(height).unwrap_or(i32::MAX),
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0,
                ..Default::default()
            },
            ..Default::default()
        };
        // SAFETY: no arguments; a null DC is refused right below and the handle is
        // then either released on that error path or owned by the returned `Dib`.
        let dc = unsafe { CreateCompatibleDC(None) };
        if dc.is_invalid() {
            return Err(anyhow!("no GDI memory device context"));
        }
        let mut bits = std::ptr::null_mut();
        // SAFETY: `info` is a live `BITMAPINFO` local with `biSize` set, and `bits` is a
        // local out-slot that is only dereferenced once both the bitmap and it exist.
        let created = unsafe {
            CreateDIBSection(
                None,
                &raw const info,
                DIB_RGB_COLORS,
                &raw mut bits,
                None,
                0,
            )
        };
        let bitmap = match created {
            Ok(bitmap) if !bits.is_null() => bitmap,
            _ => {
                // SAFETY: `dc` came from `CreateCompatibleDC` above and was never
                // stored in a `Dib`; this is its only release.
                unsafe {
                    let _ = DeleteDC(dc);
                }
                return Err(anyhow!("no GDI DIB section"));
            }
        };
        // SAFETY: both handles are live and owned here — the DC just created and the
        // bitmap just returned — and the previous object is kept so `Drop` can undo it.
        let previous = unsafe { SelectObject(dc, HGDIOBJ(bitmap.0)) };
        Ok(Self {
            dc,
            bitmap,
            previous,
            bits,
            width,
            height,
        })
    }

    fn read_back(&self) -> anyhow::Result<Capture> {
        if self.bits.is_null() {
            return Err(anyhow!("the DIB section has no pixels"));
        }
        // SAFETY: `bits` is the base of a 32-bit top-down DIB section of
        // `width * height` that this struct owns and frees only in `drop`, so exactly
        // `width * height * 4` bytes are live, `u8`-aligned, read-only here.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.bits.cast::<u8>(),
                self.width as usize * self.height as usize * 4,
            )
        };
        let mut rgba = Vec::with_capacity(bytes.len());
        for pixel in bytes.as_chunks::<4>().0 {
            push_pixel(&mut rgba, pixel);
        }
        Ok(Capture {
            width: self.width,
            height: self.height,
            rgba,
        })
    }
}

impl Drop for Dib {
    fn drop(&mut self) {
        // SAFETY: every handle is this struct's own, stored once at creation and
        // released once here; the DC goes last, so the old object is back in it first.
        unsafe {
            SelectObject(self.dc, self.previous);
            let _ = DeleteObject(HGDIOBJ(self.bitmap.0));
            let _ = DeleteDC(self.dc);
        }
    }
}

/// The desktop over a window's rect: what a person looking at the screen sees,
/// occluders included. `window_capture_rect` answers in buffer space (relative to
/// the desktop origin), while `BitBlt` wants absolute screen coordinates, so both
/// are converted here. Off-screen parts stay black, keeping the image aligned with
/// the window's own surface for the zoom regions.
fn blit_window(surface: &SurfaceGeometry) -> anyhow::Result<Capture> {
    let desktop = window::screen_geometry()?;
    let (buffer_x, buffer_y, width, height) = core::window_capture_rect(surface, &desktop)?;
    let (full_width, full_height) = edge(surface)?;
    let (source_x, source_y) = (
        buffer_x as i32 + desktop.x.round() as i32,
        buffer_y as i32 + desktop.y.round() as i32,
    );
    blit_rect(
        (
            source_x - surface.x.round() as i32,
            source_y - surface.y.round() as i32,
        ),
        (full_width, full_height),
        (source_x, source_y),
        (width, height),
    )
}

/// The whole virtual desktop, whose origin can be negative.
fn blit_screen(geometry: &SurfaceGeometry) -> anyhow::Result<Capture> {
    let width = pixels(geometry.width)?;
    let height = pixels(geometry.height)?;
    let (x, y) = (geometry.x.round() as i32, geometry.y.round() as i32);
    blit_rect((0, 0), (width, height), (x, y), (width, height))
}

/// Copy a screen rect into a `canvas`-sized DIB at `dest`. The canvas keeps the
/// image aligned with the surface the model's coordinates refer to.
fn blit_rect(
    dest: (i32, i32),
    canvas: (u32, u32),
    source: (i32, i32),
    region: (u32, u32),
) -> anyhow::Result<Capture> {
    let dib = Dib::create(canvas.0, canvas.1)?;
    // SAFETY: no arguments; a null DC is refused right below and released in here.
    let screen = unsafe { GetDC(None) };
    if screen.is_invalid() {
        return Err(anyhow!("no screen device context"));
    }
    // CAPTUREBLT includes layered windows (menus, tooltips) in the frame.
    // SAFETY: `dib.dc` holds this call's own `canvas`-sized 32-bit DIB and `screen` is
    // the DC from above; GDI clips the copy to the destination bitmap.
    let copied = unsafe {
        BitBlt(
            dib.dc,
            dest.0,
            dest.1,
            i32::try_from(region.0).unwrap_or(i32::MAX),
            i32::try_from(region.1).unwrap_or(i32::MAX),
            Some(screen),
            source.0,
            source.1,
            SRCCOPY | CAPTUREBLT,
        )
    };
    // SAFETY: `screen` came from `GetDC(None)` above and is released here exactly once.
    unsafe { ReleaseDC(None, screen) };
    copied?;
    // GDI batches; the DIB bits are not guaranteed to hold the drawing until this.
    // SAFETY: no arguments and no state of ours; it only flushes GDI's own batch.
    let _ = unsafe { GdiFlush() };
    dib.read_back()
}

// ── Frame checks and errors ─────────────────────────────────────────────

/// A black, fully opaque canvas: gaps no display covers stay black rather than
/// transparent, which would read as "nothing there" in the PNG.
fn opaque_canvas(width: u32, height: u32) -> Vec<u8> {
    let mut rgba = vec![0u8; width as usize * height as usize * 4];
    for pixel in rgba.as_chunks_mut::<4>().0 {
        pixel[3] = 255;
    }
    rgba
}

/// BGRA in, opaque RGBA out — the one place that byte order is stated.
fn push_pixel(rgba: &mut Vec<u8>, pixel: &[u8]) {
    rgba.extend_from_slice(&[pixel[2], pixel[1], pixel[0], 255]);
}

/// Crop a window frame to the surface its coordinates are expressed in. A capture
/// item may report the visible DWM frame (`surface`) or the window rect with its
/// invisible resize border; both offsets are known exactly, so a larger frame is
/// cropped to the surface instead of being rejected. Any other extent means the
/// window changed size during the grab.
///
/// [`core::crop_rgba`] clamps to the frame it is given, so a frame within
/// [`EXTENT_TOLERANCE`] of the surface comes back a pixel or two narrower — the
/// coordinates still name the surface.
fn align_to_surface(
    frame: Capture,
    hwnd: HWND,
    surface: &SurfaceGeometry,
) -> anyhow::Result<Capture> {
    let (width, height) = edge(surface)?;
    if matches_extent(&frame, width, height) {
        return Ok(frame);
    }
    let outer = window::window_rect(hwnd)?;
    let (outer_width, outer_height) = edge(&outer)?;
    if matches_extent(&frame, outer_width, outer_height) {
        return core::crop_rgba(
            &frame,
            (
                crop_offset(surface.x - outer.x)?,
                crop_offset(surface.y - outer.y)?,
                width,
                height,
            ),
        );
    }
    Err(anyhow!(
        "the frame is {}x{} but the window surface is {width}x{height} — the window changed size \
         during the capture",
        frame.width,
        frame.height
    ))
}

/// Where the visible frame starts inside the window rect. An offset outside the rect
/// the compositor just reported means a resize during the grab, refused rather than
/// clamped onto the wrong pixel.
fn crop_offset(value: f64) -> anyhow::Result<u32> {
    let value = value.round();
    if !(0.0..=f64::from(u32::MAX)).contains(&value) {
        return Err(anyhow!(
            "the visible frame starts outside the window rect — the window changed size during the \
             capture"
        ));
    }
    Ok(value as u32)
}

/// Whether a frame is exactly the surface's size, up to [`EXTENT_TOLERANCE`].
fn matches_extent(frame: &Capture, width: u32, height: u32) -> bool {
    frame.width.abs_diff(width) <= EXTENT_TOLERANCE
        && frame.height.abs_diff(height) <= EXTENT_TOLERANCE
}

/// One capture route's answer: the frame, or the route's labelled cause appended to
/// `causes`. This is the one place the route rule is stated — a route that returns an
/// empty or all-black frame is a FAILURE, labelled with the route that produced it,
/// and the next route runs.
fn take_route(
    route: anyhow::Result<Capture>,
    label: &str,
    causes: &mut Vec<String>,
) -> Option<Capture> {
    match route {
        Ok(frame) if !is_blank(&frame) => return Some(frame),
        Ok(_) => causes.push(format!("{label}: {BLANK_CAUSE}")),
        Err(e) => causes.push(format!("{label}: {e:#}")),
    }
    None
}

/// Whether a frame carries no picture at all: every pixel is at or below
/// [`BLACK_LEVEL`]. The whole buffer is scanned — sampling a stride can call a small
/// mostly-black window (one line of text in a 100x30 frame) blank and report a
/// failure for a capture that worked.
fn is_blank(capture: &Capture) -> bool {
    if capture.width == 0 || capture.height == 0 {
        return true;
    }
    let expected = capture.width as usize * capture.height as usize * 4;
    if capture.rgba.len() < expected {
        return true;
    }
    !capture
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .any(|pixel| pixel[0] > BLACK_LEVEL || pixel[1] > BLACK_LEVEL || pixel[2] > BLACK_LEVEL)
}

fn no_frame_error(target: &str, causes: &[String]) -> anyhow::Error {
    core::taxonomy_error(
        core::ERR_UNSUPPORTED,
        format!(
            "capture of {target} produced no usable frame — every route failed: {}; use observe for \
             the structure, or ask the owner to restore/focus the window",
            causes.join("; ")
        ),
    )
}

/// A surface edge in pixels, refusing a degenerate or absurd size.
fn edge(surface: &SurfaceGeometry) -> anyhow::Result<(u32, u32)> {
    if surface.width < 1.0 || surface.height < 1.0 {
        return Err(core::taxonomy_error(
            core::ERR_NOT_MATCHED,
            "the target has no visible area — re-enumerate windows",
        ));
    }
    if surface.width > MAX_CAPTURE_EDGE || surface.height > MAX_CAPTURE_EDGE {
        return Err(anyhow!(
            "the target is larger than a capture can hold ({}x{})",
            surface.width,
            surface.height
        ));
    }
    Ok((pixels(surface.width)?, pixels(surface.height)?))
}

fn pixels(value: f64) -> anyhow::Result<u32> {
    if !value.is_finite() || value < 1.0 || value > MAX_CAPTURE_EDGE {
        return Err(anyhow!("invalid capture extent {value}"));
    }
    Ok(value.round() as u32)
}
