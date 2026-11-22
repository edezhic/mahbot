// The bindings `alsa` 0.11 compiles against, wrapped so every call is resolved through `dlopen`.
//
// Generated — do not edit by hand: `regenerate.sh` beside this crate is the recipe. The items below
// are the part of the published `alsa-sys` 0.4.0 `src/generated.rs` (MIT, see `LICENSE-ALSA-SYS`)
// that `alsa` 0.11 reaches — the names its own sources spell, and whatever the signatures of those
// need — with every `extern "C"` declaration replaced by a `dlopen` wrapper of the same name and
// signature. The rest of the published file binds libasound beyond any caller of the crate and is
// not carried. The variadic declarations, which Rust cannot express and `alsa` does not call, are
// dropped. One surface serves both Linux targets: it was generated on x86_64, so `__va_list_tag`
// carries that layout on either architecture, which is safe because `alsa` only forwards one as an
// opaque pointer.
use std::sync::OnceLock;

struct __AlsaLibrary(*mut core::ffi::c_void);
// The handle lives for the process and is only read through `dlsym`.
unsafe impl Send for __AlsaLibrary {}
unsafe impl Sync for __AlsaLibrary {}

fn __alsa_library() -> Option<&'static __AlsaLibrary> {
    static LIBRARY: OnceLock<Option<__AlsaLibrary>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            // The soname every glibc distribution ships. `RTLD_LAZY` keeps one symbol this
            // build does not know from failing the whole load; `RTLD_LOCAL` keeps ALSA's
            // symbols out of every other library's namespace.
            let handle = unsafe {
                libc::dlopen(
                    c"libasound.so.2".as_ptr(),
                    libc::RTLD_LAZY | libc::RTLD_LOCAL,
                )
            };
            (!handle.is_null()).then_some(__AlsaLibrary(handle))
        })
        .as_ref()
}

unsafe fn __alsa_symbol<F>(name: &[u8]) -> Option<F> {
    let library = __alsa_library()?;
    let symbol = unsafe { libc::dlsym(library.0, name.as_ptr().cast()) };
    if symbol.is_null() {
        None
    } else {
        Some(unsafe { core::mem::transmute_copy(&symbol) })
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct __BindgenBitfieldUnit<Storage> {
    storage: Storage,
}
impl<Storage> __BindgenBitfieldUnit<Storage> {
    #[inline]
    pub const fn new(storage: Storage) -> Self {
        Self { storage }
    }
}
impl<Storage> __BindgenBitfieldUnit<Storage>
where
    Storage: AsRef<[u8]> + AsMut<[u8]>,
{
    #[inline]
    fn extract_bit(byte: u8, index: usize) -> bool {
        let bit_index = if cfg!(target_endian = "big") {
            7 - (index % 8)
        } else {
            index % 8
        };
        let mask = 1 << bit_index;
        byte & mask == mask
    }
    #[inline]
    pub fn get_bit(&self, index: usize) -> bool {
        debug_assert!(index / 8 < self.storage.as_ref().len());
        let byte_index = index / 8;
        let byte = self.storage.as_ref()[byte_index];
        Self::extract_bit(byte, index)
    }
    #[inline]
    pub unsafe fn raw_get_bit(this: *const Self, index: usize) -> bool {
        debug_assert!(index / 8 < core::mem::size_of::<Storage>());
        let byte_index = index / 8;
        let byte = unsafe {
            *(core::ptr::addr_of!((*this).storage) as *const u8).offset(byte_index as isize)
        };
        Self::extract_bit(byte, index)
    }
    #[inline]
    fn change_bit(byte: u8, index: usize, val: bool) -> u8 {
        let bit_index = if cfg!(target_endian = "big") {
            7 - (index % 8)
        } else {
            index % 8
        };
        let mask = 1 << bit_index;
        if val { byte | mask } else { byte & !mask }
    }
    #[inline]
    pub fn set_bit(&mut self, index: usize, val: bool) {
        debug_assert!(index / 8 < self.storage.as_ref().len());
        let byte_index = index / 8;
        let byte = &mut self.storage.as_mut()[byte_index];
        *byte = Self::change_bit(*byte, index, val);
    }
    #[inline]
    pub unsafe fn raw_set_bit(this: *mut Self, index: usize, val: bool) {
        debug_assert!(index / 8 < core::mem::size_of::<Storage>());
        let byte_index = index / 8;
        let byte = unsafe {
            (core::ptr::addr_of_mut!((*this).storage) as *mut u8).offset(byte_index as isize)
        };
        unsafe { *byte = Self::change_bit(*byte, index, val) };
    }
    #[inline]
    pub fn get(&self, bit_offset: usize, bit_width: u8) -> u64 {
        debug_assert!(bit_width <= 64);
        debug_assert!(bit_offset / 8 < self.storage.as_ref().len());
        debug_assert!((bit_offset + (bit_width as usize)) / 8 <= self.storage.as_ref().len());
        let mut val = 0;
        for i in 0..(bit_width as usize) {
            if self.get_bit(i + bit_offset) {
                let index = if cfg!(target_endian = "big") {
                    bit_width as usize - 1 - i
                } else {
                    i
                };
                val |= 1 << index;
            }
        }
        val
    }
    #[inline]
    pub unsafe fn raw_get(this: *const Self, bit_offset: usize, bit_width: u8) -> u64 {
        debug_assert!(bit_width <= 64);
        debug_assert!(bit_offset / 8 < core::mem::size_of::<Storage>());
        debug_assert!((bit_offset + (bit_width as usize)) / 8 <= core::mem::size_of::<Storage>());
        let mut val = 0;
        for i in 0..(bit_width as usize) {
            if unsafe { Self::raw_get_bit(this, i + bit_offset) } {
                let index = if cfg!(target_endian = "big") {
                    bit_width as usize - 1 - i
                } else {
                    i
                };
                val |= 1 << index;
            }
        }
        val
    }
    #[inline]
    pub fn set(&mut self, bit_offset: usize, bit_width: u8, val: u64) {
        debug_assert!(bit_width <= 64);
        debug_assert!(bit_offset / 8 < self.storage.as_ref().len());
        debug_assert!((bit_offset + (bit_width as usize)) / 8 <= self.storage.as_ref().len());
        for i in 0..(bit_width as usize) {
            let mask = 1 << i;
            let val_bit_is_set = val & mask == mask;
            let index = if cfg!(target_endian = "big") {
                bit_width as usize - 1 - i
            } else {
                i
            };
            self.set_bit(index + bit_offset, val_bit_is_set);
        }
    }
    #[inline]
    pub unsafe fn raw_set(this: *mut Self, bit_offset: usize, bit_width: u8, val: u64) {
        debug_assert!(bit_width <= 64);
        debug_assert!(bit_offset / 8 < core::mem::size_of::<Storage>());
        debug_assert!((bit_offset + (bit_width as usize)) / 8 <= core::mem::size_of::<Storage>());
        for i in 0..(bit_width as usize) {
            let mask = 1 << i;
            let val_bit_is_set = val & mask == mask;
            let index = if cfg!(target_endian = "big") {
                bit_width as usize - 1 - i
            } else {
                i
            };
            unsafe { Self::raw_set_bit(this, index + bit_offset, val_bit_is_set) };
        }
    }
}
#[repr(C)]
#[derive(Default)]
pub struct __IncompleteArrayField<T>(::core::marker::PhantomData<T>, [T; 0]);
impl<T> __IncompleteArrayField<T> {
    #[inline]
    pub const fn new() -> Self {
        __IncompleteArrayField(::core::marker::PhantomData, [])
    }
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self as *const _ as *const T
    }
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self as *mut _ as *mut T
    }
    #[inline]
    pub unsafe fn as_slice(&self, len: usize) -> &[T] {
        ::core::slice::from_raw_parts(self.as_ptr(), len)
    }
    #[inline]
    pub unsafe fn as_mut_slice(&mut self, len: usize) -> &mut [T] {
        ::core::slice::from_raw_parts_mut(self.as_mut_ptr(), len)
    }
}
impl<T> ::core::fmt::Debug for __IncompleteArrayField<T> {
    fn fmt(&self, fmt: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        fmt.write_str("__IncompleteArrayField")
    }
}
pub type snd_htimestamp_t = timespec;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_output {
    _unused: [u8; 0],
}
pub type snd_output_t = _snd_output;
pub type snd_local_error_handler_t = ::core::option::Option<
    unsafe extern "C" fn(
        file: *const ::core::ffi::c_char,
        line: ::core::ffi::c_int,
        func: *const ::core::ffi::c_char,
        err: ::core::ffi::c_int,
        fmt: *const ::core::ffi::c_char,
        arg: *mut __va_list_tag,
    ),
>;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_config {
    _unused: [u8; 0],
}
pub type snd_config_t = _snd_config;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_pcm_info {
    _unused: [u8; 0],
}
pub type snd_pcm_info_t = _snd_pcm_info;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_pcm_hw_params {
    _unused: [u8; 0],
}
pub type snd_pcm_hw_params_t = _snd_pcm_hw_params;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_pcm_sw_params {
    _unused: [u8; 0],
}
pub type snd_pcm_sw_params_t = _snd_pcm_sw_params;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_pcm_status {
    _unused: [u8; 0],
}
pub type snd_pcm_status_t = _snd_pcm_status;
pub const SND_PCM_STREAM_PLAYBACK: _snd_pcm_stream = 0;
pub const SND_PCM_STREAM_CAPTURE: _snd_pcm_stream = 1;
pub type _snd_pcm_stream = ::core::ffi::c_uint;
pub use self::_snd_pcm_stream as snd_pcm_stream_t;
pub const SND_PCM_ACCESS_MMAP_INTERLEAVED: _snd_pcm_access = 0;
pub const SND_PCM_ACCESS_MMAP_NONINTERLEAVED: _snd_pcm_access = 1;
pub const SND_PCM_ACCESS_MMAP_COMPLEX: _snd_pcm_access = 2;
pub const SND_PCM_ACCESS_RW_INTERLEAVED: _snd_pcm_access = 3;
pub const SND_PCM_ACCESS_RW_NONINTERLEAVED: _snd_pcm_access = 4;
pub type _snd_pcm_access = ::core::ffi::c_uint;
pub use self::_snd_pcm_access as snd_pcm_access_t;
pub const SND_PCM_FORMAT_UNKNOWN: _snd_pcm_format = -1;
pub const SND_PCM_FORMAT_S8: _snd_pcm_format = 0;
pub const SND_PCM_FORMAT_U8: _snd_pcm_format = 1;
pub const SND_PCM_FORMAT_S16_LE: _snd_pcm_format = 2;
pub const SND_PCM_FORMAT_S16_BE: _snd_pcm_format = 3;
pub const SND_PCM_FORMAT_U16_LE: _snd_pcm_format = 4;
pub const SND_PCM_FORMAT_U16_BE: _snd_pcm_format = 5;
pub const SND_PCM_FORMAT_S24_LE: _snd_pcm_format = 6;
pub const SND_PCM_FORMAT_S24_BE: _snd_pcm_format = 7;
pub const SND_PCM_FORMAT_U24_LE: _snd_pcm_format = 8;
pub const SND_PCM_FORMAT_U24_BE: _snd_pcm_format = 9;
pub const SND_PCM_FORMAT_S32_LE: _snd_pcm_format = 10;
pub const SND_PCM_FORMAT_S32_BE: _snd_pcm_format = 11;
pub const SND_PCM_FORMAT_U32_LE: _snd_pcm_format = 12;
pub const SND_PCM_FORMAT_U32_BE: _snd_pcm_format = 13;
pub const SND_PCM_FORMAT_FLOAT_LE: _snd_pcm_format = 14;
pub const SND_PCM_FORMAT_FLOAT_BE: _snd_pcm_format = 15;
pub const SND_PCM_FORMAT_FLOAT64_LE: _snd_pcm_format = 16;
pub const SND_PCM_FORMAT_FLOAT64_BE: _snd_pcm_format = 17;
pub const SND_PCM_FORMAT_IEC958_SUBFRAME_LE: _snd_pcm_format = 18;
pub const SND_PCM_FORMAT_IEC958_SUBFRAME_BE: _snd_pcm_format = 19;
pub const SND_PCM_FORMAT_MU_LAW: _snd_pcm_format = 20;
pub const SND_PCM_FORMAT_A_LAW: _snd_pcm_format = 21;
pub const SND_PCM_FORMAT_IMA_ADPCM: _snd_pcm_format = 22;
pub const SND_PCM_FORMAT_MPEG: _snd_pcm_format = 23;
pub const SND_PCM_FORMAT_GSM: _snd_pcm_format = 24;
pub const SND_PCM_FORMAT_S20_LE: _snd_pcm_format = 25;
pub const SND_PCM_FORMAT_S20_BE: _snd_pcm_format = 26;
pub const SND_PCM_FORMAT_U20_LE: _snd_pcm_format = 27;
pub const SND_PCM_FORMAT_U20_BE: _snd_pcm_format = 28;
pub const SND_PCM_FORMAT_SPECIAL: _snd_pcm_format = 31;
pub const SND_PCM_FORMAT_S24_3LE: _snd_pcm_format = 32;
pub const SND_PCM_FORMAT_S24_3BE: _snd_pcm_format = 33;
pub const SND_PCM_FORMAT_U24_3LE: _snd_pcm_format = 34;
pub const SND_PCM_FORMAT_U24_3BE: _snd_pcm_format = 35;
pub const SND_PCM_FORMAT_S20_3LE: _snd_pcm_format = 36;
pub const SND_PCM_FORMAT_S20_3BE: _snd_pcm_format = 37;
pub const SND_PCM_FORMAT_U20_3LE: _snd_pcm_format = 38;
pub const SND_PCM_FORMAT_U20_3BE: _snd_pcm_format = 39;
pub const SND_PCM_FORMAT_S18_3LE: _snd_pcm_format = 40;
pub const SND_PCM_FORMAT_S18_3BE: _snd_pcm_format = 41;
pub const SND_PCM_FORMAT_U18_3LE: _snd_pcm_format = 42;
pub const SND_PCM_FORMAT_U18_3BE: _snd_pcm_format = 43;
pub const SND_PCM_FORMAT_G723_24: _snd_pcm_format = 44;
pub const SND_PCM_FORMAT_G723_24_1B: _snd_pcm_format = 45;
pub const SND_PCM_FORMAT_G723_40: _snd_pcm_format = 46;
pub const SND_PCM_FORMAT_G723_40_1B: _snd_pcm_format = 47;
pub const SND_PCM_FORMAT_DSD_U8: _snd_pcm_format = 48;
pub const SND_PCM_FORMAT_DSD_U16_LE: _snd_pcm_format = 49;
pub const SND_PCM_FORMAT_DSD_U32_LE: _snd_pcm_format = 50;
pub const SND_PCM_FORMAT_DSD_U16_BE: _snd_pcm_format = 51;
pub const SND_PCM_FORMAT_DSD_U32_BE: _snd_pcm_format = 52;
pub type _snd_pcm_format = ::core::ffi::c_int;
pub use self::_snd_pcm_format as snd_pcm_format_t;
pub const SND_PCM_STATE_OPEN: _snd_pcm_state = 0;
pub const SND_PCM_STATE_SETUP: _snd_pcm_state = 1;
pub const SND_PCM_STATE_PREPARED: _snd_pcm_state = 2;
pub const SND_PCM_STATE_RUNNING: _snd_pcm_state = 3;
pub const SND_PCM_STATE_XRUN: _snd_pcm_state = 4;
pub const SND_PCM_STATE_DRAINING: _snd_pcm_state = 5;
pub const SND_PCM_STATE_PAUSED: _snd_pcm_state = 6;
pub const SND_PCM_STATE_SUSPENDED: _snd_pcm_state = 7;
pub const SND_PCM_STATE_DISCONNECTED: _snd_pcm_state = 8;
pub type _snd_pcm_state = ::core::ffi::c_uint;
pub use self::_snd_pcm_state as snd_pcm_state_t;
pub const SND_PCM_TSTAMP_NONE: _snd_pcm_tstamp = 0;
pub const SND_PCM_TSTAMP_ENABLE: _snd_pcm_tstamp = 1;
pub type _snd_pcm_tstamp = ::core::ffi::c_uint;
pub use self::_snd_pcm_tstamp as snd_pcm_tstamp_t;
pub const SND_PCM_TSTAMP_TYPE_GETTIMEOFDAY: _snd_pcm_tstamp_type = 0;
pub const SND_PCM_TSTAMP_TYPE_MONOTONIC: _snd_pcm_tstamp_type = 1;
pub const SND_PCM_TSTAMP_TYPE_MONOTONIC_RAW: _snd_pcm_tstamp_type = 2;
pub type _snd_pcm_tstamp_type = ::core::ffi::c_uint;
pub use self::_snd_pcm_tstamp_type as snd_pcm_tstamp_type_t;
pub const SND_PCM_AUDIO_TSTAMP_TYPE_COMPAT: _snd_pcm_audio_tstamp_type = 0;
pub const SND_PCM_AUDIO_TSTAMP_TYPE_DEFAULT: _snd_pcm_audio_tstamp_type = 1;
pub const SND_PCM_AUDIO_TSTAMP_TYPE_LINK: _snd_pcm_audio_tstamp_type = 2;
pub const SND_PCM_AUDIO_TSTAMP_TYPE_LINK_ABSOLUTE: _snd_pcm_audio_tstamp_type = 3;
pub const SND_PCM_AUDIO_TSTAMP_TYPE_LINK_ESTIMATED: _snd_pcm_audio_tstamp_type = 4;
pub const SND_PCM_AUDIO_TSTAMP_TYPE_LINK_SYNCHRONIZED: _snd_pcm_audio_tstamp_type = 5;
pub type _snd_pcm_audio_tstamp_type = ::core::ffi::c_uint;
#[repr(C)]
#[repr(align(4))]
#[derive(Debug, Copy, Clone)]
pub struct _snd_pcm_audio_tstamp_config {
    pub _bitfield_align_1: [u8; 0],
    pub _bitfield_1: __BindgenBitfieldUnit<[u8; 1usize]>,
    pub __bindgen_padding_0: [u8; 3usize],
}
impl _snd_pcm_audio_tstamp_config {
    #[inline]
    pub fn type_requested(&self) -> ::core::ffi::c_uint {
        unsafe { ::core::mem::transmute(self._bitfield_1.get(0usize, 4u8) as u32) }
    }
    #[inline]
    pub fn set_type_requested(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::core::mem::transmute(val);
            self._bitfield_1.set(0usize, 4u8, val as u64)
        }
    }
    #[inline]
    pub unsafe fn type_requested_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::core::mem::transmute(<__BindgenBitfieldUnit<[u8; 1usize]>>::raw_get(
                ::core::ptr::addr_of!((*this)._bitfield_1),
                0usize,
                4u8,
            ) as u32)
        }
    }
    #[inline]
    pub unsafe fn set_type_requested_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::core::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 1usize]>>::raw_set(
                ::core::ptr::addr_of_mut!((*this)._bitfield_1),
                0usize,
                4u8,
                val as u64,
            )
        }
    }
    #[inline]
    pub fn report_delay(&self) -> ::core::ffi::c_uint {
        unsafe { ::core::mem::transmute(self._bitfield_1.get(4usize, 1u8) as u32) }
    }
    #[inline]
    pub fn set_report_delay(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::core::mem::transmute(val);
            self._bitfield_1.set(4usize, 1u8, val as u64)
        }
    }
    #[inline]
    pub unsafe fn report_delay_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::core::mem::transmute(<__BindgenBitfieldUnit<[u8; 1usize]>>::raw_get(
                ::core::ptr::addr_of!((*this)._bitfield_1),
                4usize,
                1u8,
            ) as u32)
        }
    }
    #[inline]
    pub unsafe fn set_report_delay_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::core::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 1usize]>>::raw_set(
                ::core::ptr::addr_of_mut!((*this)._bitfield_1),
                4usize,
                1u8,
                val as u64,
            )
        }
    }
    #[inline]
    pub fn new_bitfield_1(
        type_requested: ::core::ffi::c_uint,
        report_delay: ::core::ffi::c_uint,
    ) -> __BindgenBitfieldUnit<[u8; 1usize]> {
        let mut __bindgen_bitfield_unit: __BindgenBitfieldUnit<[u8; 1usize]> = Default::default();
        __bindgen_bitfield_unit.set(0usize, 4u8, {
            let type_requested: u32 = unsafe { ::core::mem::transmute(type_requested) };
            type_requested as u64
        });
        __bindgen_bitfield_unit.set(4usize, 1u8, {
            let report_delay: u32 = unsafe { ::core::mem::transmute(report_delay) };
            report_delay as u64
        });
        __bindgen_bitfield_unit
    }
}
pub type snd_pcm_audio_tstamp_config_t = _snd_pcm_audio_tstamp_config;
pub type snd_pcm_uframes_t = ::core::ffi::c_ulong;
pub type snd_pcm_sframes_t = ::core::ffi::c_long;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_pcm {
    _unused: [u8; 0],
}
pub type snd_pcm_t = _snd_pcm;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_pcm_channel_area {
    pub addr: *mut ::core::ffi::c_void,
    pub first: ::core::ffi::c_uint,
    pub step: ::core::ffi::c_uint,
}
pub type snd_pcm_channel_area_t = _snd_pcm_channel_area;
pub const SND_CHMAP_TYPE_NONE: snd_pcm_chmap_type = 0;
pub const SND_CHMAP_TYPE_FIXED: snd_pcm_chmap_type = 1;
pub const SND_CHMAP_TYPE_VAR: snd_pcm_chmap_type = 2;
pub const SND_CHMAP_TYPE_PAIRED: snd_pcm_chmap_type = 3;
pub type snd_pcm_chmap_type = ::core::ffi::c_uint;
pub const SND_CHMAP_UNKNOWN: snd_pcm_chmap_position = 0;
pub const SND_CHMAP_NA: snd_pcm_chmap_position = 1;
pub const SND_CHMAP_MONO: snd_pcm_chmap_position = 2;
pub const SND_CHMAP_FL: snd_pcm_chmap_position = 3;
pub const SND_CHMAP_FR: snd_pcm_chmap_position = 4;
pub const SND_CHMAP_RL: snd_pcm_chmap_position = 5;
pub const SND_CHMAP_RR: snd_pcm_chmap_position = 6;
pub const SND_CHMAP_FC: snd_pcm_chmap_position = 7;
pub const SND_CHMAP_LFE: snd_pcm_chmap_position = 8;
pub const SND_CHMAP_SL: snd_pcm_chmap_position = 9;
pub const SND_CHMAP_SR: snd_pcm_chmap_position = 10;
pub const SND_CHMAP_RC: snd_pcm_chmap_position = 11;
pub const SND_CHMAP_FLC: snd_pcm_chmap_position = 12;
pub const SND_CHMAP_FRC: snd_pcm_chmap_position = 13;
pub const SND_CHMAP_RLC: snd_pcm_chmap_position = 14;
pub const SND_CHMAP_RRC: snd_pcm_chmap_position = 15;
pub const SND_CHMAP_FLW: snd_pcm_chmap_position = 16;
pub const SND_CHMAP_FRW: snd_pcm_chmap_position = 17;
pub const SND_CHMAP_FLH: snd_pcm_chmap_position = 18;
pub const SND_CHMAP_FCH: snd_pcm_chmap_position = 19;
pub const SND_CHMAP_FRH: snd_pcm_chmap_position = 20;
pub const SND_CHMAP_TC: snd_pcm_chmap_position = 21;
pub const SND_CHMAP_TFL: snd_pcm_chmap_position = 22;
pub const SND_CHMAP_TFR: snd_pcm_chmap_position = 23;
pub const SND_CHMAP_TFC: snd_pcm_chmap_position = 24;
pub const SND_CHMAP_TRL: snd_pcm_chmap_position = 25;
pub const SND_CHMAP_TRR: snd_pcm_chmap_position = 26;
pub const SND_CHMAP_TRC: snd_pcm_chmap_position = 27;
pub const SND_CHMAP_TFLC: snd_pcm_chmap_position = 28;
pub const SND_CHMAP_TFRC: snd_pcm_chmap_position = 29;
pub const SND_CHMAP_TSL: snd_pcm_chmap_position = 30;
pub const SND_CHMAP_TSR: snd_pcm_chmap_position = 31;
pub const SND_CHMAP_LLFE: snd_pcm_chmap_position = 32;
pub const SND_CHMAP_RLFE: snd_pcm_chmap_position = 33;
pub const SND_CHMAP_BC: snd_pcm_chmap_position = 34;
pub const SND_CHMAP_BLC: snd_pcm_chmap_position = 35;
pub const SND_CHMAP_BRC: snd_pcm_chmap_position = 36;
pub type snd_pcm_chmap_position = ::core::ffi::c_uint;
#[repr(C)]
#[derive(Debug)]
pub struct snd_pcm_chmap {
    pub channels: ::core::ffi::c_uint,
    pub pos: __IncompleteArrayField<::core::ffi::c_uint>,
}
pub type snd_pcm_chmap_t = snd_pcm_chmap;
#[repr(C)]
#[derive(Debug)]
pub struct snd_pcm_chmap_query {
    pub type_: snd_pcm_chmap_type,
    pub map: snd_pcm_chmap_t,
}
pub type snd_pcm_chmap_query_t = snd_pcm_chmap_query;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_rawmidi_info {
    _unused: [u8; 0],
}
pub type snd_rawmidi_info_t = _snd_rawmidi_info;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_rawmidi_status {
    _unused: [u8; 0],
}
pub type snd_rawmidi_status_t = _snd_rawmidi_status;
pub const SND_RAWMIDI_STREAM_OUTPUT: _snd_rawmidi_stream = 0;
pub const SND_RAWMIDI_STREAM_INPUT: _snd_rawmidi_stream = 1;
pub type _snd_rawmidi_stream = ::core::ffi::c_uint;
pub use self::_snd_rawmidi_stream as snd_rawmidi_stream_t;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_rawmidi {
    _unused: [u8; 0],
}
pub type snd_rawmidi_t = _snd_rawmidi;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_ctl_card_info {
    _unused: [u8; 0],
}
pub type snd_ctl_card_info_t = _snd_ctl_card_info;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_ctl_elem_id {
    _unused: [u8; 0],
}
pub type snd_ctl_elem_id_t = _snd_ctl_elem_id;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_ctl_elem_list {
    _unused: [u8; 0],
}
pub type snd_ctl_elem_list_t = _snd_ctl_elem_list;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_ctl_elem_info {
    _unused: [u8; 0],
}
pub type snd_ctl_elem_info_t = _snd_ctl_elem_info;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_ctl_elem_value {
    _unused: [u8; 0],
}
pub type snd_ctl_elem_value_t = _snd_ctl_elem_value;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_ctl_event {
    _unused: [u8; 0],
}
pub type snd_ctl_event_t = _snd_ctl_event;
pub const SND_CTL_ELEM_TYPE_NONE: _snd_ctl_elem_type = 0;
pub const SND_CTL_ELEM_TYPE_BOOLEAN: _snd_ctl_elem_type = 1;
pub const SND_CTL_ELEM_TYPE_INTEGER: _snd_ctl_elem_type = 2;
pub const SND_CTL_ELEM_TYPE_ENUMERATED: _snd_ctl_elem_type = 3;
pub const SND_CTL_ELEM_TYPE_BYTES: _snd_ctl_elem_type = 4;
pub const SND_CTL_ELEM_TYPE_IEC958: _snd_ctl_elem_type = 5;
pub const SND_CTL_ELEM_TYPE_INTEGER64: _snd_ctl_elem_type = 6;
pub type _snd_ctl_elem_type = ::core::ffi::c_uint;
pub use self::_snd_ctl_elem_type as snd_ctl_elem_type_t;
pub const SND_CTL_ELEM_IFACE_CARD: _snd_ctl_elem_iface = 0;
pub const SND_CTL_ELEM_IFACE_HWDEP: _snd_ctl_elem_iface = 1;
pub const SND_CTL_ELEM_IFACE_MIXER: _snd_ctl_elem_iface = 2;
pub const SND_CTL_ELEM_IFACE_PCM: _snd_ctl_elem_iface = 3;
pub const SND_CTL_ELEM_IFACE_RAWMIDI: _snd_ctl_elem_iface = 4;
pub const SND_CTL_ELEM_IFACE_TIMER: _snd_ctl_elem_iface = 5;
pub const SND_CTL_ELEM_IFACE_SEQUENCER: _snd_ctl_elem_iface = 6;
pub type _snd_ctl_elem_iface = ::core::ffi::c_uint;
pub use self::_snd_ctl_elem_iface as snd_ctl_elem_iface_t;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_ctl {
    _unused: [u8; 0],
}
pub type snd_ctl_t = _snd_ctl;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_hctl_elem {
    _unused: [u8; 0],
}
pub type snd_hctl_elem_t = _snd_hctl_elem;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_hctl {
    _unused: [u8; 0],
}
pub type snd_hctl_t = _snd_hctl;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_mixer {
    _unused: [u8; 0],
}
pub type snd_mixer_t = _snd_mixer;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_mixer_class {
    _unused: [u8; 0],
}
pub type snd_mixer_class_t = _snd_mixer_class;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_mixer_elem {
    _unused: [u8; 0],
}
pub type snd_mixer_elem_t = _snd_mixer_elem;
pub const SND_MIXER_ELEM_SIMPLE: _snd_mixer_elem_type = 0;
pub type _snd_mixer_elem_type = ::core::ffi::c_uint;
pub use self::_snd_mixer_elem_type as snd_mixer_elem_type_t;
pub const SND_MIXER_SCHN_UNKNOWN: _snd_mixer_selem_channel_id = -1;
pub const SND_MIXER_SCHN_FRONT_LEFT: _snd_mixer_selem_channel_id = 0;
pub const SND_MIXER_SCHN_FRONT_RIGHT: _snd_mixer_selem_channel_id = 1;
pub const SND_MIXER_SCHN_REAR_LEFT: _snd_mixer_selem_channel_id = 2;
pub const SND_MIXER_SCHN_REAR_RIGHT: _snd_mixer_selem_channel_id = 3;
pub const SND_MIXER_SCHN_FRONT_CENTER: _snd_mixer_selem_channel_id = 4;
pub const SND_MIXER_SCHN_WOOFER: _snd_mixer_selem_channel_id = 5;
pub const SND_MIXER_SCHN_SIDE_LEFT: _snd_mixer_selem_channel_id = 6;
pub const SND_MIXER_SCHN_SIDE_RIGHT: _snd_mixer_selem_channel_id = 7;
pub const SND_MIXER_SCHN_REAR_CENTER: _snd_mixer_selem_channel_id = 8;
pub const SND_MIXER_SCHN_LAST: _snd_mixer_selem_channel_id = 31;
pub type _snd_mixer_selem_channel_id = ::core::ffi::c_int;
pub use self::_snd_mixer_selem_channel_id as snd_mixer_selem_channel_id_t;
pub type snd_mixer_selem_regopt_abstract = ::core::ffi::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct snd_mixer_selem_regopt {
    pub ver: ::core::ffi::c_int,
    pub abstract_: snd_mixer_selem_regopt_abstract,
    pub device: *const ::core::ffi::c_char,
    pub playback_pcm: *mut snd_pcm_t,
    pub capture_pcm: *mut snd_pcm_t,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_mixer_selem_id {
    _unused: [u8; 0],
}
pub type snd_mixer_selem_id_t = _snd_mixer_selem_id;
pub type snd_seq_event_type_t = ::core::ffi::c_uchar;
pub const SND_SEQ_EVENT_SYSTEM: snd_seq_event_type = 0;
pub const SND_SEQ_EVENT_RESULT: snd_seq_event_type = 1;
pub const SND_SEQ_EVENT_NOTE: snd_seq_event_type = 5;
pub const SND_SEQ_EVENT_NOTEON: snd_seq_event_type = 6;
pub const SND_SEQ_EVENT_NOTEOFF: snd_seq_event_type = 7;
pub const SND_SEQ_EVENT_KEYPRESS: snd_seq_event_type = 8;
pub const SND_SEQ_EVENT_CONTROLLER: snd_seq_event_type = 10;
pub const SND_SEQ_EVENT_PGMCHANGE: snd_seq_event_type = 11;
pub const SND_SEQ_EVENT_CHANPRESS: snd_seq_event_type = 12;
pub const SND_SEQ_EVENT_PITCHBEND: snd_seq_event_type = 13;
pub const SND_SEQ_EVENT_CONTROL14: snd_seq_event_type = 14;
pub const SND_SEQ_EVENT_NONREGPARAM: snd_seq_event_type = 15;
pub const SND_SEQ_EVENT_REGPARAM: snd_seq_event_type = 16;
pub const SND_SEQ_EVENT_SONGPOS: snd_seq_event_type = 20;
pub const SND_SEQ_EVENT_SONGSEL: snd_seq_event_type = 21;
pub const SND_SEQ_EVENT_QFRAME: snd_seq_event_type = 22;
pub const SND_SEQ_EVENT_TIMESIGN: snd_seq_event_type = 23;
pub const SND_SEQ_EVENT_KEYSIGN: snd_seq_event_type = 24;
pub const SND_SEQ_EVENT_START: snd_seq_event_type = 30;
pub const SND_SEQ_EVENT_CONTINUE: snd_seq_event_type = 31;
pub const SND_SEQ_EVENT_STOP: snd_seq_event_type = 32;
pub const SND_SEQ_EVENT_SETPOS_TICK: snd_seq_event_type = 33;
pub const SND_SEQ_EVENT_SETPOS_TIME: snd_seq_event_type = 34;
pub const SND_SEQ_EVENT_TEMPO: snd_seq_event_type = 35;
pub const SND_SEQ_EVENT_CLOCK: snd_seq_event_type = 36;
pub const SND_SEQ_EVENT_TICK: snd_seq_event_type = 37;
pub const SND_SEQ_EVENT_QUEUE_SKEW: snd_seq_event_type = 38;
pub const SND_SEQ_EVENT_SYNC_POS: snd_seq_event_type = 39;
pub const SND_SEQ_EVENT_TUNE_REQUEST: snd_seq_event_type = 40;
pub const SND_SEQ_EVENT_RESET: snd_seq_event_type = 41;
pub const SND_SEQ_EVENT_SENSING: snd_seq_event_type = 42;
pub const SND_SEQ_EVENT_ECHO: snd_seq_event_type = 50;
pub const SND_SEQ_EVENT_OSS: snd_seq_event_type = 51;
pub const SND_SEQ_EVENT_CLIENT_START: snd_seq_event_type = 60;
pub const SND_SEQ_EVENT_CLIENT_EXIT: snd_seq_event_type = 61;
pub const SND_SEQ_EVENT_CLIENT_CHANGE: snd_seq_event_type = 62;
pub const SND_SEQ_EVENT_PORT_START: snd_seq_event_type = 63;
pub const SND_SEQ_EVENT_PORT_EXIT: snd_seq_event_type = 64;
pub const SND_SEQ_EVENT_PORT_CHANGE: snd_seq_event_type = 65;
pub const SND_SEQ_EVENT_PORT_SUBSCRIBED: snd_seq_event_type = 66;
pub const SND_SEQ_EVENT_PORT_UNSUBSCRIBED: snd_seq_event_type = 67;
pub const SND_SEQ_EVENT_USR0: snd_seq_event_type = 90;
pub const SND_SEQ_EVENT_USR1: snd_seq_event_type = 91;
pub const SND_SEQ_EVENT_USR2: snd_seq_event_type = 92;
pub const SND_SEQ_EVENT_USR3: snd_seq_event_type = 93;
pub const SND_SEQ_EVENT_USR4: snd_seq_event_type = 94;
pub const SND_SEQ_EVENT_USR5: snd_seq_event_type = 95;
pub const SND_SEQ_EVENT_USR6: snd_seq_event_type = 96;
pub const SND_SEQ_EVENT_USR7: snd_seq_event_type = 97;
pub const SND_SEQ_EVENT_USR8: snd_seq_event_type = 98;
pub const SND_SEQ_EVENT_USR9: snd_seq_event_type = 99;
pub const SND_SEQ_EVENT_SYSEX: snd_seq_event_type = 130;
pub const SND_SEQ_EVENT_BOUNCE: snd_seq_event_type = 131;
pub const SND_SEQ_EVENT_USR_VAR0: snd_seq_event_type = 135;
pub const SND_SEQ_EVENT_USR_VAR1: snd_seq_event_type = 136;
pub const SND_SEQ_EVENT_USR_VAR2: snd_seq_event_type = 137;
pub const SND_SEQ_EVENT_USR_VAR3: snd_seq_event_type = 138;
pub const SND_SEQ_EVENT_USR_VAR4: snd_seq_event_type = 139;
pub const SND_SEQ_EVENT_NONE: snd_seq_event_type = 255;
pub type snd_seq_event_type = ::core::ffi::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct snd_seq_addr {
    pub client: ::core::ffi::c_uchar,
    pub port: ::core::ffi::c_uchar,
}
pub type snd_seq_addr_t = snd_seq_addr;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct snd_seq_connect {
    pub sender: snd_seq_addr_t,
    pub dest: snd_seq_addr_t,
}
pub type snd_seq_connect_t = snd_seq_connect;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct snd_seq_real_time {
    pub tv_sec: ::core::ffi::c_uint,
    pub tv_nsec: ::core::ffi::c_uint,
}
pub type snd_seq_real_time_t = snd_seq_real_time;
pub type snd_seq_tick_time_t = ::core::ffi::c_uint;
#[repr(C)]
#[derive(Copy, Clone)]
pub union snd_seq_timestamp {
    pub tick: snd_seq_tick_time_t,
    pub time: snd_seq_real_time,
}
pub type snd_seq_timestamp_t = snd_seq_timestamp;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct snd_seq_ev_note {
    pub channel: ::core::ffi::c_uchar,
    pub note: ::core::ffi::c_uchar,
    pub velocity: ::core::ffi::c_uchar,
    pub off_velocity: ::core::ffi::c_uchar,
    pub duration: ::core::ffi::c_uint,
}
pub type snd_seq_ev_note_t = snd_seq_ev_note;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct snd_seq_ev_ctrl {
    pub channel: ::core::ffi::c_uchar,
    pub unused: [::core::ffi::c_uchar; 3usize],
    pub param: ::core::ffi::c_uint,
    pub value: ::core::ffi::c_int,
}
pub type snd_seq_ev_ctrl_t = snd_seq_ev_ctrl;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct snd_seq_ev_raw8 {
    pub d: [::core::ffi::c_uchar; 12usize],
}
pub type snd_seq_ev_raw8_t = snd_seq_ev_raw8;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct snd_seq_ev_raw32 {
    pub d: [::core::ffi::c_uint; 3usize],
}
pub type snd_seq_ev_raw32_t = snd_seq_ev_raw32;
#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct snd_seq_ev_ext {
    pub len: ::core::ffi::c_uint,
    pub ptr: *mut ::core::ffi::c_void,
}
pub type snd_seq_ev_ext_t = snd_seq_ev_ext;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct snd_seq_result {
    pub event: ::core::ffi::c_int,
    pub result: ::core::ffi::c_int,
}
pub type snd_seq_result_t = snd_seq_result;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct snd_seq_queue_skew {
    pub value: ::core::ffi::c_uint,
    pub base: ::core::ffi::c_uint,
}
pub type snd_seq_queue_skew_t = snd_seq_queue_skew;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct snd_seq_ev_queue_control {
    pub queue: ::core::ffi::c_uchar,
    pub unused: [::core::ffi::c_uchar; 3usize],
    pub param: snd_seq_ev_queue_control__bindgen_ty_1,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union snd_seq_ev_queue_control__bindgen_ty_1 {
    pub value: ::core::ffi::c_int,
    pub time: snd_seq_timestamp_t,
    pub position: ::core::ffi::c_uint,
    pub skew: snd_seq_queue_skew_t,
    pub d32: [::core::ffi::c_uint; 2usize],
    pub d8: [::core::ffi::c_uchar; 8usize],
}
pub type snd_seq_ev_queue_control_t = snd_seq_ev_queue_control;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct snd_seq_ev_ump_notify {
    pub client: ::core::ffi::c_uchar,
    pub block: ::core::ffi::c_uchar,
}
pub type snd_seq_ev_ump_notify_t = snd_seq_ev_ump_notify;
#[repr(C)]
#[derive(Copy, Clone)]
pub union snd_seq_event_data {
    pub note: snd_seq_ev_note_t,
    pub control: snd_seq_ev_ctrl_t,
    pub raw8: snd_seq_ev_raw8_t,
    pub raw32: snd_seq_ev_raw32_t,
    pub ext: snd_seq_ev_ext_t,
    pub queue: snd_seq_ev_queue_control_t,
    pub time: snd_seq_timestamp_t,
    pub addr: snd_seq_addr_t,
    pub connect: snd_seq_connect_t,
    pub result: snd_seq_result_t,
    pub ump_notify: snd_seq_ev_ump_notify_t,
}
pub type snd_seq_event_data_t = snd_seq_event_data;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct snd_seq_event {
    pub type_: snd_seq_event_type_t,
    pub flags: ::core::ffi::c_uchar,
    pub tag: ::core::ffi::c_uchar,
    pub queue: ::core::ffi::c_uchar,
    pub time: snd_seq_timestamp_t,
    pub source: snd_seq_addr_t,
    pub dest: snd_seq_addr_t,
    pub data: snd_seq_event_data_t,
}
pub type snd_seq_event_t = snd_seq_event;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_seq {
    _unused: [u8; 0],
}
pub type snd_seq_t = _snd_seq;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_seq_client_info {
    _unused: [u8; 0],
}
pub type snd_seq_client_info_t = _snd_seq_client_info;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_seq_port_info {
    _unused: [u8; 0],
}
pub type snd_seq_port_info_t = _snd_seq_port_info;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_seq_port_subscribe {
    _unused: [u8; 0],
}
pub type snd_seq_port_subscribe_t = _snd_seq_port_subscribe;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_seq_query_subscribe {
    _unused: [u8; 0],
}
pub type snd_seq_query_subscribe_t = _snd_seq_query_subscribe;
pub const SND_SEQ_QUERY_SUBS_READ: snd_seq_query_subs_type_t = 0;
pub const SND_SEQ_QUERY_SUBS_WRITE: snd_seq_query_subs_type_t = 1;
pub type snd_seq_query_subs_type_t = ::core::ffi::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_seq_queue_status {
    _unused: [u8; 0],
}
pub type snd_seq_queue_status_t = _snd_seq_queue_status;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_seq_queue_tempo {
    _unused: [u8; 0],
}
pub type snd_seq_queue_tempo_t = _snd_seq_queue_tempo;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _snd_seq_remove_events {
    _unused: [u8; 0],
}
pub type snd_seq_remove_events_t = _snd_seq_remove_events;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct snd_midi_event {
    _unused: [u8; 0],
}
pub type snd_midi_event_t = snd_midi_event;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct __va_list_tag {
    pub gp_offset: ::core::ffi::c_uint,
    pub fp_offset: ::core::ffi::c_uint,
    pub overflow_arg_area: *mut ::core::ffi::c_void,
    pub reg_save_area: *mut ::core::ffi::c_void,
}

pub unsafe fn snd_output_buffer_open(outputp: *mut *mut snd_output_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(outputp: *mut *mut snd_output_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_output_buffer_open\0")) {
        Some(function) => function(outputp),
        None => -1,
    }
}
pub unsafe fn snd_output_buffer_string(
        output: *mut snd_output_t,
        buf: *mut *mut ::core::ffi::c_char,
    ) -> usize {
    type F = unsafe extern "C" fn(
        output: *mut snd_output_t,
        buf: *mut *mut ::core::ffi::c_char,
    ) -> usize;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_output_buffer_string\0")) {
        Some(function) => function(output, buf),
        None => 0,
    }
}
pub unsafe fn snd_output_close(output: *mut snd_output_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(output: *mut snd_output_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_output_close\0")) {
        Some(function) => function(output),
        None => -1,
    }
}
pub unsafe fn snd_output_vprintf(
        output: *mut snd_output_t,
        format: *const ::core::ffi::c_char,
        args: *mut __va_list_tag,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        output: *mut snd_output_t,
        format: *const ::core::ffi::c_char,
        args: *mut __va_list_tag,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_output_vprintf\0")) {
        Some(function) => function(output, format, args),
        None => -1,
    }
}
pub unsafe fn snd_output_puts(
        output: *mut snd_output_t,
        str_: *const ::core::ffi::c_char,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        output: *mut snd_output_t,
        str_: *const ::core::ffi::c_char,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_output_puts\0")) {
        Some(function) => function(output, str_),
        None => -1,
    }
}
pub unsafe fn snd_output_putc(output: *mut snd_output_t, c: ::core::ffi::c_int) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(output: *mut snd_output_t, c: ::core::ffi::c_int) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_output_putc\0")) {
        Some(function) => function(output, c),
        None => -1,
    }
}
pub unsafe fn snd_lib_error_set_local(func: snd_local_error_handler_t) -> snd_local_error_handler_t {
    type F = unsafe extern "C" fn(func: snd_local_error_handler_t) -> snd_local_error_handler_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_lib_error_set_local\0")) {
        Some(function) => function(func),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_config_save(config: *mut snd_config_t, out: *mut snd_output_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(config: *mut snd_config_t, out: *mut snd_output_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_config_save\0")) {
        Some(function) => function(config, out),
        None => -1,
    }
}
pub unsafe fn snd_config_update() -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn() -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_config_update\0")) {
        Some(function) => function(),
        None => -1,
    }
}
pub unsafe fn snd_config_update_free_global() -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn() -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_config_update_free_global\0")) {
        Some(function) => function(),
        None => -1,
    }
}
pub unsafe fn snd_config_update_ref(top: *mut *mut snd_config_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(top: *mut *mut snd_config_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_config_update_ref\0")) {
        Some(function) => function(top),
        None => -1,
    }
}
pub unsafe fn snd_config_unref(top: *mut snd_config_t) -> () {
    type F = unsafe extern "C" fn(top: *mut snd_config_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_config_unref\0")) {
        Some(function) => function(top),
        None => (),
    }
}
pub unsafe fn snd_pcm_open(
        pcm: *mut *mut snd_pcm_t,
        name: *const ::core::ffi::c_char,
        stream: snd_pcm_stream_t,
        mode: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut *mut snd_pcm_t,
        name: *const ::core::ffi::c_char,
        stream: snd_pcm_stream_t,
        mode: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_open\0")) {
        Some(function) => function(pcm, name, stream, mode),
        None => -1,
    }
}
pub unsafe fn snd_pcm_close(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_close\0")) {
        Some(function) => function(pcm),
        None => -1,
    }
}
pub unsafe fn snd_pcm_poll_descriptors_count(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_poll_descriptors_count\0")) {
        Some(function) => function(pcm),
        None => -1,
    }
}
pub unsafe fn snd_pcm_poll_descriptors(
        pcm: *mut snd_pcm_t,
        pfds: *mut pollfd,
        space: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        pfds: *mut pollfd,
        space: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_poll_descriptors\0")) {
        Some(function) => function(pcm, pfds, space),
        None => -1,
    }
}
pub unsafe fn snd_pcm_poll_descriptors_revents(
        pcm: *mut snd_pcm_t,
        pfds: *mut pollfd,
        nfds: ::core::ffi::c_uint,
        revents: *mut ::core::ffi::c_ushort,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        pfds: *mut pollfd,
        nfds: ::core::ffi::c_uint,
        revents: *mut ::core::ffi::c_ushort,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_poll_descriptors_revents\0")) {
        Some(function) => function(pcm, pfds, nfds, revents),
        None => -1,
    }
}
pub unsafe fn snd_pcm_info(pcm: *mut snd_pcm_t, info: *mut snd_pcm_info_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t, info: *mut snd_pcm_info_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_info\0")) {
        Some(function) => function(pcm, info),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_current(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_current\0")) {
        Some(function) => function(pcm, params),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params\0")) {
        Some(function) => function(pcm, params),
        None => -1,
    }
}
pub unsafe fn snd_pcm_sw_params_current(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_sw_params_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_sw_params_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_sw_params_current\0")) {
        Some(function) => function(pcm, params),
        None => -1,
    }
}
pub unsafe fn snd_pcm_sw_params(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_sw_params_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_sw_params_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_sw_params\0")) {
        Some(function) => function(pcm, params),
        None => -1,
    }
}
pub unsafe fn snd_pcm_prepare(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_prepare\0")) {
        Some(function) => function(pcm),
        None => -1,
    }
}
pub unsafe fn snd_pcm_reset(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_reset\0")) {
        Some(function) => function(pcm),
        None => -1,
    }
}
pub unsafe fn snd_pcm_status(pcm: *mut snd_pcm_t, status: *mut snd_pcm_status_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t, status: *mut snd_pcm_status_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_status\0")) {
        Some(function) => function(pcm, status),
        None => -1,
    }
}
pub unsafe fn snd_pcm_start(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_start\0")) {
        Some(function) => function(pcm),
        None => -1,
    }
}
pub unsafe fn snd_pcm_drop(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_drop\0")) {
        Some(function) => function(pcm),
        None => -1,
    }
}
pub unsafe fn snd_pcm_drain(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_drain\0")) {
        Some(function) => function(pcm),
        None => -1,
    }
}
pub unsafe fn snd_pcm_pause(pcm: *mut snd_pcm_t, enable: ::core::ffi::c_int) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t, enable: ::core::ffi::c_int) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_pause\0")) {
        Some(function) => function(pcm, enable),
        None => -1,
    }
}
pub unsafe fn snd_pcm_state(pcm: *mut snd_pcm_t) -> snd_pcm_state_t {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t) -> snd_pcm_state_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_state\0")) {
        Some(function) => function(pcm),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_pcm_delay(pcm: *mut snd_pcm_t, delayp: *mut snd_pcm_sframes_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t, delayp: *mut snd_pcm_sframes_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_delay\0")) {
        Some(function) => function(pcm, delayp),
        None => -1,
    }
}
pub unsafe fn snd_pcm_resume(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_resume\0")) {
        Some(function) => function(pcm),
        None => -1,
    }
}
pub unsafe fn snd_pcm_avail(pcm: *mut snd_pcm_t) -> snd_pcm_sframes_t {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t) -> snd_pcm_sframes_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_avail\0")) {
        Some(function) => function(pcm),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_pcm_avail_update(pcm: *mut snd_pcm_t) -> snd_pcm_sframes_t {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t) -> snd_pcm_sframes_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_avail_update\0")) {
        Some(function) => function(pcm),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_pcm_avail_delay(
        pcm: *mut snd_pcm_t,
        availp: *mut snd_pcm_sframes_t,
        delayp: *mut snd_pcm_sframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        availp: *mut snd_pcm_sframes_t,
        delayp: *mut snd_pcm_sframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_avail_delay\0")) {
        Some(function) => function(pcm, availp, delayp),
        None => -1,
    }
}
pub unsafe fn snd_pcm_writei(
        pcm: *mut snd_pcm_t,
        buffer: *const ::core::ffi::c_void,
        size: snd_pcm_uframes_t,
    ) -> snd_pcm_sframes_t {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        buffer: *const ::core::ffi::c_void,
        size: snd_pcm_uframes_t,
    ) -> snd_pcm_sframes_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_writei\0")) {
        Some(function) => function(pcm, buffer, size),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_pcm_readi(
        pcm: *mut snd_pcm_t,
        buffer: *mut ::core::ffi::c_void,
        size: snd_pcm_uframes_t,
    ) -> snd_pcm_sframes_t {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        buffer: *mut ::core::ffi::c_void,
        size: snd_pcm_uframes_t,
    ) -> snd_pcm_sframes_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_readi\0")) {
        Some(function) => function(pcm, buffer, size),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_pcm_writen(
        pcm: *mut snd_pcm_t,
        bufs: *mut *mut ::core::ffi::c_void,
        size: snd_pcm_uframes_t,
    ) -> snd_pcm_sframes_t {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        bufs: *mut *mut ::core::ffi::c_void,
        size: snd_pcm_uframes_t,
    ) -> snd_pcm_sframes_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_writen\0")) {
        Some(function) => function(pcm, bufs, size),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_pcm_readn(
        pcm: *mut snd_pcm_t,
        bufs: *mut *mut ::core::ffi::c_void,
        size: snd_pcm_uframes_t,
    ) -> snd_pcm_sframes_t {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        bufs: *mut *mut ::core::ffi::c_void,
        size: snd_pcm_uframes_t,
    ) -> snd_pcm_sframes_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_readn\0")) {
        Some(function) => function(pcm, bufs, size),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_pcm_wait(pcm: *mut snd_pcm_t, timeout: ::core::ffi::c_int) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t, timeout: ::core::ffi::c_int) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_wait\0")) {
        Some(function) => function(pcm, timeout),
        None => -1,
    }
}
pub unsafe fn snd_pcm_link(pcm1: *mut snd_pcm_t, pcm2: *mut snd_pcm_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm1: *mut snd_pcm_t, pcm2: *mut snd_pcm_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_link\0")) {
        Some(function) => function(pcm1, pcm2),
        None => -1,
    }
}
pub unsafe fn snd_pcm_unlink(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_unlink\0")) {
        Some(function) => function(pcm),
        None => -1,
    }
}
pub unsafe fn snd_pcm_query_chmaps(pcm: *mut snd_pcm_t) -> *mut *mut snd_pcm_chmap_query_t {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t) -> *mut *mut snd_pcm_chmap_query_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_query_chmaps\0")) {
        Some(function) => function(pcm),
        None => core::ptr::null_mut(),
    }
}
pub unsafe fn snd_pcm_free_chmaps(maps: *mut *mut snd_pcm_chmap_query_t) -> () {
    type F = unsafe extern "C" fn(maps: *mut *mut snd_pcm_chmap_query_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_free_chmaps\0")) {
        Some(function) => function(maps),
        None => (),
    }
}
pub unsafe fn snd_pcm_get_chmap(pcm: *mut snd_pcm_t) -> *mut snd_pcm_chmap_t {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t) -> *mut snd_pcm_chmap_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_get_chmap\0")) {
        Some(function) => function(pcm),
        None => core::ptr::null_mut(),
    }
}
pub unsafe fn snd_pcm_set_chmap(
        pcm: *mut snd_pcm_t,
        map: *const snd_pcm_chmap_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        map: *const snd_pcm_chmap_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_set_chmap\0")) {
        Some(function) => function(pcm, map),
        None => -1,
    }
}
pub unsafe fn snd_pcm_chmap_long_name(val: snd_pcm_chmap_position) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(val: snd_pcm_chmap_position) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_chmap_long_name\0")) {
        Some(function) => function(val),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_pcm_chmap_print(
        map: *const snd_pcm_chmap_t,
        maxlen: usize,
        buf: *mut ::core::ffi::c_char,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        map: *const snd_pcm_chmap_t,
        maxlen: usize,
        buf: *mut ::core::ffi::c_char,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_chmap_print\0")) {
        Some(function) => function(map, maxlen, buf),
        None => -1,
    }
}
pub unsafe fn snd_pcm_recover(
        pcm: *mut snd_pcm_t,
        err: ::core::ffi::c_int,
        silent: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        err: ::core::ffi::c_int,
        silent: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_recover\0")) {
        Some(function) => function(pcm, err, silent),
        None => -1,
    }
}
pub unsafe fn snd_pcm_get_params(
        pcm: *mut snd_pcm_t,
        buffer_size: *mut snd_pcm_uframes_t,
        period_size: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        buffer_size: *mut snd_pcm_uframes_t,
        period_size: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_get_params\0")) {
        Some(function) => function(pcm, buffer_size, period_size),
        None => -1,
    }
}
pub unsafe fn snd_pcm_info_malloc(ptr: *mut *mut snd_pcm_info_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ptr: *mut *mut snd_pcm_info_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_info_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_pcm_info_free(obj: *mut snd_pcm_info_t) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_pcm_info_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_info_free\0")) {
        Some(function) => function(obj),
        None => (),
    }
}
pub unsafe fn snd_pcm_info_get_device(obj: *const snd_pcm_info_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(obj: *const snd_pcm_info_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_info_get_device\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_pcm_info_get_subdevice(obj: *const snd_pcm_info_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(obj: *const snd_pcm_info_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_info_get_subdevice\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_pcm_info_get_stream(obj: *const snd_pcm_info_t) -> snd_pcm_stream_t {
    type F = unsafe extern "C" fn(obj: *const snd_pcm_info_t) -> snd_pcm_stream_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_info_get_stream\0")) {
        Some(function) => function(obj),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_pcm_info_get_card(obj: *const snd_pcm_info_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(obj: *const snd_pcm_info_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_info_get_card\0")) {
        Some(function) => function(obj),
        None => -1,
    }
}
pub unsafe fn snd_pcm_info_get_id(obj: *const snd_pcm_info_t) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(obj: *const snd_pcm_info_t) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_info_get_id\0")) {
        Some(function) => function(obj),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_pcm_info_get_name(obj: *const snd_pcm_info_t) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(obj: *const snd_pcm_info_t) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_info_get_name\0")) {
        Some(function) => function(obj),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_pcm_info_get_subdevice_name(
        obj: *const snd_pcm_info_t,
    ) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(
        obj: *const snd_pcm_info_t,
    ) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_info_get_subdevice_name\0")) {
        Some(function) => function(obj),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_pcm_info_get_subdevices_count(obj: *const snd_pcm_info_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(obj: *const snd_pcm_info_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_info_get_subdevices_count\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_pcm_info_get_subdevices_avail(obj: *const snd_pcm_info_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(obj: *const snd_pcm_info_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_info_get_subdevices_avail\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_pcm_info_set_device(obj: *mut snd_pcm_info_t, val: ::core::ffi::c_uint) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_pcm_info_t, val: ::core::ffi::c_uint) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_info_set_device\0")) {
        Some(function) => function(obj, val),
        None => (),
    }
}
pub unsafe fn snd_pcm_info_set_subdevice(obj: *mut snd_pcm_info_t, val: ::core::ffi::c_uint) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_pcm_info_t, val: ::core::ffi::c_uint) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_info_set_subdevice\0")) {
        Some(function) => function(obj, val),
        None => (),
    }
}
pub unsafe fn snd_pcm_info_set_stream(obj: *mut snd_pcm_info_t, val: snd_pcm_stream_t) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_pcm_info_t, val: snd_pcm_stream_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_info_set_stream\0")) {
        Some(function) => function(obj, val),
        None => (),
    }
}
pub unsafe fn snd_pcm_hw_params_any(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_any\0")) {
        Some(function) => function(pcm, params),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_can_pause(params: *const snd_pcm_hw_params_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(params: *const snd_pcm_hw_params_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_can_pause\0")) {
        Some(function) => function(params),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_can_resume(params: *const snd_pcm_hw_params_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(params: *const snd_pcm_hw_params_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_can_resume\0")) {
        Some(function) => function(params),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_supports_audio_ts_type(
        params: *const snd_pcm_hw_params_t,
        type_: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        type_: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_supports_audio_ts_type\0")) {
        Some(function) => function(params, type_),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_malloc(ptr: *mut *mut snd_pcm_hw_params_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ptr: *mut *mut snd_pcm_hw_params_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_free(obj: *mut snd_pcm_hw_params_t) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_pcm_hw_params_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_free\0")) {
        Some(function) => function(obj),
        None => (),
    }
}
pub unsafe fn snd_pcm_hw_params_copy(dst: *mut snd_pcm_hw_params_t, src: *const snd_pcm_hw_params_t) -> () {
    type F = unsafe extern "C" fn(dst: *mut snd_pcm_hw_params_t, src: *const snd_pcm_hw_params_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_copy\0")) {
        Some(function) => function(dst, src),
        None => (),
    }
}
pub unsafe fn snd_pcm_hw_params_get_access(
        params: *const snd_pcm_hw_params_t,
        _access: *mut snd_pcm_access_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        _access: *mut snd_pcm_access_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_access\0")) {
        Some(function) => function(params, _access),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_test_access(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        _access: snd_pcm_access_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        _access: snd_pcm_access_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_test_access\0")) {
        Some(function) => function(pcm, params, _access),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_access(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        _access: snd_pcm_access_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        _access: snd_pcm_access_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_access\0")) {
        Some(function) => function(pcm, params, _access),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_format(
        params: *const snd_pcm_hw_params_t,
        val: *mut snd_pcm_format_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut snd_pcm_format_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_format\0")) {
        Some(function) => function(params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_test_format(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: snd_pcm_format_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: snd_pcm_format_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_test_format\0")) {
        Some(function) => function(pcm, params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_format(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: snd_pcm_format_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: snd_pcm_format_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_format\0")) {
        Some(function) => function(pcm, params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_channels(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_channels\0")) {
        Some(function) => function(params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_channels_min(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_channels_min\0")) {
        Some(function) => function(params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_channels_max(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_channels_max\0")) {
        Some(function) => function(params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_test_channels(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_test_channels\0")) {
        Some(function) => function(pcm, params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_channels(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_channels\0")) {
        Some(function) => function(pcm, params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_channels_near(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_channels_near\0")) {
        Some(function) => function(pcm, params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_rate(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_rate\0")) {
        Some(function) => function(params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_rate_min(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_rate_min\0")) {
        Some(function) => function(params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_rate_max(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_rate_max\0")) {
        Some(function) => function(params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_test_rate(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: ::core::ffi::c_uint,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: ::core::ffi::c_uint,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_test_rate\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_rate(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: ::core::ffi::c_uint,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: ::core::ffi::c_uint,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_rate\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_rate_near(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_rate_near\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_rate_resample(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_rate_resample\0")) {
        Some(function) => function(pcm, params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_rate_resample(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_rate_resample\0")) {
        Some(function) => function(pcm, params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_period_time(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_period_time\0")) {
        Some(function) => function(params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_period_time_min(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_period_time_min\0")) {
        Some(function) => function(params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_period_time_max(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_period_time_max\0")) {
        Some(function) => function(params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_period_time(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: ::core::ffi::c_uint,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: ::core::ffi::c_uint,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_period_time\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_period_time_min(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_period_time_min\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_period_time_max(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_period_time_max\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_period_time_near(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_period_time_near\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_period_size(
        params: *const snd_pcm_hw_params_t,
        frames: *mut snd_pcm_uframes_t,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        frames: *mut snd_pcm_uframes_t,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_period_size\0")) {
        Some(function) => function(params, frames, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_period_size_min(
        params: *const snd_pcm_hw_params_t,
        frames: *mut snd_pcm_uframes_t,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        frames: *mut snd_pcm_uframes_t,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_period_size_min\0")) {
        Some(function) => function(params, frames, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_period_size_max(
        params: *const snd_pcm_hw_params_t,
        frames: *mut snd_pcm_uframes_t,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        frames: *mut snd_pcm_uframes_t,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_period_size_max\0")) {
        Some(function) => function(params, frames, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_period_size(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: snd_pcm_uframes_t,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: snd_pcm_uframes_t,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_period_size\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_period_size_min(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_period_size_min\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_period_size_max(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_period_size_max\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_period_size_near(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_period_size_near\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_periods(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_periods\0")) {
        Some(function) => function(params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_periods_min(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_periods_min\0")) {
        Some(function) => function(params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_periods_max(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_periods_max\0")) {
        Some(function) => function(params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_periods(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: ::core::ffi::c_uint,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: ::core::ffi::c_uint,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_periods\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_periods_min(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_periods_min\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_periods_max(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_periods_max\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_periods_near(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_periods_near\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_buffer_time(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_buffer_time\0")) {
        Some(function) => function(params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_buffer_time_min(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_buffer_time_min\0")) {
        Some(function) => function(params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_buffer_time_max(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_buffer_time_max\0")) {
        Some(function) => function(params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_buffer_time(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: ::core::ffi::c_uint,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: ::core::ffi::c_uint,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_buffer_time\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_buffer_time_min(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_buffer_time_min\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_buffer_time_max(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_buffer_time_max\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_buffer_time_near(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut ::core::ffi::c_uint,
        dir: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_buffer_time_near\0")) {
        Some(function) => function(pcm, params, val, dir),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_buffer_size(
        params: *const snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_buffer_size\0")) {
        Some(function) => function(params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_buffer_size_min(
        params: *const snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_buffer_size_min\0")) {
        Some(function) => function(params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_get_buffer_size_max(
        params: *const snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_get_buffer_size_max\0")) {
        Some(function) => function(params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_buffer_size(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_buffer_size\0")) {
        Some(function) => function(pcm, params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_buffer_size_min(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_buffer_size_min\0")) {
        Some(function) => function(pcm, params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_buffer_size_max(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_buffer_size_max\0")) {
        Some(function) => function(pcm, params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_set_buffer_size_near(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_hw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_set_buffer_size_near\0")) {
        Some(function) => function(pcm, params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_sw_params_malloc(ptr: *mut *mut snd_pcm_sw_params_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ptr: *mut *mut snd_pcm_sw_params_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_sw_params_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_pcm_sw_params_free(obj: *mut snd_pcm_sw_params_t) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_pcm_sw_params_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_sw_params_free\0")) {
        Some(function) => function(obj),
        None => (),
    }
}
pub unsafe fn snd_pcm_sw_params_get_boundary(
        params: *const snd_pcm_sw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_sw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_sw_params_get_boundary\0")) {
        Some(function) => function(params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_sw_params_set_tstamp_mode(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_sw_params_t,
        val: snd_pcm_tstamp_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_sw_params_t,
        val: snd_pcm_tstamp_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_sw_params_set_tstamp_mode\0")) {
        Some(function) => function(pcm, params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_sw_params_get_tstamp_mode(
        params: *const snd_pcm_sw_params_t,
        val: *mut snd_pcm_tstamp_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_sw_params_t,
        val: *mut snd_pcm_tstamp_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_sw_params_get_tstamp_mode\0")) {
        Some(function) => function(params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_sw_params_set_tstamp_type(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_sw_params_t,
        val: snd_pcm_tstamp_type_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_sw_params_t,
        val: snd_pcm_tstamp_type_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_sw_params_set_tstamp_type\0")) {
        Some(function) => function(pcm, params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_sw_params_get_tstamp_type(
        params: *const snd_pcm_sw_params_t,
        val: *mut snd_pcm_tstamp_type_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_sw_params_t,
        val: *mut snd_pcm_tstamp_type_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_sw_params_get_tstamp_type\0")) {
        Some(function) => function(params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_sw_params_set_avail_min(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_sw_params_t,
        val: snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_sw_params_t,
        val: snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_sw_params_set_avail_min\0")) {
        Some(function) => function(pcm, params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_sw_params_get_avail_min(
        params: *const snd_pcm_sw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_sw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_sw_params_get_avail_min\0")) {
        Some(function) => function(params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_sw_params_set_start_threshold(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_sw_params_t,
        val: snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_sw_params_t,
        val: snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_sw_params_set_start_threshold\0")) {
        Some(function) => function(pcm, params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_sw_params_get_start_threshold(
        paramsm: *const snd_pcm_sw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        paramsm: *const snd_pcm_sw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_sw_params_get_start_threshold\0")) {
        Some(function) => function(paramsm, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_sw_params_set_stop_threshold(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_sw_params_t,
        val: snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        params: *mut snd_pcm_sw_params_t,
        val: snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_sw_params_set_stop_threshold\0")) {
        Some(function) => function(pcm, params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_sw_params_get_stop_threshold(
        params: *const snd_pcm_sw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *const snd_pcm_sw_params_t,
        val: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_sw_params_get_stop_threshold\0")) {
        Some(function) => function(params, val),
        None => -1,
    }
}
pub unsafe fn snd_pcm_status_sizeof() -> usize {
    type F = unsafe extern "C" fn() -> usize;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_status_sizeof\0")) {
        Some(function) => function(),
        None => 0,
    }
}
pub unsafe fn snd_pcm_status_get_state(obj: *const snd_pcm_status_t) -> snd_pcm_state_t {
    type F = unsafe extern "C" fn(obj: *const snd_pcm_status_t) -> snd_pcm_state_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_status_get_state\0")) {
        Some(function) => function(obj),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_pcm_status_get_trigger_htstamp(
        obj: *const snd_pcm_status_t,
        ptr: *mut snd_htimestamp_t,
    ) -> () {
    type F = unsafe extern "C" fn(
        obj: *const snd_pcm_status_t,
        ptr: *mut snd_htimestamp_t,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_status_get_trigger_htstamp\0")) {
        Some(function) => function(obj, ptr),
        None => (),
    }
}
pub unsafe fn snd_pcm_status_get_htstamp(obj: *const snd_pcm_status_t, ptr: *mut snd_htimestamp_t) -> () {
    type F = unsafe extern "C" fn(obj: *const snd_pcm_status_t, ptr: *mut snd_htimestamp_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_status_get_htstamp\0")) {
        Some(function) => function(obj, ptr),
        None => (),
    }
}
pub unsafe fn snd_pcm_status_get_audio_htstamp(
        obj: *const snd_pcm_status_t,
        ptr: *mut snd_htimestamp_t,
    ) -> () {
    type F = unsafe extern "C" fn(
        obj: *const snd_pcm_status_t,
        ptr: *mut snd_htimestamp_t,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_status_get_audio_htstamp\0")) {
        Some(function) => function(obj, ptr),
        None => (),
    }
}
pub unsafe fn snd_pcm_status_set_audio_htstamp_config(
        obj: *mut snd_pcm_status_t,
        audio_tstamp_config: *mut snd_pcm_audio_tstamp_config_t,
    ) -> () {
    type F = unsafe extern "C" fn(
        obj: *mut snd_pcm_status_t,
        audio_tstamp_config: *mut snd_pcm_audio_tstamp_config_t,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_status_set_audio_htstamp_config\0")) {
        Some(function) => function(obj, audio_tstamp_config),
        None => (),
    }
}
pub unsafe fn snd_pcm_status_get_delay(obj: *const snd_pcm_status_t) -> snd_pcm_sframes_t {
    type F = unsafe extern "C" fn(obj: *const snd_pcm_status_t) -> snd_pcm_sframes_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_status_get_delay\0")) {
        Some(function) => function(obj),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_pcm_status_get_avail(obj: *const snd_pcm_status_t) -> snd_pcm_uframes_t {
    type F = unsafe extern "C" fn(obj: *const snd_pcm_status_t) -> snd_pcm_uframes_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_status_get_avail\0")) {
        Some(function) => function(obj),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_pcm_status_get_avail_max(obj: *const snd_pcm_status_t) -> snd_pcm_uframes_t {
    type F = unsafe extern "C" fn(obj: *const snd_pcm_status_t) -> snd_pcm_uframes_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_status_get_avail_max\0")) {
        Some(function) => function(obj),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_pcm_status_get_overrange(obj: *const snd_pcm_status_t) -> snd_pcm_uframes_t {
    type F = unsafe extern "C" fn(obj: *const snd_pcm_status_t) -> snd_pcm_uframes_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_status_get_overrange\0")) {
        Some(function) => function(obj),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_pcm_dump(pcm: *mut snd_pcm_t, out: *mut snd_output_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t, out: *mut snd_output_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_dump\0")) {
        Some(function) => function(pcm, out),
        None => -1,
    }
}
pub unsafe fn snd_pcm_dump_hw_setup(pcm: *mut snd_pcm_t, out: *mut snd_output_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t, out: *mut snd_output_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_dump_hw_setup\0")) {
        Some(function) => function(pcm, out),
        None => -1,
    }
}
pub unsafe fn snd_pcm_dump_sw_setup(pcm: *mut snd_pcm_t, out: *mut snd_output_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t, out: *mut snd_output_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_dump_sw_setup\0")) {
        Some(function) => function(pcm, out),
        None => -1,
    }
}
pub unsafe fn snd_pcm_hw_params_dump(
        params: *mut snd_pcm_hw_params_t,
        out: *mut snd_output_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *mut snd_pcm_hw_params_t,
        out: *mut snd_output_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_hw_params_dump\0")) {
        Some(function) => function(params, out),
        None => -1,
    }
}
pub unsafe fn snd_pcm_sw_params_dump(
        params: *mut snd_pcm_sw_params_t,
        out: *mut snd_output_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        params: *mut snd_pcm_sw_params_t,
        out: *mut snd_output_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_sw_params_dump\0")) {
        Some(function) => function(params, out),
        None => -1,
    }
}
pub unsafe fn snd_pcm_status_dump(
        status: *mut snd_pcm_status_t,
        out: *mut snd_output_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        status: *mut snd_pcm_status_t,
        out: *mut snd_output_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_status_dump\0")) {
        Some(function) => function(status, out),
        None => -1,
    }
}
pub unsafe fn snd_pcm_mmap_begin(
        pcm: *mut snd_pcm_t,
        areas: *mut *const snd_pcm_channel_area_t,
        offset: *mut snd_pcm_uframes_t,
        frames: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        areas: *mut *const snd_pcm_channel_area_t,
        offset: *mut snd_pcm_uframes_t,
        frames: *mut snd_pcm_uframes_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_mmap_begin\0")) {
        Some(function) => function(pcm, areas, offset, frames),
        None => -1,
    }
}
pub unsafe fn snd_pcm_mmap_commit(
        pcm: *mut snd_pcm_t,
        offset: snd_pcm_uframes_t,
        frames: snd_pcm_uframes_t,
    ) -> snd_pcm_sframes_t {
    type F = unsafe extern "C" fn(
        pcm: *mut snd_pcm_t,
        offset: snd_pcm_uframes_t,
        frames: snd_pcm_uframes_t,
    ) -> snd_pcm_sframes_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_mmap_commit\0")) {
        Some(function) => function(pcm, offset, frames),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_pcm_format_little_endian(format: snd_pcm_format_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(format: snd_pcm_format_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_format_little_endian\0")) {
        Some(function) => function(format),
        None => -1,
    }
}
pub unsafe fn snd_pcm_format_width(format: snd_pcm_format_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(format: snd_pcm_format_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_format_width\0")) {
        Some(function) => function(format),
        None => -1,
    }
}
pub unsafe fn snd_pcm_format_physical_width(format: snd_pcm_format_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(format: snd_pcm_format_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_format_physical_width\0")) {
        Some(function) => function(format),
        None => -1,
    }
}
pub unsafe fn snd_pcm_format_silence_16(format: snd_pcm_format_t) -> u16 {
    type F = unsafe extern "C" fn(format: snd_pcm_format_t) -> u16;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_format_silence_16\0")) {
        Some(function) => function(format),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_pcm_bytes_to_frames(pcm: *mut snd_pcm_t, bytes: isize) -> snd_pcm_sframes_t {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t, bytes: isize) -> snd_pcm_sframes_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_bytes_to_frames\0")) {
        Some(function) => function(pcm, bytes),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_pcm_frames_to_bytes(pcm: *mut snd_pcm_t, frames: snd_pcm_sframes_t) -> isize {
    type F = unsafe extern "C" fn(pcm: *mut snd_pcm_t, frames: snd_pcm_sframes_t) -> isize;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_pcm_frames_to_bytes\0")) {
        Some(function) => function(pcm, frames),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_rawmidi_open(
        in_rmidi: *mut *mut snd_rawmidi_t,
        out_rmidi: *mut *mut snd_rawmidi_t,
        name: *const ::core::ffi::c_char,
        mode: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        in_rmidi: *mut *mut snd_rawmidi_t,
        out_rmidi: *mut *mut snd_rawmidi_t,
        name: *const ::core::ffi::c_char,
        mode: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_open\0")) {
        Some(function) => function(in_rmidi, out_rmidi, name, mode),
        None => -1,
    }
}
pub unsafe fn snd_rawmidi_close(rmidi: *mut snd_rawmidi_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(rmidi: *mut snd_rawmidi_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_close\0")) {
        Some(function) => function(rmidi),
        None => -1,
    }
}
pub unsafe fn snd_rawmidi_poll_descriptors_count(rmidi: *mut snd_rawmidi_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(rmidi: *mut snd_rawmidi_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_poll_descriptors_count\0")) {
        Some(function) => function(rmidi),
        None => -1,
    }
}
pub unsafe fn snd_rawmidi_poll_descriptors(
        rmidi: *mut snd_rawmidi_t,
        pfds: *mut pollfd,
        space: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        rmidi: *mut snd_rawmidi_t,
        pfds: *mut pollfd,
        space: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_poll_descriptors\0")) {
        Some(function) => function(rmidi, pfds, space),
        None => -1,
    }
}
pub unsafe fn snd_rawmidi_poll_descriptors_revents(
        rawmidi: *mut snd_rawmidi_t,
        pfds: *mut pollfd,
        nfds: ::core::ffi::c_uint,
        revent: *mut ::core::ffi::c_ushort,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        rawmidi: *mut snd_rawmidi_t,
        pfds: *mut pollfd,
        nfds: ::core::ffi::c_uint,
        revent: *mut ::core::ffi::c_ushort,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_poll_descriptors_revents\0")) {
        Some(function) => function(rawmidi, pfds, nfds, revent),
        None => -1,
    }
}
pub unsafe fn snd_rawmidi_info_malloc(ptr: *mut *mut snd_rawmidi_info_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ptr: *mut *mut snd_rawmidi_info_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_info_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_rawmidi_info_free(obj: *mut snd_rawmidi_info_t) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_rawmidi_info_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_info_free\0")) {
        Some(function) => function(obj),
        None => (),
    }
}
pub unsafe fn snd_rawmidi_info_get_device(obj: *const snd_rawmidi_info_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(obj: *const snd_rawmidi_info_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_info_get_device\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_rawmidi_info_get_subdevice(obj: *const snd_rawmidi_info_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(obj: *const snd_rawmidi_info_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_info_get_subdevice\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_rawmidi_info_get_stream(obj: *const snd_rawmidi_info_t) -> snd_rawmidi_stream_t {
    type F = unsafe extern "C" fn(obj: *const snd_rawmidi_info_t) -> snd_rawmidi_stream_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_info_get_stream\0")) {
        Some(function) => function(obj),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_rawmidi_info_get_id(obj: *const snd_rawmidi_info_t) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(obj: *const snd_rawmidi_info_t) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_info_get_id\0")) {
        Some(function) => function(obj),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_rawmidi_info_get_subdevice_name(
        obj: *const snd_rawmidi_info_t,
    ) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(
        obj: *const snd_rawmidi_info_t,
    ) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_info_get_subdevice_name\0")) {
        Some(function) => function(obj),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_rawmidi_info_get_subdevices_count(
        obj: *const snd_rawmidi_info_t,
    ) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(
        obj: *const snd_rawmidi_info_t,
    ) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_info_get_subdevices_count\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_rawmidi_info_set_device(obj: *mut snd_rawmidi_info_t, val: ::core::ffi::c_uint) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_rawmidi_info_t, val: ::core::ffi::c_uint) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_info_set_device\0")) {
        Some(function) => function(obj, val),
        None => (),
    }
}
pub unsafe fn snd_rawmidi_info_set_subdevice(obj: *mut snd_rawmidi_info_t, val: ::core::ffi::c_uint) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_rawmidi_info_t, val: ::core::ffi::c_uint) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_info_set_subdevice\0")) {
        Some(function) => function(obj, val),
        None => (),
    }
}
pub unsafe fn snd_rawmidi_info_set_stream(obj: *mut snd_rawmidi_info_t, val: snd_rawmidi_stream_t) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_rawmidi_info_t, val: snd_rawmidi_stream_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_info_set_stream\0")) {
        Some(function) => function(obj, val),
        None => (),
    }
}
pub unsafe fn snd_rawmidi_info(
        rmidi: *mut snd_rawmidi_t,
        info: *mut snd_rawmidi_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        rmidi: *mut snd_rawmidi_t,
        info: *mut snd_rawmidi_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_info\0")) {
        Some(function) => function(rmidi, info),
        None => -1,
    }
}
pub unsafe fn snd_rawmidi_status_malloc(ptr: *mut *mut snd_rawmidi_status_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ptr: *mut *mut snd_rawmidi_status_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_status_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_rawmidi_status_free(obj: *mut snd_rawmidi_status_t) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_rawmidi_status_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_status_free\0")) {
        Some(function) => function(obj),
        None => (),
    }
}
pub unsafe fn snd_rawmidi_status_get_avail(obj: *const snd_rawmidi_status_t) -> usize {
    type F = unsafe extern "C" fn(obj: *const snd_rawmidi_status_t) -> usize;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_status_get_avail\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_rawmidi_status_get_xruns(obj: *const snd_rawmidi_status_t) -> usize {
    type F = unsafe extern "C" fn(obj: *const snd_rawmidi_status_t) -> usize;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_status_get_xruns\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_rawmidi_status(
        rmidi: *mut snd_rawmidi_t,
        status: *mut snd_rawmidi_status_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        rmidi: *mut snd_rawmidi_t,
        status: *mut snd_rawmidi_status_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_status\0")) {
        Some(function) => function(rmidi, status),
        None => -1,
    }
}
pub unsafe fn snd_rawmidi_drain(rmidi: *mut snd_rawmidi_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(rmidi: *mut snd_rawmidi_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_drain\0")) {
        Some(function) => function(rmidi),
        None => -1,
    }
}
pub unsafe fn snd_rawmidi_drop(rmidi: *mut snd_rawmidi_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(rmidi: *mut snd_rawmidi_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_drop\0")) {
        Some(function) => function(rmidi),
        None => -1,
    }
}
pub unsafe fn snd_rawmidi_write(
        rmidi: *mut snd_rawmidi_t,
        buffer: *const ::core::ffi::c_void,
        size: usize,
    ) -> isize {
    type F = unsafe extern "C" fn(
        rmidi: *mut snd_rawmidi_t,
        buffer: *const ::core::ffi::c_void,
        size: usize,
    ) -> isize;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_write\0")) {
        Some(function) => function(rmidi, buffer, size),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_rawmidi_read(
        rmidi: *mut snd_rawmidi_t,
        buffer: *mut ::core::ffi::c_void,
        size: usize,
    ) -> isize {
    type F = unsafe extern "C" fn(
        rmidi: *mut snd_rawmidi_t,
        buffer: *mut ::core::ffi::c_void,
        size: usize,
    ) -> isize;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_read\0")) {
        Some(function) => function(rmidi, buffer, size),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_rawmidi_name(rmidi: *mut snd_rawmidi_t) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(rmidi: *mut snd_rawmidi_t) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_rawmidi_name\0")) {
        Some(function) => function(rmidi),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_card_next(card: *mut ::core::ffi::c_int) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(card: *mut ::core::ffi::c_int) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_card_next\0")) {
        Some(function) => function(card),
        None => -1,
    }
}
pub unsafe fn snd_card_get_index(name: *const ::core::ffi::c_char) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(name: *const ::core::ffi::c_char) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_card_get_index\0")) {
        Some(function) => function(name),
        None => -1,
    }
}
pub unsafe fn snd_card_get_name(
        card: ::core::ffi::c_int,
        name: *mut *mut ::core::ffi::c_char,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        card: ::core::ffi::c_int,
        name: *mut *mut ::core::ffi::c_char,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_card_get_name\0")) {
        Some(function) => function(card, name),
        None => -1,
    }
}
pub unsafe fn snd_card_get_longname(
        card: ::core::ffi::c_int,
        name: *mut *mut ::core::ffi::c_char,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        card: ::core::ffi::c_int,
        name: *mut *mut ::core::ffi::c_char,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_card_get_longname\0")) {
        Some(function) => function(card, name),
        None => -1,
    }
}
pub unsafe fn snd_ctl_open(
        ctl: *mut *mut snd_ctl_t,
        name: *const ::core::ffi::c_char,
        mode: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ctl: *mut *mut snd_ctl_t,
        name: *const ::core::ffi::c_char,
        mode: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_open\0")) {
        Some(function) => function(ctl, name, mode),
        None => -1,
    }
}
pub unsafe fn snd_ctl_close(ctl: *mut snd_ctl_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ctl: *mut snd_ctl_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_close\0")) {
        Some(function) => function(ctl),
        None => -1,
    }
}
pub unsafe fn snd_ctl_poll_descriptors_count(ctl: *mut snd_ctl_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ctl: *mut snd_ctl_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_poll_descriptors_count\0")) {
        Some(function) => function(ctl),
        None => -1,
    }
}
pub unsafe fn snd_ctl_poll_descriptors(
        ctl: *mut snd_ctl_t,
        pfds: *mut pollfd,
        space: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ctl: *mut snd_ctl_t,
        pfds: *mut pollfd,
        space: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_poll_descriptors\0")) {
        Some(function) => function(ctl, pfds, space),
        None => -1,
    }
}
pub unsafe fn snd_ctl_poll_descriptors_revents(
        ctl: *mut snd_ctl_t,
        pfds: *mut pollfd,
        nfds: ::core::ffi::c_uint,
        revents: *mut ::core::ffi::c_ushort,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ctl: *mut snd_ctl_t,
        pfds: *mut pollfd,
        nfds: ::core::ffi::c_uint,
        revents: *mut ::core::ffi::c_ushort,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_poll_descriptors_revents\0")) {
        Some(function) => function(ctl, pfds, nfds, revents),
        None => -1,
    }
}
pub unsafe fn snd_ctl_subscribe_events(
        ctl: *mut snd_ctl_t,
        subscribe: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ctl: *mut snd_ctl_t,
        subscribe: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_subscribe_events\0")) {
        Some(function) => function(ctl, subscribe),
        None => -1,
    }
}
pub unsafe fn snd_ctl_card_info(
        ctl: *mut snd_ctl_t,
        info: *mut snd_ctl_card_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ctl: *mut snd_ctl_t,
        info: *mut snd_ctl_card_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_card_info\0")) {
        Some(function) => function(ctl, info),
        None => -1,
    }
}
pub unsafe fn snd_ctl_elem_list(
        ctl: *mut snd_ctl_t,
        list: *mut snd_ctl_elem_list_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ctl: *mut snd_ctl_t,
        list: *mut snd_ctl_elem_list_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_list\0")) {
        Some(function) => function(ctl, list),
        None => -1,
    }
}
pub unsafe fn snd_ctl_elem_read(
        ctl: *mut snd_ctl_t,
        data: *mut snd_ctl_elem_value_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ctl: *mut snd_ctl_t,
        data: *mut snd_ctl_elem_value_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_read\0")) {
        Some(function) => function(ctl, data),
        None => -1,
    }
}
pub unsafe fn snd_ctl_elem_write(
        ctl: *mut snd_ctl_t,
        data: *mut snd_ctl_elem_value_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ctl: *mut snd_ctl_t,
        data: *mut snd_ctl_elem_value_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_write\0")) {
        Some(function) => function(ctl, data),
        None => -1,
    }
}
pub unsafe fn snd_ctl_elem_lock(ctl: *mut snd_ctl_t, id: *mut snd_ctl_elem_id_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ctl: *mut snd_ctl_t, id: *mut snd_ctl_elem_id_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_lock\0")) {
        Some(function) => function(ctl, id),
        None => -1,
    }
}
pub unsafe fn snd_ctl_elem_unlock(
        ctl: *mut snd_ctl_t,
        id: *mut snd_ctl_elem_id_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ctl: *mut snd_ctl_t,
        id: *mut snd_ctl_elem_id_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_unlock\0")) {
        Some(function) => function(ctl, id),
        None => -1,
    }
}
pub unsafe fn snd_ctl_pcm_next_device(
        ctl: *mut snd_ctl_t,
        device: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ctl: *mut snd_ctl_t,
        device: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_pcm_next_device\0")) {
        Some(function) => function(ctl, device),
        None => -1,
    }
}
pub unsafe fn snd_ctl_pcm_info(ctl: *mut snd_ctl_t, info: *mut snd_pcm_info_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ctl: *mut snd_ctl_t, info: *mut snd_pcm_info_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_pcm_info\0")) {
        Some(function) => function(ctl, info),
        None => -1,
    }
}
pub unsafe fn snd_ctl_rawmidi_next_device(
        ctl: *mut snd_ctl_t,
        device: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ctl: *mut snd_ctl_t,
        device: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_rawmidi_next_device\0")) {
        Some(function) => function(ctl, device),
        None => -1,
    }
}
pub unsafe fn snd_ctl_rawmidi_info(
        ctl: *mut snd_ctl_t,
        info: *mut snd_rawmidi_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ctl: *mut snd_ctl_t,
        info: *mut snd_rawmidi_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_rawmidi_info\0")) {
        Some(function) => function(ctl, info),
        None => -1,
    }
}
pub unsafe fn snd_ctl_read(ctl: *mut snd_ctl_t, event: *mut snd_ctl_event_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ctl: *mut snd_ctl_t, event: *mut snd_ctl_event_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_read\0")) {
        Some(function) => function(ctl, event),
        None => -1,
    }
}
pub unsafe fn snd_ctl_wait(ctl: *mut snd_ctl_t, timeout: ::core::ffi::c_int) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ctl: *mut snd_ctl_t, timeout: ::core::ffi::c_int) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_wait\0")) {
        Some(function) => function(ctl, timeout),
        None => -1,
    }
}
pub unsafe fn snd_ctl_event_elem_get_mask(obj: *const snd_ctl_event_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(obj: *const snd_ctl_event_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_event_elem_get_mask\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_ctl_event_elem_get_id(obj: *const snd_ctl_event_t, ptr: *mut snd_ctl_elem_id_t) -> () {
    type F = unsafe extern "C" fn(obj: *const snd_ctl_event_t, ptr: *mut snd_ctl_elem_id_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_event_elem_get_id\0")) {
        Some(function) => function(obj, ptr),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_list_alloc_space(
        obj: *mut snd_ctl_elem_list_t,
        entries: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        obj: *mut snd_ctl_elem_list_t,
        entries: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_list_alloc_space\0")) {
        Some(function) => function(obj, entries),
        None => -1,
    }
}
pub unsafe fn snd_ctl_elem_list_free_space(obj: *mut snd_ctl_elem_list_t) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_ctl_elem_list_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_list_free_space\0")) {
        Some(function) => function(obj),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_id_sizeof() -> usize {
    type F = unsafe extern "C" fn() -> usize;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_id_sizeof\0")) {
        Some(function) => function(),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_id_get_numid(obj: *const snd_ctl_elem_id_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(obj: *const snd_ctl_elem_id_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_id_get_numid\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_id_get_interface(obj: *const snd_ctl_elem_id_t) -> snd_ctl_elem_iface_t {
    type F = unsafe extern "C" fn(obj: *const snd_ctl_elem_id_t) -> snd_ctl_elem_iface_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_id_get_interface\0")) {
        Some(function) => function(obj),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_ctl_elem_id_get_device(obj: *const snd_ctl_elem_id_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(obj: *const snd_ctl_elem_id_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_id_get_device\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_id_get_subdevice(obj: *const snd_ctl_elem_id_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(obj: *const snd_ctl_elem_id_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_id_get_subdevice\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_id_get_name(obj: *const snd_ctl_elem_id_t) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(obj: *const snd_ctl_elem_id_t) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_id_get_name\0")) {
        Some(function) => function(obj),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_ctl_elem_id_get_index(obj: *const snd_ctl_elem_id_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(obj: *const snd_ctl_elem_id_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_id_get_index\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_id_set_numid(obj: *mut snd_ctl_elem_id_t, val: ::core::ffi::c_uint) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_ctl_elem_id_t, val: ::core::ffi::c_uint) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_id_set_numid\0")) {
        Some(function) => function(obj, val),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_id_set_interface(obj: *mut snd_ctl_elem_id_t, val: snd_ctl_elem_iface_t) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_ctl_elem_id_t, val: snd_ctl_elem_iface_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_id_set_interface\0")) {
        Some(function) => function(obj, val),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_id_set_device(obj: *mut snd_ctl_elem_id_t, val: ::core::ffi::c_uint) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_ctl_elem_id_t, val: ::core::ffi::c_uint) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_id_set_device\0")) {
        Some(function) => function(obj, val),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_id_set_subdevice(obj: *mut snd_ctl_elem_id_t, val: ::core::ffi::c_uint) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_ctl_elem_id_t, val: ::core::ffi::c_uint) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_id_set_subdevice\0")) {
        Some(function) => function(obj, val),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_id_set_name(obj: *mut snd_ctl_elem_id_t, val: *const ::core::ffi::c_char) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_ctl_elem_id_t, val: *const ::core::ffi::c_char) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_id_set_name\0")) {
        Some(function) => function(obj, val),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_id_set_index(obj: *mut snd_ctl_elem_id_t, val: ::core::ffi::c_uint) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_ctl_elem_id_t, val: ::core::ffi::c_uint) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_id_set_index\0")) {
        Some(function) => function(obj, val),
        None => (),
    }
}
pub unsafe fn snd_ctl_card_info_malloc(ptr: *mut *mut snd_ctl_card_info_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ptr: *mut *mut snd_ctl_card_info_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_card_info_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_ctl_card_info_free(obj: *mut snd_ctl_card_info_t) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_ctl_card_info_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_card_info_free\0")) {
        Some(function) => function(obj),
        None => (),
    }
}
pub unsafe fn snd_ctl_card_info_get_card(obj: *const snd_ctl_card_info_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(obj: *const snd_ctl_card_info_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_card_info_get_card\0")) {
        Some(function) => function(obj),
        None => -1,
    }
}
pub unsafe fn snd_ctl_card_info_get_id(obj: *const snd_ctl_card_info_t) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(obj: *const snd_ctl_card_info_t) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_card_info_get_id\0")) {
        Some(function) => function(obj),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_ctl_card_info_get_driver(
        obj: *const snd_ctl_card_info_t,
    ) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_card_info_t,
    ) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_card_info_get_driver\0")) {
        Some(function) => function(obj),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_ctl_card_info_get_name(
        obj: *const snd_ctl_card_info_t,
    ) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_card_info_t,
    ) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_card_info_get_name\0")) {
        Some(function) => function(obj),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_ctl_card_info_get_longname(
        obj: *const snd_ctl_card_info_t,
    ) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_card_info_t,
    ) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_card_info_get_longname\0")) {
        Some(function) => function(obj),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_ctl_card_info_get_mixername(
        obj: *const snd_ctl_card_info_t,
    ) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_card_info_t,
    ) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_card_info_get_mixername\0")) {
        Some(function) => function(obj),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_ctl_card_info_get_components(
        obj: *const snd_ctl_card_info_t,
    ) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_card_info_t,
    ) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_card_info_get_components\0")) {
        Some(function) => function(obj),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_ctl_event_malloc(ptr: *mut *mut snd_ctl_event_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ptr: *mut *mut snd_ctl_event_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_event_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_ctl_event_free(obj: *mut snd_ctl_event_t) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_ctl_event_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_event_free\0")) {
        Some(function) => function(obj),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_list_malloc(ptr: *mut *mut snd_ctl_elem_list_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ptr: *mut *mut snd_ctl_elem_list_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_list_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_ctl_elem_list_free(obj: *mut snd_ctl_elem_list_t) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_ctl_elem_list_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_list_free\0")) {
        Some(function) => function(obj),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_list_get_used(obj: *const snd_ctl_elem_list_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(obj: *const snd_ctl_elem_list_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_list_get_used\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_list_get_count(obj: *const snd_ctl_elem_list_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(obj: *const snd_ctl_elem_list_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_list_get_count\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_list_get_id(
        obj: *const snd_ctl_elem_list_t,
        idx: ::core::ffi::c_uint,
        ptr: *mut snd_ctl_elem_id_t,
    ) -> () {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_elem_list_t,
        idx: ::core::ffi::c_uint,
        ptr: *mut snd_ctl_elem_id_t,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_list_get_id\0")) {
        Some(function) => function(obj, idx, ptr),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_list_get_numid(
        obj: *const snd_ctl_elem_list_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_elem_list_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_list_get_numid\0")) {
        Some(function) => function(obj, idx),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_list_get_interface(
        obj: *const snd_ctl_elem_list_t,
        idx: ::core::ffi::c_uint,
    ) -> snd_ctl_elem_iface_t {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_elem_list_t,
        idx: ::core::ffi::c_uint,
    ) -> snd_ctl_elem_iface_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_list_get_interface\0")) {
        Some(function) => function(obj, idx),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_ctl_elem_list_get_device(
        obj: *const snd_ctl_elem_list_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_elem_list_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_list_get_device\0")) {
        Some(function) => function(obj, idx),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_list_get_subdevice(
        obj: *const snd_ctl_elem_list_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_elem_list_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_list_get_subdevice\0")) {
        Some(function) => function(obj, idx),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_list_get_name(
        obj: *const snd_ctl_elem_list_t,
        idx: ::core::ffi::c_uint,
    ) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_elem_list_t,
        idx: ::core::ffi::c_uint,
    ) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_list_get_name\0")) {
        Some(function) => function(obj, idx),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_ctl_elem_list_get_index(
        obj: *const snd_ctl_elem_list_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_elem_list_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_list_get_index\0")) {
        Some(function) => function(obj, idx),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_info_sizeof() -> usize {
    type F = unsafe extern "C" fn() -> usize;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_info_sizeof\0")) {
        Some(function) => function(),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_info_malloc(ptr: *mut *mut snd_ctl_elem_info_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ptr: *mut *mut snd_ctl_elem_info_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_info_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_ctl_elem_info_free(obj: *mut snd_ctl_elem_info_t) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_ctl_elem_info_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_info_free\0")) {
        Some(function) => function(obj),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_info_get_type(obj: *const snd_ctl_elem_info_t) -> snd_ctl_elem_type_t {
    type F = unsafe extern "C" fn(obj: *const snd_ctl_elem_info_t) -> snd_ctl_elem_type_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_info_get_type\0")) {
        Some(function) => function(obj),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_ctl_elem_info_get_count(obj: *const snd_ctl_elem_info_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(obj: *const snd_ctl_elem_info_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_info_get_count\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_value_sizeof() -> usize {
    type F = unsafe extern "C" fn() -> usize;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_value_sizeof\0")) {
        Some(function) => function(),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_value_malloc(ptr: *mut *mut snd_ctl_elem_value_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ptr: *mut *mut snd_ctl_elem_value_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_value_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_ctl_elem_value_free(obj: *mut snd_ctl_elem_value_t) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_ctl_elem_value_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_value_free\0")) {
        Some(function) => function(obj),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_value_clear(obj: *mut snd_ctl_elem_value_t) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_ctl_elem_value_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_value_clear\0")) {
        Some(function) => function(obj),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_value_set_id(obj: *mut snd_ctl_elem_value_t, ptr: *const snd_ctl_elem_id_t) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_ctl_elem_value_t, ptr: *const snd_ctl_elem_id_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_value_set_id\0")) {
        Some(function) => function(obj, ptr),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_value_get_boolean(
        obj: *const snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_value_get_boolean\0")) {
        Some(function) => function(obj, idx),
        None => -1,
    }
}
pub unsafe fn snd_ctl_elem_value_get_integer(
        obj: *const snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_long {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_long;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_value_get_integer\0")) {
        Some(function) => function(obj, idx),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_value_get_integer64(
        obj: *const snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_longlong {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_longlong;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_value_get_integer64\0")) {
        Some(function) => function(obj, idx),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_value_get_enumerated(
        obj: *const snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_value_get_enumerated\0")) {
        Some(function) => function(obj, idx),
        None => 0,
    }
}
pub unsafe fn snd_ctl_elem_value_get_byte(
        obj: *const snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_uchar {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_uchar;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_value_get_byte\0")) {
        Some(function) => function(obj, idx),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_ctl_elem_value_set_boolean(
        obj: *mut snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
        val: ::core::ffi::c_long,
    ) -> () {
    type F = unsafe extern "C" fn(
        obj: *mut snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
        val: ::core::ffi::c_long,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_value_set_boolean\0")) {
        Some(function) => function(obj, idx, val),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_value_set_integer(
        obj: *mut snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
        val: ::core::ffi::c_long,
    ) -> () {
    type F = unsafe extern "C" fn(
        obj: *mut snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
        val: ::core::ffi::c_long,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_value_set_integer\0")) {
        Some(function) => function(obj, idx, val),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_value_set_integer64(
        obj: *mut snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
        val: ::core::ffi::c_longlong,
    ) -> () {
    type F = unsafe extern "C" fn(
        obj: *mut snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
        val: ::core::ffi::c_longlong,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_value_set_integer64\0")) {
        Some(function) => function(obj, idx, val),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_value_set_enumerated(
        obj: *mut snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
        val: ::core::ffi::c_uint,
    ) -> () {
    type F = unsafe extern "C" fn(
        obj: *mut snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
        val: ::core::ffi::c_uint,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_value_set_enumerated\0")) {
        Some(function) => function(obj, idx, val),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_value_set_byte(
        obj: *mut snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
        val: ::core::ffi::c_uchar,
    ) -> () {
    type F = unsafe extern "C" fn(
        obj: *mut snd_ctl_elem_value_t,
        idx: ::core::ffi::c_uint,
        val: ::core::ffi::c_uchar,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_value_set_byte\0")) {
        Some(function) => function(obj, idx, val),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_set_bytes(
        obj: *mut snd_ctl_elem_value_t,
        data: *mut ::core::ffi::c_void,
        size: usize,
    ) -> () {
    type F = unsafe extern "C" fn(
        obj: *mut snd_ctl_elem_value_t,
        data: *mut ::core::ffi::c_void,
        size: usize,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_set_bytes\0")) {
        Some(function) => function(obj, data, size),
        None => (),
    }
}
pub unsafe fn snd_ctl_elem_value_get_bytes(
        obj: *const snd_ctl_elem_value_t,
    ) -> *const ::core::ffi::c_void {
    type F = unsafe extern "C" fn(
        obj: *const snd_ctl_elem_value_t,
    ) -> *const ::core::ffi::c_void;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_elem_value_get_bytes\0")) {
        Some(function) => function(obj),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_ctl_get_dB_range(
        ctl: *mut snd_ctl_t,
        id: *const snd_ctl_elem_id_t,
        min: *mut ::core::ffi::c_long,
        max: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ctl: *mut snd_ctl_t,
        id: *const snd_ctl_elem_id_t,
        min: *mut ::core::ffi::c_long,
        max: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_get_dB_range\0")) {
        Some(function) => function(ctl, id, min, max),
        None => -1,
    }
}
pub unsafe fn snd_ctl_convert_to_dB(
        ctl: *mut snd_ctl_t,
        id: *const snd_ctl_elem_id_t,
        volume: ::core::ffi::c_long,
        db_gain: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ctl: *mut snd_ctl_t,
        id: *const snd_ctl_elem_id_t,
        volume: ::core::ffi::c_long,
        db_gain: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_convert_to_dB\0")) {
        Some(function) => function(ctl, id, volume, db_gain),
        None => -1,
    }
}
pub unsafe fn snd_ctl_convert_from_dB(
        ctl: *mut snd_ctl_t,
        id: *const snd_ctl_elem_id_t,
        db_gain: ::core::ffi::c_long,
        value: *mut ::core::ffi::c_long,
        xdir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ctl: *mut snd_ctl_t,
        id: *const snd_ctl_elem_id_t,
        db_gain: ::core::ffi::c_long,
        value: *mut ::core::ffi::c_long,
        xdir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_ctl_convert_from_dB\0")) {
        Some(function) => function(ctl, id, db_gain, value, xdir),
        None => -1,
    }
}
pub unsafe fn snd_hctl_open(
        hctl: *mut *mut snd_hctl_t,
        name: *const ::core::ffi::c_char,
        mode: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        hctl: *mut *mut snd_hctl_t,
        name: *const ::core::ffi::c_char,
        mode: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_hctl_open\0")) {
        Some(function) => function(hctl, name, mode),
        None => -1,
    }
}
pub unsafe fn snd_hctl_close(hctl: *mut snd_hctl_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(hctl: *mut snd_hctl_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_hctl_close\0")) {
        Some(function) => function(hctl),
        None => -1,
    }
}
pub unsafe fn snd_hctl_poll_descriptors_count(hctl: *mut snd_hctl_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(hctl: *mut snd_hctl_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_hctl_poll_descriptors_count\0")) {
        Some(function) => function(hctl),
        None => -1,
    }
}
pub unsafe fn snd_hctl_poll_descriptors(
        hctl: *mut snd_hctl_t,
        pfds: *mut pollfd,
        space: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        hctl: *mut snd_hctl_t,
        pfds: *mut pollfd,
        space: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_hctl_poll_descriptors\0")) {
        Some(function) => function(hctl, pfds, space),
        None => -1,
    }
}
pub unsafe fn snd_hctl_poll_descriptors_revents(
        ctl: *mut snd_hctl_t,
        pfds: *mut pollfd,
        nfds: ::core::ffi::c_uint,
        revents: *mut ::core::ffi::c_ushort,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ctl: *mut snd_hctl_t,
        pfds: *mut pollfd,
        nfds: ::core::ffi::c_uint,
        revents: *mut ::core::ffi::c_ushort,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_hctl_poll_descriptors_revents\0")) {
        Some(function) => function(ctl, pfds, nfds, revents),
        None => -1,
    }
}
pub unsafe fn snd_hctl_first_elem(hctl: *mut snd_hctl_t) -> *mut snd_hctl_elem_t {
    type F = unsafe extern "C" fn(hctl: *mut snd_hctl_t) -> *mut snd_hctl_elem_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_hctl_first_elem\0")) {
        Some(function) => function(hctl),
        None => core::ptr::null_mut(),
    }
}
pub unsafe fn snd_hctl_find_elem(
        hctl: *mut snd_hctl_t,
        id: *const snd_ctl_elem_id_t,
    ) -> *mut snd_hctl_elem_t {
    type F = unsafe extern "C" fn(
        hctl: *mut snd_hctl_t,
        id: *const snd_ctl_elem_id_t,
    ) -> *mut snd_hctl_elem_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_hctl_find_elem\0")) {
        Some(function) => function(hctl, id),
        None => core::ptr::null_mut(),
    }
}
pub unsafe fn snd_hctl_load(hctl: *mut snd_hctl_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(hctl: *mut snd_hctl_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_hctl_load\0")) {
        Some(function) => function(hctl),
        None => -1,
    }
}
pub unsafe fn snd_hctl_handle_events(hctl: *mut snd_hctl_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(hctl: *mut snd_hctl_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_hctl_handle_events\0")) {
        Some(function) => function(hctl),
        None => -1,
    }
}
pub unsafe fn snd_hctl_wait(hctl: *mut snd_hctl_t, timeout: ::core::ffi::c_int) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(hctl: *mut snd_hctl_t, timeout: ::core::ffi::c_int) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_hctl_wait\0")) {
        Some(function) => function(hctl, timeout),
        None => -1,
    }
}
pub unsafe fn snd_hctl_elem_next(elem: *mut snd_hctl_elem_t) -> *mut snd_hctl_elem_t {
    type F = unsafe extern "C" fn(elem: *mut snd_hctl_elem_t) -> *mut snd_hctl_elem_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_hctl_elem_next\0")) {
        Some(function) => function(elem),
        None => core::ptr::null_mut(),
    }
}
pub unsafe fn snd_hctl_elem_info(
        elem: *mut snd_hctl_elem_t,
        info: *mut snd_ctl_elem_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_hctl_elem_t,
        info: *mut snd_ctl_elem_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_hctl_elem_info\0")) {
        Some(function) => function(elem, info),
        None => -1,
    }
}
pub unsafe fn snd_hctl_elem_read(
        elem: *mut snd_hctl_elem_t,
        value: *mut snd_ctl_elem_value_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_hctl_elem_t,
        value: *mut snd_ctl_elem_value_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_hctl_elem_read\0")) {
        Some(function) => function(elem, value),
        None => -1,
    }
}
pub unsafe fn snd_hctl_elem_write(
        elem: *mut snd_hctl_elem_t,
        value: *mut snd_ctl_elem_value_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_hctl_elem_t,
        value: *mut snd_ctl_elem_value_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_hctl_elem_write\0")) {
        Some(function) => function(elem, value),
        None => -1,
    }
}
pub unsafe fn snd_hctl_elem_get_id(obj: *const snd_hctl_elem_t, ptr: *mut snd_ctl_elem_id_t) -> () {
    type F = unsafe extern "C" fn(obj: *const snd_hctl_elem_t, ptr: *mut snd_ctl_elem_id_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_hctl_elem_get_id\0")) {
        Some(function) => function(obj, ptr),
        None => (),
    }
}
pub unsafe fn snd_device_name_hint(
        card: ::core::ffi::c_int,
        iface: *const ::core::ffi::c_char,
        hints: *mut *mut *mut ::core::ffi::c_void,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        card: ::core::ffi::c_int,
        iface: *const ::core::ffi::c_char,
        hints: *mut *mut *mut ::core::ffi::c_void,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_device_name_hint\0")) {
        Some(function) => function(card, iface, hints),
        None => -1,
    }
}
pub unsafe fn snd_device_name_free_hint(hints: *mut *mut ::core::ffi::c_void) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(hints: *mut *mut ::core::ffi::c_void) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_device_name_free_hint\0")) {
        Some(function) => function(hints),
        None => -1,
    }
}
pub unsafe fn snd_device_name_get_hint(
        hint: *const ::core::ffi::c_void,
        id: *const ::core::ffi::c_char,
    ) -> *mut ::core::ffi::c_char {
    type F = unsafe extern "C" fn(
        hint: *const ::core::ffi::c_void,
        id: *const ::core::ffi::c_char,
    ) -> *mut ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_device_name_get_hint\0")) {
        Some(function) => function(hint, id),
        None => core::ptr::null_mut(),
    }
}
pub unsafe fn snd_mixer_open(
        mixer: *mut *mut snd_mixer_t,
        mode: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        mixer: *mut *mut snd_mixer_t,
        mode: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_open\0")) {
        Some(function) => function(mixer, mode),
        None => -1,
    }
}
pub unsafe fn snd_mixer_close(mixer: *mut snd_mixer_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(mixer: *mut snd_mixer_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_close\0")) {
        Some(function) => function(mixer),
        None => -1,
    }
}
pub unsafe fn snd_mixer_first_elem(mixer: *mut snd_mixer_t) -> *mut snd_mixer_elem_t {
    type F = unsafe extern "C" fn(mixer: *mut snd_mixer_t) -> *mut snd_mixer_elem_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_first_elem\0")) {
        Some(function) => function(mixer),
        None => core::ptr::null_mut(),
    }
}
pub unsafe fn snd_mixer_handle_events(mixer: *mut snd_mixer_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(mixer: *mut snd_mixer_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_handle_events\0")) {
        Some(function) => function(mixer),
        None => -1,
    }
}
pub unsafe fn snd_mixer_attach(
        mixer: *mut snd_mixer_t,
        name: *const ::core::ffi::c_char,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        mixer: *mut snd_mixer_t,
        name: *const ::core::ffi::c_char,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_attach\0")) {
        Some(function) => function(mixer, name),
        None => -1,
    }
}
pub unsafe fn snd_mixer_poll_descriptors_count(mixer: *mut snd_mixer_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(mixer: *mut snd_mixer_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_poll_descriptors_count\0")) {
        Some(function) => function(mixer),
        None => -1,
    }
}
pub unsafe fn snd_mixer_poll_descriptors(
        mixer: *mut snd_mixer_t,
        pfds: *mut pollfd,
        space: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        mixer: *mut snd_mixer_t,
        pfds: *mut pollfd,
        space: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_poll_descriptors\0")) {
        Some(function) => function(mixer, pfds, space),
        None => -1,
    }
}
pub unsafe fn snd_mixer_poll_descriptors_revents(
        mixer: *mut snd_mixer_t,
        pfds: *mut pollfd,
        nfds: ::core::ffi::c_uint,
        revents: *mut ::core::ffi::c_ushort,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        mixer: *mut snd_mixer_t,
        pfds: *mut pollfd,
        nfds: ::core::ffi::c_uint,
        revents: *mut ::core::ffi::c_ushort,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_poll_descriptors_revents\0")) {
        Some(function) => function(mixer, pfds, nfds, revents),
        None => -1,
    }
}
pub unsafe fn snd_mixer_load(mixer: *mut snd_mixer_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(mixer: *mut snd_mixer_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_load\0")) {
        Some(function) => function(mixer),
        None => -1,
    }
}
pub unsafe fn snd_mixer_wait(
        mixer: *mut snd_mixer_t,
        timeout: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        mixer: *mut snd_mixer_t,
        timeout: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_wait\0")) {
        Some(function) => function(mixer, timeout),
        None => -1,
    }
}
pub unsafe fn snd_mixer_elem_next(elem: *mut snd_mixer_elem_t) -> *mut snd_mixer_elem_t {
    type F = unsafe extern "C" fn(elem: *mut snd_mixer_elem_t) -> *mut snd_mixer_elem_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_elem_next\0")) {
        Some(function) => function(elem),
        None => core::ptr::null_mut(),
    }
}
pub unsafe fn snd_mixer_elem_get_type(obj: *const snd_mixer_elem_t) -> snd_mixer_elem_type_t {
    type F = unsafe extern "C" fn(obj: *const snd_mixer_elem_t) -> snd_mixer_elem_type_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_elem_get_type\0")) {
        Some(function) => function(obj),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_mixer_selem_channel_name(
        channel: snd_mixer_selem_channel_id_t,
    ) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(
        channel: snd_mixer_selem_channel_id_t,
    ) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_channel_name\0")) {
        Some(function) => function(channel),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_mixer_selem_register(
        mixer: *mut snd_mixer_t,
        options: *mut snd_mixer_selem_regopt,
        classp: *mut *mut snd_mixer_class_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        mixer: *mut snd_mixer_t,
        options: *mut snd_mixer_selem_regopt,
        classp: *mut *mut snd_mixer_class_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_register\0")) {
        Some(function) => function(mixer, options, classp),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_get_id(element: *mut snd_mixer_elem_t, id: *mut snd_mixer_selem_id_t) -> () {
    type F = unsafe extern "C" fn(element: *mut snd_mixer_elem_t, id: *mut snd_mixer_selem_id_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_get_id\0")) {
        Some(function) => function(element, id),
        None => (),
    }
}
pub unsafe fn snd_mixer_find_selem(
        mixer: *mut snd_mixer_t,
        id: *const snd_mixer_selem_id_t,
    ) -> *mut snd_mixer_elem_t {
    type F = unsafe extern "C" fn(
        mixer: *mut snd_mixer_t,
        id: *const snd_mixer_selem_id_t,
    ) -> *mut snd_mixer_elem_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_find_selem\0")) {
        Some(function) => function(mixer, id),
        None => core::ptr::null_mut(),
    }
}
pub unsafe fn snd_mixer_selem_is_playback_mono(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_is_playback_mono\0")) {
        Some(function) => function(elem),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_has_playback_channel(
        obj: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        obj: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_has_playback_channel\0")) {
        Some(function) => function(obj, channel),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_is_capture_mono(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_is_capture_mono\0")) {
        Some(function) => function(elem),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_has_capture_channel(
        obj: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        obj: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_has_capture_channel\0")) {
        Some(function) => function(obj, channel),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_has_playback_volume(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_has_playback_volume\0")) {
        Some(function) => function(elem),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_has_capture_volume(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_has_capture_volume\0")) {
        Some(function) => function(elem),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_has_playback_switch(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_has_playback_switch\0")) {
        Some(function) => function(elem),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_has_capture_switch(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_has_capture_switch\0")) {
        Some(function) => function(elem),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_ask_playback_vol_dB(
        elem: *mut snd_mixer_elem_t,
        value: ::core::ffi::c_long,
        dBvalue: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        value: ::core::ffi::c_long,
        dBvalue: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_ask_playback_vol_dB\0")) {
        Some(function) => function(elem, value, dBvalue),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_ask_capture_vol_dB(
        elem: *mut snd_mixer_elem_t,
        value: ::core::ffi::c_long,
        dBvalue: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        value: ::core::ffi::c_long,
        dBvalue: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_ask_capture_vol_dB\0")) {
        Some(function) => function(elem, value, dBvalue),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_ask_playback_dB_vol(
        elem: *mut snd_mixer_elem_t,
        dBvalue: ::core::ffi::c_long,
        dir: ::core::ffi::c_int,
        value: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        dBvalue: ::core::ffi::c_long,
        dir: ::core::ffi::c_int,
        value: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_ask_playback_dB_vol\0")) {
        Some(function) => function(elem, dBvalue, dir, value),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_ask_capture_dB_vol(
        elem: *mut snd_mixer_elem_t,
        dBvalue: ::core::ffi::c_long,
        dir: ::core::ffi::c_int,
        value: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        dBvalue: ::core::ffi::c_long,
        dir: ::core::ffi::c_int,
        value: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_ask_capture_dB_vol\0")) {
        Some(function) => function(elem, dBvalue, dir, value),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_get_playback_volume(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_get_playback_volume\0")) {
        Some(function) => function(elem, channel, value),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_get_capture_volume(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_get_capture_volume\0")) {
        Some(function) => function(elem, channel, value),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_get_playback_switch(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_get_playback_switch\0")) {
        Some(function) => function(elem, channel, value),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_get_capture_switch(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_get_capture_switch\0")) {
        Some(function) => function(elem, channel, value),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_set_playback_volume(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_set_playback_volume\0")) {
        Some(function) => function(elem, channel, value),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_set_capture_volume(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_set_capture_volume\0")) {
        Some(function) => function(elem, channel, value),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_set_playback_dB(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: ::core::ffi::c_long,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: ::core::ffi::c_long,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_set_playback_dB\0")) {
        Some(function) => function(elem, channel, value, dir),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_set_capture_dB(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: ::core::ffi::c_long,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: ::core::ffi::c_long,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_set_capture_dB\0")) {
        Some(function) => function(elem, channel, value, dir),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_set_playback_volume_all(
        elem: *mut snd_mixer_elem_t,
        value: ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        value: ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_set_playback_volume_all\0")) {
        Some(function) => function(elem, value),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_set_capture_volume_all(
        elem: *mut snd_mixer_elem_t,
        value: ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        value: ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_set_capture_volume_all\0")) {
        Some(function) => function(elem, value),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_set_playback_dB_all(
        elem: *mut snd_mixer_elem_t,
        value: ::core::ffi::c_long,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        value: ::core::ffi::c_long,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_set_playback_dB_all\0")) {
        Some(function) => function(elem, value, dir),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_set_capture_dB_all(
        elem: *mut snd_mixer_elem_t,
        value: ::core::ffi::c_long,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        value: ::core::ffi::c_long,
        dir: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_set_capture_dB_all\0")) {
        Some(function) => function(elem, value, dir),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_set_playback_switch(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_set_playback_switch\0")) {
        Some(function) => function(elem, channel, value),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_set_capture_switch(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        value: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_set_capture_switch\0")) {
        Some(function) => function(elem, channel, value),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_set_playback_switch_all(
        elem: *mut snd_mixer_elem_t,
        value: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        value: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_set_playback_switch_all\0")) {
        Some(function) => function(elem, value),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_set_capture_switch_all(
        elem: *mut snd_mixer_elem_t,
        value: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        value: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_set_capture_switch_all\0")) {
        Some(function) => function(elem, value),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_get_playback_volume_range(
        elem: *mut snd_mixer_elem_t,
        min: *mut ::core::ffi::c_long,
        max: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        min: *mut ::core::ffi::c_long,
        max: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_get_playback_volume_range\0")) {
        Some(function) => function(elem, min, max),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_get_playback_dB_range(
        elem: *mut snd_mixer_elem_t,
        min: *mut ::core::ffi::c_long,
        max: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        min: *mut ::core::ffi::c_long,
        max: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_get_playback_dB_range\0")) {
        Some(function) => function(elem, min, max),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_set_playback_volume_range(
        elem: *mut snd_mixer_elem_t,
        min: ::core::ffi::c_long,
        max: ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        min: ::core::ffi::c_long,
        max: ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_set_playback_volume_range\0")) {
        Some(function) => function(elem, min, max),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_get_capture_volume_range(
        elem: *mut snd_mixer_elem_t,
        min: *mut ::core::ffi::c_long,
        max: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        min: *mut ::core::ffi::c_long,
        max: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_get_capture_volume_range\0")) {
        Some(function) => function(elem, min, max),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_get_capture_dB_range(
        elem: *mut snd_mixer_elem_t,
        min: *mut ::core::ffi::c_long,
        max: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        min: *mut ::core::ffi::c_long,
        max: *mut ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_get_capture_dB_range\0")) {
        Some(function) => function(elem, min, max),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_set_capture_volume_range(
        elem: *mut snd_mixer_elem_t,
        min: ::core::ffi::c_long,
        max: ::core::ffi::c_long,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        min: ::core::ffi::c_long,
        max: ::core::ffi::c_long,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_set_capture_volume_range\0")) {
        Some(function) => function(elem, min, max),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_is_enumerated(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_is_enumerated\0")) {
        Some(function) => function(elem),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_is_enum_playback(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_is_enum_playback\0")) {
        Some(function) => function(elem),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_is_enum_capture(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_is_enum_capture\0")) {
        Some(function) => function(elem),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_get_enum_items(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(elem: *mut snd_mixer_elem_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_get_enum_items\0")) {
        Some(function) => function(elem),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_get_enum_item_name(
        elem: *mut snd_mixer_elem_t,
        idx: ::core::ffi::c_uint,
        maxlen: usize,
        str_: *mut ::core::ffi::c_char,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        idx: ::core::ffi::c_uint,
        maxlen: usize,
        str_: *mut ::core::ffi::c_char,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_get_enum_item_name\0")) {
        Some(function) => function(elem, idx, maxlen, str_),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_get_enum_item(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        idxp: *mut ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        idxp: *mut ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_get_enum_item\0")) {
        Some(function) => function(elem, channel, idxp),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_set_enum_item(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        elem: *mut snd_mixer_elem_t,
        channel: snd_mixer_selem_channel_id_t,
        idx: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_set_enum_item\0")) {
        Some(function) => function(elem, channel, idx),
        None => -1,
    }
}
pub unsafe fn snd_mixer_selem_id_sizeof() -> usize {
    type F = unsafe extern "C" fn() -> usize;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_id_sizeof\0")) {
        Some(function) => function(),
        None => 0,
    }
}
pub unsafe fn snd_mixer_selem_id_get_name(
        obj: *const snd_mixer_selem_id_t,
    ) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(
        obj: *const snd_mixer_selem_id_t,
    ) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_id_get_name\0")) {
        Some(function) => function(obj),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_mixer_selem_id_get_index(obj: *const snd_mixer_selem_id_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(obj: *const snd_mixer_selem_id_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_id_get_index\0")) {
        Some(function) => function(obj),
        None => 0,
    }
}
pub unsafe fn snd_mixer_selem_id_set_name(
        obj: *mut snd_mixer_selem_id_t,
        val: *const ::core::ffi::c_char,
    ) -> () {
    type F = unsafe extern "C" fn(
        obj: *mut snd_mixer_selem_id_t,
        val: *const ::core::ffi::c_char,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_id_set_name\0")) {
        Some(function) => function(obj, val),
        None => (),
    }
}
pub unsafe fn snd_mixer_selem_id_set_index(obj: *mut snd_mixer_selem_id_t, val: ::core::ffi::c_uint) -> () {
    type F = unsafe extern "C" fn(obj: *mut snd_mixer_selem_id_t, val: ::core::ffi::c_uint) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_mixer_selem_id_set_index\0")) {
        Some(function) => function(obj, val),
        None => (),
    }
}
pub unsafe fn snd_seq_open(
        handle: *mut *mut snd_seq_t,
        name: *const ::core::ffi::c_char,
        streams: ::core::ffi::c_int,
        mode: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut *mut snd_seq_t,
        name: *const ::core::ffi::c_char,
        streams: ::core::ffi::c_int,
        mode: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_open\0")) {
        Some(function) => function(handle, name, streams, mode),
        None => -1,
    }
}
pub unsafe fn snd_seq_close(handle: *mut snd_seq_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(handle: *mut snd_seq_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_close\0")) {
        Some(function) => function(handle),
        None => -1,
    }
}
pub unsafe fn snd_seq_poll_descriptors_count(
        handle: *mut snd_seq_t,
        events: ::core::ffi::c_short,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        events: ::core::ffi::c_short,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_poll_descriptors_count\0")) {
        Some(function) => function(handle, events),
        None => -1,
    }
}
pub unsafe fn snd_seq_poll_descriptors(
        handle: *mut snd_seq_t,
        pfds: *mut pollfd,
        space: ::core::ffi::c_uint,
        events: ::core::ffi::c_short,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        pfds: *mut pollfd,
        space: ::core::ffi::c_uint,
        events: ::core::ffi::c_short,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_poll_descriptors\0")) {
        Some(function) => function(handle, pfds, space, events),
        None => -1,
    }
}
pub unsafe fn snd_seq_poll_descriptors_revents(
        seq: *mut snd_seq_t,
        pfds: *mut pollfd,
        nfds: ::core::ffi::c_uint,
        revents: *mut ::core::ffi::c_ushort,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        seq: *mut snd_seq_t,
        pfds: *mut pollfd,
        nfds: ::core::ffi::c_uint,
        revents: *mut ::core::ffi::c_ushort,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_poll_descriptors_revents\0")) {
        Some(function) => function(seq, pfds, nfds, revents),
        None => -1,
    }
}
pub unsafe fn snd_seq_client_id(handle: *mut snd_seq_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(handle: *mut snd_seq_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_client_id\0")) {
        Some(function) => function(handle),
        None => -1,
    }
}
pub unsafe fn snd_seq_set_input_buffer_size(handle: *mut snd_seq_t, size: usize) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(handle: *mut snd_seq_t, size: usize) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_set_input_buffer_size\0")) {
        Some(function) => function(handle, size),
        None => -1,
    }
}
pub unsafe fn snd_seq_client_info_malloc(ptr: *mut *mut snd_seq_client_info_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ptr: *mut *mut snd_seq_client_info_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_client_info_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_seq_client_info_free(ptr: *mut snd_seq_client_info_t) -> () {
    type F = unsafe extern "C" fn(ptr: *mut snd_seq_client_info_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_client_info_free\0")) {
        Some(function) => function(ptr),
        None => (),
    }
}
pub unsafe fn snd_seq_client_info_get_client(info: *const snd_seq_client_info_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(info: *const snd_seq_client_info_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_client_info_get_client\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_client_info_get_name(
        info: *mut snd_seq_client_info_t,
    ) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_client_info_t,
    ) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_client_info_get_name\0")) {
        Some(function) => function(info),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_seq_client_info_get_card(info: *const snd_seq_client_info_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(info: *const snd_seq_client_info_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_client_info_get_card\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_client_info_set_client(
        info: *mut snd_seq_client_info_t,
        client: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_client_info_t,
        client: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_client_info_set_client\0")) {
        Some(function) => function(info, client),
        None => (),
    }
}
pub unsafe fn snd_seq_get_any_client_info(
        handle: *mut snd_seq_t,
        client: ::core::ffi::c_int,
        info: *mut snd_seq_client_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        client: ::core::ffi::c_int,
        info: *mut snd_seq_client_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_get_any_client_info\0")) {
        Some(function) => function(handle, client, info),
        None => -1,
    }
}
pub unsafe fn snd_seq_query_next_client(
        handle: *mut snd_seq_t,
        info: *mut snd_seq_client_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        info: *mut snd_seq_client_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_query_next_client\0")) {
        Some(function) => function(handle, info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_info_sizeof() -> usize {
    type F = unsafe extern "C" fn() -> usize;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_sizeof\0")) {
        Some(function) => function(),
        None => 0,
    }
}
pub unsafe fn snd_seq_port_info_malloc(ptr: *mut *mut snd_seq_port_info_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ptr: *mut *mut snd_seq_port_info_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_info_free(ptr: *mut snd_seq_port_info_t) -> () {
    type F = unsafe extern "C" fn(ptr: *mut snd_seq_port_info_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_free\0")) {
        Some(function) => function(ptr),
        None => (),
    }
}
pub unsafe fn snd_seq_port_info_get_client(info: *const snd_seq_port_info_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(info: *const snd_seq_port_info_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_get_client\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_info_get_port(info: *const snd_seq_port_info_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(info: *const snd_seq_port_info_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_get_port\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_info_get_name(
        info: *const snd_seq_port_info_t,
    ) -> *const ::core::ffi::c_char {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_port_info_t,
    ) -> *const ::core::ffi::c_char;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_get_name\0")) {
        Some(function) => function(info),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_seq_port_info_get_capability(
        info: *const snd_seq_port_info_t,
    ) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_port_info_t,
    ) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_get_capability\0")) {
        Some(function) => function(info),
        None => 0,
    }
}
pub unsafe fn snd_seq_port_info_get_type(info: *const snd_seq_port_info_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(info: *const snd_seq_port_info_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_get_type\0")) {
        Some(function) => function(info),
        None => 0,
    }
}
pub unsafe fn snd_seq_port_info_get_midi_channels(
        info: *const snd_seq_port_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_port_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_get_midi_channels\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_info_get_midi_voices(
        info: *const snd_seq_port_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_port_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_get_midi_voices\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_info_get_synth_voices(
        info: *const snd_seq_port_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_port_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_get_synth_voices\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_info_get_read_use(info: *const snd_seq_port_info_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(info: *const snd_seq_port_info_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_get_read_use\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_info_get_write_use(info: *const snd_seq_port_info_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(info: *const snd_seq_port_info_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_get_write_use\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_info_get_port_specified(
        info: *const snd_seq_port_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_port_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_get_port_specified\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_info_get_timestamping(
        info: *const snd_seq_port_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_port_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_get_timestamping\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_info_get_timestamp_real(
        info: *const snd_seq_port_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_port_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_get_timestamp_real\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_info_get_timestamp_queue(
        info: *const snd_seq_port_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_port_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_get_timestamp_queue\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_info_set_client(info: *mut snd_seq_port_info_t, client: ::core::ffi::c_int) -> () {
    type F = unsafe extern "C" fn(info: *mut snd_seq_port_info_t, client: ::core::ffi::c_int) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_set_client\0")) {
        Some(function) => function(info, client),
        None => (),
    }
}
pub unsafe fn snd_seq_port_info_set_port(info: *mut snd_seq_port_info_t, port: ::core::ffi::c_int) -> () {
    type F = unsafe extern "C" fn(info: *mut snd_seq_port_info_t, port: ::core::ffi::c_int) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_set_port\0")) {
        Some(function) => function(info, port),
        None => (),
    }
}
pub unsafe fn snd_seq_port_info_set_name(
        info: *mut snd_seq_port_info_t,
        name: *const ::core::ffi::c_char,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_port_info_t,
        name: *const ::core::ffi::c_char,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_set_name\0")) {
        Some(function) => function(info, name),
        None => (),
    }
}
pub unsafe fn snd_seq_port_info_set_capability(
        info: *mut snd_seq_port_info_t,
        capability: ::core::ffi::c_uint,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_port_info_t,
        capability: ::core::ffi::c_uint,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_set_capability\0")) {
        Some(function) => function(info, capability),
        None => (),
    }
}
pub unsafe fn snd_seq_port_info_set_type(info: *mut snd_seq_port_info_t, type_: ::core::ffi::c_uint) -> () {
    type F = unsafe extern "C" fn(info: *mut snd_seq_port_info_t, type_: ::core::ffi::c_uint) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_set_type\0")) {
        Some(function) => function(info, type_),
        None => (),
    }
}
pub unsafe fn snd_seq_port_info_set_midi_channels(
        info: *mut snd_seq_port_info_t,
        channels: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_port_info_t,
        channels: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_set_midi_channels\0")) {
        Some(function) => function(info, channels),
        None => (),
    }
}
pub unsafe fn snd_seq_port_info_set_midi_voices(
        info: *mut snd_seq_port_info_t,
        voices: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_port_info_t,
        voices: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_set_midi_voices\0")) {
        Some(function) => function(info, voices),
        None => (),
    }
}
pub unsafe fn snd_seq_port_info_set_synth_voices(
        info: *mut snd_seq_port_info_t,
        voices: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_port_info_t,
        voices: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_set_synth_voices\0")) {
        Some(function) => function(info, voices),
        None => (),
    }
}
pub unsafe fn snd_seq_port_info_set_port_specified(
        info: *mut snd_seq_port_info_t,
        val: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_port_info_t,
        val: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_set_port_specified\0")) {
        Some(function) => function(info, val),
        None => (),
    }
}
pub unsafe fn snd_seq_port_info_set_timestamping(
        info: *mut snd_seq_port_info_t,
        enable: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_port_info_t,
        enable: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_set_timestamping\0")) {
        Some(function) => function(info, enable),
        None => (),
    }
}
pub unsafe fn snd_seq_port_info_set_timestamp_real(
        info: *mut snd_seq_port_info_t,
        realtime: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_port_info_t,
        realtime: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_set_timestamp_real\0")) {
        Some(function) => function(info, realtime),
        None => (),
    }
}
pub unsafe fn snd_seq_port_info_set_timestamp_queue(
        info: *mut snd_seq_port_info_t,
        queue: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_port_info_t,
        queue: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_info_set_timestamp_queue\0")) {
        Some(function) => function(info, queue),
        None => (),
    }
}
pub unsafe fn snd_seq_create_port(
        handle: *mut snd_seq_t,
        info: *mut snd_seq_port_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        info: *mut snd_seq_port_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_create_port\0")) {
        Some(function) => function(handle, info),
        None => -1,
    }
}
pub unsafe fn snd_seq_delete_port(
        handle: *mut snd_seq_t,
        port: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        port: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_delete_port\0")) {
        Some(function) => function(handle, port),
        None => -1,
    }
}
pub unsafe fn snd_seq_get_any_port_info(
        handle: *mut snd_seq_t,
        client: ::core::ffi::c_int,
        port: ::core::ffi::c_int,
        info: *mut snd_seq_port_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        client: ::core::ffi::c_int,
        port: ::core::ffi::c_int,
        info: *mut snd_seq_port_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_get_any_port_info\0")) {
        Some(function) => function(handle, client, port, info),
        None => -1,
    }
}
pub unsafe fn snd_seq_set_port_info(
        handle: *mut snd_seq_t,
        port: ::core::ffi::c_int,
        info: *mut snd_seq_port_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        port: ::core::ffi::c_int,
        info: *mut snd_seq_port_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_set_port_info\0")) {
        Some(function) => function(handle, port, info),
        None => -1,
    }
}
pub unsafe fn snd_seq_query_next_port(
        handle: *mut snd_seq_t,
        info: *mut snd_seq_port_info_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        info: *mut snd_seq_port_info_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_query_next_port\0")) {
        Some(function) => function(handle, info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_subscribe_sizeof() -> usize {
    type F = unsafe extern "C" fn() -> usize;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_subscribe_sizeof\0")) {
        Some(function) => function(),
        None => 0,
    }
}
pub unsafe fn snd_seq_port_subscribe_malloc(
        ptr: *mut *mut snd_seq_port_subscribe_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ptr: *mut *mut snd_seq_port_subscribe_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_subscribe_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_subscribe_free(ptr: *mut snd_seq_port_subscribe_t) -> () {
    type F = unsafe extern "C" fn(ptr: *mut snd_seq_port_subscribe_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_subscribe_free\0")) {
        Some(function) => function(ptr),
        None => (),
    }
}
pub unsafe fn snd_seq_port_subscribe_get_sender(
        info: *const snd_seq_port_subscribe_t,
    ) -> *const snd_seq_addr_t {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_port_subscribe_t,
    ) -> *const snd_seq_addr_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_subscribe_get_sender\0")) {
        Some(function) => function(info),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_seq_port_subscribe_get_dest(
        info: *const snd_seq_port_subscribe_t,
    ) -> *const snd_seq_addr_t {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_port_subscribe_t,
    ) -> *const snd_seq_addr_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_subscribe_get_dest\0")) {
        Some(function) => function(info),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_seq_port_subscribe_get_queue(
        info: *const snd_seq_port_subscribe_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_port_subscribe_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_subscribe_get_queue\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_subscribe_get_exclusive(
        info: *const snd_seq_port_subscribe_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_port_subscribe_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_subscribe_get_exclusive\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_subscribe_get_time_update(
        info: *const snd_seq_port_subscribe_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_port_subscribe_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_subscribe_get_time_update\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_subscribe_get_time_real(
        info: *const snd_seq_port_subscribe_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_port_subscribe_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_subscribe_get_time_real\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_port_subscribe_set_sender(
        info: *mut snd_seq_port_subscribe_t,
        addr: *const snd_seq_addr_t,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_port_subscribe_t,
        addr: *const snd_seq_addr_t,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_subscribe_set_sender\0")) {
        Some(function) => function(info, addr),
        None => (),
    }
}
pub unsafe fn snd_seq_port_subscribe_set_dest(
        info: *mut snd_seq_port_subscribe_t,
        addr: *const snd_seq_addr_t,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_port_subscribe_t,
        addr: *const snd_seq_addr_t,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_subscribe_set_dest\0")) {
        Some(function) => function(info, addr),
        None => (),
    }
}
pub unsafe fn snd_seq_port_subscribe_set_queue(
        info: *mut snd_seq_port_subscribe_t,
        q: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_port_subscribe_t,
        q: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_subscribe_set_queue\0")) {
        Some(function) => function(info, q),
        None => (),
    }
}
pub unsafe fn snd_seq_port_subscribe_set_exclusive(
        info: *mut snd_seq_port_subscribe_t,
        val: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_port_subscribe_t,
        val: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_subscribe_set_exclusive\0")) {
        Some(function) => function(info, val),
        None => (),
    }
}
pub unsafe fn snd_seq_port_subscribe_set_time_update(
        info: *mut snd_seq_port_subscribe_t,
        val: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_port_subscribe_t,
        val: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_subscribe_set_time_update\0")) {
        Some(function) => function(info, val),
        None => (),
    }
}
pub unsafe fn snd_seq_port_subscribe_set_time_real(
        info: *mut snd_seq_port_subscribe_t,
        val: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_port_subscribe_t,
        val: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_port_subscribe_set_time_real\0")) {
        Some(function) => function(info, val),
        None => (),
    }
}
pub unsafe fn snd_seq_subscribe_port(
        handle: *mut snd_seq_t,
        sub: *mut snd_seq_port_subscribe_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        sub: *mut snd_seq_port_subscribe_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_subscribe_port\0")) {
        Some(function) => function(handle, sub),
        None => -1,
    }
}
pub unsafe fn snd_seq_unsubscribe_port(
        handle: *mut snd_seq_t,
        sub: *mut snd_seq_port_subscribe_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        sub: *mut snd_seq_port_subscribe_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_unsubscribe_port\0")) {
        Some(function) => function(handle, sub),
        None => -1,
    }
}
pub unsafe fn snd_seq_query_subscribe_malloc(
        ptr: *mut *mut snd_seq_query_subscribe_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ptr: *mut *mut snd_seq_query_subscribe_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_query_subscribe_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_seq_query_subscribe_free(ptr: *mut snd_seq_query_subscribe_t) -> () {
    type F = unsafe extern "C" fn(ptr: *mut snd_seq_query_subscribe_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_query_subscribe_free\0")) {
        Some(function) => function(ptr),
        None => (),
    }
}
pub unsafe fn snd_seq_query_subscribe_get_index(
        info: *const snd_seq_query_subscribe_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_query_subscribe_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_query_subscribe_get_index\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_query_subscribe_get_addr(
        info: *const snd_seq_query_subscribe_t,
    ) -> *const snd_seq_addr_t {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_query_subscribe_t,
    ) -> *const snd_seq_addr_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_query_subscribe_get_addr\0")) {
        Some(function) => function(info),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_seq_query_subscribe_get_queue(
        info: *const snd_seq_query_subscribe_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_query_subscribe_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_query_subscribe_get_queue\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_query_subscribe_get_exclusive(
        info: *const snd_seq_query_subscribe_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_query_subscribe_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_query_subscribe_get_exclusive\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_query_subscribe_get_time_update(
        info: *const snd_seq_query_subscribe_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_query_subscribe_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_query_subscribe_get_time_update\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_query_subscribe_get_time_real(
        info: *const snd_seq_query_subscribe_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_query_subscribe_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_query_subscribe_get_time_real\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_query_subscribe_set_root(
        info: *mut snd_seq_query_subscribe_t,
        addr: *const snd_seq_addr_t,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_query_subscribe_t,
        addr: *const snd_seq_addr_t,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_query_subscribe_set_root\0")) {
        Some(function) => function(info, addr),
        None => (),
    }
}
pub unsafe fn snd_seq_query_subscribe_set_type(
        info: *mut snd_seq_query_subscribe_t,
        type_: snd_seq_query_subs_type_t,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_query_subscribe_t,
        type_: snd_seq_query_subs_type_t,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_query_subscribe_set_type\0")) {
        Some(function) => function(info, type_),
        None => (),
    }
}
pub unsafe fn snd_seq_query_subscribe_set_index(
        info: *mut snd_seq_query_subscribe_t,
        _index: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_query_subscribe_t,
        _index: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_query_subscribe_set_index\0")) {
        Some(function) => function(info, _index),
        None => (),
    }
}
pub unsafe fn snd_seq_query_port_subscribers(
        seq: *mut snd_seq_t,
        subs: *mut snd_seq_query_subscribe_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        seq: *mut snd_seq_t,
        subs: *mut snd_seq_query_subscribe_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_query_port_subscribers\0")) {
        Some(function) => function(seq, subs),
        None => -1,
    }
}
pub unsafe fn snd_seq_alloc_named_queue(
        seq: *mut snd_seq_t,
        name: *const ::core::ffi::c_char,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        seq: *mut snd_seq_t,
        name: *const ::core::ffi::c_char,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_alloc_named_queue\0")) {
        Some(function) => function(seq, name),
        None => -1,
    }
}
pub unsafe fn snd_seq_alloc_queue(handle: *mut snd_seq_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(handle: *mut snd_seq_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_alloc_queue\0")) {
        Some(function) => function(handle),
        None => -1,
    }
}
pub unsafe fn snd_seq_free_queue(handle: *mut snd_seq_t, q: ::core::ffi::c_int) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(handle: *mut snd_seq_t, q: ::core::ffi::c_int) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_free_queue\0")) {
        Some(function) => function(handle, q),
        None => -1,
    }
}
pub unsafe fn snd_seq_queue_status_sizeof() -> usize {
    type F = unsafe extern "C" fn() -> usize;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_status_sizeof\0")) {
        Some(function) => function(),
        None => 0,
    }
}
pub unsafe fn snd_seq_queue_status_malloc(ptr: *mut *mut snd_seq_queue_status_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ptr: *mut *mut snd_seq_queue_status_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_status_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_seq_queue_status_free(ptr: *mut snd_seq_queue_status_t) -> () {
    type F = unsafe extern "C" fn(ptr: *mut snd_seq_queue_status_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_status_free\0")) {
        Some(function) => function(ptr),
        None => (),
    }
}
pub unsafe fn snd_seq_queue_status_get_queue(
        info: *const snd_seq_queue_status_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_queue_status_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_status_get_queue\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_queue_status_get_events(
        info: *const snd_seq_queue_status_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_queue_status_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_status_get_events\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_queue_status_get_tick_time(
        info: *const snd_seq_queue_status_t,
    ) -> snd_seq_tick_time_t {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_queue_status_t,
    ) -> snd_seq_tick_time_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_status_get_tick_time\0")) {
        Some(function) => function(info),
        None => core::mem::zeroed(),
    }
}
pub unsafe fn snd_seq_queue_status_get_real_time(
        info: *const snd_seq_queue_status_t,
    ) -> *const snd_seq_real_time_t {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_queue_status_t,
    ) -> *const snd_seq_real_time_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_status_get_real_time\0")) {
        Some(function) => function(info),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_seq_queue_status_get_status(
        info: *const snd_seq_queue_status_t,
    ) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_queue_status_t,
    ) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_status_get_status\0")) {
        Some(function) => function(info),
        None => 0,
    }
}
pub unsafe fn snd_seq_get_queue_status(
        handle: *mut snd_seq_t,
        q: ::core::ffi::c_int,
        status: *mut snd_seq_queue_status_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        q: ::core::ffi::c_int,
        status: *mut snd_seq_queue_status_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_get_queue_status\0")) {
        Some(function) => function(handle, q, status),
        None => -1,
    }
}
pub unsafe fn snd_seq_queue_tempo_sizeof() -> usize {
    type F = unsafe extern "C" fn() -> usize;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_tempo_sizeof\0")) {
        Some(function) => function(),
        None => 0,
    }
}
pub unsafe fn snd_seq_queue_tempo_malloc(ptr: *mut *mut snd_seq_queue_tempo_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(ptr: *mut *mut snd_seq_queue_tempo_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_tempo_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_seq_queue_tempo_free(ptr: *mut snd_seq_queue_tempo_t) -> () {
    type F = unsafe extern "C" fn(ptr: *mut snd_seq_queue_tempo_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_tempo_free\0")) {
        Some(function) => function(ptr),
        None => (),
    }
}
pub unsafe fn snd_seq_queue_tempo_get_queue(info: *const snd_seq_queue_tempo_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(info: *const snd_seq_queue_tempo_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_tempo_get_queue\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_queue_tempo_get_tempo(info: *const snd_seq_queue_tempo_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(info: *const snd_seq_queue_tempo_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_tempo_get_tempo\0")) {
        Some(function) => function(info),
        None => 0,
    }
}
pub unsafe fn snd_seq_queue_tempo_get_ppq(info: *const snd_seq_queue_tempo_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(info: *const snd_seq_queue_tempo_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_tempo_get_ppq\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_queue_tempo_get_skew(info: *const snd_seq_queue_tempo_t) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(info: *const snd_seq_queue_tempo_t) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_tempo_get_skew\0")) {
        Some(function) => function(info),
        None => 0,
    }
}
pub unsafe fn snd_seq_queue_tempo_get_skew_base(
        info: *const snd_seq_queue_tempo_t,
    ) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_queue_tempo_t,
    ) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_tempo_get_skew_base\0")) {
        Some(function) => function(info),
        None => 0,
    }
}
pub unsafe fn snd_seq_queue_tempo_set_tempo(
        info: *mut snd_seq_queue_tempo_t,
        tempo: ::core::ffi::c_uint,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_queue_tempo_t,
        tempo: ::core::ffi::c_uint,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_tempo_set_tempo\0")) {
        Some(function) => function(info, tempo),
        None => (),
    }
}
pub unsafe fn snd_seq_queue_tempo_set_ppq(info: *mut snd_seq_queue_tempo_t, ppq: ::core::ffi::c_int) -> () {
    type F = unsafe extern "C" fn(info: *mut snd_seq_queue_tempo_t, ppq: ::core::ffi::c_int) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_tempo_set_ppq\0")) {
        Some(function) => function(info, ppq),
        None => (),
    }
}
pub unsafe fn snd_seq_queue_tempo_set_skew(
        info: *mut snd_seq_queue_tempo_t,
        skew: ::core::ffi::c_uint,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_queue_tempo_t,
        skew: ::core::ffi::c_uint,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_tempo_set_skew\0")) {
        Some(function) => function(info, skew),
        None => (),
    }
}
pub unsafe fn snd_seq_queue_tempo_set_skew_base(
        info: *mut snd_seq_queue_tempo_t,
        base: ::core::ffi::c_uint,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_queue_tempo_t,
        base: ::core::ffi::c_uint,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_queue_tempo_set_skew_base\0")) {
        Some(function) => function(info, base),
        None => (),
    }
}
pub unsafe fn snd_seq_get_queue_tempo(
        handle: *mut snd_seq_t,
        q: ::core::ffi::c_int,
        tempo: *mut snd_seq_queue_tempo_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        q: ::core::ffi::c_int,
        tempo: *mut snd_seq_queue_tempo_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_get_queue_tempo\0")) {
        Some(function) => function(handle, q, tempo),
        None => -1,
    }
}
pub unsafe fn snd_seq_set_queue_tempo(
        handle: *mut snd_seq_t,
        q: ::core::ffi::c_int,
        tempo: *mut snd_seq_queue_tempo_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        q: ::core::ffi::c_int,
        tempo: *mut snd_seq_queue_tempo_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_set_queue_tempo\0")) {
        Some(function) => function(handle, q, tempo),
        None => -1,
    }
}
pub unsafe fn snd_seq_event_output(
        handle: *mut snd_seq_t,
        ev: *mut snd_seq_event_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        ev: *mut snd_seq_event_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_event_output\0")) {
        Some(function) => function(handle, ev),
        None => -1,
    }
}
pub unsafe fn snd_seq_event_output_buffer(
        handle: *mut snd_seq_t,
        ev: *mut snd_seq_event_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        ev: *mut snd_seq_event_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_event_output_buffer\0")) {
        Some(function) => function(handle, ev),
        None => -1,
    }
}
pub unsafe fn snd_seq_event_output_direct(
        handle: *mut snd_seq_t,
        ev: *mut snd_seq_event_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        ev: *mut snd_seq_event_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_event_output_direct\0")) {
        Some(function) => function(handle, ev),
        None => -1,
    }
}
pub unsafe fn snd_seq_event_input(
        handle: *mut snd_seq_t,
        ev: *mut *mut snd_seq_event_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        ev: *mut *mut snd_seq_event_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_event_input\0")) {
        Some(function) => function(handle, ev),
        None => -1,
    }
}
pub unsafe fn snd_seq_event_input_pending(
        seq: *mut snd_seq_t,
        fetch_sequencer: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        seq: *mut snd_seq_t,
        fetch_sequencer: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_event_input_pending\0")) {
        Some(function) => function(seq, fetch_sequencer),
        None => -1,
    }
}
pub unsafe fn snd_seq_drain_output(handle: *mut snd_seq_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(handle: *mut snd_seq_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_drain_output\0")) {
        Some(function) => function(handle),
        None => -1,
    }
}
pub unsafe fn snd_seq_drop_output(handle: *mut snd_seq_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(handle: *mut snd_seq_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_drop_output\0")) {
        Some(function) => function(handle),
        None => -1,
    }
}
pub unsafe fn snd_seq_drop_input(handle: *mut snd_seq_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(handle: *mut snd_seq_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_drop_input\0")) {
        Some(function) => function(handle),
        None => -1,
    }
}
pub unsafe fn snd_seq_remove_events_malloc(
        ptr: *mut *mut snd_seq_remove_events_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        ptr: *mut *mut snd_seq_remove_events_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events_malloc\0")) {
        Some(function) => function(ptr),
        None => -1,
    }
}
pub unsafe fn snd_seq_remove_events_free(ptr: *mut snd_seq_remove_events_t) -> () {
    type F = unsafe extern "C" fn(ptr: *mut snd_seq_remove_events_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events_free\0")) {
        Some(function) => function(ptr),
        None => (),
    }
}
pub unsafe fn snd_seq_remove_events_get_condition(
        info: *const snd_seq_remove_events_t,
    ) -> ::core::ffi::c_uint {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_remove_events_t,
    ) -> ::core::ffi::c_uint;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events_get_condition\0")) {
        Some(function) => function(info),
        None => 0,
    }
}
pub unsafe fn snd_seq_remove_events_get_queue(
        info: *const snd_seq_remove_events_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_remove_events_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events_get_queue\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_remove_events_get_time(
        info: *const snd_seq_remove_events_t,
    ) -> *const snd_seq_timestamp_t {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_remove_events_t,
    ) -> *const snd_seq_timestamp_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events_get_time\0")) {
        Some(function) => function(info),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_seq_remove_events_get_dest(
        info: *const snd_seq_remove_events_t,
    ) -> *const snd_seq_addr_t {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_remove_events_t,
    ) -> *const snd_seq_addr_t;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events_get_dest\0")) {
        Some(function) => function(info),
        None => core::ptr::null(),
    }
}
pub unsafe fn snd_seq_remove_events_get_channel(
        info: *const snd_seq_remove_events_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_remove_events_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events_get_channel\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_remove_events_get_event_type(
        info: *const snd_seq_remove_events_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_remove_events_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events_get_event_type\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_remove_events_get_tag(
        info: *const snd_seq_remove_events_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        info: *const snd_seq_remove_events_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events_get_tag\0")) {
        Some(function) => function(info),
        None => -1,
    }
}
pub unsafe fn snd_seq_remove_events_set_condition(
        info: *mut snd_seq_remove_events_t,
        flags: ::core::ffi::c_uint,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_remove_events_t,
        flags: ::core::ffi::c_uint,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events_set_condition\0")) {
        Some(function) => function(info, flags),
        None => (),
    }
}
pub unsafe fn snd_seq_remove_events_set_queue(
        info: *mut snd_seq_remove_events_t,
        queue: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_remove_events_t,
        queue: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events_set_queue\0")) {
        Some(function) => function(info, queue),
        None => (),
    }
}
pub unsafe fn snd_seq_remove_events_set_time(
        info: *mut snd_seq_remove_events_t,
        time: *const snd_seq_timestamp_t,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_remove_events_t,
        time: *const snd_seq_timestamp_t,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events_set_time\0")) {
        Some(function) => function(info, time),
        None => (),
    }
}
pub unsafe fn snd_seq_remove_events_set_dest(
        info: *mut snd_seq_remove_events_t,
        addr: *const snd_seq_addr_t,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_remove_events_t,
        addr: *const snd_seq_addr_t,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events_set_dest\0")) {
        Some(function) => function(info, addr),
        None => (),
    }
}
pub unsafe fn snd_seq_remove_events_set_channel(
        info: *mut snd_seq_remove_events_t,
        channel: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_remove_events_t,
        channel: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events_set_channel\0")) {
        Some(function) => function(info, channel),
        None => (),
    }
}
pub unsafe fn snd_seq_remove_events_set_event_type(
        info: *mut snd_seq_remove_events_t,
        type_: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_remove_events_t,
        type_: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events_set_event_type\0")) {
        Some(function) => function(info, type_),
        None => (),
    }
}
pub unsafe fn snd_seq_remove_events_set_tag(
        info: *mut snd_seq_remove_events_t,
        tag: ::core::ffi::c_int,
    ) -> () {
    type F = unsafe extern "C" fn(
        info: *mut snd_seq_remove_events_t,
        tag: ::core::ffi::c_int,
    ) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events_set_tag\0")) {
        Some(function) => function(info, tag),
        None => (),
    }
}
pub unsafe fn snd_seq_remove_events(
        handle: *mut snd_seq_t,
        info: *mut snd_seq_remove_events_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        handle: *mut snd_seq_t,
        info: *mut snd_seq_remove_events_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_remove_events\0")) {
        Some(function) => function(handle, info),
        None => -1,
    }
}
pub unsafe fn snd_seq_control_queue(
        seq: *mut snd_seq_t,
        q: ::core::ffi::c_int,
        type_: ::core::ffi::c_int,
        value: ::core::ffi::c_int,
        ev: *mut snd_seq_event_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        seq: *mut snd_seq_t,
        q: ::core::ffi::c_int,
        type_: ::core::ffi::c_int,
        value: ::core::ffi::c_int,
        ev: *mut snd_seq_event_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_control_queue\0")) {
        Some(function) => function(seq, q, type_, value, ev),
        None => -1,
    }
}
pub unsafe fn snd_seq_create_simple_port(
        seq: *mut snd_seq_t,
        name: *const ::core::ffi::c_char,
        caps: ::core::ffi::c_uint,
        type_: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        seq: *mut snd_seq_t,
        name: *const ::core::ffi::c_char,
        caps: ::core::ffi::c_uint,
        type_: ::core::ffi::c_uint,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_create_simple_port\0")) {
        Some(function) => function(seq, name, caps, type_),
        None => -1,
    }
}
pub unsafe fn snd_seq_set_client_name(
        seq: *mut snd_seq_t,
        name: *const ::core::ffi::c_char,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        seq: *mut snd_seq_t,
        name: *const ::core::ffi::c_char,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_set_client_name\0")) {
        Some(function) => function(seq, name),
        None => -1,
    }
}
pub unsafe fn snd_seq_set_client_event_filter(
        seq: *mut snd_seq_t,
        event_type: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        seq: *mut snd_seq_t,
        event_type: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_set_client_event_filter\0")) {
        Some(function) => function(seq, event_type),
        None => -1,
    }
}
pub unsafe fn snd_seq_set_client_pool_output(seq: *mut snd_seq_t, size: usize) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(seq: *mut snd_seq_t, size: usize) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_set_client_pool_output\0")) {
        Some(function) => function(seq, size),
        None => -1,
    }
}
pub unsafe fn snd_seq_set_client_pool_output_room(
        seq: *mut snd_seq_t,
        size: usize,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        seq: *mut snd_seq_t,
        size: usize,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_set_client_pool_output_room\0")) {
        Some(function) => function(seq, size),
        None => -1,
    }
}
pub unsafe fn snd_seq_set_client_pool_input(seq: *mut snd_seq_t, size: usize) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(seq: *mut snd_seq_t, size: usize) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_set_client_pool_input\0")) {
        Some(function) => function(seq, size),
        None => -1,
    }
}
pub unsafe fn snd_seq_sync_output_queue(seq: *mut snd_seq_t) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(seq: *mut snd_seq_t) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_seq_sync_output_queue\0")) {
        Some(function) => function(seq),
        None => -1,
    }
}
pub unsafe fn snd_midi_event_new(
        bufsize: usize,
        rdev: *mut *mut snd_midi_event_t,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        bufsize: usize,
        rdev: *mut *mut snd_midi_event_t,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_midi_event_new\0")) {
        Some(function) => function(bufsize, rdev),
        None => -1,
    }
}
pub unsafe fn snd_midi_event_resize_buffer(
        dev: *mut snd_midi_event_t,
        bufsize: usize,
    ) -> ::core::ffi::c_int {
    type F = unsafe extern "C" fn(
        dev: *mut snd_midi_event_t,
        bufsize: usize,
    ) -> ::core::ffi::c_int;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_midi_event_resize_buffer\0")) {
        Some(function) => function(dev, bufsize),
        None => -1,
    }
}
pub unsafe fn snd_midi_event_free(dev: *mut snd_midi_event_t) -> () {
    type F = unsafe extern "C" fn(dev: *mut snd_midi_event_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_midi_event_free\0")) {
        Some(function) => function(dev),
        None => (),
    }
}
pub unsafe fn snd_midi_event_init(dev: *mut snd_midi_event_t) -> () {
    type F = unsafe extern "C" fn(dev: *mut snd_midi_event_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_midi_event_init\0")) {
        Some(function) => function(dev),
        None => (),
    }
}
pub unsafe fn snd_midi_event_reset_encode(dev: *mut snd_midi_event_t) -> () {
    type F = unsafe extern "C" fn(dev: *mut snd_midi_event_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_midi_event_reset_encode\0")) {
        Some(function) => function(dev),
        None => (),
    }
}
pub unsafe fn snd_midi_event_reset_decode(dev: *mut snd_midi_event_t) -> () {
    type F = unsafe extern "C" fn(dev: *mut snd_midi_event_t) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_midi_event_reset_decode\0")) {
        Some(function) => function(dev),
        None => (),
    }
}
pub unsafe fn snd_midi_event_no_status(dev: *mut snd_midi_event_t, on: ::core::ffi::c_int) -> () {
    type F = unsafe extern "C" fn(dev: *mut snd_midi_event_t, on: ::core::ffi::c_int) -> ();
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_midi_event_no_status\0")) {
        Some(function) => function(dev, on),
        None => (),
    }
}
pub unsafe fn snd_midi_event_encode(
        dev: *mut snd_midi_event_t,
        buf: *const ::core::ffi::c_uchar,
        count: ::core::ffi::c_long,
        ev: *mut snd_seq_event_t,
    ) -> ::core::ffi::c_long {
    type F = unsafe extern "C" fn(
        dev: *mut snd_midi_event_t,
        buf: *const ::core::ffi::c_uchar,
        count: ::core::ffi::c_long,
        ev: *mut snd_seq_event_t,
    ) -> ::core::ffi::c_long;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_midi_event_encode\0")) {
        Some(function) => function(dev, buf, count, ev),
        None => 0,
    }
}
pub unsafe fn snd_midi_event_decode(
        dev: *mut snd_midi_event_t,
        buf: *mut ::core::ffi::c_uchar,
        count: ::core::ffi::c_long,
        ev: *const snd_seq_event_t,
    ) -> ::core::ffi::c_long {
    type F = unsafe extern "C" fn(
        dev: *mut snd_midi_event_t,
        buf: *mut ::core::ffi::c_uchar,
        count: ::core::ffi::c_long,
        ev: *const snd_seq_event_t,
    ) -> ::core::ffi::c_long;
    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();
    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b"snd_midi_event_decode\0")) {
        Some(function) => function(dev, buf, count, ev),
        None => 0,
    }
}
