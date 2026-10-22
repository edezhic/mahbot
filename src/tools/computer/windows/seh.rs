//! Structured-exception containment for the Windows backend: [`guard`] runs one
//! closure inside the C frame in `seh.c`, and reports an exception raised inside it
//! as `Err(code)` instead of letting the process die.
//!
//! The frame has to be C (Rust cannot catch a structured exception), and it is the
//! only such frame in the product. It owns every fault and every C++ exception a
//! foreign component throws across the boundary. Three things it cannot cover are
//! stated here rather than worked around:
//!
//! * A fault raised on a driver's own internal thread never passes this frame: the
//!   frame covers only the thread that called it, so nothing in this process can
//!   catch one raised elsewhere.
//! * A Rust panic passes through by design. It travels as a C++-style unwind too, and
//!   containing one would stop the panic before Rust's handler could run. The shim
//!   tells the two apart by the identity the panic runtime gives its throw (the type
//!   name on MSVC, the exception class on windows-gnu), and only contains a record
//!   whose shape it reads as a foreign throw.
//! * A code that is not an error status passes through, because it is control flow
//!   rather than failure (a debugger print, a single step, a longjmp), and so does a
//!   stack overflow: reporting one allocates and logs, which a thread with no stack
//!   left must not be asked to do.
//!
//! A fourth limit lives where it happens, in `seh.c`'s own record walk: the filter
//! trusts the offsets a recognized throw record carries, and a filter has no frame of
//! its own to catch a bad one.

use std::ffi::c_void;

// `int mb_seh_guard(void (*body)(void *), void *ctx, unsigned long *code)` from
// `seh.c`; its `unsigned long` is 32 bits on Windows, hence `u32`.
//
// `C-unwind` on both the function and the body so a Rust panic unwinding out of the
// body keeps unwinding through this frame to Rust's own handler; the calling
// convention is identical to `C`.
unsafe extern "C-unwind" {
    fn mb_seh_guard(
        body: extern "C-unwind" fn(*mut c_void),
        ctx: *mut c_void,
        code: *mut u32,
    ) -> i32;
}

// ── The guard ───────────────────────────────────────────────────────────

/// What one guarded call leaves the frame with: the body until the frame runs it,
/// then the value it returned.
struct Slot<F, T> {
    body: Option<F>,
    out: Option<T>,
}

/// The one function the C frame calls back: run the slot's body and keep its value.
extern "C-unwind" fn run<F: FnOnce() -> T, T>(ctx: *mut c_void) {
    // SAFETY: `guard` hands over the address of its own live `Slot<F, T>` for
    // exactly the duration of the `mb_seh_guard` call, and this is the only place
    // that pointer is used; the slot outlives the reference made here.
    let slot = unsafe { &mut *ctx.cast::<Slot<F, T>>() };
    let body = slot
        .body
        .take()
        .expect("the frame calls the body once, after `guard` filled the slot");
    slot.out = Some(body());
}

/// Run `body` inside a structured-exception frame; `Err(code)` is an exception it
/// raised.
///
/// Whatever the body was mutating is unknowable once an exception is caught, so the
/// caller must fail the call and never reuse what the body owned.
pub(super) fn guard<T, F: FnOnce() -> T>(body: F) -> Result<T, u32> {
    let mut slot = Slot {
        body: Some(body),
        out: None,
    };
    let mut code = 0u32;
    // SAFETY: `seh.c` only calls `run` with the context it was given, the two
    // out-parameters are live locals across the whole call, and `run::<F, T>` has
    // exactly the `void (void *)` shape the shim's body parameter declares.
    let caught =
        unsafe { mb_seh_guard(run::<F, T>, (&raw mut slot).cast::<c_void>(), &raw mut code) };
    if caught == 0 {
        // A frame that returned normally ran the body, and `run` is the only writer of
        // `out`; anything else would be a broken shim, not an exception to report as one.
        Ok(slot
            .out
            .expect("a returning frame always stored the body's value"))
    } else {
        Err(code)
    }
}

#[cfg(test)]
mod tests {
    use super::guard;

    /// Containment checked against a real fault: that the shim returns at all is the
    /// whole claim, so this test dies with the process rather than failing if the
    /// frame is ever broken.
    #[test]
    fn a_fault_inside_the_guard_is_caught() {
        // A null read is the cheapest real fault.
        let caught = guard(|| unsafe { std::ptr::read_volatile(std::ptr::null::<u8>()) });
        assert!(caught.is_err(), "the shim must report the fault");
        // Still here — which is the whole claim.
        assert_eq!(guard(|| 42), Ok(42));
    }

    /// A C++ exception raised by foreign code while the call runs is a failed call,
    /// not the end of the process.
    ///
    /// The record is built by hand in the MSVC ABI's shape — magic number, object,
    /// throw info, image base, with the throw info's own fields as image-relative
    /// offsets — so no C++ compiler is needed to raise one.
    #[cfg(all(target_env = "msvc", target_pointer_width = "64"))]
    #[test]
    fn a_foreign_cpp_exception_becomes_a_fault() {
        #[repr(C)]
        struct ThrowInfo {
            attributes: u32,
            unwind: u32,
            forward_compat: u32,
            catchable_types: u32,
        }
        #[repr(C)]
        struct CatchableTypes {
            count: i32,
            first: u32,
        }
        #[repr(C)]
        struct CatchableType {
            properties: u32,
            descriptor: u32,
        }
        #[repr(C)]
        struct TypeDescriptor {
            vtable: usize,
            spare: usize,
            name: [u8; 11],
        }

        unsafe extern "system" {
            fn RaiseException(code: u32, flags: u32, count: u32, arguments: *const usize);
        }

        let descriptor = TypeDescriptor {
            vtable: 0,
            spare: 0,
            name: *b"foreign_ex\0",
        };
        let mut catchable = CatchableType {
            properties: 0,
            descriptor: 0,
        };
        let mut types = CatchableTypes { count: 1, first: 0 };
        let mut info = ThrowInfo {
            attributes: 0,
            unwind: 0,
            forward_compat: 0,
            catchable_types: 0,
        };
        let thrown = 0x2A_u64;
        let addresses = [
            std::ptr::from_ref(&descriptor).addr(),
            std::ptr::from_ref(&catchable).addr(),
            std::ptr::from_ref(&types).addr(),
            std::ptr::from_ref(&info).addr(),
        ];
        // The record's image base is the lowest of the fabricated records, so every
        // offset it carries is small and the addresses resolve as the ABI expects.
        let base = addresses.into_iter().min().unwrap();
        let offset = |address: usize| u32::try_from(address - base).unwrap();
        catchable.descriptor = offset(std::ptr::from_ref(&descriptor).addr());
        types.first = offset(std::ptr::from_ref(&catchable).addr());
        info.catchable_types = offset(std::ptr::from_ref(&types).addr());
        let arguments = [
            0x1993_0520usize, // EH_MAGIC_NUMBER1
            std::ptr::from_ref(&thrown).addr(),
            std::ptr::from_ref(&info).addr(),
            base,
        ];
        let caught = guard(|| unsafe { RaiseException(0xE06D_7363, 0, 4, arguments.as_ptr()) });
        assert_eq!(caught.unwrap_err(), 0xE06D_7363);
    }

    /// The same route on windows-gnu, where one code carries every unwind and the
    /// exception object's class is what names the runtime that raised it.
    ///
    /// libstdc++ stores that class as the number `0x474E5543432B2B00` — the tag read as
    /// a big-endian integer — so the object here carries exactly what the runtime puts
    /// in memory on a little-endian target, not the tag's own byte order.
    #[cfg(not(target_env = "msvc"))]
    #[test]
    fn a_foreign_cpp_exception_becomes_a_fault() {
        unsafe extern "system" {
            fn RaiseException(code: u32, flags: u32, count: u32, arguments: *const usize);
        }

        // `from_be_bytes`: what libstdc++'s numeric class constant puts in memory.
        let object: [u64; 1] = [u64::from_be_bytes(*b"GNUCC++\0")];
        let arguments = [object.as_ptr().addr()];
        let caught = guard(|| unsafe { RaiseException(0x2047_4343, 0, 1, arguments.as_ptr()) });
        assert_eq!(caught.unwrap_err(), 0x2047_4343);
    }

    /// A Rust panic must keep travelling to Rust's own handler: the frame lets its
    /// record pass, and containing a panic instead would stop it before the runtime
    /// could run it.
    #[test]
    fn a_rust_panic_passes_through_the_frame() {
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            guard(|| panic!("raised inside the frame"))
        }));
        assert!(
            caught.is_err(),
            "the panic must reach Rust's handler, not become a fault"
        );
    }
}
