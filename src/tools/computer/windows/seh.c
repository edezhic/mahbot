/*
 * The one structured-exception frame in the product: Rust cannot catch a structured
 * exception, so the Windows computer-use backend owns one here, in C. A caught
 * exception is deliberately not continued — the code that raised it cannot resume —
 * so the caller is told its code; what it must do about that is `seh.rs`'s contract.
 *
 * 64-bit only: no 32-bit Windows target is built, and the 32-bit C++ exception layout
 * is a different walk this file will not carry unexercised.
 */
#include <windows.h>
#include <string.h>

#if !defined(_WIN64)
#error "seh.c decodes the 64-bit C++ exception layout only; no 32-bit Windows target is built"
#endif

/* The code MSVC's C++ runtime throws with. */
#define MB_SEH_CPP_EXCEPTION 0xE06D7363ul
/* The code libunwind raises on windows-gnu, for C++ throws and Rust panics alike. Its
 * phase-2 and forced unwinds are different codes that carry a zero severity, so the
 * filter's default arm already leaves them alone. */
#define MB_SEH_GCC_THROW 0x20474343ul
/* A code is a failure only when its severity bits say `NT_ERROR`. Everything else is
 * control flow or a status message (a debugger print, a single step, a longjmp) that
 * the frame must not turn into a failed call. */
#define MB_SEH_IS_ERROR(code) ((((code) & 0xC0000000ul)) == 0xC0000000ul)
/* The one error status the frame passes through: reporting it allocates and logs, and a
 * thread that has run out of stack cannot do either. */
#define MB_SEH_STACK_OVERFLOW 0xC00000FDul

/* `EH_MAGIC_NUMBER1..3` (`ehdata.h`): the first exception parameter of every C++
 * throw. */
#define MB_EH_MAGIC_1 0x19930520ul
#define MB_EH_MAGIC_2 0x19930521ul
#define MB_EH_MAGIC_3 0x19930522ul

/* Rust's panic runtime throws a type under this name, which is what its own catch
 * matches on (`library/panic_unwind/src/seh.rs`); a foreign C++ throw carries another. */
static const char MB_RUST_PANIC[] = "rust_panic";

/*
 * The parts of the MSVC C++ exception layout this filter reads (`ehdata.h`). The
 * fields inside these records are 32-bit offsets from the image base the exception
 * record carries as its last parameter.
 */
typedef struct _MB_TypeDescriptor {
    const void *pVFTable;
    void *spare;
    char name[1];
} MB_TypeDescriptor;

typedef struct _MB_CatchableType {
    unsigned int properties;
    unsigned int pType; /* image-relative MB_TypeDescriptor */
} MB_CatchableType;

typedef struct _MB_CatchableTypeArray {
    int nCatchableTypes;
    unsigned int arrayOfCatchableTypes[1]; /* image-relative MB_CatchableType */
} MB_CatchableTypeArray;

typedef struct _MB_ThrowInfo {
    unsigned int attributes;
    unsigned int pmfnUnwind;          /* image-relative */
    unsigned int pForwardCompat;      /* image-relative */
    unsigned int pCatchableTypeArray; /* image-relative MB_CatchableTypeArray */
} MB_ThrowInfo;

/* The classes libstdc++ ("GNUCC++") and libc++abi ("CLNGC++") stamp on the exception
 * object they raise. They are numbers, not text — a runtime builds the tag with shifts,
 * so on little-endian Windows the bytes in memory read backwards and comparing the tag
 * as text matches nothing. The `\1` variants are the runtimes' "dependent" (rethrown)
 * classes; Rust raises neither. */
static const unsigned long long MB_FOREIGN_CLASSES[] = {
    0x474E5543432B2B00ull, /* libstdc++ "GNUCC++\0" */
    0x474E5543432B2B01ull, /* libstdc++ "GNUCC++\1" */
    0x434C4E47432B2B00ull, /* libc++abi "CLNGC++\0" */
    0x434C4E47432B2B01ull, /* libc++abi "CLNGC++\1" */
};

/*
 * Whether an exception is a *foreign* C++ throw — the one class of unwind this frame
 * may own — or something it must leave alone.
 *
 * A Rust panic travels as a C++-style unwind too, so it is the exception that has to
 * be recognized, and it is: MSVC panics carry the `rust_panic` type name, windows-gnu
 * ones libgcc's Rust exception class. Only a record whose shape this filter reads as a
 * foreign throw is contained; a record whose shape it does not read keeps propagating.
 * The shape is all that is checked — a recognized record whose inner offsets do not
 * resolve would fault here, inside a filter where no frame can catch it.
 */
static int
mb_is_foreign_cpp(const EXCEPTION_RECORD *rec)
{
    if (rec->ExceptionCode == MB_SEH_CPP_EXCEPTION && rec->NumberParameters == 4) {
        const MB_ThrowInfo *info;
        const MB_CatchableTypeArray *types;
        const MB_CatchableType *catchable;
        const MB_TypeDescriptor *type;
        ULONG_PTR magic = rec->ExceptionInformation[0];
        ULONG_PTR base = rec->ExceptionInformation[3];
        if (magic != MB_EH_MAGIC_1 && magic != MB_EH_MAGIC_2 && magic != MB_EH_MAGIC_3) {
            return 0;
        }
        info = (const MB_ThrowInfo *)rec->ExceptionInformation[2];
        if (info == 0 || base == 0) {
            return 0;
        }
        types = (const MB_CatchableTypeArray *)(base + info->pCatchableTypeArray);
        if (types->nCatchableTypes < 1) {
            return 0;
        }
        catchable = (const MB_CatchableType *)(base + types->arrayOfCatchableTypes[0]);
        type = (const MB_TypeDescriptor *)(base + catchable->pType);
        return strncmp(type->name, MB_RUST_PANIC, sizeof MB_RUST_PANIC) != 0;
    }
    if (rec->ExceptionCode == MB_SEH_GCC_THROW && rec->NumberParameters >= 1) {
        /* The exception object begins with the class the runtime gave it; Rust's is
         * `u64::from_ne_bytes(*b"MOZ\0RUST")` (`panic_unwind/src/gcc.rs`), which is in
         * none of the foreign classes above. */
        const unsigned long long *class =
            (const unsigned long long *)rec->ExceptionInformation[0];
        size_t i;
        if (class == 0) {
            return 0;
        }
        for (i = 0; i < sizeof MB_FOREIGN_CLASSES / sizeof MB_FOREIGN_CLASSES[0]; i++) {
            if (*class == MB_FOREIGN_CLASSES[i]) {
                return 1;
            }
        }
        return 0;
    }
    return 0;
}

/*
 * The frame's filter. What it owns is a fault and a C++ throw a foreign component
 * raised; everything else keeps travelling — a Rust panic to Rust's handler, a
 * control-flow code to whoever raised it, a stack overflow to the process.
 */
static int
mb_seh_filter(EXCEPTION_POINTERS *info)
{
    EXCEPTION_RECORD *rec = info->ExceptionRecord;
    switch (rec->ExceptionCode) {
    case MB_SEH_CPP_EXCEPTION:
    case MB_SEH_GCC_THROW:
        return mb_is_foreign_cpp(rec) ? EXCEPTION_EXECUTE_HANDLER
                                      : EXCEPTION_CONTINUE_SEARCH;
    case MB_SEH_STACK_OVERFLOW:
        return EXCEPTION_CONTINUE_SEARCH;
    default:
        return MB_SEH_IS_ERROR(rec->ExceptionCode) ? EXCEPTION_EXECUTE_HANDLER
                                                  : EXCEPTION_CONTINUE_SEARCH;
    }
}

/* The body runs for its effect; its value is not part of the contract. */
typedef void (*mb_seh_body)(void *ctx);

/*
 * Run `body(ctx)` under the frame. Returns 0 when the body returned, 1 when an
 * exception was caught, and stores that exception's code in `*code`.
 */
int
mb_seh_guard(mb_seh_body body, void *ctx, unsigned long *code)
{
    __try {
        body(ctx);
    }
    __except (mb_seh_filter(GetExceptionInformation())) {
        *code = GetExceptionCode();
        return 1;
    }
    return 0;
}
