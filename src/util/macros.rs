//! Declarative macros for column-index constant generation.
//!
//! The [`crate::columns!`] macro eliminates the brittle hand-maintained coupling
//! between a `*_COLUMNS` SQL column-string and its `COL_*` positional-index
//! constants. Instead of 3–4 separate locations to update per column set,
//! a single `columns!` invocation serves as the single source of truth.

// ---------------------------------------------------------------------------
// Helper: join literals with ", " — used internally by columns!
// ---------------------------------------------------------------------------

/// Join literal strings with `", "` separator.
///
/// ```ignore
/// assert_eq!(__columns_join!("a", "b", "c"), "a, b, c");
/// ```
#[macro_export]
macro_rules! __columns_join {
    ($first:literal $(, $rest:literal)* $(,)?) => {
        concat!($first $(, ", ", $rest)*)
    };
}

// ---------------------------------------------------------------------------
// Helper: generate COL_* index constants — used internally by columns!
// ---------------------------------------------------------------------------

/// Generate `COL_{prefix}_{name}` index constants with positional indices.
///
/// Recursively processes each column identifier, emitting a `const COL_{P}_{N}: usize`
/// with the appropriate zero-based position.
#[macro_export]
macro_rules! __columns_gen {
    // Terminal case: last column
    ($vis:vis $prefix:ident $n:expr, $col:ident) => {
        ::paste::paste! {
            $vis const [<COL_ $prefix _ $col>]: usize = $n;
        }
    };
    // Recursive case: emit one constant, then process the rest with incremented index
    ($vis:vis $prefix:ident $n:expr, $col:ident, $($rest:ident),+) => {
        ::paste::paste! {
            $vis const [<COL_ $prefix _ $col>]: usize = $n;
        }
        $crate::__columns_gen!($vis $prefix $n + 1usize, $($rest),+);
    };
}

// ---------------------------------------------------------------------------
// Public macro: columns!
// ---------------------------------------------------------------------------

/// Generate a column-string constant and matching column-index constants from
/// a single source-of-truth list.
///
/// # Syntax
///
/// ```ignore
/// columns! {
///     /// Optional doc comment (attached to the column string constant).
///     COLUMNS_NAME [PREFIX] {
///         FIELD_NAME => "sql_column_expression",
///         ANOTHER   => "another_column",
///     }
/// }
/// ```
///
/// # Expansion
///
/// For input
/// ```ignore
/// columns! {
///     pub(crate) const MY_COLUMNS [mc] {
///         FOO => "foo",
///         BAR => "bar",
///     }
/// }
/// ```
///
/// expands to:
/// - `pub(crate) const MY_COLUMNS: &str = "foo, bar";`
/// - `const COL_MC_FOO: usize = 0;`
/// - `const COL_MC_BAR: usize = 1;`
///
/// # Expression overrides
///
/// The `=> "..."` syntax accepts any SQL expression as a string literal, so
/// complex expressions like `"json_each.value AS error"`, `"COUNT(s.id)"`,
/// or `"sm.session_key"` are fully supported.
#[macro_export]
macro_rules! columns {
    (
        $(#[$attr:meta])*
        $vis:vis $name:ident [$prefix:ident] {
            $($col:ident => $sql:literal),+ $(,)?
        }
    ) => {
        $(#[$attr])*
        $vis const $name: &str = $crate::__columns_join!($($sql),+);

        $crate::__columns_gen!($vis $prefix 0usize, $($col),+);
    };
}

// ---------------------------------------------------------------------------
// Declarative macro: define_store!
// ---------------------------------------------------------------------------

/// Generate a DB-backed store struct, its `open()` constructor, and a global
/// singleton (via [`crate::global_store!`]).
///
/// Eliminates ~64 lines of boilerplate per store module.
///
/// **Form 1 — simple (no post-open step):**
/// ```ignore
/// define_store! {
///     /// Doc comment for the global static.
///     pub static STORE_NAME: StoreType,
///     db_name = "db_file_name",
///     schema = SCHEMA,
///     // optional: expect = "custom panic message",
/// }
/// ```
///
/// **Form 2 — post-open via `&self` method:**
/// ```ignore
/// define_store! {
///     pub static USER_STORE: UserStore,
///     db_name = "users",
///     schema = SCHEMA,
///     post_open = ensure_admin_user,
/// }
/// ```
/// The method is called via `this.$method().await?` after store construction.
/// It must be `async fn(&self) -> anyhow::Result<()>`.  Defined in a separate
/// `impl Store { … }` block.
///
/// # Generated items
///
/// For both forms the macro generates:
/// - `#[derive(Clone, Debug)] pub struct $Store { pub(crate) conn: Connection }`
/// - `impl $Store { pub async fn open(root: &Path) -> anyhow::Result<Self> { … } }`
/// - A `global_store!` invocation creating the `OnceCell`, `init_global()`, and
///   `store()` singleton accessor
///
/// This macro is intentionally limited to these two forms.  An arbitrary-block
/// form is **not** provided because Rust `macro_rules!` hygiene prevents
/// user-provided `self` / `conn` tokens inside generated method bodies.
/// The `init`-method approach (Form 2) avoids this limitation entirely.
#[macro_export]
macro_rules! define_store {
    // ── Form 1: Simple (no post-open, auto expect) ──────────────────────
    (
        $(#[$attr:meta])*
        pub static $name:ident: $ty:ident,
        db_name = $db_name:literal,
        schema = $schema:ident,
    ) => {
        $crate::define_store! {
            $(#[$attr])*
            pub static $name: $ty,
            db_name = $db_name,
            schema = $schema,
            expect = concat!(
                stringify!($name),
                " not initialized — call init_global() first"
            ),
        }
    };

    // ── Form 1b: Simple with custom expect ──────────────────────────────
    (
        $(#[$attr:meta])*
        pub static $name:ident: $ty:ident,
        db_name = $db_name:literal,
        schema = $schema:ident,
        expect = $expect:expr,
    ) => {
        $(#[$attr])*
        #[derive(Clone, Debug)]
        pub struct $ty {
            pub(crate) conn: $crate::turso::Connection,
        }

        impl $ty {
            /// Open (or create) the database at `root/db/{name}.db`.
            pub async fn open(
                root: &std::path::Path,
            ) -> ::anyhow::Result<Self> {
                let conn = $crate::turso::open_store(root, $db_name, $schema).await?;
                Ok(Self { conn })
            }
        }

        $crate::global_store! {
            $(#[$attr])*
            pub static $name: $ty,
            constructor = $ty::open,
            expect = $expect,
        }
    };

    // ── Form 2: Post-open via init method (auto expect) ─────────────────
    (
        $(#[$attr:meta])*
        pub static $name:ident: $ty:ident,
        db_name = $db_name:literal,
        schema = $schema:ident,
        post_open = $method:ident,
    ) => {
        $crate::define_store! {
            $(#[$attr])*
            pub static $name: $ty,
            db_name = $db_name,
            schema = $schema,
            post_open = $method,
            expect = concat!(
                stringify!($name),
                " not initialized — call init_global() first"
            ),
        }
    };

    // ── Form 2b: Post-open via init method with custom expect ───────────
    (
        $(#[$attr:meta])*
        pub static $name:ident: $ty:ident,
        db_name = $db_name:literal,
        schema = $schema:ident,
        post_open = $method:ident,
        expect = $expect:expr,
    ) => {
        $(#[$attr])*
        #[derive(Clone, Debug)]
        pub struct $ty {
            pub(crate) conn: $crate::turso::Connection,
        }

        impl $ty {
            /// Open (or create) the database at `root/db/{name}.db`.
            pub async fn open(
                root: &std::path::Path,
            ) -> ::anyhow::Result<Self> {
                let conn = $crate::turso::open_store(root, $db_name, $schema).await?;
                let this = Self { conn };
                this.$method().await?;
                Ok(this)
            }
        }

        $crate::global_store! {
            $(#[$attr])*
            pub static $name: $ty,
            constructor = $ty::open,
            expect = $expect,
        }
    };
}
