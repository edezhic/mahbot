//! Declarative macros for column-index constant generation.
//!
//! The [`crate::columns!`] macro eliminates the brittle hand-maintained coupling
//! between a `*_COLUMNS` SQL column-string and its `COL_*` positional-index
//! constants. Instead of 3–4 separate locations to update per column set,
//! a single `columns!` invocation serves as the single source of truth.

// Helper: join literals with ", " — used internally by columns!

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

// Helper: generate COL_* index constants — used internally by columns!

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

// Public macro: columns!

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
/// or `"sm.agent_id"` are fully supported.
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

// Declarative macro: define_store!

/// Generate a DB-backed store struct, its `open()` constructor, and a global
/// singleton accessor (`store()`).
///
/// Eliminates ~64 lines of boilerplate per store module.
///
/// # Syntax
///
/// ```ignore
/// define_store! {
///     /// Doc comment for the global static.
///     pub static STORE_NAME: StoreType,
///     post_open = ensure_admin_user,  // optional; omitted when not needed
///     expect = "custom panic message",
/// }
/// ```
///
/// The `post_open` field is optional.  When present, it names an
/// `async fn(&self) -> anyhow::Result<()>` method that is called after
/// the database connection is established (via `this.$method().await?`)
/// but before the store is returned.  The method must be defined in a
/// separate `impl Store { … }` block.
///
/// # Generated items
///
/// The macro generates:
/// - `#[derive(Clone, Debug)] pub struct $Store { pub(crate) conn: Connection }`
/// - `impl $Store { pub async fn open(root: &Path) -> anyhow::Result<Self> { … } }`
/// - A `static` [`OnceCell`] plus a `store()` accessor returning `&'static $Store`
///
/// # Consolidated layout
///
/// The macro no longer takes `db_name`/`schema`: after consolidation every
/// domain store opens the ONE consolidated database file via
/// [`crate::turso::open_consolidated_store`]. The per-module `SCHEMA` const is
/// still declared (and referenced directly by
/// [`crate::turso::consolidated_schema`]) but is no longer threaded through
/// this macro.
///
/// An arbitrary-block form is **not** provided because Rust `macro_rules!`
/// hygiene prevents user-provided `self` / `conn` tokens inside generated
/// method bodies.  The `post_open` approach avoids this limitation entirely.
#[macro_export]
macro_rules! define_store {
    (
        $(#[$attr:meta])*
        $vis:vis static $name:ident: $ty:ident,
        $(post_open = $method:ident,)?
        expect = $expect:expr,
    ) => {
        $(#[$attr])*
        #[derive(Clone, Debug)]
        $vis struct $ty {
            pub(crate) conn: $crate::turso::Connection,
        }

        impl $ty {
            /// Open the store's database — the single consolidated domain file.
            ///
            /// For the domain stores this opens the consolidated database file
            /// ([`crate::turso::CONSOLIDATED_DB_NAME`]) via
            /// [`crate::turso::open_consolidated_store`], which runs the shared
            /// schema + consolidated migrations + one-time consolidation import.
            /// The connection returned is **fresh** and owned by this store
            /// (used by isolated `open_test_store!`); the production bootstrap
            /// shares one connection across all domain stores via
            /// [`crate::turso::init_all_stores`].
            ///
            /// This is intentionally **test-only**: production never calls it for
            /// the domain stores (they are constructed directly in
            /// [`crate::turso::init_all_stores`]), so the `dead_code` lint is
            /// silenced.
            #[allow(dead_code)]
            $vis async fn open(
                root: &std::path::Path,
            ) -> ::anyhow::Result<Self> {
                let conn = $crate::turso::open_consolidated_store(root).await?;
                let this = Self { conn };
                $(this.$method().await?;)?
                Ok(this)
            }
        }

        $(#[$attr])*
        $vis static $name: ::tokio::sync::OnceCell<$ty> =
            ::tokio::sync::OnceCell::const_new();

        #[must_use]
        #[doc = concat!(
            "Get a reference to the global ",
            stringify!($name),
            " store.\n\n# Panics\n\nPanics if the store has not been initialized.",
        )]
        $vis fn store() -> &'static $ty {
            $name.get().expect($expect)
        }
    };
}
