//! Declarative macros for column-index constant generation.
//!
//! The [`columns!`] macro eliminates the brittle hand-maintained coupling
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
