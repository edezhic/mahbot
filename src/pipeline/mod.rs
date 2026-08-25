//! Management pipeline — board, ticket buffer, management loop, joint verdict.

pub mod board;
pub(crate) mod dispatch_latch;
pub(crate) mod joint_verdict;
pub mod management;
pub mod ticket_buffer;
