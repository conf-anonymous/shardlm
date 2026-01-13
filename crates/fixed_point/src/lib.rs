//! ShardLM Fixed-Point Encoding
//!
//! Fixed-point representation for neural network computations.
//! Uses i32 values with configurable scaling factor 2^S.

mod error;
mod fixed;
mod vector;

pub use error::{FixedPointError, Result};
pub use fixed::{Fixed, DEFAULT_SCALE};
pub use vector::FixedVector;
