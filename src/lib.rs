//! straitjacket — Constrained decoding for local LLMs.
//!
//! Uses candle (pure Rust ML framework) for inference with custom logit masking
//! via the `DecodingConstraint` trait. No C++ deps, no cmake, no LLVM.

pub mod constraint;
pub mod engine;
pub mod error;
pub mod grammar;
pub mod schema;

/// Convenience Result alias.
pub type Result<T> = std::result::Result<T, error::StraitjacketError>;

/// Re-exports for common use.
pub mod prelude {
    pub use crate::constraint::{DecodingConstraint, XmlSchemaConstraint};
    pub use crate::engine::{EngineConfig, InferenceEngine};
    pub use crate::error::StraitjacketError;
    pub use crate::grammar::{to_gbnf, to_lark};
    pub use crate::schema::{SchemaField, ToolFieldType, ToolSchema};
    pub use crate::Result;
}
