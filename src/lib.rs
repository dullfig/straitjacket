//! codeLlm — Constrained decoding for local LLMs.
//!
//! Feed it a GBNF grammar, get structurally valid output. Phase 1 wraps
//! llama.cpp's built-in grammar sampler. Phase 2+ adds custom logit masking
//! via the `DecodingConstraint` trait.

pub mod constraint;
pub mod engine;
pub mod error;
pub mod grammar;
pub mod sampler;
pub mod schema;

/// Convenience Result alias.
pub type Result<T> = std::result::Result<T, error::CodeLlmError>;

/// Re-exports for common use.
pub mod prelude {
    pub use crate::constraint::DecodingConstraint;
    pub use crate::engine::{EngineConfig, InferenceEngine};
    pub use crate::error::CodeLlmError;
    pub use crate::grammar::to_gbnf;
    pub use crate::schema::{SchemaField, ToolFieldType, ToolSchema};
    pub use crate::Result;
}
