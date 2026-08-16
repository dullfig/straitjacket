use thiserror::Error;

/// All errors that can occur in straitjacket.
#[derive(Debug, Error)]
pub enum StraitjacketError {
    #[error("failed to load model: {0}")]
    ModelLoad(String),

    #[error("invalid grammar: {0}")]
    InvalidGrammar(String),

    #[error("decode error: {0}")]
    Decode(String),

    #[error("schema conversion error: {0}")]
    SchemaConversion(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
