use thiserror::Error;

/// All errors that can occur in codeLlm.
#[derive(Debug, Error)]
pub enum CodeLlmError {
    #[error("failed to initialize llama.cpp backend: {0}")]
    BackendInit(String),

    #[error("failed to load model: {0}")]
    ModelLoad(String),

    #[error("failed to create inference context: {0}")]
    ContextCreate(String),

    #[error("invalid GBNF grammar: {0}")]
    InvalidGrammar(String),

    #[error("decode error: {0}")]
    Decode(String),

    #[error("schema conversion error: {0}")]
    SchemaConversion(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
