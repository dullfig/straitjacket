use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;

use crate::engine::EngineConfig;
use crate::error::CodeLlmError;

/// Build a sampler chain with grammar constraint applied first.
///
/// Order matters: grammar masks invalid tokens → temperature softens distribution →
/// top_p truncates → dist samples. Grammar MUST be first so temperature never
/// operates on structurally invalid tokens.
pub fn build_constrained_sampler(
    model: &LlamaModel,
    grammar_str: &str,
    config: &EngineConfig,
) -> Result<LlamaSampler, CodeLlmError> {
    let grammar = LlamaSampler::grammar(model, grammar_str, "root")
        .map_err(|e| CodeLlmError::InvalidGrammar(format!("{e}")))?;

    Ok(LlamaSampler::chain_simple([
        grammar,
        LlamaSampler::temp(config.temperature),
        LlamaSampler::top_p(config.top_p, /* min_keep */ 1),
        LlamaSampler::dist(config.seed),
    ]))
}

/// Build an unconstrained sampler chain (no grammar).
pub fn build_sampler(config: &EngineConfig) -> LlamaSampler {
    LlamaSampler::chain_simple([
        LlamaSampler::temp(config.temperature),
        LlamaSampler::top_p(config.top_p, /* min_keep */ 1),
        LlamaSampler::dist(config.seed),
    ])
}
