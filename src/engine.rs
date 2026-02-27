use std::path::Path;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{LlamaModel, params::LlamaModelParams};
use tracing::info;

use crate::error::CodeLlmError;

/// Configuration for the inference engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Context window size in tokens.
    pub n_ctx: u32,
    /// Number of layers to offload to GPU (0 = CPU only).
    pub n_gpu_layers: u32,
    /// RNG seed for reproducible sampling.
    pub seed: u32,
    /// Sampling temperature (higher = more random).
    pub temperature: f32,
    /// Top-p nucleus sampling threshold.
    pub top_p: f32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            n_ctx: 2048,
            n_gpu_layers: 0,
            seed: 0,
            temperature: 0.7,
            top_p: 0.9,
        }
    }
}

/// Wraps llama.cpp for constrained inference.
pub struct InferenceEngine {
    #[allow(dead_code)]
    backend: LlamaBackend,
    model: LlamaModel,
    config: EngineConfig,
}

impl InferenceEngine {
    /// Load a GGUF model file and initialize the backend.
    pub fn from_gguf(path: impl AsRef<Path>, config: EngineConfig) -> Result<Self, CodeLlmError> {
        let path = path.as_ref();
        info!("loading model from {}", path.display());

        let backend = LlamaBackend::init()
            .map_err(|e| CodeLlmError::BackendInit(format!("{e}")))?;

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(config.n_gpu_layers);

        let model = LlamaModel::load_from_file(&backend, path, &model_params)
            .map_err(|e| CodeLlmError::ModelLoad(format!("{e}")))?;

        Ok(Self {
            backend,
            model,
            config,
        })
    }

    /// Unconstrained text completion.
    pub fn complete(&self, _prompt: &str, _max_tokens: u32) -> Result<String, CodeLlmError> {
        todo!("completion loop — needs a GGUF model to test")
    }

    /// Grammar-constrained completion — primary API for AgentOS integration.
    ///
    /// The grammar string must be valid GBNF. Use `grammar::to_gbnf()` to generate
    /// from a `ToolSchema`.
    pub fn complete_constrained(
        &self,
        _prompt: &str,
        _grammar_str: &str,
        _max_tokens: u32,
    ) -> Result<String, CodeLlmError> {
        todo!("constrained completion loop — needs a GGUF model to test")
    }

    /// Number of tokens in the model's vocabulary.
    pub fn vocab_size(&self) -> u32 {
        self.model.n_vocab() as u32
    }

    /// Reference to the underlying model.
    pub fn model(&self) -> &LlamaModel {
        &self.model
    }

    /// Reference to the engine configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = EngineConfig::default();
        assert_eq!(config.n_ctx, 2048);
        assert_eq!(config.n_gpu_layers, 0);
        assert_eq!(config.seed, 0);
        assert!((config.temperature - 0.7).abs() < f32::EPSILON);
        assert!((config.top_p - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    #[ignore] // requires a GGUF model file
    fn load_model() {
        let engine = InferenceEngine::from_gguf(
            "test-model.gguf",
            EngineConfig::default(),
        )
        .expect("failed to load model");
        assert!(engine.vocab_size() > 0);
    }

    #[test]
    #[ignore] // requires a GGUF model file
    fn unconstrained_completion() {
        let engine = InferenceEngine::from_gguf(
            "test-model.gguf",
            EngineConfig::default(),
        )
        .unwrap();
        let result = engine.complete("Hello", 32).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    #[ignore] // requires a GGUF model file
    fn constrained_completion() {
        use crate::grammar::to_gbnf;
        use crate::schema::{ToolFieldType, ToolSchema};

        let schema = ToolSchema::new("greet")
            .required("name", ToolFieldType::String);
        let gbnf = to_gbnf(&schema);

        let engine = InferenceEngine::from_gguf(
            "test-model.gguf",
            EngineConfig::default(),
        )
        .unwrap();
        let result = engine.complete_constrained("Generate a greeting:", &gbnf, 64).unwrap();
        assert!(result.contains("<greet>"));
        assert!(result.contains("</greet>"));
    }
}
