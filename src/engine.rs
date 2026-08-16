use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use tokenizers::Tokenizer;
use tracing::info;

use crate::constraint::DecodingConstraint;
use crate::error::StraitjacketError;

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

/// Statistics from a completion run.
#[derive(Debug, Clone)]
pub struct CompletionStats {
    /// Total tokens in output (prefix + generated).
    pub total_tokens: u32,
    /// Number of tokens force-emitted (constraint had exactly 1 valid option).
    pub forced_tokens: u32,
    /// Number of tokens sampled from the model.
    pub sampled_tokens: u32,
    /// Number of model forward passes (batched forced tokens = 1 pass).
    pub forward_passes: u32,
}

/// Pure-Rust inference engine backed by candle.
pub struct InferenceEngine {
    model: ModelWeights,
    tokenizer: Tokenizer,
    config: EngineConfig,
    eos_token_id: u32,
    device: Device,
}

impl InferenceEngine {
    /// Load a GGUF model and a HuggingFace tokenizer.json.
    ///
    /// Unlike llama-cpp-2, candle cannot extract the tokenizer from GGUF files.
    /// Download the tokenizer from HuggingFace:
    /// ```text
    /// huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct tokenizer.json --local-dir models/
    /// ```
    pub fn from_gguf(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        config: EngineConfig,
    ) -> Result<Self, StraitjacketError> {
        let model_path = model_path.as_ref();
        let tokenizer_path = tokenizer_path.as_ref();
        info!("loading model from {}", model_path.display());

        let device = Device::Cpu;

        // Load HuggingFace tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| StraitjacketError::ModelLoad(format!("tokenizer: {e}")))?;

        // Load GGUF model
        let mut file = std::fs::File::open(model_path)?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| StraitjacketError::ModelLoad(format!("GGUF: {e}")))?;
        let model = ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| StraitjacketError::ModelLoad(format!("{e}")))?;

        // Resolve EOS token — try Qwen's known EOS tokens
        let eos_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("<|im_end|>"))
            .unwrap_or(0);

        info!(
            "loaded: vocab={}, eos={}",
            tokenizer.get_vocab_size(true),
            eos_token_id
        );

        Ok(Self {
            model,
            tokenizer,
            config,
            eos_token_id,
            device,
        })
    }

    /// Unconstrained text completion.
    pub fn complete(&mut self, prompt: &str, max_tokens: u32) -> Result<String, StraitjacketError> {
        let (text, _stats) = self.run_completion(prompt, None, "", max_tokens)?;
        Ok(text)
    }

    /// Constrained completion using a `DecodingConstraint` for logit masking.
    ///
    /// `prefix` is structural XML that's known before generation starts — it gets
    /// fed through the model in one batched forward pass (cheap) instead of being
    /// generated token-by-token (expensive). For tool schemas, pass the opening
    /// tags up to the first value, e.g. `"<read_file><path>"`.
    ///
    /// During generation, tokens that the constraint forces (exactly 1 valid option)
    /// are collected and batch-forwarded, skipping the sampling overhead.
    pub fn complete_constrained(
        &mut self,
        prompt: &str,
        constraint: &mut dyn DecodingConstraint,
        prefix: &str,
        max_tokens: u32,
    ) -> Result<(String, CompletionStats), StraitjacketError> {
        constraint.reset();
        self.run_completion(prompt, Some(&*constraint), prefix, max_tokens)
    }

    /// Core completion loop with prefix injection and force-emit batching.
    ///
    /// Two optimizations for constrained decoding:
    ///
    /// 1. **Prefix injection**: Structural tokens known before generation (e.g. `<root><field>`)
    ///    are tokenized and processed in one batched forward pass alongside the prompt.
    ///    Zero per-token forward passes for the prefix.
    ///
    /// 2. **Force-emit batching**: When the constraint has exactly 1 valid token (deterministic
    ///    structural tokens like closing tags), consecutive forced tokens are collected and
    ///    forwarded in a single batch instead of one-by-one.
    fn run_completion(
        &mut self,
        prompt: &str,
        constraint: Option<&dyn DecodingConstraint>,
        prefix: &str,
        max_tokens: u32,
    ) -> Result<(String, CompletionStats), StraitjacketError> {
        let mut stats = CompletionStats {
            total_tokens: 0,
            forced_tokens: 0,
            sampled_tokens: 0,
            forward_passes: 0,
        };

        // Tokenize prompt
        let prompt_enc = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| StraitjacketError::Decode(format!("tokenize: {e}")))?;
        let prompt_tokens = prompt_enc.get_ids();

        if prompt_tokens.is_empty() {
            return Ok((String::new(), stats));
        }

        // Tokenize prefix (fed through model as part of prompt batch, not generated)
        let prefix_ids: Vec<u32> = if !prefix.is_empty() {
            self.tokenizer
                .encode(prefix, false)
                .map_err(|e| StraitjacketError::Decode(format!("prefix tokenize: {e}")))?
                .get_ids()
                .to_vec()
        } else {
            Vec::new()
        };

        // Build logits processor — must use TopP (not ArgMax) for sample_f support
        let sampling = if self.config.temperature < 1e-7 {
            Sampling::TopP {
                p: self.config.top_p as f64,
                temperature: 1e-7,
            }
        } else {
            Sampling::TopP {
                p: self.config.top_p as f64,
                temperature: self.config.temperature as f64,
            }
        };
        let mut logits_processor =
            LogitsProcessor::from_sampling(self.config.seed as u64, sampling);

        // Process prompt + prefix in one batched forward pass
        let all_input: Vec<u32> = prompt_tokens
            .iter()
            .chain(prefix_ids.iter())
            .copied()
            .collect();

        let mut current_logits = self.forward_batch(&all_input, 0)?;
        stats.forward_passes += 1;

        let vocab_size = current_logits.dims().last().copied().unwrap_or(0);

        // Output state — includes prefix as pre-generated content
        let mut output_ids: Vec<u32> = prefix_ids;
        let mut output_text = prefix.to_string();
        let mut pos = all_input.len(); // next position in KV cache
        stats.total_tokens = output_ids.len() as u32;

        // Main generation loop
        let mut tokens_remaining = max_tokens;

        loop {
            if tokens_remaining == 0 {
                break;
            }

            // Phase 1: Force-emit — collect tokens where constraint has exactly 1 valid option.
            // These are deterministic structural tokens (closing tags, next field tags).
            // Batch them into a single forward pass instead of one-by-one.
            if let Some(c) = constraint {
                let mut forced_batch: Vec<u32> = Vec::new();

                loop {
                    if tokens_remaining == 0 {
                        break;
                    }
                    if c.is_complete(&output_text) {
                        break;
                    }

                    let mask = c.valid_tokens(&output_text, vocab_size);
                    let valid: Vec<usize> = mask
                        .iter()
                        .enumerate()
                        .filter(|(_, &v)| v)
                        .map(|(i, _)| i)
                        .collect();

                    if valid.len() != 1 {
                        break; // multiple options — need the model to decide
                    }

                    let forced_id = valid[0] as u32;
                    forced_batch.push(forced_id);
                    output_ids.push(forced_id);
                    output_text = self
                        .tokenizer
                        .decode(&output_ids, false)
                        .map_err(|e| StraitjacketError::Decode(format!("decode: {e}")))?;
                    stats.forced_tokens += 1;
                    stats.total_tokens += 1;
                    tokens_remaining -= 1;
                }

                if !forced_batch.is_empty() {
                    // One batched forward pass for all forced tokens
                    current_logits = self.forward_batch(&forced_batch, pos)?;
                    pos += forced_batch.len();
                    stats.forward_passes += 1;

                    if c.is_complete(&output_text) || tokens_remaining == 0 {
                        break;
                    }
                    continue; // re-check — might be more forced tokens after the batch
                }

                if c.is_complete(&output_text) {
                    break;
                }
            }

            // Phase 2: Sample — model decides among multiple valid tokens.
            let next_token = self.sample_token(
                &current_logits,
                &mut logits_processor,
                constraint,
                &output_text,
                vocab_size,
            )?;

            output_ids.push(next_token);
            output_text = self
                .tokenizer
                .decode(&output_ids, false)
                .map_err(|e| StraitjacketError::Decode(format!("decode: {e}")))?;
            stats.sampled_tokens += 1;
            stats.total_tokens += 1;
            tokens_remaining -= 1;

            if next_token == self.eos_token_id {
                break;
            }
            if let Some(c) = constraint {
                if c.is_complete(&output_text) {
                    break;
                }
            }
            if tokens_remaining == 0 {
                break;
            }

            // Forward pass for sampled token — updates KV cache, produces next logits
            let input = Tensor::new(&[next_token], &self.device)
                .map_err(|e| StraitjacketError::Decode(format!("tensor: {e}")))?
                .unsqueeze(0)
                .map_err(|e| StraitjacketError::Decode(format!("unsqueeze: {e}")))?;
            current_logits = self
                .model
                .forward(&input, pos)
                .map_err(|e| StraitjacketError::Decode(format!("forward: {e}")))?;
            current_logits = self.squeeze_last_logits(current_logits)?;
            pos += 1;
            stats.forward_passes += 1;
        }

        Ok((output_text, stats))
    }

    /// Forward a batch of token IDs through the model, return logits for last position.
    fn forward_batch(&mut self, token_ids: &[u32], pos: usize) -> Result<Tensor, StraitjacketError> {
        let input = Tensor::new(token_ids, &self.device)
            .map_err(|e| StraitjacketError::Decode(format!("tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| StraitjacketError::Decode(format!("unsqueeze: {e}")))?;

        let logits = self
            .model
            .forward(&input, pos)
            .map_err(|e| StraitjacketError::Decode(format!("forward: {e}")))?;

        self.squeeze_last_logits(logits)
    }

    /// Squeeze batch dimension and select last position's logits.
    fn squeeze_last_logits(&self, logits: Tensor) -> Result<Tensor, StraitjacketError> {
        let logits = logits
            .squeeze(0)
            .map_err(|e| StraitjacketError::Decode(format!("squeeze: {e}")))?;

        if logits.dims().len() == 2 {
            let seq_len = logits
                .dim(0)
                .map_err(|e| StraitjacketError::Decode(format!("dim: {e}")))?;
            logits
                .get(seq_len - 1)
                .map_err(|e| StraitjacketError::Decode(format!("get last: {e}")))
        } else {
            Ok(logits)
        }
    }

    /// Sample a token, applying constraint mask if present.
    fn sample_token(
        &self,
        logits: &Tensor,
        logits_processor: &mut LogitsProcessor,
        constraint: Option<&dyn DecodingConstraint>,
        generated_text: &str,
        vocab_size: usize,
    ) -> Result<u32, StraitjacketError> {
        match constraint {
            Some(c) => {
                let mask = c.valid_tokens(generated_text, vocab_size);
                logits_processor
                    .sample_f(logits, |prs| {
                        for (i, p) in prs.iter_mut().enumerate() {
                            if i < mask.len() && !mask[i] {
                                *p = 0.0;
                            }
                        }
                    })
                    .map_err(|e| StraitjacketError::Decode(format!("sample: {e}")))
            }
            None => logits_processor
                .sample(logits)
                .map_err(|e| StraitjacketError::Decode(format!("sample: {e}"))),
        }
    }

    /// Number of tokens in the model's vocabulary.
    pub fn vocab_size(&self) -> u32 {
        self.tokenizer.get_vocab_size(true) as u32
    }

    /// Reference to the tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Reference to the engine configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_PATH: &str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    const TOKENIZER_PATH: &str = "models/tokenizer.json";

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
    #[ignore] // requires GGUF model + tokenizer
    fn load_model() {
        let engine =
            InferenceEngine::from_gguf(MODEL_PATH, TOKENIZER_PATH, EngineConfig::default())
                .expect("failed to load model");
        assert!(engine.vocab_size() > 0);
        eprintln!("vocab_size: {}", engine.vocab_size());
    }

    #[test]
    #[ignore] // requires GGUF model + tokenizer
    fn unconstrained_completion() {
        let mut engine =
            InferenceEngine::from_gguf(MODEL_PATH, TOKENIZER_PATH, EngineConfig::default())
                .unwrap();
        let result = engine
            .complete("The capital of France is", 32)
            .unwrap();
        assert!(!result.is_empty());
        eprintln!("unconstrained: {result}");
    }

    #[test]
    #[ignore] // requires GGUF model + tokenizer
    fn constrained_xml_no_prefix() {
        use crate::constraint::XmlSchemaConstraint;
        use crate::schema::{ToolFieldType, ToolSchema};

        let schema = ToolSchema::new("read_file")
            .required("path", ToolFieldType::String)
            .optional("line_count", ToolFieldType::Integer);

        let mut engine =
            InferenceEngine::from_gguf(MODEL_PATH, TOKENIZER_PATH, EngineConfig::default())
                .unwrap();

        let mut constraint = XmlSchemaConstraint::new(schema, engine.tokenizer());

        let prompt = "Fill in XML parameters for read_file to read /etc/hosts:";
        let (result, stats) = engine
            .complete_constrained(prompt, &mut constraint, "", 128)
            .unwrap();
        eprintln!("no prefix: {result}");
        eprintln!(
            "stats: {} total, {} forced, {} sampled, {} forward passes",
            stats.total_tokens, stats.forced_tokens, stats.sampled_tokens, stats.forward_passes
        );
        assert!(result.contains("<read_file>"));
        assert!(result.contains("</read_file>"));
        assert!(result.contains("<path>"));
    }

    #[test]
    #[ignore] // requires GGUF model + tokenizer
    fn constrained_xml_with_prefix() {
        use crate::constraint::XmlSchemaConstraint;
        use crate::schema::{ToolFieldType, ToolSchema};

        let schema = ToolSchema::new("read_file")
            .required("path", ToolFieldType::String)
            .optional("line_count", ToolFieldType::Integer);

        let mut engine =
            InferenceEngine::from_gguf(MODEL_PATH, TOKENIZER_PATH, EngineConfig::default())
                .unwrap();

        let mut constraint = XmlSchemaConstraint::new(schema, engine.tokenizer());

        let prompt = "Fill in XML parameters for read_file to read /etc/hosts:";
        let prefix = "<read_file><path>";
        let (result, stats) = engine
            .complete_constrained(prompt, &mut constraint, prefix, 128)
            .unwrap();
        eprintln!("with prefix: {result}");
        eprintln!(
            "stats: {} total, {} forced, {} sampled, {} forward passes",
            stats.total_tokens, stats.forced_tokens, stats.sampled_tokens, stats.forward_passes
        );
        assert!(result.starts_with("<read_file><path>"));
        assert!(result.contains("</read_file>"));
        // With prefix, the model skips generating the structural opening —
        // should have fewer forward passes than without prefix
        assert!(stats.forced_tokens > 0, "closing tags should be force-emitted");
    }
}
