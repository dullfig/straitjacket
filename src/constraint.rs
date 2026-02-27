/// Trait for custom decoding constraints (Phase 2+).
///
/// Phase 1 uses llama.cpp's built-in GBNF grammar sampler.
/// This trait is defined now for Phase 2 custom logit masking (XSD, LSP).
pub trait DecodingConstraint {
    /// Returns a bitmask over the vocabulary: `true` = token is valid given what's been generated.
    fn valid_tokens(&self, generated_text: &str, vocab_size: usize) -> Vec<bool>;

    /// Returns `true` when the generated text is a complete, valid output.
    fn is_complete(&self, generated_text: &str) -> bool;

    /// Reset internal state for a new generation.
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the trait is object-safe and implementable.
    struct AlwaysValid;

    impl DecodingConstraint for AlwaysValid {
        fn valid_tokens(&self, _generated_text: &str, vocab_size: usize) -> Vec<bool> {
            vec![true; vocab_size]
        }

        fn is_complete(&self, _generated_text: &str) -> bool {
            true
        }

        fn reset(&mut self) {}
    }

    #[test]
    fn trait_compiles_and_works() {
        let mut constraint = AlwaysValid;
        let mask = constraint.valid_tokens("hello", 100);
        assert_eq!(mask.len(), 100);
        assert!(mask.iter().all(|&v| v));
        assert!(constraint.is_complete("anything"));
        constraint.reset(); // no-op, but must compile
    }
}
