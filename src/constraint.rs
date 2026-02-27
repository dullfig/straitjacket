use crate::schema::{ToolFieldType, ToolSchema};

/// Trait for custom decoding constraints — logit masking at each generation step.
///
/// At each step, `valid_tokens()` returns a bitmask over the vocabulary.
/// The engine zeros out probabilities for masked tokens before sampling.
pub trait DecodingConstraint {
    /// Returns a bitmask over the vocabulary: `true` = token is valid given what's been generated.
    fn valid_tokens(&self, generated_text: &str, vocab_size: usize) -> Vec<bool>;

    /// Returns `true` when the generated text is a complete, valid output.
    fn is_complete(&self, generated_text: &str) -> bool;

    /// Reset internal state for a new generation.
    fn reset(&mut self);

    /// If the next output is structurally determined, return the exact text.
    ///
    /// The engine tokenizes this text and batch-forwards it without sampling —
    /// zero per-token overhead for deterministic structural regions.
    /// Returns `None` when the model needs to decide (value regions).
    fn forced_continuation(&self, _generated_text: &str) -> Option<String> {
        None
    }
}

/// XML position within the schema structure.
#[derive(Debug, PartialEq)]
enum XmlPos {
    /// Before any output.
    BeforeRoot,
    /// Partway through root open tag. `consumed` chars matched so far.
    PartialRootOpen(usize),
    /// Between fields — next field to consider is at `field_idx`.
    /// When `field_idx == fields.len()`, all fields are done.
    BetweenFields(usize),
    /// Inside a value region for the given field.
    InValue(usize),
    /// Partway through a field close tag. `consumed` chars of the close tag matched.
    PartialFieldClose { field_idx: usize, consumed: usize },
    /// Partway through root close tag. `consumed` chars matched.
    PartialRootClose(usize),
    /// Output is structurally complete.
    Complete,
}

/// Constrains decoding to produce valid XML matching a `ToolSchema`.
///
/// At each step, checks which tokens would keep the output on a valid path
/// toward a schema-conformant XML document. Tags are forced to match exactly;
/// value regions allow free text appropriate to the field type.
pub struct XmlSchemaConstraint {
    schema: ToolSchema,
    /// Cached: token_id → decoded string
    token_strings: Vec<String>,
    /// EOS token ID (allowed only when output is complete)
    eos_token_id: Option<u32>,
    /// Precomputed tag strings
    root_open: String,
    root_close: String,
    field_opens: Vec<String>,
    field_closes: Vec<String>,
}

impl XmlSchemaConstraint {
    /// Build a constraint from a schema and tokenizer.
    ///
    /// Caches the string representation of every token in the vocabulary
    /// for fast mask computation during generation.
    pub fn new(schema: ToolSchema, tokenizer: &tokenizers::Tokenizer) -> Self {
        let vocab_size = tokenizer.get_vocab_size(true);
        let token_strings: Vec<String> = (0..vocab_size)
            .map(|id| {
                tokenizer
                    .decode(&[id as u32], false)
                    .unwrap_or_default()
            })
            .collect();

        // Find EOS token
        let eos_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("<|im_end|>"));

        let root_open = format!("<{}>", schema.root_tag);
        let root_close = format!("</{}>", schema.root_tag);
        let field_opens: Vec<String> = schema
            .fields
            .iter()
            .map(|f| format!("<{}>", f.name))
            .collect();
        let field_closes: Vec<String> = schema
            .fields
            .iter()
            .map(|f| format!("</{}>", f.name))
            .collect();

        Self {
            schema,
            token_strings,
            eos_token_id,
            root_open,
            root_close,
            field_opens,
            field_closes,
        }
    }

    /// Determine the current structural position within the generated XML.
    fn find_position(&self, text: &str) -> XmlPos {
        if text.is_empty() {
            return XmlPos::BeforeRoot;
        }

        // Check root open
        if !text.starts_with(&self.root_open) {
            if self.root_open.starts_with(text) {
                return XmlPos::PartialRootOpen(text.len());
            }
            return XmlPos::BeforeRoot; // invalid, but treat as before root
        }

        let mut rest = &text[self.root_open.len()..];
        rest = rest.trim_start();

        // Walk fields
        for (i, field) in self.schema.fields.iter().enumerate() {
            if rest.is_empty() {
                return XmlPos::BetweenFields(i);
            }

            let field_open = &self.field_opens[i];
            let field_close = &self.field_closes[i];

            if rest.starts_with(field_open.as_str()) {
                rest = &rest[field_open.len()..];

                // Look for complete close tag
                if let Some(close_pos) = rest.find(field_close.as_str()) {
                    rest = &rest[close_pos + field_close.len()..];
                    rest = rest.trim_start();
                    // Field complete — continue to next
                } else {
                    // In value or partial close tag — check for '<'
                    if let Some(lt_pos) = rest.rfind('<') {
                        let tag_part = &rest[lt_pos..];
                        if field_close.starts_with(tag_part) {
                            return XmlPos::PartialFieldClose {
                                field_idx: i,
                                consumed: tag_part.len(),
                            };
                        }
                    }
                    return XmlPos::InValue(i);
                }
            } else if !field.required {
                // Optional field not present — skip
                continue;
            } else {
                // Required field expected but not started — still between fields
                return XmlPos::BetweenFields(i);
            }
        }

        // After all fields
        if rest.is_empty() {
            return XmlPos::BetweenFields(self.schema.fields.len());
        }
        if rest.starts_with(&self.root_close) {
            return XmlPos::Complete;
        }
        if self.root_close.starts_with(rest) {
            return XmlPos::PartialRootClose(rest.len());
        }

        XmlPos::BetweenFields(self.schema.fields.len())
    }

    /// Build the forced continuation from a between-fields position.
    ///
    /// Returns the next deterministic structural text, stopping at the next
    /// value region (where the model needs to decide).
    fn forced_from_between(&self, from_idx: usize) -> Option<String> {
        if from_idx >= self.schema.fields.len() {
            // All fields done — force root close
            return Some(self.root_close.clone());
        }
        if self.schema.fields[from_idx].required {
            // Next field is required — force its open tag
            return Some(self.field_opens[from_idx].clone());
        }
        // Next field is optional — model decides
        None
    }

    /// Fast-path validation for tokens appended while inside a value region.
    ///
    /// For String fields with no `<` in the token: immediately valid (O(1)).
    /// Otherwise: checks combined value against field type + close tag detection.
    /// Only falls back to full prefix check for rare boundary-spanning tokens.
    fn token_valid_in_value(
        &self,
        generated_text: &str,
        token_str: &str,
        field_idx: usize,
    ) -> bool {
        let field_type = self.schema.fields[field_idx].field_type;
        let field_close = &self.field_closes[field_idx];

        // Hot path: String field, token has no '<' → always valid.
        // Values can't contain '<', and no '<' means no close tag starting.
        if field_type == ToolFieldType::String && !token_str.contains('<') {
            return true;
        }

        // Build combined value (current value text + new token)
        let current_value = self.extract_value(generated_text, field_idx);
        let mut combined = String::with_capacity(current_value.len() + token_str.len());
        combined.push_str(current_value);
        combined.push_str(token_str);

        // Check if token completes the close tag
        if let Some(close_pos) = combined.find(field_close.as_str()) {
            let value = &combined[..close_pos];
            if !Self::validate_value_text(value, field_type) {
                return false;
            }
            let after = &combined[close_pos + field_close.len()..];
            if after.is_empty() {
                return true; // exactly at close tag boundary
            }
            // Token spans past close tag into next region — rare with BPE.
            // Fall back to full prefix check.
            let candidate = format!("{}{}", generated_text, token_str);
            return self.could_be_valid_prefix(&candidate);
        }

        // No complete close tag — check for valid value or partial close tag
        self.is_valid_value_or_partial_close(&combined, field_close, field_type)
    }

    /// Extract the current value text for a field in InValue position.
    ///
    /// Uses `rfind` for the field open tag. Safe because validated value text
    /// cannot contain `<` (enforced by `validate_value_text`), so the last
    /// occurrence is always the real tag.
    fn extract_value<'a>(&self, text: &'a str, field_idx: usize) -> &'a str {
        let field_open = &self.field_opens[field_idx];
        match text.rfind(field_open.as_str()) {
            Some(pos) => &text[pos + field_open.len()..],
            None => "",
        }
    }

    /// Original O(V×L) implementation — kept for benchmarking against incremental dispatch.
    #[cfg(test)]
    fn valid_tokens_naive(&self, generated_text: &str, vocab_size: usize) -> Vec<bool> {
        let mut mask = vec![false; vocab_size];
        for (id, token_str) in self.token_strings.iter().enumerate().take(vocab_size) {
            if token_str.is_empty() {
                continue;
            }
            let candidate = format!("{}{}", generated_text, token_str);
            if self.could_be_valid_prefix(&candidate) {
                mask[id] = true;
            }
        }
        if self.is_complete(generated_text) {
            if let Some(eos) = self.eos_token_id {
                if (eos as usize) < vocab_size {
                    mask[eos as usize] = true;
                }
            }
        }
        if !mask.iter().any(|&v| v) {
            return vec![true; vocab_size];
        }
        mask
    }

    /// Check if `text` could be the prefix of a valid XML document matching the schema.
    fn could_be_valid_prefix(&self, text: &str) -> bool {
        if text.is_empty() {
            return true;
        }

        // Phase 1: match root open tag
        if !text.starts_with(&self.root_open) {
            return self.root_open.starts_with(text);
        }
        let mut rest = &text[self.root_open.len()..];
        rest = rest.trim_start();

        // Phase 2: match fields in declaration order
        for (i, field) in self.schema.fields.iter().enumerate() {
            if rest.is_empty() {
                return true;
            }

            let field_open = &self.field_opens[i];
            let field_close = &self.field_closes[i];

            if rest.starts_with(field_open.as_str()) {
                rest = &rest[field_open.len()..];

                if let Some(close_pos) = rest.find(field_close.as_str()) {
                    let value = &rest[..close_pos];
                    if !Self::validate_value_text(value, field.field_type) {
                        return false;
                    }
                    rest = &rest[close_pos + field_close.len()..];
                    rest = rest.trim_start();
                } else {
                    return self.is_valid_value_or_partial_close(rest, field_close, field.field_type);
                }
            } else if rest.len() < field_open.len() {
                if field.required {
                    return field_open.starts_with(rest);
                }
                return field_open.starts_with(rest) || self.root_close.starts_with(rest);
            } else if !field.required {
                continue;
            } else {
                return false;
            }
        }

        // Phase 3: match root close tag
        if rest.is_empty() {
            return true;
        }
        if rest.starts_with(&self.root_close) {
            let after = &rest[self.root_close.len()..];
            return after.is_empty() || after.chars().all(|c| c.is_whitespace());
        }
        self.root_close.starts_with(rest)
    }

    /// Check if `rest` is valid value text possibly followed by a partial closing tag.
    fn is_valid_value_or_partial_close(
        &self,
        rest: &str,
        field_close: &str,
        field_type: ToolFieldType,
    ) -> bool {
        if let Some(lt_pos) = rest.rfind('<') {
            let value_part = &rest[..lt_pos];
            let tag_part = &rest[lt_pos..];
            Self::validate_value_text(value_part, field_type)
                && field_close.starts_with(tag_part)
        } else {
            Self::validate_value_text(rest, field_type)
        }
    }

    /// Check if `value` is valid text for the given field type.
    fn validate_value_text(value: &str, field_type: ToolFieldType) -> bool {
        if value.is_empty() {
            return true;
        }
        match field_type {
            ToolFieldType::String => !value.contains('<'),
            ToolFieldType::Integer => {
                value
                    .chars()
                    .all(|c| c.is_ascii_digit() || c == '-')
                    && value.matches('-').count() <= 1
                    && (value.len() == 1 || !value.ends_with('-'))
            }
            ToolFieldType::Float => {
                value
                    .chars()
                    .all(|c| c.is_ascii_digit() || c == '-' || c == '.')
                    && value.matches('-').count() <= 1
                    && value.matches('.').count() <= 1
            }
            ToolFieldType::Boolean => {
                "true".starts_with(value) || "false".starts_with(value)
            }
        }
    }
}

impl DecodingConstraint for XmlSchemaConstraint {
    fn valid_tokens(&self, generated_text: &str, vocab_size: usize) -> Vec<bool> {
        let mut mask = vec![false; vocab_size];

        // Compute position ONCE, then dispatch.
        // InValue is the hot path (most tokens generated here): O(V × T) where T = token length.
        // All other positions: rare (once per field boundary), use full prefix walk.
        let pos = self.find_position(generated_text);

        if let XmlPos::InValue(field_idx) = pos {
            // Fast path: check each token against value constraints + close tag detection.
            // Avoids re-walking the completed prefix for every candidate token.
            for (id, token_str) in self.token_strings.iter().enumerate().take(vocab_size) {
                if token_str.is_empty() {
                    continue;
                }
                mask[id] = self.token_valid_in_value(generated_text, token_str, field_idx);
            }
        } else {
            // Structural positions: full prefix check (rare — once per field boundary).
            for (id, token_str) in self.token_strings.iter().enumerate().take(vocab_size) {
                if token_str.is_empty() {
                    continue;
                }
                let candidate = format!("{}{}", generated_text, token_str);
                if self.could_be_valid_prefix(&candidate) {
                    mask[id] = true;
                }
            }
        }

        // Allow EOS only when output is complete
        if self.is_complete(generated_text) {
            if let Some(eos) = self.eos_token_id {
                if (eos as usize) < vocab_size {
                    mask[eos as usize] = true;
                }
            }
        }

        // Safety: if nothing is valid, allow everything to prevent infinite loop
        if !mask.iter().any(|&v| v) {
            return vec![true; vocab_size];
        }

        mask
    }

    fn is_complete(&self, generated_text: &str) -> bool {
        let trimmed = generated_text.trim();
        trimmed.ends_with(&self.root_close)
    }

    fn reset(&mut self) {}

    fn forced_continuation(&self, generated_text: &str) -> Option<String> {
        let pos = self.find_position(generated_text);
        let mut result = String::new();

        match pos {
            XmlPos::BeforeRoot => {
                result.push_str(&self.root_open);
                if let Some(ext) = self.forced_from_between(0) {
                    result.push_str(&ext);
                }
            }
            XmlPos::PartialRootOpen(consumed) => {
                result.push_str(&self.root_open[consumed..]);
                if let Some(ext) = self.forced_from_between(0) {
                    result.push_str(&ext);
                }
            }
            XmlPos::BetweenFields(next_idx) => {
                if let Some(ext) = self.forced_from_between(next_idx) {
                    result.push_str(&ext);
                }
            }
            XmlPos::PartialFieldClose { field_idx, consumed } => {
                result.push_str(&self.field_closes[field_idx][consumed..]);
                if let Some(ext) = self.forced_from_between(field_idx + 1) {
                    result.push_str(&ext);
                }
            }
            XmlPos::PartialRootClose(consumed) => {
                result.push_str(&self.root_close[consumed..]);
            }
            XmlPos::InValue(_) | XmlPos::Complete => return None,
        }

        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }
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
        constraint.reset();
        assert_eq!(constraint.forced_continuation("anything"), None);
    }

    // --- XmlSchemaConstraint tests ---

    use crate::schema::{ToolFieldType, ToolSchema};

    fn read_file_schema() -> ToolSchema {
        ToolSchema::new("read_file")
            .required("path", ToolFieldType::String)
            .optional("line_count", ToolFieldType::Integer)
    }

    fn make_constraint(schema: ToolSchema) -> XmlSchemaConstraint {
        XmlSchemaConstraint {
            root_open: format!("<{}>", schema.root_tag),
            root_close: format!("</{}>", schema.root_tag),
            field_opens: schema.fields.iter().map(|f| format!("<{}>", f.name)).collect(),
            field_closes: schema.fields.iter().map(|f| format!("</{}>", f.name)).collect(),
            schema,
            token_strings: Vec::new(),
            eos_token_id: None,
        }
    }

    /// Build a constraint with a synthetic vocabulary for benchmarking.
    /// Simulates a BPE tokenizer with `vocab_size` tokens of mixed lengths.
    fn make_constraint_with_vocab(schema: ToolSchema, vocab_size: usize) -> XmlSchemaConstraint {
        let mut token_strings = Vec::with_capacity(vocab_size);
        // Token 0: empty (padding)
        token_strings.push(String::new());
        // Tokens 1..256: single ASCII bytes
        for b in 1u8..=255 {
            if token_strings.len() >= vocab_size { break; }
            token_strings.push(String::from(b as char));
        }
        // Tokens 256+: synthetic multi-char tokens (2-6 chars)
        let chars = b"abcdefghijklmnopqrstuvwxyz0123456789_/.-<>";
        let mut idx = 0usize;
        while token_strings.len() < vocab_size {
            let len = 2 + (idx % 5); // 2-6 chars
            let s: String = (0..len)
                .map(|j| chars[(idx + j * 7) % chars.len()] as char)
            .collect();
            token_strings.push(s);
            idx += 1;
        }
        // Ensure a few XML-relevant tokens exist
        if vocab_size > 300 {
            token_strings[260] = "</".to_string();
            token_strings[261] = "</path>".to_string();
            token_strings[262] = "<path>".to_string();
            token_strings[263] = "</read_file>".to_string();
            token_strings[264] = "<read_file>".to_string();
        }
        XmlSchemaConstraint {
            root_open: format!("<{}>", schema.root_tag),
            root_close: format!("</{}>", schema.root_tag),
            field_opens: schema.fields.iter().map(|f| format!("<{}>", f.name)).collect(),
            field_closes: schema.fields.iter().map(|f| format!("</{}>", f.name)).collect(),
            eos_token_id: Some(0),
            schema,
            token_strings,
        }
    }

    // --- Prefix validation tests ---

    #[test]
    fn prefix_empty() {
        let c = make_constraint(read_file_schema());
        assert!(c.could_be_valid_prefix(""));
    }

    #[test]
    fn prefix_partial_root_open() {
        let c = make_constraint(read_file_schema());
        assert!(c.could_be_valid_prefix("<"));
        assert!(c.could_be_valid_prefix("<read"));
        assert!(c.could_be_valid_prefix("<read_file"));
        assert!(c.could_be_valid_prefix("<read_file>"));
    }

    #[test]
    fn prefix_invalid_root() {
        let c = make_constraint(read_file_schema());
        assert!(!c.could_be_valid_prefix("<write"));
        assert!(!c.could_be_valid_prefix("hello"));
    }

    #[test]
    fn prefix_after_root_open() {
        let c = make_constraint(read_file_schema());
        assert!(c.could_be_valid_prefix("<read_file>"));
        assert!(c.could_be_valid_prefix("<read_file><"));
        assert!(c.could_be_valid_prefix("<read_file><path"));
        assert!(c.could_be_valid_prefix("<read_file><path>"));
    }

    #[test]
    fn prefix_in_value() {
        let c = make_constraint(read_file_schema());
        assert!(c.could_be_valid_prefix("<read_file><path>/etc/hosts"));
        assert!(c.could_be_valid_prefix("<read_file><path>/etc/hosts</"));
        assert!(c.could_be_valid_prefix("<read_file><path>/etc/hosts</path>"));
    }

    #[test]
    fn prefix_between_fields() {
        let c = make_constraint(read_file_schema());
        assert!(c.could_be_valid_prefix("<read_file><path>/etc/hosts</path>"));
        assert!(c.could_be_valid_prefix("<read_file><path>/etc/hosts</path><line_count>"));
        assert!(c.could_be_valid_prefix("<read_file><path>/etc/hosts</path></read_file>"));
    }

    #[test]
    fn prefix_optional_field_with_value() {
        let c = make_constraint(read_file_schema());
        assert!(c.could_be_valid_prefix(
            "<read_file><path>/etc/hosts</path><line_count>10</line_count></read_file>"
        ));
    }

    #[test]
    fn prefix_complete() {
        let c = make_constraint(read_file_schema());
        assert!(c.is_complete("<read_file><path>/etc/hosts</path></read_file>"));
        assert!(!c.is_complete("<read_file><path>/etc/hosts</path>"));
    }

    #[test]
    fn validate_integer_value() {
        assert!(XmlSchemaConstraint::validate_value_text("123", ToolFieldType::Integer));
        assert!(XmlSchemaConstraint::validate_value_text("-42", ToolFieldType::Integer));
        assert!(XmlSchemaConstraint::validate_value_text("", ToolFieldType::Integer));
        assert!(!XmlSchemaConstraint::validate_value_text("abc", ToolFieldType::Integer));
        assert!(!XmlSchemaConstraint::validate_value_text("12.3", ToolFieldType::Integer));
    }

    #[test]
    fn validate_boolean_value() {
        assert!(XmlSchemaConstraint::validate_value_text("true", ToolFieldType::Boolean));
        assert!(XmlSchemaConstraint::validate_value_text("false", ToolFieldType::Boolean));
        assert!(XmlSchemaConstraint::validate_value_text("t", ToolFieldType::Boolean));
        assert!(XmlSchemaConstraint::validate_value_text("f", ToolFieldType::Boolean));
        assert!(!XmlSchemaConstraint::validate_value_text("yes", ToolFieldType::Boolean));
    }

    #[test]
    fn validate_float_value() {
        assert!(XmlSchemaConstraint::validate_value_text("3.14", ToolFieldType::Float));
        assert!(XmlSchemaConstraint::validate_value_text("-0.5", ToolFieldType::Float));
        assert!(XmlSchemaConstraint::validate_value_text("42", ToolFieldType::Float));
        assert!(!XmlSchemaConstraint::validate_value_text("1.2.3", ToolFieldType::Float));
    }

    #[test]
    fn all_required_schema() {
        let schema = ToolSchema::new("write_file")
            .required("path", ToolFieldType::String)
            .required("content", ToolFieldType::String);
        let c = make_constraint(schema);

        assert!(c.could_be_valid_prefix("<write_file><path>foo.rs</path><content>hello</content></write_file>"));
        assert!(!c.could_be_valid_prefix("<write_file><path>foo.rs</path></write_file>"));
    }

    // --- find_position tests ---

    #[test]
    fn position_empty() {
        let c = make_constraint(read_file_schema());
        assert_eq!(c.find_position(""), XmlPos::BeforeRoot);
    }

    #[test]
    fn position_partial_root() {
        let c = make_constraint(read_file_schema());
        assert_eq!(c.find_position("<read"), XmlPos::PartialRootOpen(5));
    }

    #[test]
    fn position_after_root_open() {
        let c = make_constraint(read_file_schema());
        assert_eq!(c.find_position("<read_file>"), XmlPos::BetweenFields(0));
    }

    #[test]
    fn position_in_value() {
        let c = make_constraint(read_file_schema());
        assert_eq!(
            c.find_position("<read_file><path>/etc/hosts"),
            XmlPos::InValue(0)
        );
    }

    #[test]
    fn position_partial_close() {
        let c = make_constraint(read_file_schema());
        assert_eq!(
            c.find_position("<read_file><path>/etc/hosts</pa"),
            XmlPos::PartialFieldClose {
                field_idx: 0,
                consumed: 4, // "</pa" = 4 chars
            }
        );
    }

    #[test]
    fn position_between_fields() {
        let c = make_constraint(read_file_schema());
        assert_eq!(
            c.find_position("<read_file><path>/etc/hosts</path>"),
            XmlPos::BetweenFields(1)
        );
    }

    #[test]
    fn position_after_all_fields() {
        let c = make_constraint(read_file_schema());
        assert_eq!(
            c.find_position("<read_file><path>/etc/hosts</path><line_count>10</line_count>"),
            XmlPos::BetweenFields(2)
        );
    }

    #[test]
    fn position_complete() {
        let c = make_constraint(read_file_schema());
        assert_eq!(
            c.find_position("<read_file><path>/etc/hosts</path></read_file>"),
            XmlPos::Complete
        );
    }

    // --- forced_continuation tests ---

    #[test]
    fn forced_from_empty() {
        let c = make_constraint(read_file_schema());
        // Empty → force root open + first required field open
        assert_eq!(
            c.forced_continuation(""),
            Some("<read_file><path>".to_string())
        );
    }

    #[test]
    fn forced_from_partial_root() {
        let c = make_constraint(read_file_schema());
        assert_eq!(
            c.forced_continuation("<read"),
            Some("_file><path>".to_string())
        );
    }

    #[test]
    fn forced_after_root_open() {
        let c = make_constraint(read_file_schema());
        // First field is required → force its open tag
        assert_eq!(
            c.forced_continuation("<read_file>"),
            Some("<path>".to_string())
        );
    }

    #[test]
    fn forced_in_value_returns_none() {
        let c = make_constraint(read_file_schema());
        assert_eq!(
            c.forced_continuation("<read_file><path>/etc/hosts"),
            None
        );
    }

    #[test]
    fn forced_partial_close_tag() {
        let c = make_constraint(read_file_schema());
        // Partway through </path> — complete it.
        // Next field (line_count) is optional → don't force it.
        assert_eq!(
            c.forced_continuation("<read_file><path>/etc/hosts</pa"),
            Some("th>".to_string())
        );
    }

    #[test]
    fn forced_between_fields_optional_next() {
        let c = make_constraint(read_file_schema());
        // After </path>, next field is optional → model decides
        assert_eq!(
            c.forced_continuation("<read_file><path>/etc/hosts</path>"),
            None
        );
    }

    #[test]
    fn forced_after_all_fields() {
        let c = make_constraint(read_file_schema());
        // All fields done → force root close
        assert_eq!(
            c.forced_continuation("<read_file><path>/etc/hosts</path><line_count>10</line_count>"),
            Some("</read_file>".to_string())
        );
    }

    #[test]
    fn forced_all_required_chain() {
        let schema = ToolSchema::new("write_file")
            .required("path", ToolFieldType::String)
            .required("content", ToolFieldType::String);
        let c = make_constraint(schema);

        // After </path>, next field is required → force its open tag
        assert_eq!(
            c.forced_continuation("<write_file><path>foo.rs</path>"),
            Some("<content>".to_string())
        );

        // After </content>, no more fields → force root close
        assert_eq!(
            c.forced_continuation("<write_file><path>foo.rs</path><content>hello</content>"),
            Some("</write_file>".to_string())
        );
    }

    #[test]
    fn forced_partial_close_then_required() {
        let schema = ToolSchema::new("write_file")
            .required("path", ToolFieldType::String)
            .required("content", ToolFieldType::String);
        let c = make_constraint(schema);

        // Partial close tag for path — complete it + force next required field
        assert_eq!(
            c.forced_continuation("<write_file><path>foo.rs</"),
            Some("path><content>".to_string())
        );
    }

    #[test]
    fn forced_complete_returns_none() {
        let c = make_constraint(read_file_schema());
        assert_eq!(
            c.forced_continuation("<read_file><path>/etc/hosts</path></read_file>"),
            None
        );
    }

    // --- InValue fast-path tests ---

    #[test]
    fn in_value_string_no_lt() {
        let c = make_constraint(read_file_schema());
        let text = "<read_file><path>";
        // String field, no '<' → O(1) accept
        assert!(c.token_valid_in_value(text, "/etc", 0));
        assert!(c.token_valid_in_value(text, "/hosts", 0));
        assert!(c.token_valid_in_value(text, "anything_at_all", 0));
    }

    #[test]
    fn in_value_string_with_close() {
        let c = make_constraint(read_file_schema());
        let text = "<read_file><path>/etc/hosts";
        assert!(c.token_valid_in_value(text, "</", 0));
        assert!(c.token_valid_in_value(text, "</path>", 0));
        assert!(!c.token_valid_in_value(text, "</wrong>", 0));
    }

    #[test]
    fn in_value_integer() {
        let schema = ToolSchema::new("test")
            .required("count", ToolFieldType::Integer);
        let c = make_constraint(schema);
        let text = "<test><count>";
        assert!(c.token_valid_in_value(text, "42", 0));
        assert!(c.token_valid_in_value(text, "-1", 0));
        assert!(!c.token_valid_in_value(text, "abc", 0));
        assert!(c.token_valid_in_value(text, "</count>", 0));
    }

    #[test]
    fn in_value_partial_close() {
        let c = make_constraint(read_file_schema());
        let text = "<read_file><path>/etc/hosts";
        assert!(c.token_valid_in_value(text, "</pa", 0));
        assert!(!c.token_valid_in_value(text, "</wr", 0));
    }

    // --- extract_value tests ---

    #[test]
    fn extract_value_simple() {
        let c = make_constraint(read_file_schema());
        assert_eq!(c.extract_value("<read_file><path>/etc/hosts", 0), "/etc/hosts");
    }

    #[test]
    fn extract_value_empty() {
        let c = make_constraint(read_file_schema());
        assert_eq!(c.extract_value("<read_file><path>", 0), "");
    }

    #[test]
    fn extract_value_second_field() {
        let c = make_constraint(read_file_schema());
        let text = "<read_file><path>/etc/hosts</path><line_count>42";
        assert_eq!(c.extract_value(text, 1), "42");
    }


    /// Verify the incremental dispatch produces identical results to the naive approach.
    #[test]
    fn incremental_matches_naive() {
        let vocab_size = 1000;
        let c = make_constraint_with_vocab(read_file_schema(), vocab_size);

        let test_cases = [
            "",                                                          // BeforeRoot
            "<read",                                                     // PartialRootOpen
            "<read_file>",                                               // BetweenFields(0) required
            "<read_file><path>",                                         // InValue(0) empty
            "<read_file><path>/etc/hosts",                               // InValue(0) with content
            "<read_file><path>/etc/hosts</pa",                           // PartialFieldClose
            "<read_file><path>/etc/hosts</path>",                        // BetweenFields(1) optional
            "<read_file><path>/etc/hosts</path><line_count>42",          // InValue(1)
            "<read_file><path>/etc/hosts</path></read_file>",            // Complete
        ];

        for text in &test_cases {
            let naive = c.valid_tokens_naive(text, vocab_size);
            let incr = c.valid_tokens(text, vocab_size);
            assert_eq!(
                naive, incr,
                "mismatch at position {:?} for text: {text}",
                c.find_position(text)
            );
        }
    }

    // --- Benchmarks (run with `cargo test bench_ -- --ignored --nocapture`) ---

    #[test]
    #[ignore]
    fn bench_valid_tokens_in_value() {
        use std::time::Instant;

        let vocab_size = 151_936; // Qwen2.5 vocab
        let c = make_constraint_with_vocab(read_file_schema(), vocab_size);
        let text = "<read_file><path>/home/user/projects/agentos/src/main.rs";
        let iterations = 20;

        // Warm up
        let _ = c.valid_tokens(text, vocab_size);
        let _ = c.valid_tokens_naive(text, vocab_size);

        // Benchmark incremental
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = c.valid_tokens(text, vocab_size);
        }
        let incr_elapsed = start.elapsed();

        // Benchmark naive
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = c.valid_tokens_naive(text, vocab_size);
        }
        let naive_elapsed = start.elapsed();

        let incr_per = incr_elapsed / iterations as u32;
        let naive_per = naive_elapsed / iterations as u32;
        let speedup = naive_elapsed.as_nanos() as f64 / incr_elapsed.as_nanos() as f64;

        eprintln!("=== InValue benchmark (vocab={vocab_size}, text_len={}) ===", text.len());
        eprintln!("  naive:       {:?} / call", naive_per);
        eprintln!("  incremental: {:?} / call", incr_per);
        eprintln!("  speedup:     {speedup:.1}x");
    }

    #[test]
    #[ignore]
    fn bench_valid_tokens_between_fields() {
        use std::time::Instant;

        let vocab_size = 151_936;
        let c = make_constraint_with_vocab(read_file_schema(), vocab_size);
        // BetweenFields(1) — optional field, falls back to full check
        let text = "<read_file><path>/home/user/projects/agentos/src/main.rs</path>";
        let iterations = 20;

        let _ = c.valid_tokens(text, vocab_size);
        let _ = c.valid_tokens_naive(text, vocab_size);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = c.valid_tokens(text, vocab_size);
        }
        let incr_elapsed = start.elapsed();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = c.valid_tokens_naive(text, vocab_size);
        }
        let naive_elapsed = start.elapsed();

        let incr_per = incr_elapsed / iterations as u32;
        let naive_per = naive_elapsed / iterations as u32;
        let speedup = naive_elapsed.as_nanos() as f64 / incr_elapsed.as_nanos() as f64;

        eprintln!("=== BetweenFields (optional) benchmark (vocab={vocab_size}) ===");
        eprintln!("  naive:       {:?} / call", naive_per);
        eprintln!("  incremental: {:?} / call", incr_per);
        eprintln!("  speedup:     {speedup:.1}x");
    }

    #[test]
    #[ignore]
    fn bench_valid_tokens_structural() {
        use std::time::Instant;

        let vocab_size = 151_936;
        let c = make_constraint_with_vocab(read_file_schema(), vocab_size);
        // BetweenFields(0) — required field, single expected continuation
        let text = "<read_file>";
        let iterations = 20;

        let _ = c.valid_tokens(text, vocab_size);
        let _ = c.valid_tokens_naive(text, vocab_size);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = c.valid_tokens(text, vocab_size);
        }
        let incr_elapsed = start.elapsed();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = c.valid_tokens_naive(text, vocab_size);
        }
        let naive_elapsed = start.elapsed();

        let incr_per = incr_elapsed / iterations as u32;
        let naive_per = naive_elapsed / iterations as u32;
        let speedup = naive_elapsed.as_nanos() as f64 / incr_elapsed.as_nanos() as f64;

        eprintln!("=== Structural (required field) benchmark (vocab={vocab_size}) ===");
        eprintln!("  naive:       {:?} / call", naive_per);
        eprintln!("  incremental: {:?} / call", incr_per);
        eprintln!("  speedup:     {speedup:.1}x");
    }
}
