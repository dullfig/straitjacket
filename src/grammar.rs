use crate::schema::{ToolFieldType, ToolSchema};

/// Generate a Lark grammar string that constrains output to valid XML
/// matching the given tool schema.
///
/// Lark is the grammar format used by llguidance (pure Rust constrained
/// decoding engine). This is the primary grammar format for codeLlm.
///
/// The grammar enforces:
/// - Correct root tag wrapping
/// - Required fields must be present
/// - Optional fields may be absent
/// - Fields appear in declaration order
/// - Values are typed (string, integer, boolean, float)
pub fn to_lark(schema: &ToolSchema) -> String {
    let mut lines = Vec::new();

    // Disable fast-forward (forced bytes) optimization — the ApproximateTokEnv
    // in llama-cpp-2 uses greedy tokenization that doesn't match the canonical
    // tokenization llguidance expects for deterministic sequences like XML tags.
    lines.push("%llguidance {\"no_forcing\": true}".to_string());

    // Start rule
    let field_refs: Vec<String> = schema
        .fields
        .iter()
        .enumerate()
        .map(|(i, _)| format!("field{}", i))
        .collect();

    let fields_seq = if field_refs.is_empty() {
        "WS".to_string()
    } else {
        field_refs.join(" ")
    };

    lines.push(format!(
        "start: \"<{tag}>\" WS {fields} WS \"</{tag}>\" WS",
        tag = schema.root_tag,
        fields = fields_seq,
    ));

    // Per-field rules
    for (i, field) in schema.fields.iter().enumerate() {
        let value_rule = type_rule_name(field.field_type);
        let field_body = format!(
            "\"<{name}>\" WS {value} WS \"</{name}>\" WS",
            name = field.name,
            value = value_rule,
        );

        if field.required {
            lines.push(format!("field{}: {}", i, field_body));
        } else {
            lines.push(format!("field{}:  {} |", i, field_body));
        }
    }

    // Primitive value rules (terminals in Lark use UPPER_CASE or regex)
    lines.push("STRING_VALUE: /[^<]+/".to_string());
    lines.push("INTEGER_VALUE: /\\-?[0-9]+/".to_string());
    lines.push("FLOAT_VALUE: /\\-?[0-9]+(\\.[0-9]+)?/".to_string());
    lines.push("BOOLEAN_VALUE: \"true\" | \"false\"".to_string());

    // Whitespace
    lines.push("WS: /[ \\t\\n]*/".to_string());

    lines.join("\n")
}

fn type_rule_name(ft: ToolFieldType) -> &'static str {
    match ft {
        ToolFieldType::String => "STRING_VALUE",
        ToolFieldType::Integer => "INTEGER_VALUE",
        ToolFieldType::Boolean => "BOOLEAN_VALUE",
        ToolFieldType::Float => "FLOAT_VALUE",
    }
}

/// Generate a GBNF grammar string (for reference/testing — llama.cpp's built-in
/// grammar engine has token mapping issues with large BPE vocabularies).
pub fn to_gbnf(schema: &ToolSchema) -> String {
    let mut rules = Vec::new();

    let field_refs: Vec<String> = schema
        .fields
        .iter()
        .enumerate()
        .map(|(i, _)| format!("field{}", i))
        .collect();

    let fields_seq = if field_refs.is_empty() {
        "ws".to_string()
    } else {
        field_refs.join(" ")
    };

    rules.push(format!(
        "root ::= \"<{tag}>\" ws {fields} ws \"</{tag}>\" ws",
        tag = schema.root_tag,
        fields = fields_seq,
    ));

    for (i, field) in schema.fields.iter().enumerate() {
        let value_rule = gbnf_type_rule_name(field.field_type);
        let field_body = format!(
            "\"<{name}>\" ws {value} ws \"</{name}>\" ws",
            name = field.name,
            value = value_rule,
        );

        if field.required {
            rules.push(format!("field{} ::= {}", i, field_body));
        } else {
            rules.push(format!("field{} ::= ({}) | \"\"", i, field_body));
        }
    }

    rules.push("string-value ::= [^<]+".to_string());
    rules.push("integer-value ::= \"-\"? [0-9]+".to_string());
    rules.push("float-value ::= \"-\"? [0-9]+ (\".\" [0-9]+)?".to_string());
    rules.push("boolean-value ::= \"true\" | \"false\"".to_string());
    rules.push("ws ::= [ \\t\\n]*".to_string());

    rules.join("\n")
}

fn gbnf_type_rule_name(ft: ToolFieldType) -> &'static str {
    match ft {
        ToolFieldType::String => "string-value",
        ToolFieldType::Integer => "integer-value",
        ToolFieldType::Boolean => "boolean-value",
        ToolFieldType::Float => "float-value",
    }
}

/// Parse a grammar string and return the list of rule names.
/// Works with both GBNF (::=) and Lark (:) formats.
pub fn rule_names(grammar: &str) -> Vec<String> {
    grammar
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            // GBNF format
            if let Some(idx) = line.find("::=") {
                return Some(line[..idx].trim().to_string());
            }
            // Lark format
            if let Some(idx) = line.find(':') {
                let name = line[..idx].trim();
                if !name.is_empty() && !name.contains(' ') {
                    return Some(name.to_string());
                }
            }
            None
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ToolSchema;

    fn read_file_schema() -> ToolSchema {
        ToolSchema::new("read_file")
            .required("path", ToolFieldType::String)
            .optional("line_count", ToolFieldType::Integer)
    }

    // --- Lark grammar tests ---

    #[test]
    fn lark_has_root_tag() {
        let lark = to_lark(&read_file_schema());
        assert!(lark.contains("<read_file>"));
        assert!(lark.contains("</read_file>"));
    }

    #[test]
    fn lark_has_required_field() {
        let lark = to_lark(&read_file_schema());
        let field0_line = lark.lines().find(|l| l.starts_with("field0")).unwrap();
        assert!(field0_line.contains("<path>"));
        // Required field has no empty alternative
        assert!(!field0_line.trim().ends_with('|'));
    }

    #[test]
    fn lark_has_optional_field() {
        let lark = to_lark(&read_file_schema());
        let field1_line = lark.lines().find(|l| l.starts_with("field1")).unwrap();
        assert!(field1_line.contains("<line_count>"));
        // Optional field ends with | (empty alternative)
        assert!(field1_line.trim().ends_with('|'));
    }

    #[test]
    fn lark_typed_values() {
        let lark = to_lark(&read_file_schema());
        let field0 = lark.lines().find(|l| l.starts_with("field0")).unwrap();
        assert!(field0.contains("STRING_VALUE"));
        let field1 = lark.lines().find(|l| l.starts_with("field1")).unwrap();
        assert!(field1.contains("INTEGER_VALUE"));
    }

    #[test]
    fn lark_boolean_field() {
        let schema = ToolSchema::new("toggle")
            .required("enabled", ToolFieldType::Boolean);
        let lark = to_lark(&schema);
        let field0 = lark.lines().find(|l| l.starts_with("field0")).unwrap();
        assert!(field0.contains("BOOLEAN_VALUE"));
    }

    #[test]
    fn lark_float_field() {
        let schema = ToolSchema::new("set_temp")
            .required("value", ToolFieldType::Float);
        let lark = to_lark(&schema);
        let field0 = lark.lines().find(|l| l.starts_with("field0")).unwrap();
        assert!(field0.contains("FLOAT_VALUE"));
    }

    #[test]
    fn lark_empty_schema() {
        let schema = ToolSchema::new("noop");
        let lark = to_lark(&schema);
        assert!(lark.contains("<noop>"));
        assert!(lark.contains("</noop>"));
        let start = lark.lines().next().unwrap();
        assert!(!start.contains("field"));
    }

    #[test]
    fn lark_rule_names_extraction() {
        let lark = to_lark(&read_file_schema());
        let names = rule_names(&lark);
        assert!(names.contains(&"start".to_string()));
        assert!(names.contains(&"field0".to_string()));
        assert!(names.contains(&"field1".to_string()));
    }

    // --- GBNF grammar tests (kept for reference) ---

    #[test]
    fn gbnf_has_root_tag() {
        let gbnf = to_gbnf(&read_file_schema());
        assert!(gbnf.contains("<read_file>"));
        assert!(gbnf.contains("</read_file>"));
    }

    #[test]
    fn gbnf_rule_names() {
        let gbnf = to_gbnf(&read_file_schema());
        let names = rule_names(&gbnf);
        assert!(names.contains(&"root".to_string()));
        assert!(names.contains(&"field0".to_string()));
    }
}
