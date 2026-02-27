use crate::schema::{ToolFieldType, ToolSchema};

/// Generate a GBNF grammar string that constrains output to valid XML
/// matching the given tool schema.
///
/// The grammar enforces:
/// - Correct root tag wrapping
/// - Required fields must be present
/// - Optional fields may be absent
/// - Fields appear in declaration order (GBNF lacks unordered set semantics)
/// - Values are typed (string, integer, boolean, float)
pub fn to_gbnf(schema: &ToolSchema) -> String {
    let mut rules = Vec::new();

    // Root rule: <root_tag> fields </root_tag>
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

    // Per-field rules
    for (i, field) in schema.fields.iter().enumerate() {
        let value_rule = type_rule_name(field.field_type);
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

    // Primitive value rules
    rules.push("string-value ::= [^<]+".to_string());
    rules.push("integer-value ::= \"-\"? [0-9]+".to_string());
    rules.push(
        "float-value ::= \"-\"? [0-9]+ (\".\" [0-9]+)?".to_string(),
    );
    rules.push("boolean-value ::= \"true\" | \"false\"".to_string());

    // Whitespace
    rules.push("ws ::= [ \\t\\n]*".to_string());

    rules.join("\n")
}

fn type_rule_name(ft: ToolFieldType) -> &'static str {
    match ft {
        ToolFieldType::String => "string-value",
        ToolFieldType::Integer => "integer-value",
        ToolFieldType::Boolean => "boolean-value",
        ToolFieldType::Float => "float-value",
    }
}

/// Parse a GBNF grammar string and return the list of rule names.
/// Useful for validation and debugging.
pub fn rule_names(gbnf: &str) -> Vec<String> {
    gbnf.lines()
        .filter_map(|line| {
            let line = line.trim();
            line.find("::=").map(|idx| line[..idx].trim().to_string())
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

    #[test]
    fn gbnf_has_root_tag() {
        let gbnf = to_gbnf(&read_file_schema());
        assert!(gbnf.contains("<read_file>"));
        assert!(gbnf.contains("</read_file>"));
    }

    #[test]
    fn gbnf_has_required_field() {
        let gbnf = to_gbnf(&read_file_schema());
        // field0 is required — no alternative with ""
        let field0_line = gbnf.lines().find(|l| l.starts_with("field0")).unwrap();
        assert!(field0_line.contains("<path>"));
        assert!(!field0_line.contains("| \"\""));
    }

    #[test]
    fn gbnf_has_optional_field() {
        let gbnf = to_gbnf(&read_file_schema());
        let field1_line = gbnf.lines().find(|l| l.starts_with("field1")).unwrap();
        assert!(field1_line.contains("<line_count>"));
        assert!(field1_line.contains("| \"\""));
    }

    #[test]
    fn gbnf_typed_values() {
        let gbnf = to_gbnf(&read_file_schema());
        // path is String → string-value
        let field0 = gbnf.lines().find(|l| l.starts_with("field0")).unwrap();
        assert!(field0.contains("string-value"));
        // line_count is Integer → integer-value
        let field1 = gbnf.lines().find(|l| l.starts_with("field1")).unwrap();
        assert!(field1.contains("integer-value"));
    }

    #[test]
    fn gbnf_boolean_field() {
        let schema = ToolSchema::new("toggle")
            .required("enabled", ToolFieldType::Boolean);
        let gbnf = to_gbnf(&schema);
        let field0 = gbnf.lines().find(|l| l.starts_with("field0")).unwrap();
        assert!(field0.contains("boolean-value"));
        assert!(gbnf.contains("\"true\" | \"false\""));
    }

    #[test]
    fn gbnf_float_field() {
        let schema = ToolSchema::new("set_temp")
            .required("value", ToolFieldType::Float);
        let gbnf = to_gbnf(&schema);
        let field0 = gbnf.lines().find(|l| l.starts_with("field0")).unwrap();
        assert!(field0.contains("float-value"));
    }

    #[test]
    fn gbnf_empty_schema() {
        let schema = ToolSchema::new("noop");
        let gbnf = to_gbnf(&schema);
        assert!(gbnf.contains("<noop>"));
        assert!(gbnf.contains("</noop>"));
        // Root should reference ws only (no field refs)
        let root = gbnf.lines().next().unwrap();
        assert!(!root.contains("field"));
    }

    #[test]
    fn rule_names_extraction() {
        let gbnf = to_gbnf(&read_file_schema());
        let names = rule_names(&gbnf);
        assert!(names.contains(&"root".to_string()));
        assert!(names.contains(&"field0".to_string()));
        assert!(names.contains(&"field1".to_string()));
        assert!(names.contains(&"ws".to_string()));
        assert!(names.contains(&"string-value".to_string()));
        assert!(names.contains(&"integer-value".to_string()));
    }
}
