//! Standalone tool schema types — no AgentOS dependency.
//! AgentOS converts its WIT `ToolInterface` → codeLlm `ToolSchema` at the integration boundary.

/// A tool's parameter schema, used to generate GBNF grammars.
#[derive(Debug, Clone)]
pub struct ToolSchema {
    /// The XML root tag name (e.g., "read_file").
    pub root_tag: String,
    /// Ordered list of fields.
    pub fields: Vec<SchemaField>,
}

/// A single field in a tool schema.
#[derive(Debug, Clone)]
pub struct SchemaField {
    /// Field name (used as XML tag).
    pub name: String,
    /// Whether this field must be present.
    pub required: bool,
    /// The field's value type.
    pub field_type: ToolFieldType,
}

/// Primitive types a schema field can hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFieldType {
    String,
    Integer,
    Boolean,
    Float,
}

impl ToolSchema {
    /// Create a new schema with the given root tag and no fields.
    pub fn new(root_tag: impl Into<String>) -> Self {
        Self {
            root_tag: root_tag.into(),
            fields: Vec::new(),
        }
    }

    /// Add a required field.
    pub fn required(mut self, name: impl Into<String>, field_type: ToolFieldType) -> Self {
        self.fields.push(SchemaField {
            name: name.into(),
            required: true,
            field_type,
        });
        self
    }

    /// Add an optional field.
    pub fn optional(mut self, name: impl Into<String>, field_type: ToolFieldType) -> Self {
        self.fields.push(SchemaField {
            name: name.into(),
            required: false,
            field_type,
        });
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_builder() {
        let schema = ToolSchema::new("read_file")
            .required("path", ToolFieldType::String)
            .optional("line_count", ToolFieldType::Integer);

        assert_eq!(schema.root_tag, "read_file");
        assert_eq!(schema.fields.len(), 2);
        assert!(schema.fields[0].required);
        assert!(!schema.fields[1].required);
    }

    #[test]
    fn field_types_eq() {
        assert_eq!(ToolFieldType::String, ToolFieldType::String);
        assert_ne!(ToolFieldType::String, ToolFieldType::Integer);
    }

    #[test]
    fn empty_schema() {
        let schema = ToolSchema::new("noop");
        assert_eq!(schema.fields.len(), 0);
        assert_eq!(schema.root_tag, "noop");
    }
}
