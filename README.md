# codeLlm

LSP-constrained local code generation. A Claude Code plugin that delegates code writing to a small local model where every token is validated by a language server.

## The Idea

Current LLMs hallucinate APIs, produce syntax errors, reference nonexistent symbols. They memorize code facts in billions of parameters, then get them wrong anyway.

codeLlm takes a different approach: wire the language server directly into the token prediction loop. The model generates code, but at every uncertain decision point — field access, method call, type annotation — the LSP constrains the output to only valid continuations. The result is a model that **cannot produce invalid code**.

## How It Works

1. **Multi-token prediction** with confidence thresholds (based on [mtp-lm](https://github.com/jwkirchenbauer/mtp-lm))
2. **High confidence (>90%)** → emit multiple tokens at once. Boilerplate, common patterns.
3. **Low confidence** → query the LSP for valid completions at this cursor position
4. **Mask logits** to only allow valid tokens, sample from constrained distribution
5. **Confidence recovers**, generation accelerates again

The LSP only fires when the model is uncertain — exactly when it's most likely to hallucinate — and exactly when type information resolves the ambiguity.

## Key Insight: It Can Only Answer in Code

LSP constraints are meaningless for natural language. This model is structurally incapable of generating English explanations. It's a **pure code function**: task in, valid code out.

This is the feature, not the limitation. The conversational layer belongs to the orchestrator (Claude, Opus). The coding layer belongs to codeLlm. Clean separation.

## Architecture

```
Claude Code (cloud, orchestrator)
  └── MCP tool: codeLlm
        ├── local model (small, code-focused, MTP-enabled)
        ├── tree-sitter (incremental parse state)
        └── LSP (rust-analyzer, pyright, etc.)
```

### As a Claude Code Plugin (MCP Server)

```json
{
  "name": "generate_code",
  "parameters": {
    "language": "rust",
    "task": "implement the Handler trait for WebSearchTool",
    "context": "// surrounding code, imports, types in scope"
  }
}
```

Returns: guaranteed-valid code block.

### As an AgentOS Listener

Same engine, different wrapper. The local model becomes a callable organism in the AgentOS pipeline — Opus orchestrates, codeLlm generates structurally correct code.

## Why Small Models Work Here

Current code models (7B-34B parameters) spend a huge chunk of their parameter budget memorizing API surfaces — every method on every type in every library. If the LSP handles that lookup at inference time, the model only needs to learn **code patterns**, not **code facts**.

Hypothesis: you can get away with a dramatically smaller model. Small enough for a Raspberry Pi.

Rust is the ideal first target — the type system is so rich that the LSP constraint signal is incredibly strong. rust-analyzer knows exactly what's valid at every cursor position.

## Components

- **Inference engine** — Modified llama.cpp (or similar) with logit constraint injection hook
- **LSP bridge** — Translates valid completions → token IDs → logit mask
- **Parse state** — tree-sitter incremental parsing alongside generation
- **MCP server** — Claude Code plugin interface
- **Model** — Fine-tuned small code model with MTP objective

## Prior Art

- [mtp-lm](https://github.com/jwkirchenbauer/mtp-lm) — Multi-token prediction via self-distillation, ConfAdapt mechanism
- [L-MTP](https://github.com/Xiaohao-Liu/L-MTP) — Leap multi-token prediction (NeurIPS 2025)
- [PICARD](https://arxiv.org/abs/2109.05093) — Constrained decoding for SQL via incremental parser
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Local inference with grammar-constrained decoding (GBNF)

## Status

Research prototype. Design doc at `C:\Users\Daniel\.claude\projects\C--src-BestCode\memory\lsp-constrained-codegen.md`.

## License

TBD
