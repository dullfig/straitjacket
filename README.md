# straitjacket

Grammar-constrained decoding for local LLMs. A small model wearing this **cannot emit
structurally invalid output** — not "is unlikely to," cannot.

Pure Rust on [candle](https://github.com/huggingface/candle). No C++ deps, no cmake, no LLVM.

## The idea

Asking a 0.5B model to emit a well-formed tool call and hoping is a losing game. Prompt it
harder, add few-shot examples, retry on parse failure — you are still negotiating with a
probability distribution.

Don't negotiate. Constrain the decoder.

At every step, the grammar knows which tokens could legally come next. Mask the logits for
everything else to `-inf` and sample from what remains. Invalid output isn't rejected after
the fact; it is never reachable. The model has exactly as much freedom as the schema allows
and not one token more.

That's the name. A straitjacket doesn't ask you not to move.

## How it works

```
ToolSchema  ──▶  grammar (GBNF / Lark)  ──▶  DecodingConstraint
                                                    │
   prompt ──▶ candle inference ──▶ logits ──▶ mask ─┘ ──▶ sample ──▶ valid output
```

1. Describe the shape you want as a `ToolSchema` (fields, types, required-ness).
2. `grammar::to_gbnf` / `to_lark` derive a grammar from it.
3. `XmlSchemaConstraint` tracks parse state and, each step, computes the legal token set.
4. `InferenceEngine` masks the logits and samples only from that set.

Incremental constraint dispatch means the legal set is computed against parse state rather
than rebuilt per token — that's where the 11x over the naive implementation came from.

## Usage

```rust
use straitjacket::prelude::*;

let mut engine = InferenceEngine::from_gguf(
    "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "models/tokenizer.json",
    EngineConfig { n_ctx: 4096, n_gpu_layers: 0, seed: 0, temperature: 0.7, top_p: 0.9 },
)?;

let mut constraint = XmlSchemaConstraint::new(schema, engine.tokenizer());
let (xml, stats) = engine.complete_constrained(prompt, &mut constraint, prefix, max_tokens)?;
// xml is guaranteed to parse and to match `schema`
```

`prefix` is injected ahead of generation — pass the opening tag when you want the model to
resume inside a partially-written structure rather than start from nothing.

## Modules

| module | what it does |
|---|---|
| `constraint` | `DecodingConstraint` trait + `XmlSchemaConstraint` — the logit masking |
| `engine` | candle inference, GGUF loading, the constrained sampling loop |
| `grammar` | `to_gbnf`, `to_lark` — schema to grammar |
| `schema` | `ToolSchema`, `SchemaField`, `ToolFieldType` |

## Who uses it

[AgentOS](https://github.com/dullfig/AgentOS)'s semantic router. When the orchestrating model
produces prose describing an action instead of a proper tool call, AgentOS matches the text to
a tool and hands the fill job here — the WIT interface for that tool becomes a `ToolSchema`,
and the local model fills it under constraint, with a cloud model as fallback.

The value scales as the driving model shrinks. A frontier model formats its own tool calls; a
local 0.5B needs the jacket.

## A note on the name

This repo was called `code-llm` until 2026-08-16. It started as an experiment in
LSP-constrained code generation — wire a language server into the token loop so a local model
could not reference a symbol that doesn't exist — as part of a Claude Code workalike.

That was never built. Claude Code outgrew the premise, and the coding-agent ambition was
dropped. What survived, and what this repo has always actually contained, is the constrained
decoding machinery. The LSP idea remains sound and could be built on top of
`DecodingConstraint` — it would be one more implementation of the trait — but no LSP code has
ever existed here.

The old README described the abandoned plan rather than the working library, which nearly got
the whole thing deleted as dead weight. Hence the rename, and hence this section.

## License

See repository.
