---
name: jero-tokenizer
description: "Tokenize prompts, format chat templates (including Dory1.0 ChatML + think), parse BPE merges, and perform streaming piece detokenization with jero-tokenizer."
---

# `jero-tokenizer` Tokenization, Chat Templates & Streaming Invariants

## 1. Core Tokenizer Architecture & Invariants

- Implement decoupled tokenization abstractions (`Tokenizer`, `SimpleTokenizer`, `GgufTokenizer`) adhering to ADR-025 without dragging in engine compute or hardware execution dependencies.
- Depend on `jero-core` for vocabulary metadata (`GgufFile`, `GgufValue`, `GgufArray`) and structured diagnostic error handling (`K3Error`).
- In `GgufTokenizer`, extract vocabulary tokens, scores, token types, BPE merge ranks, byte fallback tokens (`<0xXX>`), and special control tokens (`bos_token_id`, `eos_token_id`, `eot_token_id`, `padding_token_id`).

## 2. Streaming Detokenization & Chat Templates

- Stateful incremental token-to-UTF-8 decoding via `StreamingDetokenizer` must buffer incomplete multi-byte UTF-8 sequences across token boundaries without emitting `U+FFFD` replacement characters.
- Detect stop sequences (`<|im_end|>`, `</s>`, etc.) without leaking stop sequence prefixes or markers into user-visible output streams.
- Format user prompts according to architecture-specific templates:
  - Standard ChatML (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`)
  - NVIDIA Nemotron-Labs Dory1.0 ChatML + `<think>` formatting (`<s><|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n`) with optional reasoning effort suffixes (`" {reasoning effort: efficient}"`)

## 3. Rust Integer Sizing (`usize` vs. `u32`)

- Use `usize` for collection lengths, buffer offsets, and slice indexing.
- Use `u32` for token IDs and vocabulary indices.
