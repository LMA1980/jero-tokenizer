---
name: k3-tokenizer-fallback
description: "Implement GGUF tokenization, ChatTemplate formatting, StreamingDetokenizer UTF-8 assembly, and fallback token handling with k3-tokenizer."
---

# `k3-tokenizer` Tokenization & Fallback Token Invariants

## 1. Mandatory Fallback Token Handling

- Tokenizers (`GgufTokenizer`, `SimpleTokenizer`) must always provide deterministic fallback token handling (`padding_token_id`, unknown token ID, or `<0xXX>` byte fallback tokens) when encoding out-of-vocabulary characters.
- Never allow arbitrary user or test prompts to trigger fatal `CapabilityMismatch` aborts during automated integration tests or diagnostic probes.

## 2. Streaming UTF-8 Detokenization

- Use `StreamingDetokenizer` to buffer incomplete multi-byte UTF-8 token sequences across generation steps (`DetokenizeStep`) rather than lossily replacing partial codepoints mid-stream.
