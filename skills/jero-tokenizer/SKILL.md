---
name: jero-tokenizer
description: "Tokenize prompts, format chat templates (including Dory1.0 ChatML + think), parse BPE merges, and perform streaming piece detokenization with jero-tokenizer."
---

# `jero-tokenizer` Tokenization, Chat Templates & Streaming Invariants

## 1. Core Tokenizer Architecture & Invariants

- Implement decoupled tokenization abstractions (`Tokenizer`, `SimpleTokenizer`, `GgufTokenizer`, `PreTokenizerKind`) adhering to ADR-025 without dragging in engine compute or hardware execution dependencies.
- Depend on `jero-core` for vocabulary metadata (`GgufFile`, `GgufValue`, `GgufArray`) and structured diagnostic error handling (`K3Error`).
- In `GgufTokenizer`, extract vocabulary tokens, log-probability scores, token types, BPE merge ranks, byte fallback tokens (`<0xXX>`), GPT-2 byte-to-Unicode reversible tables (`GPT2_BYTE_TO_CHAR`), `tokenizer.ggml.pre` pre-tokenizer dispatch (`Llama3`, `Qwen2`, `DeepSeek`, `Tekken`, `DefaultGpt2`), `tokenizer.ggml.add_space_prefix`, and special control/stop tokens (`bos_token_id`, `eos_token_id`, `eos_token_ids`, `eot_token_id`, `padding_token_id`).
- Encode byte-level BPE spans with `O(k log k)` heap-priority merging (`bpe_merge_priority`) breaking equal-rank ties on leftmost index, and encode `SentencePiece` Unigram spans via forward Viterbi DP (`scores`) with `<0xXX>` byte fallback and fail-closed `K3Error::CapabilityMismatch` when required byte tokens are absent.

## 2. Streaming Detokenization & Chat Templates

- Stateful incremental token-to-UTF-8 decoding via `StreamingDetokenizer` buffers incomplete multi-byte UTF-8 sequences across token boundaries (`DetokenizeStep::Hold`), replaces genuinely malformed byte sequences or truncated EOF tails with `U+FFFD`, and strips a single synthetic `SentencePiece` dummy space at stream start only when `add_space_prefix` is active.
- Detect stop sequences (`<|im_end|>`, `</s>`, `<|eot_id|>`, `<|eom_id|>`, `<|end|>`, `<end_of_turn>`, `<｜end▁of▁sentence｜>`, control/user-defined `<|assistant|>`) without leaking stop sequence prefixes or markers into user-visible output streams.
- Format user prompts according to architecture-specific templates:
  - Standard ChatML (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`)
  - NVIDIA Nemotron-Labs Dory1.0 ChatML + `<think>` formatting (`<s><|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n`) with optional reasoning effort suffixes (`" {reasoning effort: efficient}"`)

## 3. Rust Integer Sizing (`usize` vs. `u32`)

- Use `usize` for collection lengths, buffer offsets, and slice indexing.
- Use `u32` for token IDs and vocabulary indices.
