//! Deterministic, zero-backtracking pre-tokenization boundary splitters matching `tokenizer.ggml.pre`.

/// GGUF pre-tokenizer profile parsed from `tokenizer.ggml.pre`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizerKind {
    /// Identity / disabled pre-splitting (`SentencePiece` `llama`/`gemma` or legacy literal vocabs).
    None,
    /// Original GPT-2 / Phi-4 whitespace, letter, unbounded-digit, and contraction splitting.
    DefaultGpt2,
    /// Llama 3 / 3.1 / 3.2 / Llama-Guard 3 / `BitNet` (`1..=3` digit chunking, case-insensitive contractions).
    Llama3,
    /// Qwen2 / Qwen2.5 / Qwen3 (`1` digit per pre-token, case-insensitive contractions).
    Qwen2,
    /// `DeepSeek` (`1` digit per pre-token, isolated newline spans, contractions).
    DeepSeek,
    /// Mistral Tekken (CamelCase boundaries, `1` digit per pre-token, `/`+newline punctuation).
    Tekken,
}

impl PreTokenizerKind {
    /// Resolves `tokenizer.ggml.pre` metadata into a [`PreTokenizerKind`].
    #[must_use]
    pub fn from_gguf_pre(raw: Option<&str>) -> Option<Self> {
        let pre = raw?.trim().to_ascii_lowercase();
        if pre.is_empty() {
            return None;
        }
        Some(match pre.as_str() {
            "llama-bpe" | "llama3" | "llama-v3" | "llama-v3.1" | "llama-v3.2" | "llama-guard3"
            | "bitnet-bpe" | "falcon3" => Self::Llama3,
            "qwen2" | "qwen2.5" | "qwen3" | "smollm" | "dorna-bpe" => Self::Qwen2,
            "deepseek-llm" | "deepseek-r1" | "deepseek-r1-qwen" | "deepseek-coder"
            | "deepseek-v3" | "deepseek3" => Self::DeepSeek,
            "tekken" => Self::Tekken,
            _ => Self::DefaultGpt2,
        })
    }
}

fn is_letter(ch: char) -> bool {
    ch.is_alphabetic()
}

fn is_digit(ch: char) -> bool {
    ch.is_numeric()
}

fn is_punct_or_symbol(ch: char) -> bool {
    !ch.is_whitespace() && !is_letter(ch) && !is_digit(ch)
}

fn match_contraction_len(rest: &str, case_insensitive: bool) -> Option<usize> {
    if !rest.starts_with('\'') {
        return None;
    }
    let after = &rest[1..];
    for two in ["re", "ve", "ll"] {
        if let Some(prefix) = after.get(..2) {
            let eq = if case_insensitive {
                prefix.eq_ignore_ascii_case(two)
            } else {
                prefix == two
            };
            if eq {
                return Some(3);
            }
        }
    }
    if let Some(first) = after.chars().next() {
        let lower = if case_insensitive {
            first.to_ascii_lowercase()
        } else {
            first
        };
        if matches!(lower, 's' | 't' | 'm' | 'd') {
            return Some(1 + first.len_utf8());
        }
    }
    None
}

fn take_while_len(rest: &str, mut pred: impl FnMut(char) -> bool) -> usize {
    for (idx, ch) in rest.char_indices() {
        if !pred(ch) {
            return idx;
        }
    }
    rest.len()
}

fn take_digits_max_len(rest: &str, max_digits: usize) -> usize {
    for (digits, (idx, ch)) in rest.char_indices().enumerate() {
        if !is_digit(ch) || digits >= max_digits {
            return idx;
        }
    }
    rest.len()
}

fn match_tekken_word_len(rest: &str) -> Option<usize> {
    let first = rest.chars().next()?;
    let mut cursor = 0usize;
    if !matches!(first, '\r' | '\n') && !is_letter(first) && !is_digit(first) {
        let after_lead = &rest[first.len_utf8()..];
        if after_lead.chars().next().is_some_and(is_letter) {
            cursor = first.len_utf8();
        } else {
            return None;
        }
    } else if !is_letter(first) {
        return None;
    }

    // Consume `[Upper]*[Lower]+` when any lowercase letter follows, otherwise consume `[Upper]+`.
    let tail = &rest[cursor..];
    let mut upper_end = 0usize;
    for (idx, ch) in tail.char_indices() {
        if !is_letter(ch) || ch.is_lowercase() {
            upper_end = idx;
            break;
        }
        upper_end = idx + ch.len_utf8();
    }
    if tail[upper_end..]
        .chars()
        .next()
        .is_some_and(char::is_lowercase)
    {
        let lower_len = take_while_len(&tail[upper_end..], |c| is_letter(c) && !c.is_uppercase());
        return Some(cursor + upper_end + lower_len);
    }
    if upper_end > 0 {
        return Some(cursor + upper_end);
    }
    let lower_len = take_while_len(tail, |c| is_letter(c) && !c.is_uppercase());
    if lower_len > 0 {
        Some(cursor + lower_len)
    } else {
        None
    }
}

fn match_whitespace_chunk_len(rest: &str, separate_newlines: bool) -> usize {
    let ws_end = take_while_len(rest, char::is_whitespace);
    debug_assert!(ws_end > 0);
    let ws_slice = &rest[..ws_end];

    // Match `\s*[\r\n]+` up to the last contiguous CR/LF within the whitespace run.
    if let Some(last_nl_idx) = ws_slice.rfind(['\r', '\n']) {
        if separate_newlines {
            let leading_end = take_while_len(ws_slice, |c| c != '\r' && c != '\n');
            if leading_end > 0 {
                return leading_end;
            }
            return take_while_len(ws_slice, |c| c == '\r' || c == '\n');
        }
        let ch_len = ws_slice[last_nl_idx..]
            .chars()
            .next()
            .map_or(1, char::len_utf8);
        return last_nl_idx + ch_len;
    }

    // No newlines in this whitespace run: if non-whitespace follows and there are 2+ spaces,
    // keep the last whitespace character so the next pre-token can attach it.
    if ws_end < rest.len() {
        let mut last_char_start = 0usize;
        let mut count = 0usize;
        for (idx, _) in ws_slice.char_indices() {
            last_char_start = idx;
            count += 1;
        }
        if count >= 2 {
            return last_char_start;
        }
    }
    ws_end
}

fn next_pretoken_len(kind: PreTokenizerKind, rest: &str) -> usize {
    match kind {
        PreTokenizerKind::None => rest.len(),
        PreTokenizerKind::DefaultGpt2 => next_default_gpt2_len(rest),
        PreTokenizerKind::Llama3 => next_modern_bpe_len(rest, 3, false),
        PreTokenizerKind::Qwen2 => next_modern_bpe_len(rest, 1, false),
        PreTokenizerKind::DeepSeek => next_modern_bpe_len(rest, 1, true),
        PreTokenizerKind::Tekken => next_tekken_len(rest),
    }
}

fn next_default_gpt2_len(rest: &str) -> usize {
    if let Some(len) = match_contraction_len(rest, false) {
        return len;
    }
    let first = rest.chars().next().expect("rest is non-empty");
    let has_lead_space = first == ' ';
    let after = if has_lead_space { &rest[1..] } else { rest };

    if let Some(next_ch) = after.chars().next() {
        let prefix = usize::from(has_lead_space);
        if is_letter(next_ch) {
            return prefix + take_while_len(after, is_letter);
        }
        if is_digit(next_ch) {
            return prefix + take_while_len(after, is_digit);
        }
        if is_punct_or_symbol(next_ch) {
            return prefix + take_while_len(after, is_punct_or_symbol);
        }
    }

    if first.is_whitespace() {
        return match_whitespace_chunk_len(rest, false);
    }
    first.len_utf8()
}

fn next_modern_bpe_len(rest: &str, max_digits: usize, separate_newlines: bool) -> usize {
    if let Some(len) = match_contraction_len(rest, true) {
        return len;
    }

    let first = rest.chars().next().expect("rest is non-empty");

    // `[^\r\n\p{L}\p{N}]?\p{L}+`
    if is_letter(first) {
        return take_while_len(rest, is_letter);
    }
    if !matches!(first, '\r' | '\n') && !is_digit(first) {
        let after_lead = &rest[first.len_utf8()..];
        if after_lead.chars().next().is_some_and(is_letter) {
            return first.len_utf8() + take_while_len(after_lead, is_letter);
        }
    }

    // `\p{N}{1,3}` (Llama3) or `\p{N}` (Qwen2/DeepSeek)
    if is_digit(first) {
        return take_digits_max_len(rest, max_digits);
    }

    // ` ?[^\s\p{L}\p{N}]+[\r\n]*` (with DeepSeek keeping CR/LF separated cleanly)
    let lead_space = usize::from(first == ' ');
    let after_opt_space = &rest[lead_space..];
    if after_opt_space
        .chars()
        .next()
        .is_some_and(is_punct_or_symbol)
    {
        let punct_len = take_while_len(after_opt_space, is_punct_or_symbol);
        let nl_len = if separate_newlines {
            0
        } else {
            take_while_len(&after_opt_space[punct_len..], |c| c == '\r' || c == '\n')
        };
        return lead_space + punct_len + nl_len;
    }

    if first.is_whitespace() {
        return match_whitespace_chunk_len(rest, separate_newlines);
    }

    first.len_utf8()
}

fn next_tekken_len(rest: &str) -> usize {
    if let Some(len) = match_contraction_len(rest, true) {
        return len;
    }
    if let Some(len) = match_tekken_word_len(rest) {
        return len;
    }
    let first = rest.chars().next().expect("rest is non-empty");
    if is_digit(first) {
        return first.len_utf8();
    }
    let lead_space = usize::from(first == ' ');
    let after_opt_space = &rest[lead_space..];
    if after_opt_space
        .chars()
        .next()
        .is_some_and(is_punct_or_symbol)
    {
        let punct_len = take_while_len(after_opt_space, is_punct_or_symbol);
        let tail_len = take_while_len(&after_opt_space[punct_len..], |c| {
            matches!(c, '\r' | '\n' | '/')
        });
        return lead_space + punct_len + tail_len;
    }
    if first.is_whitespace() {
        return match_whitespace_chunk_len(rest, false);
    }
    first.len_utf8()
}

/// Splits `text` into contiguous pre-tokens according to `kind`.
///
/// The returned slices always partition `text` losslessly (`concat == text`) and never loop.
#[must_use]
pub fn split_pretokens(kind: PreTokenizerKind, text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    if kind == PreTokenizerKind::None {
        return vec![text];
    }

    let mut chunks = Vec::new();
    let mut cursor = 0usize;
    while cursor < text.len() {
        let rest = &text[cursor..];
        let step = next_pretoken_len(kind, rest).max(1);
        chunks.push(&text[cursor..cursor + step]);
        cursor += step;
    }
    chunks
}
