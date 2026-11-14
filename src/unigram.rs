//! Score-weighted `SentencePiece` Unigram Viterbi dynamic programming encoder.

use std::collections::HashMap;

use jero_core::K3Error;

use crate::TokenType;

#[derive(Debug, Clone, Copy)]
pub(crate) struct UnigramVocab<'a> {
    pub(crate) token_to_id: &'a HashMap<String, u32>,
    pub(crate) scores: &'a [f32],
    pub(crate) token_types: &'a [TokenType],
    pub(crate) byte_tokens: &'a [Option<u32>; 256],
    pub(crate) max_token_byte_len: usize,
    pub(crate) min_vocab_score: f64,
}

#[derive(Debug, Clone, Copy)]
struct ViterbiNode {
    score: f64,
    token_count: usize,
    prev_pos: usize,
    token_id: Option<u32>,
}

/// Encodes `text` using forward Viterbi dynamic programming over `scores`, with `<0xXX>` UTF-8
/// scalar byte fallback for uncovered characters.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if an uncovered UTF-8 character lacks one or more
/// required `<0xXX>` fallback byte tokens.
pub(crate) fn encode_unigram_viterbi(
    text: &str,
    raw_segment: &str,
    vocab: UnigramVocab<'_>,
) -> Result<Vec<u32>, K3Error> {
    if text.is_empty() {
        return Ok(Vec::new());
    }

    let len = text.len();
    let byte_penalty_per_byte = (vocab.min_vocab_score - 100.0).min(-100.0);

    let mut dp: Vec<Option<ViterbiNode>> = vec![None; len + 1];
    dp[0] = Some(ViterbiNode {
        score: 0.0,
        token_count: 0,
        prev_pos: 0,
        token_id: None,
    });

    for (start, ch) in text.char_indices() {
        let Some(curr) = dp[start] else {
            continue;
        };

        relax_vocab_transitions(&mut dp, text, start, curr, vocab);
        relax_byte_fallback_transition(
            &mut dp,
            text,
            start,
            ch,
            curr,
            vocab.byte_tokens,
            byte_penalty_per_byte,
        );
    }

    if dp[len].is_none() {
        return Err(first_unencodable_error(
            text,
            raw_segment,
            &dp,
            vocab.byte_tokens,
        ));
    }

    reconstruct_tokens(text, &dp, vocab.byte_tokens)
}

fn relax_vocab_transitions(
    dp: &mut [Option<ViterbiNode>],
    text: &str,
    start: usize,
    curr: ViterbiNode,
    vocab: UnigramVocab<'_>,
) {
    let remaining = &text[start..];
    let limit = remaining.len().min(vocab.max_token_byte_len.max(1));

    for (rel_idx, sub_ch) in remaining.char_indices() {
        let end_rel = rel_idx + sub_ch.len_utf8();
        if end_rel > limit {
            break;
        }
        let piece = &remaining[..end_rel];
        let Some(&tok_id) = vocab.token_to_id.get(piece) else {
            continue;
        };
        let Ok(tok_idx) = usize::try_from(tok_id) else {
            continue;
        };
        let ttype = vocab
            .token_types
            .get(tok_idx)
            .copied()
            .unwrap_or(TokenType::Normal);
        if ttype != TokenType::Normal {
            continue;
        }

        let raw_score = vocab.scores.get(tok_idx).copied().unwrap_or(0.0);
        let piece_score = if raw_score.is_finite() {
            f64::from(raw_score)
        } else {
            -1.0e6
        };

        let cand = ViterbiNode {
            score: curr.score + piece_score,
            token_count: curr.token_count + 1,
            prev_pos: start,
            token_id: Some(tok_id),
        };
        update_best(&mut dp[start + end_rel], cand);
    }
}

fn relax_byte_fallback_transition(
    dp: &mut [Option<ViterbiNode>],
    text: &str,
    start: usize,
    ch: char,
    curr: ViterbiNode,
    byte_tokens: &[Option<u32>; 256],
    byte_penalty_per_byte: f64,
) {
    let next_pos = start + ch.len_utf8();
    let char_bytes = &text.as_bytes()[start..next_pos];
    if char_bytes
        .iter()
        .all(|&b| byte_tokens[usize::from(b)].is_some())
    {
        #[expect(
            clippy::cast_precision_loss,
            reason = "UTF-8 scalar length is at most 4 bytes"
        )]
        let penalty = byte_penalty_per_byte * (char_bytes.len() as f64);
        let cand = ViterbiNode {
            score: curr.score + penalty,
            token_count: curr.token_count + char_bytes.len(),
            prev_pos: start,
            token_id: None,
        };
        update_best(&mut dp[next_pos], cand);
    }
}

fn reconstruct_tokens(
    text: &str,
    dp: &[Option<ViterbiNode>],
    byte_tokens: &[Option<u32>; 256],
) -> Result<Vec<u32>, K3Error> {
    let mut reversed_segments: Vec<(usize, usize, Option<u32>)> = Vec::new();
    let mut cursor = text.len();
    while cursor > 0 {
        let node = dp[cursor].ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "tokenizer".to_owned(),
            reason: format!("unigram Viterbi backtrace stalled at offset {cursor}"),
        })?;
        reversed_segments.push((node.prev_pos, cursor, node.token_id));
        cursor = node.prev_pos;
    }

    let mut out = Vec::with_capacity(reversed_segments.len());
    for &(seg_start, seg_end, maybe_tok) in reversed_segments.iter().rev() {
        if let Some(tok_id) = maybe_tok {
            out.push(tok_id);
        } else {
            for &b in &text.as_bytes()[seg_start..seg_end] {
                if let Some(byte_id) = byte_tokens[usize::from(b)] {
                    out.push(byte_id);
                }
            }
        }
    }
    Ok(out)
}

fn update_best(slot: &mut Option<ViterbiNode>, cand: ViterbiNode) {
    let replace = match *slot {
        None => true,
        Some(existing) => {
            if (cand.score - existing.score).abs() > 1.0e-12 {
                cand.score > existing.score
            } else {
                cand.token_count < existing.token_count
            }
        }
    };
    if replace {
        *slot = Some(cand);
    }
}

fn first_unencodable_error(
    text: &str,
    raw_segment: &str,
    dp: &[Option<ViterbiNode>],
    byte_tokens: &[Option<u32>; 256],
) -> K3Error {
    let mut stall_offset = 0usize;
    for (idx, _) in text.char_indices() {
        if dp[idx].is_some() {
            stall_offset = idx;
        }
    }
    let ch = text[stall_offset..].chars().next().unwrap_or('\0');
    let char_end = (stall_offset + ch.len_utf8()).min(text.len());
    let missing_byte = text.as_bytes()[stall_offset..char_end]
        .iter()
        .copied()
        .find(|&b| byte_tokens[usize::from(b)].is_none())
        .unwrap_or_else(|| text.as_bytes()[stall_offset]);

    K3Error::CapabilityMismatch {
        architecture: "tokenizer".to_owned(),
        reason: format!(
            "unable to encode character '{ch}' (byte 0x{missing_byte:02X}) in segment '{raw_segment}'"
        ),
    }
}
