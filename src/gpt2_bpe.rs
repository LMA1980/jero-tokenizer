//! Reversible GPT-2 / `TikToken` byte-level BPE alphabet mapping and indexed priority merge engine.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// 256-entry lookup table mapping raw byte values `0x00..=0xFF` to the canonical GPT-2 Unicode codepoints.
///
/// Printable ASCII (`'!'..='~'`), Latin-1 upper printables (`0xA1..=0xAC`), and `0xAE..=0xFF` map to
/// identical scalar codepoints (`U+0021..=U+00FF`). The remaining 68 whitespace/control/soft-hyphen
/// bytes (including `0x20` -> `'Ġ'` `U+0120` and `0x0A` -> `'Ċ'` `U+010A`) map deterministically into
/// `U+0100..=U+0143`.
pub const GPT2_BYTE_TO_CHAR: [char; 256] = build_gpt2_byte_to_char();

const fn is_gpt2_direct_byte(byte: u8) -> bool {
    matches!(byte, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF)
}

const fn build_gpt2_byte_to_char() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut shifted_offset: u32 = 0;
    let mut byte_idx = 0usize;

    while byte_idx < 256 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "byte_idx is strictly bounded to 0..256"
        )]
        let byte = byte_idx as u8;
        let codepoint = if is_gpt2_direct_byte(byte) {
            byte as u32
        } else {
            let cp = 256 + shifted_offset;
            shifted_offset += 1;
            cp
        };
        table[byte_idx] = match char::from_u32(codepoint) {
            Some(ch) => ch,
            None => unreachable!(),
        };
        byte_idx += 1;
    }

    table
}

/// Returns `true` when `ch` belongs to the 68 shifted GPT-2 byte-unicode codepoints (`U+0100..=U+0143`).
#[must_use]
pub(crate) const fn is_gpt2_shifted_char(ch: char) -> bool {
    matches!(ch, '\u{0100}'..='\u{0143}')
}

/// Maps a raw byte `0x00..=0xFF` to its canonical GPT-2 byte-level BPE character.
#[must_use]
pub const fn gpt2_byte_to_char(byte: u8) -> char {
    GPT2_BYTE_TO_CHAR[byte as usize]
}

const GPT2_SHIFTED_CHAR_TO_BYTE: [u8; 68] = build_gpt2_shifted_char_to_byte();

const fn build_gpt2_shifted_char_to_byte() -> [u8; 68] {
    let mut reverse = [0u8; 68];
    let mut shifted_idx = 0usize;
    let mut byte_idx = 0usize;

    while byte_idx < 256 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "byte_idx is strictly bounded to 0..256"
        )]
        let byte = byte_idx as u8;
        if !is_gpt2_direct_byte(byte) {
            reverse[shifted_idx] = byte;
            shifted_idx += 1;
        }
        byte_idx += 1;
    }

    reverse
}

/// Maps a GPT-2 byte-level BPE Unicode character back to its original byte, if present in the 256-entry alphabet.
#[must_use]
pub const fn gpt2_char_to_byte(ch: char) -> Option<u8> {
    let code = ch as u32;
    if code <= 0xFF {
        #[expect(clippy::cast_possible_truncation, reason = "guarded by code <= 0xFF")]
        let b = code as u8;
        if is_gpt2_direct_byte(b) {
            return Some(b);
        }
        return None;
    }
    if code >= 0x0100 && code <= 0x0143 {
        let offset = (code - 0x0100) as usize;
        return Some(GPT2_SHIFTED_CHAR_TO_BYTE[offset]);
    }
    None
}

/// Encodes raw bytes using GPT-2's reversible `bytes_to_unicode` bijection.
#[must_use]
pub fn gpt2_bytes_to_unicode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(gpt2_byte_to_char(byte));
    }
    out
}

/// Decodes a GPT-2 `bytes_to_unicode`-mapped string back into raw UTF-8/arbitrary bytes.
///
/// # Errors
///
/// Returns `Err(ch)` if a character outside the 256-entry GPT-2 Unicode alphabet is encountered.
pub fn gpt2_unicode_to_bytes(text: &str) -> Result<Vec<u8>, char> {
    let mut out = Vec::with_capacity(text.len());
    for ch in text.chars() {
        let byte = gpt2_char_to_byte(ch).ok_or(ch)?;
        out.push(byte);
    }
    Ok(out)
}

/// Converts a raw GGUF `gpt2`-family vocabulary piece into raw bytes.
///
/// Characters belonging to the 256-entry GPT-2 byte alphabet invert directly back to their raw
/// byte; any unexpected non-GPT-2 scalar falls back to its literal UTF-8 representation so custom
/// tokens remain lossless.
#[must_use]
pub fn gpt2_piece_to_raw_bytes(piece: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(piece.len());
    let mut utf8_buf = [0u8; 4];
    for ch in piece.chars() {
        if let Some(byte) = gpt2_char_to_byte(ch) {
            out.push(byte);
        } else {
            let encoded = ch.encode_utf8(&mut utf8_buf);
            out.extend_from_slice(encoded.as_bytes());
        }
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MergeCandidate {
    rank: usize,
    left: usize,
    left_gen: u32,
    right: usize,
    right_gen: u32,
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap by reversed `(rank, left)` so the lowest rank pops first and equal ranks
        // break ties deterministically on the leftmost symbol index.
        (Reverse(self.rank), Reverse(self.left)).cmp(&(Reverse(other.rank), Reverse(other.left)))
    }
}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Applies BPE merges in `O(k log k)` using doubly-linked node arrays and a rank min-heap.
///
/// Tie-breaking matches the canonical reference left-to-right scan (smallest `rank`, then smallest
/// `left` index).
#[must_use]
pub(crate) fn bpe_merge_priority(
    text: &str,
    merge_ranks: &HashMap<(String, String), usize>,
) -> Vec<String> {
    let mut nodes: Vec<String> = text.chars().map(|c| c.to_string()).collect();
    let count = nodes.len();
    if count <= 1 {
        return nodes;
    }

    let mut prev: Vec<Option<usize>> = Vec::with_capacity(count);
    let mut next: Vec<Option<usize>> = Vec::with_capacity(count);
    for idx in 0..count {
        prev.push(idx.checked_sub(1));
        next.push(if idx + 1 < count { Some(idx + 1) } else { None });
    }
    let mut alive = vec![true; count];
    let mut generations = vec![0u32; count];

    let mut heap = BinaryHeap::with_capacity(count);
    for idx in 0..(count - 1) {
        let pair = (nodes[idx].clone(), nodes[idx + 1].clone());
        if let Some(&rank) = merge_ranks.get(&pair) {
            heap.push(MergeCandidate {
                rank,
                left: idx,
                left_gen: 0,
                right: idx + 1,
                right_gen: 0,
            });
        }
    }

    while let Some(cand) = heap.pop() {
        let left = cand.left;
        let right = cand.right;
        if !alive[left]
            || !alive[right]
            || generations[left] != cand.left_gen
            || generations[right] != cand.right_gen
            || next[left] != Some(right)
        {
            continue;
        }

        let right_str = std::mem::take(&mut nodes[right]);
        nodes[left].push_str(&right_str);
        alive[right] = false;
        generations[right] = generations[right].wrapping_add(1);
        generations[left] = generations[left].wrapping_add(1);

        let right_next = next[right];
        next[left] = right_next;
        if let Some(rr) = right_next {
            prev[rr] = Some(left);
            let pair = (nodes[left].clone(), nodes[rr].clone());
            if let Some(&rank) = merge_ranks.get(&pair) {
                heap.push(MergeCandidate {
                    rank,
                    left,
                    left_gen: generations[left],
                    right: rr,
                    right_gen: generations[rr],
                });
            }
        }

        if let Some(ll) = prev[left] {
            if alive[ll] {
                let pair = (nodes[ll].clone(), nodes[left].clone());
                if let Some(&rank) = merge_ranks.get(&pair) {
                    heap.push(MergeCandidate {
                        rank,
                        left: ll,
                        left_gen: generations[ll],
                        right: left,
                        right_gen: generations[left],
                    });
                }
            }
        }
    }

    let mut merged = Vec::new();
    let mut cursor = Some(0usize);
    while let Some(idx) = cursor {
        if alive[idx] {
            merged.push(std::mem::take(&mut nodes[idx]));
        }
        cursor = next[idx];
    }
    merged
}
