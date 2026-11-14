//! Golden token-ID parity and lossless round-trip detokenization tests across six synthetic
//! GGUF model families (Llama-3.2, Qwen2.5, Phi-3.5, Gemma-2, Mistral-v0.3, and `BitNet`).

use jero_core::{GgufArray, GgufFile, GgufValue};
use jero_tokenizer::{GgufTokenizer, PreTokenizerKind, Tokenizer, gpt2_byte_to_char};

fn build_gguf_v3(kvs: &[(&str, GgufValue<'_>)]) -> Vec<u8> {
    fn write_str(buf: &mut Vec<u8>, s: &str) {
        let len = u64::try_from(s.len()).expect("string len fits u64");
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3_u32.to_le_bytes());
    buf.extend_from_slice(&0_u64.to_le_bytes());
    let kv_len = u64::try_from(kvs.len()).expect("kv count fits u64");
    buf.extend_from_slice(&kv_len.to_le_bytes());

    for (key, val) in kvs {
        write_str(&mut buf, key);
        match val {
            GgufValue::Uint32(v) => {
                buf.extend_from_slice(&4_u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::Int32(v) => {
                buf.extend_from_slice(&5_u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::Bool(v) => {
                buf.extend_from_slice(&7_u32.to_le_bytes());
                buf.push(u8::from(*v));
            }
            GgufValue::String(s) => {
                buf.extend_from_slice(&8_u32.to_le_bytes());
                write_str(&mut buf, s);
            }
            GgufValue::Array(GgufArray::String(items)) => {
                buf.extend_from_slice(&9_u32.to_le_bytes());
                buf.extend_from_slice(&8_u32.to_le_bytes());
                let count = u64::try_from(items.len()).expect("array len fits u64");
                buf.extend_from_slice(&count.to_le_bytes());
                for item in items {
                    write_str(&mut buf, item);
                }
            }
            GgufValue::Array(GgufArray::Int32(items)) => {
                buf.extend_from_slice(&9_u32.to_le_bytes());
                buf.extend_from_slice(&5_u32.to_le_bytes());
                let count = u64::try_from(items.len()).expect("array len fits u64");
                buf.extend_from_slice(&count.to_le_bytes());
                for item in items {
                    buf.extend_from_slice(&item.to_le_bytes());
                }
            }
            GgufValue::Array(GgufArray::Uint32(items)) => {
                buf.extend_from_slice(&9_u32.to_le_bytes());
                buf.extend_from_slice(&4_u32.to_le_bytes());
                let count = u64::try_from(items.len()).expect("array len fits u64");
                buf.extend_from_slice(&count.to_le_bytes());
                for item in items {
                    buf.extend_from_slice(&item.to_le_bytes());
                }
            }
            GgufValue::Array(GgufArray::Float32(items)) => {
                buf.extend_from_slice(&9_u32.to_le_bytes());
                buf.extend_from_slice(&6_u32.to_le_bytes());
                let count = u64::try_from(items.len()).expect("array len fits u64");
                buf.extend_from_slice(&count.to_le_bytes());
                for item in items {
                    buf.extend_from_slice(&item.to_le_bytes());
                }
            }
            other => panic!("unsupported GgufValue variant: {other:?}"),
        }
    }

    let rem = buf.len() % 32;
    if rem != 0 {
        buf.resize(buf.len() + (32 - rem), 0);
    }
    buf
}

fn gpt2_vocab(extras: &[(&str, i32)]) -> (Vec<String>, Vec<i32>) {
    let mut tokens = Vec::with_capacity(extras.len() + 256);
    let mut types = Vec::with_capacity(extras.len() + 256);
    for &(tok, ty) in extras {
        tokens.push(tok.to_owned());
        types.push(ty);
    }
    for byte in 0_u8..=u8::MAX {
        tokens.push(gpt2_byte_to_char(byte).to_string());
        types.push(1);
    }
    (tokens, types)
}

fn sp_vocab(extras: &[(&str, i32, f32)]) -> (Vec<String>, Vec<i32>, Vec<f32>) {
    let mut tokens = Vec::with_capacity(extras.len() + 256);
    let mut types = Vec::with_capacity(extras.len() + 256);
    let mut scores = Vec::with_capacity(extras.len() + 256);
    for &(tok, ty, score) in extras {
        tokens.push(tok.to_owned());
        types.push(ty);
        scores.push(score);
    }
    for byte in 0_u8..=u8::MAX {
        tokens.push(format!("<0x{byte:02X}>"));
        types.push(6);
        scores.push(-10.0);
    }
    (tokens, types, scores)
}

fn id_of(vocab: &[String], piece: &str) -> u32 {
    u32::try_from(
        vocab
            .iter()
            .position(|candidate| candidate == piece)
            .expect("piece in synthetic vocab"),
    )
    .expect("id fits u32")
}

#[test]
fn golden_parity_llama3_2_byte_bpe() {
    let extras = [
        ("<|begin_of_text|>", 3),
        ("<|eot_id|>", 3),
        ("<|eom_id|>", 3),
        ("\u{0120}don", 1),
        ("'t", 1),
        ("\u{0120}I", 1),
        ("'ll", 1),
        ("202", 1),
    ];
    let (vocab, types) = gpt2_vocab(&extras);
    let vocab_refs: Vec<&str> = vocab.iter().map(String::as_str).collect();
    let merges = vec![
        "\u{0120} d",
        "\u{0120}d o",
        "\u{0120}do n",
        "' t",
        "\u{0120} I",
        "' l",
        "'l l",
        "2 0",
        "20 2",
    ];
    let bytes = build_gguf_v3(&[
        ("general.architecture", GgufValue::String("llama")),
        ("tokenizer.ggml.model", GgufValue::String("gpt2")),
        ("tokenizer.ggml.pre", GgufValue::String("llama3")),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(GgufArray::String(vocab_refs)),
        ),
        (
            "tokenizer.ggml.token_type",
            GgufValue::Array(GgufArray::Int32(types)),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(GgufArray::String(merges)),
        ),
        ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(0)),
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(1)),
        ("tokenizer.ggml.add_bos_token", GgufValue::Bool(true)),
    ]);
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let tok = GgufTokenizer::from_gguf(&gguf).expect("load");
    assert_eq!(tok.pre_tokenizer(), PreTokenizerKind::Llama3);
    assert!(tok.special_tokens().is_stop_token(1));
    assert!(tok.special_tokens().is_stop_token(2));

    let prompt = " don't I'll 2026\ncafé 🚀";
    let ids = tok.encode(prompt).expect("encode");
    // Expected golden prefix: BOS(0), Ġdon(3), 't(4), ĠI(5), 'll(6), Ġ(byte 0x20), 202(7), 6
    let expected_prefix = vec![
        0,
        id_of(&vocab, "\u{0120}don"),
        id_of(&vocab, "'t"),
        id_of(&vocab, "\u{0120}I"),
        id_of(&vocab, "'ll"),
        id_of(&vocab, "\u{0120}"),
        id_of(&vocab, "202"),
        id_of(&vocab, "6"),
    ];
    assert_eq!(&ids[..expected_prefix.len()], expected_prefix.as_slice());
    assert_eq!(tok.decode(&ids).expect("decode"), prompt);
}

#[test]
fn golden_parity_qwen2_5_byte_bpe() {
    let extras = [
        ("<|endoftext|>", 3),
        ("<|im_end|>", 3),
        ("<|im_start|>", 3),
        ("\u{0120}\u{0120}", 1),
        ("fn", 1),
        ("\u{0120}add", 1),
    ];
    let (vocab, types) = gpt2_vocab(&extras);
    let vocab_refs: Vec<&str> = vocab.iter().map(String::as_str).collect();
    let merges = vec![
        "\u{0120} \u{0120}",
        "f n",
        "\u{0120} a",
        "\u{0120}a d",
        "\u{0120}ad d",
    ];
    let bytes = build_gguf_v3(&[
        ("general.architecture", GgufValue::String("qwen2")),
        ("tokenizer.ggml.model", GgufValue::String("gpt2")),
        ("tokenizer.ggml.pre", GgufValue::String("qwen2")),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(GgufArray::String(vocab_refs)),
        ),
        (
            "tokenizer.ggml.token_type",
            GgufValue::Array(GgufArray::Int32(types)),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(GgufArray::String(merges)),
        ),
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(1)),
        (
            "tokenizer.ggml.eos_token_ids",
            GgufValue::Array(GgufArray::Uint32(vec![0, 1])),
        ),
    ]);
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let tok = GgufTokenizer::from_gguf(&gguf).expect("load");
    assert_eq!(tok.pre_tokenizer(), PreTokenizerKind::Qwen2);
    assert!(tok.special_tokens().is_stop_token(0));
    assert!(tok.special_tokens().is_stop_token(1));

    // Leading double space + 1-digit splitting (`2`, `0`, `2`, `6`) + newline + UTF-8.
    let prompt = "  fn add 2026\n日本";
    let ids = tok.encode(prompt).expect("encode");
    // `"  fn"` pre-tokenizes into `" "` (`Ġ`) and `" fn"`, which BPE-merges into `["Ġ", "fn"]`.
    let expected_prefix = vec![
        id_of(&vocab, "\u{0120}"),
        id_of(&vocab, "\u{0120}"),
        id_of(&vocab, "fn"),
        id_of(&vocab, "\u{0120}add"),
        id_of(&vocab, "\u{0120}"),
        id_of(&vocab, "2"),
        id_of(&vocab, "0"),
        id_of(&vocab, "2"),
        id_of(&vocab, "6"),
        id_of(&vocab, "\u{010A}"),
    ];
    assert_eq!(&ids[..expected_prefix.len()], expected_prefix.as_slice());
    assert_eq!(tok.decode(&ids).expect("decode"), prompt);
}

#[test]
fn golden_parity_phi3_5_sentencepiece_unigram() {
    let (vocab, types, scores) = sp_vocab(&[
        ("<unk>", 2, 0.0),
        ("<s>", 3, 0.0),
        ("</s>", 3, 0.0),
        ("<|end|>", 3, 0.0),
        ("\u{2581}", 1, -2.0),
        ("\u{2581}hello", 1, -1.0),
        ("\u{2581}don", 1, -1.5),
        ("'t", 1, -1.5),
        ("\u{2581}2026", 1, -1.0),
    ]);
    let vocab_refs: Vec<&str> = vocab.iter().map(String::as_str).collect();
    let bytes = build_gguf_v3(&[
        ("general.architecture", GgufValue::String("phi3")),
        ("tokenizer.ggml.model", GgufValue::String("llama")),
        ("tokenizer.ggml.add_space_prefix", GgufValue::Bool(true)),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(GgufArray::String(vocab_refs)),
        ),
        (
            "tokenizer.ggml.token_type",
            GgufValue::Array(GgufArray::Int32(types)),
        ),
        (
            "tokenizer.ggml.scores",
            GgufValue::Array(GgufArray::Float32(scores)),
        ),
        ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(1)),
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(2)),
    ]);
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let tok = GgufTokenizer::from_gguf(&gguf).expect("load");
    assert!(tok.add_space_prefix());
    assert!(tok.special_tokens().is_stop_token(3));

    let prompt = "hello don't 2026";
    let ids = tok.encode(prompt).expect("encode");
    assert_eq!(
        ids,
        vec![
            1,
            id_of(&vocab, "\u{2581}hello"),
            id_of(&vocab, "\u{2581}don"),
            id_of(&vocab, "'t"),
            id_of(&vocab, "\u{2581}2026"),
        ]
    );
    assert_eq!(tok.decode(&ids).expect("decode"), prompt);

    let utf8_prompt = "hello café\n🚀";
    let utf8_ids = tok.encode(utf8_prompt).expect("encode with byte fallback");
    assert_eq!(tok.decode(&utf8_ids).expect("decode utf8"), utf8_prompt);
}

#[test]
fn golden_parity_gemma2_sentencepiece_no_dummy_space() {
    let (vocab, types, scores) = sp_vocab(&[
        ("<pad>", 3, 0.0),
        ("<bos>", 3, 0.0),
        ("<eos>", 3, 0.0),
        ("<end_of_turn>", 3, 0.0),
        ("\u{2581}hello", 1, -1.0),
        ("hello", 1, -1.0),
        ("\u{2581}world", 1, -1.0),
    ]);
    let vocab_refs: Vec<&str> = vocab.iter().map(String::as_str).collect();
    let bytes = build_gguf_v3(&[
        ("general.architecture", GgufValue::String("gemma2")),
        ("tokenizer.ggml.model", GgufValue::String("llama")),
        ("tokenizer.ggml.add_space_prefix", GgufValue::Bool(false)),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(GgufArray::String(vocab_refs)),
        ),
        (
            "tokenizer.ggml.token_type",
            GgufValue::Array(GgufArray::Int32(types)),
        ),
        (
            "tokenizer.ggml.scores",
            GgufValue::Array(GgufArray::Float32(scores)),
        ),
        ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(1)),
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(2)),
        ("tokenizer.ggml.add_bos_token", GgufValue::Bool(true)),
    ]);
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let tok = GgufTokenizer::from_gguf(&gguf).expect("load");
    assert!(!tok.add_space_prefix());
    assert!(tok.special_tokens().is_stop_token(3));

    // Leading space stays explicit (`▁hello`), no dummy space stripped on decode (` hello world`).
    let no_lead = "hello world";
    let no_lead_ids = tok.encode(no_lead).expect("encode");
    assert_eq!(
        no_lead_ids,
        vec![1, id_of(&vocab, "hello"), id_of(&vocab, "\u{2581}world"),]
    );
    assert_eq!(tok.decode(&no_lead_ids).expect("decode"), no_lead);

    let with_lead = " hello world";
    let with_lead_ids = tok.encode(with_lead).expect("encode with leading space");
    assert_eq!(
        with_lead_ids,
        vec![
            1,
            id_of(&vocab, "\u{2581}hello"),
            id_of(&vocab, "\u{2581}world"),
        ]
    );
    assert_eq!(tok.decode(&with_lead_ids).expect("decode"), with_lead);
}

#[test]
fn golden_parity_mistral_v03_tekken() {
    let extras = [("<s>", 3), ("</s>", 3), ("foo", 1), ("Bar", 1)];
    let (vocab, types) = gpt2_vocab(&extras);
    let vocab_refs: Vec<&str> = vocab.iter().map(String::as_str).collect();
    let merges = vec!["f o", "fo o", "B a", "Ba r"];
    let bytes = build_gguf_v3(&[
        ("general.architecture", GgufValue::String("mistral")),
        ("tokenizer.ggml.model", GgufValue::String("gpt2")),
        ("tokenizer.ggml.pre", GgufValue::String("tekken")),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(GgufArray::String(vocab_refs)),
        ),
        (
            "tokenizer.ggml.token_type",
            GgufValue::Array(GgufArray::Int32(types)),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(GgufArray::String(merges)),
        ),
        ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(0)),
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(1)),
        ("tokenizer.ggml.add_bos_token", GgufValue::Bool(true)),
    ]);
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let tok = GgufTokenizer::from_gguf(&gguf).expect("load");
    assert_eq!(tok.pre_tokenizer(), PreTokenizerKind::Tekken);

    let prompt = "fooBar 2026\n🚀";
    let ids = tok.encode(prompt).expect("encode");
    let expected_prefix = vec![
        0,
        id_of(&vocab, "foo"),
        id_of(&vocab, "Bar"),
        id_of(&vocab, "\u{0120}"),
        id_of(&vocab, "2"),
        id_of(&vocab, "0"),
        id_of(&vocab, "2"),
        id_of(&vocab, "6"),
        id_of(&vocab, "\u{010A}"),
    ];
    assert_eq!(&ids[..expected_prefix.len()], expected_prefix.as_slice());
    assert_eq!(tok.decode(&ids).expect("decode"), prompt);
}

#[test]
fn golden_parity_bitnet_byte_bpe() {
    let extras = [
        ("<|begin_of_text|>", 3),
        ("<|eot_id|>", 3),
        ("\u{0120}Bit", 1),
        ("Net", 1),
        ("202", 1),
    ];
    let (vocab, types) = gpt2_vocab(&extras);
    let vocab_refs: Vec<&str> = vocab.iter().map(String::as_str).collect();
    let merges = vec![
        "\u{0120} B",
        "\u{0120}B i",
        "\u{0120}Bi t",
        "N e",
        "Ne t",
        "2 0",
        "20 2",
    ];
    let bytes = build_gguf_v3(&[
        ("general.architecture", GgufValue::String("bitnet")),
        ("tokenizer.ggml.model", GgufValue::String("gpt2")),
        ("tokenizer.ggml.pre", GgufValue::String("bitnet-bpe")),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(GgufArray::String(vocab_refs)),
        ),
        (
            "tokenizer.ggml.token_type",
            GgufValue::Array(GgufArray::Int32(types)),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(GgufArray::String(merges)),
        ),
        ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(0)),
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(1)),
        ("tokenizer.ggml.add_bos_token", GgufValue::Bool(true)),
    ]);
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let tok = GgufTokenizer::from_gguf(&gguf).expect("load");
    assert_eq!(tok.pre_tokenizer(), PreTokenizerKind::Llama3);
    assert!(tok.special_tokens().is_stop_token(1));

    let prompt = " BitNet 2026 café";
    let ids = tok.encode(prompt).expect("encode");
    let expected_prefix = vec![
        0,
        id_of(&vocab, "\u{0120}Bit"),
        id_of(&vocab, "Net"),
        id_of(&vocab, "\u{0120}"),
        id_of(&vocab, "202"),
        id_of(&vocab, "6"),
        id_of(&vocab, "\u{0120}"),
    ];
    assert_eq!(&ids[..expected_prefix.len()], expected_prefix.as_slice());
    assert_eq!(tok.decode(&ids).expect("decode"), prompt);
}
