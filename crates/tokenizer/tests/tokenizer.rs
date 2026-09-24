//! The wrapper around the Hugging Face tokenizer, checked against the toy
//! model's tokenizer.json and the oracle's recorded ids.

use testutil::{fixture, load_fixture};
use tokenizer::{Settings, Tokenizer};

const PROMPT: &str = "The quick brown fox jumps over the lazy dog. 你好 🦀 fn main() {";
const IM_START: u32 = 1022;
const IM_END: u32 = 1023;
const ENDOFTEXT: u32 = 1021;

fn bytes() -> Vec<u8> {
    std::fs::read(fixture("toy-f32/model/tokenizer.json")).unwrap()
}

/// The toy's settings: no BOS, two EOS ids.
fn settings() -> Settings {
    Settings {
        bos_token_id: None,
        eos_token_ids: vec![ENDOFTEXT, IM_END],
        add_bos_token: false,
        chat_template: None,
    }
}

fn toy() -> Tokenizer {
    Tokenizer::from_bytes(&bytes(), settings()).unwrap()
}

fn oracle_ids() -> Vec<u32> {
    load_fixture::<u32>("toy-f32/oracle/token_ids.npy").data
}

// Settings

#[test]
fn settings_parse_from_manifest_json() {
    let s: Settings = serde_json::from_str(
        r#"{"bos_token_id": null, "eos_token_ids": [1021, 1023], "add_bos_token": false, "chat_template": null}"#,
    )
    .unwrap();
    assert_eq!(s.bos_token_id, None);
    assert_eq!(s.eos_token_ids, [1021, 1023]);
    assert!(!s.add_bos_token);
    assert_eq!(s.chat_template, None);

    let s: Settings = serde_json::from_str(
        r#"{"bos_token_id": 1, "eos_token_ids": [2], "add_bos_token": true, "chat_template": "{{ messages }}"}"#,
    )
    .unwrap();
    assert_eq!(s.bos_token_id, Some(1));
    assert!(s.add_bos_token);
    assert_eq!(s.chat_template.as_deref(), Some("{{ messages }}"));
}

#[test]
fn settings_reject_unknown_and_missing_fields() {
    assert!(serde_json::from_str::<Settings>(r#"{"eos_token_ids": [2], "bogus": 1}"#).is_err());
    assert!(serde_json::from_str::<Settings>(r#"{"bos_token_id": null}"#).is_err());
}

// Construction

#[test]
fn from_bytes_matches_the_oracle() {
    assert_eq!(toy().encode(PROMPT).unwrap(), oracle_ids());
}

#[test]
fn from_bytes_rejects_invalid_json() {
    assert!(Tokenizer::from_bytes(b"not a tokenizer", settings()).is_err());
}

#[test]
fn from_bytes_rejects_add_bos_without_bos_id() {
    let s = Settings {
        add_bos_token: true,
        ..settings()
    };
    assert!(Tokenizer::from_bytes(&bytes(), s).is_err());
}

#[test]
fn settings_are_kept() {
    assert_eq!(*toy().settings(), settings());
}

// Encoding

#[test]
fn encode_prepends_bos_only_when_asked() {
    let with_bos = Settings {
        bos_token_id: Some(ENDOFTEXT),
        add_bos_token: true,
        ..settings()
    };
    let tk = Tokenizer::from_bytes(&bytes(), with_bos).unwrap();
    let mut expected = vec![ENDOFTEXT];
    expected.extend(oracle_ids());
    assert_eq!(tk.encode(PROMPT).unwrap(), expected);

    // A BOS id that is present but not enabled must not be added.
    let disabled = Settings {
        bos_token_id: Some(ENDOFTEXT),
        add_bos_token: false,
        ..settings()
    };
    let tk = Tokenizer::from_bytes(&bytes(), disabled).unwrap();
    assert_eq!(tk.encode(PROMPT).unwrap(), oracle_ids());
}

// Chat prompts carry their own special tokens, so BOS is never added.
#[test]
fn encode_chat_never_prepends_bos() {
    let with_bos = Settings {
        bos_token_id: Some(ENDOFTEXT),
        add_bos_token: true,
        ..settings()
    };
    let tk = Tokenizer::from_bytes(&bytes(), with_bos).unwrap();
    assert_eq!(tk.encode_chat(PROMPT).unwrap(), oracle_ids());
}

#[test]
fn special_tokens_encode_as_single_ids() {
    let tk = toy();
    assert_eq!(tk.encode("<|im_start|>").unwrap(), [IM_START]);
    assert_eq!(tk.encode("<|im_end|>").unwrap(), [IM_END]);
    assert_eq!(tk.encode("<|endoftext|>").unwrap(), [ENDOFTEXT]);

    let ids = tk.encode_chat("<|im_start|>user\nhi<|im_end|>\n").unwrap();
    assert_eq!(ids[0], IM_START);
    assert!(ids.contains(&IM_END));
}

#[test]
fn encode_empty_string_gives_no_ids() {
    assert!(toy().encode("").unwrap().is_empty());
}

// Decoding

#[test]
fn decode_round_trips() {
    let tk = toy();
    for text in [
        PROMPT,
        "hello",
        "  leading and trailing  ",
        "tabs\tand\nnewlines",
        "日本語",
        "🦀🦀",
    ] {
        assert_eq!(
            tk.decode(&tk.encode(text).unwrap()).unwrap(),
            text,
            "{text:?}"
        );
    }
}

#[test]
fn decode_skips_special_tokens() {
    let tk = toy();
    let mut ids = vec![IM_START];
    ids.extend(tk.encode("hi").unwrap());
    ids.push(IM_END);
    assert_eq!(tk.decode(&ids).unwrap(), "hi");
}

#[test]
fn decode_empty_gives_empty() {
    assert_eq!(toy().decode(&[]).unwrap(), "");
}

// One id at a time must reassemble multi-byte characters, so the concatenated
// pieces equal the one-shot decode and no piece holds a replacement character.
#[test]
fn stream_decode_matches_one_shot() {
    let tk = toy();
    let ids = tk.encode(PROMPT).unwrap();
    let mut stream = tk.decode_stream();
    let mut out = String::new();
    for id in &ids {
        if let Some(piece) = stream.step(*id).unwrap() {
            assert!(!piece.contains('\u{FFFD}'), "{piece:?}");
            out.push_str(&piece);
        }
    }
    assert_eq!(out, PROMPT);
}

// EOS

#[test]
fn is_eos_matches_every_eos_id() {
    let tk = toy();
    assert!(tk.is_eos(ENDOFTEXT));
    assert!(tk.is_eos(IM_END));
    assert!(!tk.is_eos(IM_START));
    assert!(!tk.is_eos(0));
}

// Llama 3's tokenizer.json adds BOS through its own post-processor. The
// manifest's add_bos_token must be the only mechanism, so that processor is
// ignored. Reproduced here by giving the toy's file such a processor.
#[test]
fn ignores_the_files_own_post_processor() {
    let mut json: serde_json::Value = serde_json::from_slice(&bytes()).unwrap();
    json["post_processor"] = serde_json::json!({
        "type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
        "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
        "special_tokens": {"<|endoftext|>": {"id": "<|endoftext|>", "ids": [ENDOFTEXT], "tokens": ["<|endoftext|>"]}}
    });
    let tk = Tokenizer::from_bytes(json.to_string().as_bytes(), settings()).unwrap();
    assert_eq!(tk.encode(PROMPT).unwrap(), oracle_ids());
    assert_eq!(tk.encode_chat(PROMPT).unwrap(), oracle_ids());
}
