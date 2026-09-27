//! Token sequence construction and answer math for Laya — a line-for-line
//! port of `laya.common` (0.3.20): `render_options`, `build_sequence`,
//! `temp_bucket`, `clamp_temperature`, `confidence_from_probs`.
//!
//! Every question becomes one encoder row:
//!
//! ```text
//! [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]
//! ```
//!
//! The decision head scores the hidden state at each `[MASK]` — the "marker"
//! positions — so the options must survive truncation or the question cannot
//! be answered. The budgets below (48 tokens per option, `head_max_len` for the
//! instruction + options block, the state filling what is left of `max_len`)
//! are the trained ones; changing any of them changes every probability.
//!
//! Nothing here touches ONNX Runtime, so it is tested in the default build
//! against a fake tokenizer, and against the real one by the parity fixture.

use anyhow::Result;

use super::pyjson;
use crate::decision::json::Json;
use crate::decision::types::{Criteria, QType, Question};

/// laya tokenizes each option with `truncation=True, max_length=48`.
pub const OPTION_TOKEN_CAP: usize = 48;

/// Below this many tokens left for the instruction, options are trimmed.
const MIN_HEAD_ROOM: isize = 16;

/// A fitted temperature outside this range is clamped. The shipped
/// `choice:11+` bucket is 0.1006, which multiplies logits ~10×: a 0.24 top
/// probability would be published as 0.99. laya refuses to apply it, and so
/// does this port.
pub const TEMP_MIN: f64 = 0.5;
pub const TEMP_MAX: f64 = 5.0;

/// Token ids for a text, without special tokens (HF `add_special_tokens=False`).
pub trait Tokenize {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
}

/// The special tokens a sequence is framed with, resolved from the
/// checkpoint's own `tokenizer_config.json` — ModernBERT uses `[CLS]`/`[SEP]`/
/// `[MASK]`, mmBERT `<bos>`/`<eos>`/`<mask>`, so none of them is hardcoded.
#[derive(Debug, Clone)]
pub struct Specials {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub pad: u32,
    /// The mask token's text, stripped out of every input so a caller cannot
    /// plant a fake marker.
    pub mask_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sequence {
    pub ids: Vec<u32>,
    /// Position of each option's `[MASK]`, in option order.
    pub markers: Vec<usize>,
}

/// One criterion as text: strings pass through, anything structured becomes
/// compact JSON (`laya.common.render_criterion`).
pub fn render_criterion(v: &Json) -> String {
    match v {
        Json::String(s) => s.clone(),
        other => pyjson::dumps(other),
    }
}

/// The state as the encoder reads it (`laya.common.serialize_state`).
pub fn serialize_state(state: &Json) -> String {
    render_criterion(state)
}

/// Option texts in label-index order; noul is always `[false, true]`.
pub fn render_options(q: &Question) -> Vec<String> {
    match &q.criteria {
        Criteria::Choice(options) => options
            .iter()
            .map(|(label, desc)| match desc {
                None => label.clone(),
                Some(d) => format!("{label}: {}", render_criterion(d)),
            })
            .collect(),
        Criteria::Score(levels) => levels
            .iter()
            .enumerate()
            .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
            .collect(),
        Criteria::Noul {
            false_desc,
            true_desc,
            false_label,
            true_label,
        } => vec![
            format!(
                "{false_label}: {}",
                false_desc
                    .as_ref()
                    .map(render_criterion)
                    .unwrap_or_else(|| "no, the statement does not hold".into())
            ),
            format!(
                "{true_label}: {}",
                true_desc
                    .as_ref()
                    .map(render_criterion)
                    .unwrap_or_else(|| "yes, the statement holds".into())
            ),
        ],
    }
}

/// Build one question's encoder row around an already-tokenized state.
///
/// `state_ids` is shared by every question of a request — laya re-tokenizes
/// the state per question, which yields the same ids. `truncate_left` keeps
/// the *end* of an over-long state (the newest turns of a conversation).
pub fn build_sequence(
    tok: &impl Tokenize,
    sp: &Specials,
    state_ids: &[u32],
    q: &Question,
    max_len: usize,
    head_max_len: usize,
    truncate_left: bool,
) -> Result<Sequence> {
    let ins = render_criterion(&q.instructions).replace(&sp.mask_text, " ");
    let mut head_ids = tok.encode(&format!("{} question: {ins}", q.qtype.as_str()))?;

    let mut opt_ids: Vec<Vec<u32>> = Vec::new();
    for opt in render_options(q) {
        let mut ids = tok.encode(&format!(" {}", opt.replace(&sp.mask_text, " ")))?;
        ids.truncate(OPTION_TOKEN_CAP);
        let mut with_marker = Vec::with_capacity(ids.len() + 1);
        with_marker.push(sp.mask);
        with_marker.extend(ids);
        opt_ids.push(with_marker);
    }

    let head = head_max_len as isize;
    let used = |o: &[Vec<u32>]| o.iter().map(Vec::len).sum::<usize>() as isize;
    let mut opt_budget = head - used(&opt_ids);
    if opt_budget < MIN_HEAD_ROOM {
        let per = ((head - MIN_HEAD_ROOM).div_euclid(opt_ids.len().max(1) as isize)).max(4) as usize;
        for o in &mut opt_ids {
            o.truncate(per);
        }
        opt_budget = head - used(&opt_ids);
    }
    head_ids.truncate(opt_budget.max(8) as usize);

    let mut ids = Vec::with_capacity(max_len);
    ids.push(sp.cls);
    ids.extend(head_ids);
    ids.push(sp.sep);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in opt_ids {
        markers.push(ids.len());
        ids.extend(o);
    }
    ids.push(sp.sep);

    let room = (max_len as isize - ids.len() as isize - 1).max(0) as usize;
    let state = if truncate_left {
        &state_ids[state_ids.len().saturating_sub(room)..]
    } else {
        &state_ids[..room.min(state_ids.len())]
    };
    ids.extend_from_slice(state);
    ids.push(sp.sep);

    ids.truncate(max_len);
    markers.retain(|&m| m < max_len);
    Ok(Sequence { ids, markers })
}

/// The temperature bucket a question reads its calibration from.
pub fn temp_bucket(qtype: QType, k: usize) -> String {
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{}:{size}", qtype.as_str())
}

/// A usable temperature: confined to `[TEMP_MIN, TEMP_MAX]`, 1.0 if not finite.
pub fn clamp_temperature(t: f64) -> f64 {
    if t.is_finite() {
        t.clamp(TEMP_MIN, TEMP_MAX)
    } else {
        1.0
    }
}

/// `softmax(logits / t)`.
pub fn softmax_scaled(logits: &[f32], t: f64) -> Vec<f64> {
    let z: Vec<f64> = logits.iter().map(|&l| f64::from(l) / t).collect();
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.into_iter().map(|v| v / sum).collect()
}

/// Normalised-entropy confidence, `1 − H(p) / ln k` — how concentrated the
/// distribution is. laya's docs are explicit that this is *not* calibrated;
/// `max(p)` is the quantity temperature scaling fits.
pub fn entropy_confidence(p: &[f64]) -> f64 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let h: f64 = -p.iter().map(|&x| x * x.clamp(1e-12, 1.0).ln()).sum::<f64>();
    (1.0 - h / (k as f64).ln()).clamp(0.0, 1.0)
}

/// Four decimals, as every laya answer is published.
pub fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

#[cfg(test)]
mod tests {
    use super::*;

    fn js(text: &str) -> Json {
        serde_json::from_str(text).unwrap()
    }

    /// One id per character, `char as u32 + 1000`, so specials (1..=4) never collide.
    struct CharTok;
    impl Tokenize for CharTok {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.chars().map(|c| c as u32 + 1000).collect())
        }
    }

    fn sp() -> Specials {
        Specials {
            cls: 1,
            sep: 2,
            mask: 3,
            pad: 4,
            mask_text: "[MASK]".into(),
        }
    }

    fn question(def: &str) -> Question {
        Question::parse("q", &js(def)).unwrap()
    }

    fn text(ids: &[u32]) -> String {
        ids.iter()
            .map(|&i| match i {
                1 => "<C>".to_string(),
                2 => "<S>".to_string(),
                3 => "<M>".to_string(),
                i => char::from_u32(i - 1000).unwrap().to_string(),
            })
            .collect()
    }

    #[test]
    fn a_row_is_head_then_marked_options_then_state() {
        let q = question(r#"{"type": "choice", "instructions": "Pick", "criteria": {"a": "x", "b": null}}"#);
        let state = CharTok.encode("hi").unwrap();
        let seq = build_sequence(&CharTok, &sp(), &state, &q, 512, 192, false).unwrap();
        assert_eq!(text(&seq.ids), "<C>choice question: Pick<S><M> a: x<M> b<S>hi<S>");
        assert_eq!(seq.markers.len(), 2);
        assert!(seq.markers.iter().all(|&m| seq.ids[m] == 3));
    }

    #[test]
    fn noul_options_are_false_then_true_with_defaults_and_labels() {
        let q = question(r#"{"type": "noul", "instructions": "Holds?", "labels": {"false": "không", "true": "có"}}"#);
        assert_eq!(
            render_options(&q),
            vec!["không: no, the statement does not hold", "có: yes, the statement holds"]
        );
    }

    #[test]
    fn structured_criteria_and_instructions_read_as_python_json() {
        let q = question(
            r#"{"type": "score", "instructions": {"task": "calm", "lang": "vi"},
                "criteria": ["angry", {"desc": "neutral", "weight": 1.5}]}"#,
        );
        assert_eq!(
            render_options(&q),
            vec!["level 0: angry", r#"level 1: {"desc": "neutral", "weight": 1.5}"#]
        );
        let seq = build_sequence(&CharTok, &sp(), &[], &q, 512, 192, false).unwrap();
        assert!(text(&seq.ids).starts_with(r#"<C>score question: {"task": "calm", "lang": "vi"}<S>"#));
    }

    #[test]
    fn a_planted_mask_token_is_blanked_everywhere() {
        let q = question(r#"{"type": "choice", "instructions": "a[MASK]b", "criteria": ["x[MASK]"]}"#);
        let seq = build_sequence(&CharTok, &sp(), &[], &q, 512, 192, false).unwrap();
        // Exactly one marker: the one this port placed, not the two planted.
        assert_eq!(seq.ids.iter().filter(|&&i| i == 3).count(), 1);
    }

    #[test]
    fn crowded_options_are_trimmed_and_the_head_keeps_eight_tokens() {
        let criteria: Vec<String> = (0..12).map(|i| format!("option {i} {}", "x".repeat(40))).collect();
        let def = serde_json::json!({"type": "choice", "instructions": "i".repeat(100), "criteria": criteria});
        let q = question(&def.to_string());
        let seq = build_sequence(&CharTok, &sp(), &[], &q, 512, 192, false).unwrap();
        // (192 - 16) // 12 = 14 tokens per option, marker included.
        let gaps: Vec<usize> = seq.markers.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(gaps.iter().all(|&g| g == 14), "{gaps:?}");
        // head budget = 192 - 12*14 = 24 → instruction trimmed to 24 tokens.
        assert_eq!(seq.markers[0], 1 + 24 + 1);
    }

    #[test]
    fn state_truncation_keeps_the_start_or_the_end() {
        let q = question(r#"{"type": "noul", "instructions": "x"}"#);
        let state = CharTok.encode(&"abcdefghij".repeat(10)).unwrap();
        let right = build_sequence(&CharTok, &sp(), &state, &q, 100, 192, false).unwrap();
        let left = build_sequence(&CharTok, &sp(), &state, &q, 100, 192, true).unwrap();
        // Frame = 1 + 16 ("noul question: x") + 1 + 40 + 32 (options) + 1 = 91,
        // so 100 - 91 - 1 = 8 state tokens fit before the closing [SEP].
        assert_eq!(right.ids.len(), 100);
        assert_eq!(left.ids.len(), 100);
        assert!(text(&right.ids).ends_with("<S>abcdefgh<S>"), "{}", text(&right.ids));
        assert!(text(&left.ids).ends_with("<S>cdefghij<S>"), "{}", text(&left.ids));
    }

    #[test]
    fn a_marker_cut_by_max_len_is_dropped() {
        let q = question(r#"{"type": "choice", "instructions": "x", "criteria": ["aaaa", "bbbb"]}"#);
        let seq = build_sequence(&CharTok, &sp(), &[], &q, 20, 192, false).unwrap();
        assert_eq!(seq.ids.len(), 20);
        assert!(seq.markers.len() < 2, "the caller must refuse this question");
    }

    #[test]
    fn buckets_temperatures_and_confidence_follow_laya() {
        assert_eq!(temp_bucket(QType::Choice, 2), "choice:2");
        assert_eq!(temp_bucket(QType::Score, 4), "score:3-5");
        assert_eq!(temp_bucket(QType::Choice, 11), "choice:11+");
        assert_eq!(clamp_temperature(0.1006), TEMP_MIN);
        assert_eq!(clamp_temperature(f64::NAN), 1.0);
        let p = softmax_scaled(&[1.0, 1.0], 1.0);
        assert!((p[0] - 0.5).abs() < 1e-12);
        assert_eq!(entropy_confidence(&[0.5, 0.5]), 0.0);
        assert_eq!(entropy_confidence(&[1.0]), 1.0);
        assert_eq!(round4(0.123456), 0.1235);
    }
}
