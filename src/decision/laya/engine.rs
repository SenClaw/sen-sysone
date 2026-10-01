//! Laya on ONNX Runtime — the Rust mirror of `laya.onnx_agent.ONNXAgent`.
//!
//! The graph is the whole `DecisionModel` (encoder + head): inputs
//! `input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype`;
//! outputs `logits` (one per option, uncalibrated, padded slots at −1e4) and
//! the act/escalate head. Exports disagree on two things, and both are read off
//! the graph at load time instead of assumed:
//!
//! - **batch** — laya's own exporter and `receptron` trace a dynamic batch (one
//!   row per question, one run per request); `ti3x-m` fixes it at 1, so each
//!   question is its own run;
//! - **act head** — `act_logits` (softmax here) or `act_probs` (already one).
//!
//! CPU only, arena off: sequence length changes with every request, and ORT's
//! arena would keep the high-water mark of all of them (the same reason the
//! VieNeu engine turns it off).

use std::borrow::Cow;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, ValueType};

use super::layout::{LayaConfig, ModelKind, ModelLayout, SpecialTokenNames};
use super::sequence::{
    build_sequence, entropy_confidence, round4, serialize_state, softmax_scaled, Sequence,
    Specials, Tokenize,
};
use crate::decision::json::{Json, OrderedMap};
use crate::decision::types::{Action, Answer, Criteria, Question};

const REQUIRED_INPUTS: [&str; 5] = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"];

/// Most rows one run may carry on a dynamic-batch graph — its traced cap.
const MAX_BATCH: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActOutput {
    ActLogits,
    ActProbs,
}

impl ActOutput {
    pub fn name(self) -> &'static str {
        match self {
            ActOutput::ActLogits => "act_logits",
            ActOutput::ActProbs => "act_probs",
        }
    }
}

struct Tok(tokenizers::Tokenizer);

impl Tokenize for Tok {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.0.encode(text, false).map_err(|e| anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }
}

pub struct LayaEngine {
    pub kind: ModelKind,
    pub act_output: ActOutput,
    /// The graph takes one row per run.
    pub fixed_batch: bool,
    pub max_len: usize,
    pub head_max_len: usize,
    pub load_ms: u64,
    cfg: LayaConfig,
    tok: Tok,
    specials: Specials,
    // `Session::run` takes `&mut self`; concurrent asks on one model queue here.
    session: Mutex<Session>,
}

/// One answered request, answers in question order.
pub struct Inference {
    pub answers: OrderedMap<Answer>,
    pub input_tokens: usize,
    pub runs: usize,
}

/// One question's encoder row.
struct Row {
    index: usize,
    qtype: i64,
    seq: Sequence,
}

impl LayaEngine {
    /// `threads` is ONNX Runtime's intra-op thread count. Two loads must never
    /// run at once: ONNX Runtime cannot safely create sessions concurrently on
    /// macOS (`runtime::ONE_LOAD_AT_A_TIME` says why; every caller takes turns
    /// on it).
    pub fn load(layout: &ModelLayout, threads: usize) -> Result<LayaEngine> {
        let started = Instant::now();
        let cfg = LayaConfig::from_file(&layout.config)?;
        let tokenizer = tokenizers::Tokenizer::from_file(&layout.tokenizer)
            .map_err(|e| anyhow!("loading {}: {e}", layout.tokenizer.display()))?;
        let names = SpecialTokenNames::from_file(&layout.tokenizer_config)?;
        let id = |t: &str| {
            tokenizer
                .token_to_id(t)
                .ok_or_else(|| anyhow!("special token {t:?} is not in {}", layout.tokenizer.display()))
        };
        let specials = Specials {
            cls: id(&names.cls)?,
            sep: id(&names.sep)?,
            mask: id(&names.mask)?,
            pad: id(&names.pad)?,
            mask_text: names.mask.clone(),
        };
        let kind = ModelKind::infer(cfg.encoder.as_deref(), tokenizer.get_vocab_size(true));

        let session = build_session(layout, threads)?;
        let inputs: Vec<(String, ValueType)> = session
            .inputs()
            .iter()
            .map(|i| (i.name().to_string(), i.dtype().clone()))
            .collect();
        let outputs: Vec<String> = session.outputs().iter().map(|o| o.name().to_string()).collect();
        for want in REQUIRED_INPUTS {
            if !inputs.iter().any(|(n, _)| n == want) {
                bail!(
                    "{} is not a Laya decision graph: no `{want}` input (inputs: {})",
                    layout.graph.display(),
                    inputs.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ")
                );
            }
        }
        if !outputs.iter().any(|o| o == "logits") {
            bail!("{} has no `logits` output", layout.graph.display());
        }
        let act_output = if outputs.iter().any(|o| o == "act_logits") {
            ActOutput::ActLogits
        } else if outputs.iter().any(|o| o == "act_probs") {
            ActOutput::ActProbs
        } else {
            bail!(
                "{} has neither `act_logits` nor `act_probs` (outputs: {})",
                layout.graph.display(),
                outputs.join(", ")
            );
        };
        let fixed_batch = inputs.iter().any(|(n, t)| {
            n == "input_ids"
                && matches!(t, ValueType::Tensor { shape, .. } if shape.first() == Some(&1))
        });

        Ok(LayaEngine {
            kind,
            act_output,
            fixed_batch,
            max_len: cfg.max_len,
            head_max_len: cfg.head_max_len,
            load_ms: started.elapsed().as_millis() as u64,
            cfg,
            tok: Tok(tokenizer),
            specials,
            session: Mutex::new(session),
        })
    }

    /// Answer every question about one state. Blocking — call it from
    /// `spawn_blocking`.
    pub fn infer(&self, state: &Json, questions: &[Question]) -> Result<Inference> {
        let state_text = serialize_state(state).replace(&self.specials.mask_text, " ");
        let state_ids = self.tok.encode(&state_text)?;
        let truncate_left = state.is_array();

        let mut answers: Vec<Option<Answer>> = vec![None; questions.len()];
        let mut rows: Vec<Row> = Vec::new();
        for (index, q) in questions.iter().enumerate() {
            // The exported graphs are traced for two or more options (the act
            // head takes the top two probabilities), and a one-option question
            // has only one possible answer anyway.
            if q.option_count() == 1 {
                answers[index] = Some(certain_answer(q));
                continue;
            }
            let seq = build_sequence(
                &self.tok,
                &self.specials,
                &state_ids,
                q,
                self.max_len,
                self.head_max_len,
                truncate_left,
            )?;
            if seq.markers.len() != q.option_count() {
                bail!(
                    "question {:?}: its options do not fit in head_max_len={} (only {} of {} \
                     survive); shorten the option texts or the instructions",
                    q.id,
                    self.head_max_len,
                    seq.markers.len(),
                    q.option_count()
                );
            }
            rows.push(Row {
                index,
                qtype: q.qtype.index() as i64,
                seq,
            });
        }

        let batch = if self.fixed_batch { 1 } else { MAX_BATCH };
        let mut input_tokens = 0;
        let mut runs = 0;
        for chunk in rows.chunks(batch) {
            let scored = self.run(chunk)?;
            runs += 1;
            for (row, (logits, act)) in chunk.iter().zip(scored) {
                input_tokens += row.seq.ids.len();
                answers[row.index] = Some(self.answer(&questions[row.index], &logits, act));
            }
        }

        let mut out = OrderedMap::default();
        for (q, a) in questions.iter().zip(answers) {
            out.push(q.id.clone(), a.expect("every question is answered"));
        }
        Ok(Inference {
            answers: out,
            input_tokens,
            runs,
        })
    }

    /// The encoder row one question becomes — the parity test compares it
    /// token for token with the Python agent's.
    #[cfg(test)]
    pub fn sequence_for(&self, state: &Json, q: &Question) -> Result<Sequence> {
        let state_ids = self
            .tok
            .encode(&serialize_state(state).replace(&self.specials.mask_text, " "))?;
        build_sequence(
            &self.tok,
            &self.specials,
            &state_ids,
            q,
            self.max_len,
            self.head_max_len,
            state.is_array(),
        )
    }

    /// One graph run over `rows` (padded to a common length); per row: its
    /// option logits and P(act).
    fn run(&self, rows: &[Row]) -> Result<Vec<(Vec<f32>, f64)>> {
        let n = rows.len();
        let l = rows.iter().map(|r| r.seq.ids.len()).max().unwrap_or(1);
        let kmax = rows.iter().map(|r| r.seq.markers.len()).max().unwrap_or(2).max(2);
        let mut ids = vec![i64::from(self.specials.pad); n * l];
        let mut mask = vec![0i64; n * l];
        let mut mpos = vec![0i64; n * kmax];
        let mut mmask = vec![false; n * kmax];
        let mut qtype = Vec::with_capacity(n);
        for (r, row) in rows.iter().enumerate() {
            for (j, &t) in row.seq.ids.iter().enumerate() {
                ids[r * l + j] = i64::from(t);
                mask[r * l + j] = 1;
            }
            for (j, &m) in row.seq.markers.iter().enumerate() {
                mpos[r * kmax + j] = m as i64;
                mmask[r * kmax + j] = true;
            }
            qtype.push(row.qtype);
        }
        let inputs: Vec<(Cow<'static, str>, SessionInputValue<'_>)> = vec![
            ("input_ids".into(), Tensor::from_array(([n, l], ids))?.into_dyn().into()),
            ("attention_mask".into(), Tensor::from_array(([n, l], mask))?.into_dyn().into()),
            ("marker_pos".into(), Tensor::from_array(([n, kmax], mpos))?.into_dyn().into()),
            ("marker_mask".into(), Tensor::from_array(([n, kmax], mmask))?.into_dyn().into()),
            ("qtype".into(), Tensor::from_array(([n], qtype))?.into_dyn().into()),
        ];

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("the Laya session lock is poisoned"))?;
        let outputs = session.run(inputs)?;
        let (lshape, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
        let (ashape, act) = outputs[self.act_output.name()].try_extract_tensor::<f32>()?;
        let lk = lshape.get(1).copied().unwrap_or(0) as usize;
        if lshape.first() != Some(&(n as i64)) || lk < kmax {
            bail!("unexpected `logits` shape {lshape:?} for {n} rows of {kmax} options");
        }
        if ashape.first() != Some(&(n as i64)) || ashape.get(1) != Some(&2) {
            bail!("unexpected `{}` shape {ashape:?}", self.act_output.name());
        }
        Ok(rows
            .iter()
            .enumerate()
            .map(|(r, row)| {
                let k = row.seq.markers.len();
                let option_logits = logits[r * lk..r * lk + k].to_vec();
                let pair = &act[r * 2..r * 2 + 2];
                let p_act = match self.act_output {
                    ActOutput::ActLogits => softmax_scaled(pair, 1.0)[0],
                    ActOutput::ActProbs => f64::from(pair[0]),
                };
                (option_logits, p_act)
            })
            .collect())
    }

    /// Calibrate one row and shape it like `ONNXAgent`'s answer.
    fn answer(&self, q: &Question, logits: &[f32], act: f64) -> Answer {
        let k = q.option_count();
        let p = softmax_scaled(&logits[..k], self.cfg.temperature_for(q.qtype, k));
        let action = Some(Action {
            act_probability: round4(act),
        });
        let confidence = round4(entropy_confidence(&p));
        let answer_confidence = round4(p.iter().copied().fold(0.0, f64::max));
        match &q.criteria {
            Criteria::Choice(options) => Answer::Choice {
                choice: options[argmax(&p)].0.clone(),
                probabilities: OrderedMap(
                    options.iter().zip(&p).map(|((label, _), v)| (label.clone(), round4(*v))).collect(),
                ),
                confidence,
                answer_confidence,
                action,
            },
            Criteria::Score(levels) => Answer::Score {
                score: round4(p.iter().enumerate().map(|(i, v)| i as f64 * v).sum()),
                legend: OrderedMap(levels.iter().enumerate().map(|(i, c)| (i.to_string(), c.clone())).collect()),
                probabilities: OrderedMap(p.iter().enumerate().map(|(i, v)| (i.to_string(), round4(*v))).collect()),
                confidence,
                answer_confidence,
                action,
            },
            Criteria::Noul { .. } => Answer::Noul {
                noul: round4(p[1]),
                confidence: round4(p[1].max(1.0 - p[1])),
                action,
            },
        }
    }
}

/// First maximum, as numpy's `argmax`.
fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in p.iter().enumerate() {
        if *v > p[best] {
            best = i;
        }
    }
    best
}

/// The only answer a one-option question has (Laya-jev's `DaemonOnnxAgent`).
fn certain_answer(q: &Question) -> Answer {
    match &q.criteria {
        Criteria::Choice(options) => Answer::Choice {
            choice: options[0].0.clone(),
            probabilities: OrderedMap(vec![(options[0].0.clone(), 1.0)]),
            confidence: 1.0,
            answer_confidence: 1.0,
            action: None,
        },
        Criteria::Score(levels) => Answer::Score {
            score: 0.0,
            legend: OrderedMap(vec![("0".into(), levels[0].clone())]),
            probabilities: OrderedMap(vec![("0".into(), 1.0)]),
            confidence: 1.0,
            answer_confidence: 1.0,
            action: None,
        },
        Criteria::Noul { .. } => unreachable!("a noul always has two options"),
    }
}

fn build_session(layout: &ModelLayout, threads: usize) -> Result<Session> {
    // ort's builder errors carry the builder (not Send + Sync), so they cannot
    // ride `?` into anyhow — stringify at each step, as the VieNeu engine does.
    let cpu = ort::execution_providers::CPUExecutionProvider::default()
        .with_arena_allocator(false)
        .build();
    Session::builder()
        .map_err(|e| anyhow!("onnx session builder: {e}"))?
        .with_execution_providers([cpu])
        .map_err(|e| anyhow!("onnx session builder: {e}"))?
        .with_memory_pattern(false)
        .map_err(|e| anyhow!("onnx session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("onnx session builder: {e}"))?
        .with_intra_threads(threads)
        .map_err(|e| anyhow!("onnx session builder: {e}"))?
        .with_inter_threads(1)
        .map_err(|e| anyhow!("onnx session builder: {e}"))?
        .commit_from_file(&layout.graph)
        .map_err(|e| anyhow!("onnx load: {e}"))
        .with_context(|| format!("loading {}", layout.graph.display()))
}
