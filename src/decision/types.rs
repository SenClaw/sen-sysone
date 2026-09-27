//! The `/v1/systemone` wire: one `state`, a map of typed questions, one typed
//! answer per question. Parsing mirrors `laya.Agent._check_question` and
//! `_to_internal` (laya 0.3.20), so a question Laya itself would refuse is
//! refused here with the same reason — naming the question and what to fix —
//! rather than three frames down as an index error inside the model.
//!
//! Everything is [`Json`], not `serde_json::Value`: option and key order is
//! part of what the model reads (see `crate::decision::json`).

use serde::{Deserialize, Serialize};

use super::json::{Json, OrderedMap};

/// Most rows one call may carry. The exported Laya graphs are traced for up to
/// 64 rows and `laya-serve` caps a request at the same number, so a bigger
/// request is refused rather than silently split.
pub const MAX_QUESTIONS: usize = 64;

#[derive(Debug, Clone, Deserialize)]
pub struct AskRequest {
    /// `local` or `online`; absent = the backend chosen in Settings.
    #[serde(default)]
    pub backend: Option<super::settings::Backend>,
    /// Local: a model id (loaded, or loaded on demand), absent / `"auto"` to
    /// use the default. Online: the upstream model id, absent = the configured one.
    #[serde(default)]
    pub model: Option<String>,
    /// What the questions are about: a string, an object, or an array of
    /// strings (a conversation, read newest-last).
    pub state: Json,
    /// Question id → definition, answered in this order.
    pub questions: Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QType {
    Choice,
    Score,
    Noul,
}

impl QType {
    /// The index the decision head's type embedding was trained with.
    pub fn index(self) -> usize {
        match self {
            QType::Choice => 0,
            QType::Score => 1,
            QType::Noul => 2,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Criteria {
    /// Label → description, in the caller's order. `None` means the label
    /// stands alone (laya treats `null` and `""` alike; `0` and `false` are
    /// real descriptions).
    Choice(Vec<(String, Option<Json>)>),
    /// Level descriptions, index 0 first. A level may be a structured
    /// rubric; the legend echoes it back unchanged.
    Score(Vec<Json>),
    /// Optional descriptions of the two options, and the labels the model
    /// reads in front of them. Semantic order is always `[false, true]`.
    Noul {
        false_desc: Option<Json>,
        true_desc: Option<Json>,
        false_label: String,
        true_label: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub id: String,
    pub qtype: QType,
    /// Usually a string; anything else is read by the model as JSON text.
    pub instructions: Json,
    pub criteria: Criteria,
}

impl Question {
    pub fn option_count(&self) -> usize {
        match &self.criteria {
            Criteria::Choice(c) => c.len(),
            Criteria::Score(levels) => levels.len(),
            Criteria::Noul { .. } => 2,
        }
    }

    /// Validate one definition and normalise it.
    pub fn parse(id: &str, def: &Json) -> Result<Question, String> {
        let obj = def.as_object().ok_or_else(|| {
            format!("question {id:?}: definition must be an object, got {}", def.kind())
        })?;
        let field = |name: &str| obj.iter().find(|(k, _)| k == name).map(|(_, v)| v);
        let qtype = match field("type").and_then(Json::as_str) {
            Some("choice") => QType::Choice,
            Some("score") => QType::Score,
            Some("noul") => QType::Noul,
            other => {
                return Err(format!(
                    "question {id:?}: unknown type {}; use one of choice, noul, score",
                    other.map(|s| format!("{s:?}")).unwrap_or_else(|| "(missing)".into())
                ))
            }
        };
        let instructions = field("instructions").cloned().ok_or_else(|| {
            format!("question {id:?}: no 'instructions'; add the text the model should answer")
        })?;
        if field("labels").is_some() && qtype != QType::Noul {
            return Err(format!(
                "question {id:?}: 'labels' is only supported for noul questions"
            ));
        }
        let crit = field("criteria").filter(|v| !v.is_null());

        let criteria = match qtype {
            QType::Choice => {
                let options = match crit {
                    Some(Json::Object(entries)) => entries
                        .iter()
                        .map(|(k, v)| (k.clone(), description(v)))
                        .collect::<Vec<_>>(),
                    Some(Json::Array(items)) => {
                        let mut labels: Vec<(String, Option<Json>)> = Vec::new();
                        for item in items {
                            let label = python_str(item).ok_or_else(|| {
                                format!(
                                    "question {id:?}: a choice label must be a string or a number, got {}",
                                    item.kind()
                                )
                            })?;
                            // laya builds `{c: None for c in crit}`, so a repeated
                            // label is one option, at its first position.
                            if !labels.iter().any(|(l, _)| *l == label) {
                                labels.push((label, None));
                            }
                        }
                        labels
                    }
                    _ => {
                        return Err(format!(
                            "question {id:?}: a choice question takes 'criteria' as an object of \
                             label -> description, or a list of labels"
                        ))
                    }
                };
                if options.is_empty() {
                    return Err(format!(
                        "question {id:?}: a choice question needs at least one criterion"
                    ));
                }
                Criteria::Choice(options)
            }
            QType::Score => match crit {
                Some(Json::Array(levels)) if !levels.is_empty() => Criteria::Score(levels.clone()),
                Some(Json::Array(_)) => {
                    return Err(format!(
                        "question {id:?}: a score question needs at least one level"
                    ))
                }
                _ => {
                    return Err(format!(
                        "question {id:?}: a score question takes 'criteria' as a list of level \
                         descriptions, index 0 first"
                    ))
                }
            },
            QType::Noul => {
                let (mut false_desc, mut true_desc) = (None, None);
                match crit {
                    None => {}
                    Some(Json::Object(entries)) => {
                        for (k, v) in entries {
                            match k.to_lowercase().as_str() {
                                "false" => false_desc = description(v),
                                "true" => true_desc = description(v),
                                _ => {
                                    let mut keys: Vec<String> =
                                        entries.iter().map(|(k, _)| k.to_lowercase()).collect();
                                    keys.sort();
                                    return Err(format!(
                                        "question {id:?}: a noul question takes 'criteria' keyed only \
                                         'true'/'false' (either or both, and omitted is fine), got \
                                         {keys:?}. To word the answer differently, keep 'criteria' \
                                         keyed 'true'/'false' and set 'labels' instead."
                                    ));
                                }
                            }
                        }
                    }
                    Some(_) => {
                        return Err(format!(
                            "question {id:?}: a noul question takes 'criteria' as an object with \
                             optional 'true'/'false' descriptions, or omits it"
                        ))
                    }
                }
                let (false_label, true_label) =
                    noul_labels(field("labels")).map_err(|e| format!("question {id:?}: {e}"))?;
                Criteria::Noul {
                    false_desc,
                    true_desc,
                    false_label,
                    true_label,
                }
            }
        };

        Ok(Question {
            id: id.to_string(),
            qtype,
            instructions,
            criteria,
        })
    }
}

/// Parse and validate every question, keeping the caller's order.
pub fn parse_questions(questions: &Json) -> Result<Vec<Question>, String> {
    let entries = questions
        .as_object()
        .ok_or_else(|| format!("'questions' must be an object of id -> definition, got {}", questions.kind()))?;
    if entries.is_empty() {
        return Err("no questions; add at least one to 'questions'".into());
    }
    if entries.len() > MAX_QUESTIONS {
        return Err(format!(
            "{} questions in one call; the most one call may carry is {MAX_QUESTIONS}",
            entries.len()
        ));
    }
    entries.iter().map(|(id, def)| Question::parse(id, def)).collect()
}

/// `null` and `""` mean "no description" — every other value is one.
fn description(v: &Json) -> Option<Json> {
    match v {
        Json::Null => None,
        Json::String(s) if s.is_empty() => None,
        other => Some(other.clone()),
    }
}

/// The labels a noul's two options are read under; `false` / `true` unless
/// the caller renames them. Mirrors `laya.common._resolve_noul_labels`.
fn noul_labels(labels: Option<&Json>) -> Result<(String, String), String> {
    const RULE: &str = "noul labels must map exactly 'false' and 'true' to distinct non-empty strings";
    // laya resolves an explicit `null` to the defaults too.
    let Some(labels) = labels.filter(|l| !l.is_null()) else {
        return Ok(("false".into(), "true".into()));
    };
    let entries = labels.as_object().ok_or(RULE)?;
    let get = |k: &str| entries.iter().find(|(ek, _)| ek == k).map(|(_, v)| v);
    if entries.len() != 2 {
        return Err(RULE.into());
    }
    let false_label = get("false").and_then(Json::as_str).ok_or(RULE)?.trim().to_string();
    let true_label = get("true").and_then(Json::as_str).ok_or(RULE)?.trim().to_string();
    if false_label.is_empty() || true_label.is_empty() || false_label == true_label {
        return Err(RULE.into());
    }
    Ok((false_label, true_label))
}

/// Python's `"%s" % value` for the scalar kinds a list of choice labels can
/// hold — laya keys the options by the list items themselves, so `1` is the
/// label `1` and `true` the label `True`.
fn python_str(v: &Json) -> Option<String> {
    match v {
        Json::String(s) => Some(s.clone()),
        Json::Number(n) => Some(n.to_string()),
        Json::Bool(true) => Some("True".into()),
        Json::Bool(false) => Some("False".into()),
        Json::Null => Some("None".into()),
        Json::Array(_) | Json::Object(_) => None,
    }
}

// ── The response side ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct Action {
    /// P(act) from the act/escalate head — how sure the model is that
    /// acting on this answer beats handing it to someone.
    pub act_probability: f64,
}

/// One answer, shaped like laya's `ONNXAgent` output, plus
/// `answer_confidence` = max(p): the quantity laya's temperature scaling
/// fits, unlike the entropy-based `confidence`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Answer {
    Choice {
        choice: String,
        probabilities: OrderedMap<f64>,
        confidence: f64,
        answer_confidence: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        action: Option<Action>,
    },
    Score {
        score: f64,
        legend: OrderedMap<Json>,
        probabilities: OrderedMap<f64>,
        confidence: f64,
        answer_confidence: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        action: Option<Action>,
    },
    Noul {
        noul: f64,
        confidence: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        action: Option<Action>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    /// Always 0 — nothing is generated.
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Routing {
    pub model: String,
    pub reason: String,
}

/// Answers as the local engine shapes them, or as an online backend returned
/// them (kept verbatim and in order — its fields are its own).
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Answers {
    Local(OrderedMap<Answer>),
    Online(Json),
}

#[derive(Debug, Clone, Serialize)]
pub struct AskResponse {
    pub model: String,
    /// `laya-onnx`, or `online:<provider>`.
    pub engine: String,
    pub answers: Answers,
    pub usage: Usage,
    pub latency_ms: f64,
    /// Graph runs this request took: 1 on a dynamic-batch graph, one per
    /// question on a fixed-batch one.
    pub runs: usize,
    pub routing: Routing,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse JSON text — `json!` would sort the keys before we saw them.
    fn js(text: &str) -> Json {
        serde_json::from_str(text).unwrap()
    }

    fn q(text: &str) -> Result<Question, String> {
        Question::parse("q", &js(text))
    }

    #[test]
    fn choice_from_an_object_keeps_order_and_treats_empty_as_no_description() {
        let parsed = q(r#"{"type": "choice", "instructions": "Pick",
            "criteria": {"zeta": "last letter", "alpha": "", "mid": 0, "off": false}}"#)
        .unwrap();
        assert_eq!(
            parsed.criteria,
            Criteria::Choice(vec![
                ("zeta".into(), Some(js(r#""last letter""#))),
                ("alpha".into(), None),
                ("mid".into(), Some(js("0"))),
                ("off".into(), Some(js("false"))),
            ])
        );
    }

    #[test]
    fn choice_from_a_list_uses_python_labels_and_drops_repeats() {
        let parsed = q(r#"{"type": "choice", "instructions": "Pick", "criteria": ["a", 1, true, "a", 2.5]}"#).unwrap();
        let labels: Vec<String> = match parsed.criteria {
            Criteria::Choice(c) => c.into_iter().map(|(l, _)| l).collect(),
            _ => unreachable!(),
        };
        assert_eq!(labels, vec!["a", "1", "True", "2.5"]);
    }

    #[test]
    fn a_question_laya_refuses_is_refused_with_its_name() {
        for (def, needle) in [
            (r#""not an object""#, "must be an object"),
            (r#"{"type": "rank", "instructions": "x"}"#, "unknown type"),
            (r#"{"type": "noul"}"#, "no 'instructions'"),
            (r#"{"type": "choice", "instructions": "x", "criteria": {}}"#, "at least one"),
            (r#"{"type": "choice", "instructions": "x"}"#, "takes 'criteria'"),
            (r#"{"type": "score", "instructions": "x", "criteria": {"a": 1}}"#, "list of level"),
            (r#"{"type": "score", "instructions": "x", "criteria": []}"#, "at least one level"),
            (r#"{"type": "noul", "instructions": "x", "criteria": {"yes": "y"}}"#, "keyed only"),
            (r#"{"type": "noul", "instructions": "x", "criteria": ["a"]}"#, "optional 'true'/'false'"),
            (r#"{"type": "choice", "instructions": "x", "criteria": ["a"], "labels": {}}"#, "only supported for noul"),
            (r#"{"type": "noul", "instructions": "x", "labels": {"false": "no", "true": "no"}}"#, "distinct"),
            (r#"{"type": "choice", "instructions": "x", "criteria": [["nested"]]}"#, "string or a number"),
        ] {
            let err = q(def).unwrap_err();
            assert!(err.contains("\"q\""), "{err}");
            assert!(err.contains(needle), "{def} → {err}");
        }
    }

    #[test]
    fn noul_keys_are_case_insensitive_and_labels_are_trimmed() {
        let parsed = q(r#"{"type": "noul", "instructions": "x",
            "criteria": {"TRUE": "it holds", "False": null},
            "labels": {"false": " no ", "true": "yes"}}"#)
        .unwrap();
        assert_eq!(
            parsed.criteria,
            Criteria::Noul {
                false_desc: None,
                true_desc: Some(js(r#""it holds""#)),
                false_label: "no".into(),
                true_label: "yes".into(),
            }
        );
        assert_eq!(parsed.option_count(), 2);
        // An explicit null resolves to the defaults, as in laya.
        let parsed = q(r#"{"type": "noul", "instructions": "x", "labels": null}"#).unwrap();
        assert!(matches!(parsed.criteria, Criteria::Noul { ref false_label, .. } if false_label == "false"));
    }

    #[test]
    fn a_request_is_bounded_and_keeps_question_order() {
        assert!(parse_questions(&js("{}")).unwrap_err().contains("no questions"));
        assert!(parse_questions(&js("[]")).unwrap_err().contains("must be an object"));
        let many: String = (0..=MAX_QUESTIONS)
            .map(|i| format!(r#""q{i}": {{"type": "noul", "instructions": "x"}}"#))
            .collect::<Vec<_>>()
            .join(",");
        assert!(parse_questions(&js(&format!("{{{many}}}"))).unwrap_err().contains("most one call"));
        let ids: Vec<String> = parse_questions(&js(
            r#"{"zz": {"type": "noul", "instructions": "x"}, "aa": {"type": "noul", "instructions": "y"}}"#,
        ))
        .unwrap()
        .into_iter()
        .map(|q| q.id)
        .collect();
        assert_eq!(ids, vec!["zz", "aa"]);
    }

    #[test]
    fn answers_serialize_tagged_and_in_order() {
        let mut probabilities = OrderedMap::default();
        probabilities.push("refund", 0.9);
        probabilities.push("other", 0.1);
        let a = Answer::Choice {
            choice: "refund".into(),
            probabilities,
            confidence: 0.53,
            answer_confidence: 0.9,
            action: Some(Action { act_probability: 0.7 }),
        };
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            r#"{"type":"choice","choice":"refund","probabilities":{"refund":0.9,"other":0.1},"confidence":0.53,"answer_confidence":0.9,"action":{"act_probability":0.7}}"#
        );
    }
}
