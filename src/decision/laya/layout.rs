//! What a Laya model directory must contain, and how to read it.
//!
//! There is no official ONNX release — `convaiinnovations/laya` ships PyTorch
//! weights only — so every export in the wild is laid out by whoever made it:
//!
//! | export | graph | config |
//! |---|---|---|
//! | `laya` scripts/export_onnx.py, Laya-jev | `laya.onnx` (+ `.data`) | `rl_agent_config.json` |
//! | `receptron/laya-onnx` | `laya.onnx` (+ `.data`) | `laya_config.json` |
//! | `ti3x-m/*` | `onnx/model.onnx` (+ `_data`) | `laya_config.json` |
//!
//! All of them carry `tokenizer/tokenizer.json` + `tokenizer_config.json`
//! (a few flatten them to the root). The graph's external-data file is found
//! by ONNX Runtime from the name recorded inside the graph, so files keep
//! their relative paths exactly as the export wrote them.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

use super::sequence::clamp_temperature;
use crate::decision::types::QType;

pub const GRAPH_CANDIDATES: &[&str] = &["laya.onnx", "onnx/model.onnx", "model.onnx"];
pub const CONFIG_CANDIDATES: &[&str] = &["rl_agent_config.json", "laya_config.json"];
pub const TOKENIZER_CANDIDATES: &[&str] = &["tokenizer/tokenizer.json", "tokenizer.json"];
pub const TOKENIZER_CONFIG_CANDIDATES: &[&str] =
    &["tokenizer/tokenizer_config.json", "tokenizer_config.json"];

#[derive(Debug, Clone)]
pub struct ModelLayout {
    pub dir: PathBuf,
    pub graph: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub tokenizer_config: PathBuf,
}

impl ModelLayout {
    /// Find the four files a checkpoint needs, or say which are missing.
    pub fn detect(dir: &Path) -> Result<ModelLayout> {
        let find = |candidates: &[&str]| candidates.iter().map(|c| dir.join(c)).find(|p| p.is_file());
        let graph = find(GRAPH_CANDIDATES);
        let config = find(CONFIG_CANDIDATES);
        let tokenizer = find(TOKENIZER_CANDIDATES);
        let tokenizer_config = find(TOKENIZER_CONFIG_CANDIDATES);
        let mut missing = Vec::new();
        if graph.is_none() {
            missing.push(format!("the ONNX graph ({})", GRAPH_CANDIDATES.join(" / ")));
        }
        if config.is_none() {
            missing.push(format!("the head config ({})", CONFIG_CANDIDATES.join(" / ")));
        }
        if tokenizer.is_none() {
            missing.push(format!("the tokenizer ({})", TOKENIZER_CANDIDATES.join(" / ")));
        }
        if tokenizer_config.is_none() {
            missing.push(format!(
                "the tokenizer config ({})",
                TOKENIZER_CONFIG_CANDIDATES.join(" / ")
            ));
        }
        if !missing.is_empty() {
            bail!("{} is not a Laya export: missing {}", dir.display(), missing.join(", "));
        }
        Ok(ModelLayout {
            dir: dir.to_path_buf(),
            graph: graph.unwrap(),
            config: config.unwrap(),
            tokenizer: tokenizer.unwrap(),
            tokenizer_config: tokenizer_config.unwrap(),
        })
    }
}

/// The decision head's settings: sequence budgets and the fitted temperatures.
#[derive(Debug, Clone)]
pub struct LayaConfig {
    pub max_len: usize,
    pub head_max_len: usize,
    /// Per question type (choice, score, noul), already clamped.
    pub temperature: [f64; 3],
    /// Per `type:size` bucket (see `temp_bucket`), already clamped.
    pub temperature_by_options: HashMap<String, f64>,
    /// The encoder the head was trained on, when the config names it.
    pub encoder: Option<String>,
}

impl LayaConfig {
    pub fn from_file(path: &Path) -> Result<LayaConfig> {
        let raw = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let v: Value = serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
        Self::from_value(&v).with_context(|| format!("in {}", path.display()))
    }

    pub fn from_value(v: &Value) -> Result<LayaConfig> {
        let usize_field = |name: &str, default: usize| -> Result<usize> {
            match v.get(name) {
                None | Some(Value::Null) => Ok(default),
                Some(x) => x
                    .as_u64()
                    .filter(|n| *n > 0)
                    .map(|n| n as usize)
                    .ok_or_else(|| anyhow!("`{name}` must be a positive integer")),
            }
        };
        let max_len = usize_field("max_len", 512)?;
        let head_max_len = usize_field("head_max_len", 192)?;
        if head_max_len >= max_len {
            bail!("`head_max_len` ({head_max_len}) must be smaller than `max_len` ({max_len})");
        }
        let mut temperature = [1.0; 3];
        if let Some(Value::Array(ts)) = v.get("temperature") {
            for (slot, t) in temperature.iter_mut().zip(ts) {
                *slot = clamp_temperature(t.as_f64().unwrap_or(f64::NAN));
            }
        }
        let temperature_by_options = v
            .get("temperature_by_options")
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .map(|(k, t)| (k.clone(), clamp_temperature(t.as_f64().unwrap_or(f64::NAN))))
                    .collect()
            })
            .unwrap_or_default();
        Ok(LayaConfig {
            max_len,
            head_max_len,
            temperature,
            temperature_by_options,
            encoder: v.get("encoder").and_then(Value::as_str).map(str::to_string),
        })
    }

    /// The bucket temperature when fitted, else the per-type one — the
    /// lookup `ONNXAgent._infer` does.
    pub fn temperature_for(&self, qtype: QType, k: usize) -> f64 {
        self.temperature_by_options
            .get(&super::sequence::temp_bucket(qtype, k))
            .copied()
            .unwrap_or(self.temperature[qtype.index()])
    }
}

/// Which language family a checkpoint reads well — what auto-routing needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    /// ModernBERT encoder: English only. Reads Vietnamese badly and
    /// confidently (a Vietnamese billing email scores 0.93 "spam").
    English,
    /// mmBERT encoder: 100+ languages.
    Multilingual,
}

impl ModelKind {
    /// From the config's `encoder`, else the vocabulary: ModernBERT has ~50k
    /// entries, mmBERT's Gemma-style vocabulary 256k. The exports that drop
    /// `encoder` from their config (receptron, ti3x-m) are told apart that way.
    pub fn infer(encoder: Option<&str>, vocab_size: usize) -> ModelKind {
        match encoder.map(str::to_ascii_lowercase) {
            Some(e) if e.contains("mmbert") => ModelKind::Multilingual,
            Some(e) if e.contains("modernbert") => ModelKind::English,
            _ if vocab_size >= 150_000 => ModelKind::Multilingual,
            _ => ModelKind::English,
        }
    }
}

/// The special-token *texts* a checkpoint frames its sequences with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialTokenNames {
    pub cls: String,
    pub sep: String,
    pub mask: String,
    pub pad: String,
}

impl SpecialTokenNames {
    pub fn from_file(path: &Path) -> Result<SpecialTokenNames> {
        let raw = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let v: Value = serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
        // A token is a plain string or an AddedToken object `{"content": …}`.
        let get = |name: &str| -> Result<String> {
            match v.get(name) {
                Some(Value::String(s)) if !s.is_empty() => Ok(s.clone()),
                Some(Value::Object(o)) => o
                    .get("content")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .ok_or_else(|| anyhow!("`{name}` has no content")),
                _ => Err(anyhow!("{} names no `{name}`", path.display())),
            }
        };
        Ok(SpecialTokenNames {
            cls: get("cls_token")?,
            sep: get("sep_token")?,
            mask: get("mask_token")?,
            pad: get("pad_token")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn touch(dir: &Path, rel: &str, body: &str) {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    #[test]
    fn every_known_export_layout_is_detected() {
        for (graph, config) in [
            ("laya.onnx", "rl_agent_config.json"),
            ("laya.onnx", "laya_config.json"),
            ("onnx/model.onnx", "laya_config.json"),
        ] {
            let tmp = tempfile::tempdir().unwrap();
            touch(tmp.path(), graph, "g");
            touch(tmp.path(), config, "{}");
            touch(tmp.path(), "tokenizer/tokenizer.json", "{}");
            touch(tmp.path(), "tokenizer/tokenizer_config.json", "{}");
            let layout = ModelLayout::detect(tmp.path()).unwrap();
            assert!(layout.graph.ends_with(graph));
            assert!(layout.config.ends_with(config));
        }
    }

    #[test]
    fn a_folder_that_is_not_an_export_names_what_is_missing() {
        let tmp = tempfile::tempdir().unwrap();
        touch(tmp.path(), "laya.onnx", "g");
        let err = ModelLayout::detect(tmp.path()).unwrap_err().to_string();
        assert!(err.contains("head config") && err.contains("tokenizer"), "{err}");
        assert!(!err.contains("ONNX graph"), "{err}");
    }

    #[test]
    fn config_reads_budgets_and_clamps_temperatures() {
        let cfg = LayaConfig::from_value(&json!({
            "max_len": 1024, "head_max_len": 256,
            "temperature": [1.6, "bad", 1.9],
            "temperature_by_options": {"choice:11+": 0.1006, "noul:2": 1.98},
            "encoder": "jhu-clsp/mmBERT-base"
        }))
        .unwrap();
        assert_eq!((cfg.max_len, cfg.head_max_len), (1024, 256));
        assert_eq!(cfg.temperature, [1.6, 1.0, 1.9]);
        assert_eq!(cfg.temperature_for(QType::Choice, 12), 0.5);
        assert_eq!(cfg.temperature_for(QType::Noul, 2), 1.98);
        assert_eq!(cfg.temperature_for(QType::Score, 4), 1.0);
        assert!(LayaConfig::from_value(&json!({"max_len": 100, "head_max_len": 200})).is_err());
    }

    #[test]
    fn kind_comes_from_the_encoder_or_the_vocabulary() {
        assert_eq!(ModelKind::infer(Some("answerdotai/ModernBERT-large"), 0), ModelKind::English);
        assert_eq!(ModelKind::infer(Some("jhu-clsp/mmBERT-base"), 0), ModelKind::Multilingual);
        assert_eq!(ModelKind::infer(None, 256_000), ModelKind::Multilingual);
        assert_eq!(ModelKind::infer(None, 50_368), ModelKind::English);
    }

    #[test]
    fn special_tokens_accept_strings_and_added_token_objects() {
        let tmp = tempfile::tempdir().unwrap();
        touch(
            tmp.path(),
            "t.json",
            r#"{"cls_token": "<bos>", "sep_token": {"content": "<eos>"}, "mask_token": "<mask>", "pad_token": "<pad>"}"#,
        );
        let names = SpecialTokenNames::from_file(&tmp.path().join("t.json")).unwrap();
        assert_eq!(names.sep, "<eos>");
        touch(tmp.path(), "bad.json", r#"{"cls_token": "[CLS]"}"#);
        assert!(SpecialTokenNames::from_file(&tmp.path().join("bad.json")).is_err());
    }
}
