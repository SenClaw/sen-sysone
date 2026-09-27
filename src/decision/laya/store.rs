//! Where Laya models live and what the Settings page shows about them.
//!
//! Each model is a directory `<local-models>/laya/<id>/` holding the export's
//! files at their original relative paths, plus `senclaw-laya.json` — written
//! **last**, after every file is in place and verified. A directory without it
//! is an interrupted download or import, never an installed model.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::download::{self, JobState};
use super::layout::{LayaConfig, ModelKind, ModelLayout};
use super::runtime::{self, LoadedInfo};

pub const METADATA_FILE: &str = "senclaw-laya.json";

/// One downloadable checkpoint. None of these is an official release —
/// `convaiinnovations/laya` publishes PyTorch weights only — so each entry
/// names an export whose maker documents a parity check against PyTorch, and
/// pins it to a commit: a moving branch would let an upstream push change the
/// weights the daemon runs.
pub struct CatalogEntry {
    pub id: &'static str,
    pub label: &'static str,
    /// What the checkpoint is — true however it got onto this machine.
    pub description: &'static str,
    /// What this particular export is — shown only when the model came from it.
    pub export_note: &'static str,
    pub kind: ModelKind,
    pub repo: &'static str,
    pub revision: &'static str,
    pub files: &'static [&'static str],
    /// A manifest listing sha256 for every file (ti3x-m publishes one). LFS
    /// files carry their sha256 in the Hub's tree listing either way.
    pub manifest: Option<&'static str>,
    pub approx_size_mb: u32,
}

const TI3X_FILES: &[&str] = &[
    "onnx/model.onnx",
    "onnx/model.onnx_data",
    "laya_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
];

pub static CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        id: "multilingual",
        label: "Laya Multilingual (mmBERT-base, 322M)",
        description: "Đọc 100+ ngôn ngữ, gồm tiếng Việt — chọn cái này cho nội dung tiếng Việt.",
        export_note: "Bản ONNX của ti3x-m: mỗi câu hỏi một lần chạy, sha256 cho từng file, \
                      parity với PyTorch trên 34 ca.",
        kind: ModelKind::Multilingual,
        repo: "ti3x-m/laya-multilingual-onnx",
        revision: "ab6836981ce0ea5937e605fb76ddac8960bd1e21",
        files: TI3X_FILES,
        manifest: Some("manifest.json"),
        approx_size_mb: 1325,
    },
    CatalogEntry {
        id: "english",
        label: "Laya English (ModernBERT-large, 421M)",
        description: "Checkpoint gốc, chỉ tiếng Anh. Đọc tiếng Việt kém mà vẫn tự tin \
                      (một email hoá đơn tiếng Việt bị chấm 0,93 là spam).",
        export_note: "Bản ONNX của receptron: batch động, lệch ~1e-5 so với PyTorch.",
        kind: ModelKind::English,
        repo: "receptron/laya-onnx",
        revision: "68f27dfe5a27a54fb2b1fefc432f43f972e90868",
        files: &[
            "laya.onnx",
            "laya.onnx.data",
            "laya_config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        ],
        manifest: None,
        approx_size_mb: 1690,
    },
    CatalogEntry {
        id: "typed-decisions",
        label: "Laya Typed Decisions (ModernBERT-large, 421M)",
        description: "Fine-tune của bản English trên 4 nhóm việc: hỗ trợ khách hàng, hoá đơn, \
                      bảo mật, observability. Chỉ tiếng Anh.",
        export_note: "Bản ONNX của ti3x-m.",
        kind: ModelKind::English,
        repo: "ti3x-m/laya-typed-decisions-onnx",
        revision: "d562cfa57474bf409fee22218611c577ee0cdc0c",
        files: TI3X_FILES,
        manifest: Some("manifest.json"),
        approx_size_mb: 1690,
    },
];

pub fn catalog_get(id: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|e| e.id == id)
}

/// `<local-models>/laya` — the shared model root, never `space-app-data/`.
pub fn root(local_models_dir: &Path) -> PathBuf {
    local_models_dir.join("laya")
}

/// Model ids name a directory, so they are one safe path component.
pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && !id.starts_with('.')
        && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    pub path: String,
    pub size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// `catalog`, `huggingface` or `folder`.
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub kind: ModelKind,
    pub source: Source,
    /// Unix millis.
    pub installed_at: i64,
    pub files: Vec<FileRecord>,
}

pub fn read_metadata(dir: &Path) -> Option<Metadata> {
    let raw = std::fs::read_to_string(dir.join(METADATA_FILE)).ok()?;
    serde_json::from_str(&raw).ok()
}

pub fn write_metadata(dir: &Path, meta: &Metadata) -> std::io::Result<()> {
    let body = serde_json::to_string_pretty(meta).map_err(std::io::Error::other)?;
    let tmp = dir.join(format!("{METADATA_FILE}.tmp"));
    std::fs::write(&tmp, body)?;
    std::fs::rename(tmp, dir.join(METADATA_FILE))
}

/// Installed = the metadata was written *and* the files it needs are there.
pub fn installed_layout(dir: &Path) -> Option<ModelLayout> {
    read_metadata(dir)?;
    ModelLayout::detect(dir).ok()
}

/// `(id, kind)` of every installed model, sorted by id — what on-demand
/// loading chooses from. Reads metadata only, never a size or a job.
pub fn installed_kinds(root: &Path) -> Vec<(String, ModelKind)> {
    let Ok(entries) = std::fs::read_dir(root) else { return Vec::new() };
    let mut out: Vec<(String, ModelKind)> = entries
        .flatten()
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter_map(|e| {
            let id = e.file_name().to_string_lossy().into_owned();
            if !valid_id(&id) {
                return None;
            }
            let meta = read_metadata(&e.path())?;
            ModelLayout::detect(&e.path()).ok()?;
            Some((id, meta.kind))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// A checkpoint's language family, read once when it is installed. The
/// tokenizer's size tells ModernBERT (~50k entries) from mmBERT (256k) when
/// the config does not name the encoder.
pub fn detect_kind(layout: &ModelLayout) -> ModelKind {
    let encoder = LayaConfig::from_file(&layout.config).ok().and_then(|c| c.encoder);
    let vocab = std::fs::read_to_string(&layout.tokenizer)
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .map(|v| match &v["model"]["vocab"] {
            Value::Object(m) => m.len(),
            Value::Array(a) => a.len(),
            _ => 0,
        })
        .unwrap_or(0);
    ModelKind::infer(encoder.as_deref(), vocab)
}

fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            match e.file_type() {
                Ok(t) if t.is_dir() => stack.push(e.path()),
                Ok(_) => total += e.metadata().map(|m| m.len()).unwrap_or(0),
                Err(_) => {}
            }
        }
    }
    total
}

/// One row of the Settings table.
#[derive(Debug, Serialize)]
pub struct ModelView {
    pub id: String,
    pub label: String,
    pub description: String,
    pub kind: Option<ModelKind>,
    pub catalog: bool,
    pub source: Value,
    pub approx_size_mb: Option<u32>,
    pub size_bytes: u64,
    pub installed: bool,
    pub path: String,
    pub job: Option<JobState>,
    pub loading: bool,
    pub loaded: Option<LoadedInfo>,
}

/// The catalog, then every other installed model, then any job for a model
/// that has no directory yet, then anything loaded that none of those named —
/// the four ways a row can exist.
pub fn list(root: &Path) -> Vec<ModelView> {
    let mut rows: Vec<ModelView> = Vec::new();
    let view = |id: &str, entry: Option<&CatalogEntry>| {
        let dir = root.join(id);
        let meta = read_metadata(&dir);
        let installed = meta.is_some() && ModelLayout::detect(&dir).is_ok();
        let loaded = runtime::loaded_info(id);
        let source = match (&meta, entry) {
            (Some(m), _) => serde_json::to_value(&m.source).unwrap_or(Value::Null),
            (None, Some(e)) => json!({"type": "catalog", "repo": e.repo, "revision": e.revision}),
            (None, None) => Value::Null,
        };
        ModelView {
            id: id.to_string(),
            label: meta
                .as_ref()
                .and_then(|m| m.label.clone())
                .or_else(|| entry.map(|e| e.label.to_string()))
                .unwrap_or_else(|| id.to_string()),
            // The export note describes the catalog's file set; a model imported
            // under a catalog id is someone else's export of the same checkpoint.
            description: entry
                .map(|e| match &meta {
                    Some(m) if m.source.kind != "catalog" => e.description.to_string(),
                    _ => format!("{} {}", e.description, e.export_note),
                })
                .unwrap_or_default(),
            kind: loaded
                .as_ref()
                .map(|l| l.kind)
                .or(meta.as_ref().map(|m| m.kind))
                .or(entry.map(|e| e.kind)),
            catalog: entry.is_some(),
            source,
            approx_size_mb: entry.map(|e| e.approx_size_mb),
            size_bytes: if dir.is_dir() { dir_size(&dir) } else { 0 },
            installed,
            path: dir.to_string_lossy().into_owned(),
            job: download::job(id),
            loading: runtime::is_loading(id),
            loaded,
        }
    };

    for e in CATALOG {
        rows.push(view(e.id, Some(e)));
    }
    if let Ok(entries) = std::fs::read_dir(root) {
        let mut ids: Vec<String> = entries
            .flatten()
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|id| valid_id(id) && catalog_get(id).is_none())
            .collect();
        ids.sort();
        for id in ids {
            rows.push(view(&id, None));
        }
    }
    for id in download::job_ids() {
        if !rows.iter().any(|r| r.id == id) {
            rows.push(view(&id, None));
        }
    }
    // Weights in RAM with no directory behind them still cost memory, and
    // only a row lets someone unload them.
    for id in runtime::loaded_ids() {
        if !rows.iter().any(|r| r.id == id) {
            rows.push(view(&id, None));
        }
    }
    rows
}

/// Unload (if loaded), then remove the directory.
pub async fn delete(root: &Path, id: &str) -> std::io::Result<()> {
    runtime::unload(id);
    download::forget(id);
    let dir = root.join(id);
    if dir.exists() {
        tokio::fs::remove_dir_all(&dir).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_one_safe_path_component() {
        for ok in ["english", "laya-browser", "my_model.v2"] {
            assert!(valid_id(ok), "{ok}");
        }
        for bad in ["", ".hidden", "a/b", "..", "x y", "a\\b", &"x".repeat(65)] {
            assert!(!valid_id(bad), "{bad}");
        }
    }

    #[test]
    fn catalog_entries_are_pinned_and_downloadable() {
        for e in CATALOG {
            assert_eq!(e.revision.len(), 40, "{} must pin a full commit sha", e.id);
            assert!(valid_id(e.id));
            let graph = e.files.iter().any(|f| super::super::layout::GRAPH_CANDIDATES.contains(f));
            let config = e.files.iter().any(|f| super::super::layout::CONFIG_CANDIDATES.contains(f));
            assert!(graph && config, "{} lists no graph or no config", e.id);
            assert!(e.files.contains(&"tokenizer/tokenizer.json"));
        }
    }

    #[test]
    fn an_import_under_a_catalog_id_does_not_claim_the_catalog_export() {
        let tmp = tempfile::tempdir().unwrap();
        let entry = catalog_get("english").unwrap();
        let dir = tmp.path().join("english");
        for rel in ["laya.onnx", "rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"] {
            let p = dir.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, "{}").unwrap();
        }
        let row = |rows: Vec<ModelView>| rows.into_iter().find(|r| r.id == "english").unwrap();
        // Not installed: the catalog row describes the export it would download.
        assert!(row(list(tmp.path())).description.contains(entry.export_note));
        write_metadata(
            &dir,
            &Metadata {
                id: "english".into(),
                label: None,
                kind: ModelKind::English,
                source: Source { kind: "folder".into(), repo: None, revision: None, path: Some("/x".into()) },
                installed_at: 0,
                files: vec![],
            },
        )
        .unwrap();
        let r = row(list(tmp.path()));
        assert!(r.installed && r.catalog);
        assert!(r.description.starts_with(entry.description));
        assert!(!r.description.contains(entry.export_note), "{}", r.description);
    }

    #[test]
    fn a_directory_without_metadata_is_not_installed() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("m");
        for rel in ["laya.onnx", "laya_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"] {
            let p = dir.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, "{}").unwrap();
        }
        assert!(installed_layout(&dir).is_none());
        write_metadata(
            &dir,
            &Metadata {
                id: "m".into(),
                label: None,
                kind: ModelKind::English,
                source: Source { kind: "folder".into(), repo: None, revision: None, path: Some("/x".into()) },
                installed_at: 0,
                files: vec![],
            },
        )
        .unwrap();
        assert!(installed_layout(&dir).is_some());
        let rows = list(tmp.path());
        let row = rows.iter().find(|r| r.id == "m").expect("an installed custom model is listed");
        assert!(row.installed && !row.catalog);
        assert_eq!(rows.iter().filter(|r| r.catalog).count(), CATALOG.len());
    }
}
