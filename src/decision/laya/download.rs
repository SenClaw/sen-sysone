//! Background jobs that put a model on disk: a Hugging Face download or a
//! copy of an export that already exists locally.
//!
//! A download is pinned to one commit for its whole run, verifies every file
//! whose sha256 is known — LFS files carry it in the Hub's tree listing, and a
//! `manifest.json` can supply the rest — and writes `<file>.part` until the
//! bytes check out, so a crash never leaves a truncated weight file under its
//! real name. Cancelling keeps the `.part`; the next download resumes it with
//! an HTTP range request instead of starting a 1.7 GB file over.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

use super::layout::ModelLayout;
use super::store::{self, FileRecord, Metadata, Source};

const HF_BASE: &str = "https://huggingface.co";

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Listing,
    Downloading,
    Verifying,
    Copying,
    Done,
    Error,
    Cancelled,
}

impl JobStatus {
    pub fn active(self) -> bool {
        matches!(
            self,
            JobStatus::Queued
                | JobStatus::Listing
                | JobStatus::Downloading
                | JobStatus::Verifying
                | JobStatus::Copying
        )
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct JobState {
    /// `download` or `import`.
    pub kind: &'static str,
    pub status: JobStatus,
    pub total_bytes: u64,
    pub done_bytes: u64,
    pub current_file: Option<String>,
    pub files_total: u32,
    pub files_done: u32,
    pub error: Option<String>,
    /// Unix millis.
    pub started_at: i64,
}

struct JobHandle {
    state: Arc<Mutex<JobState>>,
    cancel: CancellationToken,
}

static JOBS: Lazy<Mutex<HashMap<String, JobHandle>>> = Lazy::new(|| Mutex::new(HashMap::new()));

pub fn job(id: &str) -> Option<JobState> {
    JOBS.lock().unwrap().get(id).map(|h| h.state.lock().unwrap().clone())
}

pub fn job_ids() -> Vec<String> {
    let mut ids: Vec<String> = JOBS.lock().unwrap().keys().cloned().collect();
    ids.sort();
    ids
}

pub fn is_active(id: &str) -> bool {
    job(id).map(|s| s.status.active()).unwrap_or(false)
}

pub fn cancel(id: &str) -> bool {
    match JOBS.lock().unwrap().get(id) {
        Some(h) if h.state.lock().unwrap().status.active() => {
            h.cancel.cancel();
            true
        }
        _ => false,
    }
}

/// Drop a finished job's record (the row falls back to what is on disk).
pub fn forget(id: &str) {
    let mut jobs = JOBS.lock().unwrap();
    if jobs.get(id).is_some_and(|h| !h.state.lock().unwrap().status.active()) {
        jobs.remove(id);
    }
}

fn register(id: &str, kind: &'static str) -> Result<(Arc<Mutex<JobState>>, CancellationToken), String> {
    let mut jobs = JOBS.lock().unwrap();
    if jobs.get(id).is_some_and(|h| h.state.lock().unwrap().status.active()) {
        return Err(format!("`{id}` already has a {kind} in progress"));
    }
    let state = Arc::new(Mutex::new(JobState {
        kind,
        status: JobStatus::Queued,
        total_bytes: 0,
        done_bytes: 0,
        current_file: None,
        files_total: 0,
        files_done: 0,
        error: None,
        started_at: chrono::Utc::now().timestamp_millis(),
    }));
    let cancel = CancellationToken::new();
    jobs.insert(
        id.to_string(),
        JobHandle {
            state: state.clone(),
            cancel: cancel.clone(),
        },
    );
    Ok((state, cancel))
}

/// What a job's future resolved to, folded into its state.
fn finish(state: &Arc<Mutex<JobState>>, result: anyhow::Result<bool>) {
    let mut s = state.lock().unwrap();
    s.current_file = None;
    match result {
        Ok(true) => s.status = JobStatus::Done,
        Ok(false) => s.status = JobStatus::Cancelled,
        Err(e) => {
            s.status = JobStatus::Error;
            s.error = Some(format!("{e:#}"));
        }
    }
}

// ── Hugging Face ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<LfsInfo>,
}

#[derive(Deserialize)]
struct LfsInfo {
    /// The sha256 of the content for an LFS file.
    oid: String,
}

fn http() -> anyhow::Result<reqwest::Client> {
    // No total timeout: a 1.7 GB file legitimately takes minutes. A stalled
    // connection is caught by the read timeout, which resets on every byte.
    Ok(reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .read_timeout(std::time::Duration::from_secs(60))
        .build()?)
}

async fn tree(client: &reqwest::Client, repo: &str, revision: &str) -> anyhow::Result<Vec<TreeEntry>> {
    let url = format!("{HF_BASE}/api/models/{repo}/tree/{revision}?recursive=true");
    let resp = client.get(&url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("listing {repo}@{revision} failed: HTTP {}", resp.status());
    }
    Ok(resp.json::<Vec<TreeEntry>>().await?.into_iter().filter(|e| e.kind == "file").collect())
}

/// Resolve a branch or tag to the commit a download pins to.
pub async fn resolve_revision(repo: &str, revision: &str) -> anyhow::Result<String> {
    if revision.len() == 40 && revision.chars().all(|c| c.is_ascii_hexdigit()) {
        return Ok(revision.to_string());
    }
    let client = http()?;
    let url = format!("{HF_BASE}/api/models/{repo}/revision/{revision}");
    let resp = client.get(&url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("{repo}@{revision} not found on Hugging Face (HTTP {})", resp.status());
    }
    let v: serde_json::Value = resp.json().await?;
    v["sha"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("{repo}@{revision}: the Hub returned no commit sha"))
}

/// Pick the files of a Laya export out of an arbitrary repo, the way
/// [`ModelLayout::detect`] reads a directory — so a custom download fails here,
/// naming what is missing, rather than after gigabytes.
pub async fn plan_custom(repo: &str, revision: &str) -> anyhow::Result<(Vec<String>, Option<String>)> {
    use super::layout::{CONFIG_CANDIDATES, GRAPH_CANDIDATES, TOKENIZER_CANDIDATES, TOKENIZER_CONFIG_CANDIDATES};
    let client = http()?;
    let entries = tree(&client, repo, revision).await?;
    let has = |p: &str| entries.iter().any(|e| e.path == p);
    let first = |c: &[&str]| c.iter().find(|p| has(p)).map(|p| p.to_string());
    let graph = first(GRAPH_CANDIDATES).ok_or_else(|| {
        anyhow::anyhow!("{repo} has no Laya graph ({})", GRAPH_CANDIDATES.join(" / "))
    })?;
    let config = first(CONFIG_CANDIDATES).ok_or_else(|| {
        anyhow::anyhow!("{repo} has no head config ({})", CONFIG_CANDIDATES.join(" / "))
    })?;
    let tokenizer = first(TOKENIZER_CANDIDATES)
        .ok_or_else(|| anyhow::anyhow!("{repo} has no tokenizer.json"))?;
    let tokenizer_config = first(TOKENIZER_CONFIG_CANDIDATES)
        .ok_or_else(|| anyhow::anyhow!("{repo} has no tokenizer_config.json"))?;
    let mut files = vec![graph.clone()];
    // External weights sit next to the graph under whichever name the exporter used.
    for data in [format!("{graph}.data"), format!("{graph}_data")] {
        if has(&data) {
            files.push(data);
        }
    }
    files.extend([config, tokenizer, tokenizer_config]);
    let manifest = has("manifest.json").then(|| "manifest.json".to_string());
    Ok((files, manifest))
}

pub struct DownloadSpec {
    pub id: String,
    pub label: Option<String>,
    /// `catalog` or `huggingface` — recorded in the metadata.
    pub source_kind: &'static str,
    pub repo: String,
    /// A full commit sha.
    pub revision: String,
    pub files: Vec<String>,
    pub manifest: Option<String>,
}

pub fn start_download(root: PathBuf, spec: DownloadSpec) -> Result<(), String> {
    let (state, cancel) = register(&spec.id, "download")?;
    tokio::spawn(async move {
        let result = run_download(&root, &spec, &state, &cancel).await;
        if let Err(e) = &result {
            tracing::warn!("[decision] download of `{}` failed: {e:#}", spec.id);
        }
        finish(&state, result);
    });
    Ok(())
}

/// `Ok(false)` = cancelled.
async fn run_download(
    root: &Path,
    spec: &DownloadSpec,
    state: &Arc<Mutex<JobState>>,
    cancel: &CancellationToken,
) -> anyhow::Result<bool> {
    let dir = root.join(&spec.id);
    tokio::fs::create_dir_all(&dir).await?;
    state.lock().unwrap().status = JobStatus::Listing;
    let client = http()?;

    let entries = tree(&client, &spec.repo, &spec.revision).await?;
    let mut expected: HashMap<String, (u64, Option<String>)> = HashMap::new();
    for e in &entries {
        expected.insert(e.path.clone(), (e.size, e.lfs.as_ref().map(|l| l.oid.clone())));
    }
    if let Some(m) = &spec.manifest {
        let url = format!("{HF_BASE}/{}/resolve/{}/{m}", spec.repo, spec.revision);
        let manifest: serde_json::Value = client.get(&url).send().await?.error_for_status()?.json().await?;
        for f in manifest["files"].as_array().into_iter().flatten() {
            if let (Some(p), Some(sha)) = (f["path"].as_str(), f["sha256"].as_str()) {
                if let Some(slot) = expected.get_mut(p) {
                    slot.1 = Some(sha.to_ascii_lowercase());
                }
            }
        }
    }
    let mut plan = Vec::new();
    for f in &spec.files {
        let (size, sha) = expected
            .get(f)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("{}@{} has no file `{f}`", spec.repo, short(&spec.revision)))?;
        plan.push((f.clone(), size, sha));
    }
    {
        let mut s = state.lock().unwrap();
        s.total_bytes = plan.iter().map(|p| p.1).sum();
        s.files_total = plan.len() as u32;
        s.status = JobStatus::Downloading;
    }

    let mut records = Vec::new();
    for (path, size, sha) in plan {
        if cancel.is_cancelled() {
            return Ok(false);
        }
        state.lock().unwrap().current_file = Some(path.clone());
        let url = format!("{HF_BASE}/{}/resolve/{}/{path}", spec.repo, spec.revision);
        let dst = dir.join(&path);
        let Some(digest) = fetch_file(&client, &url, &dst, size, sha.as_deref(), state, cancel).await? else {
            return Ok(false);
        };
        state.lock().unwrap().files_done += 1;
        records.push(FileRecord {
            path,
            size,
            sha256: Some(digest),
        });
    }

    let layout = ModelLayout::detect(&dir)?;
    let meta = Metadata {
        id: spec.id.clone(),
        label: spec.label.clone(),
        kind: store::detect_kind(&layout),
        source: Source {
            kind: spec.source_kind.to_string(),
            repo: Some(spec.repo.clone()),
            revision: Some(spec.revision.clone()),
            path: None,
        },
        installed_at: chrono::Utc::now().timestamp_millis(),
        files: records,
    };
    store::write_metadata(&dir, &meta)?;
    tracing::info!("[decision] downloaded Laya `{}` from {}@{}", spec.id, spec.repo, short(&spec.revision));
    Ok(true)
}

fn short(rev: &str) -> &str {
    &rev[..rev.len().min(7)]
}

/// Download one file to `dst`, verifying its size and (when known) sha256.
/// Returns the sha256, or `None` when cancelled.
async fn fetch_file(
    client: &reqwest::Client,
    url: &str,
    dst: &Path,
    size: u64,
    sha: Option<&str>,
    state: &Arc<Mutex<JobState>>,
    cancel: &CancellationToken,
) -> anyhow::Result<Option<String>> {
    if let Some(parent) = dst.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    // Already complete from an earlier run: verify it rather than trust the size.
    if tokio::fs::metadata(dst).await.map(|m| m.len() == size).unwrap_or(false) {
        state.lock().unwrap().status = JobStatus::Verifying;
        let digest = hash_file(dst).await?;
        state.lock().unwrap().status = JobStatus::Downloading;
        if sha.is_none_or(|want| want == digest) {
            state.lock().unwrap().done_bytes += size;
            return Ok(Some(digest));
        }
        tokio::fs::remove_file(dst).await?;
    }

    let part = dst.with_file_name(format!(
        "{}.part",
        dst.file_name().unwrap_or_default().to_string_lossy()
    ));
    let mut have = tokio::fs::metadata(&part).await.map(|m| m.len()).unwrap_or(0);
    if have > size {
        tokio::fs::remove_file(&part).await?;
        have = 0;
    }
    let mut hasher = Sha256::new();
    if have > 0 {
        hasher = hash_prefix(&part).await?;
    }

    let mut req = client.get(url);
    if have > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }
    let resp = req.send().await?;
    let status = resp.status();
    let mut file = if have > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT {
        tokio::fs::OpenOptions::new().append(true).open(&part).await?
    } else {
        if !status.is_success() {
            anyhow::bail!("GET {url}: HTTP {status}");
        }
        // The server ignored the range: start this file over.
        have = 0;
        hasher = Sha256::new();
        tokio::fs::File::create(&part).await?
    };
    state.lock().unwrap().done_bytes += have;

    let mut written = have;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        if cancel.is_cancelled() {
            file.flush().await?;
            return Ok(None);
        }
        let bytes = chunk?;
        file.write_all(&bytes).await?;
        hasher.update(&bytes);
        written += bytes.len() as u64;
        state.lock().unwrap().done_bytes += bytes.len() as u64;
    }
    file.flush().await?;
    drop(file);

    if size > 0 && written != size {
        anyhow::bail!("{}: got {written} bytes, expected {size}", dst.display());
    }
    let digest = hex::encode(hasher.finalize());
    if let Some(want) = sha {
        if want != digest {
            tokio::fs::remove_file(&part).await?;
            anyhow::bail!(
                "{}: sha256 mismatch (expected {want}, got {digest}) — the file was discarded",
                dst.display()
            );
        }
    }
    tokio::fs::rename(&part, dst).await?;
    Ok(Some(digest))
}

async fn hash_file(path: &Path) -> anyhow::Result<String> {
    Ok(hex::encode(hash_prefix(path).await?.finalize()))
}

/// A hasher primed with a file's current contents — hashing a gigabyte is
/// CPU work, so it runs on the blocking pool.
async fn hash_prefix(path: &Path) -> anyhow::Result<Sha256> {
    let path = path.to_path_buf();
    Ok(tokio::task::spawn_blocking(move || -> std::io::Result<Sha256> {
        use std::io::Read;
        let mut f = std::fs::File::open(&path)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                return Ok(hasher);
            }
            hasher.update(&buf[..n]);
        }
    })
    .await??)
}

// ── Import from a local folder ───────────────────────────────────────────────

/// Files an import leaves behind: OS litter and other tools' partial downloads.
fn skip_on_import(rel: &Path) -> bool {
    rel.components().any(|c| {
        let s = c.as_os_str().to_string_lossy();
        s == ".DS_Store" || s == "__pycache__" || s.ends_with(".part") || s == store::METADATA_FILE
    })
}

pub fn start_import(root: PathBuf, id: String, src: PathBuf) -> Result<(), String> {
    ModelLayout::detect(&src).map_err(|e| format!("{e:#}"))?;
    let dst = root.join(&id);
    if dst.starts_with(&src) || src.starts_with(&dst) {
        return Err("the source folder overlaps the model's own folder".into());
    }
    let (state, cancel) = register(&id, "import")?;
    tokio::spawn(async move {
        let result = run_import(&dst, &id, &src, &state, &cancel).await;
        if let Err(e) = &result {
            tracing::warn!("[decision] import of `{id}` failed: {e:#}");
        }
        finish(&state, result);
    });
    Ok(())
}

async fn run_import(
    dst: &Path,
    id: &str,
    src: &Path,
    state: &Arc<Mutex<JobState>>,
    cancel: &CancellationToken,
) -> anyhow::Result<bool> {
    let src_owned = src.to_path_buf();
    let files: Vec<(PathBuf, u64)> = tokio::task::spawn_blocking(move || {
        let mut out = Vec::new();
        let mut stack = vec![src_owned.clone()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d)?.flatten() {
                let path = e.path();
                let rel = path.strip_prefix(&src_owned).unwrap_or(&path).to_path_buf();
                if skip_on_import(&rel) {
                    continue;
                }
                let t = e.file_type()?;
                if t.is_dir() {
                    stack.push(path);
                } else if t.is_file() {
                    out.push((rel, e.metadata()?.len()));
                }
            }
        }
        std::io::Result::Ok(out)
    })
    .await??;
    {
        let mut s = state.lock().unwrap();
        s.status = JobStatus::Copying;
        s.files_total = files.len() as u32;
        s.total_bytes = files.iter().map(|f| f.1).sum();
    }

    let mut records = Vec::new();
    for (rel, size) in files {
        if cancel.is_cancelled() {
            return Ok(false);
        }
        state.lock().unwrap().current_file = Some(rel.to_string_lossy().into_owned());
        let (from, to) = (src.join(&rel), dst.join(&rel));
        // std::fs::copy clones on APFS, so a same-volume import of a 1.7 GB
        // export costs no extra disk and no time.
        tokio::task::spawn_blocking(move || -> std::io::Result<()> {
            if let Some(parent) = to.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(&from, &to).map(|_| ())
        })
        .await??;
        {
            let mut s = state.lock().unwrap();
            s.done_bytes += size;
            s.files_done += 1;
        }
        records.push(FileRecord {
            path: rel.to_string_lossy().into_owned(),
            size,
            sha256: None,
        });
    }

    let installed = ModelLayout::detect(dst)?;
    let meta = Metadata {
        id: id.to_string(),
        label: None,
        kind: store::detect_kind(&installed),
        source: Source {
            kind: "folder".into(),
            repo: None,
            revision: None,
            path: Some(src.to_string_lossy().into_owned()),
        },
        installed_at: chrono::Utc::now().timestamp_millis(),
        files: records,
    };
    store::write_metadata(dst, &meta)?;
    tracing::info!("[decision] imported Laya `{id}` from {}", src.display());
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imports_skip_litter_and_foreign_partials() {
        for skip in [".DS_Store", "__pycache__/x.pyc", "laya.onnx.data.part", store::METADATA_FILE] {
            assert!(skip_on_import(Path::new(skip)), "{skip}");
        }
        for keep in ["laya.onnx", "tokenizer/tokenizer.json", "source.json"] {
            assert!(!skip_on_import(Path::new(keep)), "{keep}");
        }
    }

    #[tokio::test]
    async fn an_import_copies_an_export_and_marks_it_installed() {
        let src = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        for (rel, body) in [
            ("laya.onnx", "graph"),
            ("laya.onnx.data", "weights"),
            ("rl_agent_config.json", r#"{"encoder": "jhu-clsp/mmBERT-base", "max_len": 1024, "head_max_len": 256}"#),
            ("tokenizer/tokenizer.json", r#"{"model": {"vocab": {}}}"#),
            ("tokenizer/tokenizer_config.json", "{}"),
            (".DS_Store", "x"),
        ] {
            let p = src.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        }
        start_import(root.path().to_path_buf(), "ml".into(), src.path().to_path_buf()).unwrap();
        for _ in 0..200 {
            if !is_active("ml") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let s = job("ml").unwrap();
        assert_eq!(s.status, JobStatus::Done, "{:?}", s.error);
        let dir = root.path().join("ml");
        assert!(store::installed_layout(&dir).is_some());
        assert!(!dir.join(".DS_Store").exists());
        let meta = store::read_metadata(&dir).unwrap();
        assert_eq!(meta.kind, super::super::layout::ModelKind::Multilingual);
        assert_eq!(meta.source.kind, "folder");
    }

    #[test]
    fn an_import_refuses_a_folder_that_is_not_an_export() {
        let src = tempfile::tempdir().unwrap();
        let err = start_import(PathBuf::from("/nonexistent-root"), "x".into(), src.path().to_path_buf()).unwrap_err();
        assert!(err.contains("not a Laya export"), "{err}");
    }
}
