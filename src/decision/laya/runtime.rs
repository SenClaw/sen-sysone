//! The loaded-model registry and the one entry point that answers a request.
//!
//! **Which backend answers**: the request's `backend`, else the one chosen in
//! Settings. Online requests go to [`crate::decision::online`]; local ones to a
//! Laya checkpoint here.
//!
//! **Which local model**: the request's `model`, else the configured default
//! when it can read the text, else one picked by language (see
//! [`plan_local`]). With `auto_load` on, a model that is installed but not in
//! RAM is **loaded on demand** (hot-load); with it off the request is refused
//! and says what to load.
//!
//! **One load per model.** A load runs in a detached task that publishes its
//! result to everyone waiting on it and cleans up after itself: a client that
//! disconnects mid-load (which cancels its handler) must neither strand the id
//! in "loading" nor throw away weights that finished loading. Loads of
//! different models take turns ([`ONE_LOAD_AT_A_TIME`]): ONNX Runtime cannot
//! safely create two sessions at once on macOS.
//!
//! **When weights leave**: a sweeper wakes every [`SWEEP_EVERY`] and unloads
//! any model no request has *started* on for the configured idle time. A
//! checkpoint is 1.2–1.7 GB of RSS, so an unused one should not sit in memory
//! all day — and with hot-load on, the next request simply brings it back. A
//! request already running keeps its own reference to the engine.
//!
//! Nothing loads at boot. Without the `decision-laya` build feature this
//! module still compiles: nothing can be loaded, and every local call says so
//! instead of failing somewhere less obvious.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;
#[cfg(feature = "decision-laya")]
use std::sync::{Arc, RwLock};
#[cfg(feature = "decision-laya")]
use std::time::Instant;

use futures::future::{BoxFuture, Shared};
#[cfg(feature = "decision-laya")]
use futures::future::FutureExt;
use once_cell::sync::Lazy;
use serde::Serialize;

use super::layout::{ModelKind, ModelLayout};
use crate::decision::json::Json;
use crate::decision::settings::{Backend, DecisionSettings, DEFAULT_IDLE_UNLOAD_MINUTES};
use crate::decision::types::{AskRequest, AskResponse, Criteria, Question};

/// Whether this build can run Laya at all.
pub const COMPILED: bool = cfg!(feature = "decision-laya");

/// How often idle models are looked for.
pub const SWEEP_EVERY: Duration = Duration::from_secs(30);

/// How long a request waits on a load before giving up on it (the load
/// itself carries on). A load reads 1.2–1.7 GB and takes seconds.
pub const LOAD_WAIT: Duration = Duration::from_secs(300);

#[derive(Debug)]
pub enum RuntimeError {
    /// Built without `decision-laya`.
    NotCompiled,
    NotInstalled(String),
    NotLoaded(String),
    NoneLoaded,
    NoneInstalled,
    /// The chosen backend lacks something only the user can supply.
    NotConfigured(String),
    /// Nothing available can answer this request well; the text says what to do.
    Refused(String),
    /// The request itself is malformed — the caller has to change it.
    BadRequest(String),
    /// The online backend failed or refused.
    Upstream(String),
    Internal(String),
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::NotCompiled => write!(
                f,
                "this SenClaw build has no Laya engine (built without the `decision-laya` feature)"
            ),
            RuntimeError::NotInstalled(id) => write!(f, "model `{id}` is not installed"),
            RuntimeError::NotLoaded(id) => write!(
                f,
                "model `{id}` is not loaded — load it in Settings → Decision (Laya), or turn on loading on demand"
            ),
            RuntimeError::NoneLoaded => write!(
                f,
                "no Laya model is loaded — load one in Settings → Decision (Laya), or turn on loading on demand"
            ),
            RuntimeError::NoneInstalled => write!(
                f,
                "no Laya model is installed — download or import one in Settings → Decision (Laya)"
            ),
            RuntimeError::NotConfigured(m)
            | RuntimeError::Refused(m)
            | RuntimeError::BadRequest(m)
            | RuntimeError::Upstream(m)
            | RuntimeError::Internal(m) => f.write_str(m),
        }
    }
}

/// What the UI shows for a loaded model.
#[derive(Debug, Clone, Serialize)]
pub struct LoadedInfo {
    pub kind: ModelKind,
    pub load_ms: u64,
    /// Unix millis.
    pub loaded_at: i64,
    /// `dynamic` (one run per request) or `fixed-1` (one run per question).
    pub batch: &'static str,
    pub act_output: &'static str,
    pub max_len: usize,
    pub head_max_len: usize,
    /// ONNX Runtime intra-op threads it was loaded with.
    pub threads: usize,
    /// Requests answered since the load.
    pub asks: u64,
    /// Seconds since the last request (or the load).
    pub idle_secs: u64,
    /// `true` when a request loaded it rather than a person.
    pub on_demand: bool,
}

#[cfg(feature = "decision-laya")]
struct Loaded {
    engine: Arc<super::engine::LayaEngine>,
    info: LoadedInfo,
    last_used: Instant,
}

#[cfg(feature = "decision-laya")]
static LOADED: Lazy<RwLock<HashMap<String, Loaded>>> = Lazy::new(|| RwLock::new(HashMap::new()));

/// What a load hands everyone waiting on it.
type LoadOutcome = Result<LoadedInfo, String>;
type PendingLoad = Shared<BoxFuture<'static, LoadOutcome>>;

/// Loads in flight. An id leaves this map only *after* its model is in
/// `LOADED`, so an id in neither is truly not loaded.
static LOADING: Lazy<Mutex<HashMap<String, PendingLoad>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Models whose files are being deleted; a load refuses them. Lock order is
/// always `LOADING` then `DELETING`.
static DELETING: Lazy<Mutex<HashSet<String>>> = Lazy::new(|| Mutex::new(HashSet::new()));

/// Held by a load while it builds its engine, so two models never load at
/// once (the parity test holds it too). ONNX Runtime finds a graph's external weights from a directory it
/// gets from libc `dirname()`, and on macOS that hands every caller the same
/// static buffer: two sessions created together can each read the other's
/// directory, and one fails to find its weights. (It surfaces as
/// "Encountered unknown exception in Initialize()" — the `filesystem_error`
/// comes from the system libc++, which the statically linked ONNX Runtime
/// cannot catch as a `std::exception`.) Inference never goes there, so only
/// loads take turns — and a load is rare and takes seconds anyway.
#[cfg(feature = "decision-laya")]
pub(super) static ONE_LOAD_AT_A_TIME: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// The idle limit the sweeper applies, in minutes (0 = never). Kept current
/// by [`apply_settings`], which every settings read and save goes through.
static IDLE_UNLOAD_MINUTES: AtomicU64 = AtomicU64::new(DEFAULT_IDLE_UNLOAD_MINUTES);

/// Hand the runtime the settings that are not read per request.
pub fn apply_settings(s: &DecisionSettings) {
    IDLE_UNLOAD_MINUTES.store(s.local.idle_unload_minutes, Ordering::Relaxed);
}

pub fn is_loading(id: &str) -> bool {
    LOADING.lock().unwrap().contains_key(id)
}

/// Every model in RAM — so a list can show one whose directory is gone.
pub fn loaded_ids() -> Vec<String> {
    #[cfg(feature = "decision-laya")]
    {
        let mut ids: Vec<String> = LOADED.read().unwrap().keys().cloned().collect();
        ids.sort();
        ids
    }
    #[cfg(not(feature = "decision-laya"))]
    {
        Vec::new()
    }
}

/// Held while a model's files are removed: a load starting meanwhile would
/// put weights in RAM for a directory that is going away.
pub struct DeleteGuard(String);

impl Drop for DeleteGuard {
    fn drop(&mut self) {
        DELETING.lock().unwrap().remove(&self.0);
    }
}

/// `None` while the model is loading — wait for that to finish first.
pub fn begin_delete(id: &str) -> Option<DeleteGuard> {
    let loading = LOADING.lock().unwrap();
    if loading.contains_key(id) {
        return None;
    }
    DELETING.lock().unwrap().insert(id.to_string());
    Some(DeleteGuard(id.to_string()))
}

pub fn loaded_info(id: &str) -> Option<LoadedInfo> {
    #[cfg(feature = "decision-laya")]
    {
        LOADED.read().unwrap().get(id).map(|l| LoadedInfo {
            idle_secs: l.last_used.elapsed().as_secs(),
            ..l.info.clone()
        })
    }
    #[cfg(not(feature = "decision-laya"))]
    {
        let _ = id;
        None
    }
}

/// The intra-op thread count for a setting (`None` = cores, capped at 8).
pub fn resolve_threads(setting: Option<usize>) -> usize {
    setting.filter(|t| *t > 0).unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, 8)
    })
}

/// Load a model's weights into RAM. Idempotent: an already-loaded model
/// answers with its existing info (and keeps its thread count), and a model
/// already loading is waited on rather than loaded twice.
pub async fn load(
    id: &str,
    layout: ModelLayout,
    threads: Option<usize>,
    on_demand: bool,
) -> Result<LoadedInfo, RuntimeError> {
    #[cfg(not(feature = "decision-laya"))]
    {
        let _ = (id, layout, threads, on_demand);
        Err(RuntimeError::NotCompiled)
    }
    #[cfg(feature = "decision-laya")]
    {
        let pending = {
            let mut loading = LOADING.lock().unwrap();
            // Checked under the LOADING lock: a finished load fills LOADED
            // before it leaves LOADING, so this cannot miss one in between.
            if let Some(info) = loaded_info(id) {
                return Ok(info);
            }
            if DELETING.lock().unwrap().contains(id) {
                return Err(RuntimeError::NotInstalled(id.to_string()));
            }
            match loading.get(id) {
                Some(p) => p.clone(),
                None => {
                    let p = spawn_load(id.to_string(), layout, resolve_threads(threads), on_demand);
                    loading.insert(id.to_string(), p.clone());
                    p
                }
            }
        };
        match tokio::time::timeout(LOAD_WAIT, pending).await {
            Ok(outcome) => outcome.map_err(RuntimeError::Internal),
            Err(_) => Err(RuntimeError::Internal(format!(
                "model `{id}` is still loading after {} s",
                LOAD_WAIT.as_secs()
            ))),
        }
    }
}

/// Run one load to completion whoever is still waiting: the task is detached,
/// so a cancelled caller changes nothing.
#[cfg(feature = "decision-laya")]
fn spawn_load(key: String, layout: ModelLayout, threads: usize, on_demand: bool) -> PendingLoad {
    let task = tokio::spawn(async move {
        let result = {
            let _turn = ONE_LOAD_AT_A_TIME.lock().await;
            tokio::task::spawn_blocking(move || super::engine::LayaEngine::load(&layout, threads)).await
        };
        let outcome: LoadOutcome = match result {
            Err(e) => Err(format!("load task failed: {e}")),
            Ok(Err(e)) => Err(format!("{e:#}")),
            Ok(Ok(engine)) => {
                let info = LoadedInfo {
                    kind: engine.kind,
                    load_ms: engine.load_ms,
                    loaded_at: chrono::Utc::now().timestamp_millis(),
                    batch: if engine.fixed_batch { "fixed-1" } else { "dynamic" },
                    act_output: engine.act_output.name(),
                    max_len: engine.max_len,
                    head_max_len: engine.head_max_len,
                    threads,
                    asks: 0,
                    idle_secs: 0,
                    on_demand,
                };
                tracing::info!(
                    "[decision] loaded Laya `{key}` ({:?}, {} batch, {}, {threads} threads{}) in {} ms",
                    info.kind,
                    info.batch,
                    info.act_output,
                    if on_demand { ", on demand" } else { "" },
                    info.load_ms
                );
                LOADED.write().unwrap().insert(
                    key.clone(),
                    Loaded {
                        engine: Arc::new(engine),
                        info: info.clone(),
                        last_used: Instant::now(),
                    },
                );
                ensure_sweeper();
                Ok(info)
            }
        };
        if let Err(e) = &outcome {
            tracing::warn!("[decision] loading Laya `{key}` failed: {e}");
        }
        // Only now: whoever sees the id leave LOADING must find it in LOADED.
        LOADING.lock().unwrap().remove(&key);
        outcome
    });
    async move { task.await.unwrap_or_else(|e| Err(format!("load task failed: {e}"))) }
        .boxed()
        .shared()
}

/// Drop a model's weights. A request already running on it finishes first —
/// it holds its own reference — and the memory goes when that one returns.
pub fn unload(id: &str) -> bool {
    #[cfg(feature = "decision-laya")]
    {
        let removed = LOADED.write().unwrap().remove(id).is_some();
        if removed {
            tracing::info!("[decision] unloaded Laya `{id}`");
        }
        removed
    }
    #[cfg(not(feature = "decision-laya"))]
    {
        let _ = id;
        false
    }
}

/// Whether a model idle for `idle` has passed a limit of `minutes` (0 = never).
pub fn idle_expired(idle: Duration, minutes: u64) -> bool {
    minutes > 0 && idle >= Duration::from_secs(minutes * 60)
}

/// Start the idle sweeper, once, the first time anything is loaded.
#[cfg(feature = "decision-laya")]
fn ensure_sweeper() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        tokio::spawn(async {
            let mut tick = tokio::time::interval(SWEEP_EVERY);
            loop {
                tick.tick().await;
                let minutes = IDLE_UNLOAD_MINUTES.load(Ordering::Relaxed);
                let expired: Vec<String> = LOADED
                    .read()
                    .unwrap()
                    .iter()
                    .filter(|(_, l)| idle_expired(l.last_used.elapsed(), minutes))
                    .map(|(id, _)| id.clone())
                    .collect();
                for id in expired {
                    // Re-check under the write lock: a request may have used it
                    // between the scan and now.
                    let mut map = LOADED.write().unwrap();
                    if map.get(&id).is_some_and(|l| idle_expired(l.last_used.elapsed(), minutes)) {
                        map.remove(&id);
                        tracing::info!("[decision] unloaded Laya `{id}` after {minutes} min without a request");
                    }
                }
            }
        });
    });
}

/// What the local path does with a request.
#[derive(Debug, PartialEq, Eq)]
enum LocalPlan {
    /// Answer with this loaded model.
    Use(String, String),
    /// Load this installed model first (hot-load), then answer.
    Load(String, String),
}

/// Choose the local model. `loaded` / `installed` are `(id, kind)` sorted by
/// id; a loaded model is normally also installed, but need not be (its
/// directory can be deleted under it). `ascii_letters` is whether every letter
/// the model will read is ASCII.
///
/// English checkpoints read other languages confidently wrong (a Vietnamese
/// billing email scored 0.93 spam), so text with other letters goes to a
/// multilingual checkpoint whenever one is installed — even past the default,
/// and never by quietly handing it to English when the right one only needs
/// loading. The asymmetry is one-way: a multilingual checkpoint reads English
/// fine, so ASCII text never costs a second model.
#[cfg_attr(not(feature = "decision-laya"), allow(dead_code))]
fn plan_local(
    requested: Option<&str>,
    default_model: Option<&str>,
    loaded: &[(String, ModelKind)],
    installed: &[(String, ModelKind)],
    ascii_letters: bool,
    auto_load: bool,
) -> Result<LocalPlan, RuntimeError> {
    use ModelKind::{English, Multilingual};
    let is_loaded = |id: &str| loaded.iter().any(|(l, _)| l == id);
    let is_installed = |id: &str| installed.iter().any(|(i, _)| i == id);
    let kind_of = |id: &str| loaded.iter().chain(installed).find(|(i, _)| i == id).map(|(_, k)| *k);
    let use_or_load = |id: &str, why: &str| {
        if is_loaded(id) {
            Ok(LocalPlan::Use(id.to_string(), why.to_string()))
        } else if !is_installed(id) {
            Err(RuntimeError::NotInstalled(id.to_string()))
        } else if auto_load {
            Ok(LocalPlan::Load(id.to_string(), format!("{why}; loaded on demand")))
        } else {
            Err(RuntimeError::NotLoaded(id.to_string()))
        }
    };

    // A model the request names is always the one that answers.
    if let Some(id) = requested.filter(|m| !m.is_empty() && *m != "auto") {
        return use_or_load(id, "requested explicitly");
    }

    // The default answers whatever it can read.
    let mut note = String::new();
    if let Some(id) = default_model {
        match kind_of(id) {
            None => return Err(RuntimeError::NotInstalled(id.to_string())),
            Some(kind) if ascii_letters || kind == Multilingual => {
                return use_or_load(id, "the default model in Settings")
            }
            Some(_) => note = format!("the default `{id}` reads only English; "),
        }
    }

    if ascii_letters {
        if let Some((id, _)) = loaded.iter().find(|(_, k)| *k == English) {
            return Ok(LocalPlan::Use(id.clone(), "ASCII-only text → an English checkpoint".into()));
        }
        if let Some((id, _)) = loaded.first() {
            return Ok(LocalPlan::Use(
                id.clone(),
                "ASCII-only text; the loaded multilingual checkpoint reads English".into(),
            ));
        }
        if auto_load {
            if let Some((id, _)) = installed.iter().find(|(_, k)| *k == English) {
                return Ok(LocalPlan::Load(id.clone(), "ASCII-only text → an English checkpoint; loaded on demand".into()));
            }
            if let Some((id, _)) = installed.first() {
                return Ok(LocalPlan::Load(id.clone(), "ASCII-only text; loaded on demand".into()));
            }
        }
        return Err(if installed.is_empty() { RuntimeError::NoneInstalled } else { RuntimeError::NoneLoaded });
    }

    if let Some((id, _)) = loaded.iter().find(|(_, k)| *k == Multilingual) {
        return Ok(LocalPlan::Use(id.clone(), format!("{note}non-ASCII text → a multilingual checkpoint")));
    }
    if let Some((id, _)) = installed.iter().find(|(_, k)| *k == Multilingual) {
        return if auto_load {
            Ok(LocalPlan::Load(
                id.clone(),
                format!("{note}non-ASCII text → a multilingual checkpoint; loaded on demand"),
            ))
        } else {
            Err(RuntimeError::Refused(format!(
                "this text has non-English letters and needs the multilingual model `{id}`, which is not loaded — \
                 load it in Settings → Decision (Laya), or turn on loading on demand (an English checkpoint \
                 reads such text confidently wrong)"
            )))
        };
    }

    // No multilingual checkpoint on this machine: answer with what there is,
    // preferring the default, and say so.
    let warn = format!("{note}no multilingual checkpoint installed — an English one reads this text poorly");
    let default_loaded = default_model.filter(|d| is_loaded(d)).map(str::to_string);
    if let Some(id) = default_loaded.or_else(|| loaded.first().map(|(i, _)| i.clone())) {
        return Ok(LocalPlan::Use(id, warn));
    }
    if auto_load {
        let default_installed = default_model.filter(|d| is_installed(d)).map(str::to_string);
        if let Some(id) = default_installed.or_else(|| installed.first().map(|(i, _)| i.clone())) {
            return Ok(LocalPlan::Load(id, format!("{warn}; loaded on demand")));
        }
    }
    Err(if installed.is_empty() { RuntimeError::NoneInstalled } else { RuntimeError::NoneLoaded })
}

/// Whether every letter the model will read is ASCII. Only letters count:
/// a curly apostrophe or an emoji says nothing about the language.
#[cfg_attr(not(feature = "decision-laya"), allow(dead_code))]
fn ascii_letters_only(state: &Json, questions: &[Question]) -> bool {
    let plain = |t: &str| !t.chars().any(|c| c.is_alphabetic() && !c.is_ascii());
    let json = |v: &Json| plain(&super::sequence::render_criterion(v));
    json(state)
        && questions.iter().all(|q| {
            json(&q.instructions)
                && match &q.criteria {
                    Criteria::Choice(c) => c.iter().all(|(l, d)| plain(l) && d.as_ref().is_none_or(json)),
                    Criteria::Score(levels) => levels.iter().all(json),
                    Criteria::Noul {
                        false_desc,
                        true_desc,
                        false_label,
                        true_label,
                    } => {
                        plain(false_label)
                            && plain(true_label)
                            && false_desc.as_ref().is_none_or(json)
                            && true_desc.as_ref().is_none_or(json)
                    }
                }
        })
}

/// Answer one `/v1/systemone`-shaped request. `root` is the Laya model root.
pub async fn ask(req: AskRequest, settings: &DecisionSettings, root: &Path) -> Result<AskResponse, RuntimeError> {
    apply_settings(settings);
    // Validated here for both backends: a malformed question should fail the
    // same way whichever one would have answered it.
    let questions = crate::decision::types::parse_questions(&req.questions).map_err(RuntimeError::BadRequest)?;
    if !matches!(req.state, Json::String(_) | Json::Object(_) | Json::Array(_)) {
        return Err(RuntimeError::BadRequest(
            "state must be a string, an object or an array".into(),
        ));
    }
    match req.backend.unwrap_or(settings.backend) {
        Backend::Online => {
            crate::decision::online::ask(&settings.online, req.model.as_deref(), &req.state, &req.questions).await
        }
        Backend::Local => ask_local(req, questions, settings, root).await,
    }
}

async fn ask_local(
    req: AskRequest,
    questions: Vec<Question>,
    settings: &DecisionSettings,
    root: &Path,
) -> Result<AskResponse, RuntimeError> {
    #[cfg(not(feature = "decision-laya"))]
    {
        let _ = (req, questions, settings, root);
        Err(RuntimeError::NotCompiled)
    }
    #[cfg(feature = "decision-laya")]
    {
        use crate::decision::types::{Answers, Routing, Usage};

        let loaded: Vec<(String, ModelKind)> = {
            let map = LOADED.read().unwrap();
            let mut v: Vec<(String, ModelKind)> = map.iter().map(|(k, l)| (k.clone(), l.info.kind)).collect();
            v.sort_by(|a, b| a.0.cmp(&b.0));
            v
        };
        let installed = super::store::installed_kinds(root);
        let plan = plan_local(
            req.model.as_deref(),
            settings.local.default_model.as_deref(),
            &loaded,
            &installed,
            ascii_letters_only(&req.state, &questions),
            settings.local.auto_load,
        )?;
        let (model, reason) = match plan {
            LocalPlan::Use(id, why) => (id, why),
            LocalPlan::Load(id, why) => {
                load_installed(&id, root, settings).await?;
                (id, why)
            }
        };
        let engine = match take_engine(&model) {
            Some(engine) => engine,
            // Swept or unloaded between the plan and now: with loading on
            // demand, bring it back rather than refuse an answerable request.
            None if settings.local.auto_load => {
                load_installed(&model, root, settings).await?;
                take_engine(&model).ok_or_else(|| RuntimeError::NotLoaded(model.clone()))?
            }
            None => return Err(RuntimeError::NotLoaded(model.clone())),
        };

        let started = Instant::now();
        let state = req.state;
        let out = tokio::task::spawn_blocking(move || engine.infer(&state, &questions))
            .await
            .map_err(|e| RuntimeError::Internal(format!("inference task failed: {e}")))?
            .map_err(|e| {
                // A question whose options overflow the head budget is the
                // caller's to fix; anything else is ours.
                let msg = format!("{e:#}");
                if msg.contains("do not fit in head_max_len") {
                    RuntimeError::BadRequest(msg)
                } else {
                    RuntimeError::Internal(msg)
                }
            })?;
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        if let Some(l) = LOADED.write().unwrap().get_mut(&model) {
            l.info.asks += 1;
            l.last_used = Instant::now();
        }
        Ok(AskResponse {
            model: model.clone(),
            engine: "laya-onnx".into(),
            answers: Answers::Local(out.answers),
            usage: Usage {
                input_tokens: out.input_tokens,
                output_tokens: 0,
            },
            latency_ms: (latency_ms * 10.0).round() / 10.0,
            runs: out.runs,
            routing: Routing { model, reason },
        })
    }
}

#[cfg(feature = "decision-laya")]
async fn load_installed(id: &str, root: &Path, settings: &DecisionSettings) -> Result<LoadedInfo, RuntimeError> {
    let layout =
        super::store::installed_layout(&root.join(id)).ok_or_else(|| RuntimeError::NotInstalled(id.to_string()))?;
    load(id, layout, settings.local.threads, true).await
}

/// The engine for `id`, marked used — before inferring, so the sweeper cannot
/// count a request's own run time as idleness.
#[cfg(feature = "decision-laya")]
fn take_engine(id: &str) -> Option<Arc<super::engine::LayaEngine>> {
    let mut map = LOADED.write().unwrap();
    let l = map.get_mut(id)?;
    l.last_used = Instant::now();
    Some(l.engine.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ModelKind::{English as En, Multilingual as Ml};

    fn v(items: &[(&str, ModelKind)]) -> Vec<(String, ModelKind)> {
        items.iter().map(|(i, k)| (i.to_string(), *k)).collect()
    }

    fn js(text: &str) -> Json {
        serde_json::from_str(text).unwrap()
    }

    fn both() -> Vec<(String, ModelKind)> {
        v(&[("english", En), ("multilingual", Ml)])
    }

    #[test]
    fn a_named_model_is_used_or_loaded_on_demand() {
        let loaded = v(&[("english", En)]);
        assert_eq!(
            plan_local(Some("english"), None, &loaded, &both(), false, true).unwrap(),
            LocalPlan::Use("english".into(), "requested explicitly".into())
        );
        assert!(matches!(
            plan_local(Some("multilingual"), None, &loaded, &both(), false, true).unwrap(),
            LocalPlan::Load(ref id, _) if id == "multilingual"
        ));
        assert!(matches!(
            plan_local(Some("multilingual"), None, &loaded, &both(), false, false),
            Err(RuntimeError::NotLoaded(_))
        ));
        assert!(matches!(
            plan_local(Some("typed-decisions"), None, &loaded, &both(), true, true),
            Err(RuntimeError::NotInstalled(_))
        ));
    }

    #[test]
    fn the_default_model_answers_requests_that_name_none() {
        let plan = plan_local(None, Some("multilingual"), &[], &both(), true, true).unwrap();
        assert!(matches!(plan, LocalPlan::Load(ref id, ref why) if id == "multilingual" && why.contains("default")));
        // An explicit model still wins over the default.
        let plan = plan_local(Some("english"), Some("multilingual"), &v(&[("english", En)]), &both(), true, true).unwrap();
        assert_eq!(plan, LocalPlan::Use("english".into(), "requested explicitly".into()));
    }

    #[test]
    fn vietnamese_loads_the_multilingual_checkpoint_rather_than_use_english() {
        let loaded = v(&[("english", En)]);
        assert!(matches!(
            plan_local(None, None, &loaded, &both(), false, true).unwrap(),
            LocalPlan::Load(ref id, _) if id == "multilingual"
        ));
        // Without hot-load it refuses rather than let English read it,
        // and names the model to load.
        match plan_local(None, None, &loaded, &both(), false, false) {
            Err(RuntimeError::Refused(m)) => assert!(m.contains("`multilingual`"), "{m}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
        // ASCII text is happy with the loaded English model.
        assert!(matches!(
            plan_local(None, None, &loaded, &both(), true, true).unwrap(),
            LocalPlan::Use(ref id, _) if id == "english"
        ));
        // Both loaded: the language decides.
        assert_eq!(
            plan_local(None, None, &both(), &both(), false, true).unwrap(),
            LocalPlan::Use("multilingual".into(), "non-ASCII text → a multilingual checkpoint".into())
        );
    }

    #[test]
    fn english_text_uses_a_loaded_multilingual_model_instead_of_loading_a_second() {
        let loaded = v(&[("multilingual", Ml)]);
        assert!(matches!(
            plan_local(None, None, &loaded, &both(), true, true).unwrap(),
            LocalPlan::Use(ref id, ref why) if id == "multilingual" && why.contains("reads English")
        ));
    }

    #[test]
    fn nothing_loaded_loads_by_language_or_says_what_is_missing() {
        assert!(matches!(
            plan_local(None, None, &[], &both(), true, true).unwrap(),
            LocalPlan::Load(ref id, _) if id == "english"
        ));
        // Only an English model installed: Vietnamese still gets an answer,
        // with a reason that says why it may be poor.
        assert!(matches!(
            plan_local(None, None, &[], &v(&[("english", En)]), false, true).unwrap(),
            LocalPlan::Load(ref id, ref why) if id == "english" && why.contains("no multilingual checkpoint")
        ));
        assert!(matches!(plan_local(None, None, &[], &both(), true, false), Err(RuntimeError::NoneLoaded)));
        assert!(matches!(plan_local(None, None, &[], &[], false, true), Err(RuntimeError::NoneInstalled)));
    }

    #[test]
    fn an_english_default_hands_vietnamese_to_the_multilingual_checkpoint() {
        let installed = v(&[("english", En), ("multilingual", Ml), ("typed-decisions", En)]);
        let loaded = v(&[("typed-decisions", En)]);
        match plan_local(None, Some("typed-decisions"), &loaded, &installed, false, true).unwrap() {
            LocalPlan::Load(id, why) => {
                assert_eq!(id, "multilingual");
                assert!(why.contains("the default `typed-decisions` reads only English"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        // ASCII text: the default answers, as chosen.
        assert_eq!(
            plan_local(None, Some("typed-decisions"), &loaded, &installed, true, true).unwrap(),
            LocalPlan::Use("typed-decisions".into(), "the default model in Settings".into())
        );
        // No multilingual checkpoint at all: the default answers, with the warning.
        let english_only = v(&[("english", En), ("typed-decisions", En)]);
        assert!(matches!(
            plan_local(None, Some("typed-decisions"), &v(&[("english", En), ("typed-decisions", En)]), &english_only, false, true).unwrap(),
            LocalPlan::Use(ref id, ref why) if id == "typed-decisions" && why.contains("reads this text poorly")
        ));
        // A request that names a model still gets exactly that model.
        assert_eq!(
            plan_local(Some("typed-decisions"), Some("typed-decisions"), &loaded, &installed, false, true).unwrap(),
            LocalPlan::Use("typed-decisions".into(), "requested explicitly".into())
        );
    }

    #[test]
    fn idle_models_expire_after_the_limit_and_never_at_zero() {
        assert!(idle_expired(Duration::from_secs(15 * 60), 15));
        assert!(!idle_expired(Duration::from_secs(15 * 60 - 1), 15));
        assert!(!idle_expired(Duration::from_secs(10 * 24 * 3600), 0));
    }

    #[test]
    fn routing_reads_every_text_the_model_reads_and_only_counts_letters() {
        let q = Question::parse(
            "q",
            &js(r#"{"type": "choice", "instructions": "Pick", "criteria": {"hoàn tiền": "refund"}}"#),
        )
        .unwrap();
        assert!(!ascii_letters_only(&js(r#""plain""#), &[q]));
        let q = Question::parse("q", &js(r#"{"type": "noul", "instructions": "Is it?"}"#)).unwrap();
        assert!(ascii_letters_only(&js(r#"{"m": "plain"}"#), &[q.clone()]));
        assert!(!ascii_letters_only(&js(r#"{"m": "Chào"}"#), &[q.clone()]));
        // Punctuation and symbols say nothing about the language.
        assert!(ascii_letters_only(&js(r#""It’s done — really 👍""#), &[q]));
        // A noul's labels and descriptions are read too.
        let q = Question::parse(
            "q",
            &js(r#"{"type": "noul", "instructions": "Is it?", "labels": {"true": "Có", "false": "Không"}}"#),
        )
        .unwrap();
        assert!(!ascii_letters_only(&js(r#""plain""#), &[q]));
    }

    #[test]
    fn the_thread_setting_resolves_to_a_usable_count() {
        assert_eq!(resolve_threads(Some(3)), 3);
        let auto = resolve_threads(None);
        assert!((1..=8).contains(&auto));
        assert_eq!(resolve_threads(Some(0)), auto);
    }

    #[tokio::test]
    async fn a_malformed_request_is_refused_before_choosing_a_backend() {
        let mut s = DecisionSettings::default();
        s.backend = Backend::Online;
        let req: AskRequest = serde_json::from_str(r#"{"state": "x", "questions": {}}"#).unwrap();
        let tmp = tempfile::tempdir().unwrap();
        assert!(matches!(ask(req, &s, tmp.path()).await, Err(RuntimeError::BadRequest(_))));
    }

    /// Requests naming different checkpoints at the same moment must each get
    /// theirs: the single-flight is per id, so nothing but
    /// `ONE_LOAD_AT_A_TIME` keeps their ONNX Runtime sessions from being
    /// created together (which on macOS failed one load in a few dozen).
    /// Ignored by default because it needs exported checkpoints on disk:
    ///
    /// ```text
    /// SENCLAW_LAYA_LOAD_ROOT=<dir of two or more model folders> \
    ///   cargo test --features decision-laya same_moment -- --ignored --nocapture
    /// ```
    ///
    /// `SENCLAW_LAYA_LOAD_ROUNDS` sets how many times (default 20); linking a
    /// checkpoint into the folder under a second name adds a load per round.
    #[cfg(feature = "decision-laya")]
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs two or more exported Laya checkpoints: set SENCLAW_LAYA_LOAD_ROOT"]
    async fn checkpoints_asked_for_at_the_same_moment_all_load() {
        let root = std::env::var("SENCLAW_LAYA_LOAD_ROOT").expect("set SENCLAW_LAYA_LOAD_ROOT");
        let rounds: usize = std::env::var("SENCLAW_LAYA_LOAD_ROUNDS")
            .ok()
            .and_then(|r| r.parse().ok())
            .unwrap_or(20);
        let mut models: Vec<(String, ModelLayout)> = std::fs::read_dir(&root)
            .unwrap()
            .filter_map(|e| {
                let e = e.ok()?;
                let layout = ModelLayout::detect(&e.path()).ok()?;
                Some((e.file_name().to_string_lossy().into_owned(), layout))
            })
            .collect();
        models.sort_by(|a, b| a.0.cmp(&b.0));
        assert!(models.len() >= 2, "need two or more checkpoints under {root}");

        for round in 1..=rounds {
            let loads = models.iter().map(|(id, layout)| load(id, layout.clone(), None, true));
            let outcomes = futures::future::join_all(loads).await;
            let mut failed = Vec::new();
            for ((id, _), outcome) in models.iter().zip(outcomes) {
                match outcome {
                    Ok(info) => eprintln!("round {round}: `{id}` loaded in {} ms", info.load_ms),
                    Err(e) => failed.push(format!("`{id}`: {e}")),
                }
                unload(id);
            }
            assert!(failed.is_empty(), "round {round} of {rounds}: {}", failed.join("; "));
        }
    }

    #[cfg(not(feature = "decision-laya"))]
    #[tokio::test]
    async fn a_build_without_the_engine_says_so_for_local_requests() {
        let req: AskRequest =
            serde_json::from_str(r#"{"state": "x", "questions": {"q": {"type": "noul", "instructions": "y"}}}"#)
                .unwrap();
        let tmp = tempfile::tempdir().unwrap();
        assert!(matches!(
            ask(req, &DecisionSettings::default(), tmp.path()).await,
            Err(RuntimeError::NotCompiled)
        ));
    }
}
