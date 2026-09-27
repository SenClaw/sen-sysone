//! `sen-sysone`'s own HTTP surface: the old daemon namespace `/api/decision/*`
//! (minus `gate`/`skills`, which stayed in the daemon's control plane) plus
//! `POST /v1/systemone`, served **verbatim** — same paths, request and
//! response bodies, status codes — because desktop and web keep calling these
//! paths through the daemon's proxy. Ported from the daemon's
//! `src/gateway/ui_server/decision.rs`; only the plumbing that read the
//! daemon's `Config`/`UiState` changed, none of the behaviour.

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use sen_runtime_sdk::api::ErrorBody;
use sen_runtime_sdk::env::LaunchEnv;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::decision::laya::download::{self, DownloadSpec};
use crate::decision::laya::runtime::{self, RuntimeError};
use crate::decision::laya::store;
use crate::decision::online;
use crate::decision::settings::{Backend, DecisionSettings, DEFAULT_IDLE_UNLOAD_MINUTES, MAX_THREADS, PROVIDERS};
use crate::decision::types::{AskRequest, AskResponse};
use crate::settings_store;

/// Everything a handler needs besides the request itself.
pub struct AppState {
    pub env: LaunchEnv,
}

fn root(s: &AppState) -> PathBuf {
    store::root(&s.env.models_dir)
}

/// Read per request, so a save applies to the next ask without a restart.
fn settings(s: &AppState) -> DecisionSettings {
    let settings = settings_store::load(&s.env);
    runtime::apply_settings(&settings);
    settings
}

/// `~` and `~/…` expansion for a model-import path — the one filesystem input
/// this API takes from the caller.
fn expand_tilde(p: &str) -> PathBuf {
    if p == "~" {
        if let Some(home) = dirs::home_dir() {
            return home;
        }
    }
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(p)
}

pub struct AppError(pub StatusCode, pub String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.0, Json(ErrorBody::new(self.1))).into_response()
    }
}

fn internal(e: impl std::fmt::Display) -> AppError {
    AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

fn bad(msg: impl Into<String>) -> AppError {
    AppError(StatusCode::BAD_REQUEST, msg.into())
}

fn conflict(msg: impl Into<String>) -> AppError {
    AppError(StatusCode::CONFLICT, msg.into())
}

fn runtime_error(e: RuntimeError) -> AppError {
    let code = match &e {
        RuntimeError::NotCompiled => StatusCode::NOT_IMPLEMENTED,
        RuntimeError::NotInstalled(_) => StatusCode::NOT_FOUND,
        RuntimeError::NotLoaded(_)
        | RuntimeError::NoneLoaded
        | RuntimeError::NoneInstalled
        | RuntimeError::NotConfigured(_)
        | RuntimeError::Refused(_) => StatusCode::CONFLICT,
        // What laya-serve answers for an invalid request body.
        RuntimeError::BadRequest(_) => StatusCode::UNPROCESSABLE_ENTITY,
        RuntimeError::Upstream(_) => StatusCode::BAD_GATEWAY,
        RuntimeError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
    };
    AppError(code, e.to_string())
}

fn checked_id(id: &str) -> Result<(), AppError> {
    if store::valid_id(id) {
        Ok(())
    } else {
        Err(bad(format!(
            "`{id}` is not a model id: use letters, digits, `.`, `_` or `-` (at most 64)"
        )))
    }
}

/// A new model may take `id` only when nothing already lives there.
fn slot_is_free(root: &std::path::Path, id: &str) -> Result<(), AppError> {
    checked_id(id)?;
    if download::is_active(id) {
        return Err(conflict(format!("`{id}` already has a download or import in progress")));
    }
    if store::installed_layout(&root.join(id)).is_some() {
        return Err(conflict(format!(
            "`{id}` is already installed — delete it first to replace it"
        )));
    }
    Ok(())
}

async fn decision_models_list(State(s): State<Arc<AppState>>) -> Result<Json<Value>, AppError> {
    let root = root(&s);
    let listed = root.clone();
    // `list` sizes every model directory; keep the walk off the async workers.
    let models = tokio::task::spawn_blocking(move || store::list(&listed))
        .await
        .map_err(internal)?;
    let settings = settings(&s);
    Ok(Json(json!({
        "compiled": runtime::COMPILED,
        "root": root,
        "models": models,
        // What the rows need to say "default" and "unloads in …" — no secrets.
        "backend": settings.backend,
        "local": settings.local,
    })))
}

async fn decision_model_download(
    State(s): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    let entry = store::catalog_get(&id).ok_or_else(|| {
        AppError(
            StatusCode::NOT_FOUND,
            format!("`{id}` is not in the catalog — use the custom download for other repos"),
        )
    })?;
    let root = root(&s);
    slot_is_free(&root, &id)?;
    download::start_download(
        root,
        DownloadSpec {
            id: id.clone(),
            label: Some(entry.label.to_string()),
            source_kind: "catalog",
            repo: entry.repo.to_string(),
            revision: entry.revision.to_string(),
            files: entry.files.iter().map(|f| f.to_string()).collect(),
            manifest: entry.manifest.map(str::to_string),
        },
    )
    .map_err(conflict)?;
    Ok(Json(json!({ "ok": true, "id": id })))
}

#[derive(Deserialize)]
struct CustomBody {
    id: String,
    repo: String,
    #[serde(default)]
    revision: Option<String>,
}

/// `org/name` from a bare id or a Hub URL.
fn normalize_repo(raw: &str) -> Result<String, String> {
    let s = raw
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("huggingface.co/")
        .trim_start_matches("hf.co/")
        .trim_end_matches('/');
    let mut parts = s.split('/');
    let (Some(org), Some(name)) = (parts.next(), parts.next()) else {
        return Err(format!("expected a Hugging Face repo as `org/name`, got `{raw}`"));
    };
    let ok = |p: &str| {
        !p.is_empty() && p != "." && p != ".." && p.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    };
    if !ok(org) || !ok(name) {
        return Err(format!("`{raw}` is not a valid `org/name` repo id"));
    }
    Ok(format!("{org}/{name}"))
}

async fn decision_model_custom(
    State(s): State<Arc<AppState>>,
    Json(body): Json<CustomBody>,
) -> Result<Json<Value>, AppError> {
    let root = root(&s);
    slot_is_free(&root, &body.id)?;
    let repo = normalize_repo(&body.repo).map_err(bad)?;
    let revision = body.revision.as_deref().map(str::trim).filter(|r| !r.is_empty()).unwrap_or("main");
    if !revision.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')) {
        return Err(bad(format!("`{revision}` is not a branch, tag or commit")));
    }
    // Pin before listing, so the files chosen and the files fetched are the same commit.
    let sha = download::resolve_revision(&repo, revision)
        .await
        .map_err(|e| bad(format!("{e:#}")))?;
    let (files, manifest) = download::plan_custom(&repo, &sha)
        .await
        .map_err(|e| bad(format!("{e:#}")))?;
    download::start_download(
        root,
        DownloadSpec {
            id: body.id.clone(),
            label: Some(repo.clone()),
            source_kind: "huggingface",
            repo,
            revision: sha.clone(),
            files: files.clone(),
            manifest,
        },
    )
    .map_err(conflict)?;
    Ok(Json(json!({ "ok": true, "id": body.id, "revision": sha, "files": files })))
}

#[derive(Deserialize)]
struct ImportBody {
    path: String,
    #[serde(default)]
    id: Option<String>,
}

async fn decision_model_import(
    State(s): State<Arc<AppState>>,
    Json(body): Json<ImportBody>,
) -> Result<Json<Value>, AppError> {
    let src = expand_tilde(body.path.trim());
    if !src.is_absolute() {
        return Err(bad("the folder must be an absolute path"));
    }
    if !src.is_dir() {
        return Err(bad(format!("{} is not a folder", src.display())));
    }
    let id = match body.id.as_deref().map(str::trim).filter(|i| !i.is_empty()) {
        Some(id) => id.to_string(),
        None => src
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .ok_or_else(|| bad("name the model: the folder has no usable name"))?,
    };
    let root = root(&s);
    slot_is_free(&root, &id)?;
    download::start_import(root, id.clone(), src).map_err(bad)?;
    Ok(Json(json!({ "ok": true, "id": id })))
}

async fn decision_model_cancel(AxumPath(id): AxumPath<String>) -> Result<Json<Value>, AppError> {
    checked_id(&id)?;
    Ok(Json(json!({ "ok": true, "cancelled": download::cancel(&id) })))
}

async fn decision_model_load(
    State(s): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    checked_id(&id)?;
    if !runtime::COMPILED {
        return Err(runtime_error(RuntimeError::NotCompiled));
    }
    let layout = store::installed_layout(&root(&s).join(&id))
        .ok_or_else(|| runtime_error(RuntimeError::NotInstalled(id.clone())))?;
    let threads = settings(&s).local.threads;
    let info = runtime::load(&id, layout, threads, false).await.map_err(runtime_error)?;
    Ok(Json(json!({ "ok": true, "id": id, "loaded": info })))
}

async fn decision_model_unload(AxumPath(id): AxumPath<String>) -> Result<Json<Value>, AppError> {
    checked_id(&id)?;
    Ok(Json(json!({ "ok": true, "unloaded": runtime::unload(&id) })))
}

async fn decision_model_delete(
    State(s): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    checked_id(&id)?;
    if download::is_active(&id) {
        return Err(conflict(format!("`{id}` is downloading — cancel it first")));
    }
    // Held across unload and removal, so no hot-load can slip in between and
    // leave weights in RAM for a directory that is gone.
    let _deleting = runtime::begin_delete(&id)
        .ok_or_else(|| conflict(format!("`{id}` is being loaded — wait for it to finish")))?;
    store::delete(&root(&s), &id).await.map_err(internal)?;
    // A default that names a deleted model would fail every request that
    // names none, so the default goes with it.
    let mut settings = settings(&s);
    let default_cleared = settings.local.default_model.as_deref() == Some(id.as_str());
    if default_cleared {
        settings.local.default_model = None;
        settings_store::save(&s.env, &settings).map_err(internal)?;
    }
    Ok(Json(json!({ "ok": true, "defaultCleared": default_cleared })))
}

/// Also mounted at `POST /v1/systemone` (Jev's `{model?, state, questions}` —
/// `AskRequest.backend` is optional, so the same handler answers both).
async fn decision_ask(
    State(s): State<Arc<AppState>>,
    Json(req): Json<AskRequest>,
) -> Result<Json<AskResponse>, AppError> {
    let settings = settings(&s);
    runtime::ask(req, &settings, &root(&s)).await.map(Json).map_err(runtime_error)
}

/// The settings plus what the form needs to explain them.
fn settings_view(settings: &DecisionSettings) -> Value {
    json!({
        "compiled": runtime::COMPILED,
        "settings": settings.masked(),
        "providers": PROVIDERS,
        "defaults": {
            "threads": runtime::resolve_threads(None),
            "maxThreads": MAX_THREADS,
            "idleUnloadMinutes": DEFAULT_IDLE_UNLOAD_MINUTES,
            "typesafeUrl": online::TYPESAFE_URL,
            "typesafeModel": online::TYPESAFE_DEFAULT_MODEL,
            "cloudflareModel": online::CLOUDFLARE_DEFAULT_MODEL,
        },
    })
}

async fn decision_settings_get(State(s): State<Arc<AppState>>) -> Json<Value> {
    Json(settings_view(&settings(&s)))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SettingsBody {
    #[serde(flatten)]
    settings: DecisionSettings,
    /// The form never holds the stored key, so an empty key means "keep it";
    /// this is how a person removes it.
    #[serde(default)]
    clear_api_key: bool,
}

/// Validate a submitted form and give it the stored key when it sent none.
fn resolve_form(s: &AppState, body: SettingsBody) -> Result<DecisionSettings, AppError> {
    let previous = settings_store::load(&s.env);
    let settings = body
        .settings
        .validated()
        .map_err(|e| AppError(StatusCode::UNPROCESSABLE_ENTITY, e))?
        .merge_key_from(&previous, body.clear_api_key);
    Ok(settings)
}

async fn decision_settings_put(
    State(s): State<Arc<AppState>>,
    Json(body): Json<SettingsBody>,
) -> Result<Json<Value>, AppError> {
    let settings = resolve_form(&s, body)?;
    if let Some(id) = &settings.local.default_model {
        if store::installed_layout(&root(&s).join(id)).is_none() {
            return Err(AppError(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("`{id}` is not installed — install it before making it the default"),
            ));
        }
    }
    settings_store::save(&s.env, &settings).map_err(internal)?;
    runtime::apply_settings(&settings);
    let mut view = settings_view(&settings);
    view["ok"] = Value::Bool(true);
    Ok(Json(view))
}

/// One tiny request against the online backend as the form describes it —
/// saved or not — so a wrong key or account shows up before the first real ask.
async fn decision_online_test(
    State(s): State<Arc<AppState>>,
    Json(body): Json<SettingsBody>,
) -> Result<Json<OnlineTestResult>, AppError> {
    let mut settings = resolve_form(&s, body)?;
    settings.backend = Backend::Online;
    let questions: crate::decision::json::Json = serde_json::from_str(
        r#"{"probe": {"type": "noul", "instructions": "Is this message a greeting?"}}"#,
    )
    .map_err(internal)?;
    let state = crate::decision::json::Json::String("Hello there!".into());
    let resp = online::ask(&settings.online, None, &state, &questions)
        .await
        .map_err(runtime_error)?;
    // Serialized directly, not through `json!`: a `serde_json::Value` here
    // sorts the answer's keys, and the point is to show what the backend sent.
    Ok(Json(OnlineTestResult {
        ok: true,
        provider: settings.online.provider,
        model: resp.model,
        latency_ms: resp.latency_ms,
        answers: resp.answers,
    }))
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct OnlineTestResult {
    ok: bool,
    provider: String,
    model: String,
    latency_ms: f64,
    answers: crate::decision::types::Answers,
}

/// The full router this runtime serves.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/decision/models", get(decision_models_list))
        .route("/api/decision/models/custom", post(decision_model_custom))
        .route("/api/decision/models/import", post(decision_model_import))
        .route("/api/decision/models/:id/download", post(decision_model_download))
        .route("/api/decision/models/:id/cancel", post(decision_model_cancel))
        .route("/api/decision/models/:id/load", post(decision_model_load))
        .route("/api/decision/models/:id/unload", post(decision_model_unload))
        .route(
            "/api/decision/models/:id",
            axum::routing::delete(decision_model_delete),
        )
        .route("/api/decision/ask", post(decision_ask))
        .route(
            "/api/decision/settings",
            get(decision_settings_get).put(decision_settings_put),
        )
        .route("/api/decision/online/test", post(decision_online_test))
        .route("/v1/systemone", post(decision_ask))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_ids_come_from_ids_or_urls_and_reject_traversal() {
        assert_eq!(normalize_repo("ti3x-m/laya-onnx").unwrap(), "ti3x-m/laya-onnx");
        assert_eq!(
            normalize_repo("https://huggingface.co/receptron/laya-onnx/").unwrap(),
            "receptron/laya-onnx"
        );
        for bad in ["laya", "../x", "a/..", "a b/c", ""] {
            assert!(normalize_repo(bad).is_err(), "{bad}");
        }
    }

    fn state_at(dir: &std::path::Path) -> Arc<AppState> {
        let env = LaunchEnv::from_lookup("sen-sysone", "0.0.0-test", |k| match k {
            "SENCLAW_RUNTIME_DATA_DIR" => Some(dir.join("data").to_string_lossy().into_owned()),
            "SENCLAW_LOCAL_MODELS_DIR" => Some(dir.join("models").to_string_lossy().into_owned()),
            "SENCLAW_CONFIG_PATH" => Some(dir.join("config.json").to_string_lossy().into_owned()),
            "SENCLAW_HOME" => Some(dir.to_string_lossy().into_owned()),
            _ => None,
        });
        Arc::new(AppState { env })
    }

    /// Full bearer-token gating is the SDK server scaffold's job (tested
    /// there); this checks the routes this crate owns answer with the old
    /// shapes once mounted, and that bad ids are rejected before touching disk.
    #[tokio::test]
    async fn settings_and_bad_ids_answer_with_the_old_shapes() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let tmp = tempfile::tempdir().unwrap();
        let app = router(state_at(tmp.path()));

        let resp = app
            .clone()
            .oneshot(Request::get("/api/decision/settings").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["settings"]["backend"], "local");
        assert_eq!(v["settings"]["online"]["hasApiKey"], false);
        assert!(v.get("providers").is_some());

        let resp = app
            .oneshot(
                Request::post("/api/decision/models/../etc/load")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // Path traversal in the id segment never reaches `checked_id`: axum
        // normalises `..` out of the route match, landing on a different (and
        // here unregistered) path — proving the segment cannot escape `models/`.
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
