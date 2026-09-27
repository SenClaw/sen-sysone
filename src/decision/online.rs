//! The hosted backend: TypeSafe's Jev, or anything else that speaks the
//! `/v1/systemone` wire.
//!
//! | provider | endpoint | body | model sent |
//! |---|---|---|---|
//! | `typesafe` | `https://api.typesafe.ai/v1/systemone` | `{model, state, questions}` | `jev-1.13.0` (pinned) |
//! | `cloudflare` | `…/accounts/{account}/ai/run` | `{model, input: {state, questions}}` | `typesafe/jev` |
//! | `custom` | the configured URL (LiteLLM, laya-serve, OpenJev…) | `{model?, state, questions}` | configured, or none |
//!
//! Cloudflare's REST `/ai/run` nests the request under `input` (its Workers
//! binding does not — that is `env.AI.run(model, {state, questions})`).
//!
//! Answers come back verbatim ([`Answers::Online`]): the wire is shared, the
//! fields inside an answer are the provider's own. What we do own is the error
//! text — a 401 names the key, a 429 the rate limit, a timeout the budget —
//! because a bare "HTTP 401" is not something a person can act on.
//!
//! Using this backend sends the state to the provider; the settings UI says so.

use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use serde::Serialize;

use super::json::Json;
use super::laya::runtime::RuntimeError;
use super::settings::OnlineSettings;
use super::types::{AskResponse, Answers, Routing, Usage};

pub const TYPESAFE_URL: &str = "https://api.typesafe.ai/v1/systemone";
/// Pinned rather than `jev-latest`: a moving alias changes answers under the caller.
pub const TYPESAFE_DEFAULT_MODEL: &str = "jev-1.13.0";
pub const CLOUDFLARE_DEFAULT_MODEL: &str = "typesafe/jev";

/// Where a request goes for these settings.
pub fn endpoint(s: &OnlineSettings) -> Result<String, RuntimeError> {
    match s.provider.as_str() {
        "typesafe" => Ok(TYPESAFE_URL.to_string()),
        "cloudflare" if s.account_id.is_empty() => Err(RuntimeError::NotConfigured(
            "Cloudflare needs its account id — set it in Settings → Decision (Laya)".into(),
        )),
        "cloudflare" => Ok(format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/ai/run",
            s.account_id
        )),
        "custom" if s.url.is_empty() => Err(RuntimeError::NotConfigured(
            "the custom online backend has no URL — set it in Settings → Decision (Laya)".into(),
        )),
        "custom" => Ok(s.url.clone()),
        other => Err(RuntimeError::NotConfigured(format!("`{other}` is not an online provider"))),
    }
}

/// The model id a request carries upstream.
pub fn model_for(s: &OnlineSettings, requested: Option<&str>) -> Option<String> {
    if let Some(m) = requested.map(str::trim).filter(|m| !m.is_empty() && *m != "auto") {
        return Some(m.to_string());
    }
    if !s.model.is_empty() {
        return Some(s.model.clone());
    }
    match s.provider.as_str() {
        "typesafe" => Some(TYPESAFE_DEFAULT_MODEL.into()),
        "cloudflare" => Some(CLOUDFLARE_DEFAULT_MODEL.into()),
        _ => None,
    }
}

#[derive(Serialize)]
struct Input<'a> {
    state: &'a Json,
    questions: &'a Json,
}

/// The request as each provider wants it; fields serialize in this order.
#[derive(Serialize)]
#[serde(untagged)]
enum Body<'a> {
    /// `/v1/systemone` itself: TypeSafe and every compatible endpoint.
    Direct {
        #[serde(skip_serializing_if = "Option::is_none")]
        model: Option<&'a str>,
        state: &'a Json,
        questions: &'a Json,
    },
    /// Cloudflare's REST `/ai/run`.
    Wrapped { model: &'a str, input: Input<'a> },
}

fn body<'a>(provider: &str, model: Option<&'a str>, state: &'a Json, questions: &'a Json) -> Body<'a> {
    match (provider, model) {
        ("cloudflare", Some(model)) => Body::Wrapped {
            model,
            input: Input { state, questions },
        },
        _ => Body::Direct { model, state, questions },
    }
}

/// One client for every ask: a decision is meant to be a fast path, and a
/// fresh client pays a new connection and TLS handshake each time. The
/// per-request timeout comes from the settings.
static CLIENT: Lazy<reqwest::Client> = Lazy::new(|| {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
});

/// Cloudflare wraps every REST answer as `{"result": …, "success": true}`;
/// the others return the `/v1/systemone` body itself.
fn unwrap_envelope(v: Json) -> Json {
    if v.get("answers").is_none() {
        if let Some(inner @ Json::Object(_)) = v.get("result") {
            return inner.clone();
        }
    }
    v
}

/// A person-readable reason for a failed upstream call.
fn upstream_error(provider: &str, status: u16, body: &str) -> RuntimeError {
    // The provider's own message when it sent one (`{"error": …}` or
    // Cloudflare's `{"errors": [{"message": …}]}`), else the start of the body.
    let detail = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v["error"]["message"]
                .as_str()
                .or(v["error"].as_str())
                .or(v["errors"][0]["message"].as_str())
                .or(v["detail"].as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| body.chars().take(200).collect());
    let what = match status {
        401 | 403 => "rejected the API key".to_string(),
        404 => "does not know this endpoint or model".to_string(),
        422 => "refused the request".to_string(),
        429 => "is rate-limiting this key — retry in a moment".to_string(),
        500..=599 => "is failing or overloaded".to_string(),
        _ => "answered with an unexpected status".to_string(),
    };
    RuntimeError::Upstream(if detail.trim().is_empty() {
        format!("{provider} {what} (HTTP {status})")
    } else {
        format!("{provider} {what} (HTTP {status}): {detail}")
    })
}

/// Send one already-validated request to the online backend.
pub async fn ask(
    s: &OnlineSettings,
    requested_model: Option<&str>,
    state: &Json,
    questions: &Json,
) -> Result<AskResponse, RuntimeError> {
    let url = endpoint(s)?;
    if s.api_key.is_empty() && s.provider != "custom" {
        return Err(RuntimeError::NotConfigured(format!(
            "the {} backend needs an API key — set it in Settings → Decision (Laya)",
            s.provider
        )));
    }
    let model = model_for(s, requested_model);
    let body = serde_json::to_vec(&body(&s.provider, model.as_deref(), state, questions))
        .map_err(|e| RuntimeError::Internal(e.to_string()))?;

    let mut req = CLIENT
        .post(&url)
        .timeout(Duration::from_secs(s.timeout_secs))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body);
    if !s.api_key.is_empty() {
        req = req.bearer_auth(&s.api_key);
    }

    let started = Instant::now();
    let resp = req.send().await.map_err(|e| {
        if e.is_timeout() {
            RuntimeError::Upstream(format!("{} did not answer within {} s", s.provider, s.timeout_secs))
        } else {
            RuntimeError::Upstream(format!("could not reach {}: {e}", s.provider))
        }
    })?;
    let status = resp.status().as_u16();
    let text = resp
        .text()
        .await
        .map_err(|e| RuntimeError::Upstream(format!("{} sent an unreadable answer: {e}", s.provider)))?;
    if !(200..300).contains(&status) {
        return Err(upstream_error(&s.provider, status, &text));
    }
    let v: Json = serde_json::from_str(&text)
        .map_err(|e| RuntimeError::Upstream(format!("{} answered with non-JSON: {e}", s.provider)))?;
    let v = unwrap_envelope(v);
    let answers = v
        .get("answers")
        .filter(|a| a.as_object().is_some())
        .cloned()
        .ok_or_else(|| RuntimeError::Upstream(format!("{} answered without `answers`", s.provider)))?;
    // A partial answer is a failure: a caller reading `answers[id]` would
    // otherwise find nothing and treat the question as answered "no".
    if let (Some(asked), Some(got)) = (questions.as_object(), answers.as_object()) {
        if let Some((id, _)) = asked.iter().find(|(id, _)| !got.iter().any(|(g, _)| g == id)) {
            return Err(RuntimeError::Upstream(format!("{} left question `{id}` unanswered", s.provider)));
        }
    }
    let tokens = |field: &str| match v.get("usage").and_then(|u| u.get(field)) {
        Some(Json::Number(n)) => n.as_u64().unwrap_or(0) as usize,
        _ => 0,
    };
    let answered_by = v
        .get("model")
        .and_then(Json::as_str)
        .map(str::to_string)
        .or(model)
        .unwrap_or_else(|| s.provider.clone());
    Ok(AskResponse {
        model: answered_by.clone(),
        engine: format!("online:{}", s.provider),
        answers: Answers::Online(answers),
        usage: Usage {
            input_tokens: tokens("input_tokens"),
            output_tokens: tokens("output_tokens"),
        },
        latency_ms: (started.elapsed().as_secs_f64() * 10_000.0).round() / 10.0,
        runs: 1,
        routing: Routing {
            model: answered_by,
            reason: format!("online backend ({})", s.provider),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(provider: &str) -> OnlineSettings {
        OnlineSettings {
            provider: provider.into(),
            ..OnlineSettings::default()
        }
    }

    #[test]
    fn each_provider_has_its_endpoint_and_default_model() {
        assert_eq!(endpoint(&settings("typesafe")).unwrap(), TYPESAFE_URL);
        assert!(matches!(endpoint(&settings("cloudflare")), Err(RuntimeError::NotConfigured(_))));
        let mut cf = settings("cloudflare");
        cf.account_id = "abc123".into();
        assert_eq!(
            endpoint(&cf).unwrap(),
            "https://api.cloudflare.com/client/v4/accounts/abc123/ai/run"
        );
        assert!(matches!(endpoint(&settings("custom")), Err(RuntimeError::NotConfigured(_))));

        assert_eq!(model_for(&settings("typesafe"), None).as_deref(), Some(TYPESAFE_DEFAULT_MODEL));
        assert_eq!(model_for(&settings("cloudflare"), Some("auto")).as_deref(), Some(CLOUDFLARE_DEFAULT_MODEL));
        assert_eq!(model_for(&settings("custom"), None), None);
        assert_eq!(model_for(&settings("typesafe"), Some("jev-preview")).as_deref(), Some("jev-preview"));
    }

    #[test]
    fn a_cloudflare_envelope_is_unwrapped_and_order_is_kept() {
        let wrapped: Json = serde_json::from_str(
            r#"{"result": {"model": "jev-1.13.0", "answers": {"z": {"type": "noul", "noul": 0.1}, "a": {"type": "noul", "noul": 0.9}}}, "success": true}"#,
        )
        .unwrap();
        let inner = unwrap_envelope(wrapped);
        let keys: Vec<&str> = inner.get("answers").unwrap().as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["z", "a"]);
        let plain: Json = serde_json::from_str(r#"{"answers": {}, "result": {"answers": {"x": 1}}}"#).unwrap();
        assert!(unwrap_envelope(plain).get("result").is_some(), "a body with answers is not an envelope");
    }

    #[test]
    fn upstream_failures_read_as_what_to_do() {
        let e = upstream_error("typesafe", 401, r#"{"error": "invalid key"}"#).to_string();
        assert!(e.contains("rejected the API key") && e.contains("invalid key"), "{e}");
        let e = upstream_error("cloudflare", 429, r#"{"errors": [{"message": "slow down"}]}"#).to_string();
        assert!(e.contains("rate-limiting") && e.contains("slow down"), "{e}");
        let e = upstream_error("custom", 503, "").to_string();
        assert!(e.contains("overloaded") && e.contains("503"), "{e}");
        let e = upstream_error("custom", 405, "").to_string();
        assert_eq!(e.matches("405").count(), 1, "{e}");
    }

    #[test]
    fn cloudflare_nests_the_request_under_input_and_the_others_do_not() {
        let state: Json = serde_json::from_str(r#""Hi""#).unwrap();
        let q: Json = serde_json::from_str(r#"{"z": {"type": "noul", "instructions": "x"}, "a": {"type": "noul", "instructions": "y"}}"#).unwrap();
        let cf = serde_json::to_string(&body("cloudflare", Some(CLOUDFLARE_DEFAULT_MODEL), &state, &q)).unwrap();
        assert_eq!(
            cf,
            r#"{"model":"typesafe/jev","input":{"state":"Hi","questions":{"z":{"type":"noul","instructions":"x"},"a":{"type":"noul","instructions":"y"}}}}"#
        );
        let ts = serde_json::to_string(&body("typesafe", Some(TYPESAFE_DEFAULT_MODEL), &state, &q)).unwrap();
        assert!(ts.starts_with(r#"{"model":"jev-1.13.0","state":"Hi","questions":{"z""#), "{ts}");
        let custom = serde_json::to_string(&body("custom", None, &state, &q)).unwrap();
        assert!(custom.starts_with(r#"{"state":"Hi","#), "no model key when none is configured: {custom}");
    }

    #[tokio::test]
    async fn a_missing_key_is_refused_before_any_request() {
        let q: Json = serde_json::from_str(r#"{"q": {"type": "noul", "instructions": "x"}}"#).unwrap();
        let err = ask(&settings("typesafe"), None, &Json::String("s".into()), &q).await.unwrap_err();
        assert!(matches!(err, RuntimeError::NotConfigured(_)), "{err}");
    }
}
