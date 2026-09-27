//! What the user picked for typed decisions: which backend answers, which
//! local model is the default, how the local engine runs, and when idle
//! weights are given back. Persisted as `<data_dir>/settings.json`, seeded
//! once from the daemon's old `decisionConfig` key
//! ([`sen_runtime_sdk::legacy::load_or_import`]), and read per request so a
//! change applies to the next ask without restarting anything.
//!
//! `gate` and `skills` are **not** part of this struct: the tool-call gate and
//! the pre-skill router stayed in the daemon (they need the skills registry
//! and the permission bridge, neither of which exists here) and keep their own
//! settings there. The daemon merges its own `gate`/`skills` into what
//! `GET /api/decision/settings` shows so existing clients render unchanged.

use serde::{Deserialize, Serialize};

/// Where a request is answered when it does not say.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// A Laya checkpoint on this machine (ONNX Runtime).
    #[default]
    Local,
    /// A hosted `/v1/systemone` — TypeSafe's Jev, or anything wire-compatible.
    Online,
}

pub const DEFAULT_IDLE_UNLOAD_MINUTES: u64 = 15;
pub const MAX_IDLE_UNLOAD_MINUTES: u64 = 24 * 60;
pub const MAX_THREADS: usize = 64;
pub const DEFAULT_ONLINE_TIMEOUT_SECS: u64 = 15;
/// The desktop client gives up on any request after 30 s. This runtime must
/// answer first — even if only to say the provider timed out — or the user
/// sees a client timeout while it is still waiting.
pub const MAX_ONLINE_TIMEOUT_SECS: u64 = 25;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalSettings {
    /// The model that answers a request naming none. `None` = pick by
    /// language among the loaded (or, with `auto_load`, installed) models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// ONNX Runtime intra-op threads. `None` = the core count, capped at 8.
    /// A loaded model keeps the count it was loaded with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<usize>,
    /// Load the model a request needs instead of refusing it ("hot-load").
    #[serde(default = "yes")]
    pub auto_load: bool,
    /// Unload a model no request has used for this long. `0` = never.
    #[serde(default = "default_idle")]
    pub idle_unload_minutes: u64,
}

impl Default for LocalSettings {
    fn default() -> Self {
        LocalSettings {
            default_model: None,
            threads: None,
            auto_load: true,
            idle_unload_minutes: DEFAULT_IDLE_UNLOAD_MINUTES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OnlineSettings {
    /// `typesafe` (Jev, api.typesafe.ai), `cloudflare` (Workers AI) or
    /// `custom` (any `/v1/systemone` URL: LiteLLM, laya-serve, OpenJev…).
    #[serde(default = "default_provider")]
    pub provider: String,
    /// Full endpoint for `custom`; ignored by the other providers.
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub api_key: String,
    /// Model id sent upstream; empty = the provider's default.
    #[serde(default)]
    pub model: String,
    /// Cloudflare account id (`cloudflare` only).
    #[serde(default)]
    pub account_id: String,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

impl Default for OnlineSettings {
    fn default() -> Self {
        OnlineSettings {
            provider: default_provider(),
            url: String::new(),
            api_key: String::new(),
            model: String::new(),
            account_id: String::new(),
            timeout_secs: DEFAULT_ONLINE_TIMEOUT_SECS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct DecisionSettings {
    #[serde(default)]
    pub backend: Backend,
    #[serde(default)]
    pub local: LocalSettings,
    #[serde(default)]
    pub online: OnlineSettings,
}

fn yes() -> bool {
    true
}
fn default_idle() -> u64 {
    DEFAULT_IDLE_UNLOAD_MINUTES
}
fn default_provider() -> String {
    "typesafe".into()
}
fn default_timeout() -> u64 {
    DEFAULT_ONLINE_TIMEOUT_SECS
}

pub const PROVIDERS: &[&str] = &["typesafe", "cloudflare", "custom"];

impl DecisionSettings {
    /// Refuse what cannot work, name the field; clamp what is merely out of range.
    pub fn validated(mut self) -> Result<DecisionSettings, String> {
        if let Some(m) = &self.local.default_model {
            let m = m.trim();
            if m.is_empty() || m == "auto" {
                self.local.default_model = None;
            } else if !crate::decision::laya::store::valid_id(m) {
                return Err(format!("`{m}` is not a model id"));
            } else {
                self.local.default_model = Some(m.to_string());
            }
        }
        self.local.threads = self.local.threads.filter(|t| *t > 0).map(|t| t.min(MAX_THREADS));
        self.local.idle_unload_minutes = self.local.idle_unload_minutes.min(MAX_IDLE_UNLOAD_MINUTES);

        let o = &mut self.online;
        o.provider = o.provider.trim().to_ascii_lowercase();
        if !PROVIDERS.contains(&o.provider.as_str()) {
            return Err(format!(
                "`{}` is not an online provider; use one of {}",
                o.provider,
                PROVIDERS.join(", ")
            ));
        }
        o.url = o.url.trim().to_string();
        o.model = o.model.trim().to_string();
        o.api_key = o.api_key.trim().to_string();
        o.account_id = o.account_id.trim().to_string();
        o.timeout_secs = o.timeout_secs.clamp(5, MAX_ONLINE_TIMEOUT_SECS);
        // Each check only for the provider whose field it is: the forms hide
        // the others, and a stale value there must not block every save.
        if o.provider == "custom" && !o.url.is_empty() && !(o.url.starts_with("https://") || o.url.starts_with("http://")) {
            return Err("the custom URL must start with http:// or https://".into());
        }
        if o.provider == "cloudflare" && !o.account_id.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err("a Cloudflare account id is letters and digits only".into());
        }
        Ok(self)
    }

    /// What `GET` returns: the key is never sent back, only whether one is set.
    pub fn masked(&self) -> serde_json::Value {
        let mut v = serde_json::to_value(self).unwrap_or_default();
        v["online"]["apiKey"] = serde_json::Value::String(String::new());
        v["online"]["hasApiKey"] = serde_json::Value::Bool(!self.online.api_key.is_empty());
        v
    }

    /// A settings form never receives the stored key, so an empty key in a
    /// save means "unchanged" — unless the caller asked to clear it, or the
    /// key would now go somewhere else. A key saved for one provider (or one
    /// custom URL) is never sent to another: switching to a LAN `http://`
    /// endpoint would otherwise hand it over in cleartext, and a test request
    /// could point it at any URL at all.
    pub fn merge_key_from(mut self, previous: &DecisionSettings, clear_key: bool) -> DecisionSettings {
        if self.online.api_key.trim().is_empty() && !clear_key && self.online.same_key_scope(&previous.online) {
            self.online.api_key = previous.online.api_key.clone();
        }
        self
    }
}

impl OnlineSettings {
    /// Whether a key held for `other` may be used for these settings: the
    /// same provider, and for `custom` the same URL.
    pub fn same_key_scope(&self, other: &OnlineSettings) -> bool {
        self.provider == other.provider && (self.provider != "custom" || self.url == other.url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_config_means_local_with_hot_load_and_a_15_minute_unload() {
        let s: DecisionSettings = serde_json::from_str("{}").unwrap();
        assert_eq!(s.backend, Backend::Local);
        assert!(s.local.auto_load);
        assert_eq!(s.local.idle_unload_minutes, 15);
        assert_eq!(s.online.provider, "typesafe");
    }

    #[test]
    fn validation_clamps_ranges_and_refuses_what_cannot_work() {
        let mut s = DecisionSettings::default();
        s.local.threads = Some(0);
        s.local.idle_unload_minutes = 100_000;
        s.local.default_model = Some("auto".into());
        let v = s.validated().unwrap();
        assert_eq!(v.local.threads, None);
        assert_eq!(v.local.idle_unload_minutes, MAX_IDLE_UNLOAD_MINUTES);
        assert_eq!(v.local.default_model, None);

        let mut bad = DecisionSettings::default();
        bad.online.provider = "openai".into();
        assert!(bad.validated().unwrap_err().contains("not an online provider"));
        let mut bad = DecisionSettings::default();
        bad.local.default_model = Some("../x".into());
        assert!(bad.validated().is_err());
        let mut bad = DecisionSettings::default();
        bad.online.provider = "custom".into();
        bad.online.url = "ftp://x".into();
        assert!(bad.validated().is_err());
    }

    #[test]
    fn the_key_is_masked_and_an_empty_save_keeps_it() {
        let mut old = DecisionSettings::default();
        old.online.api_key = "sk-secret".into();
        let masked = old.masked();
        assert_eq!(masked["online"]["apiKey"], "");
        assert_eq!(masked["online"]["hasApiKey"], true);
        assert!(!masked.to_string().contains("sk-secret"));

        let incoming = DecisionSettings::default();
        assert_eq!(incoming.clone().merge_key_from(&old, false).online.api_key, "sk-secret");
        assert_eq!(incoming.merge_key_from(&old, true).online.api_key, "");
    }

    #[test]
    fn a_stored_key_never_follows_a_change_of_provider_or_custom_url() {
        let mut old = DecisionSettings::default();
        old.online.api_key = "sk-typesafe".into();
        let mut to_custom = DecisionSettings::default();
        to_custom.online.provider = "custom".into();
        to_custom.online.url = "http://192.168.1.20:8000/v1/systemone".into();
        assert_eq!(to_custom.merge_key_from(&old, false).online.api_key, "");
        let mut to_cf = DecisionSettings::default();
        to_cf.online.provider = "cloudflare".into();
        assert_eq!(to_cf.merge_key_from(&old, false).online.api_key, "");

        let mut custom = DecisionSettings::default();
        custom.online.provider = "custom".into();
        custom.online.url = "https://a.example/v1/systemone".into();
        custom.online.api_key = "sk-custom".into();
        let mut moved = custom.clone();
        moved.online.api_key = String::new();
        moved.online.url = "https://b.example/v1/systemone".into();
        assert_eq!(moved.merge_key_from(&custom, false).online.api_key, "");
        let mut same = custom.clone();
        same.online.api_key = String::new();
        assert_eq!(same.merge_key_from(&custom, false).online.api_key, "sk-custom");
    }

    #[test]
    fn a_hidden_field_does_not_block_a_save() {
        let mut s = DecisionSettings::default();
        s.online.account_id = "not valid!".into();
        assert!(s.clone().validated().is_ok(), "typesafe ignores the Cloudflare account id");
        s.online.provider = "cloudflare".into();
        assert!(s.validated().is_err());
    }
}
