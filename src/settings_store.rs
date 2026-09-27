//! `<data_dir>/settings.json` — this runtime's own copy of `decisionConfig`,
//! seeded once from the daemon's `config.json` via
//! [`sen_runtime_sdk::legacy::load_or_import`]. Read per request (a save must
//! apply to the next ask without a restart) and written back whole: unlike the
//! old daemon file this one is private to `sen-sysone`, so there are no other
//! keys to preserve.

use std::path::Path;

use sen_runtime_sdk::env::LaunchEnv;

use crate::decision::settings::DecisionSettings;

const LEGACY_KEY: &str = "decisionConfig";

/// Load settings, importing from the daemon's old `config.json` the first
/// time this runtime ever starts against a given data dir.
pub fn load(env: &LaunchEnv) -> DecisionSettings {
    match sen_runtime_sdk::legacy::load_or_import(&env.data_dir, &env.config_path, LEGACY_KEY) {
        Some(v) => serde_json::from_value(v).unwrap_or_default(),
        None => DecisionSettings::default(),
    }
}

/// Persist settings, replacing the file atomically (write to a temp path,
/// then rename) so a crash mid-save never leaves a half-written file.
pub fn save(env: &LaunchEnv, settings: &DecisionSettings) -> std::io::Result<()> {
    write_atomic(&env.data_dir, settings)
}

fn write_atomic(data_dir: &Path, settings: &DecisionSettings) -> std::io::Result<()> {
    std::fs::create_dir_all(data_dir)?;
    let body = serde_json::to_string_pretty(settings).map_err(std::io::Error::other)?;
    let tmp = data_dir.join("settings.json.tmp");
    std::fs::write(&tmp, body)?;
    std::fs::rename(tmp, data_dir.join("settings.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_at(dir: &Path) -> LaunchEnv {
        LaunchEnv::from_lookup("sen-sysone", "0.0.0-test", |k| match k {
            "SENCLAW_RUNTIME_DATA_DIR" => Some(dir.join("data").to_string_lossy().into_owned()),
            "SENCLAW_CONFIG_PATH" => Some(dir.join("config.json").to_string_lossy().into_owned()),
            "SENCLAW_HOME" => Some(dir.to_string_lossy().into_owned()),
            _ => None,
        })
    }

    #[test]
    fn imports_the_legacy_key_once_then_reads_its_own_file() {
        let tmp = tempfile::tempdir().unwrap();
        let env = env_at(tmp.path());
        std::fs::write(
            &env.config_path,
            r#"{"decisionConfig": {"backend": "online", "online": {"provider": "cloudflare"}}}"#,
        )
        .unwrap();

        let first = load(&env);
        assert_eq!(first.backend, crate::decision::settings::Backend::Online);
        assert_eq!(first.online.provider, "cloudflare");

        // A save changes only this runtime's file — the legacy file changing
        // afterwards must not leak back in.
        let mut updated = first.clone();
        updated.online.provider = "typesafe".into();
        save(&env, &updated).unwrap();
        std::fs::write(&env.config_path, r#"{"decisionConfig": {"backend": "local"}}"#).unwrap();
        let second = load(&env);
        assert_eq!(second.online.provider, "typesafe");
        assert_eq!(second.backend, crate::decision::settings::Backend::Online);
    }

    #[test]
    fn no_legacy_file_means_defaults() {
        let tmp = tempfile::tempdir().unwrap();
        let env = env_at(tmp.path());
        let s = load(&env);
        assert_eq!(s, DecisionSettings::default());
    }
}
