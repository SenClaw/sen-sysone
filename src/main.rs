//! `sen-sysone` — the SenClaw decision runtime: typed `choice`/`score`/`noul`
//! answers from a local Laya checkpoint (ONNX Runtime) or a hosted
//! Jev-compatible backend. Launched by the SenClaw daemon as a child process
//! and driven over loopback HTTP (see `senclaw/docs/runtime-protocol.md`);
//! also runnable standalone for development.

mod decision;
mod http;
mod settings_store;

use std::sync::Arc;

use sen_runtime_sdk::env::LaunchEnv;
use sen_runtime_sdk::manifest::{Capability, RunMode};
use sen_runtime_sdk::server::{serve, Readiness, ServeArgs, ServeOptions};

fn usage() -> ! {
    eprintln!(
        "usage: sen-sysone serve [--host HOST] [--port PORT]\n\n\
         Serves the SenClaw decision API (typed choice/score/noul answers) on\n\
         loopback HTTP. Standalone: with no SENCLAW_RUNTIME_TOKEN set there is\n\
         no auth, and with no SENCLAW_PARENT_PID there is no parent watchdog."
    );
    std::process::exit(1);
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    sen_runtime_sdk::server::init_tracing();
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let cmd = args.remove(0);
    match cmd.as_str() {
        "serve" => run_serve(args).await,
        "-h" | "--help" => usage(),
        other => {
            eprintln!("unknown subcommand `{other}`");
            usage();
        }
    }
}

async fn run_serve(args: Vec<String>) -> anyhow::Result<()> {
    let parsed = ServeArgs::parse(args).map_err(|e| anyhow::anyhow!(e))?;
    let env = LaunchEnv::from_env(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
    let state = Arc::new(http::AppState { env: env.clone() });
    let routes = http::router(state);
    let opts = ServeOptions {
        env,
        mode: RunMode::Service,
        capabilities: vec![Capability::Decision],
        // A service answers /health before any weights load — Laya loads on
        // the first request that needs it, never at boot.
        readiness: Readiness::ready(),
        info_detail: Some(Arc::new(|| {
            serde_json::json!({
                "compiled": decision::laya::runtime::COMPILED,
                "loaded": decision::laya::runtime::loaded_ids(),
            })
        })),
        args: parsed,
    };
    serve(routes, opts).await
}
