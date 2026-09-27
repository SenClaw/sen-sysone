# sen-sysone

SenClaw's decision runtime: typed `choice` / `score` / `noul` answers —
probabilities, not generated text — from a local **Laya** checkpoint (ONNX
Runtime, CPU) or a hosted Jev-compatible backend (TypeSafe, Cloudflare
Workers AI, or any `/v1/systemone`-speaking endpoint). Launched by the
SenClaw daemon as a child process and driven over loopback HTTP — see
[`senclaw/docs/runtime-protocol.md`](../senclaw/docs/runtime-protocol.md)
§4.3 for the wire contract this repo implements.

Ported from the SenClaw daemon's `src/decision/{laya,online.rs,types.rs,
json.rs,settings.rs}` and the model/ask/settings/online-test handlers of
`src/gateway/ui_server/decision.rs`. The tool-call gate and the pre-skill
router (`src/decision/gate`, `src/decision/skill_route.rs`) stayed in the
daemon — they need the daemon's skills registry and permission bridge — and
keep their own settings there; this runtime's settings cover only
`backend` / `local` / `online`.

## Build & run

```bash
cargo build --release            # or: make build
cargo test                       # or: make test
make package                     # dist/sen-sysone-<version>-<platform>.tar.gz + .sha256
make install-local                # senclaw runtime install-local, or extract into ~/.senclaw/runtimes
make run-dev                     # standalone serve on :4961, no token, no watchdog
```

Standalone (no daemon) for development:

```bash
cargo run -- serve --host 127.0.0.1 --port 4961
```

With no `SENCLAW_RUNTIME_TOKEN` set there is no auth; with no
`SENCLAW_PARENT_PID` there is no parent watchdog — see
[`sen_runtime_sdk::env`](../senclaw/crates/sen-runtime-sdk/src/env.rs).

## Routes

Every route the daemon proxied at `/api/decision/*` before the split, served
**verbatim** (same paths, bodies, status codes, masking), plus the common
`/health` · `/runtime/info` · `/runtime/shutdown` from the SDK server
scaffold and Jev-compatible `POST /v1/systemone`:

```
GET    /api/decision/models
POST   /api/decision/models/custom
POST   /api/decision/models/import
POST   /api/decision/models/:id/download
POST   /api/decision/models/:id/cancel
POST   /api/decision/models/:id/load
POST   /api/decision/models/:id/unload
DELETE /api/decision/models/:id
POST   /api/decision/ask
GET    /api/decision/settings
PUT    /api/decision/settings
POST   /api/decision/online/test
POST   /v1/systemone
```

`gate`/`skills` are not here — the daemon still owns
`/api/decision/{gate,skills}*` and merges its own `gate`/`skills` into what
`GET /api/decision/settings` shows so existing clients render unchanged.

## Models on disk

Laya checkpoints stay under `<SENCLAW_LOCAL_MODELS_DIR>/laya/` (the shared
model root the daemon always used — nothing is re-downloaded after the
split). Settings persist at `<SENCLAW_RUNTIME_DATA_DIR>/settings.json`,
seeded once from the daemon's old `config.json["decisionConfig"]`.

## Feature flags

- `decision-laya` (default on) — the local ONNX engine (`ort` + `tokenizers`).
  Off, the binary still serves the full API; local ask/load answer "this
  build has no Laya engine" while the online backend keeps working.

## Docs

[`docs/laya-decisions.md`](docs/laya-decisions.md) — the full guide (catalog,
settings, parity testing) moved here from the old monorepo.
