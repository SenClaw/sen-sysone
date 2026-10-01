# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`sen-sysone` is the SenClaw decision runtime — Laya (the open, Jev-compatible
"System One" model) on ONNX Runtime, ported unchanged from the daemon's
`src/decision/{laya,online.rs,types.rs,json.rs}` + the model/ask/settings/
online-test handlers of `src/gateway/ui_server/decision.rs`. It is a
standalone binary the SenClaw daemon installs, launches as a child process,
and talks to over loopback HTTP: see
`../senclaw/docs/runtime-protocol.md` §4.3 for the exact contract (routes,
bodies, status codes, masking) this repo must keep serving **verbatim**.

The tool-call gate and the pre-skill router (`src/decision/gate`,
`src/decision/skill_route.rs` in the old daemon) are **not** here — they
need the daemon's skills registry and permission bridge, so they stayed in
the daemon's control plane and keep their own settings there. This repo's
`DecisionSettings` covers only `backend` / `local` / `online`; the daemon
merges its own `gate`/`skills` into `GET /api/decision/settings` so existing
clients render unchanged, and forwards only `backend`/`local`/`online` on
`PUT`.

## Rules for Claude (carried from the daemon, engine-scoped)

- **Never parse a decision request into `serde_json::Value`.** Its
  `preserve_order` feature is off, so `Map` sorts keys — and a `choice`'s
  option order is where its markers sit, so sorting changes every
  probability (and an object state's key order changes the text the encoder
  reads). Requests go through `decision::json::Json`, answers out through
  `OrderedMap`.
- **Parity is the contract, and the fixture is its proof.**
  `src/decision/laya/testdata/parity.json` holds Laya 0.3.20's token rows and
  answers; the ignored `parity_tests` must match token ids exactly and
  probabilities within 1e-3. Anything touching `sequence.rs` or the answer
  math runs it: `SENCLAW_LAYA_PARITY_ROOT=<dir of model folders> cargo test
  --features decision-laya parity -- --ignored --nocapture`. Verified in this
  port against the real `~/.senclaw/local-models/laya/multilingual`
  checkpoint (read-only) — all 6 fixture cases matched.
- **Read the graph, never assume it.** Exports differ: dynamic batch vs fixed
  at 1 (`ti3x-m` — one run per question), `act_logits` (softmax here) vs
  `act_probs`. `LayaEngine::load` detects both, and refuses a graph missing a
  required input.
- **There is no official ONNX release.** Catalog entries are community
  exports pinned to a full commit sha (a test enforces it); LFS sha256 comes
  from the Hub's tree listing, the rest from a `manifest.json` when present.
  A directory is installed only once `senclaw-laya.json` is written —
  always last.
- **`laya-browser` is SenClaw's own export, hosted as a GitHub release of
  this repo** (`Host::GithubRelease`), so nobody needs an account to publish
  or download it. Tag `model-laya-browser-<version>` →
  `.github/workflows/model-laya-browser.yml` runs
  `tools/laya-browser-export/export.py` (pinned checkpoint commit, parity
  against PyTorch incl. format-v5 browser requests) and attaches the files,
  `/` flattened to `__`. A tag can move, so the pin is the release
  manifest's **sha256** in the catalog entry (the test requires 64 hex); copy
  it from the release notes after the workflow finishes. A new checkpoint
  means a new tag, never re-running an old one.
- **The English checkpoint must not see Vietnamese.** It reads it
  confidently wrong. Routing (`plan_local` in
  `src/decision/laya/runtime.rs`) sends text with non-ASCII *letters* to a
  multilingual checkpoint — past an English **default**, and by **refusing**
  (409, `RuntimeError::Refused`) rather than handing it to English when the
  multilingual one is installed but not loaded and on-demand loading is off.
  Only a model the request names explicitly always answers.
- **A load runs in a detached task, once per model.** `LOADING` holds a
  `Shared` future every waiter awaits; the task fills `LOADED` *then* leaves
  `LOADING`. Delete holds a `DeleteGuard` across unload and removal so no
  hot-load slips in between. **Loads of different models take turns** on
  `ONE_LOAD_AT_A_TIME`, held inside that task: ONNX Runtime finds external
  weights through libc `dirname()`, which on macOS returns one static buffer
  for every thread, so two sessions created at once can swap directories and
  one load fails as "Encountered unknown exception in Initialize()". Do not
  drop it to parallelise loads; the ignored `same_moment` test
  (`SENCLAW_LAYA_LOAD_ROOT=<dir of model folders>`) is its proof.
- **Weights load on request, never at boot** — the `/load` route, or a
  request when *load on demand* is on. Unload returns the memory. The idle
  sweeper counts from the last request's *start* and re-checks under the
  write lock; a request already running keeps its own reference to the
  engine.
- **A decision's API key never leaves this runtime.** `GET
  /api/decision/settings` masks it (`hasApiKey`); a save with an empty key
  keeps the stored one **only for the same provider (and custom URL)**
  (`OnlineSettings::same_key_scope`) and only `clearApiKey: true` removes it.
- **Cloudflare's REST `/ai/run` nests the request under `input`**
  (`{"model":"typesafe/jev","input":{state,questions}}`); TypeSafe and custom
  endpoints take it flat. `src/decision/online.rs`'s tests pin both shapes.
- **Anything that returns online answers must serialize `AskResponse` /
  `Answers` directly, never through `json!`** — sorting the keys there is the
  same trap as above.
- **`POST /v1/systemone` and `POST /api/decision/ask` are the same handler.**
  `AskRequest.backend` is `#[serde(default)]`, so a Jev-shaped body with no
  `backend` field deserializes identically either way — do not fork this into
  two implementations.
- No plan ids, phase numbers, or finding codes in code, comments, or test
  names — explain the invariant or behaviour directly.

## Docs

[`docs/laya-decisions.md`](docs/laya-decisions.md) — full guide moved here
from the old monorepo (catalog, settings, parity testing).
