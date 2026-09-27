# Typed decisions with Laya (ONNX)

Laya is Convai Innovations' open "System One" model — the Jev-compatible kind
that does not generate text. It reads one **state** and answers typed
questions with probabilities:

| type | answer |
|---|---|
| `choice` | one label out of up to 255, with a distribution over all of them |
| `score` | a position on an ordered rubric of 2–10 levels (expected level + distribution) |
| `noul` | P(the statement holds) |

This runtime (`sen-sysone`) answers it on ONNX Runtime (CPU), behind the
`decision-laya` build feature — or sends the same request **online**, to
TypeSafe's Jev or any other `/v1/systemone` endpoint. Nothing is loaded at
boot: a model loads when someone presses **Load** or, with *Load on demand*
on, when a request needs it, and is unloaded again after a configurable idle
time.

The SenClaw daemon launches this process, proxies the routes below at the
same paths clients always used, and keeps two things this runtime does not
own: the **tool-call gate** and the **pre-skill router** — both control-plane
features that judge a Laya/Jev *answer* over HTTP but need the daemon's own
skills registry and permission bridge to act on it. See the daemon's own docs
for those; this guide covers only what `sen-sysone` itself does.

Research and design behind the original integration (daemon-era, kept for
history): `research-260924-1225-jev-typesafe-system-one.md`,
`research-260925-0111-jev-local-self-hosting.md`,
`research-260925-0215-jev-senclaw-integration-design.md` (SenClaw daemon repo,
`plans/reports/`).

## Using it

**Settings → Decision (Laya)**, in the web UI and the desktop app:

- **How it runs** — the settings below, saved to this runtime's
  `<SENCLAW_RUNTIME_DATA_DIR>/settings.json` (seeded once from the daemon's
  old `config.json["decisionConfig"]`) and read per request (no restart).
- **Models** — the catalog (download), plus anything imported or downloaded
  from another repo. Per row: size, state (not installed / downloading x% /
  installed / loading / loaded with load time, threads, batch mode, ask count,
  *on demand* when a request loaded it, and when it will be unloaded if unused)
  and the actions **Download · Cancel · Load · Unload · Delete**. The default
  model is marked.
- **Try it** — answer with the configured backend or pick one, pick a model
  (loaded, or any installed one when *Load on demand* is on), a sample, edit the
  state and the questions JSON, press **Ask**. Each answer shows its
  distribution as bars, `confidence`, `answer_confidence` and `act`; an online
  answer in a shape none of the three types match is shown as it came.
- **Import from a folder** — an export already on disk (for instance one made
  with the `laya` Python package). On the same APFS volume the copy is a clone:
  instant, and no extra disk.
- **Download from another Hugging Face repo** — any export with the Laya layout.

### How it runs

| setting | default | what it does |
|---|---|---|
| backend | `local` | `local` = a Laya checkpoint here; `online` = the provider below. A request's own `backend` wins. |
| default model | none | answers requests that name no model; none = pick by language |
| threads | cores, capped at 8 | ONNX Runtime intra-op threads. A loaded model keeps the count it was loaded with. |
| load on demand | on | a request needing a model that is installed but not in RAM loads it instead of failing (hot-load) |
| unload when unused for | 15 min | a sweeper (every 30 s) unloads a model no request has used for this long; 0 = never |
| online provider | `typesafe` | `typesafe` (api.typesafe.ai, model pinned `jev-1.13.0`), `cloudflare` (Workers AI, needs the account id), `custom` (any `/v1/systemone` URL) |
| online timeout | 15 s | 5–25 s: the desktop client gives up after 30 s, and the daemon must answer first |
| API key | — | stored in this runtime's `settings.json`; `GET` never returns it, only `hasApiKey`; an empty key in a save keeps it — **only for the same provider, and for `custom` the same URL** — and `clearApiKey: true` removes it |

**Which local model answers** (`plan_local`), in order:

1. the request's `model` — always exactly that one;
2. the default model, **when it can read the text**: a multilingual default
   reads anything, an English one (`english`, `typed-decisions`) only text whose
   letters are all ASCII;
3. by language. Text with non-ASCII letters (punctuation and emoji do not
   count) goes to a multilingual checkpoint — the loaded one, or an installed
   one loaded on demand. With *Load on demand* **off** and the multilingual
   checkpoint installed but not loaded, the request is **refused** (409) with
   what to load, rather than handed to English. Only when no multilingual
   checkpoint is installed at all does English answer, and the reason says so.
   ASCII text uses whatever is loaded — a multilingual checkpoint reads English
   fine, so it never costs a second model.

The response's `routing.reason` says which rule picked the model and whether it
was loaded on demand. A load runs once per model however many requests wait on
it, and a client that disconnects mid-load does not strand it.

Language matters because the **English** checkpoint reads Vietnamese badly
*and* confidently (a Vietnamese billing email scored 0.93 "spam", 0.72
"phishing"; the multilingual one scored 0.045 and 0.002).

**Online** sends the state and the questions to the provider — as the request
body for TypeSafe and custom endpoints, nested as `{"model": "typesafe/jev",
"input": {state, questions}}` for Cloudflare's REST `/ai/run`. The answers are
passed through verbatim and in order (`engine: "online:<provider>"`), with the
provider's token usage; an answer set missing a question is a 502, not a
success. The error text is ours — a 401 names the key, a 429 the rate limit, a
timeout the budget. **Test connection** sends one tiny yes/no question with the
form as edited, saved or not (the saved key goes along only under the rule
above).

## REST

Served by this runtime, reached through the daemon's proxy at the same paths
(`/api/decision/*`, behind the daemon's own auth gate) or directly on this
runtime's own port for development (bearer token from `SENCLAW_RUNTIME_TOKEN`,
none when started by hand).

| | |
|---|---|
| `GET /api/decision/models` | `{ compiled, root, models: [...], backend, local }` |
| `POST /api/decision/models/:id/download` | download a catalog model (background job) |
| `POST /api/decision/models/custom` | `{ id, repo, revision? }` — branch/tag pinned to a commit before listing |
| `POST /api/decision/models/import` | `{ path, id? }` — absolute folder; `~` is **this runtime process's** home (`SENCLAW_HOME`) |
| `POST /api/decision/models/:id/cancel` | stop a job; a download resumes later from its `.part` |
| `POST /api/decision/models/:id/load` | answers once the weights are in RAM |
| `POST /api/decision/models/:id/unload` | frees them (measured: RSS 2.4 GB → 0.7 GB after both) |
| `DELETE /api/decision/models/:id` | unload, then delete from disk |
| `POST /api/decision/ask` | `{ backend?, model?, state, questions }` → answers, in question order |
| `GET /api/decision/settings` | `{ compiled, settings, providers, defaults }` — the key masked |
| `PUT /api/decision/settings` | the whole form (`{ backend, local, online, clearApiKey? }`); refuses a default model that is not installed |
| `POST /api/decision/online/test` | same body as `PUT`, unsaved; one tiny request to the online backend |

```bash
curl -s -X POST http://127.0.0.1:18788/api/decision/ask -H 'Content-Type: application/json' -d '{
  "state": {"body": "Chào anh chị, tháng này em bị trừ tiền hai lần cho cùng một hóa đơn."},
  "questions": {
    "category": {"type": "choice", "instructions": "Which team should handle the email in `body`?",
                 "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "none of the above"}},
    "is_phishing": {"type": "noul", "instructions": "Is this email a phishing attempt?"}
  }}'
```

Errors are `{"error": "..."}`: 409 when no model / not this model is loaded
and loading on demand is off, when non-English text needs a multilingual model
that is not loaded, when nothing is installed, when a model is still loading
(delete), or when the online backend lacks a key, account id or URL; 404 for a
model that is not installed; 422 for a question laya itself would refuse (same
wording as laya); 502 when the online backend fails, refuses or leaves a
question unanswered; 501 on a build without `decision-laya` (local only —
online works on any build). Deleting the default model resets the default to
*pick by language* (`defaultCleared: true`).

## What the daemon builds on top (not in this repo)

Two SenClaw daemon features *call* this runtime's `/v1/systemone` / `ask` path
but are not part of it — their code, settings and REST routes live in the
daemon, not here, because they need the daemon's skills registry and
permission bridge:

- **Tool-call gate** (`/api/decision/gate*`) — auto-approves a safe shell
  command before a permission prompt, judged by asking this runtime one
  `choice`/`noul` question about what the command does. A static danger list
  still runs first in the daemon and is never sent here.
- **Pre-skill router** (`/api/decision/skills*`) — before a chat turn, asks
  this runtime to co-sign (never override) a keyword-based skill match.

See the daemon's own docs for how those work; this repo only answers the
typed question either one sends it, indistinguishable from any other caller.

## Where models come from

There is **no official ONNX release** — `convaiinnovations/laya` publishes
PyTorch weights only. The catalog therefore names community exports whose
makers document parity with PyTorch, each **pinned to a commit**:

| id | repo | export |
|---|---|---|
| `multilingual` | `ti3x-m/laya-multilingual-onnx` | batch fixed at 1, `act_probs`, sha256 for every file in `manifest.json` |
| `english` | `receptron/laya-onnx` | dynamic batch, `act_probs` |
| `typed-decisions` | `ti3x-m/laya-typed-decisions-onnx` | batch fixed at 1, `act_probs` |

Every LFS file is checked against the sha256 in the Hub's tree listing (and
the manifest when there is one). Files land as `<name>.part` and are renamed
only once the bytes check out; `senclaw-laya.json` is written **last**, so a
directory without it is never treated as installed.

Checked end to end: the catalog `multilingual` answers **identically**
(|Δp| = 0 at 4 decimals) to a PyTorch-verified export on the parity cases.

Files live in `<local-models>/laya/<id>/` (`~/.senclaw/local-models/laya/`),
keeping each export's own relative paths — the external-data file name is
recorded inside the graph.

## How it matches laya

`src/decision/laya/` is a port of `laya.onnx_agent.ONNXAgent` (laya 0.3.20):
the row `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 … [SEP] <state> [SEP]`,
the 48-token option cap and `head_max_len` budget, the temperature buckets
(clamped to [0.5, 5], which neutralises the shipped `choice:11+` = 0.1006), the
entropy `confidence`. `answer_confidence` = max(p) is added — it is the
quantity laya's temperature scaling fits.

`src/decision/laya/testdata/parity.json` holds the Python agent's token rows
and answers for 12 cases × 2 checkpoints. With the exports on disk:

```bash
SENCLAW_LAYA_PARITY_ROOT=~/Projects/Laya-jev/models \
  cargo test --features decision-laya parity -- --ignored --nocapture
```

Token ids match exactly; probabilities within 1e-3. Verified in this port
against the real `multilingual` checkpoint from `~/.senclaw/local-models/laya`
(read-only) — all 6 fixture cases matched.

## Not done yet

The online path was verified against a `custom` endpoint (this runtime's own
`/api/decision/ask`), not against a real TypeSafe or Cloudflare account; the
Cloudflare body shape is checked against its documentation by a unit test.
Settings are re-read from `<data_dir>/settings.json` on every ask, and that
file is written atomically (temp file + rename) by every settings save.
