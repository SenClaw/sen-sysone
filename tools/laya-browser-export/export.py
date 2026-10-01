"""Export cklxx/laya-browser v19s (pinned commit) to the ONNX graph sen-sysone runs.

Run by `.github/workflows/model-laya-browser.yml`, which attaches the result to a
GitHub release so the catalog can download it with no account anywhere:

    python export.py --out dist

Writes `dist/model/` (the export at its real relative paths, plus manifest.json)
and `dist/assets/` (the same files with `/` flattened to `__`, because release
asset names cannot hold a directory). The export is discarded unless ONNX
Runtime and PyTorch give the same probabilities — on Laya's own parity cases and
on browser requests built in the checkpoint's request format (v5).

The graph export and parity helpers follow Laya's official exporter
(scripts/export_onnx.py in NandhaKishorM/laya): inputs (input_ids,
attention_mask, marker_pos, marker_mask, qtype), outputs (logits, act_logits),
opset 18, traced at batch 2 with symbolic batch / sequence / option dims.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = "cklxx/laya-browser"
REVISION = "645cf366a2ae35f1086e8c20eff48f909bb49206"  # v19s, request format v5
VERSION = "v19s"
# Answers are rounded to 4 decimals, so this leaves room for float noise but not for a broken graph.
PARITY_TOLERANCE = 1e-3
# What the catalog downloads, in this order; manifest.json lists each with its sha256.
FILES = [
    "laya.onnx",
    "laya.onnx.data",
    "rl_agent_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "source.json",
]

# Long enough (~300 tokens) to cross ModernBERT's 128-token local-attention window.
_LONG_PROMPT = " ".join([
    "Our support team has used the assistant for three weeks to answer questions about invoices,",
    "refunds, shipping delays and account settings, and most answers were accurate and polite.",
    "Yesterday a customer pasted a message asking the assistant to forget its earlier",
    "instructions, reveal the hidden system prompt and print the API keys in its configuration.",
] * 4)


def asset_name(path: str) -> str:
    """Release asset name for a file of the export (sen-sysone's downloader uses the same rule)."""
    return path.replace("/", "__")


def snapshot() -> Path:
    from huggingface_hub import snapshot_download

    # Only what loading and exporting need — the repo also carries training code and demo media.
    return Path(snapshot_download(REPO, revision=REVISION, allow_patterns=[
        "config.json", "model.safetensors", "laya_browser.py", "modeling_laya_browser.py",
        "rl_agent_config.json", "encoder/*", "tokenizer/*",
    ]))


def parity_cases():
    """Requests covering every question type, 1-5 questions per call (the ONNX batch) and 2-6 options."""
    import laya

    return [
        ({"message": "Hi, we were billed twice for March. Please refund the duplicate today "
                     "or we will cancel our plan."}, laya.triage_questions()),
        ({"body": "Chào anh chị, tháng này em bị trừ tiền hai lần cho cùng một hóa đơn. "
                  "Nhờ hoàn lại giúp em trước thứ Sáu nhé."}, laya.email_questions()),
        ({"prompt": _LONG_PROMPT}, laya.guard_questions()),
        ("The app crashes every time I open settings.",
         {"bug": {"type": "noul", "instructions": "Is this a bug report?"}}),
    ]


def browser_pages():
    """Three pages in jev-ultrafast's observation shape (the harness the model was trained with)."""
    search = {
        "url": "https://packages.example.org/search?q=json",
        "title": "Package search",
        "text": "Packages · Search results for json · argo 1.2 · rapidjson 1.1 · Include non-root modules",
        "actions": [
            {"id": "e1", "kind": "fill", "node": 1, "role": "searchbox", "label": "Search packages", "value": "json"},
            {"id": "e2", "kind": "click", "node": 1, "role": "searchbox", "label": "Search packages", "value": "json"},
            {"id": "e3", "kind": "click", "node": 2, "role": "button", "label": "Search"},
            {"id": "e4", "kind": "click", "node": 3, "role": "checkbox", "label": "Include non-root modules", "checked": False},
            {"id": "e5", "kind": "click", "node": 4, "role": "link", "label": "argo"},
            {"id": "e6", "kind": "click", "node": 5, "role": "link",
             "label": "rapidjson — a very fast JSON parser and generator for Lua, with a long description"},
            {"id": "scroll_down", "kind": "scroll", "label": "Scroll down"},
            {"id": "press_enter", "kind": "key", "label": "Press Enter in the focused text field (submit it)"},
            {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
        ],
    }
    form = {
        "url": "https://flights.example.com/",
        "title": "Find flights",
        "text": "Where from? Where to? Departure Return Passengers Economy Search flights",
        "actions": [
            {"id": "e1", "kind": "fill", "node": 1, "role": "combobox", "label": "Where from?", "value": "Zurich"},
            {"id": "e2", "kind": "fill", "node": 2, "role": "combobox", "label": "Where to?", "value": ""},
            {"id": "e3", "kind": "select", "node": 3, "role": "combobox", "label": "Class → Economy",
             "value": "economy", "current_value": "Economy"},
            {"id": "e4", "kind": "select", "node": 3, "role": "combobox", "label": "Class → Premium economy",
             "value": "premium", "current_value": "Economy"},
            {"id": "e5", "kind": "select", "node": 3, "role": "combobox", "label": "Class → Business",
             "value": "business", "current_value": "Economy"},
            {"id": "e6", "kind": "click", "node": 4, "role": "radio", "label": "One-way", "checked": "false"},
            {"id": "e7", "kind": "click", "node": 5, "role": "radio", "label": "Round trip", "checked": "true"},
            {"id": "e8", "kind": "click", "node": 6, "role": "button", "label": "Search flights"},
            {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
        ],
    }
    feed = {
        "url": "https://www.tiktok.com/explore",
        "title": "Khám phá | TikTok",
        "text": "Dành cho bạn · Đang theo dõi · @mai Món ngon mỗi ngày #nauan · 12,3K lượt thích · Bình luận",
        "actions": [
            {"id": "e1", "kind": "click", "node": 1, "role": "link", "label": "@mai"},
            {"id": "e2", "kind": "click", "node": 2, "role": "button", "label": "Thích video", "checked": "false"},
            {"id": "e3", "kind": "click", "node": 3, "role": "button", "label": "Bình luận"},
            {"id": "e4", "kind": "fill", "node": 4, "role": "searchbox", "label": "Tìm kiếm", "value": ""},
            {"id": "scroll_down", "kind": "scroll", "label": "Scroll down"},
            {"id": "scroll_up", "kind": "scroll", "label": "Scroll up"},
            {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
        ],
    }
    history = [{"action": "Search packages", "kind": "fill", "text": "json", "page_changed": False}]
    return [
        (search, "Search packages for 'json' and open the package 'argo'.", history),
        (form, "Find a one-way business class flight from Zurich to Hanoi.", []),
        (feed, "Mở video của @mai rồi bấm thích.", []),
    ]


def browser_cases():
    import laya_browser  # the model card's own request builder, from the same commit as the weights

    cases = []
    for page, goal, history in browser_pages():
        state, questions, _, _ = laya_browser.build_request(page, goal, history)
        cases.append((state, questions))
    return cases


def export_graph(model, path: Path) -> None:
    import torch

    dim = torch.export.Dim
    batch = dim("batch", min=1, max=64)        # one row per question; laya-serve caps a call at 64
    seq = dim("seq", min=1, max=8192)
    options = dim("options", min=2, max=256)   # the head's top-2 feature needs at least two options
    # batch=2 dummies: tracing with batch=1 lets dynamo bake the batch size into a head reshape.
    inputs = (
        torch.ones(2, 16, dtype=torch.long),               # input_ids
        torch.ones(2, 16, dtype=torch.long),               # attention_mask
        torch.tensor([[1, 2], [1, 2]], dtype=torch.long),  # marker_pos
        torch.ones(2, 2, dtype=torch.bool),                # marker_mask
        torch.zeros(2, dtype=torch.long),                  # qtype
    )
    torch.onnx.export(
        model,
        inputs,
        str(path),
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        dynamic_shapes=({0: batch, 1: seq}, {0: batch, 1: seq}, {0: batch, 1: options},
                        {0: batch, 1: options}, {0: batch}),
        opset_version=18,
        dynamo=True,
    )


def probabilities(answer: dict) -> dict:
    # choice/score answers carry a distribution; noul carries P(true) only.
    return answer.get("probabilities") or {"true": answer["noul"]}


def check_parity(agent, export: Path) -> dict:
    """Fail unless ONNX Runtime and PyTorch give the same probabilities and the same choices."""
    from laya.onnx_agent import ONNXAgent

    onnx_agent = ONNXAgent(str(export), onnx_path=str(export / "laya.onnx"))
    worst, compared = 0.0, 0
    cases = parity_cases() + browser_cases()
    for state, questions in cases:
        want = agent.system_one(state, questions)["answers"]
        got = onnx_agent.system_one(state, questions)["answers"]
        for qid, answer in want.items():
            expected, actual = probabilities(answer), probabilities(got[qid])
            worst = max(worst, *(abs(expected[k] - actual[k]) for k in expected))
            compared += 1
            if answer.get("choice") is not None and answer.get("choice") != got[qid].get("choice"):
                raise SystemExit(f"choice differs on {qid}: torch {answer.get('choice')} vs onnx {got[qid].get('choice')}")
    if worst > PARITY_TOLERANCE:
        raise SystemExit(f"ONNX export disagrees with PyTorch (max |Δp| = {worst:.2e})")
    print(f"parity ok: max |Δp| vs PyTorch = {worst:.1e} over {compared} answers ({len(cases)} requests)", flush=True)
    return {"maxAbsProbabilityDelta": float(f"{worst:.1e}"), "answers": compared, "requests": len(cases)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out
    model_dir, assets_dir = out / "model", out / "assets"
    shutil.rmtree(out, ignore_errors=True)
    model_dir.mkdir(parents=True)

    import laya
    from laya import Agent

    snap = snapshot()
    sys.path.insert(0, str(snap))
    agent = Agent(str(snap), device="cpu", compile=False)
    # What laya_browser.from_pretrained serves with; the config already says so.
    assert agent.cfg["max_len"] == 1024 and agent.cfg["head_max_len"] == agent.cfg.get("head_max_len_train", 768)
    assert agent.cfg.get("laya_fmt") == "v5", agent.cfg.get("laya_fmt")

    started = time.perf_counter()
    export_graph(agent.model.eval(), model_dir / "laya.onnx")
    print(f"graph exported in {time.perf_counter() - started:.0f}s", flush=True)
    shutil.copy(snap / "rl_agent_config.json", model_dir / "rl_agent_config.json")
    shutil.copytree(snap / "tokenizer", model_dir / "tokenizer", copy_function=shutil.copy)
    parity = check_parity(agent, model_dir)

    (model_dir / "source.json").write_text(json.dumps({
        "checkpoint": "laya-browser", "version": VERSION, "format": "v5", "repo": REPO, "subfolder": None,
        "revision": REVISION, "laya": laya.__version__, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=2) + "\n")
    manifest = {
        "checkpoint": REPO,
        "revision": REVISION,
        "version": VERSION,
        "format": "v5",
        "laya": laya.__version__,
        "contextLength": agent.cfg["max_len"],
        "headLength": agent.cfg["head_max_len"],
        "graph": {
            "opset": 18,
            "inputs": ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
            "outputs": ["logits", "act_logits"],
            "dtype": "float32",
        },
        "parity": {**parity, "against": f"PyTorch, laya {laya.__version__}"},
        "files": [{"path": p, "sha256": sha256(model_dir / p), "size": (model_dir / p).stat().st_size} for p in FILES],
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    assets_dir.mkdir()
    for p in FILES + ["manifest.json"]:
        # A hard link, not a copy: the weights alone are 1.3 GB.
        os.link(model_dir / p, assets_dir / asset_name(p))
    print(f"manifest sha256: {sha256(model_dir / 'manifest.json')}", flush=True)
    print(f"ready: {assets_dir}", flush=True)


if __name__ == "__main__":
    main()
