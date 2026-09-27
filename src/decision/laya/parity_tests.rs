//! Parity with `laya.onnx_agent.ONNXAgent` (laya 0.3.20).
//!
//! `testdata/parity.json` holds, per checkpoint and case, the token rows and
//! the answers the Python agent produced over Laya-jev's exports (which that
//! project checked against PyTorch at export time). The cases cover every
//! question type, list/dict/structured criteria, noul labels, a Vietnamese
//! state, an array state long enough to be truncated from the left, and a
//! 12-option choice that trips both the head budget and the clamped
//! `choice:11+` temperature.
//!
//! Ignored by default because it needs the exported checkpoints on disk:
//!
//! ```text
//! SENCLAW_LAYA_PARITY_ROOT=~/Projects/Laya-jev/models \
//!   cargo test --features decision-laya parity -- --ignored --nocapture
//! ```

use std::path::Path;

use serde_json::Value;

use super::engine::LayaEngine;
use super::layout::ModelLayout;
use crate::decision::json::Json;
use crate::decision::types::parse_questions;

const P_TOL: f64 = 1e-3;

fn close(a: &Value, b: &Value, tol: f64, what: &str) {
    let (x, y) = (a.as_f64().unwrap(), b.as_f64().unwrap());
    assert!((x - y).abs() <= tol, "{what}: rust {x} vs python {y}");
}

#[test]
#[ignore = "needs exported Laya checkpoints: set SENCLAW_LAYA_PARITY_ROOT"]
fn rust_matches_python_on_every_fixture_case() {
    let root = std::env::var("SENCLAW_LAYA_PARITY_ROOT").expect("set SENCLAW_LAYA_PARITY_ROOT");
    // Ordered: option and key order is part of what the model reads.
    let fixture: Json = serde_json::from_str(include_str!("testdata/parity.json")).unwrap();
    let mut checked = 0;
    for (model, spec) in fixture.get("models").unwrap().as_object().unwrap() {
        let dir = Path::new(&root).join(model);
        if !dir.exists() {
            eprintln!("skipping {model}: {} not found", dir.display());
            continue;
        }
        let engine = LayaEngine::load(&ModelLayout::detect(&dir).unwrap(), super::runtime::resolve_threads(None)).unwrap();
        eprintln!("{model}: loaded in {} ms ({} batch)", engine.load_ms, if engine.fixed_batch { "fixed" } else { "dynamic" });
        let Json::Array(cases) = spec.get("cases").unwrap() else { panic!("cases") };
        for case in cases {
            let name = case.get("name").and_then(Json::as_str).unwrap();
            let state = case.get("state").unwrap();
            let questions = parse_questions(case.get("questions").unwrap()).unwrap();
            // Lookups from here on: order no longer matters.
            let plain: Value = serde_json::to_value(case).unwrap();

            for q in &questions {
                let seq = engine.sequence_for(state, q).unwrap();
                let want = &plain["sequences"][&q.id];
                let ids: Vec<u32> = serde_json::from_value(want["ids"].clone()).unwrap();
                let markers: Vec<usize> = serde_json::from_value(want["markers"].clone()).unwrap();
                assert_eq!(seq.ids, ids, "{model}/{name}/{}: token ids differ", q.id);
                assert_eq!(seq.markers, markers, "{model}/{name}/{}: markers differ", q.id);
            }

            let started = std::time::Instant::now();
            let out = engine.infer(state, &questions).unwrap();
            let ms = started.elapsed().as_millis();
            let expected_order: Vec<&str> = case
                .get("expected")
                .and_then(|e| e.get("answers"))
                .and_then(Json::as_object)
                .unwrap()
                .iter()
                .map(|(k, _)| k.as_str())
                .collect();
            assert_eq!(out.answers.keys().collect::<Vec<_>>(), expected_order, "{model}/{name}: answer order");
            let got_all: Value = serde_json::to_value(&out.answers).unwrap();
            let expected = plain["expected"]["answers"].as_object().unwrap();
            for (qid, want) in expected {
                let got = &got_all[qid];
                let at = format!("{model}/{name}/{qid}");
                assert_eq!(got["type"], want["type"], "{at}");
                match want["type"].as_str().unwrap() {
                    "choice" => {
                        assert_eq!(got["choice"], want["choice"], "{at}: choice");
                        for (label, p) in want["probabilities"].as_object().unwrap() {
                            close(&got["probabilities"][label], p, P_TOL, &format!("{at}: p[{label}]"));
                        }
                        close(&got["confidence"], &want["confidence"], 2e-3, &format!("{at}: confidence"));
                    }
                    "score" => {
                        close(&got["score"], &want["score"], 5e-3, &format!("{at}: score"));
                        for (level, p) in want["probabilities"].as_object().unwrap() {
                            close(&got["probabilities"][level], p, P_TOL, &format!("{at}: p[{level}]"));
                        }
                    }
                    "noul" => close(&got["noul"], &want["noul"], P_TOL, &format!("{at}: noul")),
                    other => panic!("{at}: unexpected type {other}"),
                }
                close(
                    &got["action"]["act_probability"],
                    &want["action"]["act_probability"],
                    P_TOL,
                    &format!("{at}: act_probability"),
                );
                checked += 1;
            }
            eprintln!(
                "{model}/{name}: {} answers match ({} ms here, {} ms in Python)",
                expected.len(),
                ms,
                plain["python_ms"]
            );
        }
    }
    assert!(checked > 0, "no checkpoint was found under {root}");
}
