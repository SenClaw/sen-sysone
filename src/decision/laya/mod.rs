//! Laya — Convai Innovations' open, Jev-compatible System One model — run on
//! ONNX Runtime inside this runtime process.
//!
//! | module | what |
//! |---|---|
//! | [`sequence`] | token-row construction + answer math, ported from `laya.common` |
//! | [`pyjson`] | Python `json.dumps` so structured state tokenizes as trained |
//! | [`layout`] | which files an export has, its config, special tokens, language |
//! | [`store`] | the catalog, the on-disk model directories, the status list |
//! | [`download`] | background downloads (pinned revision, sha256, resume) and imports |
//! | [`runtime`] | loaded-model registry, routing, the one `ask` entry point |
//! | `engine` | the ONNX Runtime session (only with the `decision-laya` feature) |
//!
//! Ported unchanged from the SenClaw daemon's `src/decision/laya`, where it ran
//! behind the `decision-laya` feature on the same footing as VieNeu TTS (ONNX
//! on the CPU, no MLX). It moved into this standalone `sen-sysone` runtime so
//! the daemon links no inference code at all; the module boundary did not
//! change, only the process it runs in.

pub mod download;
#[cfg(feature = "decision-laya")]
mod engine;
pub mod layout;
pub mod pyjson;
pub mod runtime;
pub mod sequence;
pub mod store;

#[cfg(all(test, feature = "decision-laya"))]
mod parity_tests;
