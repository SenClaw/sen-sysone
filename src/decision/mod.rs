//! Typed decisions — questions answered with probabilities, not generated text.
//!
//! A caller hands over one `state` (a string, or JSON the model reads as
//! text) and a map of typed questions: `choice` (pick one label), `score`
//! (place the state on an ordered rubric) or `noul` (how likely a statement
//! holds). The answer to each is a distribution, never prose, so it can be
//! branched on directly. This is the `/v1/systemone` shape TypeSafe defined
//! for Jev and the open Laya checkpoints reuse; keeping it means a hosted
//! backend can slot in later without changing a caller.
//!
//! Two engines answer it: a **local Laya checkpoint run by ONNX Runtime**
//! ([`laya`], behind the `decision-laya` build feature — without it everything
//! still compiles, the model store still works, and load / ask answer "not in
//! this build"), or a hosted `/v1/systemone` ([`online`]: TypeSafe Jev,
//! Cloudflare, any compatible URL). [`settings`] picks between them.
//!
//! This crate is the `sen-sysone` runtime: the SenClaw daemon owns none of
//! this any more, only the tool-call gate and the pre-skill router (control
//! plane, which read Laya's *answers* over HTTP but never its weights).
//! `settings` here covers `backend`/`local`/`online` only — the daemon keeps
//! its own gate/skill settings and merges them into what clients see.

pub mod json;
pub mod laya;
pub mod online;
pub mod settings;
pub mod types;
