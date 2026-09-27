//! Python's `json.dumps(value, ensure_ascii=False)`, byte for byte where it
//! matters.
//!
//! Laya never sees a JSON *value*: a structured state, instruction or rubric
//! reaches the encoder as the text Python's `json.dumps` produced when the
//! checkpoint was trained. `serde_json::to_string` writes `{"a":1}` where
//! Python writes `{"a": 1}`, and the extra spaces are extra tokens — enough to
//! move every probability by a little and to shift where truncation cuts.
//!
//! Known gap: Python prints a float exponent as `1e+21`, `serde_json` as
//! `1e21`. No realistic state carries one.

use crate::decision::json::Json;

/// `json.dumps(v, ensure_ascii=False)` — separators `", "` and `": "`, key order
/// as given, non-ASCII kept literal.
pub fn dumps(v: &Json) -> String {
    let mut out = String::new();
    write(v, &mut out);
    out
}

fn write(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Number(n) => out.push_str(&n.to_string()),
        Json::String(s) => write_str(s, out),
        Json::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write(item, out);
            }
            out.push(']');
        }
        Json::Object(entries) => {
            out.push('{');
            for (i, (k, item)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_str(k, out);
                out.push_str(": ");
                write(item, out);
            }
            out.push('}');
        }
    }
}

/// Python and `serde_json` escape the same set with `ensure_ascii=False`:
/// the quote, the backslash, and control characters (short forms for
/// `\b \f \n \r \t`, `\u00XX` for the rest).
fn write_str(s: &str, out: &mut String) {
    out.push_str(&serde_json::to_string(s).expect("a str always serializes"));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn js(text: &str) -> Json {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn matches_python_separators_and_keeps_order_and_unicode() {
        // python3 -c 'import json; print(json.dumps({"z": 1, "a": [1, 2.5, None, True], "vi": "Chào \"bạn\"\n", "o": {}}, ensure_ascii=False))'
        let v = js(r#"{"z": 1, "a": [1, 2.5, null, true], "vi": "Chào \"bạn\"\n", "o": {}}"#);
        assert_eq!(
            dumps(&v),
            r#"{"z": 1, "a": [1, 2.5, null, true], "vi": "Chào \"bạn\"\n", "o": {}}"#
        );
    }

    #[test]
    fn control_characters_use_python_escapes() {
        // python3 -c 'import json; print(json.dumps("a\tb\x01c", ensure_ascii=False))'
        assert_eq!(dumps(&Json::String("a\tb\u{1}c".into())), r#""a\tb\u0001c""#);
    }

    #[test]
    fn empty_containers_have_no_inner_space() {
        assert_eq!(dumps(&js("[[], {}]")), "[[], {}]");
    }
}
