//! A JSON value that keeps object keys in the order they were written.
//!
//! `serde_json::Value` sorts keys in this crate: its `preserve_order` feature
//! is on only for build scripts, not for the daemon. For typed decisions the
//! order is data, not presentation:
//!
//! - a `choice`'s options are scored at marker positions laid out in the
//!   caller's order — sorting them changes the encoder input and so every
//!   probability (and puts "other" wherever the alphabet says);
//! - an object state reaches the encoder as the text Python's `json.dumps`
//!   wrote, keys in insertion order.
//!
//! So requests are parsed into [`Json`], and answers go out through
//! [`OrderedMap`], both of which serialize in the order they hold.

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<Json>),
    /// Keys in document order. A repeated key keeps its first position and
    /// the last value, as Python's `json.loads` does.
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(entries) => Some(entries),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    pub fn is_array(&self) -> bool {
        matches!(self, Json::Array(_))
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "a boolean",
            Json::Number(_) => "a number",
            Json::String(_) => "a string",
            Json::Array(_) => "an array",
            Json::Object(_) => "an object",
        }
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Json, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Json;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_unit<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_none<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_some<D2: Deserializer<'de>>(self, d: D2) -> Result<Json, D2::Error> {
                Json::deserialize(d)
            }
            fn visit_bool<E>(self, b: bool) -> Result<Json, E> {
                Ok(Json::Bool(b))
            }
            fn visit_i64<E>(self, n: i64) -> Result<Json, E> {
                Ok(Json::Number(n.into()))
            }
            fn visit_u64<E>(self, n: u64) -> Result<Json, E> {
                Ok(Json::Number(n.into()))
            }
            fn visit_f64<E: de::Error>(self, n: f64) -> Result<Json, E> {
                serde_json::Number::from_f64(n)
                    .map(Json::Number)
                    .ok_or_else(|| E::custom("a JSON number must be finite"))
            }
            fn visit_str<E>(self, s: &str) -> Result<Json, E> {
                Ok(Json::String(s.to_string()))
            }
            fn visit_string<E>(self, s: String) -> Result<Json, E> {
                Ok(Json::String(s))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(Json::Array(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
                let mut entries: Vec<(String, Json)> = Vec::new();
                while let Some((k, v)) = map.next_entry::<String, Json>()? {
                    match entries.iter_mut().find(|(ek, _)| *ek == k) {
                        Some(slot) => slot.1 = v,
                        None => entries.push((k, v)),
                    }
                }
                Ok(Json::Object(entries))
            }
        }
        d.deserialize_any(V)
    }
}

impl Serialize for Json {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Json::Null => s.serialize_unit(),
            Json::Bool(b) => s.serialize_bool(*b),
            Json::Number(n) => n.serialize(s),
            Json::String(v) => s.serialize_str(v),
            Json::Array(items) => {
                let mut seq = s.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Json::Object(entries) => {
                let mut map = s.serialize_map(Some(entries.len()))?;
                for (k, v) in entries {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
        }
    }
}

/// A map that serializes in insertion order — answers, probabilities and
/// score legends go out in the order the caller asked.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderedMap<T>(pub Vec<(String, T)>);

impl<T> Default for OrderedMap<T> {
    fn default() -> Self {
        OrderedMap(Vec::new())
    }
}

impl<T> OrderedMap<T> {
    pub fn push(&mut self, key: impl Into<String>, value: T) {
        self.0.push((key.into(), value));
    }

    pub fn get(&self, key: &str) -> Option<&T> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(k, _)| k.as_str())
    }
}

impl<T: Serialize> Serialize for OrderedMap<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_keep_document_order_through_a_round_trip() {
        let text = r#"{"z":1,"a":{"y":true,"b":[1,2.5,null]},"m":"x"}"#;
        let v: Json = serde_json::from_str(text).unwrap();
        let keys: Vec<&str> = v.as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["z", "a", "m"]);
        assert_eq!(serde_json::to_string(&v).unwrap(), text);
    }

    #[test]
    fn a_repeated_key_keeps_its_first_position_and_last_value() {
        let v: Json = serde_json::from_str(r#"{"a":1,"b":2,"a":3}"#).unwrap();
        assert_eq!(serde_json::to_string(&v).unwrap(), r#"{"a":3,"b":2}"#);
    }

    #[test]
    fn ordered_maps_serialize_in_insertion_order() {
        let mut m = OrderedMap::default();
        m.push("zeta", 1);
        m.push("alpha", 2);
        assert_eq!(serde_json::to_string(&m).unwrap(), r#"{"zeta":1,"alpha":2}"#);
    }
}
