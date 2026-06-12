use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    List(Vec<MetaValue>),
    Map(MetaMap),
}

pub type MetaMap = BTreeMap<String, MetaValue>;
