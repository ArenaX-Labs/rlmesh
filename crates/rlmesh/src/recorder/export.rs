//! Bundle filesystem writer (runs as owned, cancellation-aware host work).
use super::{State, os};
use crate::{Error, Metadata, Result, Value};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};
fn check() -> Result<()> {
    crate::warnings::check_cancelled()
}
fn mkdir(path: &Path) -> Result<()> {
    check()?;
    fs::create_dir_all(path).map_err(os)
}
fn copy(source: &Path, dest: &Path) -> Result<()> {
    check()?;
    let mut input = fs::File::open(source).map_err(os)?;
    check()?;
    let mut output = fs::File::create(dest).map_err(os)?;
    transfer(&mut input, &mut output)
}
fn transfer(input: &mut impl Read, output: &mut impl Write) -> Result<()> {
    let mut buf = [0u8; 64 * 1024];
    loop {
        check()?;
        let n = input.read(&mut buf).map_err(os)?;
        if n == 0 {
            return Ok(());
        }
        check()?;
        output.write_all(&buf[..n]).map_err(os)?;
    }
}

// Python's json.dumps accepts tuples and nonfinite floats, but refuses bytes
// and tensors. The wire metadata codec instead turns bytes/tensors into arrays
// and serde_json turns nonfinite floats into null: neither preserves this
// bundle contract. Quote strings with serde_json and write the JSON-native
// container/number kinds directly, including Python's NaN/Infinity tokens.
fn json(value: &Value, depth: usize) -> Result<String> {
    let quote = |text: &str| serde_json::to_string(text).expect("strings serialize");
    let container = |open: char, close: char, items: Vec<String>| {
        if items.is_empty() {
            return format!("{open}{close}");
        }
        let indent = "  ".repeat(depth + 1);
        format!(
            "{open}\n{indent}{}\n{}{close}",
            items.join(&format!(",\n{indent}")),
            "  ".repeat(depth)
        )
    };
    Ok(match value {
        Value::None => "null".into(),
        Value::Bool(value) => value.to_string(),
        Value::Int(value) => value.to_string(),
        Value::Float(value) if value.is_nan() => "NaN".into(),
        Value::Float(value) if *value == f64::INFINITY => "Infinity".into(),
        Value::Float(value) if *value == f64::NEG_INFINITY => "-Infinity".into(),
        Value::Float(value) => crate::env::factory::py_float(*value),
        Value::Text(value) => quote(value),
        Value::List(items) | Value::Tuple(items) => container(
            '[',
            ']',
            items
                .iter()
                .map(|item| json(item, depth + 1))
                .collect::<Result<_>>()?,
        ),
        Value::Dict(items) => container(
            '{',
            '}',
            items
                .iter()
                .map(|(key, item)| Ok(format!("{}: {}", quote(key), json(item, depth + 1)?)))
                .collect::<Result<_>>()?,
        ),
        Value::Bytes(_) => {
            return Err(Error::Type(
                "Object of type bytes is not JSON serializable".into(),
            ));
        }
        Value::Tensor(_) => {
            return Err(Error::Type(
                "Object of type Tensor is not JSON serializable".into(),
            ));
        }
    })
}
pub(super) fn write(
    path: &Path,
    payload: &Metadata,
    state: &State,
    archive: Option<bool>,
) -> Result<PathBuf> {
    if state.closed && !state.assets.is_empty() {
        return Err(Error::Runtime(
            "Recorder is closed (its staged media was removed); export before close()".into(),
        ));
    }
    check()?;
    let payload = json(&Value::Dict(payload.clone()), 0)?.into_bytes();
    let archive = archive.unwrap_or_else(|| {
        path.extension()
            .is_some_and(|s| s.eq_ignore_ascii_case("zip"))
    });
    if archive {
        mkdir(
            path.parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new(".")),
        )?;
        check()?;
        let file = fs::File::create(path).map_err(os)?;
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        check()?;
        zip.start_file("result.json", options).map_err(os)?;
        check()?;
        zip.write_all(&payload).map_err(os)?;
        for (source, rel) in &state.assets {
            check()?;
            if source.is_file() {
                let mut input = fs::File::open(source).map_err(os)?;
                check()?;
                zip.start_file(rel, options).map_err(os)?;
                transfer(&mut input, &mut zip)?;
            }
        }
        check()?;
        zip.finish().map_err(os)?;
    } else {
        check()?;
        if path.exists() && !path.is_dir() {
            return Err(os(format!(
                "{} exists and is not a directory; pass archive='zip' to overwrite a zip bundle, or export to a different path",
                path.display()
            )));
        }
        mkdir(path)?;
        let media = path.join("media");
        check()?;
        if media.is_dir() {
            if !path.join("result.json").is_file() {
                return Err(os(format!(
                    "refusing to overwrite {}: {} is not an rlmesh bundle (no result.json); export to a different path",
                    media.display(),
                    path.display()
                )));
            }
            check()?;
            fs::remove_dir_all(media).map_err(os)?;
        }
        check()?;
        fs::write(path.join("result.json"), payload).map_err(os)?;
        for (source, rel) in &state.assets {
            check()?;
            if source.is_file() {
                let dest = path.join(rel);
                mkdir(dest.parent().expect("media destinations have a parent"))?;
                copy(source, &dest)?;
            }
        }
    }
    Ok(path.to_path_buf())
}
