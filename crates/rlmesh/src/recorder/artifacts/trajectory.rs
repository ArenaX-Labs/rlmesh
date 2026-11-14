//! `trajectory.parquet`: one row per step, written a row group at a time.

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, BinaryArray, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use rlmesh_spaces::{DType, SpaceKind, SpaceSpec, SpaceValue, Tensor};
use serde_json::{Map, Value, json};

use super::is_image_space;
use super::os;
use super::schema::TrajectoryRecordSpec;
use crate::Result;

/// One observation or action: the JSON tree of its leaves, plus the raw
/// pixels of its image leaves concatenated, which the tree points into. The
/// pixels are shared, so a raw stream identical to the perturbed one clones
/// the value without owning the frames twice.
#[derive(Debug, Default, Clone)]
pub(super) struct StepValue {
    pub(super) json: Vec<u8>,
    pub(super) images: Bytes,
}

/// One `trajectory.parquet` row. The videos are lossy and decoupled; these
/// columns are the record of what the model actually saw and did.
#[derive(Debug, Default, Clone)]
pub(super) struct StepRow {
    pub(super) step: i64,
    pub(super) t_ns: i64,
    pub(super) predict_ms: Option<f64>,
    pub(super) step_ms: Option<f64>,
    pub(super) reward: Option<f64>,
    pub(super) info: Option<String>,
    pub(super) obs_raw: Option<StepValue>,
    pub(super) obs_perturbed: Option<StepValue>,
    pub(super) action_raw: Option<StepValue>,
    pub(super) action_perturbed: Option<StepValue>,
}

impl StepRow {
    /// Bytes this row keeps resident; pixels a raw/perturbed pair shares
    /// count once.
    fn resident_bytes(&self) -> usize {
        fn pair(a: Option<&StepValue>, b: Option<&StepValue>) -> usize {
            let shared =
                matches!((a, b), (Some(a), Some(b)) if a.images.as_ptr() == b.images.as_ptr());
            a.map_or(0, |v| v.json.len() + v.images.len())
                + b.map_or(0, |v| {
                    v.json.len() + if shared { 0 } else { v.images.len() }
                })
        }
        pair(self.obs_raw.as_ref(), self.obs_perturbed.as_ref())
            + pair(self.action_raw.as_ref(), self.action_perturbed.as_ref())
    }
}

fn value_of(name: &str) -> fn(&StepRow) -> Option<&StepValue> {
    match name {
        "obs_raw" => |row| row.obs_raw.as_ref(),
        "obs_perturbed" => |row| row.obs_perturbed.as_ref(),
        "action_raw" => |row| row.action_raw.as_ref(),
        "action_perturbed" => |row| row.action_perturbed.as_ref(),
        _ => |_| None,
    }
}

fn blob(row: &StepRow, get: fn(&StepRow) -> Option<&StepValue>) -> Option<&[u8]> {
    get(row)
        .map(|value| value.images.as_ref())
        .filter(|pixels: &&[u8]| !pixels.is_empty())
}

/// Columns follow the spec: `step`/`t_ns` always, the rest only when their
/// stream is enabled, plus the `_images` blob column of every value column
/// whose rows carry image leaves.
fn fields(rows: &[StepRow], spec: &TrajectoryRecordSpec) -> Vec<Field> {
    let mut fields = vec![
        Field::new("step", DataType::Int64, false),
        Field::new("t_ns", DataType::Int64, false),
    ];
    if spec.step_timing {
        fields.push(Field::new("predict_ms", DataType::Float64, true));
        fields.push(Field::new("step_ms", DataType::Float64, true));
    }
    if spec.reward {
        fields.push(Field::new("reward", DataType::Float64, true));
    }
    if spec.info {
        fields.push(Field::new("info", DataType::Utf8, true));
    }
    for (name, enabled) in [
        ("obs_raw", spec.obs.raw()),
        ("obs_perturbed", spec.obs.perturbed()),
        ("action_raw", spec.action.raw()),
        ("action_perturbed", spec.action.perturbed()),
    ] {
        if !enabled {
            continue;
        }
        fields.push(Field::new(name, DataType::Binary, true));
        let get = value_of(name);
        if rows.iter().any(|row| blob(row, get).is_some()) {
            fields.push(Field::new(format!("{name}_images"), DataType::Binary, true));
        }
    }
    fields
}

fn column(name: &str, rows: &[StepRow]) -> ArrayRef {
    let floats = |get: fn(&StepRow) -> Option<f64>| {
        Arc::new(Float64Array::from(rows.iter().map(get).collect::<Vec<_>>())) as ArrayRef
    };
    match name {
        "step" => Arc::new(Int64Array::from(
            rows.iter().map(|r| r.step).collect::<Vec<_>>(),
        )),
        "t_ns" => Arc::new(Int64Array::from(
            rows.iter().map(|r| r.t_ns).collect::<Vec<_>>(),
        )),
        "predict_ms" => floats(|r| r.predict_ms),
        "step_ms" => floats(|r| r.step_ms),
        "reward" => floats(|r| r.reward),
        "info" => Arc::new(StringArray::from_iter(
            rows.iter().map(|r| r.info.as_deref()),
        )),
        value => match value.strip_suffix("_images") {
            Some(value) => {
                let get = value_of(value);
                Arc::new(BinaryArray::from_iter(
                    rows.iter().map(|row| blob(row, get)),
                ))
            }
            None => {
                let get = value_of(value);
                Arc::new(BinaryArray::from_iter(
                    rows.iter().map(|row| get(row).map(|v| v.json.as_slice())),
                ))
            }
        },
    }
}

/// Trajectory bytes an open episode buffers before they go to disk as one
/// parquet row group, so what stays resident per episode is this budget, not
/// the episode's length. The flush copies the chunk into Arrow once, so an
/// episode mid-flush briefly holds twice this.
// TODO: one fixed budget per episode; a runner-wide budget shared across
// open episodes is the upgrade path if hundreds of image trajectories still
// crowd the runner's RSS.
pub(super) const CHUNK_BYTES: usize = 8 << 20;

/// Camera frames are most of a trajectory's bytes and stored as raw pixels;
/// zstd level 3 shrinks a LIBERO episode about 5x (111 MB to 21 MB) at the
/// same write cost as snappy.
fn writer_properties() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(
            ZstdLevel::try_new(3).expect("3 is a valid zstd level"),
        ))
        .build()
}

/// `trajectory.parquet` written a row group at a time to a `.rlmesh-tmp`
/// sibling (which the file listing and the uploader skip), moved into place
/// or removed at finish. The schema is fixed by the first row group: the
/// `_images` columns exist when its rows carry pixels, which for one episode's
/// observation space is every row or none.
pub(super) struct TrajectoryWriter {
    path: PathBuf,
    spec: TrajectoryRecordSpec,
    rows: Vec<StepRow>,
    buffered: usize,
    file: Option<(SchemaRef, ArrowWriter<File>)>,
}

impl TrajectoryWriter {
    pub(super) fn new(path: PathBuf, spec: TrajectoryRecordSpec) -> Self {
        Self {
            path,
            spec,
            rows: Vec::new(),
            buffered: 0,
            file: None,
        }
    }

    fn tmp(&self) -> PathBuf {
        self.path.with_extension("parquet.rlmesh-tmp")
    }

    pub(super) fn push(&mut self, row: StepRow) {
        self.buffered += row.resident_bytes();
        self.rows.push(row);
    }

    pub(super) fn over_budget(&self) -> bool {
        self.buffered >= CHUNK_BYTES
    }

    /// Encodes the buffered rows as one row group. Blocking.
    pub(super) fn flush(&mut self) -> Result<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        let rows = std::mem::take(&mut self.rows);
        self.buffered = 0;
        let (schema, writer) = match &mut self.file {
            Some(file) => file,
            None => {
                let schema = Arc::new(Schema::new(fields(&rows, &self.spec)));
                let writer = ArrowWriter::try_new(
                    File::create(self.tmp()).map_err(os)?,
                    schema.clone(),
                    Some(writer_properties()),
                )
                .map_err(os)?;
                self.file.insert((schema, writer))
            }
        };
        let arrays = schema
            .fields()
            .iter()
            .map(|field| column(field.name(), &rows))
            .collect();
        writer
            .write(&RecordBatch::try_new(schema.clone(), arrays).map_err(os)?)
            .map_err(os)?;
        writer.flush().map_err(os)?;
        Ok(())
    }

    /// Writes the tail and closes the file, which lands at the trajectory
    /// path only when `keep`; an episode that never closed a step leaves
    /// nothing.
    pub(super) fn finish(mut self, keep: bool) -> Result<()> {
        self.flush()?;
        let Some((_, writer)) = self.file.take() else {
            return Ok(());
        };
        let closed = writer.close();
        if closed.is_err() || !keep {
            let _ = std::fs::remove_file(self.tmp());
        }
        closed.map_err(os)?;
        if keep {
            std::fs::rename(self.tmp(), &self.path).map_err(os)?;
        }
        Ok(())
    }
}

impl Drop for TrajectoryWriter {
    /// An unfinished writer (a discarded episode, a failed flush) leaves no
    /// partial file behind.
    fn drop(&mut self) {
        if self.file.take().is_some() {
            let _ = std::fs::remove_file(self.tmp());
        }
    }
}

pub(super) fn step_value(value: &SpaceValue, space: &SpaceSpec, images: bool) -> StepValue {
    let mut buffer = images.then(Vec::new);
    let json = leaves_to_json(value, space, &mut buffer);
    StepValue {
        json: serde_json::to_vec(&json).unwrap_or_default(),
        images: buffer.map(Bytes::from).unwrap_or_default(),
    }
}

/// Image leaves are appended to `images` verbatim and stand in the tree as
/// `{shape, dtype, offset, len}`; every other leaf serializes inline. Without
/// a buffer the pixels are dropped and the leaf keeps only `{shape, dtype}`,
/// so a reader still sees what the model was given.
fn leaves_to_json(value: &SpaceValue, space: &SpaceSpec, images: &mut Option<Vec<u8>>) -> Value {
    match (value, space.spec.as_ref()) {
        (SpaceValue::Box(tensor), _) if is_image_space(space) => {
            let mut leaf = json!({ "shape": tensor.shape(), "dtype": "uint8" });
            if let Some(buffer) = images.as_mut() {
                let pixels = tensor.to_contiguous_bytes();
                leaf["offset"] = json!(buffer.len());
                leaf["len"] = json!(pixels.len());
                buffer.extend_from_slice(&pixels);
            }
            leaf
        }
        (SpaceValue::Box(tensor), _) => tensor_to_json(tensor),
        (SpaceValue::Discrete(value), _) => json!(value),
        (SpaceValue::MultiBinary(value), _) => json!(value),
        (SpaceValue::MultiDiscrete(value), _) => json!(value),
        (SpaceValue::Text(value), _) => json!(value),
        (SpaceValue::Dict(values), Some(SpaceKind::Dict(spec))) => {
            let mut out = Map::new();
            for (key, child_space) in spec.keys.iter().zip(spec.spaces.iter()) {
                if let Some(child_value) = values.get(key) {
                    out.insert(
                        key.clone(),
                        leaves_to_json(child_value, child_space, images),
                    );
                }
            }
            Value::Object(out)
        }
        (SpaceValue::Tuple(values), Some(SpaceKind::Tuple(spec))) => Value::Array(
            values
                .iter()
                .zip(spec.spaces.iter())
                .map(|(value, space)| leaves_to_json(value, space, images))
                .collect(),
        ),
        _ => Value::Null,
    }
}

fn tensor_to_json(tensor: &Tensor) -> Value {
    let data = tensor.to_contiguous_bytes();
    let numbers = match tensor.dtype() {
        DType::Float32 => decode_chunks(&data, 4, |b| {
            json!(f32::from_le_bytes(
                b.try_into().expect("chunk width checked")
            ))
        }),
        DType::Float64 => decode_chunks(&data, 8, |b| {
            json!(f64::from_le_bytes(
                b.try_into().expect("chunk width checked")
            ))
        }),
        DType::Int8 => Some(data.iter().map(|&b| json!(b as i8)).collect()),
        DType::Int16 => decode_chunks(&data, 2, |b| {
            json!(i16::from_le_bytes(
                b.try_into().expect("chunk width checked")
            ))
        }),
        DType::Int32 => decode_chunks(&data, 4, |b| {
            json!(i32::from_le_bytes(
                b.try_into().expect("chunk width checked")
            ))
        }),
        DType::Int64 => decode_chunks(&data, 8, |b| {
            json!(i64::from_le_bytes(
                b.try_into().expect("chunk width checked")
            ))
        }),
        DType::Uint8 => Some(data.iter().map(|&b| json!(b)).collect()),
        DType::Uint16 => decode_chunks(&data, 2, |b| {
            json!(u16::from_le_bytes(
                b.try_into().expect("chunk width checked")
            ))
        }),
        DType::Uint32 => decode_chunks(&data, 4, |b| {
            json!(u32::from_le_bytes(
                b.try_into().expect("chunk width checked")
            ))
        }),
        DType::Uint64 => decode_chunks(&data, 8, |b| {
            json!(u64::from_le_bytes(
                b.try_into().expect("chunk width checked")
            ))
        }),
        DType::Bool => Some(data.iter().map(|&b| json!(b != 0)).collect()),
        DType::Float16 => decode_chunks(&data, 2, |b| {
            json!(half::f16::from_le_bytes(b.try_into().expect("chunk width checked")).to_f32())
        }),
        DType::Unspecified => None,
    };
    match numbers {
        Some(numbers) => json!({ "shape": tensor.shape(), "data": numbers }),
        None => Value::Null,
    }
}

fn decode_chunks(data: &[u8], width: usize, decode: impl Fn(&[u8]) -> Value) -> Option<Vec<Value>> {
    if !data.len().is_multiple_of(width) {
        return None;
    }
    Some(data.chunks_exact(width).map(decode).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recorder::artifacts::schema::TrajectoryStream;
    use rlmesh_spaces::{BoxSpec, DictSpec};
    use std::collections::BTreeMap;

    fn box_spec(dtype: DType, shape: Vec<i64>) -> SpaceSpec {
        SpaceSpec {
            shape,
            dtype,
            spec: Some(SpaceKind::Box(BoxSpec { bounds: None })),
        }
    }

    /// A camera leaf plus a state leaf, the shape a VLA observation takes.
    fn image_and_state_observation() -> (SpaceSpec, SpaceValue, Vec<u8>) {
        let dict_space = SpaceSpec {
            shape: vec![],
            dtype: DType::Unspecified,
            spec: Some(SpaceKind::Dict(DictSpec {
                keys: vec!["agentview_image".into(), "robot0_eef_pos".into()],
                spaces: vec![
                    box_spec(DType::Uint8, vec![64, 64, 3]),
                    box_spec(DType::Float32, vec![3]),
                ],
            })),
        };
        let pixels = (0..64 * 64 * 3).map(|i| i as u8).collect::<Vec<_>>();
        let image_value = SpaceValue::Box(
            Tensor::from_vec(pixels.clone(), vec![64, 64, 3], DType::Uint8).unwrap(),
        );
        let state_bytes: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let state_value =
            SpaceValue::Box(Tensor::from_vec(state_bytes, vec![3], DType::Float32).unwrap());
        let mut dict = BTreeMap::new();
        dict.insert("agentview_image".to_string(), image_value);
        dict.insert("robot0_eef_pos".to_string(), state_value);
        (dict_space, SpaceValue::Dict(dict), pixels)
    }

    #[test]
    fn image_leaves_go_to_the_blob_and_state_leaves_stay_inline() {
        let (dict_space, value, pixels) = image_and_state_observation();

        let row = step_value(&value, &dict_space, true);
        let json: Value = serde_json::from_slice(&row.json).expect("row json must parse");
        let obj = json
            .as_object()
            .expect("dict leaf must serialize to an object");
        let image = &obj["agentview_image"];
        assert_eq!(image["shape"], json!([64, 64, 3]));
        assert_eq!(image["dtype"], json!("uint8"));
        assert_eq!(image["offset"], json!(0));
        assert_eq!(image["len"], json!(pixels.len()));
        let start = image["offset"].as_u64().unwrap() as usize;
        let len = image["len"].as_u64().unwrap() as usize;
        assert_eq!(&row.images[start..start + len], pixels.as_slice());
        assert_eq!(obj["robot0_eef_pos"]["shape"], json!([3]));
        assert_eq!(obj["robot0_eef_pos"]["data"], json!([1.0, 2.0, 3.0]));
    }

    #[test]
    fn state_leaves_survive_when_image_pixels_are_off() {
        let (dict_space, value, _) = image_and_state_observation();

        let row = step_value(&value, &dict_space, false);
        assert!(row.images.is_empty());
        let json: Value = serde_json::from_slice(&row.json).expect("row json must parse");
        assert_eq!(
            json["agentview_image"],
            json!({"shape": [64, 64, 3], "dtype": "uint8"})
        );
        assert_eq!(json["robot0_eef_pos"]["data"], json!([1.0, 2.0, 3.0]));
    }

    #[test]
    fn images_blob_column_appears_only_when_a_row_carries_pixels() {
        let spec = TrajectoryRecordSpec {
            obs: TrajectoryStream::Both,
            reward: false,
            info: false,
            step_timing: false,
            ..TrajectoryRecordSpec::default()
        };
        let row = |images: Vec<u8>| StepRow {
            obs_raw: Some(StepValue {
                json: b"{}".to_vec(),
                images: images.into(),
            }),
            obs_perturbed: Some(StepValue::default()),
            ..StepRow::default()
        };
        let dir = tempfile::tempdir().unwrap();
        let columns = |name: &str, rows: &[StepRow]| {
            let path = dir.path().join(name);
            let mut writer = TrajectoryWriter::new(path.clone(), spec.clone());
            for row in rows {
                writer.push(row.clone());
            }
            writer.finish(true).unwrap();
            let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                std::fs::File::open(&path).unwrap(),
            )
            .unwrap();
            for column in reader.metadata().row_group(0).columns() {
                assert!(matches!(column.compression(), Compression::ZSTD(_)));
            }
            reader
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            columns("with.parquet", &[row(Vec::new()), row(vec![7, 8])]),
            ["step", "t_ns", "obs_raw", "obs_raw_images", "obs_perturbed"]
        );
        assert_eq!(
            columns("without.parquet", &[row(Vec::new())]),
            ["step", "t_ns", "obs_raw", "obs_perturbed"]
        );
    }

    #[test]
    fn discrete_and_text_leaves_serialize_directly() {
        let bare = |dtype| SpaceSpec {
            shape: vec![],
            dtype,
            spec: None,
        };
        let mut images = Some(Vec::new());
        assert_eq!(
            leaves_to_json(&SpaceValue::Discrete(3), &bare(DType::Int64), &mut images),
            json!(3)
        );
        assert_eq!(
            leaves_to_json(
                &SpaceValue::Text("pick up the cup".into()),
                &bare(DType::Unspecified),
                &mut images,
            ),
            json!("pick up the cup")
        );
        assert!(images.expect("buffer stays available").is_empty());
    }
}
