//! Episode videos: cameras resolved from the env's tags, one mp4 per
//! `(camera, stream)` under `episodes/epNNNNNN/video/`.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use rlmesh_adapters::v1::{EnvFeature, PathSeg, SpaceView, join, roles};
use rlmesh_spaces::{DType, SpaceKind, SpaceSpec, SpaceValue, dtype_size};

use super::schema::{
    EPISODE_VIDEO_DIR, EpisodeRecording, RecordStream, VideoRecordSpec, sanitize_path_part,
};
use super::{is_image_space, os, warn};
use crate::Result;
use crate::recorder::video::{Encoding, ThreadedVideoWriter};

/// The scene-facing camera; an env that tags nothing falls back to its first
/// image leaf.
const MAIN_CAMERA_ROLE: &str = roles::core::IMAGE_PRIMARY;

/// `name` is the video file stem: the bare role (`primary` for
/// `image/primary`) for a tagged camera, the sanitized leaf name otherwise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Camera {
    pub(super) name: String,
    pub(super) role: Option<String>,
    pub(super) path: Vec<String>,
    pub(super) upside_down: bool,
}

pub(super) struct Frame {
    width: usize,
    height: usize,
    rgb: Vec<u8>,
}

/// The primary camera's position in the wire leaf order. Leaf slabs are
/// lane-contiguous fixed-stride, so a lane's frame is a direct slice of the
/// raw leaves, byte-identical to the decoded tensor's contiguous bytes (the
/// first-frame hash depends on that), with no decode.
#[derive(Debug, Clone)]
pub(super) struct PrimaryLeaf {
    leaf_index: usize,
    leaf_count: usize,
    frame_bytes: usize,
}

impl PrimaryLeaf {
    /// One lane's frame, or `None` when the payload does not match the
    /// resolved layout.
    pub(super) fn frame<'a>(
        &self,
        leaves: &'a [bytes::Bytes],
        lanes: usize,
        lane: usize,
    ) -> Option<&'a [u8]> {
        let slab = leaves.get(self.leaf_index)?;
        (leaves.len() == self.leaf_count && slab.len() == self.frame_bytes * lanes)
            .then(|| &slab[lane * self.frame_bytes..][..self.frame_bytes])
    }
}

struct Writer {
    writer: ThreadedVideoWriter,
    path: PathBuf,
    width: usize,
    height: usize,
    frames: u64,
}

/// One episode's encoders and the stride counter of its frames.
#[derive(Default)]
pub(super) struct EpisodeVideo {
    frames: u64,
    active: BTreeMap<(String, RecordStream), Writer>,
    /// Perturbed streams byte-identical to their raw stream so far: served
    /// by the raw file instead of a second encode.
    aliased: BTreeMap<(String, RecordStream), (String, RecordStream)>,
    /// Streams dropped for the rest of the episode after a frame changed shape.
    skipped: HashSet<(String, RecordStream)>,
}

/// The cameras each stream records, resolved once per session.
#[derive(Default)]
pub(super) struct Video {
    pub(super) fps: u32,
    pub(super) encoding: Encoding,
    stride: u32,
    cameras: Vec<(RecordStream, Camera)>,
    pub(super) primary: Option<PrimaryLeaf>,
}

impl Video {
    /// Resolve the spec's camera selectors against the env's tags (or every
    /// image leaf when it publishes none), and the primary camera the
    /// episode invariants watch whether or not it is recorded.
    pub(super) fn resolve(
        spec: &VideoRecordSpec,
        encoding: Encoding,
        contract: &rlmesh_proto::core::v1::EnvContract,
    ) -> Result<Self> {
        let mut video = Self {
            fps: spec.fps.max(1),
            encoding,
            stride: spec.stride.max(1),
            ..Self::default()
        };
        let contract = rlmesh_grpc::wire::env_contract_from_proto(contract.clone()).map_err(os)?;
        let Some(observation_space) = contract.observation_space.clone() else {
            return Ok(video);
        };
        let action_space = contract.action_space.clone().unwrap_or_default();
        let facade = crate::EnvContract::from_wire(&contract)?;
        let tagged = match crate::adapters::contract_tags(&facade) {
            Ok(tags) => join(
                &tags,
                &SpaceView::from(&observation_space),
                &SpaceView::from(&action_space),
            )
            .map_err(|err| {
                crate::Error::adapter_resolution(format!(
                    "env tags do not join against the env spaces: {err}"
                ))
            })?
            .observation
            .into_iter()
            .filter_map(|feature| match feature {
                EnvFeature::Image(image) => Some(Camera {
                    name: sanitize_path_part(bare_role(&image.role)),
                    role: Some(image.role.clone()),
                    path: image
                        .source
                        .0
                        .iter()
                        .map(|segment| match segment {
                            PathSeg::Key(key) => key.clone(),
                            PathSeg::Index(index) => index.to_string(),
                        })
                        .collect(),
                    upside_down: image.upside_down,
                }),
                _ => None,
            })
            .collect(),
            Err(_) => Vec::new(),
        };
        let mut leaves = Vec::new();
        collect_image_leaves(&observation_space, Vec::new(), &mut leaves);
        video.primary = primary_camera(&tagged, &leaves)
            .and_then(|camera| primary_leaf(&observation_space, &camera.path));
        let (streams, unmatched) = resolve_streams(spec, &tagged, &leaves);
        if !unmatched.is_empty() {
            let available = tagged
                .iter()
                .filter_map(|camera| camera.role.as_deref())
                .chain(leaves.iter().map(|leaf| leaf.name.as_str()))
                .collect::<Vec<_>>();
            warn(format!(
                "recorder: record.video selectors {unmatched:?} match no camera on this env \
                 (available: {available:?})"
            ));
        }
        video.cameras = streams;
        Ok(video)
    }

    pub(super) fn records(&self) -> bool {
        !self.cameras.is_empty()
    }

    /// Feed one lane's `(raw, perturbed)` observation into the episode's
    /// encoders. `aliasable` says the perturbed stream is the raw one (no
    /// transform replaced it).
    pub(super) fn observe(
        &self,
        episode: &mut EpisodeVideo,
        dir: &Path,
        space: &SpaceSpec,
        (raw, perturbed): (&SpaceValue, &SpaceValue),
        aliasable: bool,
    ) -> Result<()> {
        let index = episode.frames;
        episode.frames += 1;
        if !index.is_multiple_of(u64::from(self.stride)) {
            return Ok(());
        }
        for (stream, camera) in &self.cameras {
            let key = (camera.name.clone(), *stream);
            if episode.skipped.contains(&key) {
                continue;
            }
            if *stream == RecordStream::Perturbed && !episode.active.contains_key(&key) {
                let raw_key = self
                    .cameras
                    .iter()
                    .find(|(s, c)| *s == RecordStream::Raw && c.path == camera.path)
                    .map(|(s, c)| (c.name.clone(), *s));
                if let Some(raw_key) = raw_key
                    && aliasable
                {
                    episode.aliased.insert(key, raw_key);
                    continue;
                }
                if episode.aliased.remove(&key).is_some() {
                    warn(format!(
                        "recorder: perturbed frames of {} diverged from raw mid-episode; \
                         the perturbed recording starts here",
                        camera.name
                    ));
                }
            }
            let value = match stream {
                RecordStream::Perturbed => perturbed,
                RecordStream::Raw | RecordStream::Extra => raw,
            };
            let Some((leaf, leaf_space)) = value_and_space_at_path(value, space, &camera.path)
            else {
                continue;
            };
            let Some(mut frame) = frame_from_value(leaf, leaf_space) else {
                continue;
            };
            if camera.upside_down {
                rotate_180(&mut frame);
            }
            self.write(episode, key, dir, frame)?;
        }
        Ok(())
    }

    fn write(
        &self,
        episode: &mut EpisodeVideo,
        key: (String, RecordStream),
        dir: &Path,
        frame: Frame,
    ) -> Result<()> {
        if let Some(writer) = episode.active.get(&key)
            && (writer.width != frame.width || writer.height != frame.height)
        {
            warn(format!(
                "recorder: camera {} changed shape from {}x{} to {}x{}; dropping its {} stream \
                 for the rest of the episode",
                key.0,
                writer.width,
                writer.height,
                frame.width,
                frame.height,
                key.1.as_str()
            ));
            episode.skipped.insert(key);
            return Ok(());
        }
        if !episode.active.contains_key(&key) {
            let path =
                dir.join(EPISODE_VIDEO_DIR)
                    .join(format!("{}.{}.mp4", key.0, key.1.as_str()));
            std::fs::create_dir_all(path.parent().expect("video files have a parent"))
                .map_err(os)?;
            let writer = ThreadedVideoWriter::create(
                &path,
                frame.width as u32,
                frame.height as u32,
                self.fps,
                &self.encoding,
            )
            .map_err(os)?;
            episode.active.insert(
                key.clone(),
                Writer {
                    writer,
                    path,
                    width: frame.width,
                    height: frame.height,
                    frames: 0,
                },
            );
        }
        let writer = episode.active.get_mut(&key).expect("opened above");
        writer
            .writer
            .push(frame.rgb, frame.width as u32, frame.height as u32, 3)
            .map_err(os)?;
        writer.frames += 1;
        Ok(())
    }

    /// Close every encoder; a failed encode is warned about and its partial
    /// file removed, so the result lists only files that exist. Paths are
    /// relative to `root`.
    pub(super) fn finish(&self, episode: EpisodeVideo, root: &Path) -> Vec<EpisodeRecording> {
        let mut recordings = Vec::new();
        for ((camera, stream), mut writer) in episode.active {
            // The encoded dimensions: AV1 crops odd ones, H.264 pads them.
            let stats = match writer.writer.finish() {
                Ok(stats) => stats,
                Err(err) => {
                    let _ = std::fs::remove_file(&writer.path);
                    warn(format!(
                        "recorder: dropping video {}: {err}",
                        writer.path.display()
                    ));
                    continue;
                }
            };
            recordings.push(EpisodeRecording {
                camera,
                stream,
                path: relative(&writer.path, root),
                frames: writer.frames,
                fps: self.fps,
                width: stats.width,
                height: stats.height,
            });
        }
        for ((camera, stream), raw) in episode.aliased {
            if let Some(found) = recordings
                .iter()
                .find(|r| r.camera == raw.0 && r.stream == raw.1)
            {
                recordings.push(EpisodeRecording {
                    camera,
                    stream,
                    ..found.clone()
                });
            }
        }
        recordings
    }
}

pub(super) fn relative(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn bare_role(role: &str) -> &str {
    role.rsplit('/').next().unwrap_or(role)
}

/// Each stream's cameras (deduplicated by source leaf) and the selectors
/// that matched nothing.
fn resolve_streams(
    spec: &VideoRecordSpec,
    tagged: &[Camera],
    leaves: &[Camera],
) -> (Vec<(RecordStream, Camera)>, Vec<String>) {
    let mut unmatched = Vec::new();
    let streams = [
        (RecordStream::Raw, &spec.raw),
        (RecordStream::Perturbed, &spec.perturbed),
        (RecordStream::Extra, &spec.extra),
    ]
    .into_iter()
    .flat_map(|(stream, selectors)| {
        let mut cameras: Vec<Camera> = Vec::new();
        for selector in selectors {
            let selected = select_cameras(selector, tagged, leaves);
            if selected.is_empty() && !unmatched.contains(selector) {
                unmatched.push(selector.clone());
            }
            for camera in selected {
                if !cameras.iter().any(|known| known.path == camera.path) {
                    cameras.push(camera);
                }
            }
        }
        cameras.into_iter().map(move |camera| (stream, camera))
    })
    .collect();
    (streams, unmatched)
}

/// `all` selects every tagged camera; any other selector names a tagged
/// camera by bare or full role (`primary`, `image/primary`), else an image
/// leaf by name or dotted path. An env without tags resolves `all` and any
/// selector naming no leaf to every image leaf.
fn select_cameras(selector: &str, tagged: &[Camera], leaves: &[Camera]) -> Vec<Camera> {
    if selector == "all" {
        return if tagged.is_empty() {
            leaves.to_vec()
        } else {
            tagged.to_vec()
        };
    }
    let by_role = tagged
        .iter()
        .filter(|camera| {
            camera
                .role
                .as_deref()
                .is_some_and(|role| role == selector || bare_role(role) == selector)
        })
        .cloned()
        .collect::<Vec<_>>();
    if !by_role.is_empty() {
        return by_role;
    }
    let name = sanitize_path_part(selector);
    let by_leaf = leaves
        .iter()
        .filter(|leaf| {
            leaf.name == name
                || leaf.path.join(".") == selector
                || leaf.path.last().is_some_and(|last| last == selector)
        })
        .cloned()
        .collect::<Vec<_>>();
    if by_leaf.is_empty() && tagged.is_empty() {
        return leaves.to_vec();
    }
    by_leaf
}

/// The leaf at `path` (Dict keys / Tuple indices) in the wire's pre-order:
/// the total leaf count and, when found, the leaf's index and spec.
fn locate_leaf<'s>(
    space: &'s SpaceSpec,
    path: &[String],
) -> (usize, Option<(usize, &'s SpaceSpec)>) {
    fn walk<'s>(
        space: &'s SpaceSpec,
        path: &[String],
        matched: Option<usize>,
        count: &mut usize,
        found: &mut Option<(usize, &'s SpaceSpec)>,
    ) {
        let advance = |segment: &str| {
            matched
                .filter(|m| path.get(*m).map(String::as_str) == Some(segment))
                .map(|m| m + 1)
        };
        match space.spec.as_ref() {
            Some(SpaceKind::Dict(dict)) => {
                for (key, child) in dict.keys.iter().zip(dict.spaces.iter()) {
                    walk(child, path, advance(key), count, found);
                }
            }
            Some(SpaceKind::Tuple(tuple)) => {
                for (index, child) in tuple.spaces.iter().enumerate() {
                    walk(child, path, advance(&index.to_string()), count, found);
                }
            }
            _ => {
                if matched == Some(path.len()) && found.is_none() {
                    *found = Some((*count, space));
                }
                *count += 1;
            }
        }
    }
    let mut count = 0;
    let mut found = None;
    walk(space, path, Some(0), &mut count, &mut found);
    (count, found)
}

/// `None` when the camera's leaf is not a fixed-stride Box; the invariants
/// then see no frames.
fn primary_leaf(space: &SpaceSpec, path: &[String]) -> Option<PrimaryLeaf> {
    let (leaf_count, found) = locate_leaf(space, path);
    let (leaf_index, leaf) = found?;
    if !matches!(leaf.spec.as_ref(), Some(SpaceKind::Box(_))) {
        return None;
    }
    let numel = leaf.shape.iter().try_fold(1usize, |acc, dim| {
        usize::try_from(*dim).ok().and_then(|d| acc.checked_mul(d))
    })?;
    let frame_bytes = numel.checked_mul(dtype_size(leaf.dtype))?;
    (frame_bytes > 0).then_some(PrimaryLeaf {
        leaf_index,
        leaf_count,
        frame_bytes,
    })
}

fn primary_camera(tagged: &[Camera], leaves: &[Camera]) -> Option<Camera> {
    tagged
        .iter()
        .find(|camera| camera.role.as_deref() == Some(MAIN_CAMERA_ROLE))
        .or_else(|| leaves.first())
        .cloned()
}

fn collect_image_leaves(space: &SpaceSpec, path: Vec<String>, out: &mut Vec<Camera>) {
    match space.spec.as_ref() {
        Some(SpaceKind::Box(_)) if is_image_space(space) => {
            let name = if path.is_empty() {
                "observation".to_string()
            } else {
                path.join("_")
            };
            out.push(Camera {
                name: sanitize_path_part(&name),
                role: None,
                path,
                upside_down: false,
            });
        }
        Some(SpaceKind::Dict(spec)) => {
            for (key, child_space) in spec.keys.iter().zip(spec.spaces.iter()) {
                let mut child_path = path.clone();
                child_path.push(key.clone());
                collect_image_leaves(child_space, child_path, out);
            }
        }
        _ => {}
    }
}

fn value_and_space_at_path<'a>(
    value: &'a SpaceValue,
    space: &'a SpaceSpec,
    path: &[String],
) -> Option<(&'a SpaceValue, &'a SpaceSpec)> {
    let Some((head, tail)) = path.split_first() else {
        return Some((value, space));
    };
    let (SpaceValue::Dict(values), Some(SpaceKind::Dict(spec))) = (value, space.spec.as_ref())
    else {
        return None;
    };
    let index = spec.keys.iter().position(|key| key == head)?;
    value_and_space_at_path(values.get(head)?, spec.spaces.get(index)?, tail)
}

// An upside-down camera is a 180° rotation, not a vertical mirror: this must
// match the model-input path (rlmesh-adapters flip_180), or recordings show
// a mirrored scene relative to what the model consumed.
fn rotate_180(frame: &mut Frame) {
    const CHANNELS: usize = 3;
    let pixels = frame.width * frame.height;
    if pixels == 0 || frame.rgb.len() != pixels * CHANNELS {
        return;
    }
    for pixel in 0..pixels / 2 {
        let (a, b) = (pixel * CHANNELS, (pixels - 1 - pixel) * CHANNELS);
        for offset in 0..CHANNELS {
            frame.rgb.swap(a + offset, b + offset);
        }
    }
}

fn frame_from_value(value: &SpaceValue, space: &SpaceSpec) -> Option<Frame> {
    let SpaceValue::Box(value) = value else {
        return None;
    };
    if !is_image_space(space) || value.dtype() != DType::Uint8 {
        return None;
    }
    normalize_rgb(value.shape(), value.to_contiguous_bytes())
}

fn normalize_rgb(shape: &[i64], data: Cow<'_, [u8]>) -> Option<Frame> {
    let dims = shape
        .iter()
        .map(|&d| usize::try_from(d).ok())
        .collect::<Option<Vec<_>>>()?;
    let (height, width, rgb) = match *dims.as_slice() {
        [height, width] if data.len() == height * width => (
            height,
            width,
            data.iter().flat_map(|&v| [v, v, v]).collect(),
        ),
        // Already packed RGB: reuse the buffer whole.
        [height, width, 3] if data.len() == height * width * 3 => {
            (height, width, data.into_owned())
        }
        [height, width, channels @ (1 | 4)] if data.len() == height * width * channels => (
            height,
            width,
            data.chunks_exact(channels)
                .flat_map(|p| {
                    if channels == 1 {
                        [p[0]; 3]
                    } else {
                        [p[0], p[1], p[2]]
                    }
                })
                .collect(),
        ),
        [channels @ (1 | 3 | 4), height, width] if data.len() == height * width * channels => {
            let plane = height * width;
            let channel = |c: usize| if channels == 1 { 0 } else { c * plane };
            let rgb = (0..plane)
                .flat_map(|i| [0, 1, 2].map(|c| data[channel(c) + i]))
                .collect();
            (height, width, rgb)
        }
        _ => return None,
    };
    Some(Frame { width, height, rgb })
}
