//! The short random-action clip an environment probe captures from its scene
//! camera. Ported from the managed probe (`probe/preview.rs`); the clip is
//! encoded with the recorder's in-process AV1 writer, so no ffmpeg is needed.

use std::path::Path;

use image::RgbImage;
use image::imageops::FilterType;
use rlmesh_adapters::v1::{EnvFeature, EnvTags, PathSeg, SpaceView, join};
use rlmesh_spaces::spaces::SpaceKind;
use rlmesh_spaces::{DType, SpaceSpec, SpaceValue};

use super::compare::image_leaves;
use super::report::ProbePreview;
use crate::recorder::video::{Encoding, VideoWriter, rgb_from_hwc};

pub(super) const PREVIEW_FRAMES: usize = 60;
pub(super) const PREVIEW_MIN_FRAMES: usize = 10;
pub(super) const PREVIEW_FPS: u32 = 20;
const PREVIEW_MAX_WIDTH: u32 = 640;
const PREVIEW_QUALITY: u8 = 60;
/// The clip and its still, beside the report.
pub const PREVIEW_VIDEO: &str = "probe-preview.mp4";
pub const PREVIEW_IMAGE: &str = "probe-preview.png";
const MAIN_CAMERA_ROLE: &str = rlmesh_adapters::roles::core::IMAGE_PRIMARY;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TaggedImage {
    pub(super) path: Vec<String>,
    pub(super) role: String,
    pub(super) upside_down: bool,
}

/// The env's tagged image leaves, in declaration order; empty when the env
/// publishes no tags or they do not join against its live spaces.
pub(super) fn tagged_images(
    tags: Option<&EnvTags>,
    observation: &SpaceSpec,
    action: &SpaceSpec,
) -> Vec<TaggedImage> {
    let Some(tags) = tags else {
        return Vec::new();
    };
    let features = match join(
        tags,
        &SpaceView::from(observation),
        &SpaceView::from(action),
    ) {
        Ok(features) => features,
        Err(error) => {
            tracing::warn!(%error, "doctor preview ignores the env tags");
            return Vec::new();
        }
    };
    features
        .observation
        .iter()
        .filter_map(|feature| match feature {
            EnvFeature::Image(image) => Some(TaggedImage {
                path: image
                    .source
                    .0
                    .iter()
                    .map(|segment| match segment {
                        PathSeg::Key(key) => key.clone(),
                        PathSeg::Index(index) => index.to_string(),
                    })
                    .collect(),
                role: image.role.clone(),
                upside_down: image.upside_down,
            }),
            _ => None,
        })
        .collect()
}

/// The camera a preview reads: a leaf path, its name, and its orientation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Camera {
    pub(super) name: String,
    pub(super) path: Vec<String>,
    pub(super) upside_down: bool,
}

/// The image leaf to preview: the one tagged as the main camera when the tags
/// name one, else the first image leaf in the observation space. A tag on the
/// chosen leaf supplies its orientation.
pub(super) fn select_camera(space: &SpaceSpec, tagged: &[TaggedImage]) -> Option<Camera> {
    let mut leaves = Vec::new();
    image_leaves(space, Vec::new(), &mut leaves);
    let main = tagged
        .iter()
        .filter(|image| image.role == MAIN_CAMERA_ROLE)
        .find_map(|image| leaves.iter().find(|leaf| **leaf == image.path));
    let path = main.or(leaves.first())?.clone();
    let tag = tagged.iter().find(|image| image.path == path);
    Some(Camera {
        name: tag.map_or_else(
            || {
                if path.is_empty() {
                    "observation".to_string()
                } else {
                    path.join("/")
                }
            },
            |tag| tag.role.clone(),
        ),
        upside_down: tag.is_some_and(|tag| tag.upside_down),
        path,
    })
}

/// Width-capped output size, aspect kept.
fn preview_dimensions(width: u32, height: u32) -> (u32, u32) {
    if width <= PREVIEW_MAX_WIDTH {
        return (width, height);
    }
    let scaled = (u64::from(height) * u64::from(PREVIEW_MAX_WIDTH) + u64::from(width) / 2)
        / u64::from(width);
    (PREVIEW_MAX_WIDTH, u32::try_from(scaled).unwrap_or(1).max(1))
}

#[derive(Debug)]
pub(super) struct PreviewCapture {
    camera: Camera,
    width: u32,
    height: u32,
    frames: Vec<Vec<u8>>,
}

impl PreviewCapture {
    pub(super) fn new(camera: Camera) -> Self {
        Self {
            camera,
            width: 0,
            height: 0,
            frames: Vec::with_capacity(PREVIEW_FRAMES),
        }
    }

    pub(super) fn camera_name(&self) -> &str {
        &self.camera.name
    }

    pub(super) fn frames(&self) -> usize {
        self.frames.len()
    }

    pub(super) fn done(&self) -> bool {
        self.frames.len() >= PREVIEW_FRAMES
    }

    /// Keeps the camera frame of `observation`. Frames that fail to decode or
    /// that change shape after the first are dropped rather than aborting the
    /// probe.
    pub(super) fn capture(&mut self, observation: &SpaceValue, space: &SpaceSpec) {
        if self.done() {
            return;
        }
        let Some((mut image, _)) = frame_at(observation, space, &self.camera.path) else {
            return;
        };
        if self.camera.upside_down {
            image::imageops::rotate180_in_place(&mut image);
        }
        let (width, height) = preview_dimensions(image.width(), image.height());
        if (width, height) != image.dimensions() {
            image = image::imageops::resize(&image, width, height, FilterType::Lanczos3);
        }
        if self.frames.is_empty() {
            self.width = width;
            self.height = height;
        } else if (width, height) != (self.width, self.height) {
            return;
        }
        self.frames.push(image.into_raw());
    }

    /// Encodes the clip and its first frame under `output_dir`; `None` when
    /// nothing was captured. The still is the post-reset scene, before random
    /// actions disturb it. A failed encode leaves no partial mp4 behind.
    pub(super) fn write(self, output_dir: &Path) -> Result<Option<ProbePreview>, String> {
        let Some(still) = self.frames.first().cloned() else {
            return Ok(None);
        };
        let (width, height) = (self.width, self.height);
        let video_path = output_dir.join(PREVIEW_VIDEO);
        let encoded = (|| {
            let mut writer = VideoWriter::create(
                &video_path,
                width,
                height,
                PREVIEW_FPS,
                &Encoding::Av1 {
                    quality: PREVIEW_QUALITY,
                },
            )?;
            for frame in &self.frames {
                writer.push(frame, width, height, 3)?;
            }
            writer.finish()
        })();
        let stats = match encoded {
            Ok(stats) => stats,
            Err(error) => {
                let _ = std::fs::remove_file(&video_path);
                return Err(format!("encode {}: {error}", video_path.display()));
            }
        };
        let image = RgbImage::from_raw(width, height, still)
            .ok_or_else(|| format!("preview frame does not match {width}x{height}"))?;
        let image_path = output_dir.join(PREVIEW_IMAGE);
        image
            .save(&image_path)
            .map_err(|error| format!("write {}: {error}", image_path.display()))?;
        Ok(Some(ProbePreview {
            video: PREVIEW_VIDEO.to_string(),
            image: PREVIEW_IMAGE.to_string(),
            frames: stats.frame_count,
            width: stats.width,
            height: stats.height,
            fps: PREVIEW_FPS,
        }))
    }
}

/// The RGB frame at `path` in `value`, and its channel count; `None` when the
/// leaf is not a uint8 image.
fn frame_at(value: &SpaceValue, space: &SpaceSpec, path: &[String]) -> Option<(RgbImage, u32)> {
    let (mut value, mut space) = (value, space);
    for key in path {
        match (space.spec.as_ref()?, value) {
            (SpaceKind::Dict(spec), SpaceValue::Dict(values)) => {
                let index = spec.keys.iter().position(|k| k == key)?;
                space = &spec.spaces[index];
                value = values.get(key)?;
            }
            _ => return None,
        }
    }
    let SpaceValue::Box(tensor) = value else {
        return None;
    };
    if tensor.dtype() != DType::Uint8 {
        return None;
    }
    let bytes = tensor.to_contiguous_bytes();
    let dims: Vec<u32> = tensor
        .shape()
        .iter()
        .map(|dim| u32::try_from(*dim).ok())
        .collect::<Option<_>>()?;
    let (height, width, channels, chw) = match dims.as_slice() {
        [h, w] => (*h, *w, 1, false),
        [h, w, c @ (1 | 3 | 4)] => (*h, *w, *c, false),
        [c @ (1 | 3 | 4), h, w] => (*h, *w, *c, true),
        _ => return None,
    };
    let hwc = if chw {
        let plane = (height * width) as usize;
        let mut out = vec![0u8; plane * channels as usize];
        for c in 0..channels as usize {
            for i in 0..plane {
                out[i * channels as usize + c] = bytes[c * plane + i];
            }
        }
        out
    } else {
        bytes.into_owned()
    };
    Some((rgb_from_hwc(&hwc, width, height, channels)?, channels))
}

#[cfg(test)]
mod tests {
    use rlmesh_spaces::Tensor;
    use rlmesh_spaces::spaces::{BoxSpaceBuilder, DictSpaceBuilder};

    use super::*;

    fn image_space(height: i64, width: i64) -> SpaceSpec {
        BoxSpaceBuilder::scalar(0.0, 255.0, vec![height, width, 3])
            .dtype(DType::Uint8)
            .build()
            .expect("image")
    }

    fn two_camera_space() -> SpaceSpec {
        DictSpaceBuilder::new()
            .insert("agentview", image_space(4, 4))
            .insert("wrist", image_space(4, 4))
            .build()
            .expect("dict")
    }

    fn tagged(path: &[&str], role: &str, upside_down: bool) -> TaggedImage {
        TaggedImage {
            path: path.iter().map(ToString::to_string).collect(),
            role: role.to_string(),
            upside_down,
        }
    }

    #[test]
    fn the_main_camera_role_wins_over_declaration_order() {
        let space = two_camera_space();
        let camera = select_camera(
            &space,
            &[
                tagged(&["agentview"], "image/wrist", false),
                tagged(&["wrist"], MAIN_CAMERA_ROLE, true),
            ],
        )
        .expect("camera");
        assert_eq!(camera.path, ["wrist"]);
        assert_eq!(camera.name, MAIN_CAMERA_ROLE);
        assert!(camera.upside_down);

        // Untagged: the first image leaf, named by its path.
        let camera = select_camera(&space, &[]).expect("camera");
        assert_eq!(
            (camera.path, camera.name),
            (vec!["agentview".to_string()], "agentview".to_string())
        );
        assert!(
            select_camera(&BoxSpaceBuilder::unbounded(vec![3]).build().unwrap(), &[]).is_none()
        );
    }

    #[test]
    fn preview_dimensions_cap_the_width_and_keep_aspect() {
        assert_eq!(preview_dimensions(320, 240), (320, 240));
        assert_eq!(preview_dimensions(1280, 720), (640, 360));
        assert_eq!(preview_dimensions(4000, 1), (640, 1));
    }

    #[test]
    fn capture_reads_hwc_and_chw_and_drops_shape_changes() {
        let frame = |shape: Vec<i64>, fill: u8| {
            let len = shape.iter().product::<i64>() as usize;
            SpaceValue::Box(Tensor::from_vec(vec![fill; len], shape, DType::Uint8).unwrap())
        };
        let hwc = image_space(16, 24);
        let chw = BoxSpaceBuilder::scalar(0.0, 255.0, vec![3, 16, 24])
            .dtype(DType::Uint8)
            .build()
            .unwrap();
        let camera = |path: Vec<String>| Camera {
            name: "cam".into(),
            path,
            upside_down: false,
        };
        let mut capture = PreviewCapture::new(camera(Vec::new()));
        capture.capture(&frame(vec![16, 24, 3], 7), &hwc);
        capture.capture(&frame(vec![3, 16, 24], 9), &chw);
        assert_eq!(
            (capture.frames(), capture.width, capture.height),
            (2, 24, 16)
        );
        // A frame of another size, and a non-image value, are dropped.
        capture.capture(&frame(vec![2, 2, 3], 7), &image_space(2, 2));
        capture.capture(&SpaceValue::Discrete(1), &hwc);
        assert_eq!(capture.frames(), 2);
        assert!(capture.frames[1].iter().all(|byte| *byte == 9));

        let dir = tempfile::tempdir().unwrap();
        let written = capture.write(dir.path()).unwrap().expect("a clip");
        assert_eq!((written.frames, written.width, written.height), (2, 24, 16));
        assert!(dir.path().join(PREVIEW_VIDEO).is_file());
        assert!(dir.path().join(PREVIEW_IMAGE).is_file());
        assert_eq!(
            PreviewCapture::new(camera(Vec::new())).write(dir.path()),
            Ok(None)
        );
    }
}
