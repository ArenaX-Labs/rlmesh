//! A canned registry and the fixtures the publish tests share.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

use anyhow::{Result, anyhow};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::build::Builder;
use super::imagetools::Imagetools;
use super::media::{DOCKER_MANIFEST_LIST, DOCKER_MEDIA_PREFIX, OCI_INDEX};
use super::publish_with;
use super::source::Reference;
use crate::cli::PublishArgs;
use crate::image_check::PACKAGE_LABEL;
use crate::render::Style;

/// Canned registry contents, in the shapes `docker buildx imagetools
/// inspect` prints for `.Manifest` and `.Image`.
#[derive(Default)]
pub(super) struct FakeRegistry {
    pub(super) manifests: RefCell<BTreeMap<String, String>>,
    pub(super) raws: RefCell<BTreeMap<String, String>>,
    pub(super) configs: BTreeMap<String, String>,
    pub(super) created: RefCell<Vec<Vec<String>>>,
    /// Applied to each index a push assembles before it is stored, as a
    /// registry or docker that loses part of it would.
    pub(super) mangle: Cell<Option<fn(&mut Value)>>,
}

/// The index `imagetools create ARGS` assembles, reduced to what the
/// tests compare: one child per source, the index annotation. Like
/// docker, it is a Docker manifest list, without annotations, when every
/// descriptor the sources carry is a Docker manifest.
fn assembled(registry: &FakeRegistry, args: &[String]) -> (Vec<String>, Value) {
    let (mut tags, mut manifests, mut annotations) = (Vec::new(), Vec::new(), Map::new());
    let mut docker_only = true;
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dry-run" => {}
            "--tag" => tags.push(args.next().unwrap().clone()),
            "--annotation" => {
                let (key, value) = args.next().unwrap()["index:".len()..]
                    .split_once('=')
                    .unwrap();
                annotations.insert(key.to_owned(), Value::from(value));
            }
            source => {
                let digest = source.split_once('@').unwrap().1;
                docker_only &= registry
                    .media_types(digest)
                    .iter()
                    .all(|media_type| media_type.starts_with(DOCKER_MEDIA_PREFIX));
                manifests.push(json!({"digest": digest}));
            }
        }
    }
    let mut index = json!({"schemaVersion": 2, "mediaType": OCI_INDEX, "manifests": manifests});
    if docker_only {
        index["mediaType"] = Value::from(DOCKER_MANIFEST_LIST);
    } else if !annotations.is_empty() {
        index["annotations"] = Value::Object(annotations);
    }
    (tags, index)
}

impl Imagetools for FakeRegistry {
    fn manifest(&self, reference: &str) -> Result<String> {
        let key = self.resolve(reference);
        match self.manifests.borrow().get(&key) {
            // A canned failure other than "not found".
            Some(error) if error.starts_with("ERROR:") => Err(anyhow!("{error}")),
            Some(manifest) => Ok(manifest.clone()),
            None => Err(anyhow!("ERROR: {reference}: not found")),
        }
    }

    fn config(&self, reference: &str) -> Result<String> {
        self.configs
            .get(reference)
            .cloned()
            .ok_or_else(|| anyhow!("ERROR: {reference}: not found"))
    }

    /// The canned raw manifest when one was set, else the `.Manifest`
    /// view without the descriptor fields docker adds to it.
    fn raw(&self, reference: &str) -> Result<String> {
        if let Some(raw) = self.raws.borrow().get(&self.resolve(reference)) {
            return Ok(raw.clone());
        }
        let mut view: Value = serde_json::from_str(&self.manifest(reference)?)?;
        if let Some(view) = view.as_object_mut() {
            view.remove("digest");
            view.remove("size");
        }
        Ok(view.to_string())
    }

    /// Like docker: a lone index source without annotations is copied
    /// byte for byte; anything else is assembled into a new index.
    fn create(&self, args: &[String]) -> Result<String> {
        self.created.borrow_mut().push(args.to_vec());
        let (tags, mut index) = assembled(self, args);
        if args.iter().any(|a| a == "--dry-run") {
            return Ok(format!("{}\n", serde_json::to_string_pretty(&index)?));
        }
        let (mut sources, mut rest, mut annotated) = (Vec::new(), args.iter(), false);
        while let Some(arg) = rest.next() {
            match arg.as_str() {
                "--tag" => _ = rest.next(),
                "--annotation" => annotated = rest.next().is_some(),
                source => sources.push(source),
            }
        }
        let copied = match sources.as_slice() {
            [source] if !annotated => {
                let manifest: Value = serde_json::from_str(&self.manifest(source)?)?;
                if manifest.get("manifests").is_some() {
                    Some(self.raw(source)?)
                } else {
                    None
                }
            }
            _ => None,
        };
        let raw = match copied {
            Some(raw) => raw,
            None => {
                if let Some(mangle) = self.mangle.get() {
                    mangle(&mut index);
                }
                serde_json::to_string_pretty(&index)?
            }
        };
        for tag in tags {
            self.store(&tag, &raw);
        }
        Ok(String::new())
    }
}

pub(super) fn digest(seed: char) -> String {
    format!("sha256:{}", seed.to_string().repeat(64))
}

/// The digest of a manifest stored as `raw`, as a registry computes it.
pub(super) fn sha256(raw: &str) -> String {
    let hex: String = Sha256::digest(raw.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("sha256:{hex}")
}

/// A BuildKit push: an index of one image plus its attestation manifest.
pub(super) fn buildkit_index(image: &str, platform: (&str, &str)) -> Value {
    json!({
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "digest": digest('i'),
        "manifests": [
            {"mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": image,
             "platform": {"os": platform.0, "architecture": platform.1}},
            {"mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": digest('a'),
             "annotations": {"vnd.docker.reference.digest": image,
                             "vnd.docker.reference.type": "attestation-manifest"},
             "platform": {"os": "unknown", "architecture": "unknown"}}
        ]
    })
}

pub(super) fn oci_config(env: &[&str], cmd: &[&str], package: Option<Value>) -> String {
    let mut labels = Map::new();
    if let Some(package) = package {
        labels.insert(PACKAGE_LABEL.to_owned(), Value::from(package.to_string()));
    }
    json!({"os": "linux", "architecture": "amd64",
           "config": {"Env": env, "Cmd": cmd, "Labels": labels}})
    .to_string()
}

pub(super) const MODEL: &[&str] = &["python", "-m", "rlmesh.serve", "pi0:Policy"];
pub(super) const ENV: &[&str] = &["python", "-m", "rlmesh.serve", "--env", "sim:Env"];

impl FakeRegistry {
    /// Register `reference` as a BuildKit index (when `index_digest` is
    /// set) or a bare manifest, with `config` as its image's config.
    pub(super) fn add(
        &mut self,
        reference: &str,
        index_digest: Option<char>,
        image: char,
        config: String,
    ) {
        let repository = Reference::parse(reference).unwrap().repository;
        let image = digest(image);
        let manifest = match index_digest {
            Some(seed) => {
                let mut index = buildkit_index(&image, ("linux", "amd64"));
                index["digest"] = Value::from(digest(seed));
                index
            }
            None => json!({"mediaType": "application/vnd.oci.image.manifest.v1+json",
                           "digest": image, "size": 481}),
        };
        self.insert(reference, &manifest);
        self.configs.insert(format!("{repository}@{image}"), config);
    }

    /// The media types of the descriptors the manifest at `digest`
    /// brings into an index: its children, or itself.
    fn media_types(&self, digest: &str) -> Vec<String> {
        let manifests = self.manifests.borrow();
        let Some(manifest) = manifests
            .values()
            .filter_map(|raw| serde_json::from_str::<Value>(raw).ok())
            .find(|manifest| manifest["digest"] == digest)
        else {
            return vec![OCI_INDEX.to_owned()];
        };
        let media_type = |value: &Value| value["mediaType"].as_str().unwrap_or("").to_owned();
        match manifest["manifests"].as_array() {
            Some(children) => children.iter().map(media_type).collect(),
            None => vec![media_type(&manifest)],
        }
    }

    /// The key `reference` is stored under: itself, or for
    /// `repository@digest`, a tag of that repository holding the digest.
    fn resolve(&self, reference: &str) -> String {
        let manifests = self.manifests.borrow();
        if manifests.contains_key(reference) {
            return reference.to_owned();
        }
        let Some((repository, digest)) = reference.split_once('@') else {
            return reference.to_owned();
        };
        manifests
            .iter()
            .find(|(key, view)| {
                key.starts_with(&format!("{repository}:"))
                    && serde_json::from_str::<Value>(view)
                        .is_ok_and(|view| view["digest"] == digest)
            })
            .map_or_else(|| reference.to_owned(), |(key, _)| key.clone())
    }

    /// Store `raw` under `reference` as a registry would: at the sha256
    /// of its bytes, docker's view adding that digest and the size.
    pub(super) fn store(&self, reference: &str, raw: &str) {
        let mut view: Value = serde_json::from_str(raw).unwrap();
        view["digest"] = Value::from(sha256(raw));
        view["size"] = Value::from(raw.len());
        self.insert(reference, &view);
        self.raws
            .borrow_mut()
            .insert(reference.to_owned(), raw.to_owned());
    }

    /// The digest `reference` points at.
    pub(super) fn digest_of(&self, reference: &str) -> String {
        let view: Value = serde_json::from_str(&self.manifest(reference).unwrap()).unwrap();
        view["digest"].as_str().unwrap().to_owned()
    }

    pub(super) fn insert(&self, reference: &str, manifest: &Value) {
        self.insert_raw(reference, &manifest.to_string());
    }

    pub(super) fn insert_raw(&self, reference: &str, raw: &str) {
        self.raws.borrow_mut().remove(reference);
        self.manifests
            .borrow_mut()
            .insert(reference.to_owned(), raw.to_owned());
    }

    /// The non-dry-run `create` calls.
    pub(super) fn pushes(&self) -> usize {
        self.created
            .borrow()
            .iter()
            .filter(|args| !args.contains(&"--dry-run".to_owned()))
            .count()
    }
}

pub(super) fn three_variants() -> FakeRegistry {
    let mut registry = FakeRegistry::default();
    registry.add(
        "reg.example/ns/pi0:v3-cuda12",
        Some('1'),
        'c',
        oci_config(
            &["CUDA_VERSION=12.4.1", "NVIDIA_REQUIRE_CUDA=cuda>=12.4"],
            MODEL,
            Some(
                json!({"schemaVersion": 1, "variant": {"key": "torch-cuda12",
                "facets": {"framework": "torch", "accel": "cuda"},
                "requires": {"accel.vendor": "nvidia", "accel.cuda": ">=12.4"}, "priority": 10}}),
            ),
        ),
    );
    registry.add(
        "reg.example/ns/pi0:v3-rocm6",
        Some('2'),
        'r',
        oci_config(
            &["ROCM_VERSION=6.2.1"],
            MODEL,
            Some(json!({"schemaVersion": 1, "variant": {"key": "torch-rocm6",
                "requires": {"accel.vendor": "amd", "accel.gfx": ["gfx942", "gfx90a"]}, "priority": 5}})),
        ),
    );
    registry.add(
        "reg.example/ns/pi0:v3-jax",
        None,
        'j',
        oci_config(
            &[],
            MODEL,
            Some(json!({"schemaVersion": 1, "variant": {"key": "jax", "facets": {"framework": "jax"}},
                "profiles": [{"key": "osmesa", "default": true, "envVars": {"MUJOCO_GL": "osmesa"}},
                             {"key": "egl", "envVars": {"MUJOCO_GL": "egl"}, "gpu": {"count": 1}}]})),
        ),
    );
    registry
}

pub(super) fn args(target: &str, sources: &[&str]) -> PublishArgs {
    PublishArgs {
        target: target.to_owned(),
        sources: sources.iter().map(|s| (*s).to_owned()).collect(),
        tags: Vec::new(),
        channel: None,
        config: None,
        index_package: None,
        dry_run: false,
        json: false,
        force: false,
    }
}

pub(super) fn run(registry: &FakeRegistry, args: &PublishArgs) -> (Result<i32>, String) {
    let mut out = Vec::new();
    let result = publish_with(
        registry,
        &FakeBuilder::default(),
        args,
        &mut out,
        Style::for_terminal(false),
    );
    (result, String::from_utf8(out).unwrap())
}

pub(super) const SOURCES: &[&str] = &[
    "reg.example/ns/pi0:v3-cuda12",
    "reg.example/ns/pi0:v3-rocm6",
    "reg.example/ns/pi0:v3-jax",
];

/// A `variant`-only package label for a child keyed `key`.
pub(super) fn keyed(key: &str) -> String {
    oci_config(
        &[],
        MODEL,
        Some(json!({"schemaVersion": 1, "variant": {"key": key}})),
    )
}

pub(super) fn version_json(dir: &tempfile::TempDir, raw: &str) -> std::path::PathBuf {
    let path = dir.path().join("version.json");
    std::fs::write(&path, raw).unwrap();
    path
}

/// Builds nothing: answers each build with the digest seeded for its
/// variant key (read off the label argument), recording the arguments.
#[derive(Default)]
pub(super) struct FakeBuilder {
    pub(super) digests: BTreeMap<String, char>,
    pub(super) builds: RefCell<Vec<Vec<String>>>,
}

impl Builder for FakeBuilder {
    fn build(&self, args: &[String]) -> Result<String> {
        self.builds.borrow_mut().push(args.to_vec());
        let label = args
            .iter()
            .find_map(|arg| arg.strip_prefix(&format!("{PACKAGE_LABEL}=")))
            .ok_or_else(|| anyhow!("no label argument"))?;
        let label: Value = serde_json::from_str(label)?;
        let key = label["variant"]["key"].as_str().unwrap_or_default();
        match self.digests.get(key) {
            Some(seed) => Ok(digest(*seed)),
            None => Err(anyhow!("ERROR: failed to build {key}")),
        }
    }
}

pub(super) fn run_with(
    registry: &FakeRegistry,
    builder: &FakeBuilder,
    args: &PublishArgs,
) -> (Result<i32>, String) {
    let mut out = Vec::new();
    let result = publish_with(
        registry,
        builder,
        args,
        &mut out,
        Style::for_terminal(false),
    );
    (result, String::from_utf8(out).unwrap())
}
