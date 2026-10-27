//! `rlmesh eval run`: an evaluation request on this machine, in Docker.
//!
//! The request compiles to a plan here (`rlmesh::binding::plan`) against the
//! images it names: their `dev.rlmesh.package` labels, and for an env its
//! describe envelope (its `dev.rlmesh.describe` label, else `python -m
//! rlmesh._describe` run in the image). `python -m rlmesh._eval_run` then
//! runs the workloads in sandboxes and writes their bundles, and the report
//! is read back from them. `<out>/index.json` holds the request, the plan and
//! each workload's status.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use rlmesh::binding::plan::{self, EvalRequest, ImageEntry, Images, Plan, Workload};
use rlmesh::binding::report::{EvalReport, read_bundles};
use serde_json::{Value, json};

use crate::check::{docker_inspect, python_command};
use crate::cli::EvalRunArgs;
use crate::image_check::{DESCRIBE_LABEL, ImageConfig, PACKAGE_LABEL, parse_serve_command};

/// Where a catalog name resolves when no local build carries it.
const CATALOG_REGISTRY: &str = "registry.rlmesh.dev";
const INDEX_SCHEMA: &str = "rlmesh.eval-run.v1";
const RUN_MODULE: &str = "rlmesh._eval_run";

pub(crate) fn eval_run(
    args: &EvalRunArgs,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
) -> Result<i32> {
    let document = crate::platform::read_request(&args.request)?;
    let request = EvalRequest::parse(&document)?;
    let images = resolve_images(&request, args.catalog.as_deref(), stderr)?;
    let plan = plan::compile(&request, &images)?;
    for warning in &plan.warnings {
        writeln!(stderr, "warning: {warning}")?;
    }
    if args.gpus.is_none() {
        let needs = plan
            .models
            .iter()
            .filter(|m| m.gpu)
            .map(|m| &m.model)
            .chain(plan.workloads.iter().filter(|w| w.env_gpu).map(|w| &w.env));
        if let Some(name) = needs.into_iter().next() {
            bail!("{name} needs a GPU (its package declares resourceClass gpu); pass --gpus all");
        }
    }

    let index_path = args.out.join("index.json");
    if index_path.exists() {
        if !args.resume {
            bail!(
                "{} holds a previous run; pass --resume to continue it, or another --out",
                args.out.display()
            );
        }
        let previous: Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)
            .with_context(|| format!("reading {}", index_path.display()))?;
        if previous["request"] != document {
            bail!(
                "{} holds a run of another request; --resume continues the same one",
                args.out.display()
            );
        }
    }
    std::fs::create_dir_all(args.out.join("workloads"))?;
    let mut status: BTreeMap<String, Value> = BTreeMap::new();
    for workload in &plan.workloads {
        let bundle = bundle_dir(&args.out, workload);
        if args.resume && recorded(&bundle, workload) {
            status.insert(
                workload.id.clone(),
                json!({"status": "done", "resumed": true}),
            );
        } else if bundle.exists() {
            std::fs::remove_dir_all(&bundle)?;
        }
    }
    let pending: Vec<&str> = plan
        .workloads
        .iter()
        .map(|w| w.id.as_str())
        .filter(|id| !status.contains_key(*id))
        .collect();
    write_index(&index_path, &document, &plan, &status)?;
    if !pending.is_empty() {
        writeln!(
            stderr,
            "eval run: {} of {} workloads to run, {} recorded",
            pending.len(),
            plan.workloads.len(),
            plan.workloads.len() - pending.len()
        )?;
        status.extend(run_workloads(args, &plan, &pending)?);
    }
    for id in pending {
        status.entry(id.to_string()).or_insert_with(
            || json!({"status": "failed", "error": "the runner exited without a verdict"}),
        );
    }
    write_index(&index_path, &document, &plan, &status)?;

    let done: Vec<PathBuf> = plan
        .workloads
        .iter()
        .filter(|w| status[&w.id]["status"] == "done")
        .map(|w| bundle_dir(&args.out, w))
        .collect();
    let report = EvalReport::load(&done)?;
    if args.json {
        writeln!(
            stdout,
            "{}",
            serde_json::to_string_pretty(&report.to_json())?
        )?;
    } else {
        writeln!(stdout, "{}", report.format_table())?;
    }
    let failed: Vec<(&String, &Value)> = status
        .iter()
        .filter(|(_, s)| s["status"] != "done")
        .collect();
    for (id, s) in &failed {
        writeln!(
            stderr,
            "failed: {id}: {}",
            s["error"].as_str().unwrap_or_default()
        )?;
    }
    Ok(i32::from(!failed.is_empty()))
}

fn bundle_dir(out: &Path, workload: &Workload) -> PathBuf {
    out.join("workloads").join(&workload.id)
}

/// The workload's bundle holds every planned episode, seeded as planned.
fn recorded(bundle: &Path, workload: &Workload) -> bool {
    let Ok(bundles) = read_bundles([bundle]) else {
        return false;
    };
    let [recorded] = bundles.as_slice() else {
        return false;
    };
    if recorded.episodes.len() as u64 != workload.episodes {
        return false;
    }
    let mut seeds: Vec<Option<i64>> = recorded.episodes.iter().map(|e| e.seed).collect();
    if seeds.iter().all(Option::is_none) {
        return true;
    }
    let mut want: Vec<Option<i64>> = workload.episode_seeds().into_iter().map(Some).collect();
    seeds.sort_unstable();
    want.sort_unstable();
    seeds == want
}

fn write_index(
    path: &Path,
    request: &Value,
    plan: &Plan,
    status: &BTreeMap<String, Value>,
) -> Result<()> {
    let workloads: Vec<Value> = plan
        .workloads
        .iter()
        .map(|w| {
            let mut entry = json!({"id": w.id, "bundle": format!("workloads/{}", w.id)});
            let state = status
                .get(&w.id)
                .cloned()
                .unwrap_or(json!({"status": "pending"}));
            for (key, value) in state.as_object().into_iter().flatten() {
                entry[key] = value.clone();
            }
            entry
        })
        .collect();
    let index = json!({
        "schema": INDEX_SCHEMA,
        "request": request,
        "plan": plan,
        "workloads": workloads,
    });
    let partial = path.with_extension("json.partial");
    std::fs::write(&partial, serde_json::to_vec_pretty(&index)?)?;
    std::fs::rename(&partial, path)?;
    Ok(())
}

/// Run the pending workloads through `python -m rlmesh._eval_run`, and read
/// back each one's verdict.
fn run_workloads(
    args: &EvalRunArgs,
    plan: &Plan,
    pending: &[&str],
) -> Result<BTreeMap<String, Value>> {
    let status_path = args.out.join("status.jsonl");
    if status_path.exists() {
        std::fs::remove_file(&status_path)?;
    }
    let job = json!({
        "plan": plan,
        "pending": pending,
        "out": args.out,
        "gpus": args.gpus,
        "volumes": args.volumes,
        "maxEnvs": args.max_envs,
        "status": status_path,
    });
    let mut child = python_command()?
        .arg("-m")
        .arg(RUN_MODULE)
        .stdin(Stdio::piped())
        .spawn()
        .with_context(|| format!("running python -m {RUN_MODULE}"))?;
    child
        .stdin
        .take()
        .context("runner stdin")?
        .write_all(&serde_json::to_vec(&job)?)?;
    let exit = child.wait()?;
    let mut verdicts = BTreeMap::new();
    if let Ok(text) = std::fs::read_to_string(&status_path) {
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let mut verdict: Value = serde_json::from_str(line)?;
            if let Some(id) = verdict
                .as_object_mut()
                .and_then(|v| v.remove("workload"))
                .and_then(|id| id.as_str().map(str::to_string))
            {
                verdicts.insert(id, verdict);
            }
        }
    }
    if !exit.success() && verdicts.is_empty() {
        bail!("python -m {RUN_MODULE} exited with {exit}");
    }
    Ok(verdicts)
}

/// The image and labels behind every model and env the request names.
fn resolve_images(
    request: &EvalRequest,
    catalog: Option<&str>,
    stderr: &mut impl Write,
) -> Result<Images> {
    let mut images = Images::default();
    for model in &request.models {
        if !images.models.contains_key(&model.model) {
            let (image, config) = inspect(&catalog_image(&model.model, catalog), stderr)?;
            images.models.insert(
                model.model.clone(),
                ImageEntry {
                    package: package_label(&image, &config)?,
                    image,
                    describe: Value::Null,
                },
            );
        }
    }
    for task in &request.tasks {
        if !images.envs.contains_key(&task.env) {
            let (image, config) = inspect(&catalog_image(&task.env, catalog), stderr)?;
            images.envs.insert(
                task.env.clone(),
                ImageEntry {
                    package: package_label(&image, &config)?,
                    describe: describe(&image, &config, stderr)?,
                    image,
                },
            );
        }
    }
    Ok(images)
}

/// A catalog name's image: `prebuilt/<path>[:tag]` is the local build
/// `rlmesh-<path with - for />:<tag>` when one exists, else the registry's.
/// Any other name, and every name without `--catalog`, is an image already.
fn catalog_image(name: &str, catalog: Option<&str>) -> String {
    let (Some(local_tag), Some(path)) = (catalog, name.strip_prefix("prebuilt/")) else {
        return name.to_string();
    };
    if path.contains('@') {
        return format!("{CATALOG_REGISTRY}/{name}");
    }
    let (repo, tag) = match path.rsplit_once(':') {
        Some((repo, tag)) => (repo, Some(tag)),
        None => (path, None),
    };
    let local = format!(
        "rlmesh-{}:{}",
        repo.replace('/', "-"),
        tag.unwrap_or(local_tag)
    );
    if docker_inspect(&local).is_ok() {
        return local;
    }
    format!(
        "{CATALOG_REGISTRY}/prebuilt/{repo}:{}",
        tag.unwrap_or("latest")
    )
}

/// The image's config, pulled first when it is not here.
fn inspect(image: &str, stderr: &mut impl Write) -> Result<(String, ImageConfig)> {
    if let Ok(config) = docker_inspect(image) {
        return Ok((image.to_string(), config));
    }
    writeln!(stderr, "eval run: pulling {image}")?;
    let pulled = Command::new("docker")
        .args(["pull", image])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .status()
        .context("running docker pull")?;
    if !pulled.success() {
        bail!("docker pull {image} failed");
    }
    Ok((image.to_string(), docker_inspect(image)?))
}

fn package_label(image: &str, config: &ImageConfig) -> Result<Value> {
    config
        .labels
        .get(PACKAGE_LABEL)
        .map_or(Ok(Value::Null), |raw| {
            serde_json::from_str(raw)
                .with_context(|| format!("{image}: {PACKAGE_LABEL} is not JSON"))
        })
}

/// An env's describe envelope: its label, else `rlmesh._describe` run in the
/// image on the target its command serves.
fn describe(image: &str, config: &ImageConfig, stderr: &mut impl Write) -> Result<Value> {
    if let Some(raw) = config.labels.get(DESCRIBE_LABEL) {
        return serde_json::from_str(raw)
            .with_context(|| format!("{image}: {DESCRIBE_LABEL} is not JSON"));
    }
    let command = parse_serve_command(config);
    let (Some(serve), Some(target)) = (command.serve, command.target) else {
        bail!("{image}: its command serves no rlmesh target, so its scenarios cannot be read");
    };
    let python = serve
        .checked_sub(2)
        .and_then(|at| command.argv.get(at))
        .map_or("python", String::as_str);
    writeln!(stderr, "eval run: describing {image}")?;
    let output = Command::new("docker")
        .args([
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            python,
            image,
        ])
        .args(["-m", "rlmesh._describe", &target])
        .stdin(Stdio::null())
        .output()
        .context("running docker run")?;
    if !output.status.success() {
        let text = String::from_utf8_lossy(&output.stderr);
        let tail: Vec<&str> = text.lines().rev().take(8).collect();
        bail!(
            "{image}: describing {target} failed:\n{}",
            tail.into_iter().rev().collect::<Vec<_>>().join("\n")
        );
    }
    serde_json::from_slice(&output.stdout)
        .with_context(|| format!("{image}: rlmesh._describe printed no JSON"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workload(episodes: u64) -> Workload {
        Workload {
            id: "m__t".into(),
            model: 0,
            task: "t".into(),
            shard: None,
            env: "e".into(),
            env_image: "e".into(),
            env_params: Default::default(),
            env_gpu: false,
            scenario: None,
            suite: None,
            episodes,
            seed: Some(0),
            seeds: Vec::new(),
            seed_key: "e".into(),
            seed_slot_offset: 4,
            trial_index_base: None,
            max_episode_steps: None,
        }
    }

    #[test]
    fn resume_skips_only_bundles_that_hold_the_planned_episodes() {
        let dir = tempfile::tempdir().unwrap();
        let planned = workload(3);
        let seeds = planned.episode_seeds();
        for (name, recorded_seeds, want) in [
            ("complete", seeds.clone(), true),
            ("short", seeds[..2].to_vec(), false),
            ("other seeds", vec![1, 2, 3], false),
        ] {
            let result = rlmesh::RunResult {
                episodes: recorded_seeds
                    .iter()
                    .enumerate()
                    .map(|(index, seed)| rlmesh::EpisodeResult {
                        index: index as i64,
                        seed: Some(*seed),
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            };
            let bundle = dir.path().join(name);
            result.save(&bundle, Default::default()).unwrap();
            assert_eq!(recorded(&bundle, &planned), want, "{name}");
        }
        assert!(!recorded(&dir.path().join("missing"), &planned));
    }
}
