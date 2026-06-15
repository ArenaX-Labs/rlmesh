use std::fs;
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use rlmesh_grpc::{EnvClient, ModelClient};
use serde::Serialize;
use tempfile::TempDir;
use uuid::Uuid;

use crate::source::ResolvedEnvironmentSourceRef;
use crate::{EffectiveSandboxSpec, EnvironmentSourceRef, hf, shell_quote};

const DEFAULT_CONTAINER_PORT: u16 = 50051;
const READY_PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const CONTAINER_LOG_TAIL_BYTES: usize = 64 * 1024;

/// Label stamped on every container rlmesh starts.
const OWNER_LABEL: &str = "rlmesh.sandbox=1";
const OWNER_LABEL_FILTER: &str = "label=rlmesh.sandbox=1";

/// Label key recording the OS process id that owns a container.
const OWNER_PID_LABEL_KEY: &str = "rlmesh.sandbox.owner-pid";
const OWNER_PID_NS_LABEL_KEY: &str = "rlmesh.sandbox.owner-pid-ns";

/// `docker ps --format` template emitting one
/// `<id>|<owner-pid>|<owner-pid-ns>|<status>` line
/// per container. A `|` delimiter (rather than whitespace) is used so the
/// *empty* owner-pid field of legacy containers that predate per-process
/// labeling is preserved as an empty middle column rather than collapsed away.
/// `.State` is the single-word state (e.g. `running`, `exited`), matching the
/// values [`ContainerState::status`] holds.
const REAP_PS_FORMAT: &str = "{{.ID}}|{{.Label \"rlmesh.sandbox.owner-pid\"}}|{{.Label \"rlmesh.sandbox.owner-pid-ns\"}}|{{.State}}";

#[derive(Debug, Clone)]
pub struct BuildArtifact {
    pub image_id: String,
}

#[derive(Debug, Clone)]
pub struct StartedContainer {
    pub container_id: String,
    pub address: String,
}

#[derive(Debug, Clone, Default)]
pub struct DockerBackend;

impl DockerBackend {
    pub fn ensure_image(&self, spec: &EffectiveSandboxSpec) -> Result<BuildArtifact> {
        self.ensure_image_tagged(spec, &spec.image_tag())
    }

    /// Like [`ensure_image`], but for the export path: keys the image by the
    /// recipe's full baked content (`export_image_tag`), not just its build phase.
    /// Two recipes that share a build phase but bake different recipe.json must NOT
    /// collide on one self-describing image, since the payload-less managed run
    /// reads the baked document.
    pub fn ensure_export_image(&self, spec: &EffectiveSandboxSpec) -> Result<BuildArtifact> {
        self.ensure_image_tagged(spec, &spec.export_image_tag())
    }

    fn ensure_image_tagged(
        &self,
        spec: &EffectiveSandboxSpec,
        image_tag: &str,
    ) -> Result<BuildArtifact> {
        // A single `docker image inspect` answers both "does it exist" and
        // "what is its id": inspect_image_id returns Ok(None) when the image is
        // absent, so a second existence probe is redundant.
        if let Some(image_id) = inspect_image_id(image_tag)? {
            return Ok(BuildArtifact { image_id });
        }

        let tempdir = tempfile::tempdir().context("failed to create sandbox build context")?;
        self.write_build_context(spec, &tempdir)?;

        // Opt-in: when a build memory ceiling is requested, route the build
        // through a bounded `docker-container` buildx builder so an OOM is a
        // clean cgroup-local build failure instead of a host freeze. Wrapping the
        // `docker` CLI in a systemd scope does NOT work -- the build runs in the
        // daemon's own cgroup (system.slice), a sibling of the CLI, so only a
        // builder whose cgroup contains the work can bound it. Without a ceiling
        // we keep the default builder (today's behaviour).
        let build_memory = match spec.build_memory.as_deref() {
            Some(raw) => resolve_build_memory(raw)?,
            None => None,
        };
        let output = match &build_memory {
            Some(memory) => {
                let builder = ensure_bounded_builder(memory)?;
                Command::new("docker")
                    .args([
                        "buildx",
                        "build",
                        "--builder",
                        &builder,
                        "--load",
                        "-t",
                        image_tag,
                        ".",
                    ])
                    .current_dir(tempdir.path())
                    .output()
                    .context("failed to invoke docker buildx build")?
            }
            None => Command::new("docker")
                .args(["build", "-t", image_tag, "."])
                .current_dir(tempdir.path())
                .output()
                .context("failed to invoke docker build")?,
        };
        if !output.status.success() {
            bail!("docker build failed:\n{}", command_output(&output));
        }

        let image_id = inspect_image_id(image_tag)?
            .ok_or_else(|| anyhow!("docker build completed but image id was not found"))?;
        Ok(BuildArtifact { image_id })
    }

    /// Add an alias tag to an already-built image (`docker tag <source> <alias>`).
    /// Docker validates the alias reference and reports a clear error if it is
    /// malformed, so no separate validation is needed here.
    pub fn tag_image(&self, source: &str, alias: &str) -> Result<()> {
        let output = Command::new("docker")
            .args(["tag", source, alias])
            .output()
            .context("failed to invoke docker tag")?;
        if !output.status.success() {
            bail!("docker tag failed:\n{}", command_output(&output));
        }
        Ok(())
    }

    pub async fn run_container_async(
        &self,
        spec: &EffectiveSandboxSpec,
        artifact: &BuildArtifact,
        mounts: &[(String, String)],
    ) -> Result<StartedContainer> {
        let container_name = format!("rlmesh-sandbox-{}-{}", spec.slug(), Uuid::new_v4().simple());
        // The bootstrap payload carries runtime-only parameters (kwargs,
        // num_envs, vectorization_mode, and similar settings). It is delivered at `docker run`
        // time via an env var rather than baked into the image, so changing a
        // runtime parameter never rebuilds the image or invalidates the pip
        // install layer.
        let bootstrap_json = render_bootstrap_json(spec)?;
        let gpu = spec.recipe.as_ref().is_some_and(|recipe| recipe.build.gpu);
        // docker --mount parses comma-separated fields with no escape for a comma
        // inside a value, so a host/target path containing one would silently
        // corrupt the mount. Reject it loudly instead. ponytail: --mount CSV
        // ceiling; move to a long-form mount API if comma paths ever need support.
        for (host, target) in mounts {
            anyhow::ensure!(
                !host.contains(',') && !target.contains(','),
                "artifact mount path must not contain a comma (docker --mount cannot escape it): host={host:?} target={target:?}"
            );
        }
        let output = Command::new("docker")
            .args(docker_run_args(
                &container_name,
                &artifact.image_id,
                &bootstrap_json,
                std::process::id(),
                gpu,
                mounts,
            ))
            .output()
            .context("failed to start docker container")?;
        if !output.status.success() {
            // `docker run` can leave a created (but not started) container behind
            // when startup fails after the container is created (e.g. a port
            // collision). Remove it by name so we do not leak it.
            let _ = self.remove_container(&container_name);
            bail!("docker run failed:\n{}", command_output(&output));
        }

        let container_id = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let host_port = match resolve_published_port(&container_id) {
            Ok(port) => port,
            Err(err) => {
                let _ = self.stop_container(&container_id);
                return Err(err);
            }
        };
        let address = format!("tcp://127.0.0.1:{host_port}");
        let is_model = spec
            .recipe
            .as_ref()
            .is_some_and(|recipe| recipe.kind == "model");
        if let Err(err) =
            wait_for_ready(&address, &container_id, Duration::from_secs(30), is_model).await
        {
            let hint = spec.rlmesh_package.skew_hint();
            let report =
                self.startup_failure_report(&container_id, &container_name, &err, hint.as_deref());
            let _ = self.stop_container(&container_id);
            return Err(report);
        }

        Ok(StartedContainer {
            container_id,
            address,
        })
    }

    pub fn stop_container(&self, container_id: &str) -> Result<()> {
        let Some(state) = inspect_container_state(container_id)? else {
            return Ok(());
        };

        if state.status == "running" {
            self.stop_running_container(container_id)?;
        }
        self.remove_container(container_id)
    }

    /// Best-effort reap of orphaned rlmesh-owned containers.
    ///
    /// Only containers whose recorded owner process is gone are removed; current
    /// process containers are excluded and individual Docker failures are skipped.
    pub fn reap_orphaned_containers(&self) -> Result<Vec<String>> {
        let candidates = list_owned_containers()?;
        let self_pid = std::process::id();
        let self_pid_namespace = current_pid_namespace_id();
        let mut reaped = Vec::new();
        for candidate in candidates {
            // Never reap our own containers, even if a pid-reuse coincidence
            // were to make the liveness check ambiguous.
            if candidate.owner_pid == Some(self_pid) {
                continue;
            }
            let owner_liveness = candidate
                .owner_pid
                .map_or(OwnerPidLiveness::Unknown, |pid| {
                    owner_pid_liveness(
                        pid,
                        candidate.owner_pid_namespace.as_deref(),
                        self_pid_namespace.as_deref(),
                    )
                });
            if !is_orphan(&candidate.status, candidate.owner_pid, owner_liveness) {
                continue;
            }
            if self.stop_container(&candidate.id).is_ok() {
                reaped.push(candidate.id);
            }
        }
        Ok(reaped)
    }

    fn stop_running_container(&self, container_id: &str) -> Result<()> {
        let output = Command::new("docker")
            .args(["stop", container_id])
            .output()
            .context("failed to stop docker container")?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("No such container") {
                return Ok(());
            }
            bail!("docker stop failed: {}", stderr.trim());
        }
        Ok(())
    }

    fn remove_container(&self, container_id: &str) -> Result<()> {
        let output = Command::new("docker")
            .args(["rm", container_id])
            .output()
            .context("failed to remove docker container")?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("No such container") || stderr.contains("No such object") {
                return Ok(());
            }
            bail!("docker rm failed: {}", stderr.trim());
        }
        Ok(())
    }

    fn startup_failure_report(
        &self,
        container_id: &str,
        container_name: &str,
        cause: &anyhow::Error,
        hint: Option<&str>,
    ) -> anyhow::Error {
        let state = match inspect_container_state(container_id) {
            Ok(Some(state)) => state.summary(),
            Ok(None) => "container state: unavailable (container not found)".to_string(),
            Err(err) => format!("container state: unavailable ({err})"),
        };
        let logs = match self.container_logs(container_id) {
            Ok(logs) if logs.trim().is_empty() => "container logs: <empty>".to_string(),
            Ok(logs) => format!(
                "container logs:\n{}",
                tail_text(&logs, CONTAINER_LOG_TAIL_BYTES)
            ),
            Err(err) => format!("container logs: unavailable ({err})"),
        };

        anyhow!(format_startup_failure_report(
            container_id,
            container_name,
            &cause.to_string(),
            &state,
            &logs,
            hint,
        ))
    }

    fn container_logs(&self, container_id: &str) -> Result<String> {
        let output = Command::new("docker")
            .args(["logs", container_id])
            .output()
            .context("failed to read docker logs")?;
        let mut logs = String::new();
        logs.push_str(&String::from_utf8_lossy(&output.stdout));
        if !output.stderr.is_empty() {
            if !logs.is_empty() {
                logs.push('\n');
            }
            logs.push_str(&String::from_utf8_lossy(&output.stderr));
        }
        Ok(logs)
    }

    fn write_build_context(&self, spec: &EffectiveSandboxSpec, tempdir: &TempDir) -> Result<()> {
        if let Some(source_path) = spec.rlmesh_package.source_path() {
            let filename = source_path
                .file_name()
                .ok_or_else(|| anyhow!("RLMesh wheel path must have a filename"))?;
            let package_dir = tempdir.path().join("packages");
            fs::create_dir_all(&package_dir)
                .context("failed to create sandbox package build context")?;
            fs::copy(source_path, package_dir.join(filename)).with_context(|| {
                format!("failed to copy RLMesh wheel {}", source_path.display())
            })?;
        }

        if let ResolvedEnvironmentSourceRef::Hf(source) = &spec.resolved_source {
            let EnvironmentSourceRef::Hf(requested_source) = &spec.requested_source else {
                bail!("resolved HF source did not match requested source");
            };
            hf::materialize_source(
                requested_source,
                &source.resolved_revision,
                &tempdir.path().join("source"),
            )?;
        }

        // Stage the recipe author's ProjectInstall tree into the build context
        // under the dir the deriver COPYs from. Rejected for Remote provenance
        // upstream (validate_recipe_build) since there is no host tree to read.
        if let Some(recipe) = &spec.recipe
            && let Some(project) = &recipe.build.project
        {
            let root = spec.context_root.as_ref().ok_or_else(|| {
                anyhow!("recipe ProjectInstall requires a context_root to stage from")
            })?;
            let src = root.join(&project.src);
            let dest = tempdir.path().join(crate::recipe::PROJECT_CONTEXT_DIR);
            copy_tree(&src, &dest)
                .with_context(|| format!("failed to stage project tree {}", src.display()))?;

            // Also stage each `include` glob match (extra non-code assets, e.g.
            // `../assets/**` siblings above src), preserving its staged layout so
            // the single `COPY {PROJECT_CONTEXT_DIR}` carries them. The same
            // matches are folded into the build hash by recipe_content_digest.
            let includes = crate::recipe::resolve_includes(&src, root, &project.include)
                .with_context(|| {
                    format!("failed to resolve project includes under {}", src.display())
                })?;
            for include in includes {
                copy_tree(&include.path, &dest.join(&include.relative)).with_context(|| {
                    format!("failed to stage project include {}", include.path.display())
                })?;
            }
        }

        // Bake the recipe's runtime half as a Dockerfile sibling (NOT under
        // project/), so the derived `COPY recipe.json` makes the image
        // self-describing without the recipe content entering project.src's digest.
        if let Some(recipe) = &spec.recipe {
            let json = serde_json::to_string(&runtime_document(recipe)?)
                .context("failed to serialize baked recipe.json")?;
            fs::write(tempdir.path().join("recipe.json"), json)
                .context("failed to stage baked recipe.json")?;
        }

        let dockerfile = render_dockerfile(spec)?;
        fs::write(tempdir.path().join("Dockerfile"), dockerfile)
            .context("failed to write generated Dockerfile")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ContainerState {
    status: String,
    exit_code: Option<i32>,
    error: String,
}

impl ContainerState {
    fn is_terminal(&self) -> bool {
        matches!(self.status.as_str(), "dead" | "exited")
    }

    fn summary(&self) -> String {
        let exit_code = self
            .exit_code
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        if self.error.trim().is_empty() {
            format!(
                "container state: status={}, exit_code={exit_code}",
                self.status
            )
        } else {
            format!(
                "container state: status={}, exit_code={exit_code}, error={}",
                self.status, self.error
            )
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct BootstrapConfigFile {
    spec: BootstrapSpec,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BootstrapSpec {
    Gym(GymBootstrapSpec),
    Hf(HfBootstrapSpec),
    Recipe(RecipeBootstrapSpec),
}

impl BootstrapSpec {
    fn from_effective_spec(spec: &EffectiveSandboxSpec) -> Result<Self> {
        Ok(match &spec.resolved_source {
            ResolvedEnvironmentSourceRef::Gym(source) => Self::Gym(GymBootstrapSpec {
                env_id: source.env_id.clone(),
                imports: spec.imports.clone(),
                kwargs: spec.kwargs.clone(),
                num_envs: spec.num_envs,
                vectorization_mode: spec.vectorization_mode.as_str().to_string(),
            }),
            ResolvedEnvironmentSourceRef::Hf(source) => Self::Hf(HfBootstrapSpec {
                source_subdir: "source".to_string(),
                suite: source.suite.clone(),
                task: source.task.clone(),
                imports: spec.imports.clone(),
                kwargs: spec.kwargs.clone(),
                num_envs: spec.num_envs,
                vectorization_mode: spec.vectorization_mode.as_str().to_string(),
            }),
            ResolvedEnvironmentSourceRef::Recipe(_) => {
                let recipe = spec
                    .recipe
                    .as_ref()
                    .ok_or_else(|| anyhow!("recipe source missing its parsed recipe"))?;
                Self::Recipe(RecipeBootstrapSpec {
                    // Only the runtime phase (setup/make/requires/adapter)
                    // rides the payload; the build phase already shaped the image
                    // and must not re-ship.
                    document: runtime_document(recipe)?,
                    num_envs: spec.num_envs,
                    vectorization_mode: spec.vectorization_mode.as_str().to_string(),
                })
            }
        })
    }
}

/// Serialize a recipe's runtime half (build phase stripped) for the bootstrap
/// payload. The container has already been built, so re-shipping the build keys
/// would only bloat the payload.
pub(crate) fn runtime_document(recipe: &crate::recipe::Recipe) -> Result<serde_json::Value> {
    let mut value =
        serde_json::to_value(recipe).context("failed to serialize recipe bootstrap document")?;
    if let Some(object) = value.as_object_mut() {
        object.remove("build");
    }
    Ok(value)
}

#[derive(Debug, Clone, Serialize)]
struct GymBootstrapSpec {
    env_id: String,
    imports: Vec<String>,
    kwargs: std::collections::BTreeMap<String, serde_json::Value>,
    num_envs: usize,
    vectorization_mode: String,
}

#[derive(Debug, Clone, Serialize)]
struct HfBootstrapSpec {
    source_subdir: String,
    suite: Option<String>,
    task: Option<String>,
    imports: Vec<String>,
    kwargs: std::collections::BTreeMap<String, serde_json::Value>,
    num_envs: usize,
    vectorization_mode: String,
}

#[derive(Debug, Clone, Serialize)]
struct RecipeBootstrapSpec {
    /// The recipe's runtime half (setup/make/requires/adapter); the build
    /// phase is stripped because it already shaped the image.
    document: serde_json::Value,
    num_envs: usize,
    vectorization_mode: String,
}

async fn wait_for_ready(
    address: &str,
    container_id: &str,
    timeout: Duration,
    is_model: bool,
) -> Result<()> {
    let deadline = Instant::now() + timeout;

    loop {
        let probe = tokio::time::timeout(READY_PROBE_TIMEOUT, async {
            // A model container serves ModelService; an env container serves
            // EnvService. Probe with the matching handshake so readiness reflects
            // the service the recipe's kind actually brings up.
            if is_model {
                let mut client = ModelClient::connect(address, "").await?;
                client.handshake().await?;
            } else {
                let mut client = EnvClient::connect(address).await?;
                client.handshake().await?;
            }
            Ok::<_, rlmesh_grpc::error::Error>(())
        })
        .await;

        match probe {
            Ok(Ok(())) => return Ok(()),
            Ok(Err(err)) => {
                if let Some(state) = container_terminated(container_id) {
                    bail!("sandbox container exited before ready ({state})");
                }
                if err.is_fatal_handshake() || Instant::now() >= deadline {
                    return Err(err.into());
                }
                tokio::time::sleep(Duration::from_millis(300)).await;
            }
            Err(_) if Instant::now() < deadline => {
                if let Some(state) = container_terminated(container_id) {
                    bail!("sandbox container exited before ready ({state})");
                }
                tokio::time::sleep(Duration::from_millis(300)).await;
            }
            Err(_) => bail!(
                "sandbox container did not respond within {} seconds",
                timeout.as_secs()
            ),
        }
    }
}

/// Whether the container is *confirmed* to have reached a terminal state.
///
/// A transient `docker inspect` failure (e.g. the daemon is briefly busy) is
/// treated as "not confirmed terminal" so the readiness wait keeps polling
/// until its own deadline, rather than tearing down a container that is about
/// to become ready. Only an inspect that succeeds and reports a terminal
/// status returns `true`.
/// When the container is confirmed terminal, returns its state summary for
/// the error message; None while it is still running or not yet inspectable.
fn container_terminated(container_id: &str) -> Option<String> {
    confirmed_terminal_summary(inspect_container_state(container_id))
}

/// Decide whether an inspect result confirms the container is terminal. A
/// transient inspect error or a not-yet-found container is treated as "not
/// confirmed", so the readiness wait keeps polling instead of aborting.
fn confirmed_terminal_summary(inspected: Result<Option<ContainerState>>) -> Option<String> {
    match inspected {
        Ok(Some(state)) if state.is_terminal() => Some(state.summary()),
        _ => None,
    }
}

fn render_dockerfile(spec: &EffectiveSandboxSpec) -> Result<String> {
    validate_dockerfile_token("base_image", &spec.base_image)?;

    // A recipe source drives the Dockerfile through the language-neutral deriver
    // (the §5A contract); gym/hf use the fixed preamble below. The deriver is fed
    // the upstream-resolved base image, rlmesh package, and extra packages (the
    // same values the build hash covers) rather than re-deriving them, plus the
    // recipe's provenance so it can skip the implicit unpinned gymnasium for a
    // Remote recipe (which the reproducibility gate forces fully-pinned).
    if let Some(recipe) = &spec.recipe {
        let provenance = match &spec.requested_source {
            EnvironmentSourceRef::Recipe(reference) => reference.provenance,
            // gym/hf never reach the deriver (no recipe), but a recipe spec must
            // carry a Recipe source; treat any other source as the trusted,
            // gymnasium-injecting path rather than silently dropping it.
            _ => crate::RecipeProvenance::Installed,
        };
        return crate::recipe::derive_dockerfile(
            recipe,
            &spec.base_image,
            &spec.rlmesh_package,
            &spec.packages,
            provenance,
        )
        .map_err(|err| anyhow!("failed to derive recipe Dockerfile: {err}"));
    }

    let source_copy = match &spec.resolved_source {
        ResolvedEnvironmentSourceRef::Gym(_) => "",
        ResolvedEnvironmentSourceRef::Hf(_) => "COPY source /opt/rlmesh/source\n",
        ResolvedEnvironmentSourceRef::Recipe(_) => "",
    };
    let package_copy = if spec.rlmesh_package.source_path().is_some() {
        "COPY packages /opt/rlmesh/packages\n"
    } else {
        ""
    };
    let package_command = render_package_install_command(spec);

    // The bootstrap payload is supplied at run time via the
    // RLMESH_BOOTSTRAP_JSON env var (see docker_run_args), not COPY'd into the
    // image, so runtime-only parameters never invalidate the image cache.
    Ok(format!(
        "# syntax=docker/dockerfile:1.7\n\n\
FROM {}\n\n\
ENV RLMESH_PORT={DEFAULT_CONTAINER_PORT}\n\
ENV RLMESH_ENV_PORT={DEFAULT_CONTAINER_PORT}\n\
ENV PYTHONUNBUFFERED=1\n\n\
WORKDIR /opt/rlmesh\n\
{}\
{}\
\n\
RUN sh -lc {}\n\n\
EXPOSE {DEFAULT_CONTAINER_PORT}\n\
{}\n",
        spec.base_image,
        source_copy,
        package_copy,
        shell_quote(&package_command),
        crate::recipe::entrypoint_for("env"),
    ))
}

/// Serialize the bootstrap payload that is injected at run time via the
/// `RLMESH_BOOTSTRAP_JSON` env var.
fn render_bootstrap_json(spec: &EffectiveSandboxSpec) -> Result<String> {
    serde_json::to_string(&BootstrapConfigFile {
        spec: BootstrapSpec::from_effective_spec(spec)?,
    })
    .context("failed to serialize sandbox bootstrap payload")
}

fn render_package_install_command(spec: &EffectiveSandboxSpec) -> String {
    let mut parts = vec![
        "python -m pip install --no-cache-dir --upgrade pip".to_string(),
        format!(
            "python -m pip install --no-cache-dir {}",
            shell_quote(spec.rlmesh_package.install_ref())
        ),
        "python -m pip install --no-cache-dir gymnasium".to_string(),
    ];

    if matches!(&spec.resolved_source, ResolvedEnvironmentSourceRef::Hf(_)) {
        parts.push(
            "if [ -f /opt/rlmesh/source/requirements.txt ]; then python -m pip install --no-cache-dir -r /opt/rlmesh/source/requirements.txt; fi"
                .to_string(),
        );
    }

    if !spec.packages.is_empty() {
        let package_args = spec
            .packages
            .iter()
            .map(|package| shell_quote(package))
            .collect::<Vec<_>>()
            .join(" ");
        parts.push(format!(
            "python -m pip install --no-cache-dir {package_args}"
        ));
    }

    parts.join(" && ")
}

fn validate_dockerfile_token(label: &str, value: &str) -> Result<()> {
    anyhow::ensure!(
        !value.contains('\n') && !value.contains('\r'),
        "{label} must not contain newlines"
    );
    Ok(())
}

fn command_output(output: &std::process::Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = stdout.trim();
    let stderr = stderr.trim();

    let mut sections = Vec::new();
    if !stdout.is_empty() {
        sections.push(format!("stdout:\n{stdout}"));
    }
    if !stderr.is_empty() {
        sections.push(format!("stderr:\n{stderr}"));
    }
    if sections.is_empty() {
        sections.push(format!("exit status: {}", output.status));
    }
    sections.join("\n")
}

/// Fixed cushion left for the already-running host (desktop + services) when
/// `build_memory="auto"` computes the ceiling. The box is never idle, so a
/// percentage cap would still co-exhaust with the desktop -- the trap that froze
/// the host. Leave a flat headroom instead.
const BUILD_MEMORY_HEADROOM: u64 = 8 * 1024 * 1024 * 1024;
/// Floor so a small box still gets a usable (if tight) build ceiling.
const BUILD_MEMORY_FLOOR: u64 = 4 * 1024 * 1024 * 1024;

/// Interpret a requested `build_memory` value into a concrete `--driver-opt
/// memory=` value, or `None` to use the default builder. `"auto"` sizes a
/// ceiling from host RAM; `"off"`/`"none"`/`"unbounded"`/empty opt out; a docker
/// size string (e.g. `"20g"`) passes through. A non-size string (typo) errors up
/// front rather than failing confusingly inside `docker buildx create`.
fn resolve_build_memory(raw: &str) -> Result<Option<String>> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "off" | "none" | "unbounded" => Ok(None),
        "auto" => Ok(Some(auto_build_memory())),
        _ => {
            let value = raw.trim();
            if is_docker_size(value) {
                Ok(Some(value.to_string()))
            } else {
                bail!(
                    "build_memory must be a docker size like \"20g\", or \"auto\"/\"off\"; got {value:?}"
                )
            }
        }
    }
}

/// Whether `value` is a docker memory size: a positive decimal with an optional
/// `b`/`k`/`m`/`g`/`t` unit (case-insensitive, optional trailing `b`). Gates the
/// passthrough above so a typo can't produce a degenerate builder name or a
/// late, opaque docker error.
fn is_docker_size(value: &str) -> bool {
    let lower = value.trim().to_ascii_lowercase();
    let without_b = lower.strip_suffix('b').unwrap_or(&lower);
    let number = without_b
        .strip_suffix(['k', 'm', 'g', 't'])
        .unwrap_or(without_b);
    !number.is_empty() && number.parse::<f64>().is_ok_and(|n| n > 0.0)
}

fn auto_build_memory() -> String {
    let total = host_total_ram_bytes().unwrap_or(BUILD_MEMORY_FLOOR + BUILD_MEMORY_HEADROOM);
    bounded_build_memory(total).to_string()
}

fn bounded_build_memory(total_ram: u64) -> u64 {
    total_ram
        .saturating_sub(BUILD_MEMORY_HEADROOM)
        .max(BUILD_MEMORY_FLOOR)
}

/// Total physical RAM in bytes: `/proc/meminfo` on Linux, else `sysctl
/// hw.memsize` on macOS. `None` when neither is readable.
///
/// macOS caveat: `hw.memsize` is HOST RAM, but a docker build runs in a Linux VM
/// whose (smaller) RAM is the real ceiling, so an `"auto"` cap here can land
/// above the VM size. That is safe -- the VM already contains an OOM, so the
/// host can't freeze and an over-high cap is merely inert. On macOS prefer an
/// explicit `build_memory` (or raise the VM's memory) over `"auto"`.
fn host_total_ram_bytes() -> Option<u64> {
    proc_meminfo_total().or_else(sysctl_memsize)
}

fn proc_meminfo_total() -> Option<u64> {
    fs::read_to_string("/proc/meminfo")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|kb| kb.parse::<u64>().ok())
        .map(|kb| kb * 1024)
}

fn sysctl_memsize() -> Option<u64> {
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout).trim().parse().ok()
}

/// The per-ceiling builder name, e.g. `rlmesh-build-20g`. Encoding the cap in
/// the name keeps a re-requested ceiling stable (reused builder, warm cache)
/// while a changed ceiling lands a fresh builder. Old ones are stopped
/// containers costing only disk; clean up with `docker buildx rm`.
fn builder_name(memory: &str) -> String {
    let suffix: String = memory
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    format!("rlmesh-build-{suffix}")
}

/// Inspect-or-create a bounded `docker-container` buildx builder. `memory-swap
/// == memory` disables swap inside the builder so the cap is a hard wall (prompt
/// cgroup OOM-kill, no thrash). Returns the builder name to build through.
fn ensure_bounded_builder(memory: &str) -> Result<String> {
    let name = builder_name(memory);
    if builder_exists(&name)? {
        return Ok(name);
    }
    let output = Command::new("docker")
        .args([
            "buildx",
            "create",
            "--name",
            &name,
            "--driver",
            "docker-container",
            "--driver-opt",
            &format!("memory={memory}"),
            "--driver-opt",
            &format!("memory-swap={memory}"),
        ])
        .output()
        .context("failed to create bounded docker buildx builder")?;
    if !output.status.success() {
        // A concurrent build may have created the builder between our inspect
        // and create. Re-inspect rather than parse docker's version-varying
        // error text: if it now exists, the race resolved in our favour.
        if builder_exists(&name)? {
            return Ok(name);
        }
        bail!(
            "failed to create bounded docker buildx builder:\n{}",
            command_output(&output)
        );
    }
    Ok(name)
}

fn builder_exists(name: &str) -> Result<bool> {
    Ok(Command::new("docker")
        .args(["buildx", "inspect", name])
        .output()
        .context("failed to inspect docker buildx builder")?
        .status
        .success())
}

fn format_startup_failure_report(
    container_id: &str,
    container_name: &str,
    cause: &str,
    state: &str,
    logs: &str,
    hint: Option<&str>,
) -> String {
    let mut report = format!(
        "sandbox container did not become ready: {cause}\ncontainer id: {container_id}\ncontainer name: {container_name}\n{state}\n{logs}"
    );
    if let Some(hint) = hint {
        report.push_str("\nhint: ");
        report.push_str(hint);
    }
    report
}

/// Recursively copy a file or directory tree from `src` to `dest`.
///
/// Symlink children are skipped for safety and consistency. A child entry is
/// classified by its own `entry.file_type()`, which reports the link itself
/// without dereferencing or erroring on a dangling/looping target. The symlink
/// is not copied or recursed into, which keeps an outward link from leaking
/// foreign bytes into the image and stops cyclic/dangling links from aborting the
/// build. `fs::metadata` is used only to classify the passed-in root, so an
/// explicitly named `src` that is itself a symlink-to-dir is still honored; every
/// child is filtered before recursion. `hash_path_tree` in lib.rs makes the same
/// skip decision so the staged bytes equal the hashed bytes.
/// Linked/out-of-tree assets are carried explicitly via `ProjectInstall::include`
/// (a guarded, canonicalized glob), not by silently following links here.
fn copy_tree(src: &std::path::Path, dest: &std::path::Path) -> Result<()> {
    let metadata =
        fs::metadata(src).with_context(|| format!("failed to stat {}", src.display()))?;
    if metadata.is_dir() {
        fs::create_dir_all(dest).with_context(|| format!("failed to create {}", dest.display()))?;
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            // Skip symlink children identically to hash_path_tree: file_type()
            // reports the link itself, so a dangling/cyclic/escaping link is
            // skipped without ever dereferencing it.
            if entry.file_type()?.is_symlink() {
                continue;
            }
            copy_tree(&entry.path(), &dest.join(entry.file_name()))?;
        }
    } else {
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(src, dest)
            .with_context(|| format!("failed to copy {} to {}", src.display(), dest.display()))?;
    }
    Ok(())
}

fn docker_run_args(
    container_name: &str,
    image_id: &str,
    bootstrap_json: &str,
    owner_pid: u32,
    gpu: bool,
    mounts: &[(String, String)],
) -> Vec<String> {
    let mut args = vec![
        "run".to_string(),
        "-d".to_string(),
        "--cap-drop".to_string(),
        "ALL".to_string(),
        "--security-opt".to_string(),
        "no-new-privileges".to_string(),
        // Mark the container as rlmesh-owned so orphans can be reaped.
        "--label".to_string(),
        OWNER_LABEL.to_string(),
        // Record the owning process so the reaper can tell a live owner's
        // container apart from an orphan: only containers whose owner process
        // is gone are reaped.
        "--label".to_string(),
        format!("{OWNER_PID_LABEL_KEY}={owner_pid}"),
    ];
    if let Some(pid_namespace) = current_pid_namespace_id() {
        args.extend([
            "--label".to_string(),
            format!("{OWNER_PID_NS_LABEL_KEY}={pid_namespace}"),
        ]);
    }
    if gpu {
        // GPU access is via the nvidia runtime, not Linux capabilities, so it
        // coexists with the --cap-drop ALL / no-new-privileges hardening above.
        args.extend([
            "--gpus".to_string(),
            "all".to_string(),
            "--env".to_string(),
            "NVIDIA_VISIBLE_DEVICES=all".to_string(),
        ]);
    }
    // Runtime artifact mounts: a declared input's resolved host directory bound
    // read-only at its in-container target. Read-only keeps the container from
    // mutating the host weights and coexists with the hardening above.
    for (host, target) in mounts {
        args.push("--mount".to_string());
        args.push(format!("type=bind,source={host},target={target},readonly"));
    }
    args.extend([
        // Deliver the bootstrap payload (runtime-only parameters) at run time
        // so changing it never rebuilds the image.
        "--env".to_string(),
        format!("RLMESH_BOOTSTRAP_JSON={bootstrap_json}"),
        "--name".to_string(),
        container_name.to_string(),
        // Let docker pick a free host port (binding it atomically) instead of
        // reserving one ourselves and racing between releasing it and `docker
        // run`. The assigned port is read back via `docker port`.
        "-p".to_string(),
        format!("127.0.0.1:0:{DEFAULT_CONTAINER_PORT}"),
        image_id.to_string(),
    ]);
    args
}

/// Read back the host port docker published for the container's gRPC port.
fn resolve_published_port(container_id: &str) -> Result<u16> {
    let output = Command::new("docker")
        .args(["port", container_id, &DEFAULT_CONTAINER_PORT.to_string()])
        .output()
        .context("failed to read published docker port")?;
    if !output.status.success() {
        bail!(
            "docker port failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_published_port(&stdout).ok_or_else(|| {
        anyhow!("could not parse published port from docker port output: {stdout:?}")
    })
}

/// Parse the host port from `docker port` output lines like
/// `127.0.0.1:49153` or `0.0.0.0:49153` (one mapping per line).
fn parse_published_port(raw: &str) -> Option<u16> {
    raw.lines()
        .filter_map(|line| line.trim().rsplit_once(':'))
        .find_map(|(_, port)| port.trim().parse::<u16>().ok())
}

/// A rlmesh-owned container candidate for reaping: its id, the recorded owner
/// process id (absent for legacy containers), and its docker state word.
#[derive(Debug, Clone, PartialEq, Eq)]
struct OwnedContainer {
    id: String,
    owner_pid: Option<u32>,
    owner_pid_namespace: Option<String>,
    status: String,
}

/// List every rlmesh-owned container (running or stopped), together with its
/// recorded owner pid and state, by filtering on the `rlmesh.sandbox` label.
fn list_owned_containers() -> Result<Vec<OwnedContainer>> {
    let output = Command::new("docker")
        .args([
            "ps",
            "--all",
            "--no-trunc",
            "--filter",
            OWNER_LABEL_FILTER,
            "--format",
            REAP_PS_FORMAT,
        ])
        .output()
        .context("failed to list rlmesh sandbox containers")?;
    if !output.status.success() {
        bail!(
            "docker ps failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(parse_owned_containers(&String::from_utf8_lossy(
        &output.stdout,
    )))
}

/// Parse pipe-delimited `docker ps --format` output (see [`REAP_PS_FORMAT`])
/// into [`OwnedContainer`] rows. Lines without an id and a non-empty state are
/// skipped; an empty or non-numeric owner-pid field (legacy containers) parses
/// to `None`.
fn parse_owned_containers(raw: &str) -> Vec<OwnedContainer> {
    raw.lines()
        .filter_map(|line| {
            let mut fields = line.split('|');
            let id = fields.next()?.trim();
            let owner_pid = fields.next().unwrap_or("").trim();
            let owner_pid_namespace_or_status = fields.next().unwrap_or("").trim();
            let status = fields.next();
            let (owner_pid_namespace, status) = match status {
                Some(status) => (
                    (!owner_pid_namespace_or_status.is_empty())
                        .then(|| owner_pid_namespace_or_status.to_string()),
                    status.trim(),
                ),
                None => (None, owner_pid_namespace_or_status),
            };
            if id.is_empty() || status.is_empty() {
                return None;
            }
            Some(OwnedContainer {
                id: id.to_string(),
                owner_pid: owner_pid.parse::<u32>().ok(),
                owner_pid_namespace,
                status: status.to_string(),
            })
        })
        .collect()
}

/// Decide whether a rlmesh-owned container is an orphan that may be reaped.
///
/// This is the pure core of the reaper, factored out so it can be tested
/// without Docker. Inputs are the container's docker state word, its recorded
/// owner pid (`None` for legacy containers that predate per-process labeling),
/// and whether that pid is currently alive.
///
/// Rules:
/// - A container whose owner process is still alive is never an orphan (this is
///   the live-session case the reaper must not disturb).
/// - A container whose recorded owner process is gone is an orphan regardless
///   of state (running leftovers from a hard-killed owner are exactly what we
///   reap).
/// - A legacy container *without* an owner-pid label is treated as an orphan
///   only when it is not running, since we cannot prove an owner is gone; a
///   running unlabeled container is left alone to avoid killing a live session
///   started by an older rlmesh build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OwnerPidLiveness {
    Alive,
    Dead,
    Unknown,
}

fn is_orphan(status: &str, owner_pid: Option<u32>, owner_liveness: OwnerPidLiveness) -> bool {
    match owner_pid {
        Some(_) => owner_liveness == OwnerPidLiveness::Dead,
        None => !is_running_status(status),
    }
}

/// Whether a docker state word denotes a running container.
fn is_running_status(status: &str) -> bool {
    status.eq_ignore_ascii_case("running")
}

fn owner_pid_liveness(
    pid: u32,
    owner_pid_namespace: Option<&str>,
    self_pid_namespace: Option<&str>,
) -> OwnerPidLiveness {
    match (owner_pid_namespace, self_pid_namespace) {
        (Some(owner), Some(current)) if owner == current => {
            if pid_is_alive(pid) {
                OwnerPidLiveness::Alive
            } else {
                OwnerPidLiveness::Dead
            }
        }
        _ => OwnerPidLiveness::Unknown,
    }
}

fn current_pid_namespace_id() -> Option<String> {
    fs::read_link("/proc/self/ns/pid")
        .ok()
        .map(|path| path.to_string_lossy().into_owned())
}

/// Best-effort liveness check for a process id, std-only via `/proc/<pid>`
/// existence (Linux). This carries an inherent pid-reuse race: if the original
/// owner died and the OS recycled its pid for an unrelated process, the
/// container is mistaken for live and skipped this sweep (it will be reaped on
/// a later sweep once the recycled pid also exits). That residual race is
/// acceptable: erring toward *not* reaping is the safe direction, since the
/// cost is a leaked container, not a killed live session.
fn pid_is_alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

fn inspect_container_state(container_id: &str) -> Result<Option<ContainerState>> {
    let output = Command::new("docker")
        .args([
            "inspect",
            "--format",
            "{{.State.Status}}\n{{.State.ExitCode}}\n{{.State.Error}}",
            container_id,
        ])
        .output()
        .context("failed to inspect docker container")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No such container") || stderr.contains("No such object") {
            return Ok(None);
        }
        bail!("docker inspect failed: {}", stderr.trim());
    }
    parse_container_state(&String::from_utf8_lossy(&output.stdout)).map(Some)
}

fn parse_container_state(raw: &str) -> Result<ContainerState> {
    let mut lines = raw.lines();
    let status = lines.next().unwrap_or_default().trim().to_string();
    if status.is_empty() {
        bail!("docker inspect did not report container status");
    }
    let exit_code = lines
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::parse::<i32>)
        .transpose()
        .context("docker inspect reported invalid container exit code")?;
    let error = lines.collect::<Vec<_>>().join("\n").trim().to_string();
    Ok(ContainerState {
        status,
        exit_code,
        error,
    })
}

fn tail_text(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }

    let mut start = value.len() - max_bytes;
    while !value.is_char_boundary(start) {
        start += 1;
    }
    format!(
        "[truncated to last {max_bytes} bytes]\n{}",
        value[start..].trim_start()
    )
}

fn inspect_image_id(image_ref: &str) -> Result<Option<String>> {
    let output = Command::new("docker")
        .args(["image", "inspect", "--format", "{{.Id}}", image_ref])
        .output()
        .context("failed to inspect docker image id")?;
    if !output.status.success() {
        return Ok(None);
    }
    Ok(Some(
        String::from_utf8_lossy(&output.stdout).trim().to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use anyhow::anyhow;
    use serde_json::json;

    use super::{
        BootstrapSpec, ContainerState, OwnedContainer, OwnerPidLiveness, bounded_build_memory,
        builder_name, confirmed_terminal_summary, copy_tree, current_pid_namespace_id,
        docker_run_args, format_startup_failure_report, is_docker_size, is_orphan,
        owner_pid_liveness, parse_container_state, parse_owned_containers, parse_published_port,
        pid_is_alive, render_bootstrap_json, render_dockerfile, resolve_build_memory, shell_quote,
        tail_text,
    };
    use crate::source::{ResolvedEnvironmentSourceRef, ResolvedHfSourceRef};
    use crate::{
        EffectiveSandboxSpec, EnvironmentSourceRef, GymSourceRef, ResolvedRlmeshPackage,
        VectorizationMode,
    };

    fn pip_rlmesh_package() -> ResolvedRlmeshPackage {
        ResolvedRlmeshPackage::Pip {
            spec: "rlmesh==0.1.0b2".to_string(),
        }
    }

    /// A baseline CartPole gym spec; tests override only the fields they exercise
    /// via `..gym_spec()`.
    fn gym_spec() -> EffectiveSandboxSpec {
        EffectiveSandboxSpec {
            schema_version: crate::BOOTSTRAP_SCHEMA_VERSION,
            requested_source: EnvironmentSourceRef::parse("CartPole-v1").unwrap(),
            resolved_source: ResolvedEnvironmentSourceRef::Gym(GymSourceRef {
                env_id: "CartPole-v1".to_string(),
            }),
            base_image: "python:3.12-slim".to_string(),
            rlmesh_package: pip_rlmesh_package(),
            packages: vec![],
            imports: vec![],
            kwargs: BTreeMap::new(),
            num_envs: 1,
            vectorization_mode: VectorizationMode::Sync,
            recipe: None,
            context_root: None,
            build_memory: None,
            build_hash: "abcdef0123456789".to_string(),
        }
    }

    #[test]
    fn dockerfile_installs_rlmesh_gymnasium_and_packages() {
        let spec = EffectiveSandboxSpec {
            packages: vec!["pygame".to_string()],
            ..gym_spec()
        };

        let dockerfile = render_dockerfile(&spec).unwrap();

        assert!(dockerfile.contains("FROM python:3.12-slim"));
        assert!(dockerfile.contains("rlmesh==0.1.0b2"));
        assert!(dockerfile.contains("gymnasium"));
        assert!(dockerfile.contains("pygame"));
        assert!(dockerfile.contains("rlmesh._bootstrap.sandbox_env"));
        assert!(!dockerfile.contains("COPY source"));
        // The bootstrap payload is delivered at run time, not baked into the
        // image, so runtime-only parameter changes never invalidate the build.
        assert!(!dockerfile.contains("COPY bootstrap.json"));
        assert!(!dockerfile.contains("bootstrap.json"));
    }

    #[test]
    fn build_memory_resolution_ceiling_and_builder_name() {
        const GIB: u64 = 1024 * 1024 * 1024;
        // Headroom math (the trap that froze the host): leave 8 GiB for the
        // loaded desktop; clamp small boxes up to the 4 GiB floor, never 0.
        assert_eq!(bounded_build_memory(32 * GIB), 24 * GIB);
        assert_eq!(bounded_build_memory(8 * GIB), 4 * GIB);
        assert_eq!(bounded_build_memory(0), 4 * GIB);

        // Opt-out keywords -> no builder; a size string passes through; "auto"
        // expands to a concrete byte count.
        assert_eq!(resolve_build_memory("off").unwrap(), None);
        assert_eq!(resolve_build_memory("").unwrap(), None);
        assert_eq!(resolve_build_memory("20g").unwrap().as_deref(), Some("20g"));
        assert!(
            resolve_build_memory("auto")
                .unwrap()
                .is_some_and(|m| m.parse::<u64>().is_ok())
        );
        // A typo'd size is rejected up front, not handed to docker.
        for bad in ["garbage", ";;;", "20 g", "-5g", "0"] {
            assert!(
                resolve_build_memory(bad).is_err(),
                "{bad} should be rejected"
            );
        }

        // is_docker_size accepts the forms docker accepts and nothing else.
        assert!(
            ["20g", "2048m", "1.5g", "1073741824", "512k", "1t"]
                .iter()
                .all(|s| is_docker_size(s))
        );
        assert!(
            !["", "g", "0", "-5g", "garbage", "20 g", "b"]
                .iter()
                .any(|s| is_docker_size(s))
        );

        // Builder name encodes the cap and is a valid buildx name.
        assert_eq!(builder_name("20g"), "rlmesh-build-20g");
        assert_eq!(builder_name("25769803776"), "rlmesh-build-25769803776");
    }

    #[test]
    fn docker_run_args_do_not_auto_remove_container() {
        let args = docker_run_args("rlmesh-sandbox-test", "sha256:abc", "{}", 4242, false, &[]);

        assert_eq!(args.first().map(String::as_str), Some("run"));
        assert!(args.iter().any(|arg| arg == "-d"));
        assert!(!args.iter().any(|arg| arg == "--rm"));
        assert!(args.iter().any(|arg| arg == "--cap-drop"));
        assert!(args.iter().any(|arg| arg == "no-new-privileges"));
        assert!(args.iter().any(|arg| arg == "rlmesh-sandbox-test"));
    }

    #[test]
    fn docker_run_args_publish_ephemeral_host_port() {
        let args = docker_run_args("rlmesh-sandbox-test", "sha256:abc", "{}", 4242, false, &[]);

        // Docker assigns the host port atomically; we must not bake a fixed one in.
        assert!(args.iter().any(|arg| arg == "127.0.0.1:0:50051"));
        assert!(!args.iter().any(|arg| arg.starts_with("127.0.0.1:4")));
    }

    #[test]
    fn docker_run_args_label_container_for_reaping() {
        let args = docker_run_args("rlmesh-sandbox-test", "sha256:abc", "{}", 4242, false, &[]);

        let label_idx = args.iter().position(|arg| arg == "--label");
        assert!(label_idx.is_some(), "containers must carry an owner label");
        assert_eq!(
            args.get(label_idx.unwrap() + 1).map(String::as_str),
            Some("rlmesh.sandbox=1")
        );
    }

    #[test]
    fn docker_run_args_stamp_owner_pid_label() {
        let args = docker_run_args("rlmesh-sandbox-test", "sha256:abc", "{}", 4242, false, &[]);

        // The owner-pid label must be present so the reaper can tell a live
        // owner's container apart from an orphan.
        assert!(
            args.iter()
                .any(|arg| arg == "rlmesh.sandbox.owner-pid=4242"),
            "containers must record their owner pid: {args:?}"
        );
    }

    #[test]
    fn docker_run_args_stamp_owner_pid_namespace_when_available() {
        let Some(pid_namespace) = current_pid_namespace_id() else {
            return;
        };
        let args = docker_run_args("rlmesh-sandbox-test", "sha256:abc", "{}", 4242, false, &[]);

        assert!(
            args.iter()
                .any(|arg| arg == &format!("rlmesh.sandbox.owner-pid-ns={pid_namespace}")),
            "containers must record their owner pid namespace when available: {args:?}"
        );
    }

    #[test]
    fn docker_run_args_inject_bootstrap_payload_at_run_time() {
        let args = docker_run_args(
            "rlmesh-sandbox-test",
            "sha256:abc",
            "{\"spec\":{\"kind\":\"gym\"}}",
            4242,
            false,
            &[],
        );

        let env_idx = args.iter().position(|arg| arg == "--env");
        assert!(env_idx.is_some(), "bootstrap must be passed via --env");
        assert_eq!(
            args.get(env_idx.unwrap() + 1).map(String::as_str),
            Some("RLMESH_BOOTSTRAP_JSON={\"spec\":{\"kind\":\"gym\"}}")
        );
    }

    #[test]
    fn docker_run_args_bind_mount_artifacts_read_only() {
        let mounts = vec![(
            "/host/weights".to_string(),
            "/rlmesh/input/model/weights".to_string(),
        )];
        let args = docker_run_args("n", "img", "{}", 1, false, &mounts);

        let idx = args
            .iter()
            .position(|arg| arg == "--mount")
            .expect("a declared mount must emit --mount");
        assert_eq!(
            args[idx + 1],
            "type=bind,source=/host/weights,target=/rlmesh/input/model/weights,readonly"
        );
    }

    #[test]
    fn docker_run_args_emit_no_mount_flag_without_artifacts() {
        let args = docker_run_args("n", "img", "{}", 1, false, &[]);
        assert!(
            !args.iter().any(|arg| arg == "--mount"),
            "no declared mounts must emit no --mount flag (gym/hf paths unchanged)"
        );
    }

    #[test]
    fn render_bootstrap_json_carries_runtime_params() {
        let mut kwargs = BTreeMap::new();
        kwargs.insert("render_mode".to_string(), json!("rgb_array"));
        let spec = EffectiveSandboxSpec {
            kwargs,
            num_envs: 4,
            vectorization_mode: VectorizationMode::Async,
            ..gym_spec()
        };

        let json = render_bootstrap_json(&spec).unwrap();
        assert!(json.contains("rgb_array"));
        assert!(json.contains("\"num_envs\":4"));
        assert!(json.contains("async"));
    }

    #[test]
    fn transient_inspect_failure_does_not_confirm_termination() {
        let running = ContainerState {
            status: "running".to_string(),
            exit_code: None,
            error: String::new(),
        };
        let exited = ContainerState {
            status: "exited".to_string(),
            exit_code: Some(0),
            error: String::new(),
        };

        // Only a successful inspect reporting a terminal state confirms exit.
        assert!(confirmed_terminal_summary(Ok(Some(exited))).is_some());
        assert!(confirmed_terminal_summary(Ok(Some(running))).is_none());
        // A transient inspect error or a not-yet-found container must not abort
        // the readiness wait; that would tear down a healthy container.
        assert!(confirmed_terminal_summary(Ok(None)).is_none());
        assert!(
            confirmed_terminal_summary(Err(anyhow!("docker inspect failed: daemon busy")))
                .is_none()
        );
    }

    #[test]
    fn parse_owned_containers_reads_id_pid_and_status() {
        let parsed = parse_owned_containers(
            "abc123|4242|pid:[4026531836]|running\ndef456|7|pid:[4026531837]|exited\n",
        );
        assert_eq!(
            parsed,
            vec![
                OwnedContainer {
                    id: "abc123".to_string(),
                    owner_pid: Some(4242),
                    owner_pid_namespace: Some("pid:[4026531836]".to_string()),
                    status: "running".to_string(),
                },
                OwnedContainer {
                    id: "def456".to_string(),
                    owner_pid: Some(7),
                    owner_pid_namespace: Some("pid:[4026531837]".to_string()),
                    status: "exited".to_string(),
                },
            ]
        );
    }

    #[test]
    fn parse_owned_containers_handles_legacy_unlabeled_pid() {
        let parsed = parse_owned_containers("abc123|||running\nxyz789|notapid||exited\n");
        assert_eq!(
            parsed,
            vec![
                OwnedContainer {
                    id: "abc123".to_string(),
                    owner_pid: None,
                    owner_pid_namespace: None,
                    status: "running".to_string(),
                },
                OwnedContainer {
                    id: "xyz789".to_string(),
                    owner_pid: None,
                    owner_pid_namespace: None,
                    status: "exited".to_string(),
                },
            ]
        );
        assert!(parse_owned_containers("\n  \n").is_empty());
        assert!(parse_owned_containers("").is_empty());
    }

    #[test]
    fn is_orphan_spares_containers_owned_by_a_live_process() {
        assert!(!is_orphan("running", Some(4242), OwnerPidLiveness::Alive));
        assert!(!is_orphan("exited", Some(4242), OwnerPidLiveness::Alive));
    }

    #[test]
    fn is_orphan_reaps_containers_whose_owner_is_gone() {
        // Owner process gone: reap regardless of container state, including a
        // running leftover from a hard-killed owner.
        assert!(is_orphan("running", Some(4242), OwnerPidLiveness::Dead));
        assert!(is_orphan("exited", Some(4242), OwnerPidLiveness::Dead));
    }

    #[test]
    fn is_orphan_spares_containers_when_owner_liveness_is_unknown() {
        assert!(!is_orphan("running", Some(4242), OwnerPidLiveness::Unknown));
        assert!(!is_orphan("exited", Some(4242), OwnerPidLiveness::Unknown));
    }

    #[test]
    fn is_orphan_treats_legacy_unlabeled_containers_conservatively() {
        // No owner-pid label: only reap when not running, since we cannot prove
        // the owner is gone. A running unlabeled container is left alone.
        assert!(is_orphan("exited", None, OwnerPidLiveness::Unknown));
        assert!(is_orphan("dead", None, OwnerPidLiveness::Unknown));
        assert!(is_orphan("created", None, OwnerPidLiveness::Unknown));
        assert!(!is_orphan("running", None, OwnerPidLiveness::Unknown));
        // State word casing must not change the decision.
        assert!(!is_orphan("Running", None, OwnerPidLiveness::Unknown));
    }

    #[test]
    fn owner_pid_liveness_is_unknown_without_matching_namespace() {
        assert_eq!(
            owner_pid_liveness(4242, None, Some("pid:[1]")),
            OwnerPidLiveness::Unknown
        );
        assert_eq!(
            owner_pid_liveness(4242, Some("pid:[2]"), Some("pid:[1]")),
            OwnerPidLiveness::Unknown
        );
        assert_eq!(
            owner_pid_liveness(4242, Some("pid:[1]"), None),
            OwnerPidLiveness::Unknown
        );
    }

    #[test]
    fn pid_is_alive_detects_current_process() {
        // The test process itself is alive; a very high pid is overwhelmingly
        // unlikely to exist on a normal system.
        assert!(pid_is_alive(std::process::id()));
        assert!(!pid_is_alive(u32::MAX));
    }

    #[test]
    fn parse_published_port_reads_host_port() {
        assert_eq!(parse_published_port("127.0.0.1:49153\n"), Some(49153));
        assert_eq!(parse_published_port("0.0.0.0:50000"), Some(50000));
        // IPv6 mappings appear with bracketed hosts; last colon-separated field is the port.
        assert_eq!(parse_published_port("[::]:51000\n"), Some(51000));
        // First parseable mapping wins when docker lists several bindings.
        assert_eq!(
            parse_published_port("0.0.0.0:49153\n[::]:49153\n"),
            Some(49153)
        );
        assert_eq!(parse_published_port(""), None);
        assert_eq!(parse_published_port("garbage"), None);
    }

    #[test]
    fn container_state_parses_exit_details() {
        let state = parse_container_state("exited\n2\nboom\n").unwrap();

        assert_eq!(
            state,
            ContainerState {
                status: "exited".to_string(),
                exit_code: Some(2),
                error: "boom".to_string(),
            }
        );
        assert!(state.is_terminal());
        assert_eq!(
            state.summary(),
            "container state: status=exited, exit_code=2, error=boom"
        );
    }

    #[test]
    fn startup_failure_report_includes_container_context() {
        let message = format_startup_failure_report(
            "abc123",
            "rlmesh-sandbox-test",
            "connection refused",
            "container state: status=exited, exit_code=1",
            "container logs:\ntraceback",
            None,
        );

        assert!(message.contains("connection refused"));
        assert!(message.contains("container id: abc123"));
        assert!(message.contains("container name: rlmesh-sandbox-test"));
        assert!(message.contains("exit_code=1"));
        assert!(message.contains("traceback"));
    }

    #[test]
    fn startup_failure_report_appends_skew_hint() {
        let with_hint = format_startup_failure_report(
            "abc123",
            "rlmesh-sandbox-test",
            "handshake failed",
            "container state: status=exited, exit_code=2",
            "container logs:\nusage: ...",
            Some("pass rlmesh_package=\"local\""),
        );
        assert!(with_hint.contains("hint: pass rlmesh_package=\"local\""));
    }

    #[test]
    fn tail_text_keeps_log_tail() {
        let value = "alpha\nbeta\ngamma";

        assert_eq!(tail_text(value, value.len()), value);
        let tail = tail_text(value, 10);
        assert!(tail.starts_with("[truncated to last 10 bytes]"));
        assert!(tail.ends_with("eta\ngamma"));
        assert!(!tail.ends_with("alpha"));
    }

    #[test]
    fn dockerfile_copies_hf_source_and_installs_requirements() {
        let spec = EffectiveSandboxSpec {
            schema_version: crate::BOOTSTRAP_SCHEMA_VERSION,
            requested_source: EnvironmentSourceRef::parse("hf://org/repo@main:suite").unwrap(),
            resolved_source: ResolvedEnvironmentSourceRef::Hf(ResolvedHfSourceRef {
                repo: "org/repo".to_string(),
                resolved_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
                suite: Some("suite".to_string()),
                task: None,
            }),
            base_image: "python:3.12-slim".to_string(),
            rlmesh_package: pip_rlmesh_package(),
            packages: vec!["numpy==2.0.0".to_string()],
            imports: vec![],
            kwargs: BTreeMap::new(),
            num_envs: 1,
            vectorization_mode: VectorizationMode::Sync,
            recipe: None,
            context_root: None,
            build_memory: None,
            build_hash: "abcdef0123456789".to_string(),
        };

        let dockerfile = render_dockerfile(&spec).unwrap();

        assert!(dockerfile.contains("COPY source /opt/rlmesh/source"));
        assert!(dockerfile.contains("/opt/rlmesh/source/requirements.txt"));
        assert!(dockerfile.contains("numpy==2.0.0"));
    }

    #[test]
    fn dockerfile_copies_and_installs_rlmesh_wheel_package() {
        let spec = EffectiveSandboxSpec {
            rlmesh_package: ResolvedRlmeshPackage::Wheel {
                source_path: "/tmp/rlmesh-0.1.0b2-cp311-abi3-manylinux_x86_64.whl".into(),
                install_path: "/opt/rlmesh/packages/rlmesh-0.1.0b2-cp311-abi3-manylinux_x86_64.whl"
                    .to_string(),
                sha256: "abc".to_string(),
            },
            ..gym_spec()
        };

        let dockerfile = render_dockerfile(&spec).unwrap();

        assert!(dockerfile.contains("COPY packages /opt/rlmesh/packages"));
        assert!(
            dockerfile
                .contains("/opt/rlmesh/packages/rlmesh-0.1.0b2-cp311-abi3-manylinux_x86_64.whl")
        );
    }

    #[test]
    fn hf_bootstrap_spec_includes_suite_and_task() {
        let spec = EffectiveSandboxSpec {
            schema_version: crate::BOOTSTRAP_SCHEMA_VERSION,
            requested_source: EnvironmentSourceRef::parse("hf://org/repo@main:suite/0").unwrap(),
            resolved_source: ResolvedEnvironmentSourceRef::Hf(ResolvedHfSourceRef {
                repo: "org/repo".to_string(),
                resolved_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
                suite: Some("suite".to_string()),
                task: Some("0".to_string()),
            }),
            base_image: "python:3.12-slim".to_string(),
            rlmesh_package: pip_rlmesh_package(),
            packages: vec![],
            imports: vec![],
            kwargs: BTreeMap::new(),
            num_envs: 1,
            vectorization_mode: VectorizationMode::Sync,
            recipe: None,
            context_root: None,
            build_memory: None,
            build_hash: "abcdef0123456789".to_string(),
        };

        let BootstrapSpec::Hf(bootstrap) = BootstrapSpec::from_effective_spec(&spec).unwrap()
        else {
            panic!("expected HF bootstrap spec");
        };

        assert_eq!(bootstrap.suite.as_deref(), Some("suite"));
        assert_eq!(bootstrap.task.as_deref(), Some("0"));
    }

    #[test]
    fn shell_quote_handles_single_quotes() {
        assert_eq!(shell_quote("pkg=='x'"), "'pkg=='\"'\"'x'\"'\"''");
    }

    #[test]
    fn bootstrap_spec_includes_kwargs() {
        let mut kwargs = BTreeMap::new();
        kwargs.insert("render_mode".to_string(), json!("rgb_array"));
        let spec = EffectiveSandboxSpec {
            imports: vec!["my_envs".to_string()],
            kwargs,
            ..gym_spec()
        };

        match &spec.resolved_source {
            ResolvedEnvironmentSourceRef::Gym(source) => assert_eq!(source.env_id, "CartPole-v1"),
            _ => panic!("expected gym source"),
        }
        assert_eq!(spec.imports, vec!["my_envs"]);
        assert_eq!(spec.kwargs["render_mode"], json!("rgb_array"));
    }

    fn recipe_spec(document: serde_json::Value) -> EffectiveSandboxSpec {
        let parsed = crate::recipe::Recipe::from_json(&document.to_string()).unwrap();
        let reference = crate::RecipeSourceRef {
            name: "acme/env".to_string(),
            document,
            provenance: crate::RecipeProvenance::Installed,
        };
        EffectiveSandboxSpec {
            schema_version: crate::BOOTSTRAP_SCHEMA_VERSION,
            requested_source: EnvironmentSourceRef::Recipe(reference.clone()),
            resolved_source: ResolvedEnvironmentSourceRef::Recipe(reference),
            base_image: parsed
                .build
                .base
                .clone()
                .unwrap_or_else(|| crate::DEFAULT_BASE_IMAGE.to_string()),
            rlmesh_package: pip_rlmesh_package(),
            packages: vec![],
            imports: vec![],
            kwargs: BTreeMap::new(),
            num_envs: 1,
            vectorization_mode: VectorizationMode::Sync,
            recipe: Some(parsed),
            context_root: None,
            build_memory: None,
            build_hash: "abcdef0123456789".to_string(),
        }
    }

    #[test]
    fn docker_run_args_add_gpu_flags_iff_gpu() {
        let without = docker_run_args("n", "img", "{}", 1, false, &[]);
        assert!(!without.iter().any(|arg| arg == "--gpus"));

        let with = docker_run_args("n", "img", "{}", 1, true, &[]);
        let idx = with.iter().position(|arg| arg == "--gpus").expect("--gpus");
        assert_eq!(with[idx + 1], "all");
        assert!(with.iter().any(|arg| arg == "NVIDIA_VISIBLE_DEVICES=all"));
        // GPU access must not weaken the existing hardening.
        assert!(with.iter().any(|arg| arg == "--cap-drop"));
    }

    #[test]
    fn recipe_spec_renders_via_the_deriver() {
        let spec = recipe_spec(json!({
            "name": "acme/env",
            "make": {"kind": "gym", "env_id": "CartPole-v1"},
            "build": {"pip": [{"packages": ["pygame"]}]}
        }));
        let dockerfile = render_dockerfile(&spec).unwrap();
        assert!(dockerfile.contains("FROM python:3.11-slim"));
        assert!(dockerfile.contains("'pygame'"));
        assert!(
            dockerfile
                .contains("ENTRYPOINT [\"python\", \"-m\", \"rlmesh._bootstrap.sandbox_env\"]")
        );
    }

    #[test]
    fn recipe_bootstrap_strips_the_build_phase() {
        let spec = recipe_spec(json!({
            "name": "acme/env",
            "make": {"kind": "gym", "env_id": "CartPole-v1"},
            "build": {"pip": [{"packages": ["pygame"]}]},
            "setup": {"env": {"K": "V"}}
        }));
        let BootstrapSpec::Recipe(bootstrap) = BootstrapSpec::from_effective_spec(&spec).unwrap()
        else {
            panic!("expected recipe bootstrap spec");
        };
        let object = bootstrap.document.as_object().unwrap();
        assert!(
            !object.contains_key("build"),
            "build phase must not re-ship"
        );
        assert!(object.contains_key("make"));
        assert!(object.contains_key("setup"));
    }

    #[cfg(unix)]
    #[test]
    fn copy_tree_skips_every_symlink_kind_without_aborting() {
        use std::fs;
        use std::os::unix::fs::symlink;

        // copy_tree must SKIP a symlink child of any kind -- never copying the
        // link, never following it -- so a real (kept) file always stages while
        // the link name never does: a symlinked dir/file does not duplicate its
        // target, a cyclic/dangling link does not abort the walk, and an escaping
        // link does not leak an out-of-tree target into the image.
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret.txt"), b"secret").unwrap();

        // (kind, build the `link` entry under `src`).
        type Case<'a> = (&'a str, &'a dyn Fn(&std::path::Path));
        let cases: [Case; 5] = [
            ("symlinked-dir", &|src| {
                let real = src.join("real");
                fs::create_dir_all(&real).unwrap();
                fs::write(real.join("asset.txt"), b"payload").unwrap();
                symlink(&real, src.join("link")).unwrap();
            }),
            ("symlinked-file", &|src| {
                fs::write(src.join("target.txt"), b"hello").unwrap();
                symlink(src.join("target.txt"), src.join("link")).unwrap();
            }),
            ("cyclic", &|src| symlink(".", src.join("link")).unwrap()),
            ("dangling", &|src| {
                symlink(src.join("does-not-exist"), src.join("link")).unwrap()
            }),
            ("escaping", &|src| {
                symlink(outside.path().join("secret.txt"), src.join("link")).unwrap()
            }),
        ];

        for (kind, build_link) in cases {
            let src_root = tempfile::tempdir().unwrap();
            fs::write(src_root.path().join("keep.txt"), b"keep").unwrap();
            build_link(src_root.path());

            let dest = tempfile::tempdir().unwrap();
            let out = dest.path().join("staged");
            copy_tree(src_root.path(), &out)
                .unwrap_or_else(|e| panic!("a {kind} symlink must not abort the copy: {e}"));

            assert_eq!(
                fs::read(out.join("keep.txt")).unwrap(),
                b"keep".to_vec(),
                "{kind}: the real file must stage"
            );
            assert!(
                !out.join("link").exists(),
                "{kind}: the symlink entry must not be staged"
            );
            // A real (non-link) sibling dir adjacent to a skipped symlink must
            // still stage under its own name; skipping the link must not skip it.
            if kind == "symlinked-dir" {
                assert_eq!(
                    fs::read(out.join("real/asset.txt")).unwrap(),
                    b"payload".to_vec()
                );
            }
        }
    }
}
