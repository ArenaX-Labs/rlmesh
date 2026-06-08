use std::fs;
use std::io::Write;
use std::net::TcpListener;
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use rlmesh_grpc::EnvClient;
use serde::Serialize;
use tempfile::TempDir;
use uuid::Uuid;

use crate::source::ResolvedEnvironmentSourceRef;
use crate::{EffectiveSandboxSpec, EnvironmentSourceRef, hf};

const DEFAULT_CONTAINER_PORT: u16 = 50051;
const READY_PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const CONTAINER_LOG_TAIL_BYTES: usize = 64 * 1024;

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
        let image_tag = format!(
            "rlmesh-sandbox-{}:{}",
            spec.slug(),
            &spec.build_hash[..12.min(spec.build_hash.len())]
        );

        if docker_image_exists(&image_tag)?
            && let Some(image_id) = inspect_image_id(&image_tag)?
        {
            return Ok(BuildArtifact { image_id });
        }

        let tempdir = tempfile::tempdir().context("failed to create sandbox build context")?;
        self.write_build_context(spec, &tempdir)?;

        let output = Command::new("docker")
            .args(["build", "-t", &image_tag, "."])
            .current_dir(tempdir.path())
            .output()
            .context("failed to invoke docker build")?;
        if !output.status.success() {
            bail!("docker build failed:\n{}", command_output(&output));
        }

        let image_id = inspect_image_id(&image_tag)?
            .ok_or_else(|| anyhow!("docker build completed but image id was not found"))?;
        Ok(BuildArtifact { image_id })
    }

    pub fn run_container(
        &self,
        spec: &EffectiveSandboxSpec,
        artifact: &BuildArtifact,
    ) -> Result<StartedContainer> {
        let host_port = reserve_host_port()?;
        let container_name = format!("rlmesh-sandbox-{}-{}", spec.slug(), Uuid::new_v4().simple());
        let output = Command::new("docker")
            .args(docker_run_args(
                &container_name,
                host_port,
                &artifact.image_id,
            ))
            .output()
            .context("failed to start docker container")?;
        if !output.status.success() {
            bail!("docker run failed:\n{}", command_output(&output));
        }

        let container_id = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let address = format!("tcp://127.0.0.1:{host_port}");
        if let Err(err) = wait_for_ready(&address, &container_id, Duration::from_secs(30)) {
            let report = self.startup_failure_report(&container_id, &container_name, &err);
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

        write_json(
            &tempdir.path().join("bootstrap.json"),
            &BootstrapConfigFile {
                spec: BootstrapSpec::from_effective_spec(spec),
            },
        )?;

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
}

impl BootstrapSpec {
    fn from_effective_spec(spec: &EffectiveSandboxSpec) -> Self {
        match &spec.resolved_source {
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
        }
    }
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

fn wait_for_ready(address: &str, container_id: &str, timeout: Duration) -> Result<()> {
    let deadline = Instant::now() + timeout;
    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;

    loop {
        match runtime.block_on(async {
            tokio::time::timeout(READY_PROBE_TIMEOUT, async {
                let mut client = EnvClient::connect(address).await?;
                client.handshake().await?;
                Ok::<_, rlmesh_grpc::error::Error>(())
            })
            .await
        }) {
            Ok(Ok(())) => return Ok(()),
            Ok(Err(err)) => {
                if let Some(state) = inspect_container_state(container_id)?
                    && state.is_terminal()
                {
                    bail!("sandbox container exited before ready: {}", state.summary());
                }
                if Instant::now() >= deadline {
                    return Err(err.into());
                }
                thread::sleep(Duration::from_millis(300));
            }
            Err(_) if Instant::now() < deadline => {
                if let Some(state) = inspect_container_state(container_id)?
                    && state.is_terminal()
                {
                    bail!("sandbox container exited before ready: {}", state.summary());
                }
                thread::sleep(Duration::from_millis(300));
            }
            Err(_) => bail!(
                "sandbox container did not respond within {} seconds",
                timeout.as_secs()
            ),
        }
    }
}

fn render_dockerfile(spec: &EffectiveSandboxSpec) -> Result<String> {
    validate_dockerfile_token("base_image", &spec.base_image)?;

    let source_copy = match &spec.resolved_source {
        ResolvedEnvironmentSourceRef::Gym(_) => "",
        ResolvedEnvironmentSourceRef::Hf(_) => "COPY source /opt/rlmesh/source\n",
    };
    let package_copy = if spec.rlmesh_package.source_path().is_some() {
        "COPY packages /opt/rlmesh/packages\n"
    } else {
        ""
    };
    let package_command = render_package_install_command(spec);

    Ok(format!(
        "# syntax=docker/dockerfile:1.7\n\n\
FROM {}\n\n\
ENV RLMESH_ENV_PORT={DEFAULT_CONTAINER_PORT}\n\
ENV PYTHONUNBUFFERED=1\n\n\
WORKDIR /opt/rlmesh\n\
COPY bootstrap.json /opt/rlmesh/bootstrap.json\n\
{}\
{}\
\n\
RUN sh -lc {}\n\n\
EXPOSE {DEFAULT_CONTAINER_PORT}\n\
ENTRYPOINT [\"python\", \"-m\", \"rlmesh._bootstrap.sandbox_env\", \"/opt/rlmesh/bootstrap.json\"]\n",
        spec.base_image,
        source_copy,
        package_copy,
        shell_quote(&package_command),
    ))
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

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn reserve_host_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").context("failed to reserve a local port")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
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

fn format_startup_failure_report(
    container_id: &str,
    container_name: &str,
    cause: &str,
    state: &str,
    logs: &str,
) -> String {
    format!(
        "sandbox container did not become ready: {cause}\ncontainer id: {container_id}\ncontainer name: {container_name}\n{state}\n{logs}"
    )
}

fn docker_run_args(container_name: &str, host_port: u16, image_id: &str) -> Vec<String> {
    vec![
        "run".to_string(),
        "-d".to_string(),
        "--cap-drop".to_string(),
        "ALL".to_string(),
        "--security-opt".to_string(),
        "no-new-privileges".to_string(),
        "--name".to_string(),
        container_name.to_string(),
        "-p".to_string(),
        format!("127.0.0.1:{host_port}:{DEFAULT_CONTAINER_PORT}"),
        image_id.to_string(),
    ]
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

fn docker_image_exists(image_ref: &str) -> Result<bool> {
    let output = Command::new("docker")
        .args(["image", "inspect", image_ref])
        .output()
        .context("failed to inspect docker image")?;
    Ok(output.status.success())
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

    use serde_json::json;

    use super::{
        BootstrapSpec, ContainerState, docker_run_args, format_startup_failure_report,
        parse_container_state, render_dockerfile, shell_quote, tail_text,
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

    #[test]
    fn dockerfile_installs_rlmesh_gymnasium_and_packages() {
        let spec = EffectiveSandboxSpec {
            schema_version: crate::BOOTSTRAP_SCHEMA_VERSION,
            requested_source: EnvironmentSourceRef::parse("CartPole-v1").unwrap(),
            resolved_source: ResolvedEnvironmentSourceRef::Gym(GymSourceRef {
                env_id: "CartPole-v1".to_string(),
            }),
            base_image: "python:3.12-slim".to_string(),
            rlmesh_package: pip_rlmesh_package(),
            packages: vec!["pygame".to_string()],
            imports: vec![],
            kwargs: BTreeMap::new(),
            num_envs: 1,
            vectorization_mode: VectorizationMode::Sync,
            build_hash: "abcdef0123456789".to_string(),
        };

        let dockerfile = render_dockerfile(&spec).unwrap();

        assert!(dockerfile.contains("FROM python:3.12-slim"));
        assert!(dockerfile.contains("rlmesh==0.1.0b2"));
        assert!(dockerfile.contains("gymnasium"));
        assert!(dockerfile.contains("pygame"));
        assert!(dockerfile.contains("rlmesh._bootstrap.sandbox_env"));
        assert!(!dockerfile.contains("COPY source"));
    }

    #[test]
    fn docker_run_args_do_not_auto_remove_container() {
        let args = docker_run_args("rlmesh-sandbox-test", 49152, "sha256:abc");

        assert_eq!(args.first().map(String::as_str), Some("run"));
        assert!(args.iter().any(|arg| arg == "-d"));
        assert!(!args.iter().any(|arg| arg == "--rm"));
        assert!(args.iter().any(|arg| arg == "--cap-drop"));
        assert!(args.iter().any(|arg| arg == "no-new-privileges"));
        assert!(args.iter().any(|arg| arg == "rlmesh-sandbox-test"));
        assert!(args.iter().any(|arg| arg == "127.0.0.1:49152:50051"));
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
        );

        assert!(message.contains("connection refused"));
        assert!(message.contains("container id: abc123"));
        assert!(message.contains("container name: rlmesh-sandbox-test"));
        assert!(message.contains("exit_code=1"));
        assert!(message.contains("traceback"));
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
                requested_revision: Some("main".to_string()),
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
            schema_version: crate::BOOTSTRAP_SCHEMA_VERSION,
            requested_source: EnvironmentSourceRef::parse("CartPole-v1").unwrap(),
            resolved_source: ResolvedEnvironmentSourceRef::Gym(GymSourceRef {
                env_id: "CartPole-v1".to_string(),
            }),
            base_image: "python:3.12-slim".to_string(),
            rlmesh_package: ResolvedRlmeshPackage::Wheel {
                source_path: "/tmp/rlmesh-0.1.0b2-cp311-abi3-manylinux_x86_64.whl".into(),
                install_path: "/opt/rlmesh/packages/rlmesh-0.1.0b2-cp311-abi3-manylinux_x86_64.whl"
                    .to_string(),
                sha256: "abc".to_string(),
            },
            packages: vec![],
            imports: vec![],
            kwargs: BTreeMap::new(),
            num_envs: 1,
            vectorization_mode: VectorizationMode::Sync,
            build_hash: "abcdef0123456789".to_string(),
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
                requested_revision: Some("main".to_string()),
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
            build_hash: "abcdef0123456789".to_string(),
        };

        let BootstrapSpec::Hf(bootstrap) = BootstrapSpec::from_effective_spec(&spec) else {
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
            schema_version: crate::BOOTSTRAP_SCHEMA_VERSION,
            requested_source: EnvironmentSourceRef::parse("CartPole-v1").unwrap(),
            resolved_source: ResolvedEnvironmentSourceRef::Gym(GymSourceRef {
                env_id: "CartPole-v1".to_string(),
            }),
            base_image: "python:3.11-slim".to_string(),
            rlmesh_package: pip_rlmesh_package(),
            packages: vec![],
            imports: vec!["my_envs".to_string()],
            kwargs,
            num_envs: 1,
            vectorization_mode: VectorizationMode::Sync,
            build_hash: "abcdef0123456789".to_string(),
        };

        match &spec.resolved_source {
            ResolvedEnvironmentSourceRef::Gym(source) => assert_eq!(source.env_id, "CartPole-v1"),
            ResolvedEnvironmentSourceRef::Hf(_) => panic!("expected gym source"),
        }
        assert_eq!(spec.imports, vec!["my_envs"]);
        assert_eq!(spec.kwargs["render_mode"], json!("rgb_array"));
    }
}
