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
            .args([
                "run",
                "-d",
                "--rm",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--name",
                &container_name,
                "-p",
                &format!("127.0.0.1:{host_port}:{DEFAULT_CONTAINER_PORT}"),
                &artifact.image_id,
            ])
            .output()
            .context("failed to start docker container")?;
        if !output.status.success() {
            bail!("docker run failed:\n{}", command_output(&output));
        }

        let container_id = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let address = format!("tcp://127.0.0.1:{host_port}");
        if let Err(err) = wait_for_ready(&address, Duration::from_secs(30)) {
            let logs = self.container_logs(&container_id).unwrap_or_default();
            let _ = self.stop_container(&container_id);
            return Err(anyhow!("{err}. container logs:\n{logs}"));
        }

        Ok(StartedContainer {
            container_id,
            address,
        })
    }

    pub fn stop_container(&self, container_id: &str) -> Result<()> {
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
    imports: Vec<String>,
    kwargs: std::collections::BTreeMap<String, serde_json::Value>,
    num_envs: usize,
    vectorization_mode: String,
}

fn wait_for_ready(address: &str, timeout: Duration) -> Result<()> {
    let deadline = Instant::now() + timeout;
    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;

    loop {
        match runtime.block_on(async {
            let mut client = EnvClient::connect(address).await?;
            client.handshake().await?;
            Ok::<_, rlmesh_grpc::error::Error>(())
        }) {
            Ok(()) => return Ok(()),
            Err(_) if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(300));
            }
            Err(err) => return Err(err.into()),
        }
    }
}

fn render_dockerfile(spec: &EffectiveSandboxSpec) -> Result<String> {
    validate_dockerfile_token("base_image", &spec.base_image)?;

    let base_packages = [spec.package_spec.clone(), "gymnasium".to_string()];
    let base_package_args = base_packages
        .iter()
        .map(|package| shell_quote(package))
        .collect::<Vec<_>>()
        .join(" ");
    let package_command = render_package_install_command(spec, &base_package_args);
    let source_copy = match &spec.resolved_source {
        ResolvedEnvironmentSourceRef::Gym(_) => "",
        ResolvedEnvironmentSourceRef::Hf(_) => "COPY source /opt/rlmesh/source\n",
    };

    Ok(format!(
        "# syntax=docker/dockerfile:1.7\n\n\
FROM {}\n\n\
ENV RLMESH_ENV_PORT={DEFAULT_CONTAINER_PORT}\n\
ENV PYTHONUNBUFFERED=1\n\n\
WORKDIR /opt/rlmesh\n\
COPY bootstrap.json /opt/rlmesh/bootstrap.json\n\
{}\
\n\
RUN sh -lc {}\n\n\
EXPOSE {DEFAULT_CONTAINER_PORT}\n\
ENTRYPOINT [\"python\", \"-m\", \"rlmesh._bootstrap.sandbox_env\", \"/opt/rlmesh/bootstrap.json\"]\n",
        spec.base_image,
        source_copy,
        shell_quote(&package_command),
    ))
}

fn render_package_install_command(spec: &EffectiveSandboxSpec, base_package_args: &str) -> String {
    let mut parts = vec![
        "python -m pip install --no-cache-dir --upgrade pip".to_string(),
        format!("python -m pip install --no-cache-dir {base_package_args}"),
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

    use super::{render_dockerfile, shell_quote};
    use crate::source::{ResolvedEnvironmentSourceRef, ResolvedHfSourceRef};
    use crate::{EffectiveSandboxSpec, EnvironmentSourceRef, GymSourceRef, VectorizationMode};

    #[test]
    fn dockerfile_installs_rlmesh_gymnasium_and_packages() {
        let spec = EffectiveSandboxSpec {
            schema_version: crate::BOOTSTRAP_SCHEMA_VERSION,
            requested_source: EnvironmentSourceRef::parse("CartPole-v1").unwrap(),
            resolved_source: ResolvedEnvironmentSourceRef::Gym(GymSourceRef {
                env_id: "CartPole-v1".to_string(),
            }),
            base_image: "python:3.12-slim".to_string(),
            package_spec: "rlmesh==0.1.0b1".to_string(),
            packages: vec!["pygame".to_string()],
            imports: vec![],
            kwargs: BTreeMap::new(),
            num_envs: 1,
            vectorization_mode: VectorizationMode::Sync,
            build_hash: "abcdef0123456789".to_string(),
        };

        let dockerfile = render_dockerfile(&spec).unwrap();

        assert!(dockerfile.contains("FROM python:3.12-slim"));
        assert!(dockerfile.contains("rlmesh==0.1.0b1"));
        assert!(dockerfile.contains("gymnasium"));
        assert!(dockerfile.contains("pygame"));
        assert!(dockerfile.contains("rlmesh._bootstrap.sandbox_env"));
        assert!(!dockerfile.contains("COPY source"));
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
            }),
            base_image: "python:3.12-slim".to_string(),
            package_spec: "rlmesh==0.1.0b1".to_string(),
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
            package_spec: "rlmesh==0.1.0b1".to_string(),
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
