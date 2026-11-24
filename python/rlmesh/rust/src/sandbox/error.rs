use thiserror::Error;

/// Errors returned by the Docker sandbox.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SandboxError {
    #[error("invalid sandbox source: {message}")]
    InvalidSource { message: String },

    #[error("invalid sandbox option: {message}")]
    InvalidOption { message: String },

    #[error("hugging face source policy violation: {message}")]
    HuggingFacePolicy { message: String },

    #[error("failed to resolve hugging face source: {message}")]
    SourceResolution { message: String },

    #[error("rlmesh wheel resolution failed: {message}")]
    Wheel { message: String },

    #[error("sandbox image build failed: {message}")]
    ImageBuild { message: String },

    #[error("sandbox container failed to start: {message}")]
    ContainerStartup { message: String },

    #[error("docker command failed: {message}")]
    Docker { message: String },
}

impl SandboxError {
    pub(crate) fn invalid_source(err: impl std::fmt::Display) -> Self {
        Self::InvalidSource {
            message: err.to_string(),
        }
    }

    pub(crate) fn invalid_option(err: impl std::fmt::Display) -> Self {
        Self::InvalidOption {
            message: err.to_string(),
        }
    }

    pub(crate) fn huggingface_policy(err: impl std::fmt::Display) -> Self {
        Self::HuggingFacePolicy {
            message: err.to_string(),
        }
    }

    pub(crate) fn source_resolution(err: impl std::fmt::Display) -> Self {
        Self::SourceResolution {
            message: err.to_string(),
        }
    }

    pub(crate) fn wheel(err: impl std::fmt::Display) -> Self {
        Self::Wheel {
            message: err.to_string(),
        }
    }

    pub(crate) fn container_startup(err: impl std::fmt::Display) -> Self {
        Self::ContainerStartup {
            message: err.to_string(),
        }
    }

    /// Classify an internal docker-backend error: a failure to spawn the `docker` CLI is always
    /// [`SandboxError::Docker`]; otherwise the supplied operation-specific variant is used.
    pub(crate) fn from_docker_op(
        err: anyhow::Error,
        operation: impl FnOnce(String) -> Self,
    ) -> Self {
        if let Some(io_err) = err
            .chain()
            .find_map(|cause| cause.downcast_ref::<std::io::Error>())
            && io_err.kind() == std::io::ErrorKind::NotFound
        {
            return Self::Docker {
                message: format!("docker CLI not found: {err:#}"),
            };
        }
        operation(format!("{err:#}"))
    }
}
