//! Public error type for the rlmesh-sandbox crate.
//!
//! The crate uses [`anyhow`] internally for ergonomic context propagation but
//! exposes this typed [`SandboxError`] enum on its public API so consumers can
//! discriminate failure classes (e.g. "docker unavailable" vs "unpinned HF
//! source" vs "invalid options") without matching on `Display` text.

use thiserror::Error;

/// Errors returned by the public rlmesh-sandbox API.
///
/// This enum is `#[non_exhaustive]`: new variants may be added in future
/// releases, so downstream `match` expressions must include a wildcard arm.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SandboxError {
    /// An environment source reference could not be parsed.
    #[error("invalid sandbox source: {message}")]
    InvalidSource {
        /// Human-readable description of the parse failure.
        message: String,
    },

    /// A sandbox option (base image, packages, imports, num_envs,
    /// vectorization mode, rlmesh package spec, ...) was invalid.
    #[error("invalid sandbox option: {message}")]
    InvalidOption {
        /// Human-readable description of the invalid option.
        message: String,
    },

    /// An `hf://` source was rejected by the trust/pinning policy (remote code
    /// not trusted, or revision not pinned to a full git SHA).
    #[error("hugging face source policy violation: {message}")]
    HuggingFacePolicy {
        /// Human-readable description of the policy violation.
        message: String,
    },

    /// Resolving an `hf://` revision (via `git ls-remote`) or materializing the
    /// source tree failed.
    #[error("failed to resolve hugging face source: {message}")]
    SourceResolution {
        /// Human-readable description of the resolution failure.
        message: String,
    },

    /// Selecting or reading a local RLMesh wheel failed.
    #[error("rlmesh wheel resolution failed: {message}")]
    Wheel {
        /// Human-readable description of the wheel resolution failure.
        message: String,
    },

    /// Building the sandbox image failed (e.g. `docker build` returned an
    /// error, or the build context could not be assembled).
    #[error("sandbox image build failed: {message}")]
    ImageBuild {
        /// Human-readable description of the build failure.
        message: String,
    },

    /// Starting the container or waiting for it to become ready failed.
    #[error("sandbox container failed to start: {message}")]
    ContainerStartup {
        /// Human-readable description of the startup failure, including any
        /// captured container state and logs.
        message: String,
    },

    /// Invoking the `docker` CLI failed (e.g. docker is not installed or the
    /// daemon is unavailable), or a docker subcommand returned an error.
    #[error("docker command failed: {message}")]
    Docker {
        /// Human-readable description of the docker failure.
        message: String,
    },

    /// A recipe's build phase was rejected by the provenance/trust gate (e.g. a
    /// `Remote` recipe with unpinned fetches, an unpinned `build.pip`, or
    /// `build.commands`).
    #[error("recipe build policy violation: {message}")]
    RecipeBuildPolicy {
        /// Human-readable description of the policy violation.
        message: String,
    },
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

    pub(crate) fn recipe_build_policy(err: impl std::fmt::Display) -> Self {
        Self::RecipeBuildPolicy {
            message: err.to_string(),
        }
    }

    /// Classify an internal docker-backend error. If the failure was an
    /// inability to spawn the `docker` CLI (e.g. it is not installed or the
    /// daemon socket is unreachable), it is reported as [`SandboxError::Docker`]
    /// regardless of which operation triggered it; otherwise the supplied
    /// operation-specific variant is used.
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
