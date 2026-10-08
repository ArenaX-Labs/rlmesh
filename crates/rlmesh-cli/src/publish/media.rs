//! The media types and annotation keys a publish reads off the registry.

/// The media type of an OCI image index, the only kind of index that carries
/// annotations.
pub(super) const OCI_INDEX: &str = "application/vnd.oci.image.index.v1+json";
/// The media-type prefix of Docker's own (non-OCI) manifests.
pub(super) const DOCKER_MEDIA_PREFIX: &str = "application/vnd.docker.";
/// The media type of the Docker manifest list docker writes when every
/// descriptor it assembles is a Docker manifest.
#[cfg(test)]
pub(super) const DOCKER_MANIFEST_LIST: &str =
    "application/vnd.docker.distribution.manifest.list.v2+json";

/// The annotation BuildKit puts on an attestation manifest inside an index to
/// say what it is ([`ATTESTATION`]).
pub(super) const REFERENCE_TYPE: &str = "vnd.docker.reference.type";
/// The annotation naming the digest of the image an attestation describes.
pub(super) const REFERENCE_DIGEST: &str = "vnd.docker.reference.digest";
/// The [`REFERENCE_TYPE`] of an attestation manifest.
pub(super) const ATTESTATION: &str = "attestation-manifest";
