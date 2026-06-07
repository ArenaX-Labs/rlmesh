use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let spec = root.join("proto");
    tonic_prost_build::configure()
        .enum_attribute(
            "rlmesh.model.v1.JoinRequest.kind",
            "#[allow(clippy::large_enum_variant)]",
        )
        .compile_protos(
            &[
                // Common
                spec.join("rlmesh/common/v1/payload.proto"),
                // Core
                spec.join("rlmesh/core/v1/telemetry.proto"),
                // Env
                spec.join("rlmesh/env/v1/contract.proto"),
                spec.join("rlmesh/env/v1/interaction.proto"),
                spec.join("rlmesh/env/v1/service.proto"),
                // Spaces
                spec.join("rlmesh/spaces/v1/spaces.proto"),
                spec.join("rlmesh/spaces/v1/types.proto"),
                spec.join("rlmesh/spaces/v1/value.proto"),
                // Model
                spec.join("rlmesh/model/v1/interaction.proto"),
                spec.join("rlmesh/model/v1/service.proto"),
            ],
            &[spec],
        )?;

    Ok(())
}
