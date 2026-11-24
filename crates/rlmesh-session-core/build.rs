use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let proto = root.join("proto");
    let out = PathBuf::from(std::env::var("OUT_DIR")?);
    // prost-build emits no rerun-if-changed for its inputs.
    println!("cargo:rerun-if-changed={}", proto.display());
    tonic_prost_build::configure()
        .build_client(false)
        .build_server(false)
        .bytes(".")
        // `HashMap` needs `std`.
        .btree_map(".")
        .file_descriptor_set_path(out.join("session_descriptor.bin"))
        .compile_protos(&[proto.join("rlmesh/session/v2/session.proto")], &[proto])?;
    Ok(())
}
