use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let proto = root.join("proto");
    let out = PathBuf::from(std::env::var("OUT_DIR")?);
    // prost-build emits no rerun-if-changed for its inputs; watch the tree.
    println!("cargo:rerun-if-changed={}", proto.display());
    // Messages only: the session wire rides Zenoh, not gRPC.
    tonic_prost_build::configure()
        .build_client(false)
        .build_server(false)
        // `bytes` fields decode as `Bytes` slices of the received buffer.
        .bytes(".")
        // Map fields as `BTreeMap`: `HashMap` needs `std`.
        .btree_map(".")
        // The descriptors recordings carry as their MCAP schemas (§13).
        .file_descriptor_set_path(out.join("session_descriptor.bin"))
        .compile_protos(&[proto.join("rlmesh/session/v2/session.proto")], &[proto])?;
    Ok(())
}
