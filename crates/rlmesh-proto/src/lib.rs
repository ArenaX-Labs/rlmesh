//! Generated RLMesh protobuf bindings and protocol-level constants.

/// Current RLMesh wire ABI version.
pub const ABI_VERSION: &str = "0.1.0";

/// Return whether a client ABI version can speak to a server ABI version.
pub fn is_abi_compatible(client: &str, server: &str) -> bool {
    // Require exact match until the protocol has an explicit semver policy.
    client == server
}

pub mod common {
    pub mod v1 {
        tonic::include_proto!("rlmesh.common.v1");
    }
}

pub mod core {
    pub mod v1 {
        tonic::include_proto!("rlmesh.core.v1");
    }
}

pub mod env {
    pub mod v1 {
        tonic::include_proto!("rlmesh.env.v1");
    }
}

pub mod spaces {
    pub mod v1 {
        tonic::include_proto!("rlmesh.spaces.v1");
    }
}

pub mod model {
    pub mod v1 {
        tonic::include_proto!("rlmesh.model.v1");
    }
}
