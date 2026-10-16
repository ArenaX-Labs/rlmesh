//! The describe envelope of a served env (ENV-13): the env-only envelope a
//! server publishes on its handshake when asked to.

use rlmesh_adapters::v1::{ENV_METADATA_KEY, build_describe_envelope, space_spec_to_json};
use serde_json::{Value as Json, json};

use crate::specs::{EnvContract, value_to_json};
use crate::{Error, Result};

/// Who is describing: the envelope's `runtime` component and language, plus
/// any host-supplied runtime fields (a C++ toolchain).
#[derive(Clone, Debug)]
pub(crate) struct DescribingHost {
    pub(crate) component: &'static str,
    pub(crate) language: &'static str,
    pub(crate) extra: serde_json::Map<String, Json>,
}

impl DescribingHost {
    pub(crate) fn rust() -> DescribingHost {
        DescribingHost {
            component: "rlmesh-rust",
            language: "rust",
            extra: serde_json::Map::new(),
        }
    }
}

/// The env envelope for `contract`: `target` `native:<id>` (or the given
/// qualname), its spaces, tags and runtime, and an optional variant catalog.
pub(crate) fn env_envelope(
    contract: &EnvContract,
    qualname: Option<&str>,
    catalog: Option<&Json>,
    host: &DescribingHost,
    workflow_edition: Option<&str>,
    vector: bool,
) -> Result<String> {
    let space = |spec: &crate::spaces::SpaceSpec, which: &str| {
        space_spec_to_json(spec.wire())
            .map_err(|err| Error::invalid_argument(format!("{which}: {err}")))
    };
    let env_tags = match contract
        .metadata()
        .and_then(|metadata| metadata.get(ENV_METADATA_KEY))
    {
        Some(tags) => value_to_json(tags)?,
        None => Json::Null,
    };
    let preferred = workflow_edition
        .map(str::trim)
        .filter(|edition| !edition.is_empty())
        .unwrap_or(rlmesh_proto::CURRENT_WORKFLOW_EDITION);
    let qualname = qualname.map_or_else(|| format!("native:{}", contract.id()), str::to_string);
    let mut runtime = json!({
        "component": host.component,
        "language": host.language,
        "package_version": crate::VERSION,
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "protocol_generation": rlmesh_proto::PROTOCOL_GENERATION,
        "supported_workflow_editions": rlmesh_proto::supported_workflow_editions(),
        "preferred_workflow_edition": preferred,
    });
    if let Some(object) = runtime.as_object_mut() {
        for (key, value) in &host.extra {
            object.insert(key.clone(), value.clone());
        }
    }
    let mut pieces = json!({
        "target": {"entrypoint": null, "qualname": qualname},
        "params": {"param_spec":null,"signature_tier":[]},
        "env_spec": {
            "observation_space": space(contract.observation_space(), "observation_space")?,
            "action_space": space(contract.action_space(), "action_space")?,
        },
        "env_tags": env_tags,
        "runtime": runtime,
    });
    if vector {
        pieces["env_spec"]["num_envs"] = json!(contract.num_envs());
    }
    if let Some(catalog) = catalog {
        pieces["variants"] = json!({ "catalog": catalog });
    }
    build_describe_envelope("env", &pieces.to_string(), None)
        .map_err(|err| Error::invalid_argument(format!("invalid describe envelope: {err}")))
}
