//! Every public type that holds an address prints it without its key, and
//! no refusal of a keyed address repeats it.

use super::*;
use crate::{
    BindOptions, EnvTarget, Model, ModelOptions, ModelServeOptions, RemoteModel,
    RemoteModelOptions, RunOptions,
};

/// The key of a keyed address (`…#{key}/…`).
fn key_of(address: &str) -> String {
    let (_, rest) = address.split_once('#').expect("a keyed address");
    rest.split('/').next().unwrap_or_default().to_string()
}

fn assert_keyless(key: &str, shown: &[(&str, String)]) {
    for (name, text) in shown {
        assert!(
            !text.contains(key),
            "{name} shows the key: {}",
            text.replace(key, "<KEY>")
        );
    }
}

#[test]
fn display_and_debug_leave_the_key_out() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::new(Counter::new(&counters), quick()).unwrap();
    server.start().unwrap();
    let address = server.address().to_string();
    let key = key_of(&address);
    let session = address.rsplit_once('/').unwrap().0.to_string();
    let vector = EnvServer::vector(
        Pair {
            num_envs: 2,
            seen: Arc::default(),
        },
        EnvServerOptions {
            address: Some(":0/pair".into()),
            ..quick()
        },
    )
    .unwrap();
    vector.start().unwrap();
    let vector_key = key_of(vector.address());

    let mut remote = dial(&address).unwrap();
    let mut remote_vector =
        RemoteVectorEnv::new(RemoteEnvOptions::default().address(vector.address())).unwrap();
    let model_address = format!("{session}/policy");
    let remote_model =
        RemoteModel::new(RemoteModelOptions::default().address(&model_address)).unwrap();
    let connect = crate::ext::ConnectAddress::parse(&address).unwrap();
    let shown = [
        ("ConnectAddress", format!("{connect} {connect:?}")),
        (
            "RunLocalOptions",
            format!(
                "{:?}",
                crate::model::RunLocalOptions::parse(&address).unwrap()
            ),
        ),
        (
            "RemoteEnvOptions",
            format!("{:?}", RemoteEnvOptions::default().address(&address)),
        ),
        (
            "RemoteModelOptions",
            format!(
                "{:?}",
                RemoteModelOptions::default().address(&model_address)
            ),
        ),
        ("RemoteModel", format!("{remote_model} {remote_model:?}")),
        (
            "EnvServerOptions",
            format!(
                "{:?}",
                EnvServerOptions {
                    address: Some(address.clone()),
                    ..quick()
                }
            ),
        ),
        ("EnvServer", format!("{server} {server:?}")),
        ("RemoteEnv", format!("{remote:?}")),
        ("RemoteVectorEnv", format!("{remote_vector:?}")),
        (
            "EnvTarget::Address",
            format!("{:?}", EnvTarget::from(address.as_str())),
        ),
        (
            "EnvTarget::Server",
            format!("{:?}", EnvTarget::from(&server)),
        ),
        (
            "EnvTarget::Remote",
            format!("{:?}", EnvTarget::from(&mut remote)),
        ),
        (
            "EnvTarget::Remote (vector)",
            format!("{:?}", EnvTarget::from(&mut remote_vector)),
        ),
    ];
    assert_keyless(&key, &shown);
    assert_keyless(&vector_key, &shown);
    for (name, text) in &shown {
        assert!(
            text.contains("#…"),
            "{name}: {}",
            text.replace(&key, "<KEY>")
        );
    }
    remote.close().unwrap();
    remote_vector.close().unwrap();
    vector.shutdown().unwrap();
    server.shutdown().unwrap();
}

#[test]
fn a_refused_keyed_address_is_not_repeated() {
    // A key of the right shape; nothing listens behind these addresses.
    let (_, key) = rlmesh_session::address::launch_identity().unwrap();
    let key = key.to_string();
    let v1 = format!("tcp://10.0.0.5:1#{key}/env");
    let model = Model::new(|_: Value| Ok(Value::Int(0)), ModelOptions::default()).unwrap();
    let error = |result: crate::Result<()>| result.expect_err("refused").to_string();
    let shown = [
        (
            "served_model (v1)",
            error(crate::binding::model::served_model(&v1).map(drop)),
        ),
        (
            "served_model (no model)",
            error(crate::binding::model::served_model(&format!("10.0.0.5:1#{key}")).map(drop)),
        ),
        (
            "Model::serve (v1)",
            error(model.serve(
                &format!("tcp://10.0.0.5:1#{key}"),
                ModelServeOptions {
                    name: Some("p".into()),
                    ..Default::default()
                },
            )),
        ),
        (
            "Model::run (v1)",
            error(model.run(v1.as_str(), RunOptions::default()).map(drop)),
        ),
        (
            "Model::bind (v1)",
            error(model.bind(v1.as_str(), BindOptions::default()).map(drop)),
        ),
        ("RemoteEnv (v1)", error(dial(&v1).map(drop))),
        (
            "ConnectAddress (v1)",
            error(crate::ext::ConnectAddress::parse(&v1).map(drop)),
        ),
    ];
    assert_keyless(&key, &shown);
}
