//! Served-model messages.

use super::*;

#[test]
fn model_messages_round_trip() {
    let mut g = Goldens::new("model");
    let slab = |b: u8| vec![Bytes::from(vec![b; 8]), Bytes::from(vec![b; 3])];
    let episode = |id: &str| ModelEpisode {
        id: id.into(),
        seed: Some(-4),
    };
    for request in [
        ModelRequest::Resolve(ModelResolve {
            route: "cartpole".into(),
            contract_json: "{}".into(),
            execution_horizon: 4,
            delivers_history: true,
            workflow_edition: "2026.06".into(),
            reads_advisories: true,
        }),
        ModelRequest::Predict(vec![ModelGroup {
            route: "cartpole".into(),
            episodes: vec![episode("a"), episode("b")],
            observation: slab(1),
            history: vec![ModelHistoryRow {
                observation: slab(2),
                episodes: vec![episode("a")],
                step: 3,
            }],
            step: Some(4),
        }]),
        ModelRequest::Reset {
            route: "cartpole".into(),
            episode_ids: vec!["a".into()],
        },
        ModelRequest::Release {
            route: "cartpole".into(),
        },
        ModelRequest::Describe,
        ModelRequest::ResetOptions {
            route: "cartpole".into(),
            episode: episode("a"),
            trial_index: Some(2),
        },
        ModelRequest::Join {
            session: "10.0.0.5:7447/bench".into(),
        },
        ModelRequest::Leave {
            session: "10.0.0.5:7447/bench".into(),
        },
    ] {
        let decoded = decode_model_request(Bytes::copy_from_slice(
            g.add("model_request", &encode_model_request(&request)),
        ));
        assert_eq!(decoded.expect("decode"), request);
    }
    let fault = ModelFault {
        code: "INTERNAL".into(),
        message: "no".into(),
        recoverable: true,
    };
    for reply in [
        ModelReply::Resolved(ModelResolved {
            native_chunk: Some(8),
            history: Some((vec!["image".into()], false)),
            reset_options: true,
            advisories: vec![("caution".into(), "rebind_camera (allowed): ...".into())],
        }),
        ModelReply::Predicted(vec![Ok(vec![slab(5), slab(6)]), Err(fault.clone())]),
        ModelReply::Done,
        ModelReply::Described("{\"kind\":\"model\"}".into()),
        ModelReply::Options("{\"friction\":0.1}".into()),
        ModelReply::Fault(fault),
    ] {
        let timing = EndpointTiming {
            user_ns: Some(7),
            held_episodes: Some(2),
            ..EndpointTiming::default()
        };
        let decoded = decode_model_reply(Bytes::copy_from_slice(
            g.add("model_reply", &encode_model_reply(&reply, Some(&timing))),
        ));
        assert_eq!(decoded.expect("decode"), (reply, Some(timing)));
    }
    assert!(decode_model_request(Bytes::new()).is_err());
    g.check();
}
