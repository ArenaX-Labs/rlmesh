//! Lockstep messages.

use super::*;

#[test]
fn lockstep_messages_round_trip() {
    let mut g = Goldens::new("lockstep");
    let reset = LockstepReset {
        epoch: Epoch(2),
        lanes: vec![0, 3],
        seeds: vec![11, 12],
        options_json: "{}".into(),
        episode_ids: vec![100, 101],
        lease_token: LeaseToken(3),
    };
    let request = |op| LockstepRequest { request_id: 5, op };
    let round_trip =
        |r: &LockstepRequest| decode_lockstep_request(Bytes::from(encode_lockstep_request(r)));
    for op in [
        LockstepOp::Reset(reset.clone()),
        LockstepOp::Render(2),
        LockstepOp::Probe,
    ] {
        assert_eq!(
            round_trip(&request(op.clone())).expect("decode"),
            request(op)
        );
    }
    let mut bad = reset.clone();
    bad.seeds.push(13);
    assert!(round_trip(&request(LockstepOp::Reset(bad))).is_err());
    let step = LockstepStep {
        epoch: Epoch(2),
        step: 41,
        lease_token: LeaseToken(3),
        lanes: Vec::new(),
        commands: vec![PartValues {
            channel: ChannelId(1),
            payload: Bytes::from(vec![1_u8; 32]),
        }],
    };
    let buf = Bytes::from(encode_lockstep_step(&step));
    g.add("lockstep_step", &buf);
    let decoded = decode_lockstep_step(buf.clone()).expect("decode");
    assert_eq!(decoded, step);
    assert!(
        buf.as_ptr_range()
            .contains(&decoded.commands[0].payload.as_ptr())
    );
    let reply = LockstepReply {
        epoch: Epoch(2),
        step: 41,
        observations: vec![PartValues {
            channel: ChannelId(4),
            payload: Bytes::from(vec![2_u8; 16]),
        }],
        reward: Bytes::from(1.5_f32.to_le_bytes().to_vec()),
        terminated: Bytes::from_static(&[0]),
        truncated: Bytes::from_static(&[1]),
        episode_ids: vec![7],
        info_json: String::new(),
        timing: Some(EndpointTiming {
            total_ns: Some(9),
            lane_skew_ns: Some(1),
            ..EndpointTiming::default()
        }),
    };
    assert_eq!(
        decode_lockstep_reply(Bytes::copy_from_slice(
            g.add("lockstep_reply", &encode_lockstep_reply(&reply),)
        ))
        .expect("decode"),
        reply
    );
    for result in [
        LockstepResult::Obs(Box::new(reply)),
        LockstepResult::Frame(Bytes::from_static(b"png")),
        LockstepResult::Failure("{}".into()),
    ] {
        let response = LockstepResponse {
            request_id: 5,
            result,
        };
        let buf = Bytes::from(encode_lockstep_response(&response));
        g.add("lockstep_response", &buf);
        assert_eq!(decode_lockstep_response(buf).expect("decode"), response);
    }
    g.check();
}
