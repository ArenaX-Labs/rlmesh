//! Samples, batches and chunks.

use super::*;

fn applied_sample() -> Sample {
    Sample {
        channel: ChannelId(2),
        epoch: Epoch(3),
        seq: 41,
        capture_ns: 100,
        publish_ns: 120,
        tick: 7,
        payload: Bytes::from(vec![7_u8; 24]),
        applied: Some(Applied {
            mode: EdgeMode::Hold,
            present: true,
            lease_token: LeaseToken(4),
            chunk_seq: 9,
            action_index: 3,
            reason: ReasonCode::QueueDry,
        }),
    }
}

#[test]
fn samples_are_plain_protobuf_with_the_payload_neither_copied_on_send_nor_on_receipt() {
    let mut g = Goldens::new("sample");
    let sample = applied_sample();
    let encoded = encode_sample(&sample);
    g.add("sample", &encoded.to_vec());
    assert!(
        encoded
            .segments
            .iter()
            .any(|s| s.as_ptr() == sample.payload.as_ptr()),
        "the payload is a segment of its own"
    );
    let buf = Bytes::from(encoded.to_vec());
    // Any protobuf implementation reads it.
    let plain = pb::Sample::decode(&buf[..]).expect("plain protobuf");
    assert_eq!((plain.seq, plain.tick, plain.payload.len()), (41, 7, 24));
    assert_eq!(plain.applied.map(|a| a.chunk_seq), Some(9));
    let decoded = decode_sample(buf.clone()).expect("decode");
    assert_eq!(decoded, sample);
    assert!(
        buf.as_ptr_range().contains(&decoded.payload.as_ptr()),
        "the payload slices the received buffer"
    );
    // An applied tick with nothing dispatched has no payload.
    let absent = Sample {
        payload: Bytes::new(),
        applied: Some(Applied::absent(EdgeMode::Safe)),
        ..sample
    };
    assert_eq!(
        decode_sample(Bytes::from(encode_sample(&absent).to_vec())).expect("decode"),
        absent
    );
    let mut bad = pb::Sample::decode(&buf[..]).expect("plain");
    bad.applied = Some(pb::Applied {
        mode: 99,
        ..pb::Applied::default()
    });
    assert!(decode_sample(Bytes::from(bad.encode_to_vec())).is_err());
    assert!(decode_sample(Bytes::from_static(b"\xff\xff")).is_err());
    g.check();
}

#[test]
fn batches_round_trip_with_their_gaps() {
    let mut g = Goldens::new("batch");
    let batch = SampleBatch {
        channel: ChannelId(2),
        epoch: Epoch(3),
        publish_ns: 5,
        samples: vec![
            SampleMeta {
                seq: 4,
                capture_ns: 1,
                tick: 0,
                applied: None,
            },
            SampleMeta {
                seq: 6,
                capture_ns: 2,
                tick: 0,
                applied: applied_sample().applied,
            },
        ],
        payload: Bytes::from(vec![1_u8; 48]),
        truncated: true,
        gaps: vec![SeqRange { first: 5, last: 5 }],
        request_id: 0,
    };
    let buf = Bytes::from(encode_batch(&batch).expect("encode").to_vec());
    g.add("batch", &buf);
    let decoded = decode_batch(buf.clone()).expect("decode");
    assert_eq!(decoded, batch);
    assert!(buf.as_ptr_range().contains(&decoded.payload.as_ptr()));
    let mut huge = batch;
    huge.samples = vec![huge.samples[0]; MAX_BATCH + 1];
    assert!(encode_batch(&huge).is_err());
    g.check();
}

fn chunk() -> ActionChunk {
    ActionChunk {
        channel: ChannelId(1),
        epoch: Epoch(2),
        lease_token: LeaseToken(3),
        seq: 9,
        source: pid("policy"),
        binding: Fingerprint::from(0xfeed),
        based_on: vec![
            InputRange::one(ChannelId(2), 41),
            InputRange {
                channel: ChannelId(5),
                seq_first: 7,
                seq_last: 10,
            },
        ],
        t0_ns: 1_000,
        dt_ns: 1_000_000,
        replace_from_ns: 900,
        valid_until_ns: 500_000_000,
        count: 5,
        actions: Bytes::from((0..120_u8).collect::<Vec<_>>()),
        publish_ns: 78,
        tick: 0,
    }
}

#[test]
fn chunks_round_trip_through_plain_protobuf_without_copying_the_actions() {
    let mut g = Goldens::new("chunk");
    let chunk = chunk();
    let encoded = encode_chunk(&chunk).expect("encode");
    g.add("chunk", &encoded.to_vec());
    let buf = Bytes::from(encoded.to_vec());
    let plain = pb::ActionChunk::decode(&buf[..]).expect("plain protobuf");
    assert_eq!((plain.count, plain.actions.len()), (5, 120));
    let decoded = decode_chunk(buf.clone()).expect("decode");
    assert_eq!(decoded, chunk);
    assert!(buf.as_ptr_range().contains(&decoded.actions.as_ptr()));
    assert_eq!(
        decoded.action(4, 24).map(|a| a[0]),
        Some(96),
        "action 4 at a 24-byte stride"
    );
    assert_eq!(decoded.action(5, 24), None);
    assert_eq!(decoded.reference().seq, 9);

    let mut huge = chunk.clone();
    huge.count = u32::try_from(MAX_CHUNK_ACTIONS + 1).expect("count");
    assert!(encode_chunk(&huge).is_err());
    let mut no_binding = pb::ActionChunk::decode(&buf[..]).expect("plain");
    no_binding.binding = Bytes::new();
    assert!(decode_chunk(Bytes::from(no_binding.encode_to_vec())).is_err());
    g.check();
}
