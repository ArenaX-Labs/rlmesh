//! The request/reply floor of the session transport: two OS processes
//! ping-pong free-running, with no host and no tick pacing, over the
//! realtime class session. This is the budget a lockstep (Gym-style step)
//! path has to live within.
//!
//! ```text
//! cargo run --release -p rlmesh-session --example rtt_bench -- [ROUND_TRIPS] [shared|per-class] [MESSAGE]
//! ```
//!
//! `MESSAGE` is what each round trip carries:
//!
//! - `sample` (default): a 72-byte `Sample` each way (the raw floor);
//! - `lockstep`: a `LockstepStep` with one T1 command (`f32[6]`) out, and a
//!   `LockstepReply` with one T1 observation (`f32[18]`), the reward and the
//!   done flags back, on publish/subscribe like `sample`;
//! - `lockstep-query`: the same messages as a Zenoh query and its reply on
//!   `l/{robot}` (the §12.3 same-host transport).
//!
//! It spawns its own `pong` child, prints one JSON line with the round-trip
//! percentiles (µs) and the rate, and exits.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use rlmesh_session::transport::{frame_zbytes, free_local_base, payload_bytes, zenoh_priority};
use rlmesh_session::wire::codec;
use rlmesh_session::wire::{
    ChannelId, ChannelPriority, Epoch, LeaseToken, LockstepReply, LockstepStep, PartValues, Sample,
};
use rlmesh_session::{TrafficClass, Transport, TransportConfig};
use zenoh::Wait;
use zenoh::qos::CongestionControl;
use zenoh::query::ConsolidationMode;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

const PING: &str = "rlmesh/bench/rtt/ping";
const PONG: &str = "rlmesh/bench/rtt/pong";
const STEP: &str = "rlmesh/bench/rtt/l/robot";

fn config(
    layout: &str,
    listen: Option<std::net::SocketAddr>,
    connect: Option<std::net::SocketAddr>,
) -> Result<TransportConfig, BoxError> {
    Ok(match layout {
        "shared" => {
            let listen: Vec<String> = listen.iter().map(|a| format!("tcp/{a}")).collect();
            let connect: Vec<String> = connect.iter().map(|a| format!("tcp/{a}")).collect();
            let listen: Vec<&str> = listen.iter().map(String::as_str).collect();
            let connect: Vec<&str> = connect.iter().map(String::as_str).collect();
            TransportConfig::shared(&listen, &connect)
        }
        _ => TransportConfig::per_class_tcp(listen, connect)?,
    })
}

fn sample(seq: u64) -> Sample {
    Sample {
        channel: ChannelId(1),
        seq,
        epoch: Epoch(1),
        capture_ns: 0,
        publish_ns: 0,
        tick: seq,
        payload: Bytes::from(vec![0_u8; 72]),
        applied: None,
    }
}

/// One T1 step: a 1-lane `f32[6]` command.
fn step(n: u64, command: &Bytes) -> LockstepStep {
    LockstepStep {
        epoch: Epoch(1),
        step: n,
        lease_token: LeaseToken(1),
        lanes: Vec::new(),
        commands: vec![PartValues {
            channel: ChannelId(1),
            payload: command.clone(),
        }],
    }
}

/// The reply to a T1 step: a 1-lane `f32[18]` observation, reward and flags.
fn reply(n: u64, observation: &Bytes) -> LockstepReply {
    LockstepReply {
        epoch: Epoch(1),
        step: n,
        observations: vec![PartValues {
            channel: ChannelId(3),
            payload: observation.clone(),
        }],
        reward: Bytes::from(0.5_f32.to_le_bytes().to_vec()),
        terminated: Bytes::from_static(&[0]),
        truncated: Bytes::from_static(&[0]),
        episode_ids: vec![7],
        info_json: String::new(),
        timing: None,
    }
}

/// The step number a pong carries, if it decodes.
fn pong_seq(message: &str, payload: &zenoh::bytes::ZBytes) -> Option<u64> {
    let bytes = payload_bytes(payload);
    match message {
        "sample" => codec::decode_sample(bytes).ok().map(|s| s.seq),
        _ => codec::decode_lockstep_reply(bytes).ok().map(|r| r.step),
    }
}

/// Answers every ping with a pong carrying the same seq.
async fn pong(layout: &str, message: &str, base: std::net::SocketAddr) -> Result<(), BoxError> {
    let transport = Transport::open(&config(layout, Some(base), None)?).await?;
    let session = transport.session(TrafficClass::Realtime).clone();
    let observation = Bytes::from(vec![0_u8; 72]);
    let reply_to = session
        .declare_publisher(PONG)
        .priority(zenoh_priority(ChannelPriority::Realtime))
        .congestion_control(CongestionControl::Block)
        .express(true)
        .await?;
    let _sub = session
        .declare_subscriber(PING)
        .callback({
            let (message, observation) = (message.to_string(), observation.clone());
            move |s| {
                let bytes = payload_bytes(s.payload());
                let out = match message.as_str() {
                    "sample" => codec::decode_sample(bytes)
                        .ok()
                        .map(|ping| frame_zbytes(codec::encode_sample(&sample(ping.seq)))),
                    _ => codec::decode_lockstep_step(bytes).ok().map(|step| {
                        zenoh::bytes::ZBytes::from(codec::encode_lockstep_reply(&reply(
                            step.step,
                            &observation,
                        )))
                    }),
                };
                if let Some(out) = out {
                    let _ = reply_to.put(out).wait();
                }
            }
        })
        .await?;
    let _queryable = session
        .declare_queryable(STEP)
        .callback(move |query| {
            let Some(step) = query
                .payload()
                .and_then(|p| codec::decode_lockstep_step(payload_bytes(p)).ok())
            else {
                return;
            };
            let _ = query
                .reply(
                    STEP,
                    codec::encode_lockstep_reply(&reply(step.step, &observation)),
                )
                .wait();
        })
        .await?;
    println!("ready");
    // Exit when the parent closes stdin.
    let mut line = String::new();
    let _ = std::io::stdin().read_line(&mut line);
    Ok(())
}

async fn ping(layout: &str, message: &str, round_trips: usize) -> Result<(), BoxError> {
    let base = free_local_base("127.0.0.1".parse()?)?;
    let mut child = Command::new(std::env::current_exe()?)
        .args(["pong", layout, message, &base.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let mut ready = String::new();
    std::io::BufRead::read_line(
        &mut std::io::BufReader::new(child.stdout.take().ok_or("child stdout")?),
        &mut ready,
    )?;
    let transport = Transport::open(&config(layout, None, Some(base))?).await?;
    let session = transport.session(TrafficClass::Realtime).clone();
    let (tx, rx) = mpsc::sync_channel::<u64>(16);
    let _sub = session
        .declare_subscriber(PONG)
        .callback({
            let message = message.to_string();
            move |s| {
                if let Some(seq) = pong_seq(&message, s.payload()) {
                    let _ = tx.try_send(seq);
                }
            }
        })
        .await?;
    let publisher = session
        .declare_publisher(PING)
        .priority(zenoh_priority(ChannelPriority::Realtime))
        .congestion_control(CongestionControl::Block)
        .express(true)
        .await?;
    let command = Bytes::from(vec![0_u8; 24]);
    // Wait for the route to exist, then warm up.
    let warm = 2_000;
    let mut rtts = Vec::with_capacity(round_trips);
    let mut seq = 0_u64;
    let started_at = Instant::now();
    let mut measured_from = None;
    while rtts.len() < round_trips {
        seq += 1;
        let sent = Instant::now();
        let got = match message {
            "lockstep-query" => {
                let replies = session
                    .get(STEP)
                    .payload(codec::encode_lockstep_step(&step(seq, &command)))
                    .consolidation(ConsolidationMode::None)
                    .priority(zenoh_priority(ChannelPriority::Realtime))
                    .congestion_control(CongestionControl::Block)
                    .express(true)
                    .timeout(Duration::from_millis(200))
                    .wait()?;
                replies
                    .recv()
                    .ok()
                    .and_then(|r| r.result().ok().and_then(|s| pong_seq(message, s.payload())))
                    .ok_or(())
            }
            _ => {
                let out = match message {
                    "sample" => frame_zbytes(codec::encode_sample(&sample(seq))),
                    _ => zenoh::bytes::ZBytes::from(codec::encode_lockstep_step(&step(
                        seq, &command,
                    ))),
                };
                publisher.put(out).wait()?;
                rx.recv_timeout(Duration::from_millis(200)).map_err(|_| ())
            }
        };
        match got {
            Ok(got) if got == seq => {
                if seq > warm {
                    measured_from.get_or_insert_with(Instant::now);
                    rtts.push(sent.elapsed().as_nanos() as u64);
                }
            }
            Ok(_) | Err(()) => {
                // Before the route is up pings are lost; drain and retry.
                while rx.try_recv().is_ok() {}
                if started_at.elapsed() > Duration::from_secs(20) {
                    return Err("no pong".into());
                }
            }
        }
    }
    let elapsed = measured_from.map_or(Duration::ZERO, |t| t.elapsed());
    drop(child.stdin.take());
    let _ = child.wait();
    rtts.sort_unstable();
    let at = |q: f64| {
        rtts[((q * rtts.len() as f64).ceil() as usize).clamp(1, rtts.len()) - 1] as f64 / 1e3
    };
    println!(
        "{{\"layout\":\"{layout}\",\"message\":\"{message}\",\"round_trips\":{},\"p50_us\":{:.1},\"p99_us\":{:.1},\"p999_us\":{:.1},\"max_us\":{:.1},\"per_s\":{:.0}}}",
        rtts.len(),
        at(0.5),
        at(0.99),
        at(0.999),
        at(1.0),
        rtts.len() as f64 / elapsed.as_secs_f64().max(1e-9)
    );
    Ok(())
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), BoxError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("pong") {
        let usage = "pong LAYOUT MESSAGE BASE";
        let layout = args.get(1).ok_or(usage)?;
        let message = args.get(2).ok_or(usage)?;
        let base = args.get(3).ok_or(usage)?.parse()?;
        return pong(layout, message, base).await;
    }
    let round_trips = args.first().map_or(Ok(20_000), |n| n.parse())?;
    let layout = args.get(1).map_or("per-class", String::as_str);
    let message = args.get(2).map_or("sample", String::as_str);
    if !matches!(message, "sample" | "lockstep" | "lockstep-query") {
        return Err(
            format!("unknown message {message}: sample, lockstep or lockstep-query").into(),
        );
    }
    ping(layout, message, round_trips).await
}
