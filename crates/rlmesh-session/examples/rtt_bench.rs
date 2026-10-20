//! The request/reply floor of the session transport: two OS processes
//! ping-pong a 72-byte sample (encoded and decoded with the session codec)
//! over the realtime class, free-running, with no host and no tick pacing.
//! This is the budget a lockstep (Gym-style step) path has to live within.
//!
//! ```text
//! cargo run --release -p rlmesh-session --example rtt_bench -- [ROUND_TRIPS] [shared|per-class]
//! ```
//!
//! It spawns its own `pong` child, prints one JSON line with the round-trip
//! percentiles (µs) and the rate, and exits.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use rlmesh_session::transport::{frame_zbytes, free_local_base, payload_bytes, zenoh_priority};
use rlmesh_session::{TrafficClass, Transport, TransportConfig};
use rlmesh_wire::codec;
use rlmesh_wire::{ChannelId, ChannelPriority, Epoch, Sample};
use zenoh::Wait;
use zenoh::qos::CongestionControl;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

const PING: &str = "rlmesh/bench/rtt/ping";
const PONG: &str = "rlmesh/bench/rtt/pong";

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
        epoch: Epoch(1),
        seq,
        capture_ns: 0,
        publish_ns: 0,
        tick: seq,
        payload: Bytes::from(vec![0_u8; 72]),
        applied: None,
    }
}

/// Answers every ping with a pong carrying the same seq.
async fn pong(layout: &str, base: std::net::SocketAddr) -> Result<(), BoxError> {
    let transport = Transport::open(&config(layout, Some(base), None)?).await?;
    let session = transport.session(TrafficClass::Realtime).clone();
    let reply = session
        .declare_publisher(PONG)
        .priority(zenoh_priority(ChannelPriority::Realtime))
        .congestion_control(CongestionControl::Block)
        .express(true)
        .await?;
    let _sub = session
        .declare_subscriber(PING)
        .callback(move |s| {
            if let Ok(ping) = codec::decode_sample(payload_bytes(s.payload())) {
                let _ = reply
                    .put(frame_zbytes(codec::encode_sample(&sample(ping.seq))))
                    .wait();
            }
        })
        .await?;
    println!("ready");
    // Exit when the parent closes stdin.
    let mut line = String::new();
    let _ = std::io::stdin().read_line(&mut line);
    Ok(())
}

async fn ping(layout: &str, round_trips: usize) -> Result<(), BoxError> {
    let base = free_local_base("127.0.0.1".parse()?)?;
    let mut child = Command::new(std::env::current_exe()?)
        .args(["pong", layout, &base.to_string()])
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
        .callback(move |s| {
            if let Ok(pong) = codec::decode_sample(payload_bytes(s.payload())) {
                let _ = tx.try_send(pong.seq);
            }
        })
        .await?;
    let publisher = session
        .declare_publisher(PING)
        .priority(zenoh_priority(ChannelPriority::Realtime))
        .congestion_control(CongestionControl::Block)
        .express(true)
        .await?;
    // Wait for the route to exist, then warm up.
    let warm = 2_000;
    let mut rtts = Vec::with_capacity(round_trips);
    let mut seq = 0_u64;
    let started_at = Instant::now();
    let mut measured_from = None;
    while rtts.len() < round_trips {
        seq += 1;
        let sent = Instant::now();
        publisher
            .put(frame_zbytes(codec::encode_sample(&sample(seq))))
            .wait()?;
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(got) if got == seq => {
                if seq > warm {
                    measured_from.get_or_insert_with(Instant::now);
                    rtts.push(sent.elapsed().as_nanos() as u64);
                }
            }
            Ok(_) | Err(_) => {
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
        "{{\"layout\":\"{layout}\",\"round_trips\":{},\"p50_us\":{:.1},\"p99_us\":{:.1},\"p999_us\":{:.1},\"max_us\":{:.1},\"per_s\":{:.0}}}",
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
        let layout = args.get(1).ok_or("pong LAYOUT BASE")?;
        let base = args.get(2).ok_or("pong LAYOUT BASE")?.parse()?;
        return pong(layout, base).await;
    }
    let round_trips = args.first().map_or(Ok(20_000), |n| n.parse())?;
    let layout = args.get(1).map_or("per-class", String::as_str);
    ping(layout, round_trips).await
}
