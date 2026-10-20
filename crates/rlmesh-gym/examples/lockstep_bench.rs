//! The §12 acceptance bench: bench2's T1 shape, free-running, on v1 direct
//! (gRPC Join stream) and on the v2 lockstep path, in the same build.
//!
//! T1 (bench2-spec.md): the env is the embodiment, observation
//! `Dict(ft: Box(18) f32, stamp: Box(4) i64)`, action `Box(6) f32`; the
//! model copies `ft` into the command and tags it with the step seq. No
//! camera, no pacing: one step as fast as env and model allow. Two
//! processes: the env server (a child) and `Model::run` with the model in
//! process (v1's `direct` topology, bench2's v1 best).
//!
//! Loop latency is bench2's `cmd_age`: the time the env receives an action
//! minus the capture time of the observation it was computed from.
//!
//! `--lanes N` (N > 1) is the vectorised case: a vector env of N lanes, one
//! message per step carrying N-lane slabs, and a model with a batched
//! corner (one `predict_batch` of N observations per step).
//!
//! ```text
//! cargo run --release -p rlmesh-gym --example lockstep_bench -- \
//!     --transport grpc|lockstep [--lanes 1] [--warmup 3] [--measure 30]
//! ```
//!
//! It spawns its own env child, prints one JSON line and exits.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unwrap_used,
    unsafe_code
)]

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

use rlmesh::prelude::*;
use rlmesh::spaces::{self, BoxOptions, Dict};
use rlmesh::{DType, EnvServer, EnvServerOptions, Model, ModelOptions, Tensor};

fn now_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: clock_gettime writes the timespec it is given.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

struct Args(HashMap<String, String>);

impl Args {
    fn parse() -> Args {
        let raw: Vec<String> = std::env::args().skip(1).collect();
        let mut map = HashMap::new();
        let mut it = raw.iter();
        while let Some(key) = it.next() {
            if let Some(name) = key.strip_prefix("--") {
                map.insert(name.to_string(), it.next().cloned().unwrap_or_default());
            } else {
                map.insert("role".to_string(), key.clone());
            }
        }
        Args(map)
    }
    fn get(&self, key: &str, default: &str) -> String {
        self.0
            .get(key)
            .cloned()
            .unwrap_or_else(|| default.to_string())
    }
    fn num<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        self.0
            .get(key)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
}

fn f32_box(n: i64) -> rlmesh::Result<Space> {
    Ok(spaces::Box::new(
        f64::NEG_INFINITY,
        f64::INFINITY,
        vec![n],
        BoxOptions::default(),
    )?
    .into())
}

fn spaces_t1() -> rlmesh::Result<(Space, Space)> {
    let stamp = spaces::Box::new(
        f64::NEG_INFINITY,
        f64::INFINITY,
        vec![4],
        BoxOptions {
            dtype: DType::Int64,
        },
    )?;
    let obs = Dict::new(vec![("ft", f32_box(18)?), ("stamp", stamp.into())])?;
    Ok((obs.into(), f32_box(6)?))
}

/// The env side's measurements.
struct Clock {
    capture_by_seq: Vec<u64>,
    /// (ft capture ns, action arrival ns), measured window only.
    applied: Vec<(u64, u64)>,
    start_ns: u64,
    measure_from_ns: u64,
    end_ns: u64,
    seq: u64,
    out: String,
    flushed: bool,
    lanes: usize,
    warmup: f64,
    measure: f64,
}

impl Clock {
    fn new(args: &Args, lanes: usize) -> Self {
        Self {
            capture_by_seq: Vec::with_capacity(4_000_000),
            applied: Vec::with_capacity(4_000_000),
            start_ns: 0,
            measure_from_ns: u64::MAX,
            end_ns: u64::MAX,
            seq: 0,
            out: args.get("out", "/tmp/lockstep_bench.json"),
            flushed: false,
            lanes,
            warmup: args.num("warmup", 3.0),
            measure: args.num("measure", 30.0),
        }
    }

    fn reset(&mut self) {
        self.start_ns = now_ns();
        self.measure_from_ns = self.start_ns + (self.warmup * 1e9) as u64;
        self.end_ns = self.measure_from_ns + (self.measure * 1e9) as u64;
    }

    /// One observation's stamp: `[capture_ns, 0, 0, seq]`.
    fn capture(&mut self) -> (u64, Vec<u8>, Vec<u8>) {
        let capture_ns = now_ns();
        let seq = self.seq;
        self.seq += 1;
        self.capture_by_seq.push(capture_ns);
        let t = (capture_ns % 1_000_000) as f32 * 1e-6;
        let ft: Vec<u8> = (0..18).flat_map(|i| (t + i as f32).to_le_bytes()).collect();
        let stamp: Vec<u8> = [capture_ns as i64, 0, 0, seq as i64]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        (seq, ft, stamp)
    }

    fn arrive(&mut self, action: &[u8]) {
        let arrival = now_ns();
        let seq = f32::from_le_bytes(action[0..4].try_into().unwrap()) as usize;
        let capture = self.capture_by_seq[seq.min(self.capture_by_seq.len() - 1)];
        if arrival >= self.measure_from_ns && arrival < self.end_ns {
            self.applied.push((capture, arrival));
        }
    }

    fn done(&mut self) -> bool {
        let done = now_ns() >= self.end_ns;
        if done {
            self.flush();
        }
        done
    }

    fn flush(&mut self) {
        if self.flushed {
            return;
        }
        self.flushed = true;
        let measure = self.measure;
        let mut ages: Vec<f64> = self
            .applied
            .iter()
            .map(|(c, a)| (a - c) as f64 / 1e6)
            .collect();
        ages.sort_by(f64::total_cmp);
        let at = |q: f64| {
            if ages.is_empty() {
                return f64::NAN;
            }
            ages[((q * ages.len() as f64).ceil() as usize).clamp(1, ages.len()) - 1]
        };
        let steps = self.applied.len() as f64;
        let json = format!(
            "{{\"lanes\":{},\"steps_per_s\":{:.0},\"lane_steps_per_s\":{:.0},\"p50_ms\":{:.4},\"p99_ms\":{:.4},\"p999_ms\":{:.4},\"max_ms\":{:.4},\"steps\":{}}}",
            self.lanes,
            steps / measure,
            steps * self.lanes as f64 / measure,
            at(0.5),
            at(0.99),
            at(0.999),
            at(1.0),
            self.applied.len()
        );
        std::fs::write(&self.out, json).expect("write result");
    }
}

/// The T1 embodiment (one lane).
struct Embodiment {
    obs: Space,
    act: Space,
    clock: Clock,
}

impl Embodiment {
    fn observation(&mut self) -> rlmesh::Result<Value> {
        let (_, ft, stamp) = self.clock.capture();
        let mut obs = Metadata::new();
        obs.insert("ft", Tensor::new(&ft, &[18], DType::Float32)?);
        obs.insert("stamp", Tensor::new(&stamp, &[4], DType::Int64)?);
        Ok(Value::Dict(obs))
    }
}

impl Env for Embodiment {
    fn observation_space(&self) -> Space {
        self.obs.clone()
    }
    fn action_space(&self) -> Space {
        self.act.clone()
    }
    fn id(&self) -> Option<String> {
        Some("TeleopEmbodiment-v0".to_string())
    }
    fn reset(
        &mut self,
        _seed: Option<i64>,
        _options: Option<&InfoDict>,
    ) -> rlmesh::Result<(Value, InfoDict)> {
        self.clock.reset();
        Ok((self.observation()?, InfoDict::new()))
    }
    fn step(
        &mut self,
        action: Option<Value>,
    ) -> rlmesh::Result<(Value, f64, bool, bool, InfoDict)> {
        let action = action.expect("action");
        self.clock.arrive(action.as_tensor()?.buffer());
        let obs = self.observation()?;
        let truncated = self.clock.done();
        Ok((obs, 0.0, false, truncated, InfoDict::new()))
    }
}

/// The vectorised T1: N lanes stepped in lockstep, one stamp per step.
struct VectorEmbodiment {
    obs: Space,
    act: Space,
    lanes: usize,
    clock: Clock,
}

impl VectorEmbodiment {
    fn observation(&mut self) -> rlmesh::Result<Value> {
        let (_, ft, stamp) = self.clock.capture();
        let mut obs = Metadata::new();
        let n = self.lanes;
        obs.insert("ft", Tensor::new(&ft.repeat(n), &[n, 18], DType::Float32)?);
        obs.insert(
            "stamp",
            Tensor::new(&stamp.repeat(n), &[n, 4], DType::Int64)?,
        );
        Ok(Value::Dict(obs))
    }
}

impl VectorEnv for VectorEmbodiment {
    fn num_envs(&self) -> u64 {
        self.lanes as u64
    }
    fn single_observation_space(&self) -> Space {
        self.obs.clone()
    }
    fn single_action_space(&self) -> Space {
        self.act.clone()
    }
    fn id(&self) -> Option<String> {
        Some("TeleopEmbodimentVec-v0".to_string())
    }
    fn metadata(&self) -> Option<Metadata> {
        let mut metadata = Metadata::new();
        metadata.insert("autoreset_mode", "NextStep");
        Some(metadata)
    }
    fn reset(
        &mut self,
        _seed: Option<VectorSeed>,
        _options: Option<&InfoDict>,
    ) -> rlmesh::Result<(Value, InfoDict)> {
        self.clock.reset();
        Ok((self.observation()?, InfoDict::new()))
    }
    #[allow(clippy::type_complexity)]
    fn step(
        &mut self,
        actions: Value,
    ) -> rlmesh::Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        self.clock.arrive(actions.as_tensor()?.buffer());
        let obs = self.observation()?;
        let truncated = self.clock.done();
        let n = self.lanes;
        Ok((
            obs,
            vec![0.0; n],
            vec![false; n],
            vec![truncated; n],
            InfoDict::new(),
        ))
    }
}

/// The trivial controller: copies ft into the command, tagged with the seq.
struct Controller;

fn command(observation: &Value) -> rlmesh::Result<[f32; 6]> {
    let dict = observation.as_dict()?;
    let stamp = dict.get("stamp").expect("stamp").as_tensor()?.buffer();
    let seq = i64::from_le_bytes(stamp[24..32].try_into().unwrap());
    let ft = dict.get("ft").expect("ft").as_tensor()?.buffer();
    let f = |i: usize| f32::from_le_bytes(ft[i * 4..i * 4 + 4].try_into().unwrap());
    Ok([seq as f32, f(0), f(1), f(2), f(3), f(4)])
}

fn action_value(action: &[f32; 6]) -> rlmesh::Result<Value> {
    let bytes: Vec<u8> = action.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok(Value::Tensor(Tensor::new(&bytes, &[6], DType::Float32)?))
}

impl Policy for Controller {
    const CORNERS: Corners = Corners::PREDICT.union(Corners::PREDICT_BATCH);

    fn predict(&mut self, observation: Value) -> rlmesh::Result<Value> {
        action_value(&command(&observation)?)
    }

    fn predict_batch(&mut self, observations: Vec<Value>) -> rlmesh::Result<Vec<Value>> {
        observations
            .iter()
            .map(|o| action_value(&command(o)?))
            .collect()
    }
}

fn run_env(args: &Args) -> rlmesh::Result<()> {
    let lanes: usize = args.num("lanes", 1);
    let (obs, act) = spaces_t1()?;
    let address = args.get("addr", "tcp://127.0.0.1:0");
    let options = EnvServerOptions {
        address: Some(address),
        ..Default::default()
    };
    let clock = Clock::new(args, lanes);
    let server = if lanes > 1 {
        EnvServer::vector(
            VectorEmbodiment {
                obs,
                act,
                lanes,
                clock,
            },
            options,
        )?
    } else {
        EnvServer::new(Embodiment { obs, act, clock }, options)?
    };
    println!("{}", server.address());
    match server.serve() {
        Err(rlmesh::Error::Cancelled) => Ok(()),
        other => other,
    }
}

fn run(args: &Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let transport = args.get("transport", "grpc");
    let addr = match transport.as_str() {
        "grpc" => "tcp://127.0.0.1:0",
        "lockstep" => "lockstep://127.0.0.1:0/embodiment",
        other => return Err(format!("unknown transport {other}").into()),
    };
    let out = format!("/tmp/lockstep_bench-{}.json", std::process::id());
    let mut child = Command::new(std::env::current_exe()?)
        .args(["env", "--addr", addr, "--out", &out])
        .args(["--lanes", &args.get("lanes", "1")])
        .args(["--warmup", &args.get("warmup", "3")])
        .args(["--measure", &args.get("measure", "30")])
        .stdout(Stdio::piped())
        .spawn()?;
    /// Stops the env child whatever happens to the run.
    struct Reap(std::process::Child);
    impl Drop for Reap {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let stdout = child.stdout.take().ok_or("child stdout")?;
    let _reap = Reap(child);
    let mut address = String::new();
    BufReader::new(stdout).read_line(&mut address)?;
    let address = address.trim().to_string();
    let lanes: usize = args.num("lanes", 1);
    let model = Model::new(Controller, ModelOptions::default())?;
    let result = model.run(
        address.as_str(),
        rlmesh::RunOptions {
            // A vector env truncates every lane on the same step.
            episodes: Some(lanes as u64),
            // NEXT_STEP (the vector env) owns its truncation.
            max_episode_steps: (lanes == 1).then_some(u64::MAX / 4),
            ..Default::default()
        },
    )?;
    model.close()?;
    let measured = std::fs::read_to_string(&out)?;
    let _ = std::fs::remove_file(&out);
    println!(
        "{{\"transport\":\"{transport}\",\"total_steps\":{},\"result\":{measured}}}",
        result.total_steps()
    );
    Ok(())
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.get("role", "run").as_str() {
        "env" => run_env(&args)?,
        _ => run(&args)?,
    }
    Ok(())
}
