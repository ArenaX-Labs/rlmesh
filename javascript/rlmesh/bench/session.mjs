// The native binding's costs on loopback, the twin of bench/session.py (run both: `mise run bench:js:native`).
//
// One process: a host, the §14.1 UR10e and a policy. The robot's control loop runs at 1 kHz
// (publish the arm, the wrench and the gripper, tick both edges, sleep); its cameras publish
// 640x480 and 1280x720 frames at 30 Hz. The policy reads the joints, the wrench, the wrist camera
// (resized to 96x96) and the task at 50 Hz and holds the arm. Prints one JSON line:
//
//   loop_work_us    one control tick's publish + due calls
//   loop_late_us    how late each tick woke (Rate.sleep's return), and periods missed
//   predict_ms      completion latency, snapshot to predict's answer (the run loop's own measure)
//   publish_us      `publish('arm', ...)` alone, in a tight loop
//   cpu_s           process CPU over the run
import { readFileSync } from 'node:fs';
import * as rlmesh from '../dist/node.js';

const SECONDS = Number(process.env.BENCH_SECONDS ?? 10);
const SPEC = new URL('../../../crates/rlmesh-session-core/tests/fixtures/', import.meta.url);
const UR10E = JSON.parse(readFileSync(new URL('robots/ur10e.json', SPEC), 'utf8'));
const JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'];
const POLICY = {
  hz: 50,
  inputs: {
    q: { kind: 'joints', at: 'arm', fields: ['pos'], names: JOINTS },
    ft: { kind: 'wrench', at: 'wrist/ft', frame: 'tool0' },
    wrist: { kind: 'camera', at: 'wrist/cam', size: [96, 96] },
    prompt: { kind: 'text', at: '/task' },
  },
  actions: { arm: { kind: 'joints', at: 'arm', command: 'position', names: JOINTS, hz: 100, horizon: 10 } },
  allow: ['resize', 'crop'],
};
const HOME = Float32Array.of(0, -1.57, 1.57, -1.57, -1.57, 0);

const quantiles = (xs) => {
  const s = Float64Array.from(xs).sort();
  const at = (q) => Math.round(s[Math.min(s.length - 1, Math.floor(q * s.length))] * 10) / 10;
  return { p50: at(0.5), p99: at(0.99), max: at(1) };
};

const cpu0 = process.cpuUsage();
await using host = await rlmesh.Host.open();
await using robot = await rlmesh.Robot.connect('ur10e', UR10E, { session: host });
await robot.arm();
await using policy = new rlmesh.Policy(POLICY, { session: host, name: 'bench' });

let stop = false;
const work = [];
const late = [];
const control = (async () => {
  const rate = new rlmesh.Rate(1000);
  let q = HOME;
  const zeros = new Float32Array(6);
  const force = Float32Array.of(0, 0, 9.8);
  const torque = new Float32Array(3);
  const gripper = Float32Array.of(1);
  while (!stop) {
    const t0 = process.hrtime.bigint();
    robot.publish('arm', { pos: q, vel: zeros, effort: zeros });
    robot.publish('wrist/ft', { force, torque });
    robot.publish('gripper', gripper);
    const due = robot.due('arm');
    if (due?.value) q = due.value.data;
    robot.due('gripper');
    work.push(Number(process.hrtime.bigint() - t0) / 1e3);
    late.push((await rate.sleep()) / 1e3);
  }
  return rate.missed;
})();
const cameras = (async () => {
  const wrist = new Uint8Array(480 * 640 * 3);
  const scene = new Uint8Array(720 * 1280 * 3);
  const rate = new rlmesh.Rate(30);
  while (!stop) {
    robot.publish('wrist/cam', wrist);
    robot.publish('scene/cam', scene);
    robot.publish('task', 'hold the arm');
    await rate.sleep();
  }
})();

await policy.bind('ur10e');
const hold = { arm: { data: new Float32Array(60), shape: [10, 6] } };
const stats = await policy.run(
  (obs) => {
    hold.arm.data.set(obs.inputs.q.data);
    return hold;
  },
  { duration: SECONDS },
);
stop = true;
const missed = await control;
await cameras;

// publish alone, in a tight loop (the edge is idle: no lease).
await policy.release();
const n = 100_000;
const zeros = new Float32Array(6);
const t0 = process.hrtime.bigint();
for (let i = 0; i < n; i++) robot.publish('arm', { pos: HOME, vel: zeros, effort: zeros });
const publishUs = Number(process.hrtime.bigint() - t0) / 1e3 / n;

const cpu = process.cpuUsage(cpu0);
console.log(
  JSON.stringify({
    binding: 'node',
    seconds: SECONDS,
    loop_ticks: work.length,
    loop_work_us: quantiles(work),
    loop_late_us: { ...quantiles(late), missed },
    predict_ms: { completions: stats.completions, p50: stats.latencyP50Ms, p99: stats.latencyP99Ms, max: stats.latencyMaxMs },
    skipped: stats.skipped,
    publish_us: Math.round(publishUs * 100) / 100,
    cpu_s: Math.round(((cpu.user + cpu.system) / 1e6) * 100) / 100,
  }),
);
