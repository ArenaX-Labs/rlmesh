// A Node robot and a Node policy in one session: the UR10e of the conformance fixtures, faked (commanded
// joints become measured ones), driven for two seconds by a policy that holds the arm where it is.
//
//   mise run build:js:native && (cd javascript/rlmesh && node build.mjs --node-only)
//   node examples/node/hold.mjs
//
// Plain try/finally, so it runs on Node 22; on Node 24 `await using` closes each one for you.
import { readFileSync } from 'node:fs';
import { Host, Policy, Rate, Robot } from '../../javascript/rlmesh/dist/node.js';

const UR10E = JSON.parse(readFileSync(new URL('../../crates/rlmesh-session-core/tests/fixtures/robots/ur10e.json', import.meta.url), 'utf8'));
const JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'];
const HOLD = {
  hz: 50,
  inputs: { q: { kind: 'joints', at: 'arm', fields: ['pos'], names: JOINTS } },
  actions: { arm: { kind: 'joints', at: 'arm', command: 'position', names: JOINTS, hz: 100, horizon: 10 } },
};

const host = await Host.open();
const robot = await Robot.connect('ur10e', UR10E, { session: host });
const policy = new Policy(HOLD, { session: host, name: 'hold' });
let running = true;
try {
  await robot.arm();

  // The robot's control loop: publish its sensors, play what its edge says is due.
  let q = Float32Array.of(0, -1.57, 1.57, -1.57, -1.57, 0);
  const zeros = new Float32Array(6);
  const control = (async () => {
    const rate = new Rate(500);
    while (running) {
      robot.publish('arm', { pos: q, vel: zeros, effort: zeros });
      robot.publish('wrist/ft', { force: [0, 0, 9.8], torque: [0, 0, 0] });
      robot.publish('gripper', [1]);
      const due = robot.due('arm'); // { mode, value: { dtype, shape, data }, ... } or null
      if (due?.value) q = Float32Array.from(due.value.data);
      await rate.sleep();
    }
  })();

  console.log(String(await policy.bind('ur10e'))); // the binding table
  const stats = await policy.run(
    async (obs) => {
      const arm = new Float32Array(60); // [horizon 10, 6 joints]
      for (let k = 0; k < 10; k++) arm.set(obs.inputs.q.data, k * 6);
      return { arm: { data: arm, shape: [10, 6] } };
    },
    { duration: 2 },
  );
  running = false;
  await control;
  console.log(`${stats.completions} starts completed, ${stats.chunksSent} chunks sent, p99 ${stats.latencyP99Ms?.toFixed(2)} ms`);
  console.log('edge:', robot.edgeStats('arm'));
} finally {
  running = false;
  await policy.close();
  await robot.close();
  await host.close();
}
