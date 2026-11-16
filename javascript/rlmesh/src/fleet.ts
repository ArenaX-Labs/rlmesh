// The lobby and scene runs: unreleased, fenced behind the addon's `fleet` feature, as Python's
// `rlmesh.realtime._fleet` is. Only an addon built with it has them; a release addon does not, and
// the package exports none of this. build.mjs bundles it as dist/fleet.js (the package plus these)
// for the native tests, and the npm package leaves that file out.

import { readFile } from 'node:fs/promises';
import { call, native, robots, wait, type Addon, type NativeRobot, type Robot } from './native.js';

export * from './node.js';

/** A refusal in a run's state. */
export interface Refusal {
  status: string;
  message: string;
}

/** A scene run's state: its stage, each member's, and why it failed. */
export interface RunState {
  run: string;
  version: number;
  stage: string;
  members: { name: string; stage: string; error: Refusal | null }[];
  error: Refusal | null;
}

interface NativeScene {
  readonly run: string;
  address(): Promise<string | null>;
  status(): Promise<RunState>;
  waitUntil(stage: string, timeoutS: number): Promise<RunState>;
  arm(): Promise<RunState>;
  stop(timeoutS: number): Promise<RunState>;
  close(timeoutS: number): Promise<void>;
}

/** What only an addon built with `fleet` has. */
interface FleetAddon {
  robotFromLobby(home: string | null, name: string, robotJson: string, timeoutS: number | null): Promise<NativeRobot>;
  runScene(home: string | null, sceneToml: string, manualArm: boolean, timeoutS: number): Promise<NativeScene>;
}

/** A lobby or scene call of the addon, which only a `fleet` build has (unreleased). */
function fleet<K extends keyof FleetAddon>(name: K): FleetAddon[K] {
  const addon = native() as Addon & Partial<FleetAddon>;
  const f = addon[name];
  if (f === undefined) throw new Error(`${name}: this rlmesh build has no lobby or scenes (the unreleased fleet feature)`);
  return f.bind(addon) as FleetAddon[K];
}

/**
 * Wait in the home rlmeshd's lobby (`home`: `'rig-pi.local'`; default this machine's) until a
 * scene's run assigns this robot; it then joins the run and declares `parts`. A refused join (its
 * pin, another map) is reported to the daemon and the wait goes on. Run until `leaving` is set,
 * close, and wait again.
 */
export async function waitInLobby(
  name: string,
  parts: Record<string, unknown>,
  opts: { home?: string; timeout?: number; onHandover?: (group: string, from: string, to: string) => void } = {},
): Promise<Robot> {
  const robot = await wait(() => fleet('robotFromLobby')(opts.home ?? null, name, JSON.stringify(parts), opts.timeout ?? null));
  return robots.adopt(robot, opts.onHandover, opts.timeout ?? 10);
}

/**
 * Why `robot` should leave its scene's run: `scene_ended` or `scene_failed` when the run closes,
 * `robot_lost` when the host lost it (wait in the lobby again: its run re-sends its join),
 * `superseded` when another process took its name over (stop); `null` while it is in the run, and
 * outside a scene.
 */
export function leaving(robot: Robot): string | null {
  return call(() => (robots.native(robot) as NativeRobot & { readonly leaving?: string | null }).leaving) ?? null;
}

/** A native run state, its absent errors `null`. */
function runState(s: RunState): RunState {
  return { ...s, members: s.members.map((m) => ({ ...m, error: m.error ?? null })), error: s.error ?? null };
}

/**
 * A scene's run on the home rlmeshd, started by `runScene`. Its members join `address()` (this
 * script's own policy among them). `close()` stops the run unless it ended.
 */
export class Scene implements AsyncDisposable {
  constructor(private native: NativeScene) {}

  /** The run's id (`r-7k2m`), as `rlmesh scene stop` takes it. */
  get run(): string {
    return this.native.run;
  }

  /** Where the run's members join, once its host is up. */
  address(): Promise<string | null> {
    return wait(async () => (await this.native.address()) ?? null);
  }

  /** The run's state now. */
  status(): Promise<RunState> {
    return wait(async () => runState(await this.native.status()));
  }

  /** Resolve once the run is at `stage` (`'running'`); a run that ends first rejects with its error. */
  waitUntil(stage: string, opts: { timeout?: number } = {}): Promise<RunState> {
    return wait(async () => runState(await this.native.waitUntil(stage, opts.timeout ?? 30)));
  }

  /** Arm a run that waits for it (`arm: 'manual'`, or after a lost member rejoined). */
  arm(): Promise<RunState> {
    return wait(async () => runState(await this.native.arm()));
  }

  /** Stop the run and wait for it to end; this process's policies' runs return. */
  stop(opts: { timeout?: number } = {}): Promise<RunState> {
    return wait(async () => runState(await this.native.stop(opts.timeout ?? 10)));
  }

  /** Stop the run unless it ended, and leave the lobby. */
  close(): Promise<void> {
    return wait(() => this.native.close(10));
  }

  [Symbol.asyncDispose](): Promise<void> {
    return this.close();
  }
}

/**
 * Run the scene file at `path` on its robots' home rlmeshd (`home`; default this machine's). The
 * daemon checks it, claims its robots, assigns them from its lobby and arms once every member
 * declared and bound (`arm: 'manual'`: on `Scene.arm`). A refused scene rejects with
 * `SessionRefused` and its code (`INVALID_REQUEST`, `NOT_AUTHORIZED`, `ROBOT_BUSY`).
 */
export async function runScene(
  path: string,
  opts: { home?: string; arm?: 'auto' | 'manual'; timeout?: number } = {},
): Promise<Scene> {
  const arm = opts.arm ?? 'auto';
  if (arm !== 'auto' && arm !== 'manual') throw new RangeError(`arm ${String(arm)}: auto or manual`);
  const text = await readFile(path, 'utf8');
  return new Scene(await wait(() => fleet('runScene')(opts.home ?? null, text, arm === 'manual', opts.timeout ?? 10)));
}
