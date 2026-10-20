// The browser example: a live RLMesh v2 session (the example robot `arm7`) through rlmesh-gateway.
// `mise run demo:web`, then open http://127.0.0.1:18080/.
import * as rlmesh from '../../javascript/rlmesh/src/browser.js';
import type { EdgeMode, Frame, Lease, Mode, Sample, TrafficClass } from '../../javascript/rlmesh/src/browser.js';

const ARM = 'arm7/arm';
const CAM = 'arm7/scene/cam';
const JOINTS = 7;
const LIMIT = 2.6;

const $ = (id: string) => document.getElementById(id)!;
const fmt = (x: number | undefined, d = 3) => (x == null || !Number.isFinite(x) ? '-' : x.toFixed(d));
const logLines: string[] = [];
const log = (s: string) => {
  logLines.unshift(`${new Date().toISOString().slice(11, 23)} ${s}`);
  logLines.length = Math.min(logLines.length, 80);
  $('log').textContent = logLines.join('\n');
};
/** Timestamped mode and edge transitions, for automated checks. */
const timeline: { t: number; what: string; value: string }[] = [];
const mark = (what: string, value: string) => timeline.push({ t: performance.now(), what, value });

const t0 = performance.now();
// A token for a non-loopback gateway rides in the URL fragment (never sent to the server): /#token=...
const token = new URLSearchParams(location.hash.slice(1)).get('token') ?? undefined;
// The operator class lets this page take over from the robot's model policy.
const session = await rlmesh.connect(undefined, { priority: 'operator', ...(token ? { token } : {}) });
const sizes = await fetch('./size.json').then((r) => r.json()).catch(() => ({}));
const showMe = () => ($('me').textContent = `(you: ${session.id}, ${session.priority})`);
showMe();
$('conn').textContent = `connected in ${fmt(session.load.connect_ms, 0)} ms`;
log(`connected to ${session.sessionId}: ${session.streams.map((s) => `${s.path}[${s.class} ${s.rate_hz} Hz]`).join(', ')}`);
const sockets: Record<string, boolean> = { control: true, state: true, bulk: true };
const showConn = () => {
  const down = Object.entries(sockets).filter(([, up]) => !up).map(([c]) => c);
  $('conn').textContent = down.length ? `reconnecting: ${down.join(', ')}` : `connected in ${fmt(session.load.connect_ms, 0)} ms`;
};
session.on('disconnected', (e) => { sockets[e.cls] = false; showConn(); log(`${e.cls} socket closed (attempt ${e.attempt})`); });
session.on('reconnected', (e) => { sockets[e.cls] = true; showConn(); showMe(); log(`${e.cls} socket reconnected`); });
session.on('worker_restart', (e) => { showMe(); log(`${e.cls} worker restarted in ${fmt(e.downtime_ms, 0)} ms (#${e.restarts}): ${e.reason}`); });

// ---------------------------------------------------------------- wasm panel
{
  const w = session.load.workers;
  const rows: [string, string][] = [
    ['size', `${(sizes.wasm / 1024).toFixed(1)} KB raw, ${(sizes.wasm_gzip9 / 1024).toFixed(1)} KB gzip (+ ${(sizes.glue_js / 1024).toFixed(1)} KB glue)`],
    ['compile (main)', `${fmt(session.load.compile_ms, 1)} ms, once; Module posted to each Worker`],
    ['instantiate', (['control', 'state', 'bulk'] as const).map((c) => `${c} ${fmt(w[c]?.instantiate_ms, 2)} ms`).join(' · ')],
    ['workers ready', `${fmt(session.load.workers_ready_ms, 1)} ms after start`],
    ['exports', `${w.state?.module_exports?.length} module exports, typeof WebAssembly = ${w.state?.typeof_WebAssembly}`],
    ['isolation', `crossOriginIsolated=${session.load.cross_origin_isolated} → latest() via ${session.load.shared_memory ? 'SharedArrayBuffer seqlock' : 'postMessage'}`],
    ['page → session', `${fmt(performance.now() - t0, 0)} ms`],
  ];
  $('wasm').innerHTML = rows.map(([k, v]) => `<span class="mut">${k}</span><span>${v}</span>`).join('');
}

// ---------------------------------------------------------------- camera
const cam = ($('cam') as HTMLCanvasElement).getContext('2d')!;
let camFrames = 0;
// The frame event hands a borrowed frame (pooled bytes + ImageBitmap), released when the listener returns.
session.on('frame', (b) => {
  if (b.bitmap) cam.drawImage(b.bitmap, 0, 0, 640, 480);
  camFrames++;
});
session.latest(CAM); // start the subscription
setInterval(() => {
  using b = session.peek<Frame>(CAM); // borrowed until the end of this block
  const f = b?.value;
  if (f) $('camInfo').textContent = `#${f.seq} · ${(f.data.length / 1024).toFixed(1)} KB ${f.kind.toUpperCase()} · age ${fmt(f.age_ms, 1)} ms · decode ${fmt(f.decode_ms, 1)} ms`;
}, 250);

// ---------------------------------------------------------------- joint plot via the async iterator
const N = 5000;
const hist = Array.from({ length: JOINTS }, () => new Float32Array(N));
let head = 0;
const joints = session.subscribe(ARM, { maxQueue: 2048 });
(async () => {
  for await (const s of joints as AsyncIterable<Sample>) {
    for (let j = 0; j < JOINTS; j++) hist[j]![head] = s.data[j] ?? 0;
    head = (head + 1) % N;
  }
})();
// The edge mode comes from the arm's applied stream.
session.latest(`${ARM}/_applied`);
const plot = ($('plot') as HTMLCanvasElement).getContext('2d')!;
const colors = ['#4c8dff', '#2fbf6a', '#e9b022', '#e5484d', '#b37feb', '#36cfc9', '#f0f0f0'];
let plotOn = true;
let busyMs = 0;
const draw = () => {
  // Debug: simulate a busy UI thread (e.g. a heavy render) for `busyMs` per animation frame.
  if (busyMs) { const e = performance.now() + busyMs; while (performance.now() < e); }
  if (!plotOn) return void requestAnimationFrame(draw);
  const W = 1200, H = 300;
  plot.clearRect(0, 0, W, H);
  plot.strokeStyle = '#2a3340';
  plot.beginPath(); plot.moveTo(0, H / 2); plot.lineTo(W, H / 2); plot.stroke();
  for (let j = 0; j < JOINTS; j++) {
    plot.strokeStyle = colors[j] ?? '#fff';
    plot.lineWidth = 2;
    plot.beginPath();
    for (let x = 0; x < W; x++) {
      const i = (head + Math.floor((x / W) * N)) % N;
      const y = H / 2 - ((hist[j]![i] ?? 0) / 2.8) * (H / 2);
      x ? plot.lineTo(x, y) : plot.moveTo(x, y);
    }
    plot.stroke();
  }
  requestAnimationFrame(draw);
};
requestAnimationFrame(draw);

// ---------------------------------------------------------------- status, clock, stats
let mode: Mode | '' = '';
let edge: EdgeMode | '' = '';
const showMode = (m: Mode) => {
  if (m !== mode) { log(`session ${m.toUpperCase()}`); mark('mode', m); }
  mode = m;
  $('mode').textContent = m;
  $('mode').className = `badge ${m}`;
};
showMode(session.mode);
session.on('state', (s) => showMode(s.mode));
session.on('status', (st) => {
  if (st.path !== ARM) return;
  showMode(st.mode);
  if (st.edge && st.edge !== edge) { edge = st.edge; mark('edge', edge); log(`arm edge ${edge.toUpperCase()}`); }
  $('edge').textContent = st.edge ?? '?';
  $('edge').className = `badge ${st.edge ?? 'unknown'}`;
  $('holder').textContent = st.holder ? `${st.holder} (${st.priority})` : '(none)';
  $('host').textContent = `epoch ${st.epoch} · token ${st.token || '-'}${st.ours ? ' (ours)' : ''}${st.estop_latched ? ' · ESTOP latched' : ''}`;
});
session.on('lease', (ev) => log(`lease ${ev.event} ${ev.path} token=${ev.token}${ev.reason ? ` (${ev.reason})` : ''}`));
session.on('fence', (f) => log(`fence${f.local ? ' (local check)' : ''}: ${f.reason}`));
session.on('console', (c) => log(`[${c.cls} ${c.lvl}] ${c.msg.slice(0, 300)}`));
session.on('lease_state', (m) => {
  const l = m.leases[ARM];
  $('lease').textContent = l ? `${l.state} token=${l.token} remaining ${fmt(l.remaining_ms, 0)} ms` : 'idle';
  $('cmds').textContent = `${fmt(l?.hz, 1)} Hz · sent ${m.cmd.sent} · refused locally ${m.cmd.refused} · gated ${m.cmd.gated}${m.cmd.lastRefusal ? ` (${m.cmd.lastRefusal})` : ''} · heartbeats ${m.cmd.heartbeats ? 'on' : 'PAUSED'} (${m.cmd.hbSent})`;
});
setInterval(() => {
  const c = session.clock;
  $('clock').innerHTML = [
    ['offset', `${fmt(c.offset_ms, 4)} ms (the host's mapping: ${fmt(c.host_offset_ms, 4)} ms)`],
    ['rtt', `${fmt(c.rtt_ms, 4)} ms (heartbeat round trip through the gateway)`],
    ['error', `± ${fmt(c.error_ms, 4)} ms`],
    ['samples', `${c.samples}`],
  ].map(([k, v]) => `<span class="mut">${k}</span><span>${v}</span>`).join('');
}, 500);
setInterval(async () => {
  const s = await session.stats();
  const rows = [
    ...s.state.streams.map((x: any) => ({ ...x, dec: s.state.decode_us_mean })),
    ...s.bulk.streams.map((x: any) => ({ ...x, dec: s.bulk.decode_us_mean })),
  ].map((x: any) => `<tr><td>${x.path}</td><td>${fmt(x.rate_hz, 1)}</td><td>${fmt(x.latency_p50_ms)}</td><td>${fmt(x.latency_p99_ms)}</td><td>${fmt(x.latency_p999_ms)}</td><td>${fmt(x.age_p50_ms)}</td><td>${fmt(x.age_p99_ms)}</td><td>${fmt(x.late_pct, 3)}</td><td>${x.lost}</td><td>${fmt(x.dec, 1)}</td></tr>`);
  const m = s.main.sample_latency;
  rows.push(`<tr class="mut"><td>${ARM} @main</td><td></td><td>${fmt(m.p50_ms)}</td><td>${fmt(m.p99_ms)}</td><td></td><td></td><td></td><td>${fmt(m.late_pct, 2)}</td><td></td><td></td></tr>`);
  $('stats').querySelector('tbody')!.innerHTML = rows.join('');
}, 1000);

// ---------------------------------------------------------------- control
let lease: Lease | null = null;
const target = new Float64Array(JOINTS);
const take = $('take') as HTMLButtonElement, rel = $('release') as HTMLButtonElement;
let lastToken = 0;
const inputs: HTMLInputElement[] = [];
const showTargets = () => target.forEach((v, j) => {
  const inp = inputs[j];
  if (inp) { inp.value = String(v); inp.parentElement!.querySelector('span:last-child')!.textContent = v.toFixed(2); }
});
take.onclick = async () => {
  take.disabled = true;
  try {
    // Start from where the arm is, so taking over does not jump.
    const now = session.latest(ARM);
    if (now) target.set(now.data.subarray(0, JOINTS));
    showTargets();
    lease = await session.lease(ARM);
    lastToken = lease.token;
    log(`took control: token ${lease.token}`);
    rel.disabled = false;
    lease.drive(target, { hz: 100 });
    lease.revoked.then((r) => { log(`our lease was revoked (${r.cause}): ${r.reason}`); lease = null; take.disabled = false; rel.disabled = true; });
  } catch (e: any) {
    log(`lease failed: ${e.message}`);
    take.disabled = false;
  }
};
rel.onclick = () => {
  lease?.release();
  lease = null;
  take.disabled = false;
  rel.disabled = true;
  log('released');
};
($('arm') as HTMLButtonElement).onclick = () => session.arm().then(() => log('re-armed'), (e) => log(`arm: ${e.message}`));
($('pausehb') as HTMLInputElement).onchange = (e) => {
  const p = (e.target as HTMLInputElement).checked;
  session.debug.pauseHeartbeats(p);
  mark('heartbeats', p ? 'paused' : 'resumed');
  log(p ? 'heartbeats paused' : 'heartbeats resumed');
};
$('stale').onclick = () => {
  const tok = lastToken ? lastToken - 1 : 1;
  session.debug.sendStaleCommand(ARM, tok, target);
  log(`sent a command with stale token ${tok}`);
};
const setTarget = () => { if (lease?.active) lease.drive(target, { hz: 100 }); };
const sliders = $('sliders');
for (let j = 0; j < JOINTS; j++) {
  const l = document.createElement('label');
  l.innerHTML = `<span class="mut">q${j}</span><input type="range" min="-${LIMIT}" max="${LIMIT}" step="0.01" value="0"><span>0.00</span>`;
  const inp = l.querySelector('input')!;
  inp.oninput = () => { target[j] = Number(inp.value); l.querySelector('span:last-child')!.textContent = (target[j] ?? 0).toFixed(2); setTarget(); };
  inputs.push(inp);
  sliders.appendChild(l);
}
const keys = 'qwertyu', neg = 'asdfghj';
const held = new Map<number, number>();
const key = (e: KeyboardEvent, down: boolean) => {
  const k = e.key.toLowerCase();
  const j = keys.indexOf(k) >= 0 ? keys.indexOf(k) : neg.indexOf(k);
  if (j < 0) return;
  if (down) held.set(j, keys.includes(k) ? 1 : -1);
  else held.delete(j);
};
setInterval(() => {
  if (!held.size) return;
  for (const [j, dir] of held) target[j] = Math.max(-LIMIT, Math.min(LIMIT, (target[j] ?? 0) + dir * 0.02));
  showTargets();
  setTarget();
}, 20);
addEventListener('keydown', (e) => !e.repeat && key(e, true));
addEventListener('keyup', (e) => key(e, false));

// ---------------------------------------------------------------- hooks for automated verification
(window as any).__rlmesh = {
  session,
  rlmesh,
  get lease() { return lease; },
  get timeline() { return timeline; },
  setTarget: (v: number[]) => { v.forEach((x, i) => (target[i] = x)); showTargets(); setTarget(); },
  take: () => take.click(),
  release: () => rel.click(),
  arm: () => session.arm(),
  pauseHeartbeats: (p: boolean) => { ($('pausehb') as HTMLInputElement).checked = p; ($('pausehb') as HTMLInputElement).dispatchEvent(new Event('change')); },
  stale: () => $('stale').click(),
  trap: (cls: TrafficClass) => session.debug.trap(cls),
  dropSocket: (cls: TrafficClass) => session.debug.dropSocket(cls),
  camFrames: () => camFrames,
  setPlot: (on: boolean) => (plotOn = on),
  setBusy: (ms: number) => (busyMs = ms),
  log: () => logLines.slice(0, 40),
  async measure(seconds = 30) {
    session.debug.resetStats();
    const f0 = camFrames;
    await new Promise((r) => setTimeout(r, seconds * 1000));
    const s = await session.stats();
    return { seconds, cam_frames_drawn: camFrames - f0, ...s, clock: session.clock };
  },
};
