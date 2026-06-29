//! HTTP backend: serves the latest frame + a poll page over `tiny_http`.
//!
//! Runs on its OWN std::thread; the viewer writes a shared latest-frame slot and
//! every request just reads it, so N browser tabs share one frame for free. The
//! browser polls `/frame`; switching source is a `GET /select?i=N` against the
//! shared `selected` atomic the viewer reads each step.

use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;

use tiny_http::{Header, Response, Server};

/// HUD payload, formatted per-backend (JSON for HTTP, a line for the terminal).
///
/// `fps` is the viewer's own draw rate (filled in by [`crate::Viewer::feed_hud`]);
/// every other field is supplied by the caller each step. A sentinel of `0` (or
/// `-1` for `seed`) means "unknown / not applicable" and the backends omit it.
#[derive(Clone)]
pub struct Hud {
    pub step: i64,
    pub reward: f64,
    pub outcome: String,
    /// Viewer draw rate (frames/sec actually painted); set by the viewer, not the caller.
    pub fps: f64,
    /// Smoothed model forward time in ms (the `predict` cost; ~0 on chunk-replay steps).
    pub model_ms: f64,
    /// Smoothed env `step()` time in ms (the simulator cost).
    pub env_ms: f64,
    /// Smoothed eval throughput in env steps/sec (distinct from the draw `fps`).
    pub sps: f64,
    /// Wall-clock seconds since the current episode's reset.
    pub elapsed_s: f64,
    /// 1-based current episode index; `0` when unknown (hand-driven loop).
    pub episode: i64,
    /// Total episodes in the run; `0` when unknown.
    pub episodes: i64,
    /// Current episode seed; `-1` when none/unknown.
    pub seed: i64,
    /// Selected source frame size; `0` until the first frame is fed.
    pub width: u32,
    pub height: u32,
    /// 1-based position within the current action chunk; `0` when not chunking.
    pub chunk_pos: i64,
    /// Action-chunk length (execution horizon); `0`/`1` when not chunking.
    pub chunk_len: i64,
}

impl Default for Hud {
    /// All-unknown HUD: zeroed counters/timings and the `-1` "no seed" sentinel (a
    /// derived default would make `seed` read as a real seed of `0`).
    fn default() -> Self {
        Self {
            step: 0,
            reward: 0.0,
            outcome: String::new(),
            fps: 0.0,
            model_ms: 0.0,
            env_ms: 0.0,
            sps: 0.0,
            elapsed_s: 0.0,
            episode: 0,
            episodes: 0,
            seed: -1,
            width: 0,
            height: 0,
            chunk_pos: 0,
            chunk_len: 0,
        }
    }
}

/// State shared between the viewer (writer) and the terminal / HTTP backends.
pub struct HttpShared {
    /// Latest encoded frame bytes (JPEG or PNG per `content_type`); HTTP only.
    pub latest_frame: Mutex<Vec<u8>>,
    pub hud: Mutex<Hud>,
    /// Source selector labels; set once the sources are known.
    pub sources: Mutex<Vec<String>>,
    /// Index into `sources` the viewer should draw. Written by the terminal key
    /// thread and the HTTP `/select` route; read by both backends.
    pub selected: AtomicUsize,
    /// Set by the terminal key thread when the user asks to quit (q / Esc / Ctrl-C).
    /// Raw mode swallows SIGINT, so the eval reads this each step (via the Python
    /// driver) and raises KeyboardInterrupt to stop the loop.
    pub quit: AtomicBool,
    /// Set by the `n` key / `/skip` route to end the *current episode* early (a soft,
    /// non-failure stop that advances to the next episode) — unlike `quit`, which stops
    /// the whole run. Consumed once per request: the eval swaps it back to false.
    pub skip: AtomicBool,
    /// Content-type for `/frame` matching the encoded `latest_frame` bytes.
    content_type: &'static str,
}

impl HttpShared {
    pub fn new(content_type: &'static str) -> Self {
        Self {
            latest_frame: Mutex::new(Vec::new()),
            hud: Mutex::new(Hud::default()),
            sources: Mutex::new(Vec::new()),
            selected: AtomicUsize::new(0),
            quit: AtomicBool::new(false),
            skip: AtomicBool::new(false),
            content_type,
        }
    }
}

/// Lock helper that recovers from poisoning instead of panicking — a viewer must
/// never take down the eval, and a poisoned display lock is harmless.
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Bind and start serving on `port`. Returns the server handle (call
/// [`Server::unblock`] to stop the accept loop), or `Err` if the bind failed.
pub fn spawn(port: u16, shared: Arc<HttpShared>) -> Result<Arc<Server>, String> {
    let server = Arc::new(Server::http(("127.0.0.1", port)).map_err(|e| e.to_string())?);
    let serve_server = Arc::clone(&server);
    thread::Builder::new()
        .name("rlmesh-viewer-http".to_string())
        .spawn(move || {
            tracing::info!("viewer: serving on http://localhost:{port}");
            for request in serve_server.incoming_requests() {
                let url = request.url().to_string();
                let response = route(&url, &shared);
                let _ = request.respond(response);
            }
        })
        .expect("spawn rlmesh-viewer http thread");
    Ok(server)
}

fn route(url: &str, shared: &HttpShared) -> Response<Cursor<Vec<u8>>> {
    let path = url.split('?').next().unwrap_or("/");
    match path {
        "/" => with_ct(
            Response::from_string(INDEX_HTML),
            "text/html; charset=utf-8",
        ),
        "/frame" => with_ct(
            Response::from_data(lock(&shared.latest_frame).clone()),
            shared.content_type,
        ),
        "/sources.json" => {
            let sources = lock(&shared.sources).clone();
            let selected = shared.selected.load(Ordering::Relaxed);
            let body = serde_json::json!({ "sources": sources, "selected": selected }).to_string();
            with_ct(Response::from_string(body), "application/json")
        }
        "/hud.json" => {
            let hud = lock(&shared.hud).clone();
            let body = serde_json::json!({
                "step": hud.step, "reward": hud.reward, "outcome": hud.outcome, "fps": hud.fps,
                "model_ms": hud.model_ms, "env_ms": hud.env_ms, "sps": hud.sps,
                "elapsed_s": hud.elapsed_s, "episode": hud.episode, "episodes": hud.episodes,
                "seed": hud.seed, "width": hud.width, "height": hud.height,
                "chunk_pos": hud.chunk_pos, "chunk_len": hud.chunk_len,
            })
            .to_string();
            with_ct(Response::from_string(body), "application/json")
        }
        "/select" => {
            if let Some(index) = query_usize(url, "i")
                && index < lock(&shared.sources).len()
            {
                shared.selected.store(index, Ordering::Relaxed);
            }
            with_ct(Response::from_string("{\"ok\":true}"), "application/json")
        }
        "/skip" => {
            // End the current episode early (the eval consumes this on its next step).
            shared.skip.store(true, Ordering::Relaxed);
            with_ct(Response::from_string("{\"ok\":true}"), "application/json")
        }
        _ => Response::from_string("not found").with_status_code(404),
    }
}

fn query_usize(url: &str, key: &str) -> Option<usize> {
    url.split('?')
        .nth(1)?
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(k, _)| *k == key)
        .and_then(|(_, v)| v.parse().ok())
}

fn with_ct(response: Response<Cursor<Vec<u8>>>, content_type: &str) -> Response<Cursor<Vec<u8>>> {
    let header = Header::from_bytes(&b"Content-Type"[..], content_type.as_bytes())
        .expect("static content-type header is valid");
    response.with_header(header)
}

const INDEX_HTML: &str = r#"<!doctype html><html><head><meta charset="utf-8">
<title>rlmesh viewer</title><style>
body{background:#111;color:#ccc;font:14px system-ui,sans-serif;margin:0;text-align:center}
#bar{padding:8px}#hud{padding:4px;color:#9c9;font-variant-numeric:tabular-nums;min-height:1.2em}
img{max-width:100vw;max-height:82vh;image-rendering:pixelated;background:#000}
button{margin:3px;padding:4px 10px;background:#222;color:#ccc;border:1px solid #444;border-radius:4px;cursor:pointer}
button.sel{font-weight:bold;border-color:#888;background:#333}
</style></head><body>
<div id="bar"></div>
<div id="ctl"><button onclick="fetch('/skip')" title="end the current episode and advance (n)">end episode ▸</button></div>
<div id="hud"></div><img id="f" alt="waiting for frames…">
<script>
const f=document.getElementById('f'),bar=document.getElementById('bar'),hud=document.getElementById('hud'),fps=15;
async function sources(){let r=null;try{r=await(await fetch('/sources.json')).json();}catch(e){}
 if(!r)return;bar.innerHTML='';r.sources.forEach((s,i)=>{const b=document.createElement('button');
  b.textContent=s;if(i===r.selected)b.className='sel';
  b.onclick=()=>fetch('/select?i='+i).then(sources);bar.appendChild(b);});}
function fmtE(s){s=Math.max(0,Math.floor(s||0));const m=Math.floor(s/60),ss=String(s%60).padStart(2,'0');
 return m>=60?Math.floor(m/60)+':'+String(m%60).padStart(2,'0')+':'+ss:m+':'+ss;}
function fmtMs(v){v=v||0;return (v<10?v.toFixed(1):Math.round(v))+'ms';}
async function hudtick(){let h=null;try{h=await(await fetch('/hud.json')).json();}catch(e){}
 if(!h)return;const p=[];
 if(h.episodes>0)p.push('ep '+h.episode+'/'+h.episodes);
 p.push('step '+h.step);
 p.push(fmtE(h.elapsed_s));
 if(h.seed>=0)p.push('seed '+h.seed);
 p.push('model '+fmtMs(h.model_ms));
 p.push('env '+fmtMs(h.env_ms));
 if(h.sps)p.push(h.sps.toFixed(1)+' sps');
 if(h.fps)p.push(Math.round(h.fps)+' fps');
 if(h.chunk_len>1)p.push('chunk '+h.chunk_pos+'/'+h.chunk_len);
 if(h.width>0)p.push(h.width+'×'+h.height);
 p.push('R '+Number(h.reward).toFixed(2));
 if(h.outcome)p.push(h.outcome);
 hud.textContent=p.join('   ·   ');}
sources();setInterval(sources,2000);setInterval(hudtick,250);
setInterval(()=>{f.src='/frame?t='+Date.now();},1000/fps);
</script></body></html>"#;
