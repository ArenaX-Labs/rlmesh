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
#[derive(Clone, Default)]
pub struct Hud {
    pub step: i64,
    pub reward: f64,
    pub outcome: String,
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
                "step": hud.step, "reward": hud.reward, "outcome": hud.outcome,
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
<div id="bar"></div><div id="hud"></div><img id="f" alt="waiting for frames…">
<script>
const f=document.getElementById('f'),bar=document.getElementById('bar'),hud=document.getElementById('hud'),fps=15;
async function sources(){let r=null;try{r=await(await fetch('/sources.json')).json();}catch(e){}
 if(!r)return;bar.innerHTML='';r.sources.forEach((s,i)=>{const b=document.createElement('button');
  b.textContent=s;if(i===r.selected)b.className='sel';
  b.onclick=()=>fetch('/select?i='+i).then(sources);bar.appendChild(b);});}
async function hudtick(){let h=null;try{h=await(await fetch('/hud.json')).json();}catch(e){}
 if(!h)return;hud.textContent=(h.step!=null?'step '+h.step:'')
  +(h.reward!=null?'   R '+Number(h.reward).toFixed(2):'')+(h.outcome?'   '+h.outcome:'');}
sources();setInterval(sources,2000);setInterval(hudtick,250);
setInterval(()=>{f.src='/frame?t='+Date.now();},1000/fps);
</script></body></html>"#;
