//! ENV-11's signal handling for a foreground `serve()`.
//!
//! A native foreground serve saves the process's SIGINT and SIGTERM
//! dispositions on entry and restores both when it returns, by any path, so
//! it does not use `tokio::signal` (whose handlers can never be removed).
//! Concurrent foreground serves share one reference-counted installation: the
//! first to enter saves and installs, a signal stops every foreground server
//! then serving, and the last to return restores. SIGINT stops a server (it
//! drains and closes its env, then `serve()` fails with the cancellation
//! class); a second SIGINT while the stop runs restores the default SIGINT
//! disposition and re-raises it. SIGTERM stops a server gracefully.
//!
//! Windows has no SIGTERM; its console-control form is not implemented yet,
//! so a native foreground serve there installs nothing.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rlmesh_grpc::lifecycle::ShutdownTrigger;

/// One foreground serve's registration; dropping it unregisters and, for the
/// last one, restores the saved dispositions.
pub(crate) struct SignalScope {
    #[cfg_attr(not(unix), allow(dead_code))]
    id: u64,
    interrupted: Arc<AtomicBool>,
}

impl SignalScope {
    /// Whether SIGINT stopped this serve.
    pub(crate) fn interrupted(&self) -> bool {
        self.interrupted.load(Ordering::SeqCst)
    }
}

/// Register a foreground serve whose server `trigger` stops.
pub(crate) fn enter(trigger: ShutdownTrigger) -> SignalScope {
    let interrupted = Arc::new(AtomicBool::new(false));
    #[cfg(unix)]
    {
        let id = unix::register(trigger, Arc::clone(&interrupted));
        SignalScope { id, interrupted }
    }
    #[cfg(not(unix))]
    {
        let _ = trigger;
        SignalScope { id: 0, interrupted }
    }
}

impl Drop for SignalScope {
    fn drop(&mut self) {
        #[cfg(unix)]
        unix::unregister(self.id);
    }
}

#[cfg(unix)]
#[allow(unsafe_code)]
mod unix {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use rlmesh_grpc::lifecycle::ShutdownTrigger;

    /// SIGINTs received since the installation; the handler re-raises from
    /// the second.
    static SIGINTS: AtomicUsize = AtomicUsize::new(0);
    static SIGTERMS: AtomicUsize = AtomicUsize::new(0);

    struct Installed {
        next_id: u64,
        servers: HashMap<u64, (ShutdownTrigger, Arc<AtomicBool>)>,
        saved_int: libc::sigaction,
        saved_term: libc::sigaction,
        stop_watcher: Arc<AtomicBool>,
    }

    // SAFETY: `libc::sigaction` holds a handler address and a signal mask:
    // plain data, moved between threads only under the mutex.
    unsafe impl Send for Installed {}

    static STATE: Mutex<Option<Installed>> = Mutex::new(None);

    extern "C" fn on_signal(signal: libc::c_int) {
        if signal == libc::SIGINT {
            if SIGINTS.fetch_add(1, Ordering::SeqCst) >= 1 {
                // SAFETY: async-signal-safe calls only (`signal`, `raise`).
                unsafe {
                    libc::signal(libc::SIGINT, libc::SIG_DFL);
                    libc::raise(libc::SIGINT);
                }
            }
        } else {
            SIGTERMS.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn install(signal: libc::c_int) -> libc::sigaction {
        // SAFETY: a zeroed sigaction is a valid "default" value to fill;
        // `sigaction` reads `action` and writes the previous one into `saved`.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_signal as extern "C" fn(libc::c_int) as usize;
            libc::sigemptyset(&mut action.sa_mask);
            action.sa_flags = libc::SA_RESTART;
            let mut saved: libc::sigaction = std::mem::zeroed();
            libc::sigaction(signal, &action, &mut saved);
            saved
        }
    }

    fn restore(signal: libc::c_int, saved: &libc::sigaction) {
        // SAFETY: `saved` came from a successful `sigaction` call.
        unsafe {
            libc::sigaction(signal, saved, std::ptr::null_mut());
        }
    }

    pub(super) fn register(trigger: ShutdownTrigger, interrupted: Arc<AtomicBool>) -> u64 {
        let mut state = STATE.lock().unwrap_or_else(|err| err.into_inner());
        let installed = state.get_or_insert_with(|| {
            SIGINTS.store(0, Ordering::SeqCst);
            SIGTERMS.store(0, Ordering::SeqCst);
            let stop_watcher = Arc::new(AtomicBool::new(false));
            let stop = Arc::clone(&stop_watcher);
            let _ = std::thread::Builder::new()
                .name("rlmesh-signals".to_string())
                .spawn(move || watch(&stop));
            Installed {
                next_id: 0,
                servers: HashMap::new(),
                saved_int: install(libc::SIGINT),
                saved_term: install(libc::SIGTERM),
                stop_watcher,
            }
        });
        installed.next_id += 1;
        let id = installed.next_id;
        installed.servers.insert(id, (trigger, interrupted));
        id
    }

    pub(super) fn unregister(id: u64) {
        let mut state = STATE.lock().unwrap_or_else(|err| err.into_inner());
        let Some(installed) = state.as_mut() else {
            return;
        };
        installed.servers.remove(&id);
        if installed.servers.is_empty()
            && let Some(installed) = state.take()
        {
            restore(libc::SIGINT, &installed.saved_int);
            restore(libc::SIGTERM, &installed.saved_term);
            installed.stop_watcher.store(true, Ordering::SeqCst);
        }
    }

    /// Turn received signals into server stops, off the handler.
    fn watch(stop: &AtomicBool) {
        let (mut ints, mut terms) = (0, 0);
        while !stop.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(20));
            let now_ints = SIGINTS.load(Ordering::SeqCst);
            let now_terms = SIGTERMS.load(Ordering::SeqCst);
            if now_ints == ints && now_terms == terms {
                continue;
            }
            let sigint = now_ints > ints;
            (ints, terms) = (now_ints, now_terms);
            let state = STATE.lock().unwrap_or_else(|err| err.into_inner());
            for (trigger, interrupted) in state.iter().flat_map(|state| state.servers.values()) {
                if sigint {
                    interrupted.store(true, Ordering::SeqCst);
                }
                trigger.trigger(if sigint { "sigint" } else { "sigterm" });
            }
        }
    }
}
