//! Signal handling for a foreground `serve()`.
//!
//! A native foreground serve saves the process's SIGINT and SIGTERM
//! dispositions on entry and restores both when it returns, by any path, so
//! it does not use `tokio::signal` (whose handlers can never be removed).
//! Concurrent foreground serves share one reference-counted installation per
//! signal: the first to enter saves and installs, a signal stops every
//! foreground server then serving that asked for it, and the last to return
//! restores. SIGINT stops a server (it drains and closes its env, then
//! `serve()` fails with the cancellation class); a second SIGINT while the
//! stop runs restores the default SIGINT disposition and re-raises it.
//! SIGTERM stops a server gracefully.
//!
//! Windows has no SIGTERM; its console-control form is not implemented yet,
//! so a native foreground serve there installs nothing.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rlmesh_grpc::lifecycle::ShutdownTrigger;

/// One foreground serve's registration; dropping it unregisters and, for the
/// last one of a signal, restores that signal's saved disposition.
pub struct SignalScope {
    #[cfg_attr(not(unix), allow(dead_code))]
    id: u64,
    interrupted: Arc<AtomicBool>,
}

impl SignalScope {
    /// Whether SIGINT stopped this serve.
    pub fn interrupted(&self) -> bool {
        self.interrupted.load(Ordering::SeqCst)
    }
}

/// Register a foreground serve that SIGINT and SIGTERM stop.
pub(crate) fn enter(trigger: ShutdownTrigger) -> SignalScope {
    enter_with(trigger, true)
}

/// Register a foreground serve that SIGTERM stops, and SIGINT too when
/// `sigint` (the Python binding handles SIGINT itself).
pub(crate) fn enter_with(trigger: ShutdownTrigger, sigint: bool) -> SignalScope {
    let interrupted = Arc::new(AtomicBool::new(false));
    #[cfg(unix)]
    {
        let id = unix::register(trigger, Arc::clone(&interrupted), sigint);
        SignalScope { id, interrupted }
    }
    #[cfg(not(unix))]
    {
        let _ = (trigger, sigint);
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

    /// Monotonic delivery generations, independent of handler installations.
    static SIGINTS: AtomicUsize = AtomicUsize::new(0);
    static SIGTERMS: AtomicUsize = AtomicUsize::new(0);
    /// SIGINTs since this installation: the second forces termination.
    static INTERRUPTS: AtomicUsize = AtomicUsize::new(0);

    struct Server {
        trigger: ShutdownTrigger,
        interrupted: Arc<AtomicBool>,
        sigint: bool,
    }

    #[derive(Default)]
    struct Registry {
        next_id: u64,
        servers: HashMap<u64, Server>,
        saved_int: Option<libc::sigaction>,
        saved_term: Option<libc::sigaction>,
        stop_watcher: Option<Arc<AtomicBool>>,
    }

    // SAFETY: `libc::sigaction` holds a handler address and a signal mask:
    // plain data, moved between threads only under the mutex.
    unsafe impl Send for Registry {}

    static STATE: Mutex<Option<Registry>> = Mutex::new(None);

    extern "C" fn on_signal(signal: libc::c_int) {
        if signal == libc::SIGINT {
            SIGINTS.fetch_add(1, Ordering::SeqCst);
            if INTERRUPTS.fetch_add(1, Ordering::SeqCst) >= 1 {
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
        // SAFETY: a zeroed sigaction is a valid value to fill; `sigaction`
        // reads `action` and writes the previous disposition into `saved`.
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

    pub(super) fn register(
        trigger: ShutdownTrigger,
        interrupted: Arc<AtomicBool>,
        sigint: bool,
    ) -> u64 {
        let mut state = STATE.lock().unwrap_or_else(|err| err.into_inner());
        let registry = state.get_or_insert_with(Registry::default);
        if registry.stop_watcher.is_none() {
            let stop = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&stop);
            // Snapshot before installing either handler, not on the new thread.
            let ints = SIGINTS.load(Ordering::SeqCst);
            let terms = SIGTERMS.load(Ordering::SeqCst);
            let _ = std::thread::Builder::new()
                .name("rlmesh-signals".to_string())
                .spawn(move || watch(&flag, ints, terms));
            registry.stop_watcher = Some(stop);
        }
        if registry.saved_term.is_none() {
            registry.saved_term = Some(install(libc::SIGTERM));
        }
        if sigint && registry.saved_int.is_none() {
            INTERRUPTS.store(0, Ordering::SeqCst);
            registry.saved_int = Some(install(libc::SIGINT));
        }
        registry.next_id += 1;
        let id = registry.next_id;
        registry.servers.insert(
            id,
            Server {
                trigger,
                interrupted,
                sigint,
            },
        );
        id
    }

    pub(super) fn unregister(id: u64) {
        let mut state = STATE.lock().unwrap_or_else(|err| err.into_inner());
        let Some(registry) = state.as_mut() else {
            return;
        };
        registry.servers.remove(&id);
        if !registry.servers.values().any(|server| server.sigint)
            && let Some(saved) = registry.saved_int.take()
        {
            restore(libc::SIGINT, &saved);
        }
        if registry.servers.is_empty() {
            if let Some(saved) = registry.saved_term.take() {
                restore(libc::SIGTERM, &saved);
            }
            if let Some(stop) = registry.stop_watcher.take() {
                stop.store(true, Ordering::SeqCst);
            }
        }
    }

    #[cfg(test)]
    pub(super) fn signal_before_watch_regression() {
        let trigger = ShutdownTrigger::new();
        let mut registry = Registry::default();
        registry.servers.insert(
            1,
            Server {
                trigger: trigger.clone(),
                interrupted: Arc::new(AtomicBool::new(false)),
                sigint: false,
            },
        );
        *STATE.lock().unwrap() = Some(registry);
        SIGTERMS.store(1, Ordering::SeqCst);
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let watcher = std::thread::spawn(move || watch(&flag, 0, 0));
        for _ in 0..100 {
            if trigger.reason().is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        stop.store(true, Ordering::SeqCst);
        watcher.join().unwrap();
        assert_eq!(trigger.reason().as_deref(), Some("sigterm"));
        *STATE.lock().unwrap() = None;
    }

    /// Turn received signals into server stops, off the handler.
    fn watch(stop: &AtomicBool, mut ints: usize, mut terms: usize) {
        while !stop.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(20));
            let now_ints = SIGINTS.load(Ordering::SeqCst);
            let now_terms = SIGTERMS.load(Ordering::SeqCst);
            let sigint = now_ints != ints;
            let sigterm = now_terms != terms;
            (ints, terms) = (now_ints, now_terms);
            if !sigint && !sigterm {
                continue;
            }
            let state = STATE.lock().unwrap_or_else(|err| err.into_inner());
            for server in state.iter().flat_map(|state| state.servers.values()) {
                if sigint && server.sigint {
                    server.interrupted.store(true, Ordering::SeqCst);
                    server.trigger.trigger("sigint");
                } else if sigterm {
                    server.trigger.trigger("sigterm");
                }
            }
        }
    }
}

#[cfg(all(test, unix))]
mod regressions {
    use super::*;

    #[test]
    fn a_signal_received_before_the_watcher_starts_is_delivered() {
        unix::signal_before_watch_regression();
    }
}
