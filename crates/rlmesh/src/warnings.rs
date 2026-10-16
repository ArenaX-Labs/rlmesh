//! Warnings: Python's `warnings.warn` with rlmesh's categories (R-23).
//!
//! A warning goes to the process-wide handler installed with
//! [`set_warning_handler`], or by default to stderr as one line,
//! `rlmesh: <Category>: <message>`, each distinct category and message once
//! per process. A handler that fails aborts the operation that warned, which
//! returns that error: the native form of Python's `-W error`.

use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};

use crate::Result;

/// A warning category: Python's warning classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WarningCategory {
    /// `UserWarning`.
    UserWarning,
    /// `RuntimeWarning`.
    RuntimeWarning,
    /// `rlmesh.WorkflowEditionWarning`, a `UserWarning`.
    WorkflowEditionWarning,
}

impl WarningCategory {
    /// Python's class name (`"WorkflowEditionWarning"`).
    pub fn name(self) -> &'static str {
        match self {
            WarningCategory::UserWarning => "UserWarning",
            WarningCategory::RuntimeWarning => "RuntimeWarning",
            WarningCategory::WorkflowEditionWarning => "WorkflowEditionWarning",
        }
    }

    /// Python's `issubclass(self, other)`: reflexive, and
    /// `WorkflowEditionWarning` is a `UserWarning`.
    pub fn is_subclass_of(self, other: WarningCategory) -> bool {
        self == other
            || (self == WarningCategory::WorkflowEditionWarning
                && other == WarningCategory::UserWarning)
    }
}

impl fmt::Display for WarningCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One warning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Warning {
    /// Its category.
    pub category: WarningCategory,
    /// Its text.
    pub message: String,
}

/// `<Category>: <message>`.
impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.category, self.message)
    }
}

/// A warning handler: returns `Err` to fail the operation that warned.
pub type WarningHandler = Box<dyn Fn(&Warning) -> Result<()> + Send + Sync>;

type SharedHandler = Arc<dyn Fn(&Warning) -> Result<()> + Send + Sync>;

static HANDLER: RwLock<Option<SharedHandler>> = RwLock::new(None);

/// Install the process-wide warning handler; `None` restores the default
/// (stderr, each distinct warning once). An emission already running keeps
/// the handler it started with, which is dropped once that call returns.
pub fn set_warning_handler(handler: Option<WarningHandler>) {
    let handler: Option<SharedHandler> = handler.map(SharedHandler::from);
    let previous = {
        let mut slot = HANDLER.write().unwrap_or_else(|err| err.into_inner());
        std::mem::replace(&mut *slot, handler)
    };
    // Dropped outside the lock: a handler's drop (a C `free_user_data`) may
    // install another handler.
    drop(previous);
}

/// Emit a warning to the current handler; its failure is the caller's error.
pub(crate) fn warn(category: WarningCategory, message: impl Into<String>) -> Result<()> {
    let warning = Warning {
        category,
        message: message.into(),
    };
    if let Some(work) = work_context() {
        check_cancelled()?;
        let (reply, result) = std::sync::mpsc::sync_channel(1);
        work.warnings
            .send((warning, reply))
            .map_err(|_| crate::Error::Cancelled)?;
        return result.recv().unwrap_or(Err(crate::Error::Cancelled));
    }
    deliver(&warning)
}

fn deliver(warning: &Warning) -> Result<()> {
    let handler = HANDLER
        .read()
        .unwrap_or_else(|err| err.into_inner())
        .clone();
    match handler {
        Some(handler) => handler(warning),
        None => {
            default_handler(warning);
            Ok(())
        }
    }
}

// The default handler is stderr by definition (R-23, D-13).
#[allow(clippy::print_stderr)]
fn default_handler(warning: &Warning) {
    static SEEN: Mutex<Option<HashSet<(WarningCategory, String)>>> = Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|err| err.into_inner());
    if seen
        .get_or_insert_with(HashSet::new)
        .insert((warning.category, warning.message.clone()))
    {
        eprintln!("rlmesh: {warning}");
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::Error;

    /// Serializes the tests that install a process-wide handler.
    pub(crate) static HANDLER_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn categories_follow_python() {
        use WarningCategory::*;
        assert!(WorkflowEditionWarning.is_subclass_of(UserWarning));
        assert!(UserWarning.is_subclass_of(UserWarning));
        assert!(!UserWarning.is_subclass_of(WorkflowEditionWarning));
        assert!(!RuntimeWarning.is_subclass_of(UserWarning));
        let warning = Warning {
            category: RuntimeWarning,
            message: "short chunk".into(),
        };
        assert_eq!(warning.to_string(), "RuntimeWarning: short chunk");
    }

    #[test]
    fn a_failing_handler_fails_the_warning_call() {
        let _guard = HANDLER_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        set_warning_handler(Some(Box::new(move |warning: &Warning| {
            sink.lock().unwrap().push(warning.clone());
            Err(Error::Runtime("warnings are errors".into()))
        })));
        let err = warn(WarningCategory::UserWarning, "careful").unwrap_err();
        set_warning_handler(None);
        assert_eq!(err.to_string(), "warnings are errors");
        assert_eq!(seen.lock().unwrap()[0].message, "careful");
        assert!(warn(WarningCategory::UserWarning, "careful").is_ok());
    }
}

// CPU work may emit warnings, but their handler belongs to the thread polling
// the call. The same context follows work through the model's serialized queue.
type WarningRequest = (Warning, std::sync::mpsc::SyncSender<Result<()>>);

#[derive(Clone)]
pub(crate) struct WorkContext {
    cancelled: Arc<std::sync::atomic::AtomicBool>,
    warnings: tokio::sync::mpsc::UnboundedSender<WarningRequest>,
}

thread_local! {
    static WORK: std::cell::RefCell<Option<WorkContext>> = const { std::cell::RefCell::new(None) };
}

pub(crate) fn work_context() -> Option<WorkContext> {
    WORK.with(|work| work.borrow().clone())
}

pub(crate) fn check_cancelled() -> Result<()> {
    if work_context().is_some_and(|work| work.cancelled.load(std::sync::atomic::Ordering::SeqCst)) {
        Err(crate::Error::Cancelled)
    } else {
        Ok(())
    }
}

pub(crate) fn with_work<R>(work: Option<WorkContext>, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<WorkContext>);
    impl Drop for Restore {
        fn drop(&mut self) {
            WORK.with(|work| *work.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(WORK.with(|slot| slot.replace(work)));
    f()
}

/// Offload host work while delivering its warnings on the polling thread and
/// refusing callbacks that have not started when the awaiting future is dropped.
pub(crate) async fn blocking<R: Send + 'static>(
    f: impl FnOnce() -> R + Send + 'static,
) -> Result<R> {
    use std::future::Future;
    use std::task::Poll;
    struct Cancel(Arc<std::sync::atomic::AtomicBool>);
    impl Drop for Cancel {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
    let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let _cancel = Cancel(cancelled.clone());
    let (warnings, mut incoming) = tokio::sync::mpsc::unbounded_channel();
    let work = WorkContext {
        cancelled,
        warnings,
    };
    let mut job = tokio::task::spawn_blocking(move || {
        with_work(Some(work), || {
            check_cancelled()?;
            Ok(f())
        })
    });
    std::future::poll_fn(move |cx| {
        while let Poll::Ready(Some((warning, reply))) = incoming.poll_recv(cx) {
            let _ = reply.send(deliver(&warning));
        }
        match std::pin::Pin::new(&mut job).poll(cx) {
            Poll::Ready(Ok(out)) => Poll::Ready(out),
            Poll::Ready(Err(err)) => Poll::Ready(Err(crate::Error::Internal(format!(
                "model task panicked: {err}"
            )))),
            Poll::Pending => Poll::Pending,
        }
    })
    .await
}
