use std::fmt;
use std::time::Duration;

use crate::adapters::AdapterResolutionError;
use rlmesh_grpc::error::{EnvError, EnvErrorCode};

/// The result type used throughout this crate: `Result<T, `[`Error`]`>`.
pub type Result<T> = std::result::Result<T, Error>;

/// Classifies an [`EnvironmentError`] reported by an environment.
///
/// The text form ([`ErrorCode::as_str`], `Display`) is the wire code of the
/// `EnvironmentException`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCode {
    /// No specific code was provided.
    Unspecified,
    /// The operation exceeded its deadline.
    Timeout,
    /// The submitted action was invalid for the action space.
    InvalidAction,
    /// The environment is not ready to serve the request yet.
    NotReady,
    /// The environment is busy with another request.
    Busy,
    /// A well-formed request for a feature the peer does not support.
    Unsupported,
    /// An internal environment error; every authored env failure carries it.
    Internal,
    /// The environment process crashed.
    Crashed,
    /// The operation was cancelled.
    Cancelled,
    /// The environment (or session) is closed.
    Closed,
}

impl ErrorCode {
    /// The code's wire spelling (`"INVALID_ACTION"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unspecified => "UNSPECIFIED",
            Self::Timeout => "TIMEOUT",
            Self::InvalidAction => "INVALID_ACTION",
            Self::NotReady => "NOT_READY",
            Self::Busy => "BUSY",
            Self::Unsupported => "UNSUPPORTED",
            Self::Internal => "INTERNAL",
            Self::Crashed => "CRASHED",
            Self::Cancelled => "CANCELLED",
            Self::Closed => "CLOSED",
        }
    }
}

/// Classifies a [`ModelError`] reported by a model.
///
/// The text form ([`ModelErrorCode::as_str`], `Display`) is the wire code of
/// the `ModelException`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelErrorCode {
    /// No specific code was provided.
    Unspecified,
    /// The request was invalid.
    InvalidRequest,
    /// The route was not configured.
    NotConfigured,
    /// The model is busy with another request.
    Busy,
    /// A well-formed request for a feature the peer does not support.
    Unsupported,
    /// An internal model error; every authored model failure carries it.
    Internal,
    /// The operation was cancelled.
    Cancelled,
    /// The model or route is closed.
    Closed,
}

impl ModelErrorCode {
    /// The code's wire spelling (`"INVALID_REQUEST"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unspecified => "UNSPECIFIED",
            Self::InvalidRequest => "INVALID_REQUEST",
            Self::NotConfigured => "NOT_CONFIGURED",
            Self::Busy => "BUSY",
            Self::Unsupported => "UNSUPPORTED",
            Self::Internal => "INTERNAL",
            Self::Cancelled => "CANCELLED",
            Self::Closed => "CLOSED",
        }
    }
}

/// A failure reported by the environment when serving a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvironmentError {
    /// Machine-readable classification of the failure.
    pub code: ErrorCode,
    /// Human-readable description.
    pub message: String,
    /// Whether the request failed but the session stays usable
    /// (`RecoverableEnvironmentException`).
    pub is_recoverable: bool,
}

/// A failure reported by a model (`ModelException`).
///
/// This is distinct from the transport/internal variants: it represents the
/// model *declining* a request (e.g. an invalid observation), so the runtime
/// can report and retry it appropriately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelError {
    /// Machine-readable classification of the failure.
    pub code: ModelErrorCode,
    /// Human-readable description of why the model declined.
    pub message: String,
    /// Whether the caller may retry the request.
    pub is_recoverable: bool,
}

/// An invalid argument value, address or options field (`ValueError`), or a
/// failed adapter resolution (`AdapterResolutionError`, a `ValueError`
/// subclass) when [`adapter_resolution`](Self::adapter_resolution) is set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidArgumentError {
    /// Human-readable description; the error's whole text.
    pub message: String,
    /// The adapter-resolution failure this error reports, if any.
    pub adapter_resolution: Option<AdapterResolutionError>,
}

impl fmt::Display for InvalidArgumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for InvalidArgumentError {}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for ModelError {}

/// The exact class of an [`Error`], using RLMesh's error-class vocabulary.
///
/// Rust cannot nest enum variants, so a subclass without a variant of its
/// own (`RecoverableEnvironmentException`, `AdapterResolutionError`) shares
/// its parent's [`Error`] variant; [`Error::class`] still reports the exact
/// class, and [`ErrorClass::is_subclass_of`] tests class inheritance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorClass {
    /// `RLMeshException` (a `RuntimeError`): an internal or server-side failure.
    RLMeshException,
    /// `ProtocolException` (an `RLMeshException`).
    ProtocolException,
    /// `EnvironmentException` (an `RLMeshException`).
    EnvironmentException,
    /// `RecoverableEnvironmentException` (an `EnvironmentException`).
    RecoverableEnvironmentException,
    /// `ModelException` (an `RLMeshException`).
    ModelException,
    /// `ValueError`.
    ValueError,
    /// `AdapterResolutionError` (a `ValueError`).
    AdapterResolutionError,
    /// `TypeError`.
    TypeError,
    /// `RuntimeError`: misuse and local refusals.
    RuntimeError,
    /// `OSError`: a filesystem failure.
    OSError,
    /// `ConnectionError` (an `OSError`).
    ConnectionError,
    /// `TimeoutError` (an `OSError`).
    TimeoutError,
    /// `KeyboardInterrupt`: cancellation.
    KeyboardInterrupt,
}

impl ErrorClass {
    /// Every class, in the order of the cross-language error table (the C
    /// `RLMeshErrorClass` values 1-13).
    pub const ALL: [ErrorClass; 13] = [
        Self::RLMeshException,
        Self::ProtocolException,
        Self::EnvironmentException,
        Self::RecoverableEnvironmentException,
        Self::ModelException,
        Self::ValueError,
        Self::AdapterResolutionError,
        Self::TypeError,
        Self::RuntimeError,
        Self::OSError,
        Self::ConnectionError,
        Self::TimeoutError,
        Self::KeyboardInterrupt,
    ];

    /// The class name (`"RecoverableEnvironmentException"`).
    pub fn name(self) -> &'static str {
        match self {
            Self::RLMeshException => "RLMeshException",
            Self::ProtocolException => "ProtocolException",
            Self::EnvironmentException => "EnvironmentException",
            Self::RecoverableEnvironmentException => "RecoverableEnvironmentException",
            Self::ModelException => "ModelException",
            Self::ValueError => "ValueError",
            Self::AdapterResolutionError => "AdapterResolutionError",
            Self::TypeError => "TypeError",
            Self::RuntimeError => "RuntimeError",
            Self::OSError => "OSError",
            Self::ConnectionError => "ConnectionError",
            Self::TimeoutError => "TimeoutError",
            Self::KeyboardInterrupt => "KeyboardInterrupt",
        }
    }

    /// The direct base class within this vocabulary, following the
    /// error-class hierarchy (`None` where the base is `Exception` or
    /// `BaseException`).
    pub fn base(self) -> Option<ErrorClass> {
        match self {
            Self::RLMeshException => Some(Self::RuntimeError),
            Self::ProtocolException | Self::EnvironmentException | Self::ModelException => {
                Some(Self::RLMeshException)
            }
            Self::RecoverableEnvironmentException => Some(Self::EnvironmentException),
            Self::AdapterResolutionError => Some(Self::ValueError),
            Self::ConnectionError | Self::TimeoutError => Some(Self::OSError),
            Self::ValueError
            | Self::TypeError
            | Self::RuntimeError
            | Self::OSError
            | Self::KeyboardInterrupt => None,
        }
    }

    /// Class inheritance test: reflexive and transitive over the class hierarchy
    /// (`ConnectionError` is a subclass of `OSError`).
    pub fn is_subclass_of(self, base: ErrorClass) -> bool {
        let mut class = Some(self);
        while let Some(current) = class {
            if current == base {
                return true;
            }
            class = current.base();
        }
        false
    }
}

impl fmt::Display for ErrorClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The error type for every fallible operation in this crate.
///
/// Each variant identifies an error class (see [`ErrorClass`]); a subclass
/// with no variant of its own uses its parent's, and [`Error::class`]
/// reports the exact class. `Display` formats the message.
/// [`Error::is_recoverable`] reports whether the session remains usable.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// An internal or server-side failure (`RLMeshException`).
    Internal(String),
    /// A protocol failure: generation mismatch, refused edition, failed
    /// handshake or malformed message (`ProtocolException`).
    Protocol(String),
    /// The environment reported a failure; see [`EnvironmentError`]
    /// (`EnvironmentException`, or `RecoverableEnvironmentException` when
    /// recoverable).
    Environment(EnvironmentError),
    /// The model reported a failure; see [`ModelError`] (`ModelException`).
    Model(ModelError),
    /// An invalid argument value, address or options field (`ValueError`, or
    /// `AdapterResolutionError`).
    InvalidArgument(InvalidArgumentError),
    /// The wrong kind of value or object (`TypeError`).
    Type(String),
    /// Misuse (closed handle, overlapping call, ...) or a local refusal
    /// (builtin `RuntimeError`).
    Runtime(String),
    /// A filesystem failure (`OSError`).
    Os(String),
    /// A transport failure (`ConnectionError`).
    Connection(String),
    /// A connect or request deadline (`TimeoutError`).
    Timeout(Duration),
    /// The call was cancelled (`KeyboardInterrupt`).
    Cancelled,
    /// In-process host exception transport; only bindings construct this.
    #[doc(hidden)]
    Attached(crate::binding::AttachedError),
}

impl Error {
    /// Construct a non-recoverable model failure (`ModelException`, code
    /// `Internal`).
    ///
    /// Use this from a [`crate::binding::model::ModelHandler::predict`]
    /// implementation to signal that the model declined the request (e.g. a
    /// malformed observation), rather than misreporting it as an
    /// internal/transport fault.
    pub fn model(message: impl Into<String>) -> Self {
        Self::Model(ModelError {
            code: ModelErrorCode::Internal,
            message: message.into(),
            is_recoverable: false,
        })
    }

    /// Construct a recoverable model failure (code `Internal`; the caller
    /// may retry).
    pub fn model_recoverable(message: impl Into<String>) -> Self {
        Self::Model(ModelError {
            code: ModelErrorCode::Internal,
            message: message.into(),
            is_recoverable: true,
        })
    }

    /// Construct a non-recoverable environment failure
    /// (`EnvironmentException`, code `Internal`).
    pub fn environment(message: impl Into<String>) -> Self {
        Self::Environment(EnvironmentError {
            code: ErrorCode::Internal,
            message: message.into(),
            is_recoverable: false,
        })
    }

    /// Construct a recoverable environment failure
    /// (`RecoverableEnvironmentException`, code `Internal`).
    pub fn environment_recoverable(message: impl Into<String>) -> Self {
        Self::Environment(EnvironmentError {
            code: ErrorCode::Internal,
            message: message.into(),
            is_recoverable: true,
        })
    }

    /// Construct an invalid-argument error (`ValueError`).
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::InvalidArgument(InvalidArgumentError {
            message: message.into(),
            adapter_resolution: None,
        })
    }

    /// Tags that do not fit their spaces (`AdapterResolutionError`).
    pub fn adapter_resolution(message: impl Into<String>) -> Self {
        let message = message.into();
        Self::InvalidArgument(InvalidArgumentError {
            message: message.clone(),
            adapter_resolution: Some(AdapterResolutionError::new()),
        })
    }

    /// An address that could not be parsed: a `ValueError` whose text is
    /// `invalid address: message`.
    pub(crate) fn invalid_address(message: impl fmt::Display) -> Self {
        Self::invalid_argument(format!("invalid address: {message}"))
    }

    /// A server-side failure (bind failed, ...): an `RLMeshException` whose
    /// text uses `server error: message`.
    pub(crate) fn server(message: impl fmt::Display) -> Self {
        Self::Internal(format!("server error: {message}"))
    }

    /// The exact exception class this error reports.
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::Internal(_) => ErrorClass::RLMeshException,
            Self::Protocol(_) => ErrorClass::ProtocolException,
            Self::Environment(error) if error.is_recoverable => {
                ErrorClass::RecoverableEnvironmentException
            }
            Self::Environment(_) => ErrorClass::EnvironmentException,
            Self::Model(_) => ErrorClass::ModelException,
            Self::InvalidArgument(error) if error.adapter_resolution.is_some() => {
                ErrorClass::AdapterResolutionError
            }
            Self::InvalidArgument(_) => ErrorClass::ValueError,
            Self::Type(_) => ErrorClass::TypeError,
            Self::Runtime(_) => ErrorClass::RuntimeError,
            Self::Os(_) => ErrorClass::OSError,
            Self::Connection(_) => ErrorClass::ConnectionError,
            Self::Timeout(_) => ErrorClass::TimeoutError,
            Self::Cancelled => ErrorClass::KeyboardInterrupt,
            Self::Attached(attached) => attached.error.class(),
        }
    }

    /// Whether the session remains usable: the Environment or Model error's
    /// own flag, and false for every other class.
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Attached(attached) => attached.error.is_recoverable(),
            Self::Environment(error) => error.is_recoverable,
            Self::Model(error) => error.is_recoverable,
            _ => false,
        }
    }

    /// Whether retrying the same request may succeed: the recoverable flag,
    /// plus transport failures and deadlines. The facade's own retry and
    /// driver plumbing use it; it is not the user-facing
    /// [`Error::is_recoverable`].
    pub(crate) fn is_transient(&self) -> bool {
        match self {
            Self::Attached(attached) => attached.error.is_transient(),
            _ => matches!(self, Self::Connection(_) | Self::Timeout(_)) || self.is_recoverable(),
        }
    }

    /// Map a wire-level protocol failure onto [`Error::Protocol`].
    pub(crate) fn from_protocol(value: rlmesh_grpc::error::ProtocolError) -> Self {
        Self::Protocol(value.to_string())
    }

    /// An adapter-apply failure is the model declining a malformed
    /// observation/action, not a transport fault: surface it as a model error.
    pub(crate) fn from_apply(value: rlmesh_adapters::v1::ApplyError) -> Self {
        value
            .source
            .as_ref()
            .and_then(|source| source.downcast_ref::<Error>())
            .cloned()
            .unwrap_or_else(|| Self::invalid_argument(value.message))
    }

    /// Map a gRPC client/server error onto the facade taxonomy.
    pub(crate) fn from_grpc(value: rlmesh_grpc::error::Error) -> Self {
        use rlmesh_grpc::error::TransportError;
        match value {
            rlmesh_grpc::error::Error::Transport(error) => match error {
                TransportError::InvalidAddress(message) => Self::invalid_address(message),
                TransportError::BindFailed(message) => Self::server(message),
                TransportError::ConnectFailed(message) => Self::Connection(message),
                TransportError::ConnectionClosed => {
                    Self::Connection("connection closed".to_string())
                }
                TransportError::Io(error) => Self::Connection(error.to_string()),
                TransportError::MessageTooLarge { size, max } => {
                    Self::Connection(format!("message too large: {size} > {max}"))
                }
                TransportError::Unavailable(message) => Self::Connection(message),
                // The server refused the key: say where a key comes from,
                // never what was sent (R-22).
                TransportError::Status {
                    code: tonic::Code::Unauthenticated,
                    ..
                } => Self::Connection(KEY_REJECTED.to_string()),
                TransportError::Status { code, message } => {
                    Self::Connection(format!("{code:?}: {message}"))
                }
                // TransportError is #[non_exhaustive].
                other => Self::Connection(other.to_string()),
            },
            rlmesh_grpc::error::Error::Protocol(error) => Self::from_protocol(error),
            rlmesh_grpc::error::Error::Environment(error) => {
                Self::Environment(EnvironmentError::from_wire(error))
            }
            rlmesh_grpc::error::Error::Model(error) => Self::Model(ModelError {
                code: map_model_error_code(error.code),
                message: error.message,
                is_recoverable: error.is_recoverable,
            }),
            rlmesh_grpc::error::Error::Timeout(duration) => Self::Timeout(duration),
            // A cancelled connect/handshake is a connection-establishment
            // failure (e.g. the server's port is published but it isn't serving
            // yet during boot), not an internal fault -- surface it as Connection
            // so dial-retry treats it as transient.
            rlmesh_grpc::error::Error::Cancelled(message) => Self::Connection(message),
            // A token a header cannot carry is a bad argument, not a
            // transport failure (RMT-5).
            rlmesh_grpc::error::Error::Client(
                error @ rlmesh_grpc::error::ClientError::InvalidToken(_),
            ) => Self::invalid_argument(error.to_string()),
            rlmesh_grpc::error::Error::Client(
                rlmesh_grpc::error::ClientError::InvalidTlsCaFile(message),
            ) => Self::invalid_argument(message),
            rlmesh_grpc::error::Error::Client(error) => Self::Connection(error.to_string()),
            // rlmesh_grpc::error::Error is #[non_exhaustive].
            other => Self::Internal(other.to_string()),
        }
    }
}

/// A server's refusal of the client's key (gRPC `Unauthenticated`), the same
/// text in every language. It names where a key comes from, never the key.
pub(crate) const KEY_REJECTED: &str = "the server rejected the API key; pass the key it \
     expects as api_key (or set RLMESH_API_KEY for an https:// address)";

/// A client's dial failure, the one formatter for env and model clients:
/// `could not connect to {peer} at '{address}'; {hint} ({cause})`, or
/// `could not connect to {peer} at '{address}': {cause}` when the server
/// rejected the key (serving is not in doubt then). Other classes pass through.
pub(crate) fn dial_failure(err: Error, peer: &str, address: &str, hint: &str) -> Error {
    match err {
        Error::Connection(cause) if cause == KEY_REJECTED => Error::Connection(format!(
            "could not connect to {peer} at '{address}': {cause}"
        )),
        Error::Connection(cause) => Error::Connection(format!(
            "could not connect to {peer} at '{address}'; {hint} ({cause})"
        )),
        other => other,
    }
}

/// Combine a primary result (carrying a value `T`) with a close-hook result. If
/// both fail, fold them into one [`Error::Internal`] prefixed with `ctx`
/// ("<ctx>: <a>; close hook failed: <b>").
pub(crate) fn join_results<T>(a: Result<T>, b: Result<()>, ctx: &str) -> Result<T> {
    match (a, b) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(err), Ok(())) => Err(err),
        (Ok(_), Err(err)) => Err(err),
        (Err(a_err), Err(b_err)) => Err(Error::Internal(format!(
            "{ctx}: {a_err}; close hook failed: {b_err}"
        ))),
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Display for ModelErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for EnvironmentError {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Attached(attached) => attached.display(f),
            Self::Internal(message)
            | Self::Protocol(message)
            | Self::Type(message)
            | Self::Runtime(message)
            | Self::Os(message)
            | Self::Connection(message) => f.write_str(message),
            Self::Environment(error) => error.fmt(f),
            Self::Model(error) => error.fmt(f),
            Self::InvalidArgument(error) => error.fmt(f),
            Self::Timeout(duration) => write!(f, "operation timed out after {duration:?}"),
            Self::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl std::error::Error for Error {}

fn map_env_error_code(code: EnvErrorCode) -> ErrorCode {
    match code {
        EnvErrorCode::Unspecified => ErrorCode::Unspecified,
        EnvErrorCode::Timeout => ErrorCode::Timeout,
        EnvErrorCode::InvalidAction => ErrorCode::InvalidAction,
        EnvErrorCode::NotReady => ErrorCode::NotReady,
        EnvErrorCode::Busy => ErrorCode::Busy,
        EnvErrorCode::Unsupported => ErrorCode::Unsupported,
        EnvErrorCode::Internal => ErrorCode::Internal,
        EnvErrorCode::Crashed => ErrorCode::Crashed,
        EnvErrorCode::Cancelled => ErrorCode::Cancelled,
        EnvErrorCode::Closed => ErrorCode::Closed,
        // EnvErrorCode is #[non_exhaustive]; map unknown codes conservatively.
        _ => ErrorCode::Internal,
    }
}

fn map_model_error_code(code: rlmesh_grpc::error::ModelErrorCode) -> ModelErrorCode {
    use rlmesh_grpc::error::ModelErrorCode as Wire;
    match code {
        Wire::Unspecified => ModelErrorCode::Unspecified,
        Wire::InvalidRequest => ModelErrorCode::InvalidRequest,
        Wire::NotConfigured => ModelErrorCode::NotConfigured,
        Wire::Busy => ModelErrorCode::Busy,
        Wire::Unsupported => ModelErrorCode::Unsupported,
        Wire::Internal => ModelErrorCode::Internal,
        Wire::Cancelled => ModelErrorCode::Cancelled,
        Wire::Closed => ModelErrorCode::Closed,
    }
}

impl EnvironmentError {
    /// Map a wire environment error onto the facade's detail fields.
    pub(crate) fn from_wire(value: EnvError) -> Self {
        Self {
            code: map_env_error_code(value.code),
            message: value.message,
            is_recoverable: value.is_recoverable,
        }
    }

    /// The inverse of [`Self::from_wire`].
    #[cfg(feature = "lockstep")]
    pub(crate) fn to_wire(&self) -> EnvError {
        let code = match self.code {
            ErrorCode::Unspecified => EnvErrorCode::Unspecified,
            ErrorCode::Timeout => EnvErrorCode::Timeout,
            ErrorCode::InvalidAction => EnvErrorCode::InvalidAction,
            ErrorCode::NotReady => EnvErrorCode::NotReady,
            ErrorCode::Busy => EnvErrorCode::Busy,
            ErrorCode::Unsupported => EnvErrorCode::Unsupported,
            ErrorCode::Internal => EnvErrorCode::Internal,
            ErrorCode::Crashed => EnvErrorCode::Crashed,
            ErrorCode::Cancelled => EnvErrorCode::Cancelled,
            ErrorCode::Closed => EnvErrorCode::Closed,
        };
        EnvError {
            is_recoverable: self.is_recoverable,
            ..EnvError::new(code, self.message.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grpc_model_error_maps_to_model_variant_preserving_code_and_recoverability() {
        let grpc = rlmesh_grpc::error::Error::Model(rlmesh_grpc::error::ModelError {
            code: rlmesh_grpc::error::ModelErrorCode::Busy,
            message: "handler declined".to_string(),
            is_recoverable: true,
            debug_info: None,
        });
        match Error::from_grpc(grpc) {
            Error::Model(model) => {
                assert_eq!(model.code, ModelErrorCode::Busy);
                assert_eq!(model.message, "handler declined");
                assert!(model.is_recoverable);
            }
            other => panic!("expected Error::Model, got {other:?}"),
        }
    }

    #[test]
    fn model_error_constructors_set_recoverability() {
        assert!(!Error::model("nope").is_recoverable());
        assert!(Error::model_recoverable("retry").is_recoverable());
        assert!(!Error::environment("nope").is_recoverable());
        assert!(Error::environment_recoverable("retry").is_recoverable());
    }

    #[test]
    fn only_environment_and_model_flags_are_recoverable() {
        assert!(!Error::Connection("reset".into()).is_recoverable());
        assert!(!Error::Timeout(Duration::from_secs(1)).is_recoverable());
        assert!(Error::Connection("reset".into()).is_transient());
        assert!(Error::Timeout(Duration::from_secs(1)).is_transient());
        assert!(!Error::Internal("boom".into()).is_transient());
    }

    #[test]
    fn unsupported_env_code_survives_the_wire_mapping() {
        let wire = EnvError::new(EnvErrorCode::Unsupported, "no render");
        let error = EnvironmentError::from_wire(wire);
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert_eq!(error.code.as_str(), "UNSUPPORTED");
    }

    #[test]
    fn protocol_failures_map_to_the_protocol_variant() {
        let grpc = rlmesh_grpc::error::Error::Protocol(
            rlmesh_grpc::error::ProtocolError::HandshakeFailed("generation 1 != 2".into()),
        );
        let error = Error::from_grpc(grpc);
        assert!(matches!(error, Error::Protocol(_)), "{error:?}");
        assert_eq!(error.class(), ErrorClass::ProtocolException);
    }

    #[test]
    fn text_is_python_str() {
        let cases = [
            (Error::environment("bad"), "[INTERNAL] bad"),
            (
                Error::Environment(EnvironmentError {
                    code: ErrorCode::InvalidAction,
                    message: "bad action".into(),
                    is_recoverable: true,
                }),
                "[INVALID_ACTION] bad action",
            ),
            (Error::model("declined"), "[INTERNAL] declined"),
            (Error::Internal("boom".into()), "boom"),
            (Error::server("bind failed"), "server error: bind failed"),
            (Error::Protocol("bad frame".into()), "bad frame"),
            (Error::Type("not a Box".into()), "not a Box"),
            (Error::Runtime("closed".into()), "closed"),
            (Error::Os("exists".into()), "exists"),
            (Error::Connection("reset".into()), "reset"),
            (Error::invalid_argument("bad seed"), "bad seed"),
            (Error::invalid_address("nope"), "invalid address: nope"),
            (
                Error::Timeout(Duration::from_secs(5)),
                "operation timed out after 5s",
            ),
            (Error::Cancelled, "cancelled"),
        ];
        for (error, text) in cases {
            assert_eq!(error.to_string(), text, "{error:?}");
        }
    }

    #[test]
    fn class_reports_the_exact_python_class() {
        assert_eq!(
            Error::environment_recoverable("x").class(),
            ErrorClass::RecoverableEnvironmentException
        );
        assert_eq!(
            Error::environment("x").class(),
            ErrorClass::EnvironmentException
        );
        assert_eq!(Error::server("x").class(), ErrorClass::RLMeshException);
        assert_eq!(Error::invalid_address("x").class(), ErrorClass::ValueError);
        assert_eq!(Error::Cancelled.class(), ErrorClass::KeyboardInterrupt);
    }

    #[test]
    fn is_subclass_of_follows_python_mro() {
        use ErrorClass as C;
        // Reflexive, as issubclass.
        for class in C::ALL {
            assert!(class.is_subclass_of(class), "{class}");
        }
        assert!(C::RecoverableEnvironmentException.is_subclass_of(C::EnvironmentException));
        assert!(C::RecoverableEnvironmentException.is_subclass_of(C::RLMeshException));
        assert!(C::RecoverableEnvironmentException.is_subclass_of(C::RuntimeError));
        assert!(C::ModelException.is_subclass_of(C::RLMeshException));
        assert!(C::ProtocolException.is_subclass_of(C::RuntimeError));
        assert!(C::AdapterResolutionError.is_subclass_of(C::ValueError));
        assert!(C::ConnectionError.is_subclass_of(C::OSError));
        assert!(C::TimeoutError.is_subclass_of(C::OSError));
        // The builtins are not under RLMeshException.
        for class in [
            C::ValueError,
            C::TypeError,
            C::RuntimeError,
            C::OSError,
            C::ConnectionError,
            C::TimeoutError,
            C::KeyboardInterrupt,
        ] {
            assert!(!class.is_subclass_of(C::RLMeshException), "{class}");
        }
        assert!(!C::KeyboardInterrupt.is_subclass_of(C::RuntimeError));
        assert!(!C::RuntimeError.is_subclass_of(C::RLMeshException));
        let under_rlmesh: Vec<_> = C::ALL
            .into_iter()
            .filter(|class| class.is_subclass_of(C::RLMeshException))
            .collect();
        assert_eq!(
            under_rlmesh,
            [
                C::RLMeshException,
                C::ProtocolException,
                C::EnvironmentException,
                C::RecoverableEnvironmentException,
                C::ModelException,
            ]
        );
        assert_eq!(C::OSError.name(), "OSError");
    }
}
