//! Connection lifecycle state shared by the env and model clients.

/// Client state for gRPC endpoint clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientState {
    /// Connected but handshake not completed.
    Connected,
    /// Connected and handshake completed, ready to use.
    Ready,
    /// Closed.
    Closed,
}
