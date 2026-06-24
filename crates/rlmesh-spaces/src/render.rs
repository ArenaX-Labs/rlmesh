#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BinaryPayload {
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RenderFrame {
    /// Rendered frame bytes. Currently PNG-encoded and experimental; the format
    /// is carried out-of-band. A future format discriminator is additive.
    pub frame: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RenderRequest {
    pub env_index: Option<usize>,
    pub timeout_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct RenderResult {
    pub frame: Option<RenderFrame>,
}
