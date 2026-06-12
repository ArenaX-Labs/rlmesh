#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BinaryPayload {
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RenderFrame {
    pub png_frame: Vec<u8>,
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
