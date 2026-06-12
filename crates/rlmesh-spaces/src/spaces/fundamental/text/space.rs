use crate::errors::{SpaceError, err_space};
use crate::spaces::{SpaceSpec, space_spec};
use crate::{DType, TextSpec};

#[macro_export]
macro_rules! text_space_v1 {
    ($n:expr $(,)?) => {
        $crate::TextBuilder::new($n)
    };
}

pub struct TextBuilder {
    min_length: i64,
    max_length: i64,
    charset: String,
}

impl TextBuilder {
    pub fn new(max_length: i64) -> Self {
        Self {
            min_length: 1,
            max_length,
            charset: String::new(),
        }
    }
    pub fn min_length(mut self, min_length: i64) -> Self {
        self.min_length = min_length;
        self
    }
    pub fn max_length(mut self, max_length: i64) -> Self {
        self.max_length = max_length;
        self
    }
    pub fn charset(mut self, charset: impl Into<String>) -> Self {
        self.charset = charset.into();
        self
    }
    pub fn build(self) -> Result<SpaceSpec, SpaceError> {
        make_text_at(self.min_length, self.max_length, self.charset)
    }
}

fn make_text_at(
    min_length: i64,
    max_length: i64,
    charset: String,
) -> Result<SpaceSpec, SpaceError> {
    let spec = SpaceSpec {
        shape: vec![],
        dtype: DType::Uint8,
        spec: Some(space_spec::Spec::Text(TextSpec {
            min_length,
            max_length,
            charset,
        })),
    };
    crate::spaces::validate_space(&spec)?;
    Ok(spec)
}

pub(crate) fn validate_text_at(spec: &SpaceSpec, path: &str) -> Result<(), SpaceError> {
    if !spec.shape.is_empty() {
        return err_space!(path, "Text", "shape must be empty");
    }

    if spec.dtype != DType::Uint8 {
        return err_space!(path, "Text", "dtype must be uint8");
    }

    let t = match &spec.spec {
        Some(space_spec::Spec::Text(t)) => t,
        _ => return err_space!(path, "Text", "spec.text must be set"),
    };

    if t.min_length <= 0 {
        return err_space!(path, "Text", "min_length must be > 0");
    }

    if t.max_length <= 0 {
        return err_space!(path, "Text", "max_length must be > 0");
    }

    Ok(())
}
