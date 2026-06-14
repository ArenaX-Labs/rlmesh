use crate::errors::{SpaceError, err_space};
use crate::spaces::{SpaceKind, SpaceSpec, SpaceValue};
use crate::{DType, TextSpec};

#[macro_export]
macro_rules! text_space_v1 {
    ($n:expr $(,)?) => {
        $crate::TextBuilder::new($n)
    };
}

#[must_use = "a space builder does nothing until .build() is called"]
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
        spec: Some(SpaceKind::Text(TextSpec {
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
        Some(SpaceKind::Text(t)) => t,
        _ => return err_space!(path, "Text", "spec.text must be set"),
    };

    if t.min_length <= 0 {
        return err_space!(path, "Text", "min_length must be > 0");
    }

    if t.max_length <= 0 {
        return err_space!(path, "Text", "max_length must be > 0");
    }

    if t.min_length > t.max_length {
        return err_space!(path, "Text", "min_length must be <= max_length");
    }

    Ok(())
}

pub(crate) fn contains_text(
    space: &SpaceSpec,
    value: &SpaceValue,
    path: &str,
) -> Result<(), SpaceError> {
    let text = match value {
        SpaceValue::Text(s) => s,
        _ => return err_space!(path, "expected Text value"),
    };

    let t = match &space.spec {
        Some(SpaceKind::Text(t)) => t,
        _ => return err_space!(path, "space is not Text"),
    };

    // Check length in characters (Gymnasium Text counts characters, not UTF-8 bytes)
    let len = text.chars().count() as i64;
    if len < t.min_length {
        return err_space!(
            path,
            format!("text length {} below minimum {}", len, t.min_length)
        );
    }
    if len > t.max_length {
        return err_space!(
            path,
            format!("text length {} exceeds maximum {}", len, t.max_length)
        );
    }

    // Check charset if specified
    if !t.charset.is_empty() {
        for c in text.chars() {
            if !t.charset.contains(c) {
                return err_space!(path, format!("character '{}' not in charset", c));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::spaces::fundamental::TextBuilder;
    use crate::spaces::{SpaceValue, contains};

    #[test]
    fn test_text_contains() {
        let unrestricted = TextBuilder::new(32).build().unwrap();
        assert!(
            contains(
                &unrestricted,
                &SpaceValue::Text("pick up the object!".to_string())
            )
            .is_ok()
        );

        let space = TextBuilder::new(5)
            .min_length(2)
            .charset("abc".to_string())
            .build()
            .unwrap();

        assert!(contains(&space, &SpaceValue::Text("abc".to_string())).is_ok());
        assert!(contains(&space, &SpaceValue::Text("ab".to_string())).is_ok());
        assert!(contains(&space, &SpaceValue::Text("a".to_string())).is_err()); // too short
        assert!(contains(&space, &SpaceValue::Text("abcdef".to_string())).is_err()); // too long
        assert!(contains(&space, &SpaceValue::Text("abc!".to_string())).is_err()); // '!' not in charset
    }

    #[test]
    fn test_text_length_counts_chars_not_bytes() {
        // max_length is in characters; "ééé" is 3 chars but 6 UTF-8 bytes.
        let space = TextBuilder::new(3)
            .charset("éàü".to_string())
            .build()
            .unwrap();

        // 3 multi-byte chars (6 bytes) must be accepted, not rejected for length.
        assert!(contains(&space, &SpaceValue::Text("ééé".to_string())).is_ok());
        // 4 chars exceeds the 3-char maximum.
        assert!(contains(&space, &SpaceValue::Text("éééà".to_string())).is_err());
    }

    #[test]
    fn test_text_rejects_vacuous_bounds() {
        // min_length > max_length describes a space no value can satisfy; reject at build time.
        assert!(TextBuilder::new(5).min_length(10).build().is_err());
        // Equal bounds remain valid (fixed-length text).
        assert!(TextBuilder::new(5).min_length(5).build().is_ok());
    }
}
