use crate::errors::{SpaceError, err_space};
use crate::spaces::{SpaceSpec, SpaceValue, space_spec};

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
        Some(space_spec::Spec::Text(t)) => t,
        _ => return err_space!(path, "space is not Text"),
    };

    // Check length
    let len = text.len() as i64;
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
}
