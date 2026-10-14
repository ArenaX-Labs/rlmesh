//! Bearer-token authentication helpers shared by the env and model services.
//!
//! Token comparison is constant-time (a manual byte-fold) so a network peer
//! cannot recover the secret one byte at a time from response timing. An empty
//! configured token means authentication is disabled.

/// Compare two byte slices in constant time relative to their contents.
///
/// Returns `false` immediately for differing lengths (the length of a bearer
/// token is not itself a secret worth protecting), but never short-circuits on
/// the first differing byte for equal-length inputs.
#[must_use]
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Decide whether a request bearing `provided` satisfies the `configured`
/// bearer token.
///
/// **An empty `configured` token disables authentication**: every request is
/// accepted (this is the explicit opt-out for unauthenticated endpoints).
/// Otherwise the provided token must match in constant time, sent either raw
/// or in the standard `Bearer <token>` form (scheme matched case-insensitively).
#[must_use]
pub fn bearer_token_matches(configured: &str, provided: &str) -> bool {
    if configured.is_empty() {
        return true;
    }
    constant_time_eq(configured.as_bytes(), provided.as_bytes())
        || strip_bearer_scheme(provided)
            .is_some_and(|token| constant_time_eq(configured.as_bytes(), token.as_bytes()))
}

/// Why `token` cannot travel as an `authorization` header value, if it cannot.
///
/// A token must be printable ASCII without leading or trailing whitespace:
/// header values cannot carry control characters such as a newline, a server
/// reads the header only as visible ASCII, and the `Bearer <token>` form drops
/// the spaces after the scheme. The returned phrase completes "the token ..."
/// and never includes the token itself, which is a secret.
#[must_use]
pub fn token_header_problem(token: &str) -> Option<&'static str> {
    if token.trim() != token {
        Some("has leading or trailing whitespace (such as a trailing newline from a secret file)")
    } else if token.chars().any(char::is_control) {
        Some("contains a control character")
    } else if !token.is_ascii() {
        Some("contains a non-ASCII character")
    } else {
        None
    }
}

/// Attach `token` to a client request's `authorization` metadata. An empty
/// token sends none; a token a header cannot carry fails with a message naming
/// the problem (never the token) instead of a transport error.
pub(crate) fn authorize<T>(
    request: &mut tonic::Request<T>,
    token: &str,
) -> Result<(), crate::error::ClientError> {
    use crate::error::ClientError;
    if token.is_empty() {
        return Ok(());
    }
    if let Some(problem) = token_header_problem(token) {
        return Err(ClientError::InvalidToken(problem));
    }
    let value = token
        .parse()
        .map_err(|_| ClientError::InvalidToken("is not a valid header value"))?;
    request.metadata_mut().insert("authorization", value);
    Ok(())
}

/// The credential of an `authorization` value in `Bearer <token>` form.
fn strip_bearer_scheme(provided: &str) -> Option<&str> {
    let (scheme, token) = provided.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("bearer")
        .then_some(token.trim_start_matches(' '))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_eq_matches_only_identical_bytes() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secref"));
        assert!(!constant_time_eq(b"secret", b"secretx"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn empty_configured_token_disables_auth() {
        assert!(bearer_token_matches("", ""));
        assert!(bearer_token_matches("", "anything"));
    }

    #[test]
    fn bearer_scheme_is_accepted() {
        assert!(bearer_token_matches("s3cret", "Bearer s3cret"));
        assert!(bearer_token_matches("s3cret", "bearer s3cret"));
        assert!(bearer_token_matches("s3cret", "BEARER  s3cret"));
        assert!(!bearer_token_matches("s3cret", "Bearer wrong"));
        assert!(!bearer_token_matches("s3cret", "Bearer "));
        assert!(!bearer_token_matches("s3cret", "Basic s3cret"));
        // A configured token that itself contains a space still matches raw.
        assert!(bearer_token_matches("Bearer s3cret", "Bearer s3cret"));
    }

    #[test]
    fn token_header_problem_flags_values_a_header_cannot_carry() {
        for token in ["s3cret", "Bearer-ish.token_~!", "a b"] {
            assert_eq!(token_header_problem(token), None, "{token:?}");
            // Every accepted token is one the client can put in a header.
            assert!(
                token
                    .parse::<tonic::metadata::MetadataValue<tonic::metadata::Ascii>>()
                    .is_ok()
            );
        }
        for (token, problem) in [
            ("s3cret\n", "whitespace"),
            ("s3cret\r\n", "whitespace"),
            (" s3cret", "whitespace"),
            ("s3cret\t", "whitespace"),
            ("s3c\nret", "control character"),
            ("s3c\u{7f}ret", "control character"),
            ("s3c\0ret", "control character"),
            ("s3cr\u{e9}t", "non-ASCII"),
        ] {
            let found = token_header_problem(token).unwrap_or_else(|| panic!("{token:?}"));
            assert!(found.contains(problem), "{token:?}: {found}");
        }
    }

    #[test]
    fn authorize_names_the_token_problem_without_the_token() {
        let mut request = tonic::Request::new(());
        authorize(&mut request, "").unwrap();
        assert!(request.metadata().get("authorization").is_none());
        authorize(&mut request, "s3cret").unwrap();
        assert_eq!(request.metadata().get("authorization").unwrap(), "s3cret");

        let error = crate::error::Error::from(
            authorize(&mut tonic::Request::new(()), "s3cret\n").unwrap_err(),
        );
        let message = error.to_string();
        assert!(message.contains("endpoint token"), "{message}");
        assert!(message.contains("trailing whitespace"), "{message}");
        assert!(!message.contains("s3cret"), "{message}");
        assert!(!message.contains("invalid address"), "{message}");
    }

    #[test]
    fn nonempty_token_requires_exact_match() {
        assert!(bearer_token_matches("s3cret", "s3cret"));
        assert!(!bearer_token_matches("s3cret", ""));
        assert!(!bearer_token_matches("s3cret", "wrong"));
    }
}
