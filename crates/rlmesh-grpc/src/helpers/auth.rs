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
/// Otherwise the provided token must match in constant time.
#[must_use]
pub fn bearer_token_matches(configured: &str, provided: &str) -> bool {
    if configured.is_empty() {
        return true;
    }
    constant_time_eq(configured.as_bytes(), provided.as_bytes())
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
    fn nonempty_token_requires_exact_match() {
        assert!(bearer_token_matches("s3cret", "s3cret"));
        assert!(!bearer_token_matches("s3cret", ""));
        assert!(!bearer_token_matches("s3cret", "wrong"));
    }
}
