//! The `rlmesh.toml` array scan `build.rs` reads `[workflow] supported_editions`
//! with.
//!
//! It lives beside `build.rs` instead of inside it because a `#[test]` in a
//! build script never runs: `src/lib.rs` includes this file with `#[path]` under
//! `cfg(test)`, so `cargo test -p rlmesh-proto` covers the real parser. (A
//! `build/` directory would be gitignored repo-wide and never ship in the
//! published crate, so the module is a sibling file.)

/// Values of a TOML array of strings (`key = ["a", "b"]`), written on one line
/// or spread across several — the array ends at the first `]`. `#` comments and
/// trailing commas are ignored. `scripts/check_rlmesh_policy.py --selfcheck`
/// mirrors this scan and asserts it agrees with a real TOML parse, so the two
/// readers of the manifest cannot drift.
pub fn manifest_string_list(text: &str, key: &str) -> Vec<String> {
    let prefix = format!("{key} = [");
    let mut lines = text.lines().map(str::trim);
    let Some(first) = lines.find_map(|line| line.strip_prefix(prefix.as_str())) else {
        return Vec::new();
    };
    let mut items = String::new();
    for line in std::iter::once(first).chain(lines) {
        let line = line.split_once('#').map_or(line, |(head, _)| head);
        if let Some((head, _)) = line.split_once(']') {
            items.push_str(head);
            break;
        }
        items.push_str(line);
        // A line break separates two entries exactly as a comma does; empty
        // entries (a trailing comma, a comment-only line) drop out below.
        items.push(',');
    }
    items
        .split(',')
        .filter_map(|item| {
            let (value, _) = item.trim().strip_prefix('"')?.split_once('"')?;
            Some(value.to_string())
        })
        .collect()
}
