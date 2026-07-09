//! Minimal ANSI styling for CLI output.
//!
//! Escape codes are emitted only when the target stream is a terminal and the
//! environment has not opted out (any non-empty `NO_COLOR`, or `TERM=dumb`),
//! so piped, captured, and CI output stays plain. The handful of codes the CLI
//! uses are universal; no styling dependency is needed.

use std::io::IsTerminal;

/// Whether to decorate output for one stream, and the decorations themselves.
#[derive(Clone, Copy)]
pub struct Style {
    enabled: bool,
}

impl Style {
    /// Styling for text written to the process's stdout.
    pub fn stdout() -> Self {
        Self::for_terminal(std::io::stdout().is_terminal())
    }

    /// Styling for text written to the process's stderr.
    pub fn stderr() -> Self {
        Self::for_terminal(std::io::stderr().is_terminal())
    }

    fn for_terminal(is_terminal: bool) -> Self {
        Style {
            enabled: is_terminal && env_allows_color(),
        }
    }

    /// Whether this stream is an interactive, color-capable terminal. Also the
    /// right gate for other interactive-only behavior (spinners, progress
    /// dots, opening a browser).
    pub fn interactive(&self) -> bool {
        self.enabled
    }

    fn paint(&self, code: &str, text: &str) -> String {
        if self.enabled {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_string()
        }
    }

    pub fn bold(&self, text: &str) -> String {
        self.paint("1", text)
    }

    pub fn green(&self, text: &str) -> String {
        self.paint("32", text)
    }

    pub fn cyan(&self, text: &str) -> String {
        self.paint("36", text)
    }

    pub fn red_bold(&self, text: &str) -> String {
        self.paint("1;31", text)
    }
}

/// The conventional opt-outs: any non-empty `NO_COLOR` (https://no-color.org)
/// or `TERM=dumb`.
fn env_allows_color() -> bool {
    let no_color = std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty());
    let dumb_term = std::env::var_os("TERM").is_some_and(|term| term == "dumb");
    !no_color && !dumb_term
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_style_passes_text_through() {
        let style = Style { enabled: false };
        assert_eq!(style.bold("code"), "code");
        assert_eq!(style.green("ok"), "ok");
        assert!(!style.interactive());
    }

    #[test]
    fn enabled_style_wraps_in_ansi_codes() {
        let style = Style { enabled: true };
        assert_eq!(style.bold("code"), "\x1b[1mcode\x1b[0m");
        assert_eq!(style.red_bold("Error:"), "\x1b[1;31mError:\x1b[0m");
        assert!(style.interactive());
    }
}
