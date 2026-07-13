//! Minimal ANSI styling for CLI output.
//!
//! Escape codes are emitted only when the target stream is a terminal and the
//! environment has not opted out (any non-empty `NO_COLOR`, or `TERM=dumb`),
//! so piped, captured, and CI output stays plain. The handful of codes the CLI
//! uses are universal; no styling dependency is needed.
//!
//! Terminality (`interactive`) and colorability (`color`) are tracked
//! separately: `NO_COLOR` is a color-only opt-out, so a user who sets it in an
//! interactive terminal still gets the browser opened and progress feedback,
//! just without ANSI codes.

/// Whether to decorate output for one stream, and whether that stream is an
/// interactive terminal. Construct with [`Style::for_terminal`] at the point
/// that knows the real sink, rather than probing process fds from deep in the
/// call tree (the injected writer may not be the process stream).
#[derive(Clone, Copy)]
pub struct Style {
    color: bool,
    interactive: bool,
}

impl Style {
    /// Styling for a stream, given whether that stream is a terminal. Color is
    /// additionally gated on the `NO_COLOR`/`TERM=dumb` opt-outs; interactivity
    /// is not.
    pub fn for_terminal(is_terminal: bool) -> Self {
        Style {
            color: is_terminal && env_allows_color(),
            interactive: is_terminal,
        }
    }

    /// Whether this stream is an interactive terminal. The right gate for
    /// interactive-only behavior (progress dots, opening a browser) — distinct
    /// from colorability, which `NO_COLOR` can disable independently.
    pub fn interactive(&self) -> bool {
        self.interactive
    }

    fn paint(&self, code: &str, text: &str) -> String {
        if self.color {
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
        let style = Style {
            color: false,
            interactive: false,
        };
        assert_eq!(style.bold("code"), "code");
        assert_eq!(style.green("ok"), "ok");
        assert!(!style.interactive());
    }

    #[test]
    fn enabled_style_wraps_in_ansi_codes() {
        let style = Style {
            color: true,
            interactive: true,
        };
        assert_eq!(style.bold("code"), "\x1b[1mcode\x1b[0m");
        assert_eq!(style.red_bold("Error:"), "\x1b[1;31mError:\x1b[0m");
        assert!(style.interactive());
    }

    #[test]
    fn interactive_is_independent_of_color() {
        let style = Style {
            color: false,
            interactive: true,
        };
        assert_eq!(style.bold("code"), "code");
        assert!(style.interactive());
    }
}
