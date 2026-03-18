use std::env;

/// Color depth supported by the terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorDepth {
    TrueColor,
    Colors256,
    Colors16,
}

/// Detected terminal capabilities.
#[derive(Debug, Clone)]
pub struct TerminalCapabilities {
    pub supports_kitty_graphics: bool,
    pub supports_iterm2: bool,
    pub supports_sixel: bool,
    pub color_depth: ColorDepth,
    pub unicode_support: bool,
    pub terminal_name: Option<String>,
}

/// Detect terminal capabilities from environment variables.
///
/// This is detection only — actual image rendering is Phase 3.
pub fn detect_capabilities() -> TerminalCapabilities {
    let term_program = env::var("TERM_PROGRAM").ok();
    let term = env::var("TERM").ok();
    let colorterm = env::var("COLORTERM").ok();
    let lang = env::var("LANG").ok();

    let term_program_lower = term_program
        .as_deref()
        .unwrap_or("")
        .to_lowercase();

    // Kitty graphics protocol
    let supports_kitty = term_program_lower == "kitty"
        || term_program_lower == "wezterm"
        || term_program_lower == "ghostty"
        || env::var("KITTY_WINDOW_ID").is_ok();

    // iTerm2 inline images
    let supports_iterm2 = term_program_lower.contains("iterm")
        || term_program_lower == "wezterm"
        || env::var("ITERM_SESSION_ID").is_ok();

    // Sixel support (heuristic)
    let supports_sixel = term
        .as_deref()
        .map(|t| t.contains("xterm") || t.contains("foot") || t.contains("mlterm"))
        .unwrap_or(false)
        || term_program_lower == "wezterm";

    // Color depth
    let color_depth = if colorterm.as_deref() == Some("truecolor")
        || colorterm.as_deref() == Some("24bit")
        || supports_kitty
    {
        ColorDepth::TrueColor
    } else if term.as_deref().is_some_and(|t| t.contains("256color")) {
        ColorDepth::Colors256
    } else {
        ColorDepth::Colors16
    };

    // Unicode support
    let unicode_support = lang
        .as_deref()
        .map(|l| l.contains("UTF-8") || l.contains("utf-8") || l.contains("UTF8"))
        .unwrap_or(false)
        || supports_kitty;

    TerminalCapabilities {
        supports_kitty_graphics: supports_kitty,
        supports_iterm2,
        supports_sixel,
        color_depth,
        unicode_support,
        terminal_name: term_program,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_returns_valid_struct() {
        let caps = detect_capabilities();
        // Just verify it doesn't panic and returns something sensible
        let _ = caps.color_depth;
        let _ = caps.supports_kitty_graphics;
        let _ = caps.supports_sixel;
        let _ = caps.supports_iterm2;
        let _ = caps.unicode_support;
    }

    #[test]
    fn test_color_depth_variants() {
        // Verify enum variants exist and can be compared
        assert_ne!(ColorDepth::TrueColor, ColorDepth::Colors256);
        assert_ne!(ColorDepth::Colors256, ColorDepth::Colors16);
    }
}
