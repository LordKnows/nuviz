//! Terminal image capabilities.
//!
//! The protocol is found by *asking the terminal* (device-attribute and
//! graphics queries, via `ratatui-image`) rather than guessing from `TERM`:
//! most terminals report `xterm-256color` whether or not they can show
//! images. Without a positive answer the half-block fallback is used, which
//! works everywhere.

use std::env;
use std::io::IsTerminal;

use clap::ValueEnum;
use ratatui_image::picker::{Picker, ProtocolType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ImageProtocol {
    Kitty,
    Iterm2,
    Sixel,
    /// Unicode half blocks with ANSI colors (works in any terminal)
    Halfblock,
}

impl From<ProtocolType> for ImageProtocol {
    fn from(p: ProtocolType) -> Self {
        match p {
            ProtocolType::Kitty => ImageProtocol::Kitty,
            ProtocolType::Iterm2 => ImageProtocol::Iterm2,
            ProtocolType::Sixel => ImageProtocol::Sixel,
            ProtocolType::Halfblocks => ImageProtocol::Halfblock,
        }
    }
}

/// Color depth supported by the terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorDepth {
    TrueColor,
    Colors256,
    /// No reliable color: render with shading characters only.
    Mono,
}

/// Detected terminal capabilities.
#[derive(Debug, Clone)]
pub struct TerminalCapabilities {
    pub protocol: ImageProtocol,
    pub color_depth: ColorDepth,
    /// Terminal size in character cells.
    pub cols: u16,
    pub rows: u16,
    /// Size of one character cell in pixels.
    pub cell_px: (u16, u16),
}

/// What a terminal query told us.
struct Queried {
    protocol: ImageProtocol,
    cell_px: (u16, u16),
}

const DEFAULT_CELL_PX: (u16, u16) = (8, 16);

/// Detect capabilities. `requested` (from `--protocol`) and then
/// `NUVIZ_IMAGE_PROTOCOL` override detection.
pub fn detect_capabilities(requested: Option<ImageProtocol>) -> TerminalCapabilities {
    let interactive = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    // The query writes escape sequences and reads the reply, so it is only
    // attempted on a real terminal.
    let queried = if interactive { query_terminal() } else { None };

    let env_choice = env::var("NUVIZ_IMAGE_PROTOCOL")
        .ok()
        .and_then(|v| ImageProtocol::from_str(&v, true).ok());
    let protocol = choose_protocol(requested, env_choice, queried.as_ref().map(|q| q.protocol));

    let (cols, rows) = terminal_size(
        std::io::stdout().is_terminal(),
        env::var("COLUMNS").ok().as_deref(),
        env::var("LINES").ok().as_deref(),
    );
    TerminalCapabilities {
        protocol,
        color_depth: detect_color_depth(
            env::var("COLORTERM").ok().as_deref(),
            env::var("TERM").ok().as_deref(),
            env::var("NO_COLOR").is_ok_and(|v| !v.is_empty()),
            protocol,
        ),
        cols,
        rows,
        cell_px: queried.map(|q| q.cell_px).unwrap_or(DEFAULT_CELL_PX),
    }
}

/// Window size in cells. When stdout is not a terminal (piped or
/// redirected) honour `COLUMNS` / `LINES`, defaulting to 80x24.
fn terminal_size(stdout_is_tty: bool, columns: Option<&str>, lines: Option<&str>) -> (u16, u16) {
    if stdout_is_tty {
        if let Ok((cols, rows)) = crossterm::terminal::size() {
            if cols > 0 && rows > 0 {
                return (cols, rows);
            }
        }
    }
    let parse = |v: Option<&str>, default: u16| {
        v.and_then(|s| s.trim().parse::<u16>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(default)
    };
    (parse(columns, 80), parse(lines, 24))
}

fn query_terminal() -> Option<Queried> {
    let picker = Picker::from_query_stdio().ok()?;
    let (w, h) = picker.font_size();
    Some(Queried {
        protocol: picker.protocol_type().into(),
        cell_px: if w > 0 && h > 0 {
            (w, h)
        } else {
            DEFAULT_CELL_PX
        },
    })
}

/// Explicit request, then environment override, then what the terminal
/// reported; otherwise the universal fallback.
fn choose_protocol(
    requested: Option<ImageProtocol>,
    env_choice: Option<ImageProtocol>,
    queried: Option<ImageProtocol>,
) -> ImageProtocol {
    requested
        .or(env_choice)
        .or(queried)
        .unwrap_or(ImageProtocol::Halfblock)
}

fn detect_color_depth(
    colorterm: Option<&str>,
    term: Option<&str>,
    no_color: bool,
    protocol: ImageProtocol,
) -> ColorDepth {
    if no_color {
        ColorDepth::Mono
    } else if matches!(colorterm, Some("truecolor") | Some("24bit"))
        || matches!(protocol, ImageProtocol::Kitty | ImageProtocol::Iterm2)
    {
        ColorDepth::TrueColor
    } else if term.is_some_and(|t| t.contains("256color")) {
        ColorDepth::Colors256
    } else if term.is_some_and(|t| t == "dumb") || term.is_none() {
        ColorDepth::Mono
    } else {
        // Every terminal emulator in current use understands 256 colors.
        ColorDepth::Colors256
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_precedence() {
        use ImageProtocol::*;
        assert_eq!(
            choose_protocol(Some(Sixel), Some(Kitty), Some(Iterm2)),
            Sixel
        );
        assert_eq!(choose_protocol(None, Some(Kitty), Some(Iterm2)), Kitty);
        assert_eq!(choose_protocol(None, None, Some(Iterm2)), Iterm2);
    }

    #[test]
    fn test_no_positive_answer_means_halfblock_not_sixel() {
        // TERM=xterm-256color says nothing about image support.
        assert_eq!(choose_protocol(None, None, None), ImageProtocol::Halfblock);
    }

    #[test]
    fn test_color_depth() {
        use ImageProtocol::Halfblock;
        assert_eq!(
            detect_color_depth(Some("truecolor"), Some("xterm"), false, Halfblock),
            ColorDepth::TrueColor
        );
        assert_eq!(
            detect_color_depth(None, Some("xterm-256color"), false, Halfblock),
            ColorDepth::Colors256
        );
        assert_eq!(
            detect_color_depth(None, Some("xterm"), false, ImageProtocol::Kitty),
            ColorDepth::TrueColor
        );
        assert_eq!(
            detect_color_depth(Some("truecolor"), Some("xterm"), true, Halfblock),
            ColorDepth::Mono
        );
        assert_eq!(
            detect_color_depth(None, Some("dumb"), false, Halfblock),
            ColorDepth::Mono
        );
    }

    #[test]
    fn test_detect_outside_a_terminal_does_not_query_or_panic() {
        // Under `cargo test` stdio is not a terminal.
        let caps = detect_capabilities(None);
        assert!(caps.cols > 0 && caps.rows > 0);
        assert_eq!(caps.cell_px, DEFAULT_CELL_PX);
    }

    #[test]
    fn test_size_when_not_a_terminal() {
        assert_eq!(terminal_size(false, None, None), (80, 24));
        assert_eq!(terminal_size(false, Some("132"), Some("50")), (132, 50));
        assert_eq!(terminal_size(false, Some("0"), Some("junk")), (80, 24));
    }

    #[test]
    fn test_env_values_parse() {
        assert_eq!(
            ImageProtocol::from_str("kitty", true),
            Ok(ImageProtocol::Kitty)
        );
        assert_eq!(
            ImageProtocol::from_str("HALFBLOCK", true),
            Ok(ImageProtocol::Halfblock)
        );
        assert!(ImageProtocol::from_str("ascii", true).is_err());
    }
}
