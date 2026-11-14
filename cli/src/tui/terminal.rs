//! Terminal setup and guaranteed teardown.

use std::io::{self, Stdout};

use anyhow::Result;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

/// Puts the terminal into raw mode on the alternate screen and restores it
/// when dropped — on normal exit, on `?` early returns, and on panic.
pub struct TerminalGuard {
    pub terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalGuard {
    pub fn enter() -> Result<Self> {
        install_panic_hook();
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        if let Err(e) = execute!(stdout, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(e.into());
        }
        match Terminal::new(CrosstermBackend::new(stdout)) {
            Ok(terminal) => Ok(Self { terminal }),
            Err(e) => {
                restore();
                Err(e.into())
            }
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore();
        let _ = self.terminal.show_cursor();
    }
}

fn restore() {
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen);
}

/// Restore the terminal before the default panic message is printed, so a
/// bug never leaves the user's shell in raw mode with the message lost on
/// the alternate screen.
fn install_panic_hook() {
    use std::sync::Once;
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            restore();
            default(info);
        }));
    });
}
