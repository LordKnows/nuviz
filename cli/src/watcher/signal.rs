//! Filesystem change notifications, used only to make the UI react faster.
//!
//! The dashboard polls its sources on a timer regardless, so everything
//! keeps working when notifications are unavailable (NFS, some containers)
//! or disabled with `--poll`.

use std::path::PathBuf;
use std::sync::mpsc;

use notify::{RecursiveMode, Watcher};

pub struct FsSignal {
    rx: Option<mpsc::Receiver<()>>,
    // Dropping the watcher stops notifications; keep it alive.
    _watcher: Option<notify::RecommendedWatcher>,
}

impl FsSignal {
    /// Never signals; callers rely on their polling interval.
    pub fn disabled() -> Self {
        Self {
            rx: None,
            _watcher: None,
        }
    }

    /// Watch the given directories. Falls back to [`FsSignal::disabled`] if
    /// the platform watcher cannot be set up.
    pub fn watch(dirs: &[PathBuf]) -> Self {
        let (tx, rx) = mpsc::channel();
        let watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
            if event.is_ok() {
                let _ = tx.send(());
            }
        });
        let Ok(mut watcher) = watcher else {
            return Self::disabled();
        };
        let mut watching = false;
        for dir in dirs {
            watching |= watcher.watch(dir, RecursiveMode::NonRecursive).is_ok();
        }
        if !watching {
            return Self::disabled();
        }
        Self {
            rx: Some(rx),
            _watcher: Some(watcher),
        }
    }

    /// Whether anything changed since the last call.
    pub fn changed(&self) -> bool {
        let Some(rx) = &self.rx else {
            return false;
        };
        let mut any = false;
        while rx.try_recv().is_ok() {
            any = true;
        }
        any
    }
}
