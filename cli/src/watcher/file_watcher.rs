use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

use notify::{EventKind, RecursiveMode, Watcher};

use crate::data::metrics::MetricRecord;
use crate::watcher::tail::TailReader;

/// Watch a metrics.jsonl file for new records.
///
/// Returns a receiver that yields batches of new `MetricRecord`s
/// as they are appended to the file. The watcher thread runs in
/// the background until the receiver is dropped.
pub fn watch_metrics(
    jsonl_path: &Path,
    use_polling: bool,
) -> anyhow::Result<mpsc::Receiver<Vec<MetricRecord>>> {
    let (record_tx, record_rx) = mpsc::channel::<Vec<MetricRecord>>();
    let (notify_tx, notify_rx) = mpsc::channel::<notify::Result<notify::Event>>();

    let parent = jsonl_path
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();

    let path = jsonl_path.to_path_buf();

    if use_polling {
        let config =
            notify::Config::default().with_poll_interval(Duration::from_secs(1));
        let mut watcher = notify::PollWatcher::new(notify_tx, config)?;
        if parent.exists() {
            watcher.watch(&parent, RecursiveMode::NonRecursive)?;
        }
        std::thread::spawn(move || {
            let _watcher = watcher; // keep alive
            watch_loop(&path, notify_rx, record_tx);
        });
    } else {
        let mut watcher = notify::recommended_watcher(notify_tx)?;
        if parent.exists() {
            watcher.watch(&parent, RecursiveMode::NonRecursive)?;
        }
        std::thread::spawn(move || {
            let _watcher = watcher; // keep alive
            watch_loop(&path, notify_rx, record_tx);
        });
    }

    Ok(record_rx)
}

fn watch_loop(
    path: &Path,
    notify_rx: mpsc::Receiver<notify::Result<notify::Event>>,
    record_tx: mpsc::Sender<Vec<MetricRecord>>,
) {
    let mut reader = TailReader::new(path);

    // Initial read of existing data
    let records = reader.read_new();
    if !records.is_empty() {
        if record_tx.send(records).is_err() {
            return;
        }
    }

    // Process file change events
    loop {
        match notify_rx.recv_timeout(Duration::from_millis(500)) {
            Ok(Ok(event)) => {
                if matches!(
                    event.kind,
                    EventKind::Modify(_) | EventKind::Create(_)
                ) {
                    read_and_send(&mut reader, &record_tx);
                }
            }
            Ok(Err(_)) => {}
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Periodic fallback poll
                read_and_send(&mut reader, &record_tx);
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}

fn read_and_send(
    reader: &mut TailReader,
    tx: &mpsc::Sender<Vec<MetricRecord>>,
) {
    let records = reader.read_new();
    if !records.is_empty() {
        let _ = tx.send(records);
    }
}
