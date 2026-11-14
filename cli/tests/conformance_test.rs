//! Cross-language wire-format conformance.
//!
//! `spec/fixtures/records.json` holds logical records and
//! `spec/fixtures/metrics.jsonl` their golden wire encoding. The Python writer
//! is tested against the same files (`python/tests/test_conformance.py`).

use std::path::PathBuf;

use nuviz_cli::data::metrics::read_metrics;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("spec")
        .join("fixtures")
}

fn expect_number(v: &serde_json::Value) -> f64 {
    match v {
        serde_json::Value::String(s) => match s.as_str() {
            "NaN" | "Infinity" | "-Infinity" => f64::NAN, // non-finite travels as null
            other => panic!("unexpected marker {other}"),
        },
        other => other.as_f64().expect("number"),
    }
}

#[test]
fn golden_file_parses_to_logical_records() {
    let logical: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("records.json")).unwrap())
            .unwrap();
    let parsed = read_metrics(&fixtures().join("metrics.jsonl"));

    assert_eq!(
        parsed.len(),
        logical.len(),
        "every golden line must parse, including lines with null metrics"
    );

    for (got, want) in parsed.iter().zip(&logical) {
        assert_eq!(got.step, want["step"].as_u64().unwrap());
        assert_eq!(got.timestamp, want["timestamp"].as_f64().unwrap());

        let want_metrics = want["metrics"].as_object().unwrap();
        assert_eq!(got.metrics.len(), want_metrics.len(), "step {}", got.step);
        for (name, value) in want_metrics {
            let got_value = got
                .metrics
                .get(name)
                .unwrap_or_else(|| panic!("metric {name:?} missing at step {}", got.step));
            let want_value = expect_number(value);
            if want_value.is_nan() {
                assert!(got_value.is_nan(), "{name} at step {}", got.step);
            } else {
                assert_eq!(*got_value, want_value, "{name} at step {}", got.step);
            }
        }

        match want.get("gpu") {
            Some(gpu) => {
                let got_gpu = got.gpu.as_ref().expect("gpu present");
                for (name, value) in gpu.as_object().unwrap() {
                    assert_eq!(got_gpu[name], value.as_f64().unwrap());
                }
            }
            None => assert!(got.gpu.is_none()),
        }
    }
}
