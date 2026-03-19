# Contributing to NuViz

## Development Setup

### Prerequisites

- Python 3.10+
- Rust stable (latest)
- (Optional) nvidia-smi for GPU metrics testing

### Python Library

```bash
cd python
pip install -e ".[dev,images,yaml]"

# Run tests
pytest tests/ -v
pytest tests/ --cov=nuviz --cov-report=term-missing --cov-fail-under=80

# Lint and type check
ruff check src/ tests/
mypy src/nuviz/ --ignore-missing-imports
```

### Rust CLI

```bash
cd cli
cargo build
cargo test --lib
cargo clippy -- -D warnings
cargo fmt -- --check
```

### Integration Tests

Integration tests require both Python and Rust to be set up:

```bash
cd python && pip install -e ".[dev,images,yaml]"
cd ../cli && cargo test --test integration_test --test hardening_test
```

## Project Structure

```
python/src/nuviz/    # Python logger library
cli/src/             # Rust CLI application
  commands/          # CLI command implementations
  data/              # Data parsing and experiment discovery
  tui/               # Terminal UI (ratatui)
  watcher/           # File watching and tail reading
  terminal/          # Terminal capability detection and image rendering
docs/                # Design docs and usage guide
examples/            # Demo scripts
```

## Code Style

- **Python**: Enforced by `ruff` (PEP 8 + extras). All dataclasses are frozen/immutable.
- **Rust**: Enforced by `cargo fmt` and `cargo clippy -D warnings`.
- Minimum 80% test coverage for Python.

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests first (TDD encouraged)
3. Ensure all CI checks pass
4. Keep commits focused with conventional commit messages (`feat:`, `fix:`, `test:`, etc.)

## Data Format

Python writes, Rust reads. If changing the data format:
- Update the Python writer and Rust parser in the same PR
- Ensure backwards compatibility (new fields should use `#[serde(default)]` in Rust)
- Add integration tests covering the new format
