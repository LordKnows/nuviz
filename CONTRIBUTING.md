# Contributing to NuViz

## Development Setup

### Prerequisites

- Python 3.10+
- Rust 1.88 or newer (CI lints with a pinned toolchain; see `.github/workflows/ci.yml`)
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
cargo clippy --all-targets -- -D warnings
cargo fmt -- --check
```

### Integration Tests

These need both the Python package and the Rust toolchain:

```bash
cd python && pip install -e ".[dev,images,yaml]"
cd ../cli && cargo test --test integration_test --test hardening_test \
                        --test conformance_test --test e2e_test
```

- `conformance_test` parses the golden files in `spec/fixtures/`; the Python
  suite (`tests/test_conformance.py`) writes the same files byte for byte.
- `e2e_test` has the Python logger write experiments and then drives the real
  `nuviz` binary against them.

### Trying the dashboard

```bash
python examples/demo_train.py &          # writes to ~/.nuviz/experiments
cd cli && cargo run -- watch
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
spec/fixtures/       # Golden files shared by the Python and Rust test suites
docs/                # Usage guide, data format, design doc
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

Python writes, Rust reads. The contract is [docs/FORMAT.md](docs/FORMAT.md).
If you change it:

- Update `docs/FORMAT.md`, the Python writer and the Rust reader in the same PR.
- Update `spec/fixtures/` so both test suites cover the change.
- Keep old data readable: new keys must be optional on the reader side, and a
  change to the meaning of an existing key needs a `schema_version` bump.

## Releasing

Releases are cut by pushing a `v<version>` tag; `.github/workflows/release.yml`
tests, builds and publishes. The workflow refuses to run unless the tag
matches the version in `cli/Cargo.toml`, `python/pyproject.toml` and
`python/src/nuviz/__init__.py`.

One-time setup:

- **PyPI** uses Trusted Publishing, so no API token is stored. On pypi.org,
  add a trusted publisher to both projects, `nuviz` and `nuviz-cli`
  (repository `LordKnows/nuviz`, workflow `release.yml`, environment `pypi`),
  and create a `pypi` environment in the GitHub repository settings.
- **crates.io** uses the `CARGO_REGISTRY_TOKEN` repository secret.

A failed publish fails the workflow. Re-running it is safe: versions that are
already published are skipped.
