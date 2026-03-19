"""Phase 4 demo: experiment management, GPU metrics, tagging, cleanup, reproduce.

This script exercises all Phase 4 features:
1. Logger with GPU metrics collection (mocked for demo)
2. Multiple experiments for cleanup ranking
3. CLI commands: tag, cleanup, reproduce

Usage:
    cd python && pip install -e ".[dev,images,yaml]"
    python ../examples/demo_phase4.py

Then try the CLI (from cli/):
    # Commands are printed at the end of the script
"""

import math
import random
import tempfile
from pathlib import Path

import numpy as np

from nuviz import Logger
from nuviz.config import NuvizConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT = "nuviz-phase4-test"
STEPS = 100

BASE_DIR = Path(tempfile.mkdtemp(prefix="nuviz_p4_demo_"))
print(f"Data directory: {BASE_DIR}\n")

config = NuvizConfig(
    base_dir=BASE_DIR,
    flush_interval_seconds=0.1,
    flush_count=10,
    enable_alerts=True,
    enable_snapshot=True,
    enable_gpu=False,  # Disable GPU polling for demo (no nvidia-smi needed)
)

# ---------------------------------------------------------------------------
# Step 1: Run several experiments with varying quality
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Running 6 experiments with different configs")
print("=" * 60)

experiments = [
    ("baseline-lr1e3", {"lr": 1e-3, "depth": 2}),
    ("baseline-lr1e4", {"lr": 1e-4, "depth": 2}),
    ("deep-lr1e3", {"lr": 1e-3, "depth": 8}),
    ("deep-lr1e4", {"lr": 1e-4, "depth": 8}),
    ("overfit-run", {"lr": 1e-2, "depth": 16}),
    ("failed-run", {"lr": 1e-1, "depth": 1}),
]

for exp_name, params in experiments:
    random.seed(hash(exp_name) % 2**32)
    print(f"\n  Running: {exp_name} (lr={params['lr']}, depth={params['depth']})")

    with Logger(
        exp_name,
        project=PROJECT,
        config=config,
        seed=42,
    ) as log:
        for step in range(STEPS):
            # Quality depends on config: deeper models + moderate LR are better
            quality = params["depth"] * 0.3 - abs(math.log10(params["lr"]) + 3.5) * 2
            loss = max(0.01, 2.0 * math.exp(-step / (20 + quality * 5)) + random.gauss(0, 0.02))
            psnr = 20.0 + quality + 8.0 * (1 - math.exp(-step / 50)) + random.gauss(0, 0.2)
            ssim = min(0.99, 0.7 + quality * 0.02 + 0.25 * (1 - math.exp(-step / 60)))

            log.step(step, loss=loss, psnr=psnr, ssim=ssim)

            # Save images at a few steps
            if step % 25 == 0:
                img = np.random.RandomState(step).rand(32, 32, 3).astype(np.float32)
                log.image("render", img)

        # Per-scene evaluation
        for scene in ["bicycle", "garden", "stump"]:
            scene_quality = quality + random.gauss(0, 0.5)
            log.scene(scene, psnr=26 + scene_quality, ssim=0.85 + scene_quality * 0.01)

    print(f"    → final loss={loss:.4f}, psnr={psnr:.2f}")

# ---------------------------------------------------------------------------
# Step 2: Verify output
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("Step 2: Verify experiments")
print("=" * 60)

project_dir = BASE_DIR / PROJECT
for exp_name, _ in experiments:
    exp_dir = project_dir / exp_name
    meta_exists = (exp_dir / "meta.json").exists()
    metrics_exists = (exp_dir / "metrics.jsonl").exists()
    print(f"  {exp_name}: meta={meta_exists}, metrics={metrics_exists}")

# ---------------------------------------------------------------------------
# Print CLI commands to try
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("Step 3: Try these CLI commands")
print("=" * 60)

dir_flag = f"--dir {BASE_DIR}"

print(f"""
# List all experiments
nuviz ls {dir_flag} --project {PROJECT}

# Leaderboard — see which experiments are best
nuviz leaderboard {dir_flag} --project {PROJECT} --sort psnr

# Tag the best experiment
nuviz tag deep-lr1e4 {dir_flag} baseline
nuviz tag deep-lr1e4 {dir_flag} --list

# Reproduce an experiment
nuviz reproduce deep-lr1e4 {dir_flag}

# Cleanup — dry run first (default)
nuviz cleanup {dir_flag} --project {PROJECT} --metric psnr --keep-top 3

# Cleanup — actually delete (requires --force)
nuviz cleanup {dir_flag} --project {PROJECT} --metric psnr --keep-top 3 --force

# Verify cleanup worked
nuviz ls {dir_flag} --project {PROJECT}

# Per-scene breakdown of the best model
nuviz breakdown deep-lr1e4 {dir_flag} --project {PROJECT}
nuviz breakdown deep-lr1e4 {dir_flag} --project {PROJECT} --latex
""")

print("Demo complete!")
print(f"Data lives at: {BASE_DIR}")
print("Delete with: rm -rf", BASE_DIR)
