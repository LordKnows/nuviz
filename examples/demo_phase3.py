"""Phase 3 demo: image browsing, visual diff, point clouds, PLY stats, JSONL rotation.

This script exercises all Phase 3 features:
1. Logger with image saving — render + ground truth images per step
2. Logger.pointcloud() — save 3DGS-style PLY files
3. JSONL rotation — triggered by small line limit
4. CLI commands to verify: image, diff, view, ls (rotation)

Usage:
    cd python && pip install -e ".[dev,images,yaml]"
    python ../examples/demo_phase3.py

Then try the CLI (from cli/):
    # Commands are printed at the end of the script
"""

import math
import random
import struct
import tempfile
from pathlib import Path

import numpy as np

from nuviz import Logger
from nuviz.config import NuvizConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT = "nuviz-phase3-test"
STEPS = 50
ROTATION_LINE_LIMIT = 20  # Low limit to trigger rotation during demo

BASE_DIR = Path(tempfile.mkdtemp(prefix="nuviz_p3_demo_"))
print(f"Data directory: {BASE_DIR}\n")

config = NuvizConfig(
    base_dir=BASE_DIR,
    flush_interval_seconds=0.1,
    flush_count=5,
    enable_alerts=False,
    enable_snapshot=False,
    rotate_max_size_bytes=50_000_000,
    rotate_max_lines=ROTATION_LINE_LIMIT,
)

# ---------------------------------------------------------------------------
# Helper: generate synthetic images
# ---------------------------------------------------------------------------
def make_gradient_image(w: int, h: int, step: int, seed: int) -> np.ndarray:
    """Create a simple gradient image that changes with step."""
    rng = np.random.RandomState(seed + step)
    base = np.zeros((h, w, 3), dtype=np.float32)
    # Horizontal gradient shifts with step
    for c in range(3):
        shift = (step * 5 + c * 80) % 256
        base[:, :, c] = np.linspace(shift / 255, (shift + 128) / 255, w)[None, :]
    # Add some noise
    base += rng.randn(h, w, 3).astype(np.float32) * 0.05
    return np.clip(base, 0, 1)


def make_gt_image(w: int, h: int, seed: int) -> np.ndarray:
    """Create a fixed ground truth image."""
    rng = np.random.RandomState(seed)
    base = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        base[:, :, c] = np.linspace(c * 0.3, 0.5 + c * 0.2, w)[None, :]
    base += rng.randn(h, w, 3).astype(np.float32) * 0.02
    return np.clip(base, 0, 1)


# ---------------------------------------------------------------------------
# Step 1: Training run with images
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Training run with images + point clouds")
print("=" * 60)

EXP_A = "phase3-model-a"
EXP_B = "phase3-model-b"

experiment_names = [EXP_A, EXP_B]
img_w, img_h = 64, 48  # Small images for demo speed

for exp_idx, exp_name in enumerate(experiment_names):
    random.seed(exp_idx * 42)
    print(f"\nRunning: {exp_name}")

    with Logger(
        exp_name,
        project=PROJECT,
        config=config,
        snapshot=False,
    ) as log:
        gt_img = make_gt_image(img_w, img_h, seed=999)

        for step in range(STEPS):
            # Metrics — model B converges slightly better
            quality_offset = exp_idx * 1.5
            loss = 2.0 * math.exp(-step / (30 + exp_idx * 10)) + random.gauss(0, 0.03)
            psnr = 22.0 + quality_offset + 6.0 * (1 - math.exp(-step / 40)) + random.gauss(0, 0.3)

            log.step(step, loss=loss, psnr=psnr)

            # Save images every 10 steps
            if step % 10 == 0:
                render = make_gradient_image(img_w, img_h, step, seed=exp_idx)
                log.image("render", render)
                log.image("gt", gt_img)

                # Save a depth-like image
                depth = np.random.RandomState(step + exp_idx).rand(img_h, img_w).astype(np.float32)
                log.image("depth", depth, cmap="turbo")

        # -----------------------------------------------------------------
        # Step 2: Save point clouds at final step
        # -----------------------------------------------------------------
        num_points = 500
        rng = np.random.RandomState(exp_idx)

        xyz = rng.randn(num_points, 3).astype(np.float32) * 2.0
        colors = (rng.rand(num_points, 3) * 255).astype(np.uint8)
        opacities = rng.rand(num_points).astype(np.float32)

        log.pointcloud("gaussians", xyz, colors=colors, opacities=opacities)
        print(f"  Saved {num_points}-point PLY")

    print(f"  → {log.experiment_dir}")

# ---------------------------------------------------------------------------
# Step 3: Verify output files
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("Step 2: Verify output")
print("=" * 60)

project_dir = BASE_DIR / PROJECT
for exp_name in experiment_names:
    exp_dir = project_dir / exp_name
    if not exp_dir.exists():
        print(f"  WARNING: {exp_name} not found!")
        continue

    # Count files
    metrics_files = sorted(exp_dir.glob("metrics*.jsonl"))
    images = sorted((exp_dir / "images").glob("*.png")) if (exp_dir / "images").exists() else []
    plys = sorted((exp_dir / "pointclouds").glob("*.ply")) if (exp_dir / "pointclouds").exists() else []

    print(f"\n  {exp_name}/")
    print(f"    JSONL files:  {len(metrics_files)} ({', '.join(f.name for f in metrics_files)})")

    total_lines = sum(len(f.read_text().strip().splitlines()) for f in metrics_files)
    print(f"    Total records: {total_lines}")
    print(f"    Images:        {len(images)} PNG files")
    print(f"    Point clouds:  {len(plys)} PLY files")

    if plys:
        ply_path = plys[0]
        ply_size = ply_path.stat().st_size
        # Quick header peek
        with open(ply_path, "rb") as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if b"end_header" in line:
                    break
        header_str = header.decode("ascii", errors="replace")
        vertex_line = [l for l in header_str.splitlines() if "element vertex" in l]
        if vertex_line:
            print(f"    PLY: {vertex_line[0].strip()}, {ply_size} bytes")

# ---------------------------------------------------------------------------
# Print CLI commands to try
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("Step 3: Try these CLI commands")
print("=" * 60)

dir_flag = f"--dir {BASE_DIR}"

print(f"""
# List experiments (note: JSONL rotation visible in file listing)
nuviz ls {dir_flag} --project {PROJECT}

# Browse images interactively (arrow keys to navigate, q to quit)
nuviz image {EXP_A} {dir_flag} --project {PROJECT}

# Show latest render image
nuviz image {EXP_A} {dir_flag} --project {PROJECT} --tag render --latest

# Side-by-side: render vs ground truth
nuviz image {EXP_A} {dir_flag} --project {PROJECT} --side-by-side gt --step 40

# Visual diff between two experiments
nuviz diff {EXP_A} {EXP_B} {dir_flag} --project {PROJECT} --tag render --step 40

# Visual diff with error heatmap
nuviz diff {EXP_A} {EXP_B} {dir_flag} --project {PROJECT} --tag render --step 40 --heatmap

# Point cloud statistics
nuviz view {EXP_A} {dir_flag} --project {PROJECT}

# Point cloud with histograms
nuviz view {EXP_A} {dir_flag} --project {PROJECT} --histogram

# Direct PLY file path also works:
nuviz view {project_dir / EXP_A / "pointclouds" / "gaussians_step_000049.ply"}

# Leaderboard (still works — reads rotated JSONL seamlessly)
nuviz leaderboard {dir_flag} --project {PROJECT} --sort psnr
""")

print("Demo complete!")
print(f"Data lives at: {BASE_DIR}")
print("Delete with: rm -rf", BASE_DIR)
