"""Phase 2 demo: ablation configs, multi-seed training, scene evaluation, and export.

This script exercises all Phase 2 features:
1. Ablation class — generate experiment configs
2. Logger with seed/config_hash — simulate multi-seed runs
3. log.scene() — per-scene evaluation metrics
4. CLI commands to verify: ls, leaderboard, breakdown, export, matrix

Usage:
    cd python && pip install -e ".[dev,images,yaml]"
    python ../examples/demo_phase2.py

Then try the CLI (from cli/):
    cargo run -- ls --project nuviz-phase2-test
    cargo run -- leaderboard --project nuviz-phase2-test --format latex
    cargo run -- breakdown demo-lr_0.001-sh_2_seed0 --project nuviz-phase2-test
    cargo run -- breakdown demo-lr_0.001-sh_2_seed0 --project nuviz-phase2-test --latex
    cargo run -- breakdown demo-lr_0.001-sh_2_seed0 --diff demo-lr_0.0001-sh_2_seed0 --project nuviz-phase2-test
    cargo run -- export demo-lr_0.001-sh_2_seed0 --project nuviz-phase2-test --format csv
    cargo run -- export demo-lr_0.001-sh_2_seed0 --project nuviz-phase2-test --format json
    cargo run -- matrix --rows lr --cols sh --metric psnr --project nuviz-phase2-test
    cargo run -- matrix --rows lr --cols sh --metric psnr --project nuviz-phase2-test --format latex
    cargo run -- compare demo-lr_0.001-sh_2_seed0 demo-lr_0.0001-sh_2_seed0 --metric loss
"""

import math
import random
import shutil
import sys
import tempfile
from pathlib import Path

from nuviz import Ablation, Logger
from nuviz.config import NuvizConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT = "nuviz-phase2-test"
STEPS_PER_RUN = 150
SEEDS = [0, 1]
SCENES = ["bicycle", "garden", "stump", "room", "counter"]

# Use a temp dir so we don't pollute ~/.nuviz
BASE_DIR = Path(tempfile.mkdtemp(prefix="nuviz_demo_"))
print(f"Data directory: {BASE_DIR}\n")

config = NuvizConfig(
    base_dir=BASE_DIR,
    flush_interval_seconds=0.1,
    flush_count=10,
    enable_alerts=True,
    enable_snapshot=False,
)

# ---------------------------------------------------------------------------
# Step 1: Ablation — generate experiment configs
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Ablation config generation")
print("=" * 60)

ab = Ablation("demo", base_config={"lr": 1e-3, "sh": 2, "dense": True})
ab.vary("lr", [1e-3, 1e-4])
ab.vary("sh", [2, 3])

configs = ab.generate()
print(f"Generated {len(configs)} configs (2 lr x 2 sh):")
for c in configs:
    print(f"  lr={c['lr']:.0e}  sh={c['sh']}  hash={c['_config_hash']}")

# Export to YAML
yaml_dir = BASE_DIR / "configs"
paths = ab.export(yaml_dir)
print(f"\nExported {len(paths)} YAML files to {yaml_dir}/")
for p in paths:
    print(f"  {p.name}")

# ---------------------------------------------------------------------------
# Step 2: Simulated training runs (multi-seed)
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("Step 2: Simulated training runs")
print("=" * 60)

experiment_names = []

for cfg in configs:
    lr = cfg["lr"]
    sh = cfg["sh"]
    config_hash = cfg["_config_hash"]

    for seed in SEEDS:
        random.seed(seed * 1000 + int(lr * 1e6) + sh)
        exp_name = f"demo-lr_{lr}-sh_{sh}_seed{seed}"
        experiment_names.append(exp_name)

        print(f"\nRunning: {exp_name}  (lr={lr:.0e}, sh={sh}, seed={seed})")

        with Logger(
            exp_name,
            project=PROJECT,
            config=config,
            snapshot=False,
            seed=seed,
            config_hash=config_hash,
        ) as log:
            for step in range(STEPS_PER_RUN):
                # Higher lr → faster initial drop but noisier
                decay_rate = 40 + lr * 2e4
                noise_scale = 0.02 + lr * 10
                # Higher sh → better final quality
                quality_bonus = sh * 0.8

                loss = 2.0 * math.exp(-step / decay_rate) + random.gauss(0, noise_scale)
                psnr = (
                    20.0
                    + quality_bonus
                    + 8.0 * (1 - math.exp(-step / 60))
                    + random.gauss(0, 0.2)
                )
                ssim = min(
                    0.99,
                    0.85 + 0.1 * (1 - math.exp(-step / 70)) + random.gauss(0, 0.005),
                )
                lpips = max(
                    0.01,
                    0.2 * math.exp(-step / 50) + 0.05 + random.gauss(0, 0.01),
                )

                log.step(step, loss=loss, psnr=psnr, ssim=ssim, lpips=lpips)

            # -----------------------------------------------------------------
            # Step 3: Per-scene evaluation
            # -----------------------------------------------------------------
            for scene in SCENES:
                scene_seed = hash(f"{exp_name}-{scene}") % 10000
                rng = random.Random(scene_seed)

                scene_psnr = psnr + rng.gauss(0, 2.0)
                scene_ssim = max(0.0, min(1.0, ssim + rng.gauss(0, 0.02)))
                scene_lpips = max(0.0, lpips + rng.gauss(0, 0.02))

                log.scene(scene, psnr=round(scene_psnr, 2), ssim=round(scene_ssim, 4), lpips=round(scene_lpips, 4))

        print(f"  → {log.experiment_dir}")

# ---------------------------------------------------------------------------
# Step 4: Verify output files
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("Step 3: Verify output")
print("=" * 60)

project_dir = BASE_DIR / PROJECT
exp_dirs = sorted(project_dir.iterdir()) if project_dir.exists() else []
print(f"\nExperiments in {project_dir}/:")
for d in exp_dirs:
    files = [f.name for f in d.iterdir()]
    print(f"  {d.name}/  →  {', '.join(sorted(files))}")

# Check a specific experiment
sample = exp_dirs[0] if exp_dirs else None
if sample:
    metrics_lines = len((sample / "metrics.jsonl").read_text().strip().splitlines())
    scenes_lines = len((sample / "scenes.jsonl").read_text().strip().splitlines())
    print(f"\nSample ({sample.name}):")
    print(f"  metrics.jsonl: {metrics_lines} records")
    print(f"  scenes.jsonl:  {scenes_lines} scene records")

    import json
    meta = json.loads((sample / "meta.json").read_text())
    print(f"  meta.json: status={meta.get('status')}, seed={meta.get('seed')}, config_hash={meta.get('config_hash', 'N/A')[:8]}...")

# ---------------------------------------------------------------------------
# Print CLI commands to try
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("Step 4: Try these CLI commands")
print("=" * 60)

dir_flag = f"--dir {BASE_DIR}"

print(f"""
# List all experiments
nuviz ls {dir_flag} --project {PROJECT}

# Leaderboard (table / LaTeX / markdown)
nuviz leaderboard {dir_flag} --project {PROJECT} --sort psnr
nuviz leaderboard {dir_flag} --project {PROJECT} --sort psnr --format latex
nuviz leaderboard {dir_flag} --project {PROJECT} --sort psnr --format markdown

# Per-scene breakdown
nuviz breakdown {experiment_names[0]} {dir_flag} --project {PROJECT}
nuviz breakdown {experiment_names[0]} {dir_flag} --project {PROJECT} --latex

# Diff two experiments
nuviz breakdown {experiment_names[0]} --diff {experiment_names[2]} {dir_flag} --project {PROJECT}

# Export raw data
nuviz export {experiment_names[0]} {dir_flag} --project {PROJECT} --format csv
nuviz export {experiment_names[0]} {dir_flag} --project {PROJECT} --format json

# Ablation matrix
nuviz matrix --rows lr --cols sh --metric psnr {dir_flag} --project {PROJECT}
nuviz matrix --rows lr --cols sh --metric psnr {dir_flag} --project {PROJECT} --format latex

# Compare TUI (interactive — press q to quit, m to cycle metrics)
nuviz compare {experiment_names[0]} {experiment_names[2]} {dir_flag}
""")

print("Demo complete!")
print(f"Data lives at: {BASE_DIR}")
print("Delete with: rm -rf", BASE_DIR)
