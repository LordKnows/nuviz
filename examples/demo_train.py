"""Pseudo training loop to test nuviz logging → CLI pipeline."""

import math
import random
import time

from nuviz import Logger

random.seed(42)

with Logger("demo-phase1", project="nuviz-test") as log:
    for step in range(200):
        # Simulated loss: decaying with noise
        loss = 2.0 * math.exp(-step / 60) + random.gauss(0, 0.05)
        # Simulated PSNR: climbing with noise
        psnr = 20.0 + 10.0 * (1 - math.exp(-step / 80)) + random.gauss(0, 0.3)
        # Simulated learning rate: cosine decay
        lr = 1e-3 * (0.5 * (1 + math.cos(math.pi * step / 200)))

        log.step(step, loss=loss, psnr=psnr, lr=lr)

        # ~50 steps/sec to simulate real training
        time.sleep(0.02)

    print(f"Done. Data written to: {log.experiment_dir}")
