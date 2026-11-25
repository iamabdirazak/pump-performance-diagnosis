"""Learn the unit from its healthy month, then record what healthy looks like.

1. Fit the correction on the first three weeks; pick its degree on the last.
2. Run the health estimate over the held-out week to get each parameter's
   healthy centre and spread, kept as the record of what healthy looks like.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.residuals import NAMES, compute
from src.models.correction import Correction
from src.models.health import PARAMETERS, estimate

TRAIN_DAYS = 21
TOLERANCE = 0.02  # a higher degree must cut validation error by more than 2%


def rms(residuals: pd.DataFrame) -> float:
    """Mean of each residual's RMS, each scaled by its datasheet-only RMS so units don't matter."""
    return float(np.mean([np.sqrt((residuals[name] ** 2).mean()) for name in NAMES]))


def main() -> None:
    healthy = pd.read_parquet("data/healthy.parquet")
    split = TRAIN_DAYS * 24 * 60
    train, validate = healthy.iloc[:split], healthy.iloc[split:]
    train_residuals = compute(train)

    scale = compute(validate)[list(NAMES)].pow(2).mean().pow(0.5)
    best, scores = None, {}
    for degree in range(1, 5):
        candidate = Correction.fit(train_residuals, train["flow"], train["speed"], degree)
        scores[degree] = float((compute(validate, candidate)[list(NAMES)].pow(2).mean().pow(0.5) / scale).mean())
        if best is None or scores[degree] < scores[best.degree] * (1 - TOLERANCE):
            best = candidate
    correction = Correction.fit(train_residuals, train["flow"], train["speed"], best.degree)

    health = estimate(validate, compute(validate, correction)).dropna()
    baseline = {name: {"centre": float(health[name].mean()), "spread": float(health[name].std())}
                for name in PARAMETERS}

    directory = Path("models")
    directory.mkdir(exist_ok=True)
    correction.save(directory / "correction.json")
    (directory / "baseline.json").write_text(json.dumps(baseline, indent=2))

    print("validation error vs datasheet:", {d: round(s, 3) for d, s in scores.items()}, "-> degree", best.degree)
    for name, values in baseline.items():
        print(f"{name:<10} centre {values['centre']:.4f}  spread {values['spread']:.4f}")


if __name__ == "__main__":
    main()
