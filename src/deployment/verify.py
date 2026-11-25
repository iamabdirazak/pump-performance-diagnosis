"""Check the ONNX graph against the Python pipeline, then time it on one CPU thread.

Equivalence: on every run, the graph's residuals and health estimates must
match the reference implementation. Timing: one minute of readings at a
time, as the edge device would call it, against a 50 ms budget.
"""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.deployment.export import INPUTS, RESIDUALS, TERMS
from src.features.residuals import compute
from src.models.correction import Correction
from src.models.health import MARGIN_WINDOW, WINDOW, estimate
from src.plant.pump import System

BUDGET_MS = 50.0


def session() -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession("models/pump.onnx", options, providers=["CPUExecutionProvider"])


def health_from_terms(terms: pd.DataFrame, system: System = System()) -> pd.DataFrame:
    """What the host loop does with the graph's output: rolling sums and one division each."""
    total = terms.rolling(WINDOW, min_periods=WINDOW).sum()
    return pd.DataFrame({
        "head": 1 + total["head_fit"] / total["head_norm"],
        "power": 1 + total["power_fit"] / total["power_norm"],
        "discharge": 1 + total["discharge_fit"] / total["flow_norm"] / system.friction_discharge,
        "suction": 1 - total["suction_fit"] / total["flow_norm"] / system.friction_suction,
        "margin": terms["margin"].rolling(MARGIN_WINDOW, min_periods=MARGIN_WINDOW).mean(),
    })


def main() -> None:
    model = session()
    correction = Correction.load(Path("models/correction.json"))
    worst = {"residuals": 0.0, "health": 0.0}
    for run in ("healthy", "wear", "blockage", "clogging"):
        frame = pd.read_parquet(f"data/{run}.parquet")
        residuals, terms = model.run(None, {"readings": frame[list(INPUTS)].to_numpy(np.float32)})
        reference = compute(frame, correction)
        worst["residuals"] = max(worst["residuals"], float(np.abs(residuals - reference[list(RESIDUALS)].to_numpy()).max()))
        health = health_from_terms(pd.DataFrame(terms, columns=TERMS, dtype=np.float64)).dropna()
        expected = estimate(frame, reference).loc[health.index]
        worst["health"] = max(worst["health"], float((health - expected).abs().to_numpy().max()))

    one_minute = frame[list(INPUTS)].to_numpy(np.float32)[:1]
    for _ in range(200):
        model.run(None, {"readings": one_minute})
    timings = []
    for _ in range(5000):
        start = time.perf_counter()
        model.run(None, {"readings": one_minute})
        timings.append((time.perf_counter() - start) * 1000)

    edge = {
        "size_kb": round(Path("models/pump.onnx").stat().st_size / 1024, 1),
        "latency_ms_median": round(float(np.median(timings)), 3),
        "latency_ms_p99": round(float(np.percentile(timings, 99)), 3),
        "budget_ms": BUDGET_MS,
        "max_residual_difference": float(f"{worst['residuals']:.2e}"),
        "max_health_difference": float(f"{worst['health']:.2e}"),
    }
    results = json.loads(Path("reports/results.json").read_text())
    results["edge"] = edge
    Path("reports/results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(edge, indent=2))


if __name__ == "__main__":
    main()
