"""Score the advisories the way an operator would.

    lead time          days between the advisory and the fault reaching
                       maintenance severity (severity 1)
    true health        what the parameter really was when the advisory came
    false advisories   advisories on healthy data, per machine-month

The learned unit and the datasheet alone are scored side by side.
"""

import json
from pathlib import Path

import pandas as pd

from src.features.residuals import compute
from src.models.correction import Correction
from src.models.health import estimate
from src.monitoring.advisory import RULES, advise

FAULT_RUNS = ("wear", "blockage", "clogging")
TRUTH = {"wear": lambda s: 1 - 0.12 * s, "blockage": lambda s: 1 + s, "clogging": lambda s: 1 + 17 * s}
HOURS_PER_MONTH = 30 * 24


def run(name: str, correction: Correction | None):
    frame = pd.read_parquet(f"data/{name}.parquet")
    return frame, advise(estimate(frame, compute(frame, correction)))


def score(correction: Correction | None) -> dict:
    result = {}
    for name in FAULT_RUNS:
        frame, advice = run(name, correction)
        raised = advice.index[advice[name]]
        healthy_hours = advice.index[frame.loc[advice.index, "severity"] == 0]
        wrong = advice.drop(columns=name).loc[healthy_hours].any(axis=1).sum() + advice.loc[healthy_hours, name].sum()
        if len(raised) == 0:
            result[name] = {"lead_days": None}
            continue
        first = raised[0]
        due = frame["time"].iloc[-1]
        severity = float(frame["severity"].iloc[first])
        result[name] = {
            "lead_days": round((due - frame["time"].iloc[first]).total_seconds() / 86400, 1),
            "true_health": round(float(TRUTH[name](severity)), 3),
            "severity": round(severity, 2),
            "other_causes_raised": [c for c in advice.columns if c != name and advice[c].any()],
            "false_hours_before_onset": int(wrong),
        }

    frame, advice = run("holdout", correction)
    onsets = int((advice & ~advice.shift(fill_value=False)).sum().sum())
    result["holdout"] = {"months": round(len(advice) / HOURS_PER_MONTH, 2),
                         "false_advisories_per_month": round(onsets / (len(advice) / HOURS_PER_MONTH), 2)}
    return result


if __name__ == "__main__":
    results = {"learned": score(Correction.load(Path("models/correction.json"))), "datasheet": score(None)}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
