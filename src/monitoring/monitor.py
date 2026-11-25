"""The loop that runs on site: one call per minute of sensor readings.

The ONNX graph turns the readings into residuals and fit terms. The monitor
keeps the last day of terms as running sums, turns them into health, and
updates the advisory rules once an hour.
"""

from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.deployment.export import INPUTS, TERMS
from src.models.health import MARGIN_WINDOW, WINDOW
from src.monitoring.advisory import RULES, Latch
from src.plant.pump import System


class Monitor:
    def __init__(self, model: Path = Path("models/pump.onnx"), system: System = System()):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.system = system
        self.day = deque(maxlen=WINDOW)
        self.hour = deque(maxlen=MARGIN_WINDOW)
        self.sums = np.zeros(len(TERMS) - 1)
        self.latches = [Latch(rule) for rule in RULES]
        self.minutes = 0
        self.health: dict[str, float] = {}
        self.advisories: list[str] = []

    def step(self, readings: dict[str, float]) -> None:
        """Take one minute of readings, keyed as in INPUTS (m, °C, kPa gauge, m³/h, rpm, kW)."""
        row = np.array([[readings[name] for name in INPUTS]], dtype=np.float32)
        _, terms = self.session.run(None, {"readings": row})
        fits, margin = terms[0, :-1].astype(float), float(terms[0, -1])
        if len(self.day) == WINDOW:
            self.sums -= self.day[0]
        self.day.append(fits)
        self.sums += fits
        self.hour.append(margin)
        self.minutes += 1

        if len(self.day) == WINDOW:
            head_fit, head_norm, power_fit, power_norm, discharge_fit, suction_fit, flow_norm = self.sums
            self.health = {
                "head": 1 + head_fit / head_norm,
                "power": 1 + power_fit / power_norm,
                "discharge": 1 + discharge_fit / flow_norm / self.system.friction_discharge,
                "suction": 1 - suction_fit / flow_norm / self.system.friction_suction,
                "margin": float(np.mean(self.hour)),
            }
            if self.minutes % 60 == 0:
                self.advisories = [latch.rule.advice for latch in self.latches
                                   if latch.update(self.health[latch.rule.parameter])]
