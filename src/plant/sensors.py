"""What the seven instruments read at an operating point.

Pressures are gauge readings at the pump centreline. Noise levels are
typical accuracies for industrial transmitters.
"""

import numpy as np

from src.plant.pump import DENSITY, GRAVITY, OperatingPoint, System

PIPE_AREA = np.pi * 0.1023**2 / 4  # m^2, DN100 schedule 40

NOISE = {  # one standard deviation
    "level": 0.01,  # m
    "temperature": 0.1,  # C
    "suction_pressure": 0.25,  # kPa
    "discharge_pressure": 0.5,  # kPa
    "flow": 0.3,  # m^3/h
    "speed": 2.0,  # rpm
    "power": 0.02,  # kW
}


def read(point: OperatingPoint, system: System, speed: float, temperature: float,
         rng: np.random.Generator | None = None, flow_gain: float = 1.0) -> dict[str, float]:
    """Readings at an operating point. flow_gain is the flowmeter's calibration error."""
    velocity = point.flow / 3600 / PIPE_AREA
    suction = DENSITY * GRAVITY * (system.level_suction - system.friction_suction * point.flow**2) \
        - 0.5 * DENSITY * velocity**2
    discharge = suction + DENSITY * GRAVITY * point.head  # same pipe size both sides
    clean = {
        "level": system.level_suction,
        "temperature": temperature,
        "suction_pressure": suction / 1000,
        "discharge_pressure": discharge / 1000,
        "flow": point.flow * flow_gain,
        "speed": speed,
        "power": point.power_electrical / 1000,
    }
    if rng is None:
        return clean
    return {name: value + rng.normal(0, NOISE[name]) for name, value in clean.items()}
