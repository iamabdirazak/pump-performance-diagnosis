"""Centrifugal pump and its pipe system, from first principles.

The pump is described by its best efficiency point (BEP) at rated speed.
Similitude carries that point to any speed (Q ~ n, H ~ n^2, P ~ n^3), and
energy conservation along the pipe gives the system curve. Faults later
enter as changes to these parameters, never as new equations.
"""

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81  # m/s^2
DENSITY = 998.2  # kg/m^3, water at 20 C
ATMOSPHERE = 101_325.0  # Pa


def vapour_pressure(temperature: float) -> float:
    """Saturation pressure of water in Pa (Antoine equation, 1-100 C)."""
    return 133.322 * 10 ** (8.07131 - 1730.63 / (233.426 + temperature))


@dataclass(frozen=True)
class Pump:
    """End suction 3x2-10 at 1750 rpm, defined by its BEP."""

    speed_rated: float = 1750.0  # rpm
    flow_bep: float = 57.0  # m^3/h
    head_bep: float = 18.9  # m
    efficiency_bep: float = 0.68
    shutoff_ratio: float = 1.15  # shut-off head / BEP head
    npsh_required_bep: float = 1.8  # m
    drive_efficiency: float = 0.90  # motor and VFD together
    # How this particular unit differs from its datasheet. Zero for the datasheet pump.
    head_error: tuple[float, float, float] = (0.0, 0.0, 0.0)  # fractional, quadratic in flow ratio
    efficiency_error: float = 0.0

    def flow_ratio(self, flow, speed):
        """Flow relative to the BEP flow at this speed (the flow coefficient ratio)."""
        return np.asarray(flow) / (self.flow_bep * speed / self.speed_rated)

    def head(self, flow, speed):
        """Head in m. A parabola through shut-off and the BEP, scaled by speed."""
        ratio = speed / self.speed_rated
        drop = (self.shutoff_ratio - 1) * self.head_bep / self.flow_bep**2
        datasheet = self.shutoff_ratio * self.head_bep * ratio**2 - drop * np.square(flow)
        a, b, c = self.head_error
        x = self.flow_ratio(flow, speed) - 1
        return datasheet * (1 + a + b * x + c * x**2)

    def efficiency(self, flow, speed):
        """Hydraulic efficiency, peaking at the BEP flow for this speed."""
        x = self.flow_ratio(flow, speed)
        return (self.efficiency_bep + self.efficiency_error) * np.clip(2 * x - x**2, 0.0, None)

    def npsh_required(self, flow, speed):
        """Suction head the pump needs to avoid cavitation, in m."""
        x = self.flow_ratio(flow, speed)
        return self.npsh_required_bep * (speed / self.speed_rated) ** 2 * x**2


@dataclass(frozen=True)
class System:
    """Pipe from the suction tank to the reservoir. Elevations from the pump centreline."""

    level_suction: float = 2.0  # m
    level_reservoir: float = 14.0  # m
    friction_suction: float = 0.5 / 57.0**2  # m per (m^3/h)^2, tank to pump
    friction_discharge: float = 6.4 / 57.0**2  # m per (m^3/h)^2, pump to reservoir

    @property
    def static_head(self) -> float:
        return self.level_reservoir - self.level_suction

    @property
    def friction(self) -> float:
        return self.friction_suction + self.friction_discharge

    def head(self, flow):
        """Head the pipe demands at a flow: lift plus friction (Darcy-Weisbach)."""
        return self.static_head + self.friction * np.square(flow)

    def npsh_available(self, flow, temperature: float = 20.0):
        """Absolute suction head above vapour pressure at the pump inlet, in m."""
        pressure = (ATMOSPHERE - vapour_pressure(temperature)) / (DENSITY * GRAVITY)
        return pressure + self.level_suction - self.friction_suction * np.square(flow)


@dataclass(frozen=True)
class OperatingPoint:
    flow: float  # m^3/h
    head: float  # m
    efficiency: float
    power_hydraulic: float  # W
    power_electrical: float  # W
    npsh_available: float  # m
    npsh_required: float  # m


def operating_point(pump: Pump, system: System, speed: float, temperature: float = 20.0) -> OperatingPoint:
    """Where the pump curve meets the system curve.

    For the datasheet pump both curves are parabolas in flow, so the crossing
    has a closed form. A real unit's curve differs slightly, so a few Newton
    steps from that answer finish the job.
    """
    ratio = speed / pump.speed_rated
    drop = (pump.shutoff_ratio - 1) * pump.head_bep / pump.flow_bep**2
    surplus = pump.shutoff_ratio * pump.head_bep * ratio**2 - system.static_head
    if surplus <= 0:
        raise ValueError(f"At {speed:.0f} rpm the pump cannot lift {system.static_head:.1f} m")
    flow = float(np.sqrt(surplus / (system.friction + drop)))
    for _ in range(4):
        gap = pump.head(flow, speed) - system.head(flow)
        slope = (pump.head(flow + 1e-3, speed) - system.head(flow + 1e-3) - gap) / 1e-3
        flow -= float(gap / slope)
    head = float(system.head(flow))
    efficiency = float(pump.efficiency(flow, speed))
    power_hydraulic = DENSITY * GRAVITY * (flow / 3600) * head
    return OperatingPoint(
        flow=flow,
        head=head,
        efficiency=efficiency,
        power_hydraulic=power_hydraulic,
        power_electrical=power_hydraulic / (efficiency * pump.drive_efficiency),
        npsh_available=float(system.npsh_available(flow, temperature)),
        npsh_required=float(pump.npsh_required(flow, speed)),
    )
