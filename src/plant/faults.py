"""The three faults, each a change to one physical parameter.

Severity runs from 0 (healthy) to 1 (due for maintenance). The equations
never change; only the numbers inside them do.
"""

from dataclasses import replace

from src.plant.pump import Pump, System

FAULTS = ("wear", "blockage", "clogging")


def apply(pump: Pump, system: System, fault: str | None, severity: float) -> tuple[Pump, System]:
    if fault is None or severity == 0:
        return pump, system
    if fault == "wear":
        # Worn wear rings leak flow back to suction: the impeller works harder for less head.
        return replace(pump, head_bep=pump.head_bep * (1 - 0.12 * severity),
                       efficiency_bep=pump.efficiency_bep - 0.12 * severity), system
    if fault == "blockage":
        # Fouled discharge line or a valve left partly shut: the pipe resists more.
        return pump, replace(system, friction_discharge=system.friction_discharge * (1 + 1.0 * severity))
    if fault == "clogging":
        # Clogging suction strainer: the inlet loses pressure and the cavitation margin shrinks.
        return pump, replace(system, friction_suction=system.friction_suction * (1 + 17.0 * severity))
    raise ValueError(f"Unknown fault: {fault}")
