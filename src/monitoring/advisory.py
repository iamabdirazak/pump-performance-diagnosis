"""From health to advice: one rule per fault, in engineering units.

A rule raises its advisory when its parameter has crossed the threshold
for a full on-delay, and clears it only once the parameter is back past a
deadband. So a value hovering at the threshold cannot make the advisory
flicker, and a single bad hour cannot raise one.

Thresholds are maintenance criteria, not statistics: they say when the
work is worth doing, whatever the noise happens to be.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Rule:
    cause: str
    parameter: str
    threshold: float
    clear: float  # the deadband edge the parameter must return past
    rising: bool  # True if the parameter grows as the fault grows
    advice: str

    def exceeded(self, value: float) -> bool:
        return value > self.threshold if self.rising else value < self.threshold

    def cleared(self, value: float) -> bool:
        return value < self.clear if self.rising else value > self.clear


RULES = (
    Rule("wear", "head", 0.97, 0.98, False, "Impeller wear: head down 3%. Plan wear-ring inspection."),
    Rule("blockage", "discharge", 1.30, 1.20, True, "Discharge blockage: line friction up 30%. Check valves and pipe."),
    Rule("clogging", "suction", 3.00, 2.50, True, "Suction clogging: strainer loss tripled. Clean the strainer."),
    Rule("cavitation", "margin", 3.00, 3.50, False, "Cavitation risk: NPSH margin below 3 m. Reduce speed."),
)

ON_DELAY = 6  # hours the threshold must stay crossed


class Latch:
    """One rule's state, updated once an hour."""

    def __init__(self, rule: Rule):
        self.rule, self.active, self.count = rule, False, 0

    def update(self, value: float) -> bool:
        if pd.isna(value):
            return self.active
        if self.active:
            self.active = not self.rule.cleared(value)
        else:
            self.count = self.count + 1 if self.rule.exceeded(value) else 0
            self.active = self.count >= ON_DELAY
        return self.active


def advise(health: pd.DataFrame) -> pd.DataFrame:
    """Hourly advisory state for each rule. health is the minute-by-minute estimate."""
    hourly = health.iloc[59::60]
    states = {}
    for rule in RULES:
        latch = Latch(rule)
        states[rule.cause] = [latch.update(value) for value in hourly[rule.parameter]]
    return pd.DataFrame(states, index=hourly.index)
