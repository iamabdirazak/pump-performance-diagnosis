"""Write the per-minute path as one ONNX graph: physics, learned unit, residuals.

The graph takes the seven sensor readings and returns, for each minute,
the four residuals and the terms that the health estimate sums over its
window. The window sums and the advisory rules stay in the host loop:
they are state, a few additions per minute, and need no graph.

Every operation here is arithmetic, so the graph is written directly with
ONNX operators through a small expression builder, rather than through a
deep-learning framework.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from src.models.correction import Correction
from src.plant.pump import ATMOSPHERE, DENSITY, GRAVITY, Pump, System
from src.plant.sensors import PIPE_AREA

INPUTS = ("level", "temperature", "suction_pressure", "discharge_pressure", "flow", "speed", "power")
RESIDUALS = ("head", "power", "suction", "discharge")
TERMS = ("head_fit", "head_norm", "power_fit", "power_norm", "discharge_fit", "suction_fit", "flow_norm", "margin")


class Graph:
    def __init__(self):
        self.nodes, self.initializers, self.count = [], [], 0

    def name(self, prefix: str) -> str:
        self.count += 1
        return f"{prefix}_{self.count}"

    def constant(self, value) -> "Expression":
        name = self.name("c")
        self.initializers.append(numpy_helper.from_array(np.array(value, dtype=np.float32), name))
        return Expression(self, name)

    def op(self, kind: str, *inputs, **attributes) -> "Expression":
        output = self.name(kind.lower())
        self.nodes.append(helper.make_node(kind, [i.output for i in inputs], [output], **attributes))
        return Expression(self, output)


class Expression:
    """A value in the graph. Arithmetic on it adds ONNX nodes."""

    def __init__(self, graph: Graph, output: str):
        self.graph, self.output = graph, output

    def wrap(self, other):
        return other if isinstance(other, Expression) else self.graph.constant(other)

    def __add__(self, other): return self.graph.op("Add", self, self.wrap(other))
    def __radd__(self, other): return self.graph.op("Add", self.wrap(other), self)
    def __sub__(self, other): return self.graph.op("Sub", self, self.wrap(other))
    def __rsub__(self, other): return self.graph.op("Sub", self.wrap(other), self)
    def __mul__(self, other): return self.graph.op("Mul", self, self.wrap(other))
    def __rmul__(self, other): return self.graph.op("Mul", self.wrap(other), self)
    def __truediv__(self, other): return self.graph.op("Div", self, self.wrap(other))
    def __rtruediv__(self, other): return self.graph.op("Div", self.wrap(other), self)
    def __rpow__(self, base): return self.graph.op("Pow", self.wrap(base), self)

    def __pow__(self, exponent: int):
        result = self
        for _ in range(exponent - 1):
            result = result * self
        return result

    def at_least(self, floor: float): return self.graph.op("Max", self, self.wrap(floor))


def build(correction: Correction, pump: Pump = Pump(), system: System = System()) -> onnx.ModelProto:
    graph = Graph()
    readings = Expression(graph, "readings")
    column = {name: graph.op("Gather", readings, _index(graph, i), axis=1)
              for i, name in enumerate(INPUTS)}
    level, temperature = column["level"], column["temperature"]
    suction_kpa, discharge_kpa = column["suction_pressure"], column["discharge_pressure"]
    flow, speed, power = column["flow"], column["speed"], column["power"]
    rho_g = DENSITY * GRAVITY

    # Datasheet physics, as in src/plant/pump.py
    r = speed / pump.speed_rated
    x = flow / (r * pump.flow_bep)
    drop = (pump.shutoff_ratio - 1) * pump.head_bep / pump.flow_bep**2
    head_e = pump.shutoff_ratio * pump.head_bep * r**2 - drop * flow**2
    efficiency = pump.efficiency_bep * (2 * x - x**2).at_least(1e-3)
    power_e = rho_g / 3_600_000 * flow * head_e / (efficiency * pump.drive_efficiency)
    velocity_head = (flow / (3600 * PIPE_AREA)) ** 2 / (2 * GRAVITY)
    suction_e = level - system.friction_suction * flow**2 - velocity_head
    discharge_e = system.level_reservoir + system.friction_discharge * flow**2 - velocity_head

    # The learned unit, as in src/models/correction.py
    c = correction.coefficients
    head_e = head_e + r**2 * _polynomial(c["head"], x)
    power_e = power_e + r**3 * _polynomial(c["power"], x)
    q = (flow / pump.flow_bep) ** 2
    suction_e = suction_e + c["suction"][0] + c["suction"][1] * q
    discharge_e = discharge_e + c["discharge"][0] + c["discharge"][1] * q

    # Residuals, as in src/features/residuals.py
    head_r = (discharge_kpa - suction_kpa) * (1000 / rho_g) - head_e
    power_r = power - power_e
    suction_r = suction_kpa * (1000 / rho_g) - suction_e
    discharge_r = discharge_kpa * (1000 / rho_g) - discharge_e

    # Per-minute terms of the window fits, as in src/models/health.py
    q2 = flow**2
    vapour = 133.322 * 10 ** (8.07131 - 1730.63 / (temperature + 233.426))
    margin = (ATMOSPHERE + suction_kpa * 1000 - vapour) / rho_g + velocity_head \
        - pump.npsh_required_bep * r**2 * x**2
    terms = [head_r * head_e, head_e**2, power_r * power_e, power_e**2,
             discharge_r * q2, suction_r * q2, q2**2, margin]

    def stack(values, name):
        columns = [graph.op("Unsqueeze", v, _axes(graph)) for v in values]
        node = helper.make_node("Concat", [col.output for col in columns], [name], axis=1)
        graph.nodes.append(node)

    stack([head_r, power_r, suction_r, discharge_r], "residuals")
    stack(terms, "terms")

    model = helper.make_model(helper.make_graph(
        graph.nodes, "pump_performance_diagnosis",
        [helper.make_tensor_value_info("readings", TensorProto.FLOAT, ["minutes", len(INPUTS)])],
        [helper.make_tensor_value_info("residuals", TensorProto.FLOAT, ["minutes", len(RESIDUALS)]),
         helper.make_tensor_value_info("terms", TensorProto.FLOAT, ["minutes", len(TERMS)])],
        graph.initializers,
    ), opset_imports=[helper.make_opsetid("", 17)], producer_name="pump-performance-diagnosis", ir_version=8)
    model.doc_string = "Seven sensor readings per minute in; residuals and health-fit terms out."
    onnx.checker.check_model(model)
    return model


def _polynomial(coefficients, x):
    return coefficients[0] + sum(coefficient * x**i for i, coefficient in enumerate(coefficients) if i)


def _index(graph: Graph, i: int) -> Expression:
    name = graph.name("index")
    graph.initializers.append(numpy_helper.from_array(np.array(i, dtype=np.int64), name))
    return Expression(graph, name)


def _axes(graph: Graph) -> Expression:
    name = graph.name("axes")
    graph.initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name))
    return Expression(graph, name)


if __name__ == "__main__":
    model = build(Correction.load(Path("models/correction.json")))
    path = Path("models/pump.onnx")
    onnx.save(model, path)
    print(f"{path}  {path.stat().st_size / 1024:.1f} KB  {len(model.graph.node)} operators")
