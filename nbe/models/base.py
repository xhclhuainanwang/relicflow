from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ThermalChannel:
    name: str
    s_min: float
    sigma_s: Callable[[float], float]
    initial_m1: float | None = None
    initial_m2: float | None = None
    breakpoints: tuple[float, ...] = ()
    thermal_average_x: Callable[[float], float] | None = None


@dataclass
class RelicFunctions:
    svx: Callable[[float], float]
    yeq: Callable[[float], float]
    channel_funcs: dict[str, Callable[[float], float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(Protocol):
    name: str

    def prepare_params(self, cards: dict) -> dict:
        ...

    def build_channels(self, params: dict) -> list[ThermalChannel]:
        ...

    def build_relic_functions(self, params: dict, settings: dict, dof: Any) -> RelicFunctions:
        ...
