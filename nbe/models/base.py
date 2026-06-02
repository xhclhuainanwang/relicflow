from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass
class ThermalChannel:
    name: str
    s_min: float
    sigma_s: Callable[[float], float]


class ModelAdapter(Protocol):
    name: str

    def prepare_params(self, cards: dict) -> dict:
        ...

    def build_channels(self, params: dict) -> list[ThermalChannel]:
        ...

