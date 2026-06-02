from .a4 import A4Model
from .base import ModelAdapter


def get_model_adapter(name: str) -> ModelAdapter:
    key = name.strip().lower()
    if key == "a4":
        return A4Model()
    raise ValueError(f"Unsupported model '{name}'. Add a new adapter in nbe/models/.")

