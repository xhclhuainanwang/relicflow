from .base import ModelAdapter


def get_model_adapter(name: str) -> ModelAdapter:
    key = name.strip().lower()
    if key == "a4":
        from .a4 import A4Model

        return A4Model()
    if key in {"disp", "dispersion"}:
        from .dispersion import DispersionModel

        return DispersionModel()
    if key in {"v4", "dispersion_v4"}:
        from .v4 import V4Model

        return V4Model()
    if key in {"z4", "vll"}:
        from .z4 import Z4Model

        return Z4Model()
    raise ValueError(f"Unsupported model '{name}'. Add a new adapter in nbe/models/.")
