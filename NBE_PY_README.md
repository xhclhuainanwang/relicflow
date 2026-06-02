# nBE Python Mini-Framework

This project is a modular Python reconstruction of the Mathematica nBE workflow.

## Run (No MMA dependency, recommended)

```bash
python run_nbe.py --config configs/A4/parameters1_settings.json --out models/A4/nbe_from_config_parameters1.json
```

Optional full-channel run for A4:

```bash
python run_nbe.py --config configs/A4/parameters1_settings.json --with-wwzz --out models/A4/nbe_from_config_parameters1_full.json
```

Legacy `.wl` mode (compatible, not required):

```bash
python run_nbe.py --model A4 --model-dir models/A4 --param parameters1.wl --settings settings.wl --out models/A4/nbe_modular_parameters1.json
```

## Structure

- `nbe/wl_cards.py`: minimal `.wl` card parser
- `nbe/dof.py`: dof table + `H(T), s(T), gtilde(T)`
- `nbe/thermal.py`: thermal average integration + adaptive log-log tabulation + interpolation
- `nbe/solver.py`: adaptive nBE solver (`Euler/Trapezoid` control)
- `nbe/models/`: model adapters
  - `a4.py`: first adapter (channels: NN/WW/ZZ, `chanZpZp=1` intentionally excluded)
- `nbe/pipeline.py`: end-to-end workflow + JSON output
- `run_nbe.py`: generic CLI
- `configs/A4/parameters1_settings.json`: native config example
- `configs/A4/parameters2_settings.json`: native config example

## Add a New Model

1. Create `nbe/models/<your_model>.py`.
2. Implement adapter with:
   - `prepare_params(cards) -> dict`
   - `build_channels(params) -> list[ThermalChannel]`
3. Register it in `nbe/models/__init__.py::get_model_adapter`.
4. Run with `--model <YourModelName>`.

Only the model adapter should change; solver and thermal pipeline stay shared.


## Split Params/Settings (recommended for your workflow)

```bash
python run_nbe.py --params-file configs/A4/parameters1_params.json --settings-file configs/A4/parameters1_settings.json --out out.json
```

- `parameters1_params.json`: physical/numerical model parameters
- `parameters1_settings.json`: solver/tabulation settings
