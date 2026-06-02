import json
from pathlib import Path


def _strip_jsonc(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    in_str = False
    escaped = False
    in_line_comment = False
    in_block_comment = False

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append(ch)
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _normalize_data(data: dict) -> tuple[str | None, dict]:
    model = data.get("model")

    if "cards" in data and isinstance(data["cards"], dict):
        return model, dict(data["cards"])

    cards: dict = {}
    if "params" in data and isinstance(data["params"], dict):
        cards.update(data["params"])
    if "settings" in data and isinstance(data["settings"], dict):
        cards.update(data["settings"])
    if cards:
        return model, cards

    reserved = {"model"}
    cards = {k: v for k, v in data.items() if k not in reserved}
    return model, cards


def load_native_config(config_path: str | Path) -> tuple[str | None, dict]:
    p = Path(config_path)
    text = p.read_text(encoding="utf-8-sig")
    data = json.loads(_strip_jsonc(text))
    return _normalize_data(data)


def load_native_split(params_path: str | Path, settings_path: str | Path) -> tuple[str | None, dict]:
    model_p, cards_p = load_native_config(params_path)
    model_s, cards_s = load_native_config(settings_path)

    if model_p and model_s and str(model_p).strip().lower() != str(model_s).strip().lower():
        raise ValueError(f"Model mismatch between params ({model_p}) and settings ({model_s}).")

    model = model_p or model_s
    cards = {}
    cards.update(cards_p)
    cards.update(cards_s)
    return model, cards
