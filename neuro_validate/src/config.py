from __future__ import annotations

from pathlib import Path

from .schema import BenchmarkConfig


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    payload = _parse_simple_yaml(config_path.read_text(encoding="utf-8"))
    frequency_bands = {
        key: tuple(float(value) for value in band_values)
        for key, band_values in payload["frequency_bands"].items()
    }
    return BenchmarkConfig(
        benchmark_name=str(payload["benchmark_name"]),
        dataset_root=Path(str(payload["dataset_root"])),
        task_name=str(payload["task_name"]),
        window_length_s=float(payload["window_length_s"]),
        window_step_s=float(payload["window_step_s"]),
        sample_rate_hz=int(payload["sample_rate_hz"]),
        grouped_cv_folds=int(payload["grouped_cv_folds"]),
        model_family=str(payload["model_family"]),
        output_dir=Path(str(payload["output_dir"])),
        frequency_bands=frequency_bands,
        metrics=tuple(str(metric) for metric in payload["metrics"]),
    )


def _parse_simple_yaml(text: str) -> dict:
    """Parse the limited YAML subset used by the benchmark configs.

    This avoids introducing a hard PyYAML dependency in the scaffold.
    """

    data: dict[str, object] = {}
    current_key: str | None = None
    nested_dict_key: str | None = None
    nested_list_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            nested_dict_key = None
            nested_list_key = None
            key, value = [part.strip() for part in line.split(":", 1)]
            if value == "":
                data[key] = {}
                current_key = key
                nested_dict_key = key
                continue
            data[key] = _parse_scalar(value)
            current_key = key
            continue

        stripped = line.strip()
        if stripped.startswith("- "):
            if current_key is None:
                raise ValueError("list item encountered before parent key")
            if not isinstance(data.get(current_key), list):
                data[current_key] = []
            assert isinstance(data[current_key], list)
            data[current_key].append(_parse_scalar(stripped[2:].strip()))
            nested_list_key = current_key
            continue

        if nested_dict_key is None:
            raise ValueError(f"unsupported YAML structure: {line}")
        subkey, subvalue = [part.strip() for part in stripped.split(":", 1)]
        assert isinstance(data[nested_dict_key], dict)
        data[nested_dict_key][subkey] = _parse_inline_list(subvalue)

    return data


def _parse_inline_list(value: str):
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    return _parse_scalar(value)


def _parse_scalar(value: str):
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value

