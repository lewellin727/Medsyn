"""Project-compatible entrypoint for the GReaT baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baselines.common import add_dataset_args, apply_dataset_overrides


ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_log(config: dict[str, Any], record: dict[str, Any]) -> None:
    log_path = Path(config["paths"]["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def configure_device(device: str) -> str:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    if device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        return "cuda"
    return device


def import_great():
    sys.path.insert(0, str(BASELINE_ROOT))
    from be_great import GReaT

    return GReaT


def load_training_data(config: dict[str, Any]) -> pd.DataFrame:
    schema = config["schema"]
    train_path = Path(config["paths"]["train_csv"])
    df = pd.read_csv(train_path)
    columns = schema["numerical_columns"] + schema["categorical_columns"] + [schema["label_column"]]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{train_path} is missing required columns: {missing}")
    if df[columns].isna().any().any():
        raise ValueError(f"GReaT input contains missing values: {train_path}")
    return df[columns].copy()


def cast_generated_columns(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    schema = config["schema"]
    columns = schema["numerical_columns"] + schema["categorical_columns"] + [schema["label_column"]]
    for column in schema["numerical_columns"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in schema["categorical_columns"] + [schema["label_column"]]:
        df[column] = pd.to_numeric(df[column], errors="coerce").round().astype("Int64")
    return df[columns]


def build_model(config: dict[str, Any]):
    GReaT = import_great()
    great = config["great"]
    model_kwargs: dict[str, Any] = {
        "llm": great["llm"],
        "experiment_dir": config["paths"]["experiment_dir"],
        "epochs": int(great["epochs"]),
        "batch_size": int(great["batch_size"]),
        "save_steps": int(great["save_steps"]),
    }
    if great.get("float_precision") is not None:
        model_kwargs["float_precision"] = int(great["float_precision"])
    return GReaT(**model_kwargs)


def train_and_sample(config: dict[str, Any]) -> Path:
    great = config["great"]
    seed = int(great["seed"])
    random.seed(seed)
    np.random.seed(seed)

    sample_device = configure_device(str(great["device"]))
    train_df = load_training_data(config)
    model = build_model(config)

    write_log(
        config,
        {
            "event": "start",
            "baseline": config["baseline"],
            "dataset": config["dataset"],
            "split_name": config["split_name"],
            "train_rows": len(train_df),
            "llm": great["llm"],
            "device": great["device"],
            "epochs": int(great["epochs"]),
            "timestamp": time.time(),
        },
    )

    started = time.time()
    model.fit(train_df)
    train_seconds = time.time() - started

    sample_df = model.sample(
        n_samples=int(great["sample"]["num_samples"]),
        temperature=float(great["temperature"]),
        k=int(great["k"]),
        max_length=int(great["max_length"]),
        drop_nan=bool(great["drop_nan"]),
        device=sample_device,
        guided_sampling=bool(great["guided_sampling"]),
        random_feature_order=bool(great["random_feature_order"]),
    )
    sample_df = cast_generated_columns(sample_df, config)

    result_path = Path(config["paths"]["result_csv"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(result_path, index=False)

    model_dir = Path(config["paths"]["model_dir"])
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))

    write_log(
        config,
        {
            "event": "complete",
            "result_csv": str(result_path),
            "model_dir": str(model_dir),
            "train_seconds": train_seconds,
            "num_samples": len(sample_df),
            "timestamp": time.time(),
        },
    )
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("baselines/great/config_baseline.yaml"))
    add_dataset_args(parser)
    parser.add_argument("--llm")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--guided-sampling", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_dataset_overrides(config, "great", args)
    if args.llm:
        config["great"]["llm"] = args.llm
    if args.epochs is not None:
        config["great"]["epochs"] = args.epochs
    if args.num_samples is not None:
        config["great"]["sample"]["num_samples"] = args.num_samples
    if args.device:
        config["great"]["device"] = args.device
    if args.seed is not None:
        config["great"]["seed"] = args.seed
    if args.max_length is not None:
        config["great"]["max_length"] = args.max_length
    if args.temperature is not None:
        config["great"]["temperature"] = args.temperature
    if args.guided_sampling:
        config["great"]["guided_sampling"] = True

    result_path = train_and_sample(config)
    print(f"wrote generated samples: {result_path}")


if __name__ == "__main__":
    main()
