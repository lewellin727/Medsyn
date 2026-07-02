"""Project-compatible entrypoint for the TVAE baseline."""

from __future__ import annotations

import argparse
import json
import os
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


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_log(config: dict[str, Any], record: dict[str, Any]) -> None:
    log_path = Path(config["paths"]["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def configure_device(device: str) -> bool:
    if device == "cpu":
        return False
    if device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    return True


def import_tvae(config: dict[str, Any]):
    ctgan_source_root = ROOT / config["paths"]["ctgan_source_root"]
    sys.path.insert(0, str(ctgan_source_root))
    from ctgan import TVAE

    return TVAE


def load_training_data(config: dict[str, Any]) -> pd.DataFrame:
    schema = config["schema"]
    train_path = Path(config["paths"]["train_csv"])
    df = pd.read_csv(train_path)
    expected_columns = (
        schema["numerical_columns"] + schema["categorical_columns"] + [schema["label_column"]]
    )
    missing = sorted(set(expected_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {train_path}: {missing}")

    df = df[expected_columns].copy()
    for column in schema["numerical_columns"]:
        df[column] = df[column].astype(float)
    for column in schema["discrete_columns"]:
        df[column] = df[column].astype(int)

    if df.isna().any().any():
        raise ValueError(f"TVAE input contains missing values: {train_path}")
    return df


def build_model(config: dict[str, Any], enable_gpu: bool):
    TVAE = import_tvae(config)
    tvae = config["tvae"]
    model = TVAE(
        embedding_dim=int(tvae["embedding_dim"]),
        compress_dims=tuple(tvae["compress_dims"]),
        decompress_dims=tuple(tvae["decompress_dims"]),
        l2scale=float(tvae["l2scale"]),
        batch_size=int(tvae["batch_size"]),
        epochs=int(tvae["epochs"]),
        loss_factor=int(tvae["loss_factor"]),
        enable_gpu=enable_gpu,
        verbose=bool(tvae["verbose"]),
    )
    model.set_random_state(int(tvae["seed"]))
    return model


def save_loss_values(config: dict[str, Any], model: Any) -> Path | None:
    loss_values = getattr(model, "loss_values", None)
    if loss_values is None:
        return None
    loss_path = Path(config["paths"]["model_path"]).with_suffix(".loss.csv")
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    loss_values.to_csv(loss_path, index=False)
    return loss_path


def train_and_sample(config: dict[str, Any]) -> tuple[Path, Path | None]:
    seed = int(config["tvae"]["seed"])
    np.random.seed(seed)

    device = str(config["tvae"]["device"])
    enable_gpu = configure_device(device)
    train_df = load_training_data(config)
    model = build_model(config, enable_gpu=enable_gpu)

    write_log(
        config,
        {
            "event": "start",
            "baseline": config["baseline"],
            "dataset": config["dataset"],
            "split_name": config["split_name"],
            "train_rows": len(train_df),
            "device": device,
            "epochs": int(config["tvae"]["epochs"]),
            "timestamp": time.time(),
        },
    )

    started = time.time()
    model.fit(train_df, discrete_columns=list(config["schema"]["discrete_columns"]))
    train_seconds = time.time() - started

    result_path = Path(config["paths"]["result_csv"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df = model.sample(int(config["tvae"]["sample"]["num_samples"]))
    sample_df.to_csv(result_path, index=False)

    model_path = Path(config["paths"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    loss_path = save_loss_values(config, model)

    write_log(
        config,
        {
            "event": "complete",
            "result_csv": str(result_path),
            "model_path": str(model_path),
            "loss_csv": str(loss_path) if loss_path else None,
            "train_seconds": train_seconds,
            "num_samples": len(sample_df),
            "timestamp": time.time(),
        },
    )
    return result_path, loss_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("baselines/tvae/config_baseline.yaml"))
    add_dataset_args(parser)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_dataset_overrides(config, "tvae", args)
    if args.epochs is not None:
        config["tvae"]["epochs"] = args.epochs
    if args.num_samples is not None:
        config["tvae"]["sample"]["num_samples"] = args.num_samples
    if args.device:
        config["tvae"]["device"] = args.device
    if args.seed is not None:
        config["tvae"]["seed"] = args.seed

    result_path, loss_path = train_and_sample(config)
    print(f"wrote generated samples: {result_path}")
    if loss_path:
        print(f"wrote TVAE loss values: {loss_path}")


if __name__ == "__main__":
    main()
