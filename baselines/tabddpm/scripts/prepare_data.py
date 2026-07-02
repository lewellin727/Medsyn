"""Convert Medsyn CSV splits to the NumPy layout expected by TabDDPM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_split(path: Path, numerical_columns: list[str], categorical_columns: list[str], label_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    required = numerical_columns + categorical_columns + [label_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    x_num = df[numerical_columns].to_numpy(dtype=np.float32)
    x_cat = df[categorical_columns].to_numpy()
    y = df[label_column].to_numpy(dtype=np.int64)
    return x_num, x_cat, y


def prepare_tabddpm_data(config: dict[str, Any]) -> Path:
    paths = config["paths"]
    schema = config["schema"]
    output_dir = Path(paths["tabddpm_data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    numerical_columns = schema["numerical_columns"]
    categorical_columns = schema["categorical_columns"]
    label_column = schema["label_column"]

    split_map = {
        "train": Path(paths["train_csv"]),
        "val": Path(paths["val_csv"]),
        "test": Path(paths["test_csv"]),
    }
    sizes = {}
    for split, csv_path in split_map.items():
        x_num, x_cat, y = _read_split(csv_path, numerical_columns, categorical_columns, label_column)
        np.save(output_dir / f"X_num_{split}.npy", x_num)
        np.save(output_dir / f"X_cat_{split}.npy", x_cat)
        np.save(output_dir / f"y_{split}.npy", y)
        sizes[f"{split}_size"] = int(len(y))

    info = {
        "name": f"medsyn_{config['dataset']}_{config['split_name']}",
        "id": f"medsyn_{config['dataset']}_{config['split_name']}",
        "task_type": schema["task_type"],
        "n_classes": int(schema["n_classes"]),
        "n_num_features": len(numerical_columns),
        "n_cat_features": len(categorical_columns),
        "num_feature_names": numerical_columns,
        "cat_feature_names": categorical_columns,
        "label_column": label_column,
        **sizes,
    }
    (output_dir / "info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_dir


def load_generated_samples(config: dict[str, Any]) -> pd.DataFrame:
    paths = config["paths"]
    schema = config["schema"]
    exp_dir = Path(paths["tabddpm_exp_dir"])
    numerical_columns = schema["numerical_columns"]
    categorical_columns = schema["categorical_columns"]
    label_column = schema["label_column"]

    x_num_path = exp_dir / "X_num_train.npy"
    x_cat_path = exp_dir / "X_cat_train.npy"
    y_path = exp_dir / "y_train.npy"
    if not y_path.exists():
        raise FileNotFoundError(f"missing generated labels: {y_path}")

    parts = []
    if x_num_path.exists():
        parts.append(pd.DataFrame(np.load(x_num_path, allow_pickle=True), columns=numerical_columns))
    if x_cat_path.exists():
        parts.append(pd.DataFrame(np.load(x_cat_path, allow_pickle=True), columns=categorical_columns))
    if not parts:
        raise FileNotFoundError(f"missing generated feature arrays in {exp_dir}")

    df = pd.concat(parts, axis=1)
    df[label_column] = np.load(y_path, allow_pickle=True).astype(int)
    return df[numerical_columns + categorical_columns + [label_column]]
