"""Dataset loading and output cleaning for the Medsyn CLLM pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.schemas import all_columns, feature_columns, get_schema


def resolve_dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["data"]["dataset"]
    data_seed = int(config["data"]["data_seed"])
    size = int(config["data"]["size"])
    split_name = f"seed{data_seed}_n{size}"
    schema = get_schema(dataset)

    paths = config.setdefault("paths", {})
    paths["train_csv"] = f"datasets/{dataset}/splits/{split_name}_train.csv"
    paths["oracle_csv"] = f"datasets/{dataset}/splits/{split_name}_oracle.csv"
    paths["test_csv"] = f"datasets/{dataset}/splits/{split_name}_test.csv"
    paths["result_csv"] = f"results/{dataset}/medsyn_{split_name}.csv"
    paths["run_json"] = f"results/{dataset}/medsyn_{split_name}.json"
    paths["log_file"] = f"logs/{dataset}/medsyn_{split_name}.jsonl"
    paths["curation_json"] = f"results/{dataset}/medsyn_{split_name}_curation.json"
    paths["eval_json"] = f"eval/{dataset}/medsyn_{split_name}_metrics.json"

    config["data"]["split_name"] = split_name
    config["schema"] = schema
    return config


def load_split(path: str | Path, schema: dict[str, Any]) -> pd.DataFrame:
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    columns = all_columns(schema)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    return df[columns].copy()


def split_xy(df: pd.DataFrame, schema: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    label = schema["label_column"]
    return df[feature_columns(schema)].copy(), df[label].copy()


def clean_generated(df: pd.DataFrame, reference: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    columns = all_columns(schema)
    df = df.copy()
    if "target" in df.columns and schema["label_column"] not in df.columns:
        df[schema["label_column"]] = df["target"]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Generated data is missing required columns: {missing}")

    df = df[columns].dropna(how="any")
    for column in schema["numerical_columns"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in schema["categorical_columns"] + [schema["label_column"]]:
        df[column] = pd.to_numeric(df[column], errors="coerce").round()
    df = df.dropna(how="any")

    for column in columns:
        try:
            df[column] = df[column].astype(reference[column].dtype)
        except (TypeError, ValueError):
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(how="any")[columns].reset_index(drop=True)
