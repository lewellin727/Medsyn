"""Shared dataset schema and path overrides for baseline wrappers."""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any


DATASET_SCHEMAS: dict[str, dict[str, Any]] = {
    "adult": {
        "label_column": "salary",
        "task_type": "binclass",
        "n_classes": 2,
        "numerical_columns": [
            "age",
            "fnlwgt",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ],
        "categorical_columns": [
            "marital-status",
            "relationship",
            "race",
            "sex",
            "country",
            "employment_type",
        ],
    },
    "drug": {
        "label_column": "y",
        "task_type": "binclass",
        "n_classes": 2,
        "numerical_columns": [
            "Nscore",
            "Escore",
            "Oscore",
            "AScore",
            "Cscore",
            "Impulsive",
            "SS",
        ],
        "categorical_columns": [
            "Age",
            "Gender",
            "Education",
            "Country",
            "Ethnicity",
            "Alcohol",
            "Amphet",
            "Amyl",
            "Benzos",
            "Cannabis",
            "Coke",
            "Crack",
            "Ecstasy",
            "Heroin",
            "Ketamine",
            "Legalh",
            "LSD",
            "Meth",
            "Mushrooms",
            "VSA",
        ],
    },
    "compas": {
        "label_column": "y",
        "task_type": "binclass",
        "n_classes": 2,
        "numerical_columns": [
            "age",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
            "priors_count",
        ],
        "categorical_columns": [
            "sex",
            "age_cat_25-45",
            "age_cat_Greaterthan45",
            "age_cat_Lessthan25",
            "race_African-American",
            "race_Caucasian",
            "c_charge_degree_F",
            "c_charge_degree_M",
        ],
    },
}


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=sorted(DATASET_SCHEMAS))
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--size", type=int, choices=[20, 40, 100, 200])


def apply_dataset_overrides(config: dict[str, Any], baseline: str, args: argparse.Namespace) -> dict[str, Any]:
    has_override = args.dataset is not None or args.data_seed is not None or args.size is not None
    if not has_override:
        return config

    updated = deepcopy(config)
    dataset = args.dataset or updated["dataset"]
    data_seed = 0 if args.data_seed is None else args.data_seed
    size = 20 if args.size is None else args.size
    split_name = f"seed{data_seed}_n{size}"

    if dataset not in DATASET_SCHEMAS:
        raise ValueError(f"Unsupported dataset: {dataset}")

    schema = deepcopy(DATASET_SCHEMAS[dataset])
    if baseline in {"ctgan", "tvae"}:
        schema["discrete_columns"] = schema["categorical_columns"] + [schema["label_column"]]

    updated["dataset"] = dataset
    updated["split_name"] = split_name
    updated["schema"] = schema

    paths = updated["paths"]
    split_root = f"datasets/{dataset}/splits"
    paths["train_csv"] = f"{split_root}/{split_name}_train.csv"
    if "val_csv" in paths:
        paths["val_csv"] = f"{split_root}/{split_name}_oracle.csv"
    if "test_csv" in paths:
        paths["test_csv"] = f"{split_root}/{split_name}_test.csv"
    if "result_csv" in paths:
        paths["result_csv"] = f"results/{dataset}/{baseline}_{split_name}.csv"
    if "log_file" in paths:
        paths["log_file"] = f"logs/{dataset}/{baseline}_{split_name}.jsonl"

    artifact_dir = f"baselines/{baseline}/artifacts/{dataset}_{split_name}"
    if baseline in {"ctgan", "tvae"} and "model_path" in paths:
        paths["model_path"] = f"{artifact_dir}/{baseline}.pt"
    if baseline == "great":
        paths["model_dir"] = f"{artifact_dir}/model"
        paths["experiment_dir"] = f"{artifact_dir}/trainer"
    if baseline == "tabddpm":
        paths["tabddpm_data_dir"] = f"baselines/tabddpm/tab-ddpm/data/medsyn_{dataset}_{split_name}"
        paths["tabddpm_exp_dir"] = f"baselines/tabddpm/tab-ddpm/exp/{dataset}/medsyn_{split_name}"

    return updated
