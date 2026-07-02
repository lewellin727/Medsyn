"""Dataset schemas used by the Medsyn CLLM pipeline."""

from __future__ import annotations

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


def get_schema(dataset: str) -> dict[str, Any]:
    if dataset not in DATASET_SCHEMAS:
        supported = ", ".join(sorted(DATASET_SCHEMAS))
        raise ValueError(f"Unsupported dataset {dataset!r}; supported datasets: {supported}")
    return deepcopy(DATASET_SCHEMAS[dataset])


def feature_columns(schema: dict[str, Any]) -> list[str]:
    return schema["numerical_columns"] + schema["categorical_columns"]


def all_columns(schema: dict[str, Any]) -> list[str]:
    return feature_columns(schema) + [schema["label_column"]]
