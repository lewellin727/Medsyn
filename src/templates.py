"""Prompt templates for CLLM-style tabular generation."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.schemas import all_columns


DATASET_CONTEXT = {
    "adult": "salary above 50K based on demographic features",
    "drug": "features leading to drug usage and consumption",
    "compas": "criminal recidivism risk",
}


def build_format_instructions(schema: dict[str, Any]) -> str:
    fields = []
    for column in all_columns(schema):
        if column == schema["label_column"]:
            fields.append({"name": column, "description": "binary target label"})
        else:
            fields.append({"name": column, "description": "feature column"})
    return (
        "Return only JSON objects, one object per synthetic row. "
        "Each object must contain exactly these fields: "
        f"{json.dumps(fields, ensure_ascii=True)}"
    )


def build_generation_prompt(train_df: pd.DataFrame, dataset: str, schema: dict[str, Any], ic_samples: int) -> str:
    example_df = train_df.sample(n=min(ic_samples, len(train_df)), replace=len(train_df) < ic_samples, random_state=0)
    examples = example_df.to_dict(orient="records")
    context = DATASET_CONTEXT.get(dataset, "the tabular prediction task")
    return f"""You are a synthetic data generator.
Your goal is to produce data which mirrors the given examples in causal structure and feature and label distributions, while producing diverse new samples.

Leverage your knowledge of {context} to generate realistic but diverse samples.

Example data:
{json.dumps(examples, ensure_ascii=True)}

{build_format_instructions(schema)}

Do not copy the examples. Generate realistic new rows with the correct label conditioned on the features.
"""
