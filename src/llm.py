"""LLM serving adapter for CLLM-style generation."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from openai import OpenAI


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    records = []
    for candidate in re.findall(r"\{[^{}]*\}", text):
        try:
            records.append(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    return records


def generate_with_openai_compatible(config: dict[str, Any], prompt: str) -> pd.DataFrame:
    generation = config["generation"]
    api = generation.get("api", {})
    serving = generation["serving"]

    if serving == "vllm":
        client = OpenAI(api_key=api.get("api_key", "EMPTY"), base_url=api.get("base_url", "http://localhost:8000/v1"))
        model = generation["model"]
    elif serving in {"together", "openai_compatible"}:
        client = OpenAI(api_key=api["api_key"], base_url=api["base_url"])
        model = generation["model"]
    elif serving == "azure_openai":
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=api["api_key"],
            azure_endpoint=api["azure_endpoint"],
            api_version=api["api_version"],
        )
        model = generation["model"]
    else:
        raise ValueError(f"Unsupported LLM serving mode: {serving}")

    records: list[dict[str, Any]] = []
    max_rounds = int(generation["max_rounds"])
    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a tabular synthetic data generation model."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(generation["temperature"]),
            max_tokens=int(generation["max_tokens"]),
            top_p=float(generation.get("top_p", 0.95)),
            n=int(generation["n_processes"]),
        )
        for choice in response.choices:
            records.extend(_extract_json_objects(choice.message.content or ""))
        if len(records) >= int(generation["n_samples"]):
            break
    return pd.DataFrame(records).head(int(generation["n_samples"]))
