"""CLLM-style Medsyn generation runner."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.curation import curate_generated
from src.data import clean_generated, load_split, resolve_dataset_config
from src.llm import generate_with_openai_compatible
from src.templates import build_generation_prompt


def write_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def run_cllm(config: dict[str, Any]) -> dict[str, Any]:
    config = resolve_dataset_config(config)
    schema = config["schema"]
    paths = config["paths"]
    started = time.time()
    train_df = load_split(paths["train_csv"], schema)

    prompt = build_generation_prompt(
        train_df=train_df,
        dataset=config["data"]["dataset"],
        schema=schema,
        ic_samples=int(config["generation"]["ic_samples"]),
    )
    write_jsonl(
        paths["log_file"],
        {
            "event": "start",
            "dataset": config["data"]["dataset"],
            "split_name": config["data"]["split_name"],
            "dry_run": bool(config["generation"]["dry_run"]),
            "timestamp": started,
        },
    )

    run_record: dict[str, Any] = {
        "sysname": config["sysname"],
        "dataset": config["data"]["dataset"],
        "split_name": config["data"]["split_name"],
        "dry_run": bool(config["generation"]["dry_run"]),
        "prompt_preview": prompt[:1000],
        "result_csv": paths["result_csv"],
        "run_json": paths["run_json"],
    }

    if config["generation"]["dry_run"]:
        output_df = train_df.sample(
            n=min(int(config["generation"]["n_samples"]), len(train_df)),
            replace=len(train_df) < int(config["generation"]["n_samples"]),
            random_state=int(config["run"]["seed"]),
        ).reset_index(drop=True)
        run_record["dry_run_note"] = "Wrote resampled training rows for pipeline validation; disable dry_run for LLM generation."
    else:
        raw_df = generate_with_openai_compatible(config, prompt)
        output_df = clean_generated(raw_df, train_df, schema)

    result_path = Path(paths["result_csv"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(result_path, index=False)
    run_record["num_rows"] = int(len(output_df))

    if config["curation"]["enabled"] and len(output_df):
        curation = curate_generated(train_df, output_df, schema, config)
        curation_path = Path(paths["curation_json"])
        curation_path.parent.mkdir(parents=True, exist_ok=True)
        curation_path.write_text(json.dumps(curation, indent=2) + "\n", encoding="utf-8")
        run_record["curation_json"] = str(curation_path)

    run_json = Path(paths["run_json"])
    run_json.parent.mkdir(parents=True, exist_ok=True)
    run_record["seconds"] = time.time() - started
    run_json.write_text(json.dumps(run_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(paths["log_file"], {"event": "complete", **run_record, "timestamp": time.time()})
    return run_record
