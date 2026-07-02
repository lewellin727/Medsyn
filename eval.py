"""Evaluation entrypoint for medsyn outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate medsyn outputs.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = Path(config["eval"]["input_file"])
    output_path = Path(config["eval"]["output_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.exists():
        with input_path.open("r", encoding="utf-8") as input_file:
            predictions = json.load(input_file)
        record_count = len(predictions.get("records", []))
    else:
        record_count = 0

    metrics = {
        "sysname": config["sysname"],
        "dataset": config["run"]["dataset"],
        "record_count": record_count,
        "status": "placeholder",
    }

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
