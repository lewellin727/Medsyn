"""Main experiment entrypoint for medsyn."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.medsyn.runner import run_placeholder


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run medsyn experiment.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = Path(config["run"]["output_file"])
    log_path = Path(config["run"]["log_file"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_placeholder(config)

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps({"event": "run_complete", "output": str(output_path)}) + "\n")
        log_file.flush()

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
