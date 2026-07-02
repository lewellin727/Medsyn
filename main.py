"""Main entrypoint for the Medsyn CLLM-style generation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.runner import run_cllm


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--dataset", choices=["adult", "drug", "compas"])
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--size", type=int, choices=[20, 40, 100, 200])
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--serving", choices=["vllm", "together", "azure_openai", "openai_compatible"])
    parser.add_argument("--model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-dry-run", action="store_true")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset:
        config["data"]["dataset"] = args.dataset
    if args.data_seed is not None:
        config["data"]["data_seed"] = args.data_seed
    if args.size is not None:
        config["data"]["size"] = args.size
    if args.n_samples is not None:
        config["generation"]["n_samples"] = args.n_samples
    if args.serving:
        config["generation"]["serving"] = args.serving
    if args.model:
        config["generation"]["model"] = args.model
    if args.dry_run:
        config["generation"]["dry_run"] = True
    if args.no_dry_run:
        config["generation"]["dry_run"] = False
    return config


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    result = run_cllm(config)
    print(f"wrote generated samples: {result['result_csv']}")
    print(f"wrote run metadata: {result['run_json']}")


if __name__ == "__main__":
    main()
