"""Project-compatible entrypoint for the TabDDPM baseline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baselines.common import add_dataset_args, apply_dataset_overrides
from scripts.prepare_data import load_generated_samples, prepare_tabddpm_data


ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return '"' + str(value) + '"'


def write_tabddpm_config(config: dict[str, Any]) -> Path:
    paths = config["paths"]
    schema = config["schema"]
    tab = config["tabddpm"]
    exp_dir = Path(paths["tabddpm_exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / "config.toml"

    data_dir = Path(paths["tabddpm_data_dir"])
    lines = [
        f"parent_dir = {_toml_value(str(exp_dir.relative_to(Path(paths['tabddpm_root']))))}",
        f"real_data_path = {_toml_value(str(data_dir.relative_to(Path(paths['tabddpm_root']))))}",
        f"num_numerical_features = {len(schema['numerical_columns'])}",
        f"model_type = {_toml_value(tab['model_type'])}",
        f"seed = {int(tab['seed'])}",
        f"device = {_toml_value(tab['device'])}",
        "",
        "[model_params]",
        f"num_classes = {int(tab['model_params']['num_classes'])}",
        f"is_y_cond = {_toml_value(tab['model_params']['is_y_cond'])}",
        "",
        "[model_params.rtdl_params]",
        f"d_layers = {_toml_value(tab['model_params']['rtdl_params']['d_layers'])}",
        f"dropout = {_toml_value(tab['model_params']['rtdl_params']['dropout'])}",
        "",
        "[diffusion_params]",
        f"num_timesteps = {int(tab['diffusion_params']['num_timesteps'])}",
        f"gaussian_loss_type = {_toml_value(tab['diffusion_params']['gaussian_loss_type'])}",
        f"scheduler = {_toml_value(tab['diffusion_params']['scheduler'])}",
        "",
        "[train.main]",
        f"steps = {int(tab['train']['steps'])}",
        f"lr = {_toml_value(tab['train']['lr'])}",
        f"weight_decay = {_toml_value(tab['train']['weight_decay'])}",
        f"batch_size = {int(tab['train']['batch_size'])}",
        "",
        "[train.T]",
    ]
    for key, value in tab["transform"].items():
        lines.append(f"{key} = {_toml_value(value)}")
    lines.extend(
        [
            "",
            "[sample]",
            f"num_samples = {int(tab['sample']['num_samples'])}",
            f"batch_size = {int(tab['sample']['batch_size'])}",
            f"seed = {int(tab['sample']['seed'])}",
            "",
            "[eval.type]",
            f"eval_model = {_toml_value(tab['eval']['eval_model'])}",
            f"eval_type = {_toml_value(tab['eval']['eval_type'])}",
            "",
            "[eval.T]",
        ]
    )
    for key, value in tab.get("eval_transform", tab["transform"]).items():
        lines.append(f"{key} = {_toml_value(value)}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def run_pipeline(config: dict[str, Any], config_path: Path, train: bool, sample: bool, evaluate: bool) -> None:
    tabddpm_root = Path(config["paths"]["tabddpm_root"])
    command = ["python", "scripts/pipeline.py", "--config", str(config_path.relative_to(tabddpm_root))]
    if train:
        command.append("--train")
    if sample:
        command.append("--sample")
    if evaluate:
        command.append("--eval")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tabddpm_root.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=tabddpm_root, env=env, check=True)


def export_results(config: dict[str, Any]) -> Path:
    result_path = Path(config["paths"]["result_csv"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_generated_samples(config)
    df.to_csv(result_path, index=False)
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("baselines/tabddpm/config_baseline.yaml"))
    add_dataset_args(parser)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--num-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_dataset_overrides(config, "tabddpm", args)
    if args.device:
        config["tabddpm"]["device"] = args.device
    if args.steps is not None:
        config["tabddpm"]["train"]["steps"] = args.steps
    if args.num_samples is not None:
        config["tabddpm"]["sample"]["num_samples"] = args.num_samples

    prepare_tabddpm_data(config)
    config_path = write_tabddpm_config(config)
    print(f"prepared TabDDPM data: {config['paths']['tabddpm_data_dir']}")
    print(f"wrote TabDDPM config: {config_path}")

    if args.prepare_only:
        return
    if args.train or args.sample or args.eval:
        run_pipeline(config, config_path, train=args.train, sample=args.sample, evaluate=args.eval)
    if args.sample or args.export:
        result_path = export_results(config)
        print(f"wrote generated samples: {result_path}")


if __name__ == "__main__":
    main()
