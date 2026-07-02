"""Evaluation entrypoint for Medsyn synthetic tabular outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data import load_split, resolve_dataset_config, split_xy


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--dataset", choices=["adult", "drug", "compas"])
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--size", type=int, choices=[20, 40, 100, 200])
    parser.add_argument("--input-csv", type=Path)
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset:
        config["data"]["dataset"] = args.dataset
    if args.data_seed is not None:
        config["data"]["data_seed"] = args.data_seed
    if args.size is not None:
        config["data"]["size"] = args.size
    config = resolve_dataset_config(config)
    if args.input_csv:
        config["paths"]["result_csv"] = str(args.input_csv)
    return config


def _score(y_true, y_pred, y_score) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["auc"] = 0.0
    return metrics


def evaluate_frame(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: dict[str, Any],
    seed: int,
    n_estimators: int,
) -> dict[str, Any]:
    x_train, y_train = split_xy(train_df, schema)
    x_test, y_test = split_xy(test_df, schema)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    classifiers = {
        "xgb": XGBClassifier(n_estimators=n_estimators, random_state=seed),
        "rf": RandomForestClassifier(n_estimators=n_estimators, random_state=seed),
        "lr": LogisticRegression(random_state=seed, max_iter=1000),
        "dt": DecisionTreeClassifier(random_state=seed),
    }
    scores = {}
    for name, clf in classifiers.items():
        clf.fit(x_train_scaled, y_train)
        y_pred = clf.predict(x_test_scaled)
        try:
            y_score = clf.predict_proba(x_test_scaled)[:, 1]
        except AttributeError:
            y_score = y_pred
        scores[name] = _score(y_test, y_pred, y_score)
    return scores


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    schema = config["schema"]
    paths = config["paths"]

    synthetic_df = load_split(paths["result_csv"], schema)
    original_df = load_split(paths["train_csv"], schema)
    oracle_df = load_split(paths["oracle_csv"], schema)
    test_df = load_split(paths["test_csv"], schema)

    seed = int(config["run"]["seed"])
    n_estimators = int(config["eval"]["n_estimators"])
    metrics = {
        "sysname": config["sysname"],
        "dataset": config["data"]["dataset"],
        "split_name": config["data"]["split_name"],
        "input_csv": paths["result_csv"],
        "rows": {
            "synthetic": int(len(synthetic_df)),
            "original": int(len(original_df)),
            "oracle": int(len(oracle_df)),
            "test": int(len(test_df)),
        },
        "downstream": {
            "synthetic": evaluate_frame(synthetic_df, test_df, schema, seed, n_estimators),
            "original": evaluate_frame(original_df, test_df, schema, seed, n_estimators),
            "oracle": evaluate_frame(oracle_df, test_df, schema, seed, n_estimators),
        },
    }

    output_path = Path(paths["eval_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote evaluation metrics: {output_path}")


if __name__ == "__main__":
    main()
