"""Project-compatible entrypoint for the SMOTE baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baselines.common import add_dataset_args, apply_dataset_overrides


class RangeStepSMOTE(SMOTE):
    """SMOTE variant matching TabDDPM's lam1/lam2 interpolation range."""

    def __init__(
        self,
        lam1: float = 0.0,
        lam2: float = 1.0,
        *,
        sampling_strategy: str | dict[int, int] = "auto",
        random_state: int | None = None,
        k_neighbors: int = 5,
    ) -> None:
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )
        self.lam1 = lam1
        self.lam2 = lam2

    def _make_samples(
        self,
        X,
        y_dtype,
        y_type,
        nn_data,
        nn_num,
        n_samples,
        step_size=1.0,
        y=None,
    ):
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)
        steps = step_size * random_state.uniform(
            low=self.lam1,
            high=self.lam2,
            size=n_samples,
        )[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])
        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps, y_type, y)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new


class RangeStepSMOTENC(SMOTENC):
    """SMOTENC variant matching TabDDPM's lam1/lam2 interpolation range."""

    def __init__(
        self,
        lam1: float = 0.0,
        lam2: float = 1.0,
        *,
        categorical_features: list[int],
        sampling_strategy: str | dict[int, int] = "auto",
        random_state: int | None = None,
        k_neighbors: int = 5,
    ) -> None:
        super().__init__(
            categorical_features=categorical_features,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )
        self.lam1 = lam1
        self.lam2 = lam2

    def _make_samples(
        self,
        X,
        y_dtype,
        y_type,
        nn_data,
        nn_num,
        n_samples,
        step_size=1.0,
        y=None,
    ):
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)
        steps = step_size * random_state.uniform(
            low=self.lam1,
            high=self.lam2,
            size=n_samples,
        )[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])
        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps, y_type, y)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_log(config: dict[str, Any], record: dict[str, Any]) -> None:
    log_path = Path(config["paths"]["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def load_training_data(config: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    schema = config["schema"]
    train_path = Path(config["paths"]["train_csv"])
    df = pd.read_csv(train_path)
    columns = schema["numerical_columns"] + schema["categorical_columns"] + [schema["label_column"]]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{train_path} is missing required columns: {missing}")
    if df[columns].isna().any().any():
        raise ValueError(f"SMOTE input contains missing values: {train_path}")

    x_num = df[schema["numerical_columns"]].to_numpy(dtype=float)
    x_cat = df[schema["categorical_columns"]].to_numpy(dtype=int)
    y = df[schema["label_column"]].to_numpy(dtype=int)
    return df[columns].copy(), x_num, x_cat, y


def build_sampling_strategy(y: np.ndarray, config: dict[str, Any], num_samples: int | None) -> dict[int, int]:
    labels, counts = np.unique(y, return_counts=True)
    if num_samples is None:
        frac_samples = float(config["smote"]["frac_samples"])
        additions = {int(label): int(round(frac_samples * count)) for label, count in zip(labels, counts)}
    else:
        weights = counts / counts.sum()
        raw = weights * int(num_samples)
        base = np.floor(raw).astype(int)
        remainder = int(num_samples) - int(base.sum())
        if remainder:
            order = np.argsort(raw - base)[::-1]
            base[order[:remainder]] += 1
        additions = {int(label): int(addition) for label, addition in zip(labels, base)}
    return {
        int(label): int(count + additions[int(label)])
        for label, count in zip(labels, counts)
        if additions[int(label)] > 0
    }


def round_discrete_numeric_columns(reference: np.ndarray, generated: np.ndarray) -> np.ndarray:
    rounded = generated.copy()
    for column_idx in range(reference.shape[1]):
        unique_values = np.unique(reference[:, column_idx])
        if len(unique_values) <= 32 and np.all(unique_values == np.round(unique_values)):
            rounded[:, column_idx] = np.round(rounded[:, column_idx])
    return rounded


def sample_smote(config: dict[str, Any]) -> pd.DataFrame:
    schema = config["schema"]
    smote_cfg = config["smote"]
    _, x_num, x_cat, y = load_training_data(config)

    min_count = int(pd.Series(y).value_counts().min())
    k_neighbours = int(smote_cfg["k_neighbours"])
    if min_count <= k_neighbours:
        raise ValueError(
            f"k_neighbours={k_neighbours} requires every class to have at least "
            f"{k_neighbours + 1} rows; smallest class has {min_count}"
        )

    lam_delta = float(smote_cfg["frac_lam_del"])
    lam1 = lam_delta / 2.0
    lam2 = 1.0 - lam_delta / 2.0
    scaler = MinMaxScaler().fit(x_num)
    x_scaled = scaler.transform(x_num).astype(object)
    x_train = np.concatenate([x_scaled, x_cat], axis=1, dtype=object)
    categorical_features = list(range(x_num.shape[1], x_train.shape[1]))
    sampling_strategy = build_sampling_strategy(
        y,
        config,
        int(smote_cfg["sample"]["num_samples"]) if smote_cfg["sample"]["num_samples"] is not None else None,
    )

    sampler = RangeStepSMOTENC(
        lam1=lam1,
        lam2=lam2,
        random_state=int(smote_cfg["seed"]),
        k_neighbors=k_neighbours,
        categorical_features=categorical_features,
        sampling_strategy=sampling_strategy,
    )
    x_res, y_res = sampler.fit_resample(x_train, y)

    if smote_cfg["eval_type"] == "synthetic":
        x_res = x_res[x_train.shape[0] :]
        y_res = y_res[x_train.shape[0] :]

    x_num_res = scaler.inverse_transform(x_res[:, : x_num.shape[1]].astype(float))
    x_num_res = round_discrete_numeric_columns(x_num, x_num_res)
    x_cat_res = x_res[:, x_num.shape[1] :].astype(int)
    y_res = y_res.astype(int)

    result = pd.concat(
        [
            pd.DataFrame(x_num_res, columns=schema["numerical_columns"]),
            pd.DataFrame(x_cat_res, columns=schema["categorical_columns"]),
            pd.Series(y_res, name=schema["label_column"]),
        ],
        axis=1,
    )
    return result[schema["numerical_columns"] + schema["categorical_columns"] + [schema["label_column"]]]


def run(config: dict[str, Any]) -> Path:
    started = time.time()
    write_log(
        config,
        {
            "event": "start",
            "baseline": config["baseline"],
            "dataset": config["dataset"],
            "split_name": config["split_name"],
            "timestamp": started,
        },
    )
    result = sample_smote(config)
    result_path = Path(config["paths"]["result_csv"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(result_path, index=False)
    write_log(
        config,
        {
            "event": "complete",
            "result_csv": str(result_path),
            "num_samples": len(result),
            "seconds": time.time() - started,
            "timestamp": time.time(),
        },
    )
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("baselines/smote/config_baseline.yaml"))
    add_dataset_args(parser)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--k-neighbours", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_dataset_overrides(config, "smote", args)
    if args.num_samples is not None:
        config["smote"]["sample"]["num_samples"] = args.num_samples
    if args.seed is not None:
        config["smote"]["seed"] = args.seed
    if args.k_neighbours is not None:
        config["smote"]["k_neighbours"] = args.k_neighbours

    result_path = run(config)
    print(f"wrote SMOTE samples: {result_path}")


if __name__ == "__main__":
    main()
