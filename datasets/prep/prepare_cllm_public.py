"""Prepare public/reference datasets for CLLM-style experiments.

The script keeps raw copies under ``datasets/<name>/raw`` and writes processed
CSV files plus balanced low-data splits under ``datasets/<name>/processed``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REF_DATA = ROOT / "ref" / "CLLM-main" / "data"
DATASETS = ROOT / "datasets"


def write_manifest(dataset_dir: Path, manifest: dict) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def copy_raw(source: Path, raw_dir: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"missing reference data file: {source}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / source.name
    shutil.copy2(source, target)
    return target


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object" or str(out[col].dtype) == "category":
            out[col] = out[col].astype("category").cat.codes
    return out


def prepare_adult() -> pd.DataFrame:
    raw = copy_raw(REF_DATA / "adult.csv", DATASETS / "adult" / "raw")
    df = pd.read_csv(raw)

    df["salary"] = df["salary"].map({" <=50K": 1, " >50K": 0}).astype(int)
    df["sex"] = df["sex"].map({" Male": 1, " Female": 0}).astype(int)
    df["country"] = df["country"].replace(" ?", pd.NA)
    df["workclass"] = df["workclass"].replace(" ?", pd.NA)
    df["occupation"] = df["occupation"].replace(" ?", pd.NA)
    df = df.dropna(how="any").copy()

    df.loc[df["country"] != " United-States", "country"] = "Non-US"
    df.loc[df["country"] == " United-States", "country"] = "US"
    df["country"] = df["country"].map({"US": 1, "Non-US": 0}).astype(int)

    df["marital-status"] = df["marital-status"].replace(
        [" Divorced", " Married-spouse-absent", " Never-married", " Separated", " Widowed"],
        "Single",
    )
    df["marital-status"] = df["marital-status"].replace(
        [" Married-AF-spouse", " Married-civ-spouse"],
        "Couple",
    )
    df["marital-status"] = df["marital-status"].map({"Couple": 0, "Single": 1})

    df["relationship"] = df["relationship"].map(
        {
            " Unmarried": 0,
            " Wife": 1,
            " Husband": 2,
            " Not-in-family": 3,
            " Own-child": 4,
            " Other-relative": 5,
        }
    )
    df["race"] = df["race"].map(
        {
            " White": 0,
            " Amer-Indian-Eskimo": 1,
            " Asian-Pac-Islander": 2,
            " Black": 3,
            " Other": 4,
        }
    )

    def employment_type(workclass: str) -> str:
        if workclass in {" Federal-gov", " Local-gov", " State-gov"}:
            return "govt"
        if workclass == " Private":
            return "private"
        if workclass in {" Self-emp-inc", " Self-emp-not-inc"}:
            return "self_employed"
        return "without_pay"

    df["employment_type"] = df["workclass"].apply(employment_type)
    df["employment_type"] = df["employment_type"].map(
        {"govt": 0, "private": 1, "self_employed": 2, "without_pay": 3}
    )
    df = df.drop(columns=["workclass", "education", "occupation"])
    df["capital-gain"] = (df["capital-gain"] > 0).astype(int)
    df["capital-loss"] = (df["capital-loss"] > 0).astype(int)
    return df


def prepare_drug() -> pd.DataFrame:
    raw = copy_raw(REF_DATA / "Drug_Consumption.csv", DATASETS / "drug" / "raw")
    data = pd.read_csv(raw)
    data = data.drop(data[data["Semer"] != "CL0"].index)
    data = data.drop(columns=["Semer", "Caff", "Choc"])
    data = data.reset_index(drop=True)
    data["Gender"] = data["Gender"].apply(lambda x: 1 if x == "M" else 0)

    ordinal_features = [
        "Age",
        "Education",
        "Alcohol",
        "Amyl",
        "Amphet",
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
        "Nicotine",
        "VSA",
    ]
    ordinal_orderings = [
        ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        [
            "Left school before 16 years",
            "Left school at 16 years",
            "Left school at 17 years",
            "Left school at 18 years",
            "Some college or university, no certificate or degree",
            "Professional certificate/ diploma",
            "University degree",
            "Masters degree",
            "Doctorate degree",
        ],
    ] + [["CL0", "CL1", "CL2", "CL3", "CL4", "CL5", "CL6"] for _ in range(16)]

    for col, ordering in zip(ordinal_features, ordinal_orderings):
        data[col] = data[col].apply(lambda x: ordering.index(x))
    for col in ["Country", "Ethnicity"]:
        data[col] = data[col].astype("category").cat.codes

    data["y"] = data["Nicotine"].apply(lambda x: 1 if x not in [0, 1] else 0)
    return data.drop(columns=["ID", "Nicotine"])


def prepare_higgs() -> pd.DataFrame:
    raw = copy_raw(REF_DATA / "training.csv", DATASETS / "higgs" / "raw")
    df = pd.read_csv(raw)
    df = df.drop(columns=["EventId"])
    drop_columns = [
        "DER_deltaeta_jet_jet",
        "DER_mass_jet_jet",
        "DER_prodeta_jet_jet",
        "DER_lep_eta_centrality",
        "PRI_jet_subleading_pt",
        "PRI_jet_subleading_eta",
        "PRI_jet_subleading_phi",
    ]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])
    for weight_col in ["Weight", "Weigh"]:
        if weight_col in df.columns:
            df = df.drop(columns=[weight_col])
    df["Label"] = df["Label"].replace({"s": 0, "b": 1}).astype(int)
    return df.rename(columns={"Label": "y"})


def prepare_compas() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    cache_dir = DATASETS / "compas" / "raw" / "openml_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fetched = fetch_openml(data_id=42192, as_frame=True, parser="auto", data_home=cache_dir)
    df = fetched.frame.copy()
    target = fetched.target_names[0] if fetched.target_names else fetched.target.name
    if target not in df.columns:
        df["y"] = fetched.target
    else:
        df = df.rename(columns={target: "y"})
    df["y"] = df["y"].astype("category").cat.codes
    return encode_categories(df)


PREPARE = {
    "adult": prepare_adult,
    "drug": prepare_drug,
    "higgs": prepare_higgs,
    "compas": prepare_compas,
}


def write_low_data_splits(df: pd.DataFrame, dataset: str, label: str, sample_sizes: list[int], seeds: list[int]) -> None:
    split_dir = DATASETS / dataset / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for total_n in sample_sizes:
        class_count = df[label].nunique()
        if total_n % class_count != 0:
            raise ValueError(f"{dataset}: sample size {total_n} is not divisible by {class_count} classes")
        per_class = total_n // class_count
        min_class_size = int(df[label].value_counts().min())
        if per_class > min_class_size:
            raise ValueError(f"{dataset}: requested {per_class} per class, but smallest class has {min_class_size}")
        for seed in seeds:
            train = df.groupby(label, group_keys=False).sample(n=per_class, random_state=seed).sort_index()
            remaining = df.drop(train.index)
            half = len(remaining) // 2
            oracle = remaining.sample(half, random_state=seed)
            test = remaining.drop(oracle.index)
            prefix = split_dir / f"seed{seed}_n{total_n}"
            train.to_csv(f"{prefix}_train.csv", index=False)
            oracle.to_csv(f"{prefix}_oracle.csv", index=False)
            test.to_csv(f"{prefix}_test.csv", index=False)


def prepare_dataset(name: str, sample_sizes: list[int], seeds: list[int]) -> None:
    df = PREPARE[name]()
    dataset_dir = DATASETS / name
    processed_dir = dataset_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / f"{name}_cllm.csv"
    df.to_csv(processed_path, index=False)

    label = "salary" if name == "adult" else "y"
    write_low_data_splits(df, name, label, sample_sizes, seeds)
    write_manifest(
        dataset_dir,
        {
            "dataset": name,
            "source": "ref/CLLM-main/data" if name in {"adult", "drug", "higgs"} else "OpenML dataset id 42192",
            "processed_file": str(processed_path.relative_to(ROOT)),
            "label_column": label,
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "sample_sizes_total": sample_sizes,
            "seeds": seeds,
        },
    )
    print(f"prepared {name}: {len(df)} rows -> {processed_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(PREPARE),
        default=["adult", "drug", "compas"],
    )
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[20, 40, 100, 200])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        prepare_dataset(dataset, args.sample_sizes, args.seeds)


if __name__ == "__main__":
    main()
