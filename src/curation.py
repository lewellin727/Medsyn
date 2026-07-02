"""Data-centric curation utilities adapted from the CLLM reference code."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def curate_generated(
    train_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    schema: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    label = schema["label_column"]
    x_train = train_df.drop(columns=[label])
    y_train = train_df[label]
    x_check = generated_df.drop(columns=[label])
    y_check = generated_df[label]

    nest = int(config["curation"]["n_estimators"])
    clf = XGBClassifier(n_estimators=nest, random_state=int(config["run"]["seed"]))
    clf.fit(x_train, y_train)
    probabilities = clf.predict_proba(x_check)
    gold_prob = probabilities[np.arange(len(y_check)), y_check.astype(int).to_numpy()]
    predictions = probabilities.argmax(axis=1)

    low = float(config["curation"]["confidence_low"])
    high = float(config["curation"]["confidence_high"])
    hard = np.where((gold_prob <= low) & (predictions != y_check.to_numpy()))[0].tolist()
    easy = np.where(gold_prob >= high)[0].tolist()
    excluded = set(hard + easy)
    ambiguous = [idx for idx in range(len(gold_prob)) if idx not in excluded]

    return {
        "easy": easy,
        "ambiguous": ambiguous,
        "hard": hard,
        "confidence": gold_prob.tolist(),
    }
