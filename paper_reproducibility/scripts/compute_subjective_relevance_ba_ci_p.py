"""Recompute the post hoc subjective-relevance auditory classification.

Input:
    paper_reproducibility/data/features_auditory_subjective_relevance_anonymized.csv

The CSV contains derived, anonymized epoch-level features only. It does not
contain raw XDF data, absolute timestamps, webcam data, eye-tracking logs, or
private recording identifiers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "FZ_mean_uv",
    "CZ_mean_uv",
    "P3_mean_uv",
    "PZ_mean_uv",
    "P4_mean_uv",
    "centro_parietal_mean_uv",
    "centro_parietal_peak_uv",
]

PARTICIPANT_ORDER = {"Participante 1": 1, "Participante 2": 2, "Participante con NAION": 3}
MODULE_ORDER = {"audio2": 1, "audio4": 2}


def run_cv_predictions(x: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> tuple[float, float, np.ndarray]:
    pred = np.empty_like(y, dtype=int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(x, y):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        model.fit(x_train, y[train_idx])
        pred[test_idx] = model.predict(x_test)
    return float(accuracy_score(y, pred)), float(balanced_accuracy_score(y, pred)), pred


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    scores = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = np.concatenate(
            [
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            ]
        )
        scores[i] = balanced_accuracy_score(y[sample], pred[sample])
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def leave_one_block_out(data: pd.DataFrame, seed: int) -> dict[str, Any]:
    sub = data.dropna(subset=FEATURE_COLS).copy()
    predictions: list[int] = []
    observed: list[int] = []
    valid_blocks: list[str] = []
    skipped_blocks: list[str] = []
    for block in sorted(sub["block"].dropna().astype(int).unique()):
        train = sub[sub["block"].astype(int).ne(int(block))]
        test = sub[sub["block"].astype(int).eq(int(block))]
        if train["label"].nunique() < 2 or test["label"].nunique() < 2:
            skipped_blocks.append(str(block))
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[FEATURE_COLS].to_numpy(dtype=float))
        x_test = scaler.transform(test[FEATURE_COLS].to_numpy(dtype=float))
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        model.fit(x_train, train["label"].to_numpy(dtype=int))
        test_pred = model.predict(x_test)
        predictions.extend([int(v) for v in test_pred])
        observed.extend([int(v) for v in test["label"].to_numpy(dtype=int)])
        valid_blocks.append(str(block))
    if not observed or len(set(observed)) < 2:
        return {
            "lobo_feasible": False,
            "lobo_valid_blocks": ",".join(valid_blocks),
            "lobo_skipped_blocks": ",".join(skipped_blocks),
        }
    y = np.asarray(observed, dtype=int)
    pred = np.asarray(predictions, dtype=int)
    ci_low, ci_high = bootstrap_ci(y, pred, n_bootstrap=5000, seed=seed + 30_000)
    return {
        "lobo_feasible": True,
        "lobo_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "lobo_ba_ci_low": ci_low,
        "lobo_ba_ci_high": ci_high,
        "lobo_n": int(len(y)),
        "lobo_n_positive": int(np.sum(y == 1)),
        "lobo_n_negative": int(np.sum(y == 0)),
        "lobo_valid_blocks": ",".join(valid_blocks),
        "lobo_skipped_blocks": ",".join(skipped_blocks),
    }


def classify_group(data: pd.DataFrame, permutations: int, bootstrap: int, seed: int) -> dict[str, Any]:
    sub = data.dropna(subset=FEATURE_COLS).copy()
    x = sub[FEATURE_COLS].to_numpy(dtype=float)
    y = sub["label"].to_numpy(dtype=int)
    n_splits = min(5, int(pd.Series(y).value_counts().min()))
    accuracy, ba, pred = run_cv_predictions(x, y, n_splits, seed)
    ci_low, ci_high = bootstrap_ci(y, pred, bootstrap, seed + 10_000)

    rng = np.random.default_rng(seed + 20_000)
    perm_scores = np.empty(permutations, dtype=float)
    for i in range(permutations):
        _, perm_scores[i], _ = run_cv_predictions(x, rng.permutation(y), n_splits, seed + i + 1)
        if (i + 1) % 1000 == 0:
            print(f"{data.name}: {i + 1}/{permutations} permutations", flush=True)
    p_value = float((np.sum(perm_scores >= ba) + 1) / (permutations + 1))

    return {
        "participant": str(sub["participant"].iloc[0]),
        "recording_public_id": str(sub["recording_public_id"].iloc[0]),
        "module": str(sub["module"].iloc[0]),
        "window": str(sub["window"].iloc[0]),
        "contrast": "subjective_relevant_hit_vs_standard",
        "positive_label": "objetivo_subjetivo_relevante_con_respuesta",
        "negative_label": "estandar_no_objetivo_estricto",
        "n": int(len(y)),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "n_splits": int(n_splits),
        "accuracy": accuracy,
        "balanced_accuracy": ba,
        "ba_ci_low": ci_low,
        "ba_ci_high": ci_high,
        "permutation_mean_balanced_accuracy": float(np.mean(perm_scores)),
        "permutation_p_value": p_value,
        "n_permutations": int(permutations),
        "n_bootstrap": int(bootstrap),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("paper_reproducibility/data/features_auditory_subjective_relevance_anonymized.csv"))
    parser.add_argument("--out", type=Path, default=Path("paper_reproducibility/results/auditory_subjective_relevance_recomputed.csv"))
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    if "epoch_order" in frame.columns:
        frame = frame.sort_values("epoch_order").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["participant", "recording_public_id", "module"], sort=False):
        group = group.copy()
        group.name = "__".join(str(x) for x in keys)
        result = classify_group(group, args.permutations, args.bootstrap, args.seed)
        result.update(leave_one_block_out(group, args.seed))
        rows.append(result)

    out = pd.DataFrame(rows)
    out["participant_order"] = out["participant"].map(PARTICIPANT_ORDER).fillna(99)
    out["module_order"] = out["module"].map(MODULE_ORDER).fillna(99)
    out = out.sort_values(["participant_order", "module_order"]).drop(columns=["participant_order", "module_order"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
