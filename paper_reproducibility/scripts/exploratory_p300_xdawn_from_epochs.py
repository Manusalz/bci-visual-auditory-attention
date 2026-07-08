"""Optional exploratory ERP/P300 classification from derived epochs.

This script is intentionally generic. It does not read private XDF recordings.
It expects a local NPZ file with already-derived, anonymized epochs:

  epochs: array, shape (n_epochs, n_channels, n_times)
  times: array, shape (n_times,), seconds relative to auditory stimulus onset
  labels: binary array, shape (n_epochs,), with 0/1 class labels
  groups: optional array, shape (n_epochs,), block/run identifiers
  channel_names: optional array, shape (n_channels,)

The goal is to document and test an exploratory P300-oriented pipeline:
xDAWN -> flattened epoch features or OAS covariance -> tangent-space features
-> balanced logistic regression.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.covariance import OAS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import mne
    from mne.preprocessing import Xdawn
except ImportError as exc:  # pragma: no cover - depends on optional install
    raise SystemExit(
        "This optional script requires MNE-Python. Install with: "
        "python -m pip install -r paper_reproducibility/requirements_xdawn_optional.txt"
    ) from exc


mne.set_log_level("WARNING")


def load_epoch_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=True)
    epochs = np.asarray(data["epochs"], dtype=float)
    times = np.asarray(data["times"], dtype=float)
    labels = np.asarray(data["labels"], dtype=int)
    if "groups" in data:
        groups = np.asarray(data["groups"]).astype(str)
    else:
        groups = np.array(["all"] * len(labels), dtype=object)
    if "channel_names" in data:
        channel_names = [str(item) for item in np.asarray(data["channel_names"]).tolist()]
    else:
        channel_names = [f"EEG{i + 1}" for i in range(epochs.shape[1])]

    if epochs.ndim != 3:
        raise ValueError("epochs must have shape (n_epochs, n_channels, n_times)")
    if len(labels) != len(epochs):
        raise ValueError("labels length must match number of epochs")
    if len(groups) != len(epochs):
        raise ValueError("groups length must match number of epochs")
    if len(times) != epochs.shape[2]:
        raise ValueError("times length must match epoch time dimension")
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("labels must be binary values 0/1")
    return epochs, times, labels, groups, channel_names


def infer_sfreq(times: np.ndarray) -> float:
    diffs = np.diff(times)
    if len(diffs) == 0 or not np.all(np.isfinite(diffs)):
        raise ValueError("times must contain at least two finite samples")
    return float(1.0 / np.median(diffs))


def make_mne_epochs(
    epochs: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    channel_names: list[str],
) -> mne.EpochsArray:
    sfreq = infer_sfreq(times)
    info = mne.create_info(channel_names, sfreq=sfreq, ch_types="eeg")
    events = np.column_stack(
        [np.arange(len(labels), dtype=int), np.zeros(len(labels), dtype=int), labels.astype(int) + 1]
    )
    event_id = {}
    if np.any(labels == 0):
        event_id["class0"] = 1
    if np.any(labels == 1):
        event_id["class1"] = 2
    return mne.EpochsArray(
        epochs,
        info,
        events=events,
        event_id=event_id,
        tmin=float(times[0]),
        baseline=None,
        verbose=False,
    )


def tabular_window_features(epochs: np.ndarray, times: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    mask = (times >= window[0]) & (times <= window[1])
    if not np.any(mask):
        raise ValueError("feature window does not overlap epoch times")
    channel_means = np.nanmean(epochs[:, :, mask], axis=2)
    global_mean = np.nanmean(channel_means, axis=1, keepdims=True)
    global_peak = np.nanmax(np.nanmean(epochs[:, :, mask], axis=1), axis=1, keepdims=True)
    return np.hstack([channel_means, global_mean, global_peak])


def logistic_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
    )
    clf.fit(x_train, y_train)
    return clf.predict(x_test), clf.predict_proba(x_test)[:, 1]


def fit_predict_xdawn_flat(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    times: np.ndarray,
    channel_names: list[str],
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    xdawn = Xdawn(n_components=n_components, reg="oas", correct_overlap="auto")
    train_epochs = make_mne_epochs(x_train, y_train, times, channel_names)
    test_epochs = make_mne_epochs(x_test, np.zeros(len(x_test), dtype=int), times, channel_names)
    xdawn.fit(train_epochs)
    train_x = xdawn.transform(train_epochs).reshape(len(x_train), -1)
    test_x = xdawn.transform(test_epochs).reshape(len(x_test), -1)
    return logistic_predict(train_x, y_train, test_x)


def spd_from_epoch(epoch: np.ndarray) -> np.ndarray:
    cov = OAS().fit(epoch.T).covariance_
    cov = 0.5 * (cov + cov.T)
    eps = 1e-7 * np.trace(cov) / max(1, cov.shape[0])
    return cov + np.eye(cov.shape[0]) * eps


def logm_spd(mat: np.ndarray) -> np.ndarray:
    vals, vecs = linalg.eigh(0.5 * (mat + mat.T))
    vals = np.maximum(vals, 1e-12)
    return (vecs * np.log(vals)) @ vecs.T


def expm_spd(mat: np.ndarray) -> np.ndarray:
    vals, vecs = linalg.eigh(0.5 * (mat + mat.T))
    return (vecs * np.exp(vals)) @ vecs.T


def invsqrtm_spd(mat: np.ndarray) -> np.ndarray:
    vals, vecs = linalg.eigh(0.5 * (mat + mat.T))
    vals = np.maximum(vals, 1e-12)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def fit_tangent_reference(covs: np.ndarray) -> np.ndarray:
    return expm_spd(np.mean(np.stack([logm_spd(cov) for cov in covs], axis=0), axis=0))


def vectorize_symmetric(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(mat.shape[0])
    out = mat[idx].astype(float)
    out[idx[0] != idx[1]] *= math.sqrt(2.0)
    return out


def tangent_transform(covs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    inv_sqrt = invsqrtm_spd(reference)
    rows = []
    for cov in covs:
        projected = inv_sqrt @ cov @ inv_sqrt
        rows.append(vectorize_symmetric(logm_spd(projected)))
    return np.vstack(rows)


def fit_predict_xdawn_tangent(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    times: np.ndarray,
    channel_names: list[str],
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    xdawn = Xdawn(n_components=n_components, reg="oas", correct_overlap="auto")
    train_epochs = make_mne_epochs(x_train, y_train, times, channel_names)
    test_epochs = make_mne_epochs(x_test, np.zeros(len(x_test), dtype=int), times, channel_names)
    xdawn.fit(train_epochs)
    train_xd = xdawn.transform(train_epochs)
    test_xd = xdawn.transform(test_epochs)
    train_covs = np.stack([spd_from_epoch(epoch) for epoch in train_xd], axis=0)
    test_covs = np.stack([spd_from_epoch(epoch) for epoch in test_xd], axis=0)
    reference = fit_tangent_reference(train_covs)
    return logistic_predict(tangent_transform(train_covs, reference), y_train, tangent_transform(test_covs, reference))


def make_folds(labels: np.ndarray, groups: np.ndarray, validation: str, seed: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
    if validation == "leave_one_group_out":
        folds = []
        for group in sorted(pd.unique(groups)):
            test = groups == group
            train = ~test
            folds.append((np.where(train)[0], np.where(test)[0], str(group)))
        return folds

    counts = pd.Series(labels).value_counts()
    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        return []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train, test, f"fold_{i + 1}") for i, (train, test) in enumerate(cv.split(np.zeros(len(labels)), labels))]


def metric_row(labels: np.ndarray, pred: np.ndarray, score: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    y_true = labels[valid]
    y_pred = pred[valid]
    y_score = score[valid]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"n_valid": int(len(y_true)), "balanced_accuracy": np.nan}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out: dict[str, float | int] = {
        "n_valid": int(len(y_true)),
        "n_class0": int(np.sum(y_true == 0)),
        "n_class1": int(np.sum(y_true == 1)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["roc_auc"] = np.nan
    try:
        out["average_precision"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        out["average_precision"] = np.nan
    return out


def run_cv(
    epochs: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    times: np.ndarray,
    channel_names: list[str],
    validation: str,
    seed: int,
    pipeline: str,
    n_components: int,
    feature_window: tuple[float, float],
) -> tuple[dict[str, float | int], pd.DataFrame]:
    pred = np.full(len(labels), -1, dtype=int)
    score = np.full(len(labels), np.nan, dtype=float)
    valid = np.zeros(len(labels), dtype=bool)
    rows = []

    for train_idx, test_idx, fold_id in make_folds(labels, groups, validation, seed):
        if len(np.unique(labels[train_idx])) < 2:
            continue
        if pipeline == "tabular_window_lr":
            x_all = tabular_window_features(epochs, times, feature_window)
            fold_pred, fold_score = logistic_predict(x_all[train_idx], labels[train_idx], x_all[test_idx])
        elif pipeline == "xdawn_flat_lr":
            fold_pred, fold_score = fit_predict_xdawn_flat(
                epochs[train_idx], labels[train_idx], epochs[test_idx], times, channel_names, n_components
            )
        elif pipeline == "xdawn_oas_tangent_lr":
            fold_pred, fold_score = fit_predict_xdawn_tangent(
                epochs[train_idx], labels[train_idx], epochs[test_idx], times, channel_names, n_components
            )
        else:
            raise ValueError(f"unknown pipeline: {pipeline}")

        pred[test_idx] = fold_pred
        score[test_idx] = fold_score
        valid[test_idx] = True
        for idx, y_hat, y_score in zip(test_idx, fold_pred, fold_score):
            rows.append(
                {
                    "epoch_index": int(idx),
                    "fold": fold_id,
                    "label": int(labels[idx]),
                    "prediction": int(y_hat),
                    "score": float(y_score),
                }
            )

    out = metric_row(labels, pred, score, valid)
    out["n_folds_valid"] = int(len(set(row["fold"] for row in rows)))
    return out, pd.DataFrame(rows)


def permute_within_groups(labels: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = labels.copy()
    for group in pd.unique(groups):
        idx = np.where(groups == group)[0]
        out[idx] = rng.permutation(out[idx])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Optional exploratory xDAWN/OAS/Tangent P300 pipeline from epochs.")
    parser.add_argument("--epochs-npz", required=True)
    parser.add_argument("--out-dir", default="outputs/xdawn_optional")
    parser.add_argument("--validation", choices=["leave_one_group_out", "stratified_epoch"], default="leave_one_group_out")
    parser.add_argument("--pipelines", default="tabular_window_lr,xdawn_flat_lr,xdawn_oas_tangent_lr")
    parser.add_argument("--feature-start", type=float, default=0.25)
    parser.add_argument("--feature-end", type=float, default=0.60)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--permutations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    epochs, times, labels, groups, channel_names = load_epoch_npz(Path(args.epochs_npz))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipelines = [item.strip() for item in args.pipelines.split(",") if item.strip()]
    feature_window = (args.feature_start, args.feature_end)

    result_rows = []
    prediction_tables = []
    rng = np.random.default_rng(args.seed)
    for pipeline in pipelines:
        metrics, preds = run_cv(
            epochs=epochs,
            labels=labels,
            groups=groups,
            times=times,
            channel_names=channel_names,
            validation=args.validation,
            seed=args.seed,
            pipeline=pipeline,
            n_components=args.n_components,
            feature_window=feature_window,
        )
        metrics.update({"pipeline": pipeline, "validation": args.validation, "p_permutation": np.nan})

        if args.permutations > 0 and np.isfinite(metrics.get("balanced_accuracy", np.nan)):
            null_ba = []
            for _ in range(args.permutations):
                perm_labels = permute_within_groups(labels, groups, rng)
                perm_metrics, _ = run_cv(
                    epochs=epochs,
                    labels=perm_labels,
                    groups=groups,
                    times=times,
                    channel_names=channel_names,
                    validation=args.validation,
                    seed=args.seed,
                    pipeline=pipeline,
                    n_components=args.n_components,
                    feature_window=feature_window,
                )
                ba = perm_metrics.get("balanced_accuracy", np.nan)
                if np.isfinite(ba):
                    null_ba.append(float(ba))
            if null_ba:
                null_arr = np.asarray(null_ba)
                metrics["p_permutation"] = float(
                    (np.sum(null_arr >= float(metrics["balanced_accuracy"])) + 1) / (len(null_arr) + 1)
                )
                metrics["permutation_mean_ba"] = float(np.mean(null_arr))
                metrics["permutation_sd_ba"] = float(np.std(null_arr, ddof=1)) if len(null_arr) > 1 else 0.0

        result_rows.append(metrics)
        if not preds.empty:
            preds.insert(0, "pipeline", pipeline)
            prediction_tables.append(preds)

    results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True) if prediction_tables else pd.DataFrame()
    results.to_csv(out_dir / "xdawn_optional_results.csv", index=False)
    predictions.to_csv(out_dir / "xdawn_optional_predictions.csv", index=False)
    (out_dir / "xdawn_optional_config.json").write_text(
        json.dumps(
            {
                "epochs_npz": str(args.epochs_npz),
                "validation": args.validation,
                "pipelines": pipelines,
                "feature_window": feature_window,
                "n_components": args.n_components,
                "permutations": args.permutations,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(results.to_string(index=False))
    print(f"saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
