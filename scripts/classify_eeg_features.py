"""Classify tabular EEG features with cross-validation, permutation and bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _fit_predict_cv(x: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(clf, x, y, cv=cv)


def run_classifier(
    frame: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    positive_label: str,
    negative_label: str | None = None,
    permutations: int = 1000,
    bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    sub = frame.dropna(subset=feature_cols + [label_col]).copy()
    if negative_label is not None:
        sub = sub[sub[label_col].isin([positive_label, negative_label])].copy()
    else:
        labels = [x for x in sub[label_col].drop_duplicates().tolist() if x != positive_label]
        if len(labels) != 1:
            raise ValueError("Pass --negative when the label column has more than two classes.")
        negative_label = labels[0]

    sub["y"] = (sub[label_col] == positive_label).astype(int)
    counts = sub["y"].value_counts()
    if len(counts) != 2:
        raise ValueError("Need exactly two classes after filtering.")
    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        raise ValueError("Need at least two samples in the smallest class.")

    x = sub[feature_cols].to_numpy(dtype=float)
    y = sub["y"].to_numpy(dtype=int)
    pred = _fit_predict_cv(x, y, n_splits=n_splits, seed=seed)

    rng = np.random.default_rng(seed)
    observed_ba = float(balanced_accuracy_score(y, pred))
    perm_bas = []
    for _ in range(permutations):
        y_perm = rng.permutation(y)
        pred_perm = _fit_predict_cv(x, y_perm, n_splits=n_splits, seed=seed)
        perm_bas.append(float(balanced_accuracy_score(y_perm, pred_perm)))
    perm_bas_arr = np.asarray(perm_bas, dtype=float)
    p_perm = float((np.sum(perm_bas_arr >= observed_ba) + 1) / (len(perm_bas_arr) + 1))

    boot_bas = []
    idx = np.arange(len(y))
    for _ in range(bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[sample])) < 2:
            continue
        boot_bas.append(float(balanced_accuracy_score(y[sample], pred[sample])))
    boot_arr = np.asarray(boot_bas, dtype=float)
    ci = np.nanpercentile(boot_arr, [2.5, 97.5]).tolist() if len(boot_arr) else [None, None]

    return {
        "n_total": int(len(y)),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "positive_label": positive_label,
        "negative_label": negative_label,
        "feature_cols": feature_cols,
        "n_splits": int(n_splits),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": observed_ba,
        "ba_bootstrap_ci95": ci,
        "p_permutation": p_perm,
        "n_permutations": int(permutations),
        "n_bootstrap": int(bootstrap),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="CSV with one row per epoch/trial.")
    parser.add_argument("--label", required=True, help="Column containing class labels.")
    parser.add_argument("--positive", required=True, help="Positive class label.")
    parser.add_argument("--negative", help="Negative class label. Required if >2 classes are present.")
    parser.add_argument("--feature-cols", required=True, help="Comma-separated numeric feature columns.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.features)
    feature_cols = [col.strip() for col in args.feature_cols.split(",") if col.strip()]
    result = run_classifier(
        frame,
        feature_cols=feature_cols,
        label_col=args.label,
        positive_label=args.positive,
        negative_label=args.negative,
        permutations=args.permutations,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
