"""Recompute Table 3 BA, bootstrap CI and permutation p-values.

This script starts from anonymized epoch-level feature CSVs in
``paper_reproducibility/data``. It does not require raw XDF files.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VISUAL_FEATURE_COLS = [
    "PO7_alpha_db",
    "PO8_alpha_db",
    "OZ_alpha_db",
    "PZ_alpha_db",
    "po8_minus_po7_db",
    "contra_minus_ipsi_db",
]

AUDITORY_FEATURE_COLS = [
    "FZ_mean_uv",
    "CZ_mean_uv",
    "P3_mean_uv",
    "PZ_mean_uv",
    "P4_mean_uv",
    "centro_parietal_mean_uv",
    "centro_parietal_peak_uv",
]

PARTICIPANT_ORDER = {
    "Participante 1": 1,
    "Participante 2": 2,
    "Participante con NAION": 3,
}
MODULE_ORDER = {"visual": 0, "audio2": 1, "audio4": 2}
CONTRAST_ORDER = {
    "visual_alpha_cue_left_vs_right": 0,
    "auditory_main_attended_targets_vs_ignored_targets": 1,
    "auditory_secondary_target_hit_vs_standard": 2,
}


def classifier() -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
    )


def run_cv_predictions(x: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> tuple[float, float, np.ndarray]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(classifier(), x, y, cv=cv)
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


def permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    observed_ba: float,
    n_splits: int,
    n_permutations: int,
    seed: int,
    progress_label: str,
) -> tuple[float, float, float]:
    if n_permutations <= 0:
        return np.nan, np.nan, 0.0
    rng = np.random.default_rng(seed)
    perm = np.empty(n_permutations, dtype=float)
    start = time.perf_counter()
    for i in range(n_permutations):
        _, perm[i], _ = run_cv_predictions(x, rng.permutation(y), n_splits, seed + i + 1)
        if (i + 1) % 2000 == 0:
            elapsed = time.perf_counter() - start
            print(f"{progress_label}: {i + 1}/{n_permutations} permutations in {elapsed:.1f}s", flush=True)
    p_value = float((np.sum(perm >= observed_ba) + 1) / (n_permutations + 1))
    return float(np.mean(perm)), p_value, float(time.perf_counter() - start)


def classify_task(
    task: dict[str, Any],
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    sub = task["data"].dropna(subset=task["feature_cols"]).copy()
    x = sub[task["feature_cols"]].to_numpy(dtype=float)
    y = sub["label"].to_numpy(dtype=int)
    if len(sub) < 8 or len(np.unique(y)) < 2:
        raise ValueError(f"Not enough data for {task['task_id']}")
    n_splits = min(5, int(pd.Series(y).value_counts().min()))
    if n_splits < 2:
        raise ValueError(f"Not enough folds for {task['task_id']}")

    accuracy, ba, pred = run_cv_predictions(x, y, n_splits, seed)
    ci_low, ci_high = bootstrap_ci(y, pred, n_bootstrap, seed + 10_000)
    perm_mean, p_value, perm_elapsed = permutation_p_value(
        x=x,
        y=y,
        observed_ba=ba,
        n_splits=n_splits,
        n_permutations=n_permutations,
        seed=seed + 20_000,
        progress_label=task["task_id"],
    )

    return {
        "task_id": task["task_id"],
        "participant": task["participant"],
        "recording_public_id": task.get("recording_public_id", ""),
        "modality": task["modality"],
        "module": task["module"],
        "window": task["window"],
        "contrast": task["contrast"],
        "positive_label": task["positive_label"],
        "negative_label": task["negative_label"],
        "n": int(len(sub)),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "n_splits": int(n_splits),
        "accuracy": accuracy,
        "balanced_accuracy": ba,
        "ba_ci_low": ci_low,
        "ba_ci_high": ci_high,
        "permutation_mean_balanced_accuracy": perm_mean,
        "permutation_p_value": p_value,
        "p_source": "computed",
        "n_permutations": int(n_permutations),
        "n_bootstrap": int(n_bootstrap),
        "permutation_elapsed_s": perm_elapsed,
    }


def build_tasks(data_dir: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []

    visual = pd.read_csv(data_dir / "features_visual_alpha_anonymized.csv")
    visual = visual[visual["window"].eq("cue_0p5_1p2") & visual["included"].astype(bool)].copy()
    for keys, group in visual.groupby(["recording_public_id", "participant"], dropna=False):
        recording_public_id, participant = keys
        task_df = group.copy()
        task_df["label"] = task_df["cue_side"].map({"L": 0, "R": 1}).astype(int)
        tasks.append(
            {
                "task_id": f"visual__{participant}",
                "participant": participant,
                "recording_public_id": recording_public_id,
                "modality": "visual",
                "module": "visual",
                "window": "cue_0p5_1p2",
                "contrast": "visual_alpha_cue_left_vs_right",
                "positive_label": "cue_right",
                "negative_label": "cue_left",
                "feature_cols": VISUAL_FEATURE_COLS,
                "data": task_df,
            }
        )

    auditory = pd.read_csv(data_dir / "features_auditory_erp_anonymized.csv")
    auditory = auditory[auditory["window"].eq("p300_250_600")].copy()
    for keys, group in auditory.groupby(["participant", "recording_public_id", "module"], dropna=False):
        participant, recording_public_id, module = keys
        main = group[group["condition"].isin(["target_atendido_hit", "target_atendido_miss", "target_ignorado"])].copy()
        main["label"] = np.where(main["condition"].eq("target_ignorado"), 0, 1)
        tasks.append(
            {
                "task_id": f"auditory_main__{participant}__{module}",
                "participant": participant,
                "recording_public_id": recording_public_id,
                "modality": "auditory",
                "module": module,
                "window": "p300_250_600",
                "contrast": "auditory_main_attended_targets_vs_ignored_targets",
                "positive_label": "target_atendido_hit_or_miss",
                "negative_label": "target_ignorado",
                "feature_cols": AUDITORY_FEATURE_COLS,
                "data": main,
            }
        )

        secondary = group[group["condition"].isin(["target_atendido_hit", "standard"])].copy()
        secondary["label"] = np.where(secondary["condition"].eq("target_atendido_hit"), 1, 0)
        tasks.append(
            {
                "task_id": f"auditory_secondary__{participant}__{module}",
                "participant": participant,
                "recording_public_id": recording_public_id,
                "modality": "auditory",
                "module": module,
                "window": "p300_250_600",
                "contrast": "auditory_secondary_target_hit_vs_standard",
                "positive_label": "target_atendido_hit",
                "negative_label": "standard",
                "feature_cols": AUDITORY_FEATURE_COLS,
                "data": secondary,
            }
        )

    return tasks


def write_compact_table(results: pd.DataFrame, out_dir: Path) -> None:
    ordered = results.copy()
    ordered["participant_order"] = ordered["participant"].map(PARTICIPANT_ORDER).fillna(99)
    ordered["contrast_order"] = ordered["contrast"].map(CONTRAST_ORDER).fillna(99)
    ordered["module_order"] = ordered["module"].map(MODULE_ORDER).fillna(99)
    ordered = ordered.sort_values(["participant_order", "contrast_order", "module_order"])
    ordered.to_csv(out_dir / "table3_audit_recomputed_sorted.csv", index=False)

    rows: list[dict[str, Any]] = []
    for participant, group in ordered.groupby("participant", sort=False):
        visual = group[group["contrast"].eq("visual_alpha_cue_left_vs_right")].iloc[0]
        aud_main = group[group["contrast"].eq("auditory_main_attended_targets_vs_ignored_targets")]
        aud_sec = group[group["contrast"].eq("auditory_secondary_target_hit_vs_standard")]
        row: dict[str, Any] = {
            "participant": participant,
            "visual_ba": visual["balanced_accuracy"],
            "visual_ci": f"{visual['ba_ci_low']:.3f}-{visual['ba_ci_high']:.3f}",
            "visual_p": visual["permutation_p_value"],
        }
        for module in ["audio2", "audio4"]:
            main = aud_main[aud_main["module"].eq(module)].iloc[0]
            sec = aud_sec[aud_sec["module"].eq(module)].iloc[0]
            row[f"{module}_main_ba"] = main["balanced_accuracy"]
            row[f"{module}_main_ci"] = f"{main['ba_ci_low']:.3f}-{main['ba_ci_high']:.3f}"
            row[f"{module}_main_p"] = main["permutation_p_value"]
            row[f"{module}_secondary_ba"] = sec["balanced_accuracy"]
            row[f"{module}_secondary_ci"] = f"{sec['ba_ci_low']:.3f}-{sec['ba_ci_high']:.3f}"
            row[f"{module}_secondary_p"] = sec["permutation_p_value"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "table3_compact_recomputed.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="paper_reproducibility/data")
    parser.add_argument("--out-dir", default="paper_reproducibility/results")
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for task in build_tasks(data_dir):
        results.append(classify_task(task, args.permutations, args.bootstrap, args.seed))
    result_df = pd.DataFrame(results)
    result_df.to_csv(out_dir / "table3_audit_recomputed.csv", index=False)
    write_compact_table(result_df, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
