"""Create reproducibility figures from anonymized public CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PARTICIPANT_ORDER = ["Participante 1", "Participante 2", "Participante con NAION"]


def ordered_participants(values: pd.Series) -> list[str]:
    present = set(values.dropna().astype(str))
    return [p for p in PARTICIPANT_ORDER if p in present] + sorted(present - set(PARTICIPANT_ORDER))


def plot_table3_summary(results_dir: Path, out_dir: Path) -> None:
    table = pd.read_csv(results_dir / "table3_audit_final.csv")
    table["ci_low"] = table["ba_ci_low"].astype(float)
    table["ci_high"] = table["ba_ci_high"].astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    specs = [
        ("Visual alfa", "visual", "visual_alpha_cue_left_vs_right"),
        ("Auditivo principal", "auditory", "auditory_main_attended_targets_vs_ignored_targets"),
        ("Auditivo secundario", "auditory", "auditory_secondary_target_hit_vs_standard"),
    ]
    for ax, (title, modality, contrast) in zip(axes, specs):
        sub = table[table["contrast"].eq(contrast)].copy()
        if modality == "auditory":
            sub["label"] = sub["participant"] + " " + sub["module"].str.replace("audio", "A", regex=False)
        else:
            sub["label"] = pd.Categorical(sub["participant"], ordered_participants(sub["participant"]), ordered=True)
            sub = sub.sort_values("label")
            sub["label"] = sub["participant"]
        x = range(len(sub))
        y = sub["balanced_accuracy"].astype(float)
        yerr = [y - sub["ci_low"], sub["ci_high"] - y]
        ax.bar(x, y, color="#8da0cb")
        ax.errorbar(x, y, yerr=yerr, fmt="none", color="#222222", capsize=3, linewidth=1)
        ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(sub["label"], rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Balanced accuracy")
    fig.tight_layout()
    fig.savefig(out_dir / "table3_ba_ci_summary.png", dpi=200)
    plt.close(fig)


def plot_feature_distributions(data_dir: Path, out_dir: Path) -> None:
    visual = pd.read_csv(data_dir / "features_visual_alpha_anonymized.csv")
    visual = visual[visual["window"].eq("cue_0p5_1p2") & visual["included"].astype(bool)].copy()
    auditory = pd.read_csv(data_dir / "features_auditory_erp_anonymized.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    participants = ordered_participants(visual["participant"])
    visual.boxplot(column="contra_minus_ipsi_db", by="participant", ax=axes[0], grid=False)
    axes[0].set_title("Visual alfa 0,5-1,2 s")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Contra - ipsi (dB)")
    axes[0].set_xticklabels(participants, rotation=25, ha="right")
    axes[0].axhline(0, color="#333333", linewidth=1)

    erp = auditory[auditory["condition"].isin(["target_atendido_hit", "standard"])].copy()
    erp.boxplot(column="centro_parietal_mean_uv", by="condition", ax=axes[1], grid=False)
    axes[1].set_title("Auditivo ERP 250-600 ms")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Media centro-parietal (uV)")
    axes[1].tick_params(axis="x", labelrotation=20)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_distributions.png", dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="paper_reproducibility/data")
    parser.add_argument("--results-dir", default="paper_reproducibility/results")
    parser.add_argument("--out-dir", default="paper_reproducibility/figures")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_table3_summary(results_dir, out_dir)
    plot_feature_distributions(data_dir, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
