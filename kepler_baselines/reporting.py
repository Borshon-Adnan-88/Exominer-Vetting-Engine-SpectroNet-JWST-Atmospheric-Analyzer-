"""Compact aggregate plots; no model selection or calibration fitting."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_plots(metrics, bins, diagnostic, destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    families = ("logistic_regression", "random_forest")
    strategies = ("object_stratified", "host_group_stratified")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for i, family in enumerate(families):
        selected = metrics.loc[(metrics.model == family) & (metrics.partition == "test") & (metrics.track == "primary")]
        for j, metric in enumerate(("roc_auc", "average_precision", "brier")):
            ax = axes[i, j]
            for repeat, group in selected.groupby("repeat_id"):
                values = group.set_index("strategy").loc[list(strategies), metric]
                ax.plot([0, 1], values, "o-", alpha=.65, linewidth=1, label=str(repeat))
            ax.set_xticks([0, 1], ["Object", "Host grouped"])
            ax.set_title(f"{family}\n{metric}"); ax.grid(alpha=.2)
    fig.suptitle("Primary test comparison: ten paired repeats; no confidence intervals")
    fig.savefig(destination / "paired_test_metrics.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for i, family in enumerate(families):
        for j, strategy in enumerate(strategies):
            ax = axes[i, j]; ax.plot([0, 1], [0, 1], "k--", linewidth=1)
            selected = bins.loc[(bins.model == family) & (bins.strategy == strategy) & (bins.partition == "test") & (bins.track == "primary")]
            for _, repeat in selected.groupby("repeat_id"):
                ax.plot(repeat.mean_probability, repeat.observed_fraction, "o-", alpha=.35, markersize=3)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed positive fraction", title=f"{family}\n{strategy}")
    fig.suptitle("Raw test calibration: ten fixed bins, one curve per repeat")
    fig.savefig(destination / "test_calibration.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, family in zip(axes, families):
        selected = metrics.loc[(metrics.model == family) & (metrics.partition == "test")]
        for strategy in strategies:
            table = selected.loc[selected.strategy == strategy].pivot(index="repeat_id", columns="track", values="roc_auc")
            ax.scatter(table.primary, table.qc_pass, label=strategy, alpha=.8)
        ax.plot([0, 1], [0, 1], "k--", alpha=.5)
        ax.set(xlabel="Primary ROC-AUC", ylabel="QC-pass evaluation ROC-AUC", title=family)
        ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.suptitle("Evaluation sensitivity only: identical fitted models")
    fig.savefig(destination / "qc_sensitivity.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, family in zip(axes, families):
        selected = diagnostic.loc[diagnostic.model == family]
        for _, repeat in selected.groupby("repeat_id"):
            row = repeat.set_index("host_group").reindex(["seen_in_training", "unseen_in_training"])
            ax.plot([0, 1], row.roc_auc, "o-", alpha=.6)
        ax.set_xticks([0, 1], ["Seen host", "Unseen host"])
        ax.set(title=family, ylabel="Test ROC-AUC"); ax.grid(alpha=.2)
    fig.suptitle("Secondary diagnostic only: object-split test hosts")
    fig.savefig(destination / "host_diagnostic.png", dpi=150); plt.close(fig)
