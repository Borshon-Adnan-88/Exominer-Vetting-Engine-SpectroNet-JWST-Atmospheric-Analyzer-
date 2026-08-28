"""Compact label-blind global/local pilot review figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_overview(processing_manifest: str | Path, arrays_dir: str | Path, output: str | Path) -> None:
    manifest = pd.read_csv(processing_manifest)
    constructed = manifest.loc[manifest["processing_status"] == "constructed"].sort_values("pilot_order")
    fig, axes = plt.subplots(len(constructed), 2, figsize=(12, max(3, len(constructed) * 1.6)), squeeze=False)
    for row_index, (_, row) in enumerate(constructed.iterrows()):
        with np.load(Path(arrays_dir) / row["array_filename"]) as arrays:
            axes[row_index, 0].plot(np.linspace(-0.5, 0.5, len(arrays["global_flux"]), endpoint=False), arrays["global_flux"], lw=0.6)
            axes[row_index, 1].plot(np.linspace(-2.5, 2.5, len(arrays["local_flux"]), endpoint=False), arrays["local_flux"], lw=0.8)
        axes[row_index, 0].set_ylabel(str(row["kepoi_name"]), rotation=0, ha="right")
        axes[row_index, 1].axvline(0, color="gray", lw=0.5)
        axes[row_index, 1].axvspan(-0.5, 0.5, color="gray", alpha=0.08)
    axes[0, 0].set_title("Global phase view")
    axes[0, 1].set_title("Local view (transit durations)")
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
