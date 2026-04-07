"""
Figures for multivariate analysis: log series, FEVD, IRF export.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_log_series(
    df: pd.DataFrame,
    log_cols: list[str],
    title: str = "Log-transformed series",
    save_path: Path | None = None,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for c in log_cols:
        ax.plot(df.index, df[c], label=c, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi)
        print(f"[Saved figure] {save_path}")
    plt.close(fig)


def plot_fevd_stacked(
    decomp: np.ndarray,
    names: list[str],
    food_name: str,
    max_periods: int,
    save_path: Path | None = None,
    dpi: int = 150,
) -> None:
    """Stacked bar FEVD for the Food equation."""
    if food_name not in names:
        return
    i_eq = names.index(food_name)
    k = len(names)
    periods = min(max_periods, decomp.shape[1])
    cum = np.zeros(periods)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, periods + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, k))
    for j in range(k):
        contrib = decomp[i_eq, :periods, j]
        ax.bar(
            x,
            contrib,
            bottom=cum,
            label=f"Shock: {names[j]}",
            color=colors[j],
            width=0.85,
        )
        cum = cum + contrib
    ax.set_ylim(0, 1)
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("Share of forecast error variance")
    ax.set_title(f"FEVD for {food_name} (Cholesky)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi)
        print(f"[Saved figure] {save_path}")
    plt.close(fig)


def save_irf_figure(fig, out_dir: Path, stem: str) -> None:
    """Persist IRF plot from statsmodels (Figure, ndarray of axes, or fallback)."""
    if isinstance(fig, np.ndarray):
        for i, f in enumerate(fig.ravel()):
            if hasattr(f, "figure"):
                f.figure.savefig(out_dir / f"{stem}_{i}.png", dpi=150)
                plt.close(f.figure)
    elif hasattr(fig, "savefig"):
        fig.savefig(out_dir / f"{stem}.png", dpi=150)
        plt.close(fig)
    else:
        plt.savefig(out_dir / f"{stem}.png", dpi=150)
        plt.close("all")
