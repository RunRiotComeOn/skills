"""Matplotlib figures: 4 PNGs per spec §7."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skillmap_eval.types import EvalReport


_COLORS = {
    "stateless": "#888888",
    "declarative_memory": "#3366cc",
    "skillmap": "#cc3333",
}


def make_all_figures(report: EvalReport, out_dir: Path | str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["correction_rate_curve"] = _plot_correction_curve(report, out)
    paths["preference_recovery_bar"] = _plot_preference_recovery(report, out)
    paths["generalization_comparison"] = _plot_generalization(report, out)
    paths["correctness_sanity"] = _plot_correctness_sanity(report, out)
    return paths


def _plot_correction_curve(report: EvalReport, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for curve in report.correction_curves:
        color = _COLORS.get(curve.condition_name, None)
        ax.plot(
            curve.task_indices,
            curve.rolling_mean_window_3,
            label=curve.condition_name,
            color=color,
            linewidth=2,
        )
    ax.set_xlabel("task index in stream")
    ax.set_ylabel("corrections per task (rolling mean, w=3)")
    ax.set_title("Correction-rate decay across the stream")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out / "correction_rate_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_preference_recovery(report: EvalReport, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    names = [r.condition_name for r in report.preference_recovery]
    rates = [r.recovery_rate for r in report.preference_recovery]
    colors = [_COLORS.get(n, "#555555") for n in names]
    ax.bar(names, rates, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("preferences recovered (fraction)")
    ax.set_title("Preference recovery by condition")
    for i, v in enumerate(rates):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    path = out / "preference_recovery_bar.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_generalization(report: EvalReport, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    names = [r.condition_name for r in report.generalization]
    avgs = [r.held_out_avg_correction_count for r in report.generalization]
    colors = [_COLORS.get(n, "#555555") for n in names]
    ax.bar(names, avgs, color=colors)
    ax.set_ylabel("held-out avg corrections / task")
    ax.set_title("Generalization (held-out correction rate)")
    path = out / "generalization_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_correctness_sanity(report: EvalReport, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    names = [r.condition_name for r in report.correctness_sanity]
    rates = [r.avg_test_pass_rate for r in report.correctness_sanity]
    colors = [_COLORS.get(n, "#555555") for n in names]
    ax.bar(names, rates, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("avg test pass rate")
    ax.set_title("Correctness sanity (should be approximately level)")
    path = out / "correctness_sanity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
