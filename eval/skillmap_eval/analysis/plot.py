"""Matplotlib figures: per-axis correction curves + correctness trajectory."""

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
    paths["correction_rate_total"] = _plot_correction_curve(
        report, out, axis="total", filename="correction_rate_total.png",
        title="Correction-rate decay (total)"
    )
    paths["correction_rate_preference"] = _plot_correction_curve(
        report, out, axis="preference", filename="correction_rate_preference.png",
        title="Correction-rate decay (preference axis)"
    )
    paths["correction_rate_correctness"] = _plot_correction_curve(
        report, out, axis="correctness", filename="correction_rate_correctness.png",
        title="Correction-rate decay (correctness axis)"
    )
    paths["preference_recovery_bar"] = _plot_preference_recovery(report, out)
    paths["generalization_comparison"] = _plot_generalization(report, out)
    paths["correctness_trajectory"] = _plot_correctness_trajectory(report, out)
    paths["preference_trajectory"] = _plot_preference_trajectory(report, out)
    return paths


def _plot_correction_curve(
    report: EvalReport, out: Path, axis: str, filename: str, title: str
) -> Path:
    series_attr = {
        "total": "rolling_mean_window_3_total",
        "preference": "rolling_mean_window_3_preference",
        "correctness": "rolling_mean_window_3_correctness",
    }[axis]
    fig, ax = plt.subplots(figsize=(8, 5))
    for curve in report.correction_curves:
        color = _COLORS.get(curve.condition_name, None)
        ax.plot(
            curve.task_indices,
            getattr(curve, series_attr),
            label=curve.condition_name,
            color=color,
            linewidth=2,
        )
    ax.set_xlabel("task index in stream")
    ax.set_ylabel(f"{axis} corrections per task (rolling mean, w=3)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out / filename
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
    ax.set_title("Preference recovery (preference-axis skills only)")
    for i, v in enumerate(rates):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    path = out / "preference_recovery_bar.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_generalization(report: EvalReport, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r.condition_name for r in report.generalization]
    pref = [r.held_out_avg_preference_corrections for r in report.generalization]
    corr = [r.held_out_avg_correctness_corrections for r in report.generalization]
    x = list(range(len(names)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], pref, width=width, label="preference", color="#3366cc")
    ax.bar([i + width / 2 for i in x], corr, width=width, label="correctness", color="#cc6633")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("held-out avg corrections / task")
    ax.set_title("Generalization (held-out per-axis correction rate)")
    ax.legend()
    path = out / "generalization_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_correctness_trajectory(report: EvalReport, out: Path) -> Path:
    """Bar chart: early vs late vs held-out average pass rate, per condition."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r.condition_name for r in report.correctness_trajectory]
    early = [r.avg_pass_rate_early for r in report.correctness_trajectory]
    late = [r.avg_pass_rate_late for r in report.correctness_trajectory]
    held = [r.avg_pass_rate_held_out for r in report.correctness_trajectory]
    x = list(range(len(names)))
    width = 0.27
    ax.bar([i - width for i in x], early, width=width, label="early stream", color="#cccccc")
    ax.bar(x, late, width=width, label="late stream", color="#cc3333")
    ax.bar([i + width for i in x], held, width=width, label="held-out", color="#666666")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("avg test-case pass rate")
    ax.set_title("Correctness trajectory (early vs late vs held-out)")
    ax.legend()
    path = out / "correctness_trajectory.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_preference_trajectory(report: EvalReport, out: Path) -> Path:
    """Bar chart: early vs late vs held-out average preference acceptance rate."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r.condition_name for r in report.preference_trajectory]
    early = [r.avg_acceptance_rate_early for r in report.preference_trajectory]
    late = [r.avg_acceptance_rate_late for r in report.preference_trajectory]
    held = [r.avg_acceptance_rate_held_out for r in report.preference_trajectory]
    x = list(range(len(names)))
    width = 0.27
    ax.bar([i - width for i in x], early, width=width, label="early stream", color="#cccccc")
    ax.bar(x, late, width=width, label="late stream", color="#3366cc")
    ax.bar([i + width for i in x], held, width=width, label="held-out", color="#666666")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("avg preference acceptance rate")
    ax.set_title("Preference trajectory (early vs late vs held-out)")
    ax.legend()
    path = out / "preference_trajectory.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
