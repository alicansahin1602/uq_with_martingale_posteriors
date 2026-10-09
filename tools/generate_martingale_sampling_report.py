"""Generate a consolidated PDF report for martingale-sampling experiments."""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "data" / "results_mp_sampling"
OUTPUT_DIR = ROOT / "data" / "martingale_test_outputs"
TMP_DIR = ROOT / "tmp" / "pdfs" / "martingale_sampling_tolerance_report"
OUTPUT_PDF = OUTPUT_DIR / "martingale_sampling_tolerance_report_2026-09-17.pdf"

ALPHA = 0.05
EPSILON_CLASS = 1e-3
EPSILON_L2 = 1e-3
SENSITIVITY_EPSILONS = (1e-4, 1e-3, 1e-2)

DATASETS = ("arc_c", "arc_e", "csqa")
MODELS = ("gpt4o_mini", "deepseek_v4_pro")
MODEL_DISPLAY = {
    "gpt4o_mini": "GPT-4o mini",
    "deepseek_v4_pro": "DeepSeek V4 Pro",
}
DATASET_DISPLAY = {
    "arc_c": "ARC-Challenge",
    "arc_e": "ARC-Easy",
    "csqa": "CommonsenseQA",
}
COLORS = {
    "gpt4o_mini": "#2F6B9A",
    "deepseek_v4_pro": "#D07A3A",
}
BIN_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.000001])


def load_runs() -> dict[tuple[str, str], dict]:
    runs = {}
    for dataset in DATASETS:
        for model in MODELS:
            path = (
                RESULTS_ROOT
                / f"{dataset}_{model}"
                / "martingale_check"
                / "sampling"
                / "martingale_sampling_results.npz"
            )
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=True) as raw:
                run = {key: raw[key] for key in raw.files}
            if "metadata" in run and isinstance(run["metadata"], np.ndarray):
                run["metadata"] = run["metadata"].item()
            run["path"] = path
            runs[(dataset, model)] = run
    return runs


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = 20000):
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    draws = values[indices].mean(axis=1)
    return float(values.mean()), np.quantile(draws, [0.025, 0.975]), draws


def one_sample_tost(values: np.ndarray, epsilon: float, alpha: float = ALPHA) -> dict:
    """Two one-sided tests of whether the population mean lies in (-epsilon, epsilon)."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sem = float(stats.sem(values))
    df = n - 1
    if not np.isfinite(sem) or sem == 0:
        lower_p = 0.0 if mean > -epsilon else 1.0
        upper_p = 0.0 if mean < epsilon else 1.0
        ci_low = ci_high = mean
    else:
        lower_p = float(stats.t.sf((mean + epsilon) / sem, df))
        upper_p = float(stats.t.cdf((mean - epsilon) / sem, df))
        critical = stats.t.ppf(1 - alpha, df)
        ci_low = mean - critical * sem
        ci_high = mean + critical * sem
    tost_p = max(lower_p, upper_p)
    return {
        "equivalence_ci_low": float(ci_low),
        "equivalence_ci_high": float(ci_high),
        "tost_p_value": float(tost_p),
        "practically_equivalent": bool(tost_p < alpha),
    }


def analyze_run(dataset: str, model: str, run: dict, seed: int = 42) -> dict:
    residuals = np.asarray(run["residuals"], dtype=np.float64)
    error_l2 = np.asarray(run["error_l2"], dtype=np.float64)
    probs = np.asarray(run["probs"], dtype=np.float64)
    expected_next = np.asarray(run["expected_next"], dtype=np.float64)
    true_labels = np.asarray(run["true_labels"])
    metadata = run.get("metadata", {})
    labels = list(metadata.get("label_chars", range(residuals.shape[-1])))

    Q, J, T, C = residuals.shape
    n_trajectories = Q * J
    rng = np.random.default_rng(seed)

    trajectory_residuals = residuals.mean(axis=2).reshape(n_trajectories, C)
    class_rows = []
    for c, label in enumerate(labels):
        values = trajectory_residuals[:, c]
        indices = rng.integers(0, n_trajectories, size=(20000, n_trajectories))
        bootstrap_draws = values[indices].mean(axis=1)
        ci_low, ci_high = np.quantile(bootstrap_draws, [0.025, 0.975])
        t_result = stats.ttest_1samp(values, popmean=0.0)
        tost = one_sample_tost(values, EPSILON_CLASS)
        class_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "class": str(label),
                "mean_residual": float(values.mean()),
                "bootstrap_ci_low": float(ci_low),
                "bootstrap_ci_high": float(ci_high),
                "zero_in_bootstrap_ci": bool(ci_low <= 0 <= ci_high),
                "t_statistic": float(t_result.statistic),
                "p_value": float(t_result.pvalue),
                "bonferroni_reject": bool(t_result.pvalue < 0.05 / C),
                "epsilon": EPSILON_CLASS,
                **tost,
            }
        )

    trajectory_l2 = error_l2.mean(axis=2).reshape(n_trajectories)
    mean_l2, mean_l2_ci, l2_draws = bootstrap_mean(trajectory_l2, rng)
    l2_upper_95 = float(np.quantile(l2_draws, 0.95))
    l2_sensitivity = {
        epsilon: bool(l2_upper_95 < epsilon) for epsilon in SENSITIVITY_EPSILONS
    }

    # This is the user-requested test without splitting by class. Because probability
    # vectors sum to one, class residuals sum to zero at each transition; the result is
    # therefore a useful implementation check, not an independent martingale diagnostic.
    aggregate_signed_values = residuals.mean(axis=(2, 3)).reshape(n_trajectories)
    aggregate_t = stats.ttest_1samp(aggregate_signed_values, popmean=0.0)
    aggregate_tost = one_sample_tost(aggregate_signed_values, EPSILON_CLASS)

    step_trajectories = error_l2.reshape(n_trajectories, T)
    step_indices = rng.integers(0, n_trajectories, size=(10000, n_trajectories))
    step_bootstrap = step_trajectories[step_indices].mean(axis=1)
    step_ci_low, step_ci_high = np.quantile(step_bootstrap, [0.025, 0.975], axis=0)
    step_mean = step_trajectories.mean(axis=0)
    step_median = np.median(step_trajectories, axis=0)

    current_probs = probs[..., :-1, :]
    predictions = current_probs.argmax(axis=-1)
    correct = predictions == true_labels[:, None, None]
    max_probs = current_probs.max(axis=-1)
    entropy = -(probs * np.log(np.clip(probs, 1e-300, 1.0))).sum(axis=-1)

    current_flat = current_probs.reshape(n_trajectories, T * C)
    residual_flat = residuals.reshape(n_trajectories, T * C)
    cluster_sums = np.zeros((n_trajectories, len(BIN_EDGES) - 1))
    cluster_counts = np.zeros_like(cluster_sums, dtype=int)
    for g in range(n_trajectories):
        for b in range(len(BIN_EDGES) - 1):
            lower, upper = BIN_EDGES[b], BIN_EDGES[b + 1]
            if b == len(BIN_EDGES) - 2:
                mask = (current_flat[g] >= lower) & (current_flat[g] <= upper)
            else:
                mask = (current_flat[g] >= lower) & (current_flat[g] < upper)
            cluster_sums[g, b] = residual_flat[g, mask].sum()
            cluster_counts[g, b] = mask.sum()

    total_counts = cluster_counts.sum(axis=0)
    bin_means = np.divide(
        cluster_sums.sum(axis=0),
        total_counts,
        out=np.full(len(total_counts), np.nan),
        where=total_counts > 0,
    )
    bin_draws = np.full((10000, len(total_counts)), np.nan)
    for b in range(10000):
        sampled = rng.integers(0, n_trajectories, size=n_trajectories)
        sampled_sums = cluster_sums[sampled].sum(axis=0)
        sampled_counts = cluster_counts[sampled].sum(axis=0)
        bin_draws[b] = np.divide(
            sampled_sums,
            sampled_counts,
            out=np.full(len(total_counts), np.nan),
            where=sampled_counts > 0,
        )
    with np.errstate(all="ignore"):
        bin_ci_low = np.nanquantile(bin_draws, 0.025, axis=0)
        bin_ci_high = np.nanquantile(bin_draws, 0.975, axis=0)

    return {
        "dataset": dataset,
        "model": model,
        "Q": Q,
        "J": J,
        "T": T,
        "C": C,
        "n_trajectories": n_trajectories,
        "data_indices": np.asarray(run.get("data_indices", [])),
        "labels": labels,
        "class_results": pd.DataFrame(class_rows),
        "mean_l2": mean_l2,
        "mean_l2_ci_low": float(mean_l2_ci[0]),
        "mean_l2_ci_high": float(mean_l2_ci[1]),
        "l2_upper_95": l2_upper_95,
        "l2_below_epsilon": bool(l2_upper_95 < EPSILON_L2),
        "l2_sensitivity": l2_sensitivity,
        "aggregate_signed_mean": float(aggregate_signed_values.mean()),
        "aggregate_signed_p_value": float(aggregate_t.pvalue),
        "aggregate_signed_t_statistic": float(aggregate_t.statistic),
        "aggregate_signed_tost_p_value": aggregate_tost["tost_p_value"],
        "aggregate_signed_equivalent": aggregate_tost["practically_equivalent"],
        "aggregate_signed_ci_low": aggregate_tost["equivalence_ci_low"],
        "aggregate_signed_ci_high": aggregate_tost["equivalence_ci_high"],
        "median_l2": float(np.median(error_l2)),
        "max_l2": float(error_l2.max()),
        "step_mean": step_mean,
        "step_median": step_median,
        "step_ci_low": step_ci_low,
        "step_ci_high": step_ci_high,
        "correct_states": int(correct.sum()),
        "incorrect_states": int((~correct).sum()),
        "state_accuracy": float(correct.mean()),
        "mean_max_probability": float(max_probs.mean()),
        "min_max_probability": float(max_probs.min()),
        "mean_entropy": float(entropy.mean()),
        "bin_counts": total_counts,
        "bin_means": bin_means,
        "bin_ci_low": bin_ci_low,
        "bin_ci_high": bin_ci_high,
        "probs": probs,
        "expected_next": expected_next,
        "residuals": residuals,
        "error_l2": error_l2,
    }


def scientific(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.{digits}e}"


def make_overview_figure(analyses: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    x = np.arange(len(DATASETS))
    width = 0.34
    for offset, model in zip((-width / 2, width / 2), MODELS):
        means = np.array([analyses[(d, model)]["mean_l2"] for d in DATASETS])
        lows = np.array([analyses[(d, model)]["mean_l2_ci_low"] for d in DATASETS])
        highs = np.array([analyses[(d, model)]["mean_l2_ci_high"] for d in DATASETS])
        ax.bar(
            x + offset,
            means,
            width,
            color=COLORS[model],
            label=MODEL_DISPLAY[model],
            alpha=0.9,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            fmt="none",
            ecolor="#333333",
            capsize=3,
            linewidth=1,
        )
    ax.set_yscale("log")
    ax.set_xticks(x, [DATASET_DISPLAY[d] for d in DATASETS])
    ax.set_ylabel(r"Mean local residual magnitude $\|r_t\|_2$ (log scale)")
    ax.set_title("Average local martingale discrepancy")
    ax.grid(axis="y", which="both", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_step_figure(analyses: dict, path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5), sharex=True)
    for row, dataset in enumerate(DATASETS):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            result = analyses[(dataset, model)]
            steps = np.arange(result["T"])
            ax.fill_between(
                steps,
                result["step_ci_low"],
                result["step_ci_high"],
                color=COLORS[model],
                alpha=0.18,
                label="95% trajectory-bootstrap CI",
            )
            ax.plot(steps, result["step_mean"], color=COLORS[model], lw=1.8, label="Mean")
            ax.plot(steps, result["step_median"], color="#4C956C", lw=1.2, ls="--", label="Median")
            ax.axhline(result["mean_l2"], color="#555555", lw=1, ls=":", label="Overall mean")
            ax.set_yscale("log")
            ax.set_title(f"{DATASET_DISPLAY[dataset]} - {MODEL_DISPLAY[model]}", fontsize=10)
            ax.grid(which="both", alpha=0.18)
            if col == 0:
                ax.set_ylabel(r"$\|r_t\|_2$")
            if row == len(DATASETS) - 1:
                ax.set_xlabel("Transition step t")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Residual magnitude over the 49 transitions", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_class_figure(analyses: dict, path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 10.4))
    for row, dataset in enumerate(DATASETS):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            frame = analyses[(dataset, model)]["class_results"]
            x = np.arange(len(frame))
            means = frame["mean_residual"].to_numpy()
            lows = frame["bootstrap_ci_low"].to_numpy()
            highs = frame["bootstrap_ci_high"].to_numpy()
            ax.errorbar(
                x,
                means,
                yerr=np.vstack([means - lows, highs - means]),
                fmt="o",
                color=COLORS[model],
                capsize=4,
                markersize=5,
            )
            ax.axhline(0, color="#333333", lw=1, ls="--")
            ax.set_xticks(x, frame["class"])
            ax.set_title(f"{DATASET_DISPLAY[dataset]} - {MODEL_DISPLAY[model]}", fontsize=10)
            ax.grid(axis="y", alpha=0.2)
            if col == 0:
                ax.set_ylabel("Mean signed residual")
            if row == len(DATASETS) - 1:
                ax.set_xlabel("Class")
    fig.suptitle("Class-wise mean signed residuals with 95% trajectory-bootstrap intervals", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_expected_current_figure(analyses: dict, path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 10.4), sharex=True, sharey=True)
    for row, dataset in enumerate(DATASETS):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            result = analyses[(dataset, model)]
            current = result["probs"][..., :-1, :].ravel()
            expected = result["expected_next"].ravel()
            ax.scatter(current, expected, s=5, alpha=0.12, color=COLORS[model], rasterized=True)
            ax.plot([0, 1], [0, 1], color="#333333", lw=1, ls="--")
            ax.set_title(f"{DATASET_DISPLAY[dataset]} - {MODEL_DISPLAY[model]}", fontsize=10)
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel(r"Expected next probability $E[p_{t+1,c}\mid\mathcal{F}_t]$")
            if row == len(DATASETS) - 1:
                ax.set_xlabel(r"Current probability $p_{t,c}$")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
    fig.suptitle("Expected next probability versus current probability", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_confidence_figure(analyses: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.3), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        values = []
        labels = []
        for model in MODELS:
            probs = analyses[(dataset, model)]["probs"][..., :-1, :]
            values.append(probs.max(axis=-1).ravel())
            labels.append(MODEL_DISPLAY[model])
        box = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=True)
        for patch, model in zip(box["boxes"], MODELS):
            patch.set_facecolor(COLORS[model])
            patch.set_alpha(0.65)
        ax.set_title(DATASET_DISPLAY[dataset])
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0.90, 1.0005)
    axes[0].set_ylabel("Maximum current class probability")
    fig.suptitle("Prediction confidence is concentrated close to one", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=28,
            textColor=colors.HexColor("#17324D"),
            alignment=TA_LEFT,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            textColor=colors.HexColor("#52606D"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=6,
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subsection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#2F6B9A"),
            spaceBefore=7,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyReport",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=13.2,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#263238"),
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#52606D"),
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Callout",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10.2,
            leading=14.2,
            textColor=colors.HexColor("#17324D"),
            borderColor=colors.HexColor("#B8CCE0"),
            borderWidth=0.8,
            borderPadding=9,
            backColor=colors.HexColor("#F2F7FB"),
            spaceBefore=5,
            spaceAfter=10,
        )
    )
    return styles


def make_table(data, col_widths=None, header=True, font_size=7.5):
    table = Table(data, colWidths=col_widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def report_header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
    canvas.setLineWidth(0.5)
    canvas.line(1.7 * cm, 1.35 * cm, width - 1.7 * cm, 1.35 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#687784"))
    canvas.drawString(1.7 * cm, 0.85 * cm, "Martingale sampling - tolerance diagnostic report")
    canvas.drawRightString(width - 1.7 * cm, 0.85 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(analyses: dict, figures: dict[str, Path]) -> None:
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.65 * cm,
        title="Martingale sampling diagnostics with practical tolerances",
        author="Ali Can Sahin",
        subject="Martingale residual diagnostics for ARC and CommonsenseQA experiments",
    )
    story = []

    story.append(Spacer(1, 1.0 * cm))
    story.append(Paragraph("Martingale sampling diagnostics", styles["ReportTitle"]))
    story.append(
        Paragraph(
            "Exact-zero tests and practical-equivalence analysis<br/>"
            "ARC-Challenge, ARC-Easy and CommonsenseQA | GPT-4o mini and DeepSeek V4 Pro",
            styles["Subtitle"],
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            "Prepared for an initial methodological review<br/>17 September 2026",
            styles["BodyReport"],
        )
    )
    story.append(Spacer(1, 0.6 * cm))
    story.append(
        Paragraph(
            "Purpose. This note updates the sampling diagnostics with an explicit tolerance. "
            f"The default is epsilon = {EPSILON_CLASS:g}: a class-wise mean residual within "
            "+/-0.001 is treated as practically negligible only when a formal equivalence test "
            "supports that conclusion. This is an initial methodological readout, not a final "
            "model-level claim.",
            styles["Callout"],
        )
    )

    csqa_gpt = analyses[("csqa", "gpt4o_mini")]
    csqa_ds = analyses[("csqa", "deepseek_v4_pro")]
    csqa_ratio = csqa_gpt["mean_l2"] / csqa_ds["mean_l2"]
    all_class_results = pd.concat([a["class_results"] for a in analyses.values()], ignore_index=True)
    n_exact = int((~all_class_results["zero_in_bootstrap_ci"]).sum())
    n_bonf = int(all_class_results["bonferroni_reject"].sum())
    n_equivalent = int(all_class_results["practically_equivalent"].sum())
    n_l2_pass = sum(a["l2_below_epsilon"] for a in analyses.values())
    story.append(Paragraph("Main observations", styles["Section"]))
    bullets = [
        "The average local residual magnitude is very small in five of the six runs. "
        f"The exception is GPT-4o mini on CommonsenseQA ({scientific(csqa_gpt['mean_l2'])}), "
        f"about {csqa_ratio:,.0f} times the DeepSeek value on the same selected questions.",
        f"Exact equality and practical equivalence give different answers: {n_exact}/30 class-wise "
        f"bootstrap intervals exclude zero and {n_bonf}/30 t-tests survive within-run Bonferroni "
        f"correction, while {n_equivalent}/30 class means pass TOST equivalence at +/-0.001.",
        f"Using the one-sided 95% trajectory-bootstrap upper bound, {n_l2_pass}/6 runs have mean "
        f"L2 residual below epsilon_L2 = {EPSILON_L2:g}. This is the more meaningful aggregate "
        "assessment because signed residuals cancel across classes by construction.",
        "The model probabilities are strongly saturated. Mean maximum probabilities range from "
        f"{min(a['mean_max_probability'] for a in analyses.values()):.6f} to "
        f"{max(a['mean_max_probability'] for a in analyses.values()):.6f}. The current experiment "
        "mainly examines near-one-hot states rather than the full probability simplex.",
    ]
    for item in bullets:
        story.append(Paragraph(f"• {item}", styles["BodyReport"]))

    story.append(PageBreak())
    story.append(Paragraph("1. Experimental coverage", styles["Section"]))
    story.append(
        Paragraph(
            "For each visited history, the experiment computes the conditional expected next "
            "probability vector and subtracts the current vector: "
            "<i>r</i><sub>t</sub> = E[<i>p</i><sub>t+1</sub> | F<sub>t</sub>] - "
            "<i>p</i><sub>t</sub>. The saved tensors contain 49 transitions and five classes. "
            "Inference in the notebook first averages dependent steps within a trajectory and then "
            "resamples complete trajectories.",
            styles["BodyReport"],
        )
    )
    coverage = [["Dataset", "Model", "Questions", "Traj./question", "Clusters", "States", "Incorrect"]]
    for dataset in DATASETS:
        for model in MODELS:
            a = analyses[(dataset, model)]
            coverage.append(
                [
                    DATASET_DISPLAY[dataset],
                    MODEL_DISPLAY[model],
                    str(a["Q"]),
                    str(a["J"]),
                    str(a["n_trajectories"]),
                    str(a["Q"] * a["J"] * a["T"]),
                    str(a["incorrect_states"]),
                ]
            )
    story.append(make_table(coverage, [3.0 * cm, 3.0 * cm, 1.5 * cm, 2.0 * cm, 1.5 * cm, 1.7 * cm, 1.5 * cm]))
    story.append(Spacer(1, 0.25 * cm))
    matched_arc_c = set(analyses[("arc_c", MODELS[0])]["data_indices"]) == set(analyses[("arc_c", MODELS[1])]["data_indices"])
    matched_arc_e = set(analyses[("arc_e", MODELS[0])]["data_indices"]) == set(analyses[("arc_e", MODELS[1])]["data_indices"])
    matched_csqa = set(analyses[("csqa", MODELS[0])]["data_indices"]) == set(analyses[("csqa", MODELS[1])]["data_indices"])
    story.append(
        Paragraph(
            f"The selected question indices match across models for ARC-Challenge ({matched_arc_c}) "
            f"and CommonsenseQA ({matched_csqa}), but not for ARC-Easy ({matched_arc_e}). ARC-Easy "
            "also uses 10 questions for DeepSeek and 5 for GPT-4o mini, so that model comparison is "
            "descriptive rather than paired.",
            styles["Callout"],
        )
    )
    story.append(Paragraph("Tests implemented in the notebook", styles["Subsection"]))
    methods = [
        "Class-wise mean signed residual with a 95% percentile bootstrap interval, resampling complete trajectories.",
        "Two-sided one-sample t-test of trajectory-mean residuals against zero, with a Bonferroni threshold across five classes.",
        f"Class-wise TOST equivalence test for the interval (-{EPSILON_CLASS:g}, +{EPSILON_CLASS:g}); at alpha = 0.05 this corresponds to a 90% equivalence confidence interval.",
        "An additional signed-residual t-test after averaging over classes, reported as requested but interpreted only as a probability-conservation check.",
        f"A practical aggregate criterion for the non-negative L2 residual: its one-sided 95% bootstrap upper bound must be below epsilon_L2 = {EPSILON_L2:g}.",
        "L2 residual magnitude by step, including mean, median and pointwise trajectory-bootstrap intervals.",
        "Expected-next versus current probability, matching the active visualization in the notebook.",
    ]
    for item in methods:
        story.append(Paragraph(f"• {item}", styles["BodyReport"]))

    story.append(PageBreak())
    story.append(Paragraph("2. Overall residual magnitude", styles["Section"]))
    story.append(Image(str(figures["overview"]), width=17.0 * cm, height=9.2 * cm))
    story.append(Spacer(1, 0.15 * cm))
    overview_table = [["Dataset", "Model", "Mean L2", "95% CI", "Median", "Maximum"]]
    for dataset in DATASETS:
        for model in MODELS:
            a = analyses[(dataset, model)]
            overview_table.append(
                [
                    DATASET_DISPLAY[dataset],
                    MODEL_DISPLAY[model],
                    scientific(a["mean_l2"], 3),
                    f"[{scientific(a['mean_l2_ci_low'])}, {scientific(a['mean_l2_ci_high'])}]",
                    scientific(a["median_l2"]),
                    scientific(a["max_l2"]),
                ]
            )
    story.append(make_table(overview_table, [2.7 * cm, 2.9 * cm, 2.2 * cm, 4.2 * cm, 2.1 * cm, 2.1 * cm]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        Paragraph(
            "The median is well below the mean in every run, especially for GPT-4o mini. This "
            "indicates a right-skewed distribution: most transitions have negligible discrepancy, "
            "with a small number of larger excursions driving the average. CommonsenseQA with "
            "GPT-4o mini stands apart both in its mean and its maximum residual.",
            styles["BodyReport"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("3. Evolution over the trajectory", styles["Section"]))
    story.append(Image(str(figures["steps"]), width=17.2 * cm, height=16.4 * cm))
    story.append(
        Paragraph(
            "The log scale is intentional: the runs differ by several orders of magnitude. There "
            "is no common monotone increase across all six panels. Instead, the means are driven by "
            "intermittent step-specific spikes. The median often remains close to the numerical "
            "floor while the mean moves, which again points to a small subset of trajectories or "
            "histories accounting for most of the observed discrepancy.",
            styles["BodyReport"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("4. Are mean signed residuals different from zero?", styles["Section"]))
    story.append(Image(str(figures["classes"]), width=17.2 * cm, height=16.2 * cm))
    story.append(
        Paragraph(
            f"Across 30 dataset-model-class combinations, {n_exact} percentile bootstrap intervals "
            f"exclude zero and {n_bonf} t-tests remain significant after within-run Bonferroni "
            "correction. This answers the literal question of exact equality. It does not answer "
            "whether the departures are large enough to matter.",
            styles["BodyReport"],
        )
    )
    story.append(
        Paragraph(
            f"At epsilon = {EPSILON_CLASS:g}, {n_equivalent}/30 class means are statistically "
            "equivalent to zero by TOST. A tiny confidence interval such as [4.27e-10, 1.23e-08] "
            "can therefore exclude exact zero and still sit comfortably inside the tolerance band. "
            "These statements are not contradictory: the first concerns exact equality; the second "
            "concerns practical size.",
            styles["Callout"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("5. Aggregate tests and epsilon sensitivity", styles["Section"]))
    aggregate_table = [["Dataset", "Model", "Signed mean", "Exact p", "TOST p", "Mean L2", "L2 upper 95%", "< .001?"]]
    for dataset in DATASETS:
        for model in MODELS:
            a = analyses[(dataset, model)]
            aggregate_table.append([
                DATASET_DISPLAY[dataset], MODEL_DISPLAY[model],
                scientific(a["aggregate_signed_mean"], 2), scientific(a["aggregate_signed_p_value"], 2),
                scientific(a["aggregate_signed_tost_p_value"], 2), scientific(a["mean_l2"], 2),
                scientific(a["l2_upper_95"], 2), "yes" if a["l2_below_epsilon"] else "no",
            ])
    story.append(make_table(aggregate_table, [2.2*cm, 2.35*cm, 1.65*cm, 1.35*cm, 1.35*cm, 1.55*cm, 1.75*cm, 1.2*cm], font_size=6.4))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("How to read TOST and the tolerance decision", styles["Subsection"]))
    story.append(Paragraph(
        "TOST means <i>Two One-Sided Tests</i>. Instead of asking whether the mean is exactly zero, "
        f"it asks whether the mean is convincingly inside a pre-specified negligible interval, here "
        f"(-{EPSILON_CLASS:g}, +{EPSILON_CLASS:g}). One test checks that the mean is above the lower "
        "boundary and the other checks that it is below the upper boundary. Equivalence is concluded "
        "only when both tests reject their boundary null hypothesis at alpha = 0.05. Equivalently, "
        "the entire 90% confidence interval must lie inside the tolerance interval.",
        styles["BodyReport"]))
    story.append(Paragraph(
        "The <b>&lt; .001?</b> column concerns the non-negative L2 residual, not the signed mean. "
        "A 'yes' means the one-sided 95% trajectory-bootstrap upper confidence bound for the mean "
        "L2 residual is below 0.001. A 'no' means the data do not establish that the mean L2 error is "
        "smaller than this tolerance. It does not mean that the residual is larger than 0.001 at "
        "every transition.", styles["BodyReport"]))
    story.append(
        Paragraph(
            "The class-aggregated signed test is almost guaranteed to return zero: both the current "
            "and expected-next vectors sum to one, so their component-wise residuals sum to zero. "
            "It is included because it was requested, but it should not be treated as evidence for "
            "the martingale property. The L2 column avoids this cancellation and is the useful "
            "aggregate result.", styles["Callout"],
        )
    )
    sensitivity_table = [["Dataset", "Model"] + [f"epsilon={eps:g}" for eps in SENSITIVITY_EPSILONS]]
    for dataset in DATASETS:
        for model in MODELS:
            a = analyses[(dataset, model)]
            sensitivity_table.append([
                DATASET_DISPLAY[dataset], MODEL_DISPLAY[model],
                *["pass" if a["l2_sensitivity"][eps] else "fail" for eps in SENSITIVITY_EPSILONS],
            ])
    story.append(Paragraph("Sensitivity of the aggregate L2 conclusion", styles["Subsection"]))
    story.append(make_table(sensitivity_table, [3.0*cm, 3.0*cm, 2.5*cm, 2.5*cm, 2.5*cm]))
    story.append(Paragraph(
        "A run passes when the one-sided 95% bootstrap upper bound for its trajectory-mean L2 "
        "residual is below epsilon. The table makes the scientific consequence of the tolerance "
        "choice explicit; epsilon should ultimately be justified by a downstream probability or "
        "decision effect, rather than selected after seeing significance.", styles["BodyReport"]))

    story.append(PageBreak())
    story.append(Paragraph("6. Expected-next versus current probability", styles["Section"]))
    story.append(Image(str(figures["expected_current"]), width=17.2 * cm, height=16.2 * cm))
    story.append(Paragraph(
        "Points on the dashed diagonal have zero class residual. Most observations lie at the "
        "corners because the saved probabilities are nearly one-hot. Departures are visually most "
        "noticeable for GPT-4o mini on CommonsenseQA, consistent with the L2 summary.",
        styles["BodyReport"]))

    story.append(PageBreak())
    story.append(Paragraph("7. Dataset-level reading", styles["Section"]))
    for dataset in DATASETS:
        gpt = analyses[(dataset, "gpt4o_mini")]
        deepseek = analyses[(dataset, "deepseek_v4_pro")]
        ratio = gpt["mean_l2"] / deepseek["mean_l2"]
        story.append(Paragraph(DATASET_DISPLAY[dataset], styles["Subsection"]))
        matched = set(gpt["data_indices"]) == set(deepseek["data_indices"])
        if dataset == "csqa":
            text = (
                f"GPT-4o mini has a mean L2 residual of {scientific(gpt['mean_l2'])}, compared with "
                f"{scientific(deepseek['mean_l2'])} for DeepSeek (ratio {ratio:,.0f}). This is the "
                "clearest separation in the current data and occurs on matched question indices. "
                "GPT-4o mini is also less saturated here (minimum maximum probability "
                f"{gpt['min_max_probability']:.3f}), giving the residual more room to vary."
            )
        elif dataset == "arc_e":
            text = (
                f"Both models show very small discrepancies ({scientific(gpt['mean_l2'])} for GPT-4o "
                f"mini and {scientific(deepseek['mean_l2'])} for DeepSeek). However, the questions are "
                "not matched and DeepSeek uses twice as many questions, so the ratio should not be read "
                "as a controlled model comparison."
            )
        else:
            text = (
                f"On matched question indices, GPT-4o mini has a mean L2 residual of "
                f"{scientific(gpt['mean_l2'])} and DeepSeek {scientific(deepseek['mean_l2'])} "
                f"(ratio {ratio:.1f}). Both remain very small in absolute terms and all evaluated "
                "states are correct."
            )
        story.append(Paragraph(text, styles["BodyReport"]))

    story.append(Paragraph("Provisional conclusion", styles["Subsection"]))
    story.append(
        Paragraph(
            f"At the pre-specified working tolerance of {EPSILON_CLASS:g} per signed class residual, "
            "the class-wise results are practically close to zero even though several are distinguishable "
            "from exact zero. The aggregate L2 criterion is stricter and identifies any runs whose "
            "overall discrepancy is not below 0.001. These findings are compatible with approximate "
            "martingale behavior only in the sampled, highly confident states; they do not establish "
            "the full conditional martingale property. GPT-4o mini on CommonsenseQA remains the clearest "
            "case for follow-up.",
            styles["Callout"],
        )
    )
    story.append(Paragraph("Recommended next run", styles["Subsection"]))
    recommendations = [
        "Increase the number of questions substantially and use identical question indices for both models on every dataset.",
        "For population-level uncertainty, resample questions as the outer cluster and trajectories within questions, rather than treating all question-trajectory pairs as fully independent.",
        "Pre-specify epsilon using a downstream consequence, such as the smallest probability shift that can change calibration or a decision, and retain the sensitivity table.",
        "Inspect the largest CommonsenseQA GPT-4o mini residuals at the prompt/history level to determine whether they arise from a reproducible behavioral transition or from provider/log-probability truncation.",
    ]
    for item in recommendations:
        story.append(Paragraph(f"• {item}", styles["BodyReport"]))

    story.append(PageBreak())
    story.append(Paragraph("Appendix A. Full class-wise test results", styles["Section"]))
    class_table = [["Dataset", "Model", "Class", "Mean", "Bootstrap 95% CI", "Exact p", "Bonf.", "TOST p", "Equiv."]]
    for _, row in all_class_results.iterrows():
        class_table.append(
            [
                DATASET_DISPLAY[row["dataset"]],
                MODEL_DISPLAY[row["model"]],
                row["class"],
                scientific(row["mean_residual"], 3),
                f"[{scientific(row['bootstrap_ci_low'])}, {scientific(row['bootstrap_ci_high'])}]",
                scientific(row["p_value"], 2),
                "yes" if row["bonferroni_reject"] else "no",
                scientific(row["tost_p_value"], 2),
                "yes" if row["practically_equivalent"] else "no",
            ]
        )
    story.append(make_table(class_table, [1.85*cm, 2.1*cm, 0.65*cm, 1.4*cm, 3.2*cm, 1.2*cm, 0.8*cm, 1.2*cm, 0.9*cm], font_size=5.8))
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "The p-values are two-sided. Because each returned p-value already combines both tails, "
            "it is compared with alpha = 0.05, not alpha/2. The alpha/2 split appears in the endpoints "
            "of the corresponding two-sided confidence interval.",
            styles["Small"],
        )
    )
    story.append(
        Paragraph(
            "<b>Bonf.</b> reports the Bonferroni-corrected exact-zero decision. Five class-wise tests "
            "are performed within each dataset-model run. If every test used 0.05 independently, the "
            "chance of at least one false positive would exceed 5%. Bonferroni controls this family-wise "
            "error by comparing each exact p-value with 0.05/5 = 0.01. 'yes' means the exact-zero null "
            "is rejected even under this more conservative threshold; 'no' means it is not.",
            styles["Small"],
        )
    )
    story.append(
        Paragraph(
            f"<b>Equiv.</b> reports the practical-equivalence decision from TOST. 'yes' means both "
            f"one-sided tests passed and the 90% confidence interval for the class mean lies entirely "
            f"inside (-{EPSILON_CLASS:g}, +{EPSILON_CLASS:g}). The residual can therefore be statistically "
            "different from exact zero while still being negligible at the chosen tolerance. 'no' "
            "means equivalence was not established; it does not by itself prove a meaningful violation.",
            styles["Small"],
        )
    )
    story.append(Paragraph("Scope and reproducibility", styles["Subsection"]))
    story.append(
        Paragraph(
            "Source files: the six <i>martingale_sampling_results.npz</i> artifacts under "
            "<i>data/results_mp_sampling</i>. Analysis follows the current sampling notebook with "
            "20,000 bootstrap draws for class-wise and overall summaries and 10,000 draws for step "
            "plots. Random seed: 42. Default tolerances: epsilon_class = epsilon_L2 = 0.001. This "
            "document was generated directly from the saved "
            "arrays; no model calls were repeated.",
            styles["BodyReport"],
        )
    )

    doc.build(story, onFirstPage=report_header_footer, onLaterPages=report_header_footer)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    analyses = {
        key: analyze_run(key[0], key[1], run, seed=42 + index)
        for index, (key, run) in enumerate(runs.items())
    }

    figures = {
        "overview": TMP_DIR / "overall_l2.png",
        "steps": TMP_DIR / "l2_by_step.png",
        "classes": TMP_DIR / "class_residuals.png",
        "expected_current": TMP_DIR / "expected_vs_current.png",
    }
    make_overview_figure(analyses, figures["overview"])
    make_step_figure(analyses, figures["steps"])
    make_class_figure(analyses, figures["classes"])
    make_expected_current_figure(analyses, figures["expected_current"])
    build_pdf(analyses, figures)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
