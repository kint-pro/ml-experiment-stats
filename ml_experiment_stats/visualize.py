import json
from pathlib import Path

import numpy as np

from ml_experiment_stats.statistics import significance_marker


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError("matplotlib required: pip install kint-stats[plots]") from None


def _import_pyarrow():
    try:
        import pyarrow.parquet as pq

        return pq
    except ImportError:
        raise ImportError("pyarrow required: pip install kint-stats[parquet]") from None


COLORS = {
    0: "#2d3436",
    1: "#e17055",
    2: "#0984e3",
    3: "#00b894",
    4: "#6c5ce7",
}


def load_summary(results_dir: str) -> dict:
    path = Path(results_dir) / "summary.json"
    with open(path) as f:
        return json.load(f)


def load_statistics(results_dir: str) -> dict | None:
    path = Path(results_dir) / "statistics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_metrics(results_dir: str) -> list[dict]:
    pq = _import_pyarrow()
    path = Path(results_dir) / "metrics.parquet"
    table = pq.read_table(path)
    return table.to_pylist()


def _get_pairwise(statistics: dict | None, metric: str) -> list[dict]:
    if not statistics or metric not in statistics.get("metrics", {}):
        return []
    metric_data = statistics["metrics"][metric]
    if isinstance(metric_data, list):
        return metric_data
    return metric_data.get("pairwise", [])


def _build_significance_lookup(
    statistics: dict | None,
    metric: str,
) -> dict[tuple[str, str], float]:
    comparisons = _get_pairwise(statistics, metric)
    lookup = {}
    for comp in comparisons:
        a, b = comp["method_a"], comp["method_b"]
        lookup[(a, b)] = comp["corrected_p_value"]
        lookup[(b, a)] = comp["corrected_p_value"]
    return lookup


def plot_comparison_bar(
    summary: dict,
    metric: str,
    output_dir: str,
    formats: list[str],
    statistics: dict | None = None,
):
    plt = _import_matplotlib()
    methods = [m for m in summary if metric in summary[m]]
    if not methods:
        return
    means = [summary[m][metric]["mean"] for m in methods]
    stds = [summary[m][metric]["std"] for m in methods]

    sig_lookup = _build_significance_lookup(statistics, metric)
    alpha = statistics["config"]["alpha"] if statistics and "config" in statistics else 0.05
    significant_pairs = [
        (i, j)
        for i in range(len(methods))
        for j in range(i + 1, len(methods))
        if sig_lookup.get((methods[i], methods[j]), 1.0) < alpha
    ]

    fig, ax = plt.subplots(figsize=(6, 4 + 0.3 * len(significant_pairs)))
    x = np.arange(len(methods))
    colors = [COLORS.get(i, COLORS[0]) for i in range(len(methods))]
    ax.bar(x, means, yerr=stds, color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by Method")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if significant_pairs:
        y_max = max(m + s for m, s in zip(means, stds))
        y_step = (max(means) - min(means)) * 0.1 or y_max * 0.05
        for k, (i, j) in enumerate(significant_pairs):
            p = sig_lookup[(methods[i], methods[j])]
            y_bar = y_max + y_step * (k + 1)
            y_lo = y_bar - y_step * 0.3
            ax.plot(
                [x[i], x[i], x[j], x[j]],
                [y_lo, y_bar, y_bar, y_lo],
                color="black",
                linewidth=0.8,
            )
            ax.text(
                (x[i] + x[j]) / 2,
                y_bar,
                significance_marker(p),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()

    for fmt in formats:
        fig.savefig(Path(output_dir) / f"comparison_{metric}.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_seed(rows: list[dict], metric: str, output_dir: str, formats: list[str]):
    plt = _import_matplotlib()
    methods = sorted(set(r["method"] for r in rows))
    seeds = sorted(set(int(r["seed"]) for r in rows))

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, method in enumerate(methods):
        values = [
            float(r[metric]) for r in rows if r["method"] == method and r.get(metric) is not None
        ]
        color = COLORS.get(i, COLORS[0])
        ax.plot(seeds[: len(values)], values, "o-", label=method, color=color, markersize=4)

    ax.set_xlabel("Seed")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Across Seeds")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    for fmt in formats:
        fig.savefig(Path(output_dir) / f"per_seed_{metric}.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_significance_heatmap(
    statistics: dict,
    metric: str,
    output_dir: str,
    formats: list[str],
):
    plt = _import_matplotlib()
    comparisons = _get_pairwise(statistics, metric)
    if not comparisons:
        return

    methods = sorted({c["method_a"] for c in comparisons} | {c["method_b"] for c in comparisons})
    n = len(methods)
    method_idx = {m: i for i, m in enumerate(methods)}

    matrix = np.ones((n, n))
    for comp in comparisons:
        i = method_idx[comp["method_a"]]
        j = method_idx[comp["method_b"]]
        matrix[i, j] = comp["corrected_p_value"]
        matrix[j, i] = comp["corrected_p_value"]

    for i in range(n):
        matrix[i, i] = np.nan

    fig, ax = plt.subplots(figsize=(max(4, n * 1.2), max(4, n * 1.2)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticklabels(methods)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = matrix[i, j]
            marker = significance_marker(p) or "n.s."
            color = "white" if p < 0.3 else "black"
            ax.text(j, i, f"{p:.3f}\n{marker}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title(f"Pairwise p-values: {metric}")
    fig.colorbar(im, ax=ax, label="corrected p-value", shrink=0.8)
    fig.tight_layout()

    for fmt in formats:
        fig.savefig(Path(output_dir) / f"significance_{metric}.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_critical_difference(
    statistics: dict,
    metric: str,
    output_dir: str,
    formats: list[str],
):
    plt = _import_matplotlib()
    metric_data = statistics.get("metrics", {}).get(metric, {})
    if not isinstance(metric_data, dict):
        return
    nemenyi = metric_data.get("nemenyi")
    if not nemenyi:
        return

    mean_ranks = nemenyi["mean_ranks"]
    cd = nemenyi["critical_difference"]
    groups = nemenyi["groups"]

    sorted_methods = sorted(mean_ranks.keys(), key=lambda m: mean_ranks[m])
    n = len(sorted_methods)

    fig, ax = plt.subplots(figsize=(max(6, n * 1.5), max(2.5, 1.0 + len(groups) * 0.3)))

    rank_min = min(mean_ranks.values())
    rank_max = max(mean_ranks.values())
    margin = max(cd, (rank_max - rank_min) * 0.15, 0.5)
    ax.set_xlim(rank_min - margin, rank_max + margin)

    ax.hlines(0, rank_min - margin * 0.5, rank_max + margin * 0.5, color="black", linewidth=0.5)
    for r in range(int(np.floor(rank_min)), int(np.ceil(rank_max)) + 1):
        ax.plot(r, 0, "|", color="black", markersize=10)
        ax.text(r, -0.15, str(r), ha="center", va="top", fontsize=9)

    top_half = sorted_methods[: n // 2]
    bottom_half = sorted_methods[n // 2 :]

    for i, method in enumerate(top_half):
        rank = mean_ranks[method]
        y = 0.4 + i * 0.3
        ax.plot(
            rank,
            0,
            "o",
            color=COLORS.get(sorted_methods.index(method), COLORS[0]),
            markersize=6,
            zorder=5,
        )
        ax.plot([rank, rank], [0, y], color="gray", linewidth=0.5, linestyle="--")
        ax.text(rank, y + 0.05, method, ha="center", va="bottom", fontsize=9, fontweight="bold")

    for i, method in enumerate(bottom_half):
        rank = mean_ranks[method]
        y = -0.4 - i * 0.3
        ax.plot(
            rank,
            0,
            "o",
            color=COLORS.get(sorted_methods.index(method), COLORS[0]),
            markersize=6,
            zorder=5,
        )
        ax.plot([rank, rank], [0, y], color="gray", linewidth=0.5, linestyle="--")
        ax.text(rank, y - 0.05, method, ha="center", va="top", fontsize=9, fontweight="bold")

    group_y_start = 0.4 + len(top_half) * 0.3 + 0.2
    for i, group in enumerate(groups):
        if len(group) < 2:
            continue
        ranks = [mean_ranks[m] for m in group]
        y = group_y_start + i * 0.15
        ax.plot(
            [min(ranks), max(ranks)],
            [y, y],
            color="black",
            linewidth=2.5,
            solid_capstyle="round",
        )

    cd_y = group_y_start + len(groups) * 0.15 + 0.3
    cd_x = rank_min
    ax.plot([cd_x, cd_x + cd], [cd_y, cd_y], color="red", linewidth=2)
    ax.text(
        cd_x + cd / 2,
        cd_y + 0.05,
        f"CD={cd:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="red",
    )

    ax.set_title(f"Critical Difference Diagram: {metric}")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()

    for fmt in formats:
        path = Path(output_dir) / f"cd_diagram_{metric}.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_figures(config):
    try:
        _import_matplotlib()
    except ImportError:
        return

    summary = load_summary(config.output.results_dir)
    statistics = load_statistics(config.output.results_dir)

    try:
        rows = load_metrics(config.output.results_dir)
    except ImportError:
        rows = []

    Path(config.output.figures_dir).mkdir(parents=True, exist_ok=True)

    metrics = set()
    for method_data in summary.values():
        metrics.update(method_data.keys())

    for metric in sorted(metrics):
        plot_comparison_bar(
            summary,
            metric,
            config.output.figures_dir,
            config.output.figure_format,
            statistics=statistics,
        )
        if rows:
            plot_per_seed(rows, metric, config.output.figures_dir, config.output.figure_format)
        if statistics:
            plot_significance_heatmap(
                statistics,
                metric,
                config.output.figures_dir,
                config.output.figure_format,
            )
            plot_critical_difference(
                statistics,
                metric,
                config.output.figures_dir,
                config.output.figure_format,
            )
