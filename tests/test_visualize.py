import numpy as np

from ml_experiment_stats.results import ResultsCollector, RunResult
from ml_experiment_stats.visualize import (
    load_metrics,
    load_statistics,
    load_summary,
    plot_comparison_bar,
    plot_critical_difference,
    plot_per_seed,
    plot_significance_heatmap,
)


def _setup_results(tmp_path):
    collector = ResultsCollector(str(tmp_path))
    np.random.seed(42)
    for i in range(10):
        collector.add(
            RunResult(
                seed=42 + i,
                method="a",
                metrics={"acc": float(np.random.normal(0.9, 0.01))},
            )
        )
        collector.add(
            RunResult(
                seed=42 + i,
                method="b",
                metrics={"acc": float(np.random.normal(0.7, 0.01))},
            )
        )
        collector.add(
            RunResult(
                seed=42 + i,
                method="c",
                metrics={"acc": float(np.random.normal(0.5, 0.01))},
            )
        )
    collector.save()
    return tmp_path


def test_plot_comparison_bar_creates_file(tmp_path):
    results_dir = _setup_results(tmp_path)
    summary = load_summary(str(results_dir))
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    plot_comparison_bar(summary, "acc", str(figures_dir), ["png"])
    assert (figures_dir / "comparison_acc.png").exists()


def test_plot_comparison_bar_with_significance(tmp_path):
    results_dir = _setup_results(tmp_path)
    summary = load_summary(str(results_dir))
    statistics = load_statistics(str(results_dir))
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    plot_comparison_bar(summary, "acc", str(figures_dir), ["png"], statistics=statistics)
    assert (figures_dir / "comparison_acc.png").exists()


def test_plot_per_seed_creates_file(tmp_path):
    results_dir = _setup_results(tmp_path)
    rows = load_metrics(str(results_dir))
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    plot_per_seed(rows, "acc", str(figures_dir), ["png"])
    assert (figures_dir / "per_seed_acc.png").exists()


def test_plot_significance_heatmap_creates_file(tmp_path):
    results_dir = _setup_results(tmp_path)
    statistics = load_statistics(str(results_dir))
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    plot_significance_heatmap(statistics, "acc", str(figures_dir), ["png"])
    assert (figures_dir / "significance_acc.png").exists()


def test_plot_critical_difference_creates_file(tmp_path):
    results_dir = _setup_results(tmp_path)
    statistics = load_statistics(str(results_dir))
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    plot_critical_difference(statistics, "acc", str(figures_dir), ["png"])
    assert (figures_dir / "cd_diagram_acc.png").exists()


def test_plot_comparison_bar_no_metric(tmp_path):
    results_dir = _setup_results(tmp_path)
    summary = load_summary(str(results_dir))
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    plot_comparison_bar(summary, "nonexistent", str(figures_dir), ["png"])
    assert not (figures_dir / "comparison_nonexistent.png").exists()


def test_plot_critical_difference_no_nemenyi(tmp_path):
    collector = ResultsCollector(str(tmp_path))
    for i in range(10):
        collector.add(RunResult(seed=42 + i, method="a", metrics={"acc": 0.9 + i * 0.001}))
        collector.add(RunResult(seed=42 + i, method="b", metrics={"acc": 0.8 + i * 0.001}))
    collector.save()
    statistics = load_statistics(str(tmp_path))
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    plot_critical_difference(statistics, "acc", str(figures_dir), ["png"])
    assert not (figures_dir / "cd_diagram_acc.png").exists()
