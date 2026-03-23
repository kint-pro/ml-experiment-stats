import json

import pyarrow.parquet as pq

from ml_experiment_stats.config import ExperimentConfig
from ml_experiment_stats.report import save_report
from ml_experiment_stats.results import ResultsCollector, RunResult


def _run_full_pipeline(tmp_path):
    config = ExperimentConfig(name="contract_test")
    collector = ResultsCollector(str(tmp_path), statistics_config=config.statistics)

    for seed in [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]:
        collector.add(RunResult(seed=seed, method="baseline", metrics={"acc": 0.9, "loss": 0.1}))
        collector.add(RunResult(seed=seed, method="proposed", metrics={"acc": 0.95, "loss": 0.05}))

    collector.save()
    config.save(tmp_path / "config_used.json")
    save_report(str(tmp_path))
    return tmp_path


def test_summary_json_schema(tmp_path):
    results_dir = _run_full_pipeline(tmp_path)
    summary = json.loads((results_dir / "summary.json").read_text())

    assert isinstance(summary, dict)
    for method, method_data in summary.items():
        assert isinstance(method, str)
        for metric, stats in method_data.items():
            assert isinstance(metric, str)
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats
            assert "n" in stats
            assert isinstance(stats["mean"], float)
            assert isinstance(stats["n"], int)


def test_statistics_json_schema(tmp_path):
    results_dir = _run_full_pipeline(tmp_path)
    statistics = json.loads((results_dir / "statistics.json").read_text())

    assert "config" in statistics
    assert "test" in statistics["config"]
    assert "alpha" in statistics["config"]
    assert "correction" in statistics["config"]

    assert "metrics" in statistics
    for metric, metric_data in statistics["metrics"].items():
        assert "pairwise" in metric_data
        assert "friedman" in metric_data
        assert "nemenyi" in metric_data
        assert "bayesian" in metric_data
        assert "power" in metric_data

        for comp in metric_data["pairwise"]:
            assert "method_a" in comp
            assert "method_b" in comp
            assert "p_value" in comp
            assert "corrected_p_value" in comp
            assert "significant" in comp
            assert "effect_size" in comp
            assert "effect_magnitude" in comp
            assert "n_samples" in comp

        for bay in metric_data["bayesian"]:
            assert "p_a_better" in bay
            assert "p_rope" in bay
            assert "p_b_better" in bay
            assert "decision" in bay

        for pair_key, pw in metric_data["power"].items():
            assert "power" in pw
            assert "recommended_n" in pw


def test_report_json_schema(tmp_path):
    results_dir = _run_full_pipeline(tmp_path)
    report = json.loads((results_dir / "report.json").read_text())

    assert "experiment" in report
    assert "timestamp" in report
    assert "exit_codes" in report
    assert "methods" in report
    assert "comparisons" in report
    assert "statistics_config" in report

    for method, method_data in report["methods"].items():
        for metric, stats in method_data.items():
            assert "mean" in stats
            assert "std" in stats
            assert "n" in stats

    for metric, metric_data in report["comparisons"].items():
        assert "pairwise" in metric_data
        assert "bayesian" in metric_data
        assert "power" in metric_data


def test_report_md_exists(tmp_path):
    results_dir = _run_full_pipeline(tmp_path)
    md = (results_dir / "report.md").read_text()
    assert len(md) > 0
    assert "# contract_test" in md
    assert "## Results" in md
    assert "## Statistical Analysis" in md
    assert "## Config" in md


def test_metrics_parquet_schema(tmp_path):
    results_dir = _run_full_pipeline(tmp_path)
    table = pq.read_table(results_dir / "metrics.parquet")

    assert "seed" in table.column_names
    assert "method" in table.column_names
    assert "dataset" in table.column_names
    assert table.num_rows == 20


def test_all_output_files_exist(tmp_path):
    results_dir = _run_full_pipeline(tmp_path)

    assert (results_dir / "summary.json").exists()
    assert (results_dir / "statistics.json").exists()
    assert (results_dir / "metrics.parquet").exists()
    assert (results_dir / "config_used.json").exists()
    assert (results_dir / "report.json").exists()
    assert (results_dir / "report.md").exists()


def test_summary_methods_match_input(tmp_path):
    results_dir = _run_full_pipeline(tmp_path)
    summary = json.loads((results_dir / "summary.json").read_text())
    assert set(summary.keys()) == {"baseline", "proposed"}


def test_summary_metrics_match_input(tmp_path):
    results_dir = _run_full_pipeline(tmp_path)
    summary = json.loads((results_dir / "summary.json").read_text())
    for method_data in summary.values():
        assert set(method_data.keys()) == {"acc", "loss"}
