import json

import pyarrow.parquet as pq

from ml_experiment_stats.results import ResultsCollector, RunResult


def test_collector_saves_parquet(tmp_path):
    collector = ResultsCollector(str(tmp_path))
    collector.add(RunResult(seed=42, method="a", metrics={"loss": 0.5}))
    collector.add(RunResult(seed=42, method="b", metrics={"loss": 0.3}))
    collector.save_parquet()

    parquet_path = tmp_path / "metrics.parquet"
    assert parquet_path.exists()
    table = pq.read_table(parquet_path)
    assert table.num_rows == 2
    assert "seed" in table.column_names
    assert "method" in table.column_names
    assert "loss" in table.column_names


def test_collector_saves_summary(tmp_path):
    collector = ResultsCollector(str(tmp_path))
    collector.add(RunResult(seed=42, method="a", metrics={"loss": 0.5}))
    collector.add(RunResult(seed=43, method="a", metrics={"loss": 0.3}))
    collector.save_summary()

    summary_path = tmp_path / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert "a" in summary
    assert summary["a"]["loss"]["mean"] == 0.4
    assert summary["a"]["loss"]["n"] == 2


def test_run_result_dataset_defaults_to_default():
    run = RunResult(seed=42, method="a", metrics={"acc": 0.9})
    assert run.dataset == "default"


def test_run_result_dataset_can_be_set():
    run = RunResult(seed=42, method="a", metrics={"acc": 0.9}, dataset="cifar10")
    assert run.dataset == "cifar10"


def test_parquet_includes_dataset_column(tmp_path):
    collector = ResultsCollector(str(tmp_path))
    collector.add(RunResult(seed=42, method="a", metrics={"acc": 0.9}, dataset="cifar10"))
    collector.add(RunResult(seed=42, method="b", metrics={"acc": 0.8}, dataset="mnist"))
    collector.save_parquet()

    table = pq.read_table(tmp_path / "metrics.parquet")
    assert "dataset" in table.column_names
    datasets = table.column("dataset").to_pylist()
    assert "cifar10" in datasets
    assert "mnist" in datasets


def test_parquet_dataset_column_defaults_to_default(tmp_path):
    collector = ResultsCollector(str(tmp_path))
    collector.add(RunResult(seed=42, method="a", metrics={"loss": 0.5}))
    collector.save_parquet()

    table = pq.read_table(tmp_path / "metrics.parquet")
    assert "dataset" in table.column_names
    assert table.column("dataset").to_pylist() == ["default"]


def test_parquet_multi_dataset_preserves_all_rows(tmp_path):
    collector = ResultsCollector(str(tmp_path))
    for ds in ["d1", "d2", "d3"]:
        collector.add(RunResult(seed=42, method="a", metrics={"acc": 0.9}, dataset=ds))
    collector.save_parquet()

    table = pq.read_table(tmp_path / "metrics.parquet")
    assert table.num_rows == 3
    datasets = sorted(table.column("dataset").to_pylist())
    assert datasets == ["d1", "d2", "d3"]
