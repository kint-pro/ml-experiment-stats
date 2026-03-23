import json

import pytest

from ml_experiment_stats.diff import (
    diff_summaries,
    generate_diff,
    render_diff_console,
    render_diff_json,
    render_diff_markdown,
)
from ml_experiment_stats.report import save_report
from ml_experiment_stats.results import ResultsCollector, RunResult


def _setup_results(tmp_path, method_a_base=0.90, method_b_base=0.80):
    collector = ResultsCollector(str(tmp_path))
    for i in range(10):
        val_a = method_a_base + i * 0.001
        val_b = method_b_base + i * 0.001
        collector.add(RunResult(seed=42 + i, method="a", metrics={"acc": val_a}))
        collector.add(RunResult(seed=42 + i, method="b", metrics={"acc": val_b}))
    collector.save_parquet()
    collector.save_summary()


def test_diff_summaries_detects_change():
    current = {"a": {"acc": {"mean": 0.95, "std": 0.01, "n": 10}}}
    baseline = {"a": {"acc": {"mean": 0.90, "std": 0.01, "n": 10}}}
    diffs = diff_summaries(current, baseline)
    assert len(diffs) == 1
    assert diffs[0]["delta"] == pytest.approx(0.05)
    assert diffs[0]["delta_pct"] > 0


def test_diff_summaries_no_overlap():
    current = {"a": {"acc": {"mean": 0.95, "std": 0.01, "n": 10}}}
    baseline = {"b": {"acc": {"mean": 0.90, "std": 0.01, "n": 10}}}
    diffs = diff_summaries(current, baseline)
    assert diffs == []


def test_diff_summaries_negative_change():
    current = {"a": {"acc": {"mean": 0.85, "std": 0.01, "n": 10}}}
    baseline = {"a": {"acc": {"mean": 0.90, "std": 0.01, "n": 10}}}
    diffs = diff_summaries(current, baseline)
    assert diffs[0]["delta"] < 0
    assert diffs[0]["delta_pct"] < 0


def test_generate_diff_from_dirs(tmp_path):
    current_dir = tmp_path / "current"
    baseline_dir = tmp_path / "baseline"
    current_dir.mkdir()
    baseline_dir.mkdir()
    _setup_results(current_dir, 0.95, 0.85)
    _setup_results(baseline_dir, 0.90, 0.80)

    diffs = generate_diff(str(current_dir), str(baseline_dir))
    assert len(diffs) == 2
    for d in diffs:
        assert d["delta"] > 0


def test_render_diff_markdown():
    diffs = [
        {
            "method": "a", "metric": "acc",
            "baseline_mean": 0.90, "baseline_std": 0.01,
            "current_mean": 0.95, "current_std": 0.01,
            "delta": 0.05, "delta_pct": 5.56,
        },
    ]
    md = render_diff_markdown(diffs)
    assert "## Diff vs Baseline" in md
    assert "| a |" in md
    assert "+0.0500" in md


def test_render_diff_markdown_empty():
    md = render_diff_markdown([])
    assert "No comparable metrics found" in md


def test_render_diff_json_returns_list():
    diffs = [
        {
            "method": "a", "metric": "acc",
            "baseline_mean": 0.90, "baseline_std": 0.01,
            "current_mean": 0.95, "current_std": 0.01,
            "delta": 0.05, "delta_pct": 5.56,
        },
    ]
    result = render_diff_json(diffs)
    assert result == diffs


def test_render_diff_json_empty():
    assert render_diff_json([]) == []


def test_render_diff_json_preserves_all_fields():
    diffs = [
        {
            "method": "b", "metric": "loss",
            "baseline_mean": 0.5, "baseline_std": 0.02,
            "current_mean": 0.4, "current_std": 0.01,
            "delta": -0.1, "delta_pct": -20.0,
        },
    ]
    result = render_diff_json(diffs)
    assert result[0]["method"] == "b"
    assert result[0]["delta"] == pytest.approx(-0.1)
    assert result[0]["delta_pct"] == pytest.approx(-20.0)


def test_render_diff_console_prints_output(capsys):
    diffs = [
        {
            "method": "a", "metric": "acc",
            "baseline_mean": 0.90, "baseline_std": 0.01,
            "current_mean": 0.95, "current_std": 0.01,
            "delta": 0.05, "delta_pct": 5.56,
        },
    ]
    render_diff_console(diffs)
    captured = capsys.readouterr()
    assert "a" in captured.out
    assert "acc" in captured.out


def test_render_diff_console_empty_prints_no_metrics(capsys):
    render_diff_console([])
    captured = capsys.readouterr()
    assert "No comparable metrics found" in captured.out


def _setup_full_results(path, a_base, b_base):
    collector = ResultsCollector(str(path))
    for i in range(10):
        collector.add(RunResult(seed=42 + i, method="a", metrics={"acc": a_base + i * 0.001}))
        collector.add(RunResult(seed=42 + i, method="b", metrics={"acc": b_base + i * 0.001}))
    collector.save()
    config = {
        "experiment": {"name": "Diff Test", "version": "0.1.0"},
        "seed": {"base": 42, "n_runs": 10},
    }
    (path / "config_used.json").write_text(json.dumps(config))


def test_save_report_baseline_dir_markdown_has_diff_section(tmp_path):
    current_dir = tmp_path / "current"
    baseline_dir = tmp_path / "baseline"
    current_dir.mkdir()
    baseline_dir.mkdir()
    _setup_full_results(current_dir, 0.95, 0.85)
    _setup_full_results(baseline_dir, 0.90, 0.80)

    save_report(str(current_dir), baseline_dir=str(baseline_dir))
    md = (current_dir / "report.md").read_text()
    assert "## Diff vs Baseline" in md
    assert "| a |" in md


def test_save_report_baseline_dir_json_has_diff_key(tmp_path):
    current_dir = tmp_path / "current"
    baseline_dir = tmp_path / "baseline"
    current_dir.mkdir()
    baseline_dir.mkdir()
    _setup_full_results(current_dir, 0.95, 0.85)
    _setup_full_results(baseline_dir, 0.90, 0.80)

    save_report(str(current_dir), baseline_dir=str(baseline_dir))
    json_data = json.loads((current_dir / "report.json").read_text())
    assert "diff" in json_data
    assert isinstance(json_data["diff"], list)
    assert len(json_data["diff"]) == 2
    for entry in json_data["diff"]:
        assert entry["delta"] == pytest.approx(0.05, abs=0.001)


def test_save_report_no_baseline_diff_absent_from_markdown(tmp_path):
    _setup_full_results(tmp_path, 0.90, 0.80)
    save_report(str(tmp_path))
    md = (tmp_path / "report.md").read_text()
    assert "## Diff vs Baseline" not in md


def test_save_report_no_baseline_diff_absent_from_json(tmp_path):
    _setup_full_results(tmp_path, 0.90, 0.80)
    save_report(str(tmp_path))
    json_data = json.loads((tmp_path / "report.json").read_text())
    assert "diff" not in json_data
