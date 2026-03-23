import json

from ml_experiment_stats.report import (
    load_report_data,
    render_console,
    render_json,
    render_markdown,
    save_report,
)
from ml_experiment_stats.results import ResultsCollector, RunResult


def _setup_results(tmp_path):
    collector = ResultsCollector(str(tmp_path))
    for i in range(10):
        collector.add(RunResult(seed=42 + i, method="a", metrics={"acc": 0.90 + i * 0.001}))
        collector.add(RunResult(seed=42 + i, method="b", metrics={"acc": 0.80 + i * 0.001}))
    collector.save()

    config = {
        "experiment": {"name": "Test Experiment", "version": "0.1.0"},
        "seed": {"base": 42, "n_runs": 10},
    }
    (tmp_path / "config_used.json").write_text(json.dumps(config))
    return tmp_path


def test_load_report_data(tmp_path):
    results_dir = _setup_results(tmp_path)
    data = load_report_data(str(results_dir))
    assert "config" in data
    assert "summary" in data
    assert "statistics" in data
    assert "timestamp" in data


def test_render_markdown_contains_header(tmp_path):
    results_dir = _setup_results(tmp_path)
    data = load_report_data(str(results_dir))
    md = render_markdown(data)
    assert "# Test Experiment" in md


def test_render_markdown_contains_results_table(tmp_path):
    results_dir = _setup_results(tmp_path)
    data = load_report_data(str(results_dir))
    md = render_markdown(data)
    assert "| Method |" in md
    assert "| a |" in md
    assert "| b |" in md


def test_render_markdown_contains_statistics(tmp_path):
    results_dir = _setup_results(tmp_path)
    data = load_report_data(str(results_dir))
    md = render_markdown(data)
    assert "## Statistical Analysis" in md
    assert "a vs b" in md


def test_render_markdown_contains_config(tmp_path):
    results_dir = _setup_results(tmp_path)
    data = load_report_data(str(results_dir))
    md = render_markdown(data)
    assert "## Config" in md
    assert "```json" in md


def test_render_console_runs(tmp_path, capsys):
    results_dir = _setup_results(tmp_path)
    data = load_report_data(str(results_dir))
    render_console(data)
    captured = capsys.readouterr()
    assert "a vs b" in captured.out


def test_render_json_structure(tmp_path):
    results_dir = _setup_results(tmp_path)
    data = load_report_data(str(results_dir))
    report = render_json(data)
    assert "experiment" in report
    assert "methods" in report
    assert "comparisons" in report
    assert "a" in report["methods"]
    assert "b" in report["methods"]


def test_save_report_creates_both_files(tmp_path):
    _setup_results(tmp_path)
    save_report(str(tmp_path))
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.json").exists()
    json_data = json.loads((tmp_path / "report.json").read_text())
    assert "methods" in json_data


def test_render_markdown_contains_bayesian_section(tmp_path):
    results_dir = _setup_results(tmp_path)
    data = load_report_data(str(results_dir))
    md = render_markdown(data)
    assert "Bayesian" in md


def test_render_markdown_bayesian_contains_decision(tmp_path):
    results_dir = _setup_results(tmp_path)
    data = load_report_data(str(results_dir))
    md = render_markdown(data)
    assert "Decision" in md or "decision" in md.lower()


def test_render_json_report_contains_bayesian_data(tmp_path):
    results_dir = _setup_results(tmp_path)
    data = load_report_data(str(results_dir))
    statistics = data["statistics"]
    assert statistics is not None
    for metric_data in statistics["metrics"].values():
        assert "bayesian" in metric_data
        assert len(metric_data["bayesian"]) > 0
        entry = metric_data["bayesian"][0]
        assert "p_a_better" in entry
        assert "p_b_better" in entry
        assert "p_rope" in entry
        assert "decision" in entry


def _setup_two_results_dirs(tmp_path):
    current_dir = tmp_path / "current"
    baseline_dir = tmp_path / "baseline"
    current_dir.mkdir()
    baseline_dir.mkdir()

    for direction, path, a_base, b_base in [
        ("current", current_dir, 0.95, 0.85),
        ("baseline", baseline_dir, 0.90, 0.80),
    ]:
        collector = ResultsCollector(str(path))
        for i in range(10):
            collector.add(RunResult(seed=42 + i, method="a", metrics={"acc": a_base + i * 0.001}))
            collector.add(RunResult(seed=42 + i, method="b", metrics={"acc": b_base + i * 0.001}))
        collector.save()
        config = {
            "experiment": {"name": f"{direction} exp", "version": "0.1.0"},
            "seed": {"base": 42, "n_runs": 10},
        }
        (path / "config_used.json").write_text(json.dumps(config))

    return current_dir, baseline_dir


def test_save_report_with_baseline_dir_adds_diff_to_markdown(tmp_path):
    current_dir, baseline_dir = _setup_two_results_dirs(tmp_path)
    save_report(str(current_dir), baseline_dir=str(baseline_dir))
    md = (current_dir / "report.md").read_text()
    assert "## Diff vs Baseline" in md


def test_save_report_with_baseline_dir_adds_diff_to_json(tmp_path):
    current_dir, baseline_dir = _setup_two_results_dirs(tmp_path)
    save_report(str(current_dir), baseline_dir=str(baseline_dir))
    json_data = json.loads((current_dir / "report.json").read_text())
    assert "diff" in json_data
    assert len(json_data["diff"]) > 0


def test_save_report_without_baseline_dir_no_diff_section(tmp_path):
    _setup_results(tmp_path)
    save_report(str(tmp_path))
    md = (tmp_path / "report.md").read_text()
    assert "## Diff vs Baseline" not in md


def test_save_report_without_baseline_dir_no_diff_in_json(tmp_path):
    _setup_results(tmp_path)
    save_report(str(tmp_path))
    json_data = json.loads((tmp_path / "report.json").read_text())
    assert "diff" not in json_data
