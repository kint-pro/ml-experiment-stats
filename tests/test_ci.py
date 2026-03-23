from ml_experiment_stats.ci import check_thresholds
from ml_experiment_stats.config import CIConfig
from ml_experiment_stats.results import ResultsCollector, RunResult


def _setup_results(tmp_path, a_base=0.90, b_base=0.80):
    collector = ResultsCollector(str(tmp_path))
    for i in range(10):
        collector.add(RunResult(seed=42 + i, method="a", metrics={"acc": a_base + i * 0.001}))
        collector.add(RunResult(seed=42 + i, method="b", metrics={"acc": b_base + i * 0.001}))
    collector.save()
    return tmp_path


def test_check_thresholds_passes(tmp_path):
    results_dir = _setup_results(tmp_path)
    ci_config = CIConfig(enabled=True, baseline="a", fail_on_no_significance=True)
    assert check_thresholds(str(results_dir), ci_config)


def test_check_thresholds_fails_no_significance(tmp_path):
    results_dir = _setup_results(tmp_path, a_base=0.90, b_base=0.90)
    ci_config = CIConfig(enabled=True, baseline="a", fail_on_no_significance=True)
    assert not check_thresholds(str(results_dir), ci_config)


def test_check_thresholds_fails_missing_statistics(tmp_path):
    ci_config = CIConfig(enabled=True, baseline="a")
    assert not check_thresholds(str(tmp_path), ci_config)


def test_check_thresholds_fails_no_baseline(tmp_path):
    results_dir = _setup_results(tmp_path)
    ci_config = CIConfig(enabled=True, baseline="")
    assert not check_thresholds(str(results_dir), ci_config)


def test_check_thresholds_fails_min_effect_size(tmp_path):
    collector = ResultsCollector(str(tmp_path))
    import numpy as np
    np.random.seed(42)
    for i in range(20):
        val_a = np.random.normal(0.9, 0.05)
        val_b = np.random.normal(0.88, 0.05)
        collector.add(RunResult(seed=42 + i, method="a", metrics={"acc": val_a}))
        collector.add(RunResult(seed=42 + i, method="b", metrics={"acc": val_b}))
    collector.save()
    ci_config = CIConfig(
        enabled=True, baseline="a", fail_on_no_significance=False, min_effect_size=0.9,
    )
    assert not check_thresholds(str(tmp_path), ci_config)


def test_check_thresholds_passes_with_min_effect_size(tmp_path):
    results_dir = _setup_results(tmp_path, a_base=0.90, b_base=0.70)
    ci_config = CIConfig(
        enabled=True, baseline="a", fail_on_no_significance=True, min_effect_size=0.5,
    )
    assert check_thresholds(str(results_dir), ci_config)
