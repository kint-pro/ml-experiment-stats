import json

import numpy as np
import pytest

from ml_experiment_stats.results import ResultsCollector, RunResult
from ml_experiment_stats.statistics import (
    BayesianResult,
    apply_correction,
    bayesian_signed_rank,
    bootstrap_ci,
    build_method_values,
    cliffs_delta,
    cohens_d,
    compute_effect_size,
    cross_dataset_analysis,
    effect_magnitude_cliffs_delta,
    effect_magnitude_cohens_d,
    friedman_test,
    holm_bonferroni,
    is_normal,
    nemenyi_cd,
    nemenyi_test,
    pairwise_test,
    post_hoc_power,
    resolve_test,
    run_statistical_analysis,
)


def _make_runs(method_values: dict[str, list[float]], base_seed=42):
    runs = []
    for method, values in method_values.items():
        for i, val in enumerate(values):
            runs.append(RunResult(seed=base_seed + i, method=method, metrics={"acc": val}))
    return runs


def _make_two_method_runs(a_values, b_values, base_seed=42):
    return _make_runs({"a": a_values, "b": b_values}, base_seed)


def test_cohens_d_identical():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert cohens_d(a, b) == 0.0


def test_cohens_d_constant_difference():
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([1.0, 2.0, 3.0])
    d = cohens_d(a, b)
    assert d == float("inf")


def test_cohens_d_variable_difference():
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([1.0, 3.0, 5.0])
    d = cohens_d(a, b)
    assert d > 0
    assert np.isfinite(d)


def test_cliffs_delta_identical():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert cliffs_delta(a, b) == 0.0


def test_cliffs_delta_complete_dominance():
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([1.0, 2.0, 3.0])
    assert cliffs_delta(a, b) == 1.0


def test_cliffs_delta_negative_dominance():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 11.0, 12.0])
    assert cliffs_delta(a, b) == -1.0


def test_cliffs_delta_empty():
    assert cliffs_delta(np.array([]), np.array([1.0])) == 0.0


def test_effect_magnitude_cohens_d():
    assert effect_magnitude_cohens_d(0.1) == "negligible"
    assert effect_magnitude_cohens_d(0.3) == "small"
    assert effect_magnitude_cohens_d(0.6) == "medium"
    assert effect_magnitude_cohens_d(1.0) == "large"


def test_effect_magnitude_cliffs_delta():
    assert effect_magnitude_cliffs_delta(0.1) == "negligible"
    assert effect_magnitude_cliffs_delta(0.2) == "small"
    assert effect_magnitude_cliffs_delta(0.4) == "medium"
    assert effect_magnitude_cliffs_delta(0.5) == "large"


def test_compute_effect_size_ttest():
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([1.0, 3.0, 5.0])
    es, mag = compute_effect_size(a, b, "ttest")
    assert es == cohens_d(a, b)


def test_compute_effect_size_wilcoxon():
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([1.0, 2.0, 3.0])
    es, mag = compute_effect_size(a, b, "wilcoxon")
    assert es == cliffs_delta(a, b)


def test_is_normal_with_normal_data():
    np.random.seed(42)
    a = np.random.normal(10, 1, 50)
    b = np.random.normal(10, 1, 50)
    assert is_normal(a, b)


def test_is_normal_with_skewed_data():
    np.random.seed(42)
    a = np.random.exponential(1, 50)
    b = np.zeros(50)
    assert not is_normal(a, b)


def test_is_normal_too_few_samples():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    assert not is_normal(a, b)


def test_is_normal_zero_std():
    a = np.array([5.0, 5.0, 5.0, 5.0])
    b = np.array([3.0, 3.0, 3.0, 3.0])
    assert not is_normal(a, b)


def test_resolve_test_explicit():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert resolve_test("wilcoxon", a, b) == "wilcoxon"
    assert resolve_test("ttest", a, b) == "ttest"


def test_resolve_test_auto():
    np.random.seed(42)
    a = np.random.normal(10, 1, 50)
    b = np.random.normal(10, 1, 50)
    result = resolve_test("auto", a, b)
    assert result in ("wilcoxon", "ttest")


def test_holm_bonferroni_single():
    corrected = holm_bonferroni([0.03], alpha=0.05)
    assert corrected == [0.03]


def test_holm_bonferroni_multiple():
    corrected = holm_bonferroni([0.01, 0.04, 0.03], alpha=0.05)
    assert corrected[0] == pytest.approx(0.03)
    assert corrected[2] == pytest.approx(0.06)
    assert corrected[1] == pytest.approx(0.06)


def test_holm_bonferroni_empty():
    assert holm_bonferroni([], alpha=0.05) == []


def test_holm_bonferroni_clamps_to_one():
    corrected = holm_bonferroni([0.8, 0.9], alpha=0.05)
    assert all(p <= 1.0 for p in corrected)


def test_apply_correction_unknown():
    with pytest.raises(ValueError, match="Unknown correction"):
        apply_correction([0.01], "invalid", 0.05)


def test_bootstrap_ci_returns_interval():
    np.random.seed(42)
    a = np.random.normal(10, 1, 30)
    b = np.random.normal(8, 1, 30)
    ci = bootstrap_ci(a, b)
    assert ci is not None
    assert ci[0] < ci[1]
    assert ci[0] > 0


def test_bootstrap_ci_too_few_samples():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    assert bootstrap_ci(a, b) is None


def test_friedman_test_significant():
    np.random.seed(42)
    values = {
        "a": np.random.normal(0.9, 0.01, 20),
        "b": np.random.normal(0.7, 0.01, 20),
        "c": np.random.normal(0.5, 0.01, 20),
    }
    result = friedman_test(values, "acc")
    assert result is not None
    assert result.significant
    assert result.n_methods == 3
    assert result.n_samples == 20
    assert len(result.mean_ranks) == 3


def test_friedman_test_not_significant():
    np.random.seed(42)
    values = {
        "a": np.random.normal(0.9, 0.1, 20),
        "b": np.random.normal(0.9, 0.1, 20),
        "c": np.random.normal(0.9, 0.1, 20),
    }
    result = friedman_test(values, "acc")
    assert result is not None
    assert not result.significant


def test_friedman_test_two_methods():
    values = {
        "a": np.array([1.0, 2.0, 3.0]),
        "b": np.array([4.0, 5.0, 6.0]),
    }
    assert friedman_test(values, "acc") is None


def test_friedman_test_too_few_samples():
    values = {
        "a": np.array([1.0, 2.0]),
        "b": np.array([3.0, 4.0]),
        "c": np.array([5.0, 6.0]),
    }
    assert friedman_test(values, "acc") is None


def test_nemenyi_cd_value():
    cd = nemenyi_cd(3, 20, 0.05)
    assert cd > 0
    assert np.isfinite(cd)


def test_nemenyi_test_produces_groups():
    np.random.seed(42)
    values = {
        "a": np.random.normal(0.9, 0.01, 20),
        "b": np.random.normal(0.7, 0.01, 20),
        "c": np.random.normal(0.5, 0.01, 20),
    }
    fr = friedman_test(values, "acc")
    assert fr is not None
    nem = nemenyi_test(values, "acc", fr)
    assert nem.critical_difference > 0
    assert len(nem.mean_ranks) == 3
    assert len(nem.groups) > 0


def test_build_method_values():
    runs = _make_two_method_runs([0.9, 0.91, 0.92], [0.8, 0.81, 0.82])
    values = build_method_values(runs, "acc")
    assert "a" in values
    assert "b" in values
    assert len(values["a"]) == 3


def test_build_method_values_missing_metric():
    runs = [
        RunResult(seed=42, method="a", metrics={"acc": 0.9}),
        RunResult(seed=42, method="b", metrics={"loss": 0.1}),
    ]
    values = build_method_values(runs, "acc")
    assert "a" in values
    assert "b" not in values


def test_pairwise_test_significant():
    np.random.seed(42)
    a_vals = np.random.normal(0.9, 0.01, 20)
    b_vals = np.random.normal(0.7, 0.01, 20)
    values = {"a": a_vals, "b": b_vals}
    results = pairwise_test(values, "acc", test="wilcoxon")
    assert len(results) == 1
    assert results[0].significant
    assert results[0].p_value < 0.05


def test_pairwise_test_not_significant():
    np.random.seed(42)
    a_vals = np.random.normal(0.9, 0.01, 20)
    b_vals = np.random.normal(0.9, 0.01, 20)
    values = {"a": a_vals, "b": b_vals}
    results = pairwise_test(values, "acc", test="wilcoxon")
    assert len(results) == 1
    assert not results[0].significant


def test_pairwise_test_identical_values():
    values = {"a": np.array([1.0, 1.0, 1.0]), "b": np.array([1.0, 1.0, 1.0])}
    results = pairwise_test(values, "acc")
    assert len(results) == 1
    assert results[0].p_value == 1.0
    assert not results[0].significant


def test_pairwise_test_single_method():
    values = {"a": np.array([1.0, 2.0, 3.0])}
    assert pairwise_test(values, "acc") == []


def test_pairwise_test_ttest():
    np.random.seed(42)
    a_vals = np.random.normal(0.9, 0.01, 20)
    b_vals = np.random.normal(0.7, 0.01, 20)
    values = {"a": a_vals, "b": b_vals}
    results = pairwise_test(values, "acc", test="ttest")
    assert len(results) == 1
    assert results[0].significant
    assert results[0].test_name == "ttest"


def test_pairwise_test_auto():
    np.random.seed(42)
    a_vals = np.random.normal(0.9, 0.01, 20)
    b_vals = np.random.normal(0.7, 0.01, 20)
    values = {"a": a_vals, "b": b_vals}
    results = pairwise_test(values, "acc", test="auto")
    assert len(results) == 1
    assert results[0].test_name in ("wilcoxon", "ttest")


def test_pairwise_test_unknown_test():
    values = {"a": np.array([1.0]), "b": np.array([2.0])}
    with pytest.raises(ValueError, match="Unknown test"):
        pairwise_test(values, "acc", test="invalid")


def test_pairwise_test_three_methods():
    np.random.seed(42)
    values = {
        "a": np.random.normal(0.9, 0.01, 20),
        "b": np.random.normal(0.7, 0.01, 20),
        "c": np.random.normal(0.5, 0.01, 20),
    }
    results = pairwise_test(values, "acc")
    assert len(results) == 3


def test_pairwise_test_includes_ci():
    np.random.seed(42)
    a_vals = np.random.normal(0.9, 0.01, 20)
    b_vals = np.random.normal(0.7, 0.01, 20)
    values = {"a": a_vals, "b": b_vals}
    results = pairwise_test(values, "acc")
    assert results[0].confidence_interval is not None
    lo, hi = results[0].confidence_interval
    assert lo < hi


def test_pairwise_test_wilcoxon_uses_cliffs_delta():
    np.random.seed(42)
    a_vals = np.random.normal(0.9, 0.01, 20)
    b_vals = np.random.normal(0.7, 0.01, 20)
    values = {"a": a_vals, "b": b_vals}
    results = pairwise_test(values, "acc", test="wilcoxon")
    expected_es = cliffs_delta(a_vals, b_vals)
    assert results[0].effect_size == pytest.approx(expected_es)


def test_pairwise_test_ttest_uses_cohens_d():
    np.random.seed(42)
    a_vals = np.random.normal(0.9, 0.01, 20)
    b_vals = np.random.normal(0.7, 0.01, 20)
    values = {"a": a_vals, "b": b_vals}
    results = pairwise_test(values, "acc", test="ttest")
    expected_es = cohens_d(a_vals, b_vals)
    assert results[0].effect_size == pytest.approx(expected_es)


def test_run_statistical_analysis_two_methods():
    runs = _make_two_method_runs(
        [0.95, 0.94, 0.93, 0.96, 0.95, 0.94, 0.93, 0.96, 0.95, 0.94],
        [0.80, 0.81, 0.79, 0.82, 0.80, 0.81, 0.79, 0.82, 0.80, 0.81],
    )
    output = run_statistical_analysis(runs)
    assert "config" in output
    assert "metrics" in output
    metric_data = output["metrics"]["acc"]
    assert "pairwise" in metric_data
    assert metric_data["friedman"] is None
    assert len(metric_data["pairwise"]) == 1
    assert metric_data["pairwise"][0]["significant"]


def test_run_statistical_analysis_three_methods_with_friedman():
    np.random.seed(42)
    runs = _make_runs(
        {
            "a": list(np.random.normal(0.9, 0.01, 20)),
            "b": list(np.random.normal(0.7, 0.01, 20)),
            "c": list(np.random.normal(0.5, 0.01, 20)),
        }
    )
    output = run_statistical_analysis(runs)
    metric_data = output["metrics"]["acc"]
    assert metric_data["friedman"] is not None
    assert metric_data["friedman"]["significant"]
    assert metric_data["nemenyi"] is not None
    assert metric_data["nemenyi"]["critical_difference"] > 0


def test_run_statistical_analysis_sample_warning():
    runs = _make_two_method_runs([0.9, 0.91, 0.92], [0.8, 0.81, 0.82])
    output = run_statistical_analysis(runs)
    assert output["sample_warning"] is not None
    assert "3 seeds" in output["sample_warning"]


def test_run_statistical_analysis_no_sample_warning():
    runs = _make_two_method_runs(
        [0.9 + i * 0.001 for i in range(10)],
        [0.8 + i * 0.001 for i in range(10)],
    )
    output = run_statistical_analysis(runs)
    assert output["sample_warning"] is None


def test_run_statistical_analysis_ci_in_output():
    np.random.seed(42)
    runs = _make_two_method_runs(
        list(np.random.normal(0.9, 0.01, 20)),
        list(np.random.normal(0.7, 0.01, 20)),
    )
    output = run_statistical_analysis(runs)
    comp = output["metrics"]["acc"]["pairwise"][0]
    assert "ci_lower" in comp
    assert "ci_upper" in comp
    assert comp["ci_lower"] < comp["ci_upper"]


def test_collector_save_statistics(tmp_path):
    runs = _make_two_method_runs(
        [0.95, 0.94, 0.93, 0.96, 0.95, 0.94, 0.93, 0.96, 0.95, 0.94],
        [0.80, 0.81, 0.79, 0.82, 0.80, 0.81, 0.79, 0.82, 0.80, 0.81],
    )
    collector = ResultsCollector(str(tmp_path))
    for run in runs:
        collector.add(run)
    collector.save_statistics()

    stats_path = tmp_path / "statistics.json"
    assert stats_path.exists()
    data = json.loads(stats_path.read_text())
    assert data["metrics"]["acc"]["pairwise"][0]["significant"]


def test_collector_save_statistics_single_method(tmp_path):
    collector = ResultsCollector(str(tmp_path))
    collector.add(RunResult(seed=42, method="a", metrics={"acc": 0.9}))
    collector.add(RunResult(seed=43, method="a", metrics={"acc": 0.91}))
    collector.save_statistics()
    assert not (tmp_path / "statistics.json").exists()


def test_bonferroni_correction():
    np.random.seed(42)
    values = {
        "a": np.random.normal(0.9, 0.01, 20),
        "b": np.random.normal(0.7, 0.01, 20),
    }
    results = pairwise_test(values, "acc", correction="bonferroni")
    assert len(results) == 1
    assert results[0].corrected_p_value == results[0].p_value


def test_no_correction():
    np.random.seed(42)
    values = {
        "a": np.random.normal(0.9, 0.01, 20),
        "b": np.random.normal(0.7, 0.01, 20),
    }
    results = pairwise_test(values, "acc", correction="none")
    assert results[0].corrected_p_value == results[0].p_value


def test_bayesian_signed_rank_a_wins():
    np.random.seed(42)
    values = {
        "a": np.random.normal(0.9, 0.005, 30),
        "b": np.random.normal(0.7, 0.005, 30),
    }
    results = bayesian_signed_rank(values, "acc")
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, BayesianResult)
    assert r.method_a == "a"
    assert r.method_b == "b"
    assert r.metric == "acc"
    assert r.decision == "a wins"
    assert r.p_a_better > r.p_b_better
    assert r.p_a_better > r.p_rope
    assert r.n_samples == 30


def test_bayesian_signed_rank_b_wins():
    np.random.seed(42)
    values = {
        "a": np.random.normal(0.7, 0.005, 30),
        "b": np.random.normal(0.9, 0.005, 30),
    }
    results = bayesian_signed_rank(values, "acc")
    assert len(results) == 1
    assert results[0].decision == "b wins"
    assert results[0].p_b_better > results[0].p_a_better
    assert results[0].p_b_better > results[0].p_rope


def test_bayesian_signed_rank_equivalent():
    np.random.seed(42)
    base = np.random.normal(0.9, 0.001, 30)
    values = {
        "a": base,
        "b": base + 0.0001,
    }
    results = bayesian_signed_rank(values, "acc", rope=0.01)
    assert len(results) == 1
    assert results[0].decision == "equivalent"


def test_bayesian_signed_rank_single_method_returns_empty():
    values = {"a": np.array([0.9, 0.91, 0.92])}
    results = bayesian_signed_rank(values, "acc")
    assert results == []


def test_bayesian_signed_rank_probabilities_sum_to_one():
    np.random.seed(42)
    values = {
        "a": np.random.normal(0.85, 0.01, 20),
        "b": np.random.normal(0.80, 0.01, 20),
    }
    results = bayesian_signed_rank(values, "acc")
    r = results[0]
    assert r.p_a_better + r.p_rope + r.p_b_better == pytest.approx(1.0, abs=1e-6)


def test_bayesian_signed_rank_rope_width_stored():
    np.random.seed(42)
    values = {
        "a": np.random.normal(0.9, 0.01, 20),
        "b": np.random.normal(0.8, 0.01, 20),
    }
    results = bayesian_signed_rank(values, "acc", rope=0.05)
    assert results[0].rope_width == pytest.approx(0.05)


def test_bayesian_signed_rank_three_methods_produces_three_pairs():
    np.random.seed(42)
    values = {
        "a": np.random.normal(0.9, 0.01, 20),
        "b": np.random.normal(0.8, 0.01, 20),
        "c": np.random.normal(0.7, 0.01, 20),
    }
    results = bayesian_signed_rank(values, "acc")
    assert len(results) == 3


def test_post_hoc_power_large_effect_high_power():
    np.random.seed(42)
    a = np.random.normal(0.9, 0.01, 30)
    b = np.random.normal(0.7, 0.01, 30)
    result = post_hoc_power(a, b)
    assert result["power"] is not None
    assert result["power"] > 0.8
    assert result["recommended_n"] == 30


def test_post_hoc_power_small_effect_low_power():
    np.random.seed(42)
    a = np.array([0.901, 0.900, 0.902, 0.899, 0.901])
    b = np.array([0.900, 0.901, 0.900, 0.900, 0.900])
    result = post_hoc_power(a, b)
    assert result["power"] is not None
    assert result["power"] < 0.8
    assert result["recommended_n"] is not None
    assert result["recommended_n"] > 5


def test_post_hoc_power_identical_returns_none():
    a = np.array([0.9, 0.9, 0.9, 0.9])
    b = np.array([0.9, 0.9, 0.9, 0.9])
    result = post_hoc_power(a, b)
    assert result["power"] is None
    assert result["recommended_n"] is None


def test_post_hoc_power_too_few_samples():
    a = np.array([0.9, 0.8])
    b = np.array([0.7, 0.6])
    result = post_hoc_power(a, b)
    assert result["power"] is None
    assert result["recommended_n"] is None


def test_post_hoc_power_returns_dict_with_required_keys():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    result = post_hoc_power(a, b)
    assert "power" in result
    assert "recommended_n" in result


def _make_multi_dataset_runs(seed=42):
    np.random.seed(seed)
    runs = []
    datasets = ["d1", "d2", "d3"]
    methods = {"a": 0.9, "b": 0.7, "c": 0.5}
    for ds in datasets:
        for method, mean in methods.items():
            for s in range(10):
                val = float(np.random.normal(mean, 0.01))
                runs.append(
                    RunResult(seed=seed + s, method=method, metrics={"acc": val}, dataset=ds)
                )
    return runs


def test_cross_dataset_analysis_three_methods_three_datasets():
    runs = _make_multi_dataset_runs()
    result = cross_dataset_analysis(runs, "acc")
    assert result is not None
    assert "mean_ranks" in result
    assert "n_datasets" in result
    assert result["n_datasets"] == 3
    assert result["n_methods"] == 3


def test_cross_dataset_analysis_contains_friedman_when_enough_datasets():
    runs = _make_multi_dataset_runs()
    result = cross_dataset_analysis(runs, "acc")
    assert result is not None
    assert "statistic" in result
    assert "p_value" in result
    assert "significant" in result


def test_cross_dataset_analysis_single_dataset_returns_none():
    runs = [
        RunResult(seed=1, method="a", metrics={"acc": 0.9}, dataset="d1"),
        RunResult(seed=2, method="b", metrics={"acc": 0.8}, dataset="d1"),
    ]
    result = cross_dataset_analysis(runs, "acc")
    assert result is None


def test_cross_dataset_analysis_missing_metric_returns_none():
    runs = [
        RunResult(seed=1, method="a", metrics={"loss": 0.1}, dataset="d1"),
        RunResult(seed=2, method="b", metrics={"loss": 0.2}, dataset="d2"),
    ]
    result = cross_dataset_analysis(runs, "acc")
    assert result is None


def test_cross_dataset_analysis_best_method_has_lowest_mean_rank():
    runs = _make_multi_dataset_runs()
    result = cross_dataset_analysis(runs, "acc")
    assert result is not None
    ranks = result["mean_ranks"]
    assert ranks["a"] < ranks["b"] < ranks["c"]


def test_cross_dataset_analysis_two_methods_two_datasets_no_friedman():
    runs = [
        RunResult(seed=1, method="a", metrics={"acc": 0.9}, dataset="d1"),
        RunResult(seed=2, method="b", metrics={"acc": 0.8}, dataset="d1"),
        RunResult(seed=1, method="a", metrics={"acc": 0.85}, dataset="d2"),
        RunResult(seed=2, method="b", metrics={"acc": 0.75}, dataset="d2"),
    ]
    result = cross_dataset_analysis(runs, "acc")
    assert result is not None
    assert "mean_ranks" in result
    assert "statistic" not in result
