from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from ml_experiment_stats.results import RunResult


@dataclass
class PairwiseResult:
    method_a: str
    method_b: str
    metric: str
    statistic: float
    p_value: float
    corrected_p_value: float
    significant: bool
    effect_size: float
    effect_magnitude: str
    test_name: str
    n_samples: int
    confidence_interval: tuple[float, float] | None = None


@dataclass
class FriedmanResult:
    metric: str
    statistic: float
    p_value: float
    significant: bool
    n_methods: int
    n_samples: int
    mean_ranks: dict[str, float] = field(default_factory=dict)


@dataclass
class BayesianResult:
    method_a: str
    method_b: str
    metric: str
    p_a_better: float
    p_rope: float
    p_b_better: float
    rope_width: float
    decision: str
    n_samples: int


@dataclass
class NemenyiResult:
    metric: str
    critical_difference: float
    alpha: float
    n_methods: int
    n_samples: int
    mean_ranks: dict[str, float] = field(default_factory=dict)
    groups: list[list[str]] = field(default_factory=list)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    std = np.std(diff, ddof=1)
    if std == 0:
        return 0.0 if np.mean(diff) == 0 else float("inf") * np.sign(np.mean(diff))
    return float(np.mean(diff) / std)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0
    diff_matrix = np.subtract.outer(a, b)
    dominance = np.sum(np.sign(diff_matrix))
    return float(dominance / (n_a * n_b))


def effect_magnitude_cohens_d(d: float) -> str:
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    if d_abs < 0.5:
        return "small"
    if d_abs < 0.8:
        return "medium"
    return "large"


def effect_magnitude_cliffs_delta(d: float) -> str:
    d_abs = abs(d)
    if d_abs < 0.147:
        return "negligible"
    if d_abs < 0.33:
        return "small"
    if d_abs < 0.474:
        return "medium"
    return "large"


def compute_effect_size(
    a: np.ndarray,
    b: np.ndarray,
    test_name: str,
) -> tuple[float, str]:
    if test_name == "ttest":
        d = cohens_d(a, b)
        return d, effect_magnitude_cohens_d(d)
    d = cliffs_delta(a, b)
    return d, effect_magnitude_cliffs_delta(d)


def is_normal(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> bool:
    diff = a - b
    if len(diff) < 3:
        return False
    if np.std(diff) == 0:
        return False
    _, p = stats.shapiro(diff)
    return p > alpha


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 9999,
) -> tuple[float, float] | None:
    diff = a - b
    if len(diff) < 5:
        return None
    result = stats.bootstrap(
        (diff,),
        np.mean,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method="BCa",
    )
    return (float(result.confidence_interval.low), float(result.confidence_interval.high))


def post_hoc_power(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    diff = a - b
    n = len(diff)
    if n < 3 or np.std(diff, ddof=1) == 0:
        return {"power": None, "recommended_n": None}

    d = abs(float(np.mean(diff) / np.std(diff, ddof=1)))
    noncentrality = d * np.sqrt(n)
    critical_t = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    power = 1.0 - stats.t.cdf(critical_t, df=n - 1, loc=noncentrality)
    power += stats.t.cdf(-critical_t, df=n - 1, loc=noncentrality)
    power = float(power)

    recommended_n = n
    if power < 0.8:
        recommended_n = None
        for candidate_n in range(n + 1, 10001):
            nc = d * np.sqrt(candidate_n)
            ct = float(stats.t.ppf(1 - alpha / 2, df=candidate_n - 1))
            p = 1.0 - stats.t.cdf(ct, df=candidate_n - 1, loc=nc)
            p += stats.t.cdf(-ct, df=candidate_n - 1, loc=nc)
            if p >= 0.8:
                recommended_n = candidate_n
                break

    return {"power": power, "recommended_n": recommended_n}


def holm_bonferroni(p_values: list[float], alpha: float) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    for rank, (original_idx, p) in enumerate(indexed):
        corrected[original_idx] = min(p * (n - rank), 1.0)
    for i in range(1, n):
        orig_idx = indexed[i][0]
        prev_idx = indexed[i - 1][0]
        corrected[orig_idx] = max(corrected[orig_idx], corrected[prev_idx])
    return corrected


def apply_correction(
    p_values: list[float],
    correction: str,
    alpha: float,
) -> list[float]:
    if correction == "holm":
        return holm_bonferroni(p_values, alpha)
    if correction == "bonferroni":
        return [min(p * len(p_values), 1.0) for p in p_values]
    if correction == "none":
        return list(p_values)
    raise ValueError(
        f"Unknown correction: {correction}. Use 'holm', 'bonferroni', or 'none'.",
    )


def resolve_test(test: str, a: np.ndarray, b: np.ndarray) -> str:
    if test != "auto":
        return test
    return "ttest" if is_normal(a, b) else "wilcoxon"


def friedman_test(
    values_by_method: dict[str, np.ndarray],
    metric: str,
    alpha: float = 0.05,
) -> FriedmanResult | None:
    methods = sorted(values_by_method.keys())
    if len(methods) < 3:
        return None

    n = min(len(values_by_method[m]) for m in methods)
    if n < 3:
        return None

    groups = [values_by_method[m][:n] for m in methods]
    stat, p = stats.friedmanchisquare(*groups)

    rank_matrix = np.zeros((n, len(methods)))
    for i in range(n):
        row_values = [groups[j][i] for j in range(len(methods))]
        rank_matrix[i] = stats.rankdata(row_values)
    mean_ranks = {m: float(np.mean(rank_matrix[:, j])) for j, m in enumerate(methods)}

    return FriedmanResult(
        metric=metric,
        statistic=float(stat),
        p_value=float(p),
        significant=float(p) < alpha,
        n_methods=len(methods),
        n_samples=n,
        mean_ranks=mean_ranks,
    )


def nemenyi_cd(n_methods: int, n_samples: int, alpha: float = 0.05) -> float:
    q_alpha = float(stats.studentized_range.ppf(1 - alpha, n_methods, np.inf) / np.sqrt(2))
    return q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_samples))


def nemenyi_test(
    values_by_method: dict[str, np.ndarray],
    metric: str,
    friedman_result: FriedmanResult,
    alpha: float = 0.05,
) -> NemenyiResult:
    methods = sorted(values_by_method.keys())
    n = friedman_result.n_samples
    k = friedman_result.n_methods
    cd = nemenyi_cd(k, n, alpha)

    groups = []
    sorted_methods = sorted(methods, key=lambda m: friedman_result.mean_ranks[m])
    for m_i in sorted_methods:
        found_group = False
        for group in groups:
            if abs(friedman_result.mean_ranks[m_i] - friedman_result.mean_ranks[group[0]]) < cd:
                group.append(m_i)
                found_group = True
                break
        if not found_group:
            groups.append([m_i])

    return NemenyiResult(
        metric=metric,
        critical_difference=cd,
        alpha=alpha,
        n_methods=k,
        n_samples=n,
        mean_ranks=friedman_result.mean_ranks,
        groups=groups,
    )


def bayesian_signed_rank(
    values_by_method: dict[str, np.ndarray],
    metric: str,
    rope: float = 0.01,
) -> list[BayesianResult]:
    try:
        import baycomp
    except ImportError:
        raise ImportError("baycomp required: pip install kint-stats[bayesian]") from None

    methods = sorted(values_by_method.keys())
    if len(methods) < 2:
        return []

    results = []
    for method_a, method_b in combinations(methods, 2):
        a = values_by_method[method_a]
        b = values_by_method[method_b]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

        probs = baycomp.two_on_single(a, b, rope=rope)
        p_a, p_rope_val, p_b = probs

        if p_a > p_b and p_a > p_rope_val:
            decision = f"{method_a} wins"
        elif p_b > p_a and p_b > p_rope_val:
            decision = f"{method_b} wins"
        else:
            decision = "equivalent"

        results.append(
            BayesianResult(
                method_a=method_a,
                method_b=method_b,
                metric=metric,
                p_a_better=float(p_a),
                p_rope=float(p_rope_val),
                p_b_better=float(p_b),
                rope_width=rope,
                decision=decision,
                n_samples=n,
            )
        )

    return results


def pairwise_test(
    values_by_method: dict[str, np.ndarray],
    metric: str,
    test: str = "wilcoxon",
    alpha: float = 0.05,
    correction: str = "holm",
) -> list[PairwiseResult]:
    methods = sorted(values_by_method.keys())
    if len(methods) < 2:
        return []

    pairs = list(combinations(methods, 2))
    raw_results = []

    for method_a, method_b in pairs:
        a = values_by_method[method_a]
        b = values_by_method[method_b]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

        resolved_test = resolve_test(test, a, b)

        if np.array_equal(a, b):
            raw_results.append(
                (method_a, method_b, 0.0, 1.0, 0.0, "negligible", n, resolved_test, None),
            )
            continue

        if resolved_test == "wilcoxon":
            stat_result = stats.wilcoxon(a, b)
        elif resolved_test == "ttest":
            stat_result = stats.ttest_rel(a, b)
        else:
            raise ValueError(f"Unknown test: {resolved_test}. Use 'wilcoxon', 'ttest', or 'auto'.")

        es, mag = compute_effect_size(a, b, resolved_test)
        ci = bootstrap_ci(a, b)
        entry = (
            method_a,
            method_b,
            float(stat_result.statistic),
            float(stat_result.pvalue),
            es,
            mag,
            n,
            resolved_test,
            ci,
        )
        raw_results.append(entry)

    raw_p_values = [r[3] for r in raw_results]
    corrected_p_values = apply_correction(raw_p_values, correction, alpha)

    results = []
    for i, (m_a, m_b, stat, p, es, mag, n, tname, ci) in enumerate(raw_results):
        corrected_p = corrected_p_values[i]
        results.append(
            PairwiseResult(
                method_a=m_a,
                method_b=m_b,
                metric=metric,
                statistic=stat,
                p_value=p,
                corrected_p_value=corrected_p,
                significant=corrected_p < alpha,
                effect_size=es,
                effect_magnitude=mag,
                test_name=tname,
                n_samples=n,
                confidence_interval=ci,
            )
        )

    return results


def build_method_values(runs: list[RunResult], metric: str) -> dict[str, np.ndarray]:
    method_seeds: dict[str, dict[int, float]] = {}
    for run in runs:
        if metric not in run.metrics:
            continue
        method_seeds.setdefault(run.method, {})[run.seed] = run.metrics[metric]

    seed_sets = [set(seeds.keys()) for seeds in method_seeds.values()]
    common_seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []

    values_by_method = {}
    for method, seed_vals in method_seeds.items():
        paired = [seed_vals[s] for s in common_seeds]
        if paired:
            values_by_method[method] = np.array(paired, dtype=float)

    return values_by_method


def run_statistical_analysis(
    runs: list[RunResult],
    test: str = "wilcoxon",
    alpha: float = 0.05,
    correction: str = "holm",
    rope: float = 0.01,
) -> dict:
    all_metric_keys = set()
    for run in runs:
        all_metric_keys.update(run.metrics.keys())

    n_samples = len(set(r.seed for r in runs))
    sample_warning = None
    if n_samples < 10:
        sample_warning = f"Only {n_samples} seeds. Minimum 10 recommended, 20+ for publication."

    output = {
        "config": {"test": test, "alpha": alpha, "correction": correction, "rope": rope},
        "sample_warning": sample_warning,
        "metrics": {},
        "cross_dataset": {},
    }

    for metric in sorted(all_metric_keys):
        values_by_method = build_method_values(runs, metric)

        metric_output = {"pairwise": [], "friedman": None, "nemenyi": None}

        fr = friedman_test(values_by_method, metric, alpha)
        if fr:
            metric_output["friedman"] = {
                "statistic": fr.statistic,
                "p_value": fr.p_value,
                "significant": fr.significant,
                "n_methods": fr.n_methods,
                "n_samples": fr.n_samples,
                "mean_ranks": fr.mean_ranks,
            }
            if fr.significant:
                nem = nemenyi_test(values_by_method, metric, fr, alpha)
                metric_output["nemenyi"] = {
                    "critical_difference": nem.critical_difference,
                    "mean_ranks": nem.mean_ranks,
                    "groups": nem.groups,
                }

        results = pairwise_test(
            values_by_method,
            metric,
            test=test,
            alpha=alpha,
            correction=correction,
        )

        for r in results:
            entry = {
                "method_a": r.method_a,
                "method_b": r.method_b,
                "test_used": r.test_name,
                "statistic": r.statistic,
                "p_value": r.p_value,
                "corrected_p_value": r.corrected_p_value,
                "significant": r.significant,
                "effect_size": r.effect_size,
                "effect_magnitude": r.effect_magnitude,
                "n_samples": r.n_samples,
            }
            if r.confidence_interval:
                entry["ci_lower"] = r.confidence_interval[0]
                entry["ci_upper"] = r.confidence_interval[1]
            metric_output["pairwise"].append(entry)

        try:
            bayesian_results = bayesian_signed_rank(values_by_method, metric, rope=rope)
        except ImportError:
            bayesian_results = []
        metric_output["bayesian"] = [
            {
                "method_a": r.method_a,
                "method_b": r.method_b,
                "p_a_better": r.p_a_better,
                "p_rope": r.p_rope,
                "p_b_better": r.p_b_better,
                "rope_width": r.rope_width,
                "decision": r.decision,
                "n_samples": r.n_samples,
            }
            for r in bayesian_results
        ]

        power_results = {}
        methods = sorted(values_by_method.keys())
        for m_a, m_b in combinations(methods, 2):
            a = values_by_method[m_a]
            b = values_by_method[m_b]
            n = min(len(a), len(b))
            pw = post_hoc_power(a[:n], b[:n], alpha)
            power_results[f"{m_a} vs {m_b}"] = pw
        metric_output["power"] = power_results

        output["metrics"][metric] = metric_output

    datasets = set(r.dataset for r in runs)
    if len(datasets) > 1:
        for metric in sorted(all_metric_keys):
            cd_result = cross_dataset_analysis(runs, metric, alpha)
            if cd_result:
                output["cross_dataset"][metric] = cd_result

    return output


def cross_dataset_analysis(
    runs: list[RunResult],
    metric: str,
    alpha: float = 0.05,
) -> dict | None:
    dataset_method_values: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        if metric not in run.metrics:
            continue
        ds = dataset_method_values.setdefault(run.dataset, {})
        ds.setdefault(run.method, []).append(run.metrics[metric])

    datasets = sorted(dataset_method_values.keys())
    if len(datasets) < 2:
        return None

    methods = sorted({m for ds in dataset_method_values.values() for m in ds})
    if len(methods) < 2:
        return None

    mean_per_dataset = {}
    for ds in datasets:
        ds_data = dataset_method_values[ds]
        means = {}
        for method in methods:
            if method in ds_data:
                means[method] = float(np.mean(ds_data[method]))
        if len(means) == len(methods):
            mean_per_dataset[ds] = means

    if len(mean_per_dataset) < 2:
        return None

    ds_list = sorted(mean_per_dataset.keys())
    n_ds = len(ds_list)
    n_methods = len(methods)

    rank_matrix = np.zeros((n_ds, n_methods))
    for i, ds in enumerate(ds_list):
        values = [mean_per_dataset[ds][m] for m in methods]
        rank_matrix[i] = stats.rankdata([-v for v in values])

    mean_ranks = {m: float(np.mean(rank_matrix[:, j])) for j, m in enumerate(methods)}

    if n_methods >= 3 and n_ds >= 3:
        groups = [rank_matrix[:, j] for j in range(n_methods)]
        stat, p = stats.friedmanchisquare(*groups)
    else:
        return {"mean_ranks": mean_ranks, "n_datasets": n_ds, "n_methods": n_methods}

    result = {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": float(p) < alpha,
        "mean_ranks": mean_ranks,
        "n_datasets": n_ds,
        "n_methods": n_methods,
    }

    if float(p) < alpha and n_methods >= 3:
        cd = nemenyi_cd(n_methods, n_ds, alpha)
        result["nemenyi_cd"] = cd

    return result


def significance_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def save_statistics(output: dict, results_dir: str, filename: str = "statistics.json"):
    path = Path(results_dir) / filename
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
