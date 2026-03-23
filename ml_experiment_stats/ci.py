import json
from pathlib import Path

from ml_experiment_stats.config import CIConfig


def _baseline_effect(comp: dict, baseline: str) -> float:
    if comp["method_a"] == baseline:
        return comp["effect_size"]
    return -comp["effect_size"]


def check_thresholds(results_dir: str, ci_config: CIConfig) -> bool:
    stats_path = Path(results_dir) / "statistics.json"
    if not stats_path.exists():
        print("CI CHECK FAIL: statistics.json not found")
        return False

    statistics = json.loads(stats_path.read_text())
    baseline = ci_config.baseline
    if not baseline:
        print("CI CHECK FAIL: ci.baseline not set")
        return False

    passed = True

    for metric, metric_data in statistics.get("metrics", {}).items():
        pairwise = metric_data.get("pairwise", [])
        baseline_comparisons = [
            c for c in pairwise if c["method_a"] == baseline or c["method_b"] == baseline
        ]

        if not baseline_comparisons:
            print(
                f"CI CHECK FAIL [{metric}]: no comparisons found for baseline '{baseline}'",
            )
            passed = False
            continue

        for comp in baseline_comparisons:
            other = comp["method_b"] if comp["method_a"] == baseline else comp["method_a"]
            directed_es = _baseline_effect(comp, baseline)

            if ci_config.fail_on_no_significance and not comp["significant"]:
                print(
                    f"CI CHECK FAIL [{metric}]: {baseline} vs {other} "
                    f"not significant (p={comp['corrected_p_value']:.4f})",
                )
                passed = False

            if ci_config.min_effect_size > 0 and directed_es < ci_config.min_effect_size:
                print(
                    f"CI CHECK FAIL [{metric}]: {baseline} vs {other} "
                    f"effect={directed_es:+.3f} < {ci_config.min_effect_size}",
                )
                passed = False

    if passed:
        print("CI CHECK PASSED: all thresholds met")

    return passed
