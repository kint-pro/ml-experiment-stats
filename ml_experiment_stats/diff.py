import json
from pathlib import Path


def load_summary(results_dir: str) -> dict:
    path = Path(results_dir) / "summary.json"
    return json.loads(path.read_text())


def diff_summaries(current: dict, baseline: dict) -> list[dict]:
    diffs = []

    methods = sorted(set(current.keys()) & set(baseline.keys()))
    for method in methods:
        curr_metrics = current[method]
        base_metrics = baseline[method]
        shared_metrics = sorted(set(curr_metrics.keys()) & set(base_metrics.keys()))

        for metric in shared_metrics:
            curr = curr_metrics[metric]
            base = base_metrics[metric]
            delta = curr["mean"] - base["mean"]
            pct = (delta / abs(base["mean"]) * 100) if base["mean"] != 0 else 0.0
            diffs.append(
                {
                    "method": method,
                    "metric": metric,
                    "baseline_mean": base["mean"],
                    "baseline_std": base["std"],
                    "current_mean": curr["mean"],
                    "current_std": curr["std"],
                    "delta": delta,
                    "delta_pct": pct,
                }
            )

    return diffs


def render_diff_console(diffs: list[dict]):
    if not diffs:
        print("  No comparable metrics found.")
        return

    print(f"\n  {'Method':<15} {'Metric':<15} {'Baseline':>12} {'Current':>12} {'Delta':>10}")
    print(f"  {'-' * 66}")

    for d in diffs:
        direction = "+" if d["delta"] > 0 else ""
        print(
            f"  {d['method']:<15} {d['metric']:<15} "
            f"{d['baseline_mean']:>12.4f} {d['current_mean']:>12.4f} "
            f"{direction}{d['delta']:.4f} ({d['delta_pct']:+.1f}%)",
        )


def render_diff_markdown(diffs: list[dict]) -> str:
    lines = ["## Diff vs Baseline", ""]

    if not diffs:
        lines.append("No comparable metrics found.")
        return "\n".join(lines)

    lines.append("| Method | Metric | Baseline | Current | Delta | Change |")
    lines.append("|---|---|---|---|---|---|")

    for d in diffs:
        direction = "+" if d["delta"] > 0 else ""
        lines.append(
            f"| {d['method']} | {d['metric']} "
            f"| {d['baseline_mean']:.4f} | {d['current_mean']:.4f} "
            f"| {direction}{d['delta']:.4f} | {d['delta_pct']:+.1f}% |",
        )

    lines.append("")
    return "\n".join(lines)


def render_diff_json(diffs: list[dict]) -> list[dict]:
    return diffs


def generate_diff(current_dir: str, baseline_dir: str) -> list[dict]:
    current = load_summary(current_dir)
    baseline = load_summary(baseline_dir)
    return diff_summaries(current, baseline)
