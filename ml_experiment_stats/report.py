import json
from datetime import datetime, timezone
from pathlib import Path

from ml_experiment_stats.statistics import significance_marker


def load_report_data(results_dir: str) -> dict:
    results_path = Path(results_dir)
    config = json.loads((results_path / "config_used.json").read_text())
    summary = json.loads((results_path / "summary.json").read_text())

    stats_path = results_path / "statistics.json"
    statistics = json.loads(stats_path.read_text()) if stats_path.exists() else None

    return {
        "config": config,
        "summary": summary,
        "statistics": statistics,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def render_console(data: dict):
    statistics = data.get("statistics")
    if not statistics:
        return

    warning = statistics.get("sample_warning")
    if warning:
        print(f"\n  WARNING: {warning}")

    cfg = statistics["config"]
    print(f"\n  Test: {cfg['test']} | Alpha: {cfg['alpha']} | Correction: {cfg['correction']}")

    for metric, metric_data in statistics["metrics"].items():
        print(f"\n  {'=' * 60}")
        print(f"  {metric}")
        print(f"  {'=' * 60}")

        _console_friedman(metric_data)
        _console_pairwise(metric_data)

    _console_cross_dataset(statistics)


def _console_friedman(metric_data: dict):
    fr = metric_data.get("friedman")
    if not fr:
        return

    sig = "SIGNIFICANT" if fr["significant"] else "not significant"
    print(f"  Friedman: p={fr['p_value']:.4f} ({sig})")
    ranks = ", ".join(
        f"{m}={r:.1f}" for m, r in sorted(fr["mean_ranks"].items(), key=lambda x: x[1])
    )
    print(f"  Ranks: {ranks}")

    nem = metric_data.get("nemenyi")
    if not nem:
        return
    print(f"  Nemenyi CD: {nem['critical_difference']:.3f}")
    for group in nem["groups"]:
        if len(group) > 1:
            print(f"    Not significantly different: {', '.join(group)}")


def _console_pairwise(metric_data: dict):
    pairwise = metric_data.get("pairwise", [])
    if not pairwise:
        return

    print(f"\n  {'Comparison':<25} {'p-corr':>8} {'':>4} {'Effect':>7} {'Size':>10}")
    print(f"  {'-' * 58}")

    for comp in pairwise:
        pair = f"{comp['method_a']} vs {comp['method_b']}"
        p = comp["corrected_p_value"]
        marker = significance_marker(p)
        es = comp["effect_size"]
        mag = comp["effect_magnitude"]
        ci_str = ""
        if "ci_lower" in comp:
            ci_str = f"  CI: [{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}]"
        print(f"  {pair:<25} {p:>8.4f} {marker:>4} {es:>+7.3f} {mag:>10}{ci_str}")

    _console_bayesian(metric_data)
    _console_power(metric_data)


def _console_power(metric_data: dict):
    power = metric_data.get("power", {})
    if not power:
        return

    underpowered = [
        (pair, pw)
        for pair, pw in power.items()
        if pw.get("power") is not None and pw["power"] < 0.8
    ]
    if not underpowered:
        return

    print("\n  Power warnings:")
    for pair, pw in underpowered:
        rec = pw.get("recommended_n")
        rec_str = f", need n>={rec}" if rec else ""
        print(f"    {pair}: power={pw['power']:.2f}{rec_str}")


def _console_bayesian(metric_data: dict):
    bayesian = metric_data.get("bayesian", [])
    if not bayesian:
        return

    print(f"\n  {'Bayesian (ROPE)':}")
    print(f"  {'Comparison':<25} {'P(A>B)':>8} {'P(rope)':>8} {'P(B>A)':>8} {'Decision':>12}")
    print(f"  {'-' * 65}")

    for comp in bayesian:
        pair = f"{comp['method_a']} vs {comp['method_b']}"
        print(
            f"  {pair:<25} {comp['p_a_better']:>8.3f} {comp['p_rope']:>8.3f} "
            f"{comp['p_b_better']:>8.3f} {comp['decision']:>12}",
        )


def _console_cross_dataset(statistics: dict):
    cross = statistics.get("cross_dataset", {})
    if not cross:
        return

    print(f"\n  {'=' * 60}")
    print("  Cross-Dataset Analysis")
    print(f"  {'=' * 60}")

    for metric, cd_data in cross.items():
        print(f"\n  {metric}:")
        ranks = ", ".join(
            f"{m}={r:.1f}" for m, r in sorted(cd_data["mean_ranks"].items(), key=lambda x: x[1])
        )
        print(f"    Ranks across {cd_data['n_datasets']} datasets: {ranks}")
        if "statistic" in cd_data:
            sig = "SIGNIFICANT" if cd_data["significant"] else "not significant"
            print(f"    Friedman: p={cd_data['p_value']:.4f} ({sig})")
        if "nemenyi_cd" in cd_data:
            print(f"    Nemenyi CD: {cd_data['nemenyi_cd']:.3f}")


def render_markdown(data: dict) -> str:
    lines = []
    _md_header(lines, data)
    _md_method_summary(lines, data["summary"])
    if data["statistics"]:
        _md_statistical_analysis(lines, data["statistics"])
        _md_cross_dataset(lines, data["statistics"])
    _md_config_section(lines, data["config"])
    return "\n".join(lines) + "\n"


def _md_header(lines: list[str], data: dict):
    exp = data["config"].get("experiment", {})
    name = exp.get("name", "Experiment")
    lines.append(f"# {name}")
    lines.append("")
    lines.append(f"Generated: {data['timestamp']}")
    lines.append("")


def _md_method_summary(lines: list[str], summary: dict):
    lines.append("## Results")
    lines.append("")

    methods = sorted(summary.keys())
    if not methods:
        return

    metrics = sorted({m for method_data in summary.values() for m in method_data})

    lines.append(f"| Method | {' | '.join(metrics)} |")
    lines.append(f"|--------|{'|'.join(' --- ' for _ in metrics)}|")

    for method in methods:
        cells = []
        for metric in metrics:
            md = summary[method].get(metric)
            if md:
                cells.append(f"{md['mean']:.4f} +/- {md['std']:.4f}")
            else:
                cells.append("-")
        lines.append(f"| {method} | {' | '.join(cells)} |")

    lines.append("")
    first_method = methods[0]
    first_metric = metrics[0] if metrics else None
    if first_metric:
        n = summary[first_method][first_metric].get("n", "?")
        lines.append(f"n={n} seeds per method.")
        lines.append("")


def _md_statistical_analysis(lines: list[str], statistics: dict):
    lines.append("## Statistical Analysis")
    lines.append("")

    cfg = statistics.get("config", {})
    lines.append(
        f"Test: {cfg.get('test', '?')} | "
        f"Alpha: {cfg.get('alpha', '?')} | "
        f"Correction: {cfg.get('correction', '?')}"
    )
    lines.append("")

    warning = statistics.get("sample_warning")
    if warning:
        lines.append(f"> WARNING: {warning}")
        lines.append("")

    for metric, metric_data in statistics.get("metrics", {}).items():
        lines.append(f"### {metric}")
        lines.append("")
        _md_friedman(lines, metric_data)
        _md_pairwise(lines, metric_data)
        _md_bayesian(lines, metric_data)
        _md_power(lines, metric_data)


def _md_friedman(lines: list[str], metric_data: dict):
    friedman = metric_data.get("friedman")
    if not friedman:
        return

    sig = "significant" if friedman["significant"] else "not significant"
    lines.append(f"Friedman test: p={friedman['p_value']:.4f} ({sig})")
    ranks = ", ".join(
        f"{m}={r:.1f}" for m, r in sorted(friedman["mean_ranks"].items(), key=lambda x: x[1])
    )
    lines.append(f"Mean ranks: {ranks}")
    lines.append("")

    nemenyi = metric_data.get("nemenyi")
    if not nemenyi:
        return
    lines.append(f"Nemenyi CD: {nemenyi['critical_difference']:.3f}")
    for group in nemenyi.get("groups", []):
        if len(group) > 1:
            lines.append(f"- Not significantly different: {', '.join(group)}")
    lines.append("")


def _md_pairwise(lines: list[str], metric_data: dict):
    pairwise = metric_data.get("pairwise", [])
    if not pairwise:
        return

    has_ci = any("ci_lower" in c for c in pairwise)
    header = "| Comparison | p (corrected) | Sig. | Effect | Magnitude |"
    sep = "|---|---|---|---|---|"
    if has_ci:
        header += " 95% CI |"
        sep += "---|"
    lines.append(header)
    lines.append(sep)

    for comp in pairwise:
        pair = f"{comp['method_a']} vs {comp['method_b']}"
        p = comp["corrected_p_value"]
        marker = significance_marker(p)
        es = comp["effect_size"]
        mag = comp["effect_magnitude"]
        row = f"| {pair} | {p:.4f} | {marker} | {es:+.3f} | {mag} |"
        if has_ci:
            if "ci_lower" in comp:
                row += f" [{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}] |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")


def _md_power(lines: list[str], metric_data: dict):
    power = metric_data.get("power", {})
    if not power:
        return

    underpowered = [
        (pair, pw)
        for pair, pw in power.items()
        if pw.get("power") is not None and pw["power"] < 0.8
    ]
    if not underpowered:
        return

    lines.append("#### Power Analysis")
    lines.append("")
    lines.append("| Comparison | Power | Recommended n |")
    lines.append("|---|---|---|")
    for pair, pw in underpowered:
        rec = pw.get("recommended_n", "-")
        lines.append(f"| {pair} | {pw['power']:.2f} | {rec} |")
    lines.append("")


def _md_bayesian(lines: list[str], metric_data: dict):
    bayesian = metric_data.get("bayesian", [])
    if not bayesian:
        return

    lines.append("#### Bayesian Analysis (ROPE)")
    lines.append("")
    lines.append("| Comparison | P(A>B) | P(ROPE) | P(B>A) | Decision |")
    lines.append("|---|---|---|---|---|")

    for comp in bayesian:
        pair = f"{comp['method_a']} vs {comp['method_b']}"
        lines.append(
            f"| {pair} | {comp['p_a_better']:.3f} | {comp['p_rope']:.3f} "
            f"| {comp['p_b_better']:.3f} | {comp['decision']} |",
        )
    lines.append("")


def _md_cross_dataset(lines: list[str], statistics: dict):
    cross = statistics.get("cross_dataset", {})
    if not cross:
        return

    lines.append("## Cross-Dataset Analysis")
    lines.append("")

    for metric, cd_data in cross.items():
        lines.append(f"### {metric}")
        lines.append("")
        ranks = ", ".join(
            f"{m}={r:.1f}" for m, r in sorted(cd_data["mean_ranks"].items(), key=lambda x: x[1])
        )
        lines.append(f"Ranks across {cd_data['n_datasets']} datasets: {ranks}")
        lines.append("")
        if "statistic" in cd_data:
            sig = "significant" if cd_data["significant"] else "not significant"
            lines.append(f"Friedman: p={cd_data['p_value']:.4f} ({sig})")
            lines.append("")
        if "nemenyi_cd" in cd_data:
            lines.append(f"Nemenyi CD: {cd_data['nemenyi_cd']:.3f}")
            lines.append("")


def _md_config_section(lines: list[str], config: dict):
    lines.append("## Config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(config, indent=2))
    lines.append("```")


def render_json(data: dict) -> dict:
    summary = data["summary"]
    statistics = data["statistics"]

    report = {
        "experiment": data["config"].get("experiment", {}),
        "timestamp": data["timestamp"],
        "exit_codes": {
            "0": "All experiments completed, CI thresholds met (or CI disabled)",
            "1": "CI threshold check failed (significance or effect size below minimum)",
        },
        "methods": {},
        "comparisons": {},
    }

    for method, method_data in summary.items():
        report["methods"][method] = {
            metric: {"mean": vals["mean"], "std": vals["std"], "n": vals["n"]}
            for metric, vals in method_data.items()
        }

    if statistics:
        report["statistics_config"] = statistics.get("config", {})
        report["sample_warning"] = statistics.get("sample_warning")
        for metric, metric_data in statistics.get("metrics", {}).items():
            metric_report = {
                "pairwise": metric_data.get("pairwise", []),
                "friedman": metric_data.get("friedman"),
                "nemenyi": metric_data.get("nemenyi"),
                "bayesian": metric_data.get("bayesian", []),
                "power": metric_data.get("power", {}),
            }
            report["comparisons"][metric] = metric_report

        report["cross_dataset"] = statistics.get("cross_dataset", {})

    return report


def save_report(results_dir: str, baseline_dir: str = ""):
    from ml_experiment_stats.diff import generate_diff, render_diff_json, render_diff_markdown

    data = load_report_data(results_dir)
    results_path = Path(results_dir)

    diffs = generate_diff(results_dir, baseline_dir) if baseline_dir else []

    md = render_markdown(data)
    if diffs:
        md += "\n" + render_diff_markdown(diffs) + "\n"
    (results_path / "report.md").write_text(md)

    json_report = render_json(data)
    if diffs:
        json_report["diff"] = render_diff_json(diffs)
    with open(results_path / "report.json", "w") as f:
        json.dump(json_report, f, indent=2)
