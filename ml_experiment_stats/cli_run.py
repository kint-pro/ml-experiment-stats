import sys
from collections.abc import Callable
from pathlib import Path

from ml_experiment_stats.ci import check_thresholds
from ml_experiment_stats.config import ExperimentConfig, parse_args
from ml_experiment_stats.diff import generate_diff, render_diff_console
from ml_experiment_stats.report import load_report_data, render_console, save_report
from ml_experiment_stats.results import ResultsCollector, RunResult
from ml_experiment_stats.visualize import generate_figures


def run_with(
    run_single: Callable[[ExperimentConfig, int], list[RunResult]],
):
    config, args = parse_args()
    config.output.ensure_dirs()
    config.save(Path(config.output.results_dir) / "config_used.json")

    collector = ResultsCollector(config.output.results_dir, statistics_config=config.statistics)

    seeds = config.seed.seeds()
    n_total = len(seeds)
    for i, seed in enumerate(seeds):
        print(f"[seed {i + 1}/{n_total} = {seed}]")
        results = run_single(config, seed)
        for result in results:
            collector.add(result)
            primary_metric = next(iter(result.metrics), None)
            if primary_metric:
                val = result.metrics[primary_metric]
                print(f"  {result.method}: {primary_metric}={val:.4f}")

    collector.save()
    print(f"\nResults saved to {config.output.results_dir}/")

    report_data = load_report_data(config.output.results_dir)
    render_console(report_data)

    if args.baseline_dir:
        print("\n  Diff vs baseline:")
        diffs = generate_diff(config.output.results_dir, args.baseline_dir)
        render_diff_console(diffs)

    save_report(config.output.results_dir, baseline_dir=args.baseline_dir)
    print(f"\nReport saved to {config.output.results_dir}/report.md")
    print(f"Report saved to {config.output.results_dir}/report.json")

    generate_figures(config)
    print(f"Figures saved to {config.output.figures_dir}/")

    if config.ci.enabled:
        passed = check_thresholds(config.output.results_dir, config.ci)
        if not passed:
            sys.exit(1)
