import argparse
import sys

from ml_experiment_stats.ci import check_thresholds
from ml_experiment_stats.config import load_config
from ml_experiment_stats.diff import generate_diff, render_diff_console
from ml_experiment_stats.report import load_report_data, render_console, save_report


def cmd_report(args):
    save_report(args.results_dir, baseline_dir=args.baseline_dir or "")
    print(f"Report saved to {args.results_dir}/report.md")
    print(f"Report saved to {args.results_dir}/report.json")


def cmd_diff(args):
    diffs = generate_diff(args.current_dir, args.baseline_dir)
    render_diff_console(diffs)


def cmd_check(args):
    config = load_config(args.config)
    if not config.ci.enabled:
        print("CI checking is disabled in config (ci.enabled: false)")
        sys.exit(0)
    passed = check_thresholds(args.results_dir, config.ci)
    sys.exit(0 if passed else 1)


def cmd_summary(args):
    data = load_report_data(args.results_dir)
    render_console(data)


def main():
    parser = argparse.ArgumentParser(prog="mlstats")
    sub = parser.add_subparsers(dest="command")

    p_report = sub.add_parser("report", help="Generate report from results")
    p_report.add_argument("--results-dir", default="results")
    p_report.add_argument("--baseline-dir", default="")

    p_diff = sub.add_parser("diff", help="Compare two result directories")
    p_diff.add_argument("current_dir")
    p_diff.add_argument("baseline_dir")

    p_check = sub.add_parser("check", help="Run CI threshold checks")
    p_check.add_argument("--config", default="configs/default.yaml")
    p_check.add_argument("--results-dir", default="results")

    p_summary = sub.add_parser("summary", help="Print results summary to console")
    p_summary.add_argument("--results-dir", default="results")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "report": cmd_report,
        "diff": cmd_diff,
        "check": cmd_check,
        "summary": cmd_summary,
    }
    commands[args.command](args)
