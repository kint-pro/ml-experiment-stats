import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RunResult:
    seed: int
    method: str
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    dataset: str = "default"


class ResultsCollector:
    def __init__(self, results_dir: str = "results", statistics_config=None):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.runs: list[RunResult] = []
        self.statistics_config = statistics_config

    def add(self, result: RunResult):
        self.runs.append(result)

    def save_parquet(self, filename: str = "metrics.parquet"):
        if not self.runs:
            return

        all_metric_keys = set()
        for run in self.runs:
            all_metric_keys.update(run.metrics.keys())
        metric_keys = sorted(all_metric_keys)

        data = {
            "seed": [r.seed for r in self.runs],
            "method": [r.method for r in self.runs],
            "dataset": [r.dataset for r in self.runs],
        }
        for key in metric_keys:
            data[key] = [r.metrics.get(key) for r in self.runs]

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow required: pip install kint-stats[parquet]") from None
        table = pa.table(data)
        pq.write_table(table, self.results_dir / filename)

    def save_summary(self, filename: str = "summary.json"):
        if not self.runs:
            return

        methods = sorted(set(r.method for r in self.runs))
        all_metric_keys = set()
        for run in self.runs:
            all_metric_keys.update(run.metrics.keys())
        metric_keys = sorted(all_metric_keys)

        summary = {}
        for method in methods:
            method_runs = [r for r in self.runs if r.method == method]
            method_summary = {}
            for key in metric_keys:
                values = [r.metrics[key] for r in method_runs if key in r.metrics]
                if values:
                    arr = np.array(values, dtype=float)
                    method_summary[key] = {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "n": len(values),
                    }
            summary[method] = method_summary

        path = self.results_dir / filename
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    def save_statistics(self, filename: str = "statistics.json"):
        if not self.runs:
            return
        methods = set(r.method for r in self.runs)
        if len(methods) < 2:
            return

        from ml_experiment_stats.statistics import run_statistical_analysis, save_statistics

        cfg = self.statistics_config
        test = cfg.test if cfg else "auto"
        alpha = cfg.alpha if cfg else 0.05
        correction = cfg.correction if cfg else "holm"
        rope = cfg.rope if cfg else 0.01

        output = run_statistical_analysis(
            self.runs,
            test=test,
            alpha=alpha,
            correction=correction,
            rope=rope,
        )
        save_statistics(output, str(self.results_dir), filename)

    def save(self):
        if not self.runs:
            raise RuntimeError("ResultsCollector.save() called with no results")
        self.save_parquet()
        self.save_summary()
        self.save_statistics()
