# ml-experiment-stats

Statistical analysis engine for ML experiments with multi-seed evaluation.

## Install

```bash
pip install ml-experiment-stats              # core: numpy, scipy, pyyaml
pip install ml-experiment-stats[parquet]     # + pyarrow
pip install ml-experiment-stats[plots]       # + matplotlib
pip install ml-experiment-stats[bayesian]    # + baycomp
pip install ml-experiment-stats[all]         # everything
```

## Usage

### SDK

```python
from ml_experiment_stats import RunResult, ResultsCollector, ExperimentConfig
from ml_experiment_stats.statistics import run_statistical_analysis
from ml_experiment_stats.report import save_report

collector = ResultsCollector("results/")
collector.add(RunResult(seed=42, method="baseline", metrics={"mse": 0.12}))
collector.add(RunResult(seed=42, method="proposed", metrics={"mse": 0.08}))
collector.save()
save_report("results/")
```

### CLI

```bash
mlstats summary --results-dir results/
mlstats report --results-dir results/
mlstats diff results_new/ results_baseline/
mlstats check --config experiment.yaml --results-dir results/
```

### Orchestrator

```python
from ml_experiment_stats import ExperimentConfig, RunResult, set_seed
from ml_experiment_stats.cli_run import run_with

def run_single(config: ExperimentConfig, seed: int) -> list[RunResult]:
    set_seed(seed)
    # your experiment logic here
    return [RunResult(seed=seed, method="my_method", metrics={"acc": 0.95})]

run_with(run_single)
```

## Statistical Methods

- **Pairwise**: Wilcoxon signed-rank, paired t-test, auto (Shapiro-Wilk selection)
- **Omnibus**: Friedman test, Nemenyi post-hoc with Critical Difference diagrams
- **Bayesian**: Signed-rank test with ROPE (Region of Practical Equivalence)
- **Effect sizes**: Cliff's delta (non-parametric), Cohen's d (parametric)
- **Corrections**: Holm-Bonferroni, Bonferroni
- **Confidence intervals**: BCa bootstrap
- **Power analysis**: Post-hoc power with recommended sample size
- **Multi-dataset**: Cross-dataset Friedman analysis (Demsar 2006)

## Output

`make run` / `run_with()` produces:

| File | Format | For |
|---|---|---|
| `summary.json` | JSON | Per-method mean/std/min/max |
| `statistics.json` | JSON | All pairwise tests, Friedman, Bayesian, power |
| `report.json` | JSON | Structured report for LLM agents |
| `report.md` | Markdown | Human-readable report |
| `figures/` | PNG/PDF | Bar plots, per-seed, heatmaps, CD diagrams |

## License

[Apache 2.0](LICENSE)
