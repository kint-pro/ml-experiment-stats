import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class StatisticsConfig:
    alpha: float = 0.05
    test: str = "auto"
    correction: str = "holm"
    rope: float = 0.01


@dataclass
class CIConfig:
    enabled: bool = False
    baseline: str = ""
    fail_on_no_significance: bool = True
    min_effect_size: float = 0.0


@dataclass
class SeedConfig:
    base: int = 42
    n_runs: int = 10

    def seeds(self) -> list[int]:
        return [self.base + i for i in range(self.n_runs)]


@dataclass
class DataConfig:
    n_train: int = 1000
    n_val: int = 200
    n_test: int = 200


@dataclass
class ModelConfig:
    hidden_layers: list[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 64
    optimizer: str = "adam"
    early_stopping_patience: int = 10


@dataclass
class OutputConfig:
    results_dir: str = "results"
    figures_dir: str = "results/figures"
    save_models: bool = True
    figure_format: list[str] = field(default_factory=lambda: ["png", "pdf"])

    def ensure_dirs(self):
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.figures_dir).mkdir(parents=True, exist_ok=True)
        if self.save_models:
            Path(self.results_dir, "models").mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    name: str = "EXPERIMENT_NAME"
    description: str = "EXPERIMENT_DESCRIPTION"
    version: str = "0.1.0"
    seed: SeedConfig = field(default_factory=SeedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    ci: CIConfig = field(default_factory=CIConfig)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> dict:
        return {
            "experiment": {
                "name": self.name,
                "description": self.description,
                "version": self.version,
            },
            "seed": {"base": self.seed.base, "n_runs": self.seed.n_runs},
            "data": {
                "n_train": self.data.n_train,
                "n_val": self.data.n_val,
                "n_test": self.data.n_test,
            },
            "model": {
                "hidden_layers": self.model.hidden_layers,
                "activation": self.model.activation,
                "dropout": self.model.dropout,
            },
            "training": {
                "epochs": self.training.epochs,
                "learning_rate": self.training.learning_rate,
                "batch_size": self.training.batch_size,
                "optimizer": self.training.optimizer,
                "early_stopping_patience": self.training.early_stopping_patience,
            },
            "output": {
                "results_dir": self.output.results_dir,
                "figures_dir": self.output.figures_dir,
                "save_models": self.output.save_models,
                "figure_format": self.output.figure_format,
            },
            "statistics": {
                "alpha": self.statistics.alpha,
                "test": self.statistics.test,
                "correction": self.statistics.correction,
                "rope": self.statistics.rope,
            },
            "ci": {
                "enabled": self.ci.enabled,
                "baseline": self.ci.baseline,
                "fail_on_no_significance": self.ci.fail_on_no_significance,
                "min_effect_size": self.ci.min_effect_size,
            },
        }


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    exp = raw.get("experiment", {})
    return ExperimentConfig(
        name=exp.get("name", "EXPERIMENT_NAME"),
        description=exp.get("description", "EXPERIMENT_DESCRIPTION"),
        version=exp.get("version", "0.1.0"),
        seed=SeedConfig(**raw.get("seed", {})),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        output=OutputConfig(**raw.get("output", {})),
        statistics=StatisticsConfig(**raw.get("statistics", {})),
        ci=CIConfig(**raw.get("ci", {})),
    )


def parse_args() -> tuple[ExperimentConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--baseline-dir", type=str, default="")
    args = parser.parse_args()
    return load_config(args.config), args
