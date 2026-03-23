from ml_experiment_stats.config import ExperimentConfig, SeedConfig, load_config


def test_default_config():
    config = ExperimentConfig()
    assert config.seed.base == 42
    assert config.seed.n_runs == 10
    assert len(config.seed.seeds()) == 10


def test_seeds_are_deterministic():
    config = SeedConfig(base=42, n_runs=5)
    assert config.seeds() == [42, 43, 44, 45, 46]


def test_load_yaml_config(tmp_path):
    yaml_content = """
experiment:
  name: "test"
  description: "test experiment"

seed:
  base: 0
  n_runs: 3

data:
  n_train: 100
  n_val: 20
  n_test: 20
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml_content)

    config = load_config(config_path)
    assert config.name == "test"
    assert config.seed.base == 0
    assert config.seed.n_runs == 3
    assert config.data.n_train == 100
