from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from lean_reward_hacking.config import (
    ConfigError,
    ExperimentConfig,
    canonical_config,
    config_hash,
    load_config,
)


PROJECT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_repository_smoke_config_loads_with_defaults(self) -> None:
        config = load_config(PROJECT / "configs" / "toy_smoke.toml")
        self.assertEqual(config.experiment, "toy_smoke")
        self.assertEqual(config.labels.c_on_min, 0.95)
        self.assertEqual(config.statistics.bootstrap_seed, 8675309)
        self.assertEqual(config["task_dim"], 4)

    def test_canonical_hash_ignores_mapping_order(self) -> None:
        first = {
            "schema_version": 1,
            "experiment": "unit",
            "execution": "local_smoke",
            "z": [2, 1],
            "a": {"second": 2, "first": 1},
        }
        second = {
            "a": {"first": 1, "second": 2},
            "z": [2, 1],
            "execution": "local_smoke",
            "experiment": "unit",
            "schema_version": 1,
        }
        self.assertEqual(canonical_config(first), canonical_config(second))
        self.assertEqual(config_hash(first), config_hash(second))

    def test_unknown_key_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            ExperimentConfig.from_mapping(
                {
                    "schema_version": 1,
                    "experiment": "unit",
                    "execution": "local_smoke",
                    "learnng_rate": 0.1,
                }
            )

    def test_unknown_nested_key_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            ExperimentConfig.from_mapping(
                {
                    "schema_version": 1,
                    "experiment": "unit",
                    "execution": "local_smoke",
                    "labels": {"c_on_min": 0.9, "typo": 0.4},
                }
            )

    def test_invalid_toml_has_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.toml"
            path.write_text("schema_version = [", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()

