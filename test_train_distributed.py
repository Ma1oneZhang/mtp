import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from train_distributed import (
    configure_target_layers,
    directory_size_bytes,
    scheduled_learning_rate,
)
from train_distributed import (
    parse_args as parse_distributed_args,
)
from train_experiment import proxy_acceptance_loss
from training_utils import parse_args as parse_single_gpu_args


class DefaultArchitectureTest(unittest.TestCase):
    def test_single_gpu_trainer_defaults_to_two_tap_markov(self) -> None:
        argv = [
            "train_experiment.py",
            "--target-model",
            "target",
            "--draft-model",
            "draft",
            "--data",
            "data",
            "--output-dir",
            "output",
        ]
        with patch("sys.argv", argv):
            args = parse_single_gpu_args()
        self.assertEqual(args.experiment, "two_tap_markov")

    def test_distributed_trainer_defaults_to_two_tap_markov(self) -> None:
        argv = [
            "train_distributed.py",
            "--target-model",
            "target",
            "--draft-model",
            "draft",
            "--data",
            "data",
            "--output-dir",
            "output",
        ]
        with patch("sys.argv", argv):
            args = parse_distributed_args()
        self.assertEqual(args.experiment, "two_tap_markov")


class LearningRateScheduleTest(unittest.TestCase):
    def test_warmup_and_cosine_endpoints(self) -> None:
        base = 6e-4
        self.assertAlmostEqual(
            scheduled_learning_rate(1, 100, base, 10, 0.1), base / 10
        )
        self.assertAlmostEqual(scheduled_learning_rate(10, 100, base, 10, 0.1), base)
        self.assertAlmostEqual(
            scheduled_learning_rate(100, 100, base, 10, 0.1), base * 0.1
        )

    def test_zero_warmup_starts_at_base_rate(self) -> None:
        self.assertAlmostEqual(scheduled_learning_rate(0, 20, 1e-3, 0, 0.2), 1e-3)


class StorageBudgetTest(unittest.TestCase):
    def test_directory_size_includes_nested_files(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "nested"
            nested.mkdir()
            (root / "first.bin").write_bytes(b"123")
            (nested / "second.bin").write_bytes(b"4567")
            self.assertEqual(directory_size_bytes(root), 7)


class ProxyAcceptanceLossTest(unittest.TestCase):
    def test_matches_direct_probability_formula_and_backpropagates(self) -> None:
        logits = torch.tensor(
            [
                [[2.0, 0.0], [0.5, 1.5], [1.0, -1.0]],
                [[0.0, 1.0], [2.0, -0.5], [-1.0, 1.0]],
            ],
            requires_grad=True,
        )
        target_ids = torch.tensor([[0, 1, 0], [1, 0, 1]])

        loss, proxy = proxy_acceptance_loss(logits, target_ids)

        probabilities = (
            logits.softmax(dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        )
        expected_per_anchor = 1 + probabilities.cumprod(dim=-1).sum(dim=-1)
        self.assertTrue(torch.allclose(proxy, expected_per_anchor.mean()))
        self.assertTrue(torch.allclose(loss, -expected_per_anchor.log().mean()))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_perfect_predictions_approach_full_block_acceptance(self) -> None:
        logits = torch.full((2, 15, 3), -30.0)
        target_ids = torch.zeros((2, 15), dtype=torch.long)
        logits[:, :, 0] = 30.0

        loss, proxy = proxy_acceptance_loss(logits, target_ids)

        self.assertAlmostEqual(proxy.item(), 16.0, places=5)
        self.assertAlmostEqual(
            loss.item(), -torch.log(torch.tensor(16.0)).item(), places=5
        )


class TargetLayerConfigurationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.draft = SimpleNamespace(target_layer_ids=[1, 9, 17, 25, 33])
        self.target = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=36))

    def test_overrides_target_layers(self) -> None:
        selected = configure_target_layers(self.draft, self.target, [1, 2, 3, 4, 5])
        self.assertEqual(selected, [1, 2, 3, 4, 5])
        self.assertEqual(self.draft.target_layer_ids, selected)

    def test_rejects_invalid_layer_sets(self) -> None:
        invalid = (
            [],
            [1, 2, 3, 4, 4],
            [1, 2, 3, 4, 36],
        )
        for layer_ids in invalid:
            with self.subTest(layer_ids=layer_ids), self.assertRaises(ValueError):
                configure_target_layers(self.draft, self.target, layer_ids)

    def test_expands_projection_without_changing_existing_output(self) -> None:
        hidden_size = 2
        original_fc = torch.nn.Linear(5 * hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            original_fc.weight.copy_(
                torch.arange(original_fc.weight.numel()).reshape_as(original_fc.weight)
            )
        self.draft.fc = original_fc
        self.target.config.hidden_size = hidden_size
        requested = [1, 5, 9, 13, 17, 21, 25, 29, 33]
        original_input = torch.arange(5 * hidden_size, dtype=torch.float32)[None]
        expanded_input = torch.zeros(1, len(requested) * hidden_size)
        for old_index, layer_id in enumerate(self.draft.target_layer_ids):
            new_index = requested.index(layer_id)
            expanded_input[
                :, new_index * hidden_size : (new_index + 1) * hidden_size
            ] = original_input[
                :, old_index * hidden_size : (old_index + 1) * hidden_size
            ]
        expected = original_fc(original_input)
        rng_before = torch.random.get_rng_state()

        selected = configure_target_layers(self.draft, self.target, requested)

        self.assertEqual(selected, requested)
        self.assertEqual(self.draft.fc.in_features, 9 * hidden_size)
        self.assertTrue(torch.equal(self.draft.fc(expanded_input), expected))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        for new_index, layer_id in enumerate(requested):
            if layer_id in (1, 9, 17, 25, 33):
                continue
            block = self.draft.fc.weight[
                :, new_index * hidden_size : (new_index + 1) * hidden_size
            ]
            self.assertEqual(torch.count_nonzero(block).item(), 0)


if __name__ == "__main__":
    unittest.main()
