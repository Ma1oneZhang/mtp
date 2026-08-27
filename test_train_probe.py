import math
import unittest

import torch
from torch import nn

from probe_utils import (
    _training_anchors,
    capture_parameter_snapshot,
    parameter_update_metrics,
)
from train_probe import pal_objective
from train_probe_muon_dflash2_conv import GroupedDynamicCausalConv


class PalObjectiveTest(unittest.TestCase):
    def test_matches_direct_value_and_analytic_gradient(self) -> None:
        probabilities = torch.tensor(
            [[0.8, 0.6, 0.25], [0.2, 0.9, 0.7]], dtype=torch.float64
        )
        log_probabilities = probabilities.log().requires_grad_()

        loss, expected_acceptance, weights = pal_objective(log_probabilities)
        loss.backward()

        prefix_probabilities = probabilities.cumprod(dim=-1)
        direct_acceptance = 1 + prefix_probabilities.sum(dim=-1)
        direct_loss = -direct_acceptance.log().mean()
        direct_weights = (
            torch.flip(
                torch.flip(prefix_probabilities, dims=(-1,)).cumsum(dim=-1),
                dims=(-1,),
            )
            / direct_acceptance[:, None]
        )

        torch.testing.assert_close(loss.detach(), direct_loss)
        torch.testing.assert_close(expected_acceptance, direct_acceptance.mean())
        torch.testing.assert_close(weights, direct_weights.mean(dim=0))
        torch.testing.assert_close(
            log_probabilities.grad, -direct_weights / probabilities.shape[0]
        )

    def test_boundaries(self) -> None:
        certain = torch.zeros((2, 15), dtype=torch.float64)
        loss, expected_acceptance, weights = pal_objective(certain)
        self.assertAlmostEqual(loss.item(), -math.log(16))
        self.assertAlmostEqual(expected_acceptance.item(), 16)
        torch.testing.assert_close(
            weights,
            torch.arange(15, 0, -1, dtype=torch.float64) / 16,
        )

        impossible = torch.full((2, 15), -1000.0, dtype=torch.float64)
        loss, expected_acceptance, weights = pal_objective(impossible)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(expected_acceptance.item(), 1)
        self.assertEqual(torch.count_nonzero(weights).item(), 0)

    def test_rejects_invalid_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "anchors, draft_positions"):
            pal_objective(torch.zeros(3))
        with self.assertRaisesRegex(ValueError, "at least one draft position"):
            pal_objective(torch.empty(2, 0))


class DFlash2ConvolutionTest(unittest.TestCase):
    def test_identity_initialization_and_gradient(self) -> None:
        convolution = GroupedDynamicCausalConv(
            hidden_size=8, kernel_size=2, group_size=4
        ).double()
        hidden = torch.randn(2, 12, 8, dtype=torch.float64, requires_grad=True)

        prepared, post_dynamic = convolution.prepare(hidden, block_size=4)
        finished = convolution.finish(prepared, post_dynamic, block_size=4)

        torch.testing.assert_close(prepared, hidden, rtol=0, atol=0)
        torch.testing.assert_close(finished, hidden, rtol=0, atol=0)
        finished.square().mean().backward()
        self.assertGreater(
            torch.linalg.vector_norm(convolution.kernel_projection.weight.grad).item(),
            0,
        )

    def test_predecessor_tap_stays_inside_each_block(self) -> None:
        convolution = GroupedDynamicCausalConv(
            hidden_size=4, kernel_size=2, group_size=2
        ).double()
        with torch.no_grad():
            convolution.base_kernel.zero_()
            convolution.base_kernel[:, 1, :] = 1
        hidden = torch.arange(32, dtype=torch.float64).view(1, 8, 4)

        prepared, post_dynamic = convolution.prepare(hidden, block_size=4)

        expected = torch.zeros_like(hidden)
        expected[:, 1:4] = hidden[:, :3]
        expected[:, 5:8] = hidden[:, 4:7]
        torch.testing.assert_close(prepared, expected)
        self.assertEqual(torch.count_nonzero(post_dynamic).item(), 0)


class ParameterUpdateMetricsTest(unittest.TestCase):
    def test_measures_actual_group_updates_and_direction(self) -> None:
        class TinyDraft(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(2, 2, bias=False)
                self.layers = nn.ModuleList([nn.Linear(2, 2, bias=False)])
                self.norm = nn.LayerNorm(2)

        draft = TinyDraft().double()
        for parameter in draft.parameters():
            parameter.grad = torch.ones_like(parameter)
        snapshot = capture_parameter_snapshot(draft)
        with torch.no_grad():
            for parameter in draft.parameters():
                parameter.add_(parameter.grad, alpha=-0.25)

        metrics = parameter_update_metrics(draft, snapshot)

        self.assertEqual(
            set(metrics["update_norm_by_group"]),
            {"global", "fc", "layer_0", "other"},
        )
        self.assertAlmostEqual(metrics["effective_step_size_by_group"]["global"], 0.25)
        self.assertAlmostEqual(
            metrics["gradient_update_cosine_by_group"]["global"], 1.0
        )
        self.assertAlmostEqual(
            metrics["changed_parameter_fraction_by_group"]["global"], 1.0
        )
        self.assertAlmostEqual(
            sum(
                share
                for group, share in metrics["update_energy_share_by_group"].items()
                if group != "global"
            ),
            1.0,
        )


class CuratedAnchorTest(unittest.TestCase):
    def test_uses_only_valid_curated_anchors(self) -> None:
        row = {"selected_anchor_positions": [11, 15, 20]}
        self.assertEqual(_training_anchors(row, list(range(10, 21))), [11, 15, 20])

    def test_rejects_anchor_from_different_tokenization(self) -> None:
        row = {"selected_anchor_positions": [11, 21]}
        with self.assertRaisesRegex(ValueError, "incompatible"):
            _training_anchors(row, list(range(10, 21)))


if __name__ == "__main__":
    unittest.main()
