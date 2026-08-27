import unittest
from itertools import pairwise
from types import SimpleNamespace

import torch
from torch import nn

from architectures import (
    CandidateSelector,
    ConvDecoderLayer,
    GatedTargetFusion,
    GroupedDynamicMixer,
    VanillaMarkovHead,
    _ngram_energy,
    configure_architecture,
    selector_objective,
)


class DummyDraftLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = nn.Identity()
        self.mlp = nn.Identity()
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()


class DummyDraft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_size = 16
        self.layers = nn.ModuleList([DummyDraftLayer(hidden_size) for _ in range(5)])
        self.block_size = 16
        self.target_layer_ids = [1, 9, 17, 25, 33]
        self.fc = nn.Linear(5 * hidden_size, hidden_size, bias=False)
        self.config = SimpleNamespace(vocab_size=32, hidden_size=hidden_size)


class ArchitectureConfigurationTest(unittest.TestCase):
    def test_two_tap_markov_combination(self) -> None:
        draft = DummyDraft()
        metadata = configure_architecture(draft, "two_tap_markov")
        self.assertTrue(
            all(isinstance(layer, ConvDecoderLayer) for layer in draft.layers)
        )
        self.assertIsInstance(draft.markov_head, VanillaMarkovHead)
        self.assertEqual(metadata["markov_rank"], 256)

    def test_markov_head_starts_as_identity_and_uses_sequential_predecessors(
        self,
    ) -> None:
        head = VanillaMarkovHead(vocab_size=5, rank=2).double()
        base = torch.zeros(1, 2, 5, dtype=torch.float64)
        predecessor = torch.tensor([[1, 2]])
        torch.testing.assert_close(head(base, predecessor), base)
        with torch.no_grad():
            head.predecessor_codebook.zero_()
            head.successor_codebook.zero_()
            head.predecessor_codebook[1, 0] = 1
            head.predecessor_codebook[3, 1] = 1
            head.successor_codebook[3, 0] = 2
            head.successor_codebook[4, 1] = 2
        selected = head.greedy_path(base, torch.tensor([1]))
        torch.testing.assert_close(selected, torch.tensor([[3, 4]]))

    def test_two_tap_selector_ngram_combination(self) -> None:
        draft = DummyDraft()
        metadata = configure_architecture(draft, "two_tap_selector_ngram")
        self.assertTrue(
            all(isinstance(layer, ConvDecoderLayer) for layer in draft.layers)
        )
        self.assertTrue(hasattr(draft, "candidate_selector"))
        self.assertEqual(metadata["ngram_beta"], 1.0)

    def test_two_tap_gated_fusion_combination(self) -> None:
        draft = DummyDraft()
        metadata = configure_architecture(draft, "two_tap_gated_fusion")
        self.assertIsInstance(draft.fc, GatedTargetFusion)
        self.assertFalse(hasattr(draft, "candidate_selector"))
        self.assertEqual(metadata["ngram_beta"], 0.0)


class DynamicMixerTest(unittest.TestCase):
    def test_identity_initialization_and_gradient(self) -> None:
        mixer = GroupedDynamicMixer(8, (0, 1, 2, 4), group_size=4).double()
        hidden = torch.randn(2, 16, 8, dtype=torch.float64, requires_grad=True)
        prepared, dynamic = mixer.prepare(hidden, block_size=8)
        finished = mixer.finish(prepared, dynamic, block_size=8)
        torch.testing.assert_close(prepared, hidden, rtol=0, atol=0)
        torch.testing.assert_close(finished, hidden, rtol=0, atol=0)
        finished.square().mean().backward()
        self.assertGreater(mixer.kernel_projection.weight.grad.norm().item(), 0)

    def test_dilated_tap_does_not_cross_blocks(self) -> None:
        mixer = GroupedDynamicMixer(4, (0, 4), group_size=2).double()
        with torch.no_grad():
            mixer.base_kernel.zero_()
            mixer.base_kernel[:, 1] = 1
        hidden = torch.arange(64, dtype=torch.float64).view(1, 16, 4)
        prepared, _ = mixer.prepare(hidden, block_size=8)
        expected = torch.zeros_like(hidden)
        expected[:, 4:8] = hidden[:, :4]
        expected[:, 12:16] = hidden[:, 8:12]
        torch.testing.assert_close(prepared, expected)


class FusionTest(unittest.TestCase):
    def test_initial_output_matches_original_projection(self) -> None:
        torch.manual_seed(7)
        original = nn.Linear(12, 4, bias=False).double()
        fusion = GatedTargetFusion(original, num_features=3).double()
        hidden = torch.randn(2, 5, 12, dtype=torch.float64)
        torch.testing.assert_close(
            fusion(hidden), original(hidden), rtol=1e-14, atol=1e-14
        )

        fusion(hidden).square().mean().backward()
        self.assertGreater(fusion.gate_projection.weight.grad.norm().item(), 0)


class SelectorTest(unittest.TestCase):
    def test_vectorized_ngram_matches_prefix_bigram_counts(self) -> None:
        sequence = torch.tensor([4, 7, 4, 7, 9, 4, 7, 3])
        anchors = [2, 4]
        predecessors = torch.tensor([[4, 7, 9], [9, 4, 7]])
        candidates = torch.tensor(
            [
                [[7, 9, 3], [4, 9, 7], [4, 7, 3]],
                [[4, 7, 3], [7, 9, 3], [4, 9, 3]],
            ]
        )
        actual = _ngram_energy(candidates, predecessors, sequence, anchors)
        expected = torch.zeros_like(actual)
        for block, anchor in enumerate(anchors):
            for position in range(candidates.shape[1]):
                prefix = sequence[: anchor + position + 1].tolist()
                pairs = list(pairwise(prefix))
                for index, candidate in enumerate(candidates[block, position]):
                    pair = (predecessors[block, position].item(), candidate.item())
                    expected[block, position, index] = torch.log1p(
                        torch.tensor(pairs.count(pair), dtype=torch.float32)
                    )
        torch.testing.assert_close(actual, expected)

    def test_identity_scores_and_gradient(self) -> None:
        selector = CandidateSelector(32, 8, rank=4, top_k=3).double()
        hidden = torch.randn(2, 5, 8, dtype=torch.float64)
        unary = torch.randn(2, 5, 3, dtype=torch.float64)
        candidates = torch.randint(0, 32, (2, 5, 3))
        predecessors = torch.randint(0, 32, (2, 5))
        scores = selector.pair_scores(hidden, unary, candidates, predecessors)
        torch.testing.assert_close(scores, unary, rtol=0, atol=0)
        scores.square().mean().backward()
        self.assertGreater(selector.successor_codebook.grad.norm().item(), 0)

    def test_missing_teacher_candidate_still_trains_selector(self) -> None:
        selector = CandidateSelector(32, 8, rank=4, top_k=3).double()
        hidden = torch.randn(2, 4, 8, dtype=torch.float64)
        logits = torch.full((2, 4, 32), -10.0, dtype=torch.float64)
        logits[..., :3] = 1.0
        targets = torch.full((2, 4), 31)
        output = selector_objective(
            selector,
            hidden,
            logits,
            targets,
            torch.tensor([4, 5]),
            torch.arange(16),
            [4, 5],
            ngram_beta=0.0,
        )
        self.assertEqual(output.candidate_recall.item(), 0)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertGreater(selector.successor_codebook.grad.norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
