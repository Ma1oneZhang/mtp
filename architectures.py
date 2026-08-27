from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _shift_inside_blocks(blocks: torch.Tensor, offset: int) -> torch.Tensor:
    if offset == 0:
        return blocks
    if offset >= blocks.shape[2]:
        return torch.zeros_like(blocks)
    return F.pad(blocks[:, :, :-offset], (0, 0, 0, 0, offset, 0))


def grouped_dynamic_mix(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base_kernel: torch.Tensor,
    offsets: tuple[int, ...],
    group_size: int,
    block_size: int,
) -> torch.Tensor:
    if hidden.ndim != 3:
        raise ValueError("dynamic mixer expects [batch, sequence, hidden]")
    batch, sequence_length, hidden_size = hidden.shape
    if sequence_length % block_size:
        raise ValueError("sequence length must be a whole number of draft blocks")
    if hidden_size % group_size:
        raise ValueError("group size must divide hidden size")
    if len(offsets) != base_kernel.shape[0]:
        raise ValueError("offset count must match base-kernel taps")
    groups = hidden_size // group_size
    blocks = hidden.view(
        batch, sequence_length // block_size, block_size, groups, group_size
    )
    dynamic = dynamic.view(
        batch,
        sequence_length // block_size,
        block_size,
        len(offsets),
        groups,
        1,
    )
    output = torch.zeros_like(blocks)
    for tap, offset in enumerate(offsets):
        values = _shift_inside_blocks(blocks, offset)
        kernel = base_kernel[tap].view(1, 1, 1, groups, group_size)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, :, tap], values)
    return output.view(batch, sequence_length, hidden_size)


class GroupedDynamicMixer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        offsets: tuple[int, ...],
        group_size: int = 16,
    ) -> None:
        super().__init__()
        if not offsets or offsets[0] != 0 or len(set(offsets)) != len(offsets):
            raise ValueError("offsets must be unique and start with zero")
        if any(offset < 0 for offset in offsets):
            raise ValueError("offsets cannot be negative")
        if hidden_size % group_size:
            raise ValueError("group size must divide hidden size")
        self.offsets = offsets
        self.group_size = group_size
        groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.zeros(2, len(offsets), hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * len(offsets) * groups, bias=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.kernel_projection.weight)
        with torch.no_grad():
            self.base_kernel.zero_()
            self.base_kernel[:, 0, :] = 1

    def prepare(
        self, hidden: torch.Tensor, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).view(
            *hidden.shape[:-1], 2, len(self.offsets), groups
        )
        mixed = grouped_dynamic_mix(
            hidden,
            dynamic[..., 0, :, :],
            self.base_kernel[0],
            self.offsets,
            self.group_size,
            block_size,
        )
        return mixed, dynamic[..., 1, :, :]

    def finish(
        self, hidden: torch.Tensor, dynamic: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        return grouped_dynamic_mix(
            hidden,
            dynamic,
            self.base_kernel[1],
            self.offsets,
            self.group_size,
            block_size,
        )


class ConvDecoderLayer(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        block_size: int,
        offsets: tuple[int, ...],
        group_size: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_size = layer.hidden_size
        self.block_size = block_size
        self.self_attn = layer.self_attn
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.attention_conv = GroupedDynamicMixer(
            self.hidden_size, offsets, group_size=group_size
        )
        self.mlp_conv = GroupedDynamicMixer(
            self.hidden_size, offsets, group_size=group_size
        )

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attention_dynamic = self.attention_conv.prepare(
            hidden_states, self.block_size
        )
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = self.attention_conv.finish(
            hidden_states, attention_dynamic, self.block_size
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, mlp_dynamic = self.mlp_conv.prepare(
            hidden_states, self.block_size
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(
            hidden_states, mlp_dynamic, self.block_size
        )
        return residual + hidden_states


class GatedTargetFusion(nn.Module):
    def __init__(self, original: nn.Linear, num_features: int) -> None:
        super().__init__()
        hidden_size = original.out_features
        if original.in_features != num_features * hidden_size:
            raise ValueError("target feature projection has an unexpected shape")
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.original = original
        self.gate_projection = nn.Linear(original.in_features, num_features, bias=False)
        nn.init.zeros_(self.gate_projection.weight)

    def forward(self, concatenated: torch.Tensor) -> torch.Tensor:
        features = concatenated.view(
            *concatenated.shape[:-1], self.num_features, self.hidden_size
        )
        projected = torch.stack(
            [
                F.linear(
                    features[..., index, :],
                    self.original.weight[
                        :, index * self.hidden_size : (index + 1) * self.hidden_size
                    ],
                )
                for index in range(self.num_features)
            ],
            dim=-2,
        )
        weights = torch.softmax(self.gate_projection(concatenated).float(), dim=-1)
        uniform = torch.full_like(weights, 1 / self.num_features)
        residual = (
            projected * (weights - uniform).to(projected.dtype).unsqueeze(-1)
        ).sum(dim=-2)
        return self.original(concatenated) + residual


class CandidateSelector(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_size: int, rank: int = 8, top_k: int = 16
    ) -> None:
        super().__init__()
        self.rank = rank
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        with torch.no_grad():
            self.predecessor_codebook.normal_(mean=0.0, std=0.02)
            self.successor_codebook.zero_()

    def pair_scores(
        self,
        hidden: torch.Tensor,
        unary: torch.Tensor,
        candidate_ids: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> torch.Tensor:
        gate = F.embedding(
            predecessor_ids, self.predecessor_codebook
        ) * self.hidden_projection(hidden)
        successors = F.embedding(candidate_ids, self.successor_codebook)
        return unary + torch.einsum("...r,...kr->...k", gate, successors)


class VanillaMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int = 256) -> None:
        super().__init__()
        self.rank = rank
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        with torch.no_grad():
            self.predecessor_codebook.normal_(mean=0.0, std=0.02)
            self.successor_codebook.zero_()

    def forward(
        self, base_logits: torch.Tensor, predecessor_ids: torch.Tensor
    ) -> torch.Tensor:
        if base_logits.shape[:-1] != predecessor_ids.shape:
            raise ValueError("predecessor ids must match the logits batch dimensions")
        predecessor = F.embedding(predecessor_ids, self.predecessor_codebook)
        transition = F.linear(predecessor, self.successor_codebook)
        return base_logits + transition

    @torch.no_grad()
    def greedy_path(
        self, base_logits: torch.Tensor, anchor_ids: torch.Tensor
    ) -> torch.Tensor:
        previous = anchor_ids
        selected = []
        for position in range(base_logits.shape[1]):
            corrected = self(base_logits[:, position], previous)
            previous = corrected.argmax(dim=-1)
            selected.append(previous)
        return torch.stack(selected, dim=-1)


@dataclass(frozen=True)
class SelectorOutput:
    loss: torch.Tensor
    candidate_recall: torch.Tensor
    teacher_accuracy: torch.Tensor
    path_acceptance: torch.Tensor


def _ngram_energy(
    candidate_ids: torch.Tensor,
    predecessor_ids: torch.Tensor,
    sequence: torch.Tensor,
    anchors: list[int],
) -> torch.Tensor:
    if sequence.ndim != 1 or candidate_ids.ndim != 3:
        raise ValueError("ngram energy expects one sequence and [block, position, k]")
    if predecessor_ids.shape != candidate_ids.shape[:2]:
        raise ValueError("predecessor shape must match candidate block positions")
    if len(anchors) != candidate_ids.shape[0]:
        raise ValueError("anchor count must match candidate blocks")

    bigram_base = 1 << 20
    pair_keys = sequence[:-1] * bigram_base + sequence[1:]
    query_keys = predecessor_ids.unsqueeze(-1) * bigram_base + candidate_ids
    pair_indices = torch.arange(pair_keys.shape[0], device=sequence.device)
    position_offsets = torch.arange(candidate_ids.shape[1], device=sequence.device)
    limits = torch.tensor(anchors, device=sequence.device).unsqueeze(-1)
    limits = limits + position_offsets
    visible = pair_indices.view(1, 1, 1, -1) < limits.unsqueeze(-1).unsqueeze(-1)
    counts = (query_keys.unsqueeze(-1).eq(pair_keys) & visible).sum(dim=-1)
    # Preserve the original CPU log1p rounding path. Muon amplifies the tiny
    # CPU/CUDA difference enough to change the training trajectory within a
    # few optimizer steps.
    return torch.log1p(counts.cpu().float()).to(candidate_ids.device)


def selector_objective(
    selector: CandidateSelector,
    hidden: torch.Tensor,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    anchor_ids: torch.Tensor,
    sequence: torch.Tensor,
    anchors: list[int],
    ngram_beta: float,
) -> SelectorOutput:
    unary, candidates = logits.topk(selector.top_k, dim=-1)
    predecessors = torch.cat((anchor_ids[:, None], target_ids[:, :-1]), dim=-1)
    target_matches = candidates.eq(target_ids.unsqueeze(-1))
    candidate_recall = target_matches.any(dim=-1).float().mean().detach()

    # A randomly initialized drafter almost never places the teacher token in
    # a 16-way candidate set. Include it in the training set so the selector
    # receives CE supervision from step one. Candidate recall and greedy path
    # decoding continue to use the unmodified model candidates.
    training_candidates = candidates.clone()
    training_unary = unary.clone()
    missing = ~target_matches.any(dim=-1)
    training_candidates[missing, -1] = target_ids[missing]
    target_unary = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    training_unary[missing, -1] = target_unary[missing]

    scores = selector.pair_scores(
        hidden, training_unary, training_candidates, predecessors
    )
    if ngram_beta:
        scores = scores + ngram_beta * _ngram_energy(
            training_candidates,
            predecessors,
            sequence,
            anchors,
        )
    training_matches = training_candidates.eq(target_ids.unsqueeze(-1))
    target_indices = training_matches.to(torch.int64).argmax(dim=-1)
    loss = F.cross_entropy(scores.float().flatten(0, 1), target_indices.flatten())
    teacher_accuracy = scores.argmax(dim=-1).eq(target_indices).float().mean().detach()

    selected = []
    previous = anchor_ids
    for position in range(target_ids.shape[1]):
        step_scores = selector.pair_scores(
            hidden[:, position],
            unary[:, position],
            candidates[:, position],
            previous,
        )
        if ngram_beta:
            step_scores = step_scores + ngram_beta * _ngram_energy(
                candidates[:, position : position + 1],
                previous[:, None],
                sequence,
                anchors,
            ).squeeze(1)
        previous = (
            candidates[:, position]
            .gather(-1, step_scores.argmax(dim=-1, keepdim=True))
            .squeeze(-1)
        )
        selected.append(previous)
    selected_ids = torch.stack(selected, dim=-1)
    path_acceptance = (
        selected_ids.eq(target_ids).float().cumprod(dim=-1).sum(dim=-1).mean().detach()
    )
    return SelectorOutput(loss, candidate_recall, teacher_accuracy, path_acceptance)


def _wrap_convolution(
    draft: nn.Module,
    offset_schedule: list[tuple[int, ...]],
    *,
    group_size: int = 16,
) -> None:
    if len(offset_schedule) != len(draft.layers):
        raise ValueError("offset schedule must specify every draft layer")
    reference = next(draft.parameters())
    wrapped_layers = []
    for layer, offsets in zip(draft.layers, offset_schedule, strict=True):
        wrapped = ConvDecoderLayer(
            layer, draft.block_size, offsets, group_size=group_size
        )
        wrapped.attention_conv.to(reference.device, reference.dtype)
        wrapped.mlp_conv.to(reference.device, reference.dtype)
        wrapped_layers.append(wrapped)
    draft.layers = nn.ModuleList(wrapped_layers)


def configure_architecture(draft: nn.Module, experiment: str) -> dict[str, Any]:
    layer_count = len(draft.layers)
    if experiment in {
        "two_tap",
        "two_tap_markov",
        "two_tap_selector",
        "two_tap_selector_ngram",
        "two_tap_gated_fusion",
    }:
        _wrap_convolution(draft, [(0, 1)] * layer_count)
    elif experiment in {
        "dilated",
        "dilated_selector",
        "dilated_selector_ngram",
        "dilated_gated_fusion",
        "combined",
        "combined_conv2x",
    }:
        dilation = [1, 2, 4, 1, 2]
        if layer_count != len(dilation):
            raise ValueError("dilated schedule expects five draft layers")
        _wrap_convolution(
            draft,
            [(0, value) for value in dilation],
            group_size=8 if experiment == "combined_conv2x" else 16,
        )
    elif experiment == "multiscale":
        _wrap_convolution(draft, [(0, 1, 2, 4)] * layer_count)

    if experiment in {
        "gated_fusion",
        "two_tap_gated_fusion",
        "dilated_gated_fusion",
        "combined",
        "combined_conv2x",
    }:
        original_fc = draft.fc
        draft.fc = GatedTargetFusion(original_fc, len(draft.target_layer_ids)).to(
            original_fc.weight.device, original_fc.weight.dtype
        )

    if experiment in {
        "selector",
        "selector_ngram",
        "two_tap_selector",
        "two_tap_selector_ngram",
        "dilated_selector",
        "dilated_selector_ngram",
        "combined",
        "combined_conv2x",
    }:
        reference = next(draft.parameters())
        draft.candidate_selector = CandidateSelector(
            draft.config.vocab_size, draft.config.hidden_size
        ).to(reference.device, reference.dtype)

    if experiment == "two_tap_markov":
        reference = next(draft.parameters())
        draft.markov_head = VanillaMarkovHead(draft.config.vocab_size).to(
            reference.device, reference.dtype
        )

    trainable = sum(parameter.numel() for parameter in draft.parameters())
    return {
        "experiment": experiment,
        "trainable_parameters": trainable,
        "markov_head_type": "vanilla" if hasattr(draft, "markov_head") else None,
        "markov_rank": draft.markov_head.rank
        if hasattr(draft, "markov_head")
        else None,
        "selector_rank": 8 if hasattr(draft, "candidate_selector") else None,
        "selector_top_k": 16 if hasattr(draft, "candidate_selector") else None,
        "ngram_beta": 1.0
        if experiment
        in {
            "selector_ngram",
            "two_tap_selector_ngram",
            "dilated_selector_ngram",
            "combined",
            "combined_conv2x",
        }
        else 0.0,
    }
