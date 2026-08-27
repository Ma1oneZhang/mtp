from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

EXPERIMENTS = (
    "baseline",
    "two_tap",
    "two_tap_markov",
    "two_tap_selector",
    "two_tap_selector_ngram",
    "two_tap_gated_fusion",
    "dilated",
    "multiscale",
    "gated_fusion",
    "selector",
    "selector_ngram",
    "dilated_selector",
    "dilated_selector_ngram",
    "dilated_gated_fusion",
    "combined",
    "combined_conv2x",
)
DEFAULT_EXPERIMENT = "two_tap_markov"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--anchors-per-sample", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--update-log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-final", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for label in ("target_model", "draft_model", "data"):
        path = getattr(args, label)
        if not path.is_dir():
            raise FileNotFoundError(f"{label} directory not found: {path}")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.max_length < 32:
        raise ValueError("max-length must be at least 32")
    if args.anchors_per_sample < 1:
        raise ValueError("anchors-per-sample must be positive")
    if args.update_log_interval < 0:
        raise ValueError("update-log-interval cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "each experiment must see exactly one GPU via CUDA_VISIBLE_DEVICES"
        )


def rows(data_dir: Path) -> Iterator[dict[str, Any]]:
    paths = sorted(data_dir.glob("train-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no train-*.parquet files under {data_dir}")
    while True:
        for path in paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=32):
                yield from batch.to_pylist()


def _visible_answer(content: str) -> str | None:
    if "</think>" in content:
        content = content.rsplit("</think>", maxsplit=1)[1]
    elif "<think>" in content:
        return None
    content = content.lstrip("\n")
    return content or None


def _training_anchors(row: dict[str, Any], valid_anchors: list[int]) -> list[int]:
    selected = row.get("selected_anchor_positions")
    if selected is None:
        return valid_anchors
    if not isinstance(selected, list) or not selected:
        raise ValueError("curated row must contain selected anchor positions")
    if any(
        not isinstance(anchor, int) or isinstance(anchor, bool) for anchor in selected
    ):
        raise TypeError("selected anchor positions must be integers")
    if selected != sorted(set(selected)):
        raise ValueError("selected anchor positions must be sorted and unique")
    valid = set(valid_anchors)
    invalid = [anchor for anchor in selected if anchor not in valid]
    if invalid:
        raise ValueError(
            f"selected anchors are incompatible with this tokenization: {invalid[:4]}"
        )
    return selected


def tokenize_last_answer(
    row: dict[str, Any], tokenizer: Any, max_length: int, block_size: int
) -> tuple[torch.Tensor, list[int]] | None:
    conversations = row.get("conversations")
    if row.get("status") not in {"success", "max_tokens"} or not conversations:
        return None
    messages = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in conversations
        if turn.get("role") in {"system", "user", "assistant"}
        and isinstance(turn.get("content"), str)
    ]
    assistant_indices = [
        index
        for index, message in enumerate(messages)
        if message["role"] == "assistant"
    ]
    if not assistant_indices:
        return None
    answer_index = assistant_indices[-1]
    answer = _visible_answer(messages[answer_index]["content"])
    if answer is None:
        return None
    prompt_text = tokenizer.apply_chat_template(
        messages[:answer_index],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    output_token_ids = row.get("output_token_ids")
    answer_ids = (
        [int(token_id) for token_id in output_token_ids]
        if output_token_ids is not None
        else tokenizer.encode(answer + "<|im_end|>\n", add_special_tokens=False)
    )
    full_ids = (prompt_ids + answer_ids)[:max_length]
    valid_anchors = list(range(len(prompt_ids), len(full_ids) - block_size + 1))
    if not valid_anchors:
        return None
    return (
        torch.tensor(full_ids, dtype=torch.long).unsqueeze(0),
        _training_anchors(row, valid_anchors),
    )


def build_dflash_batch(
    input_ids: torch.Tensor,
    anchors: torch.Tensor,
    block_size: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build draft inputs for a right-padded batch of sequences.

    Every anchor block has to lie in its own sample's valid region, so padded
    positions are never gathered and never become visible (a query only sees
    context strictly before its anchor).
    """
    assert input_ids.ndim == 2, "input_ids must be [batch, sequence]"
    assert anchors.ndim == 2 and anchors.shape[0] == input_ids.shape[0], (
        "anchors must be [batch, anchors_per_sample]"
    )
    device = input_ids.device
    anchor_positions = anchors.to(device=device, dtype=torch.long)
    batch, sequence_length = input_ids.shape
    anchors_per_sample = anchor_positions.shape[1]

    offsets = torch.arange(block_size, device=device).view(1, 1, -1)
    block_positions = anchor_positions.unsqueeze(-1) + offsets
    block_tokens = torch.gather(input_ids, 1, block_positions.flatten(1)).view(
        batch, anchors_per_sample, block_size
    )
    noise_ids = torch.full_like(block_tokens, mask_token_id)
    noise_ids[:, :, 0] = block_tokens[:, :, 0]

    query_length = anchors_per_sample * block_size
    context_positions = (
        torch.arange(sequence_length, device=device).view(1, -1).expand(batch, -1)
    )
    position_ids = torch.cat((context_positions, block_positions.flatten(1)), dim=1)
    query_indices = torch.arange(query_length, device=device).view(1, 1, -1, 1)
    kv_indices = torch.arange(sequence_length + query_length, device=device).view(
        1, 1, 1, -1
    )
    query_block_ids = query_indices // block_size
    anchor_for_query = anchor_positions.repeat_interleave(block_size, dim=1).view(
        batch, 1, query_length, 1
    )
    context_visible = (kv_indices < sequence_length) & (kv_indices < anchor_for_query)
    draft_block_ids = (kv_indices - sequence_length) // block_size
    same_draft_block = (kv_indices >= sequence_length) & (
        query_block_ids == draft_block_ids
    )
    return (
        noise_ids.flatten(1),
        position_ids,
        context_visible | same_draft_block,
        block_tokens,
    )


def tensor_l2_norm(tensors: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        norm = torch.linalg.vector_norm(tensor.detach(), dtype=torch.float32).item()
        total += norm * norm
    return math.sqrt(total)


def _metric_families(
    model: torch.nn.Module,
) -> dict[str, Callable[[str], bool]]:
    families = {
        "global": lambda name: True,
        "fc": lambda name: name.startswith("fc."),
        "conv_total": lambda name: ".attention_conv." in name or ".mlp_conv." in name,
        "conv_attention": lambda name: ".attention_conv." in name,
        "conv_mlp": lambda name: ".mlp_conv." in name,
        "conv_base_kernel": lambda name: ".base_kernel" in name,
        "conv_projection": lambda name: ".kernel_projection." in name,
        "fusion": lambda name: name.startswith("fc.gate_projection."),
        "selector": lambda name: "candidate_selector." in name,
        "markov": lambda name: "markov_head." in name,
        "non_conv": lambda name: (
            ".attention_conv." not in name and ".mlp_conv." not in name
        ),
    }
    for index in range(len(model.layers)):
        prefix = f"layers.{index}."
        families[f"layer_{index}_total"] = lambda name, prefix=prefix: name.startswith(
            prefix
        )
        families[f"layer_{index}_conv"] = lambda name, prefix=prefix: (
            name.startswith(prefix)
            and (".attention_conv." in name or ".mlp_conv." in name)
        )
        families[f"layer_{index}_non_conv"] = lambda name, prefix=prefix: (
            name.startswith(prefix)
            and ".attention_conv." not in name
            and ".mlp_conv." not in name
        )
    return families


def gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    named = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    return {
        family: tensor_l2_norm(gradient for name, gradient in named if predicate(name))
        for family, predicate in _metric_families(model).items()
    }


@torch.no_grad()
def capture_parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    snapshot = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not snapshot:
        raise ValueError(
            "cannot observe updates for a model without trainable parameters"
        )
    return snapshot


def _new_update_accumulator(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "parameter_squared_norm": reference.new_zeros((), dtype=torch.float32),
        "gradient_squared_norm": reference.new_zeros((), dtype=torch.float32),
        "update_squared_norm": reference.new_zeros((), dtype=torch.float32),
        "gradient_descent_dot": reference.new_zeros((), dtype=torch.float32),
        "changed_parameters": reference.new_zeros((), dtype=torch.int64),
        "parameters": reference.new_zeros((), dtype=torch.int64),
    }


@torch.no_grad()
def parameter_update_metrics(
    model: torch.nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> dict[str, Any]:
    named_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if named_parameters.keys() != snapshot.keys():
        raise RuntimeError("trainable parameters changed during the optimizer step")

    reference = next(iter(named_parameters.values()))
    families = _metric_families(model)
    accumulated = {family: _new_update_accumulator(reference) for family in families}

    for name, parameter in named_parameters.items():
        before = snapshot[name]
        if (
            before.shape != parameter.shape
            or before.dtype != parameter.dtype
            or before.device != parameter.device
        ):
            raise RuntimeError(f"parameter snapshot no longer matches {name}")
        before_float = before.float()
        update = parameter.detach().float().sub_(before_float)
        gradient = parameter.grad
        gradient_float = gradient.detach().float() if gradient is not None else None
        values = {
            "parameter_squared_norm": before_float.square().sum(),
            "gradient_squared_norm": (
                gradient_float.square().sum()
                if gradient_float is not None
                else update.new_zeros(())
            ),
            "update_squared_norm": update.square().sum(),
            "gradient_descent_dot": (
                -(gradient_float * update).sum()
                if gradient_float is not None
                else update.new_zeros(())
            ),
            "changed_parameters": torch.count_nonzero(parameter.detach() != before),
            "parameters": torch.tensor(
                parameter.numel(), device=parameter.device, dtype=torch.int64
            ),
        }
        matched = False
        for family, predicate in families.items():
            if not predicate(name):
                continue
            matched = True
            for metric_name, value in values.items():
                accumulated[family][metric_name].add_(value)
        if not matched:
            raise RuntimeError(f"trainable parameter has no metric family: {name}")

    parameter_norms: dict[str, float] = {}
    update_norms: dict[str, float] = {}
    relative_updates: dict[str, float | None] = {}
    effective_step_sizes: dict[str, float | None] = {}
    cosines: dict[str, float | None] = {}
    changed_fractions: dict[str, float | None] = {}
    update_energy_shares: dict[str, float | None] = {}
    global_update_energy = accumulated["global"]["update_squared_norm"].item()

    for family, values in accumulated.items():
        parameter_energy = values["parameter_squared_norm"].item()
        gradient_energy = values["gradient_squared_norm"].item()
        update_energy = values["update_squared_norm"].item()
        gradient_descent_dot = values["gradient_descent_dot"].item()
        if not all(
            math.isfinite(value)
            for value in (
                parameter_energy,
                gradient_energy,
                update_energy,
                gradient_descent_dot,
            )
        ):
            raise FloatingPointError(
                f"non-finite parameter update metric in group {family}"
            )
        parameter_norm = math.sqrt(parameter_energy)
        gradient_norm = math.sqrt(gradient_energy)
        update_norm = math.sqrt(update_energy)
        parameter_norms[family] = parameter_norm
        update_norms[family] = update_norm
        relative_updates[family] = (
            update_norm / parameter_norm if parameter_norm else None
        )
        effective_step_sizes[family] = (
            update_norm / gradient_norm if gradient_norm else None
        )
        cosine_denominator = gradient_norm * update_norm
        cosine = (
            gradient_descent_dot / cosine_denominator if cosine_denominator else None
        )
        cosines[family] = max(-1.0, min(1.0, cosine)) if cosine is not None else None
        parameter_count = values["parameters"].item()
        changed_fractions[family] = (
            values["changed_parameters"].item() / parameter_count
            if parameter_count
            else None
        )
        update_energy_shares[family] = (
            update_energy / global_update_energy if global_update_energy else None
        )

    return {
        "parameter_update_metrics_version": 1,
        "parameter_norm_by_group": parameter_norms,
        "update_norm_by_group": update_norms,
        "relative_update_by_group": relative_updates,
        "effective_step_size_by_group": effective_step_sizes,
        "gradient_update_cosine_by_group": cosines,
        "changed_parameter_fraction_by_group": changed_fractions,
        "update_energy_share_by_group": update_energy_shares,
    }
