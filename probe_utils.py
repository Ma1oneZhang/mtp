"""Data and metric helpers for the single-GPU DFlash training probe."""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument(
        "--draft-init",
        choices=("checkpoint", "random"),
        default="checkpoint",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("perfectblend-qwen3-4b-regen/data"),
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--anchors-per-sample", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--update-log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.target_model.is_dir():
        raise FileNotFoundError(
            f"target model directory not found: {args.target_model}"
        )
    if not args.draft_model.is_dir():
        raise FileNotFoundError(f"draft model directory not found: {args.draft_model}")
    if not args.data.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {args.data}")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.max_length < 32:
        raise ValueError("max-length must be at least 32")
    if args.anchors_per_sample < 1:
        raise ValueError("anchors-per-sample must be positive")
    if args.update_log_interval < 0:
        raise ValueError("update-log-interval cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires CUDA")


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
    """Select the visible answer used by the local non-thinking checkpoint."""
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
    if row.get("status") != "success" or not conversations:
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
    if output_token_ids is None:
        answer_ids = tokenizer.encode(answer + "<|im_end|>\n", add_special_tokens=False)
    else:
        answer_ids = [int(token_id) for token_id in output_token_ids]
    full_ids = (prompt_ids + answer_ids)[:max_length]
    response_start = len(prompt_ids)
    valid_anchors = list(range(response_start, len(full_ids) - block_size + 1))
    if not valid_anchors:
        return None
    return (
        torch.tensor(full_ids, dtype=torch.long).unsqueeze(0),
        _training_anchors(row, valid_anchors),
    )


def build_dflash_batch(
    input_ids: torch.Tensor,
    anchors: list[int],
    block_size: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten one sample's anchor blocks and build their visibility mask."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("parallel DFlash probe expects one sequence")

    device = input_ids.device
    sequence_length = input_ids.shape[1]
    anchor_positions = torch.tensor(anchors, dtype=torch.long, device=device).view(
        1, -1
    )
    offsets = torch.arange(block_size, device=device).view(1, 1, -1)
    block_positions = anchor_positions.unsqueeze(-1) + offsets
    block_tokens = torch.gather(
        input_ids,
        1,
        block_positions.flatten(1),
    ).view(1, len(anchors), block_size)

    noise_ids = torch.full_like(block_tokens, mask_token_id)
    noise_ids[:, :, 0] = block_tokens[:, :, 0]

    query_length = len(anchors) * block_size
    context_positions = torch.arange(sequence_length, device=device).view(1, -1)
    position_ids = torch.cat(
        [context_positions, block_positions.flatten(1)],
        dim=1,
    )

    query_indices = torch.arange(query_length, device=device).view(1, 1, -1, 1)
    kv_indices = torch.arange(sequence_length + query_length, device=device).view(
        1, 1, 1, -1
    )
    query_block_ids = query_indices // block_size
    anchor_for_query = anchor_positions.repeat_interleave(block_size, dim=1).view(
        1, 1, query_length, 1
    )
    # A block sees the clean prefix before its anchor and its own draft tokens.
    context_visible = (kv_indices < sequence_length) & (kv_indices < anchor_for_query)
    draft_block_ids = (kv_indices - sequence_length) // block_size
    same_draft_block = (kv_indices >= sequence_length) & (
        query_block_ids == draft_block_ids
    )
    attention_mask = context_visible | same_draft_block

    labels = block_tokens[:, :, 1:]
    return noise_ids.flatten(1), position_ids, attention_mask, labels


def tensor_l2_norm(tensors: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        norm = torch.linalg.vector_norm(tensor.detach(), dtype=torch.float32).item()
        total += norm * norm
    return math.sqrt(total)


def grouped_gradient_norms(draft: torch.nn.Module) -> dict[str, float]:
    groups: dict[str, list[torch.Tensor]] = {"fc": [], "other": []}
    for index in range(len(draft.layers)):
        groups[f"layer_{index}"] = []
    for name, parameter in draft.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("fc."):
            group = "fc"
        elif name.startswith("layers."):
            group = f"layer_{name.split('.')[1]}"
        else:
            group = "other"
        groups[group].append(parameter.grad)
    return {
        name: tensor_l2_norm(tensors) for name, tensors in groups.items() if tensors
    }


def _parameter_group(name: str) -> str:
    if name.startswith("fc."):
        return "fc"
    if name.startswith("layers."):
        return f"layer_{name.split('.')[1]}"
    return "other"


@torch.no_grad()
def capture_parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone trainable stored parameters immediately before an optimizer step."""
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
    """Measure the optimizer's actual change to the model's stored parameters."""
    named_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if named_parameters.keys() != snapshot.keys():
        raise RuntimeError("trainable parameters changed during the optimizer step")

    reference = next(iter(named_parameters.values()))
    groups = {"global": _new_update_accumulator(reference)}
    for name in named_parameters:
        group_name = _parameter_group(name)
        if group_name not in groups:
            groups[group_name] = _new_update_accumulator(reference)

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
        for group_name in ("global", _parameter_group(name)):
            for metric_name, value in values.items():
                groups[group_name][metric_name].add_(value)

    parameter_norms: dict[str, float] = {}
    update_norms: dict[str, float] = {}
    relative_updates: dict[str, float | None] = {}
    effective_step_sizes: dict[str, float | None] = {}
    cosines: dict[str, float | None] = {}
    changed_fractions: dict[str, float] = {}
    update_energy_shares: dict[str, float | None] = {}
    global_update_energy = groups["global"]["update_squared_norm"].item()

    for group_name, values in groups.items():
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
                f"non-finite parameter update metric in group {group_name}"
            )
        parameter_norm = math.sqrt(parameter_energy)
        gradient_norm = math.sqrt(gradient_energy)
        update_norm = math.sqrt(update_energy)
        parameter_norms[group_name] = parameter_norm
        update_norms[group_name] = update_norm
        relative_updates[group_name] = (
            update_norm / parameter_norm if parameter_norm else None
        )
        effective_step_sizes[group_name] = (
            update_norm / gradient_norm if gradient_norm else None
        )
        cosine_denominator = gradient_norm * update_norm
        cosine = (
            gradient_descent_dot / cosine_denominator if cosine_denominator else None
        )
        cosines[group_name] = (
            max(-1.0, min(1.0, cosine)) if cosine is not None else None
        )
        changed_fractions[group_name] = (
            values["changed_parameters"].item() / values["parameters"].item()
        )
        update_energy_shares[group_name] = (
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
