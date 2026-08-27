from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from architectures import configure_architecture
from distributed_data import RankRowStream
from train_experiment import (
    OptimizerBundle,
    atomic_json,
    build_optimizer,
    compute_loss,
    load_models,
    target_features,
)
from training_utils import (
    DEFAULT_EXPERIMENT,
    EXPERIMENTS,
    capture_parameter_snapshot,
    gradient_norms,
    parameter_update_metrics,
    tokenize_last_answer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--target-layers", type=int, nargs="+")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--anchors-per-sample", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--holdout-shards", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=15_000)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--max-storage-gib", type=float, default=185.0)
    parser.add_argument("--gradient-log-interval", type=int, default=100)
    parser.add_argument("--update-log-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--bucket-cap-mb", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--loss-objective", choices=("ce", "pal10"), default="ce")
    parser.add_argument("--allow-loss-objective-change", action="store_true")
    parser.add_argument("--allow-world-size-change", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, world_size: int) -> None:
    for label in ("target_model", "draft_model", "data"):
        path = getattr(args, label)
        if not path.is_dir():
            raise FileNotFoundError(f"{label} directory not found: {path}")
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"expected {args.expected_world_size} ranks, torchrun started {world_size}"
        )
    if args.steps is not None and args.steps < 1:
        raise ValueError("steps must be positive")
    if args.stop_step is not None and args.stop_step < 1:
        raise ValueError("stop-step must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.max_length < 32:
        raise ValueError("max-length must be at least 32")
    if args.anchors_per_sample < 1:
        raise ValueError("anchors-per-sample must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.checkpoint_interval < 1:
        raise ValueError("checkpoint-interval must be positive")
    if args.max_storage_gib <= 0:
        raise ValueError("max-storage-gib must be positive")
    if args.storage_root is not None:
        if not args.storage_root.is_dir():
            raise FileNotFoundError(
                f"storage-root directory not found: {args.storage_root}"
            )
        try:
            args.output_dir.resolve().relative_to(args.storage_root.resolve())
        except ValueError as error:
            raise ValueError("output-dir must be inside storage-root") from error
    if args.gradient_log_interval < 1 or args.log_interval < 1:
        raise ValueError("logging intervals must be positive")
    if args.update_log_interval < 0:
        raise ValueError("update-log-interval cannot be negative")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps cannot be negative")
    if not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("min-lr-ratio must be in [0, 1]")
    if args.max_grad_norm < 0:
        raise ValueError("max-grad-norm cannot be negative")
    if args.bucket_cap_mb < 1:
        raise ValueError("bucket-cap-mb must be positive")


def scheduled_learning_rate(
    step: int,
    total_steps: int,
    base_learning_rate: float,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if warmup_steps and step <= warmup_steps:
        return base_learning_rate * step / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_learning_rate * (min_lr_ratio + (1 - min_lr_ratio) * cosine)


def configure_target_layers(
    draft: torch.nn.Module,
    target: torch.nn.Module,
    requested: list[int] | None,
) -> list[int]:
    current = list(draft.target_layer_ids)
    if requested is None:
        return current
    if not requested:
        raise ValueError("target-layers cannot be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("target-layers must be unique")
    layer_count = target.config.num_hidden_layers
    if any(layer_id < 0 or layer_id >= layer_count for layer_id in requested):
        raise ValueError(f"target-layers must be in [0, {layer_count})")
    if len(requested) != len(current):
        original_fc = draft.fc
        if not isinstance(original_fc, torch.nn.Linear):
            raise TypeError("changing target-layer count requires a linear draft.fc")
        hidden_size = target.config.hidden_size
        expected_in_features = len(current) * hidden_size
        if original_fc.in_features != expected_in_features:
            raise ValueError(
                "draft.fc input width does not match the configured target layers"
            )
        cuda_devices = [original_fc.weight.device] if original_fc.weight.is_cuda else []
        with torch.random.fork_rng(devices=cuda_devices):
            resized_fc = torch.nn.Linear(
                len(requested) * hidden_size,
                original_fc.out_features,
                bias=original_fc.bias is not None,
                device=original_fc.weight.device,
                dtype=original_fc.weight.dtype,
            )
        old_offsets = {layer_id: index for index, layer_id in enumerate(current)}
        with torch.no_grad():
            resized_fc.weight.zero_()
            for new_index, layer_id in enumerate(requested):
                old_index = old_offsets.get(layer_id)
                if old_index is None:
                    continue
                resized_fc.weight[
                    :,
                    new_index * hidden_size : (new_index + 1) * hidden_size,
                ].copy_(
                    original_fc.weight[
                        :,
                        old_index * hidden_size : (old_index + 1) * hidden_size,
                    ]
                )
            if resized_fc.bias is not None:
                resized_fc.bias.copy_(original_fc.bias)
        resized_fc.weight.requires_grad_(original_fc.weight.requires_grad)
        if resized_fc.bias is not None:
            resized_fc.bias.requires_grad_(original_fc.bias.requires_grad)
        draft.fc = resized_fc
    draft.target_layer_ids = list(requested)
    return list(requested)


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def runtime_state(stream: RankRowStream, anchor_rng: random.Random) -> dict[str, Any]:
    return {
        "data": stream.state_dict(),
        "anchor_rng": anchor_rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }


def restore_runtime_state(
    state: dict[str, Any], stream: RankRowStream, anchor_rng: random.Random
) -> None:
    stream.load_state_dict(state["data"])
    anchor_rng.setstate(state["anchor_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])


def save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    total_steps: int,
    draft: torch.nn.Module,
    optimizer: OptimizerBundle,
    stream: RankRowStream,
    anchor_rng: random.Random,
    architecture: dict[str, Any],
    loss_objective: str,
    rank: int,
    world_size: int,
    storage_root: Path | None,
    max_storage_bytes: int,
) -> Path | None:
    local_runtime = runtime_state(stream, anchor_rng)
    gathered_runtime = [None] * world_size if rank == 0 else None
    dist.gather_object(local_runtime, gathered_runtime, dst=0)
    destination = output_dir / f"checkpoint-step-{step:09d}.pt"
    save_error = [None]
    if rank == 0:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            checkpoint = {
                "step": step,
                "total_steps": total_steps,
                "architecture": architecture,
                "loss_objective": loss_objective,
                "world_size": world_size,
                "model": cpu_tree(draft.state_dict()),
                "optimizers": cpu_tree(optimizer.state_dict()),
                "runtime_by_rank": gathered_runtime,
            }
            torch.save(checkpoint, temporary)
            os.replace(temporary, destination)
            if storage_root is not None:
                used_bytes = directory_size_bytes(storage_root)
                if used_bytes > max_storage_bytes:
                    destination.unlink()
                    raise OSError(
                        f"storage usage {used_bytes / 1024**3:.2f} GiB exceeds "
                        f"the {max_storage_bytes / 1024**3:.2f} GiB limit"
                    )
        except (OSError, RuntimeError) as error:
            temporary.unlink(missing_ok=True)
            save_error[0] = f"checkpoint save failed: {error}"
    dist.broadcast_object_list(save_error, src=0)
    if save_error[0] is not None:
        raise OSError(save_error[0])
    return destination if rank == 0 else None


def load_checkpoint(
    path: Path,
    draft: torch.nn.Module,
    optimizer: OptimizerBundle,
    stream: RankRowStream,
    anchor_rng: random.Random,
    architecture: dict[str, Any],
    rank: int,
    world_size: int,
    total_steps: int,
    loss_objective: str,
    allow_loss_objective_change: bool,
    allow_world_size_change: bool,
) -> tuple[int, str, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["architecture"] != architecture:
        raise ValueError("checkpoint architecture does not match this run")
    if checkpoint["total_steps"] != total_steps:
        raise ValueError("checkpoint total_steps does not match this run")
    checkpoint_loss_objective = checkpoint.get("loss_objective", "ce")
    if checkpoint_loss_objective != loss_objective and not allow_loss_objective_change:
        raise ValueError(
            "checkpoint loss objective does not match this run; "
            "pass --allow-loss-objective-change for an intentional transition"
        )
    checkpoint_world_size = checkpoint.get(
        "world_size", len(checkpoint["runtime_by_rank"])
    )
    if checkpoint_world_size != world_size and not allow_world_size_change:
        raise ValueError(
            "checkpoint world size does not match this run; "
            "pass --allow-world-size-change for an intentional repartition"
        )
    if rank >= len(checkpoint["runtime_by_rank"]):
        raise ValueError("checkpoint has no runtime state for this rank")
    draft.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizers"])
    restore_runtime_state(checkpoint["runtime_by_rank"][rank], stream, anchor_rng)
    return int(checkpoint["step"]), checkpoint_loss_objective, checkpoint_world_size


def reduced_metrics(output: Any, world_size: int) -> torch.Tensor:
    values = torch.cat(
        (
            torch.stack(
                (
                    output.objective.detach(),
                    output.plain_ce,
                    output.pal_loss,
                    output.proxy_acceptance_length,
                    output.greedy_acceptance_length,
                    output.selector_loss,
                    output.selector_candidate_recall,
                    output.selector_teacher_accuracy,
                    output.selector_path_acceptance_length,
                )
            ).float(),
            output.position_ce.float(),
            output.position_accuracy.float(),
        )
    )
    dist.reduce(values, dst=0, op=dist.ReduceOp.SUM)
    return values / world_size


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        validate_args(args, world_size)
        torch.manual_seed(args.seed)
        tokenizer, target, draft = load_models(args)
        target_layer_ids = configure_target_layers(draft, target, args.target_layers)
        architecture = configure_architecture(draft, args.experiment)
        architecture["target_layer_ids"] = target_layer_ids
        stream = RankRowStream(
            args.data, rank, world_size, holdout_shards=args.holdout_shards
        )
        total_steps = args.steps or math.ceil(
            stream.total_rows * args.epochs / world_size
        )
        anchor_rng = random.Random(args.seed + rank)
        optimizer = build_optimizer(draft, args.learning_rate)

        start_step = 0
        resumed_loss_objective = None
        resumed_world_size = None
        if args.resume:
            start_step, resumed_loss_objective, resumed_world_size = load_checkpoint(
                args.resume,
                draft,
                optimizer,
                stream,
                anchor_rng,
                architecture,
                rank,
                world_size,
                total_steps,
                args.loss_objective,
                args.allow_loss_objective_change,
                args.allow_world_size_change,
            )
        final_step = args.stop_step if args.stop_step is not None else total_steps
        if not start_step < final_step <= total_steps:
            raise ValueError(
                "stop-step must be after the resume step and at most steps"
            )
        distributed_draft = DistributedDataParallel(
            draft,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=args.bucket_cap_mb,
        )

        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(
                args.output_dir / "run.json",
                {
                    **architecture,
                    "world_size": world_size,
                    "training_rows": stream.total_rows,
                    "holdout_shards": args.holdout_shards,
                    "epochs_requested": args.epochs,
                    "total_steps": total_steps,
                    "stop_step": final_step,
                    "start_step": start_step,
                    "loss_objective": args.loss_objective,
                    "resumed_loss_objective": resumed_loss_objective,
                    "resumed_world_size": resumed_world_size,
                    "world_size_change_allowed": args.allow_world_size_change,
                    "loss_objective_change_allowed": (args.allow_loss_objective_change),
                    "anchors_per_sample": args.anchors_per_sample,
                    "max_length": args.max_length,
                    "learning_rate": args.learning_rate,
                    "warmup_steps": args.warmup_steps,
                    "min_lr_ratio": args.min_lr_ratio,
                    "max_grad_norm": args.max_grad_norm,
                    "checkpoint_interval": args.checkpoint_interval,
                    "storage_root": (
                        str(args.storage_root)
                        if args.storage_root is not None
                        else None
                    ),
                    "max_storage_gib": args.max_storage_gib,
                    "gradient_log_interval": args.gradient_log_interval,
                    "update_log_interval": args.update_log_interval,
                    "bucket_cap_mb": args.bucket_cap_mb,
                    "seed": args.seed,
                    "git_revision": os.environ.get("MTP_RUN_REVISION"),
                    "target_model": args.target_model.name,
                    "draft_model": args.draft_model.name,
                    "data": str(args.data),
                },
            )
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        started_run = time.time()
        last_checkpoint = None

        for step in range(start_step + 1, final_step + 1):
            torch.cuda.synchronize()
            started_step = time.perf_counter()
            prepared = None
            skipped_rows = 0
            row = None
            while prepared is None:
                row = next(stream)
                prepared = tokenize_last_answer(
                    row, tokenizer, args.max_length, draft.block_size
                )
                if prepared is None:
                    skipped_rows += 1
            input_ids, valid_anchors = prepared
            input_ids = input_ids.to("cuda")
            anchors = anchor_rng.sample(
                valid_anchors, min(args.anchors_per_sample, len(valid_anchors))
            )
            features = target_features(target, input_ids, draft.target_layer_ids)
            optimizer.zero_grad(set_to_none=True)
            output = compute_loss(
                target=target,
                draft=draft,
                input_ids=input_ids,
                features=features,
                anchors=anchors,
                ngram_beta=architecture["ngram_beta"],
                loss_objective=args.loss_objective,
                draft_forward=distributed_draft,
            )
            finite = torch.isfinite(output.objective.detach()).to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("non-finite objective on at least one rank")
            output.objective.backward()
            if any(parameter.grad is not None for parameter in target.parameters()):
                raise RuntimeError("frozen target unexpectedly received gradients")

            if args.max_grad_norm:
                global_grad_norm = torch.nn.utils.clip_grad_norm_(
                    draft.parameters(),
                    args.max_grad_norm,
                    error_if_nonfinite=True,
                    foreach=True,
                )
            else:
                global_grad_norm = None
            grad_metrics = None
            if step % args.gradient_log_interval == 0:
                if rank == 0:
                    grad_metrics = gradient_norms(draft)
                dist.barrier()

            observe_update = args.update_log_interval > 0 and (
                step == start_step + 1 or step % args.update_log_interval == 0
            )
            parameter_snapshot = (
                capture_parameter_snapshot(draft)
                if observe_update and rank == 0
                else None
            )
            if observe_update:
                dist.barrier()

            learning_rate = scheduled_learning_rate(
                step,
                total_steps,
                args.learning_rate,
                args.warmup_steps,
                args.min_lr_ratio,
            )
            optimizer.set_learning_rate(learning_rate)
            optimizer.step()
            update_metrics = (
                parameter_update_metrics(draft, parameter_snapshot)
                if parameter_snapshot is not None
                else None
            )
            del parameter_snapshot
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started_step

            if (
                step % args.log_interval == 0
                or step == start_step + 1
                or observe_update
            ):
                metrics = reduced_metrics(output, world_size)
                counts = torch.tensor(
                    [input_ids.numel(), len(anchors), skipped_rows],
                    device="cuda",
                    dtype=torch.float64,
                )
                dist.reduce(counts, dst=0, op=dist.ReduceOp.SUM)
                elapsed_tensor = torch.tensor(elapsed, device="cuda")
                dist.reduce(elapsed_tensor, dst=0, op=dist.ReduceOp.MAX)
                if rank == 0:
                    position_count = draft.block_size - 1
                    position_ce = metrics[9 : 9 + position_count]
                    position_accuracy = metrics[9 + position_count :]
                    labels = counts[1].item() * position_count
                    record = {
                        "step": step,
                        "total_steps": total_steps,
                        "epoch": step * world_size / stream.total_rows,
                        "experiment": args.experiment,
                        "loss": metrics[0].item(),
                        "plain_ce": metrics[1].item(),
                        "pal_loss": metrics[2].item(),
                        "proxy_acceptance_length": metrics[3].item(),
                        "greedy_acceptance_length": metrics[4].item(),
                        "selector_loss": metrics[5].item(),
                        "selector_candidate_recall": metrics[6].item(),
                        "selector_teacher_accuracy": metrics[7].item(),
                        "selector_path_acceptance_length": metrics[8].item(),
                        "position_ce": position_ce.tolist(),
                        "position_accuracy": position_accuracy.tolist(),
                        "global_gradient_norm_before_clip": (
                            global_grad_norm.item()
                            if global_grad_norm is not None
                            else None
                        ),
                        "gradient_norms_after_clip": grad_metrics,
                        "learning_rate": learning_rate,
                        "samples": world_size,
                        "tokens": int(counts[0].item()),
                        "supervised_labels": int(labels),
                        "skipped_rows": int(counts[2].item()),
                        "elapsed_seconds": elapsed_tensor.item(),
                        "samples_per_second": world_size / elapsed_tensor.item(),
                        "labels_per_second": labels / elapsed_tensor.item(),
                        "cuda_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated()
                        / 1024**3,
                    }
                    if update_metrics is not None:
                        record.update(update_metrics)
                        record["parameter_update_gradient_basis"] = (
                            "after_clip" if args.max_grad_norm else "raw"
                        )
                    print(json.dumps(record, ensure_ascii=False), flush=True)

            del features, input_ids, output
            if step % args.checkpoint_interval == 0 or step == final_step:
                last_checkpoint = save_checkpoint(
                    output_dir=args.output_dir,
                    step=step,
                    total_steps=total_steps,
                    draft=draft,
                    optimizer=optimizer,
                    stream=stream,
                    anchor_rng=anchor_rng,
                    architecture=architecture,
                    loss_objective=args.loss_objective,
                    rank=rank,
                    world_size=world_size,
                    storage_root=args.storage_root,
                    max_storage_bytes=int(args.max_storage_gib * 1024**3),
                )

        if rank == 0:
            atomic_json(
                args.output_dir / "completed.json",
                {
                    **architecture,
                    "loss_objective": args.loss_objective,
                    "steps": final_step,
                    "schedule_total_steps": total_steps,
                    "epochs": final_step * world_size / stream.total_rows,
                    "world_size": world_size,
                    "elapsed_seconds": time.time() - started_run,
                    "final_checkpoint": last_checkpoint.name,
                    "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated()
                    / 1024**3,
                },
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
