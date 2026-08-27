from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from architectures import configure_architecture
from train_experiment import atomic_json, compute_loss, load_models, target_features
from training_utils import EXPERIMENTS, tokenize_last_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-shards", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=1_000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--anchors-per-sample", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def holdout_rows(data_dir: Path, holdout_shards: int) -> Iterator[dict[str, Any]]:
    paths = sorted(data_dir.glob("train-*.parquet"))
    if not 0 < holdout_shards < len(paths):
        raise ValueError("holdout-shards must select a proper dataset suffix")
    for path in paths[-holdout_shards:]:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=32):
            yield from batch.to_pylist()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"weights not found: {args.weights}")
    if args.max_samples < 1 or args.anchors_per_sample < 1:
        raise ValueError("sample and anchor counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    tokenizer, target, draft = load_models(args)
    architecture = configure_architecture(draft, args.experiment)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    draft.load_state_dict(state, strict=True)
    draft.eval()

    position_count = draft.block_size - 1
    position_ce = torch.zeros(position_count, dtype=torch.float64)
    position_accuracy = torch.zeros(position_count, dtype=torch.float64)
    totals = {
        "plain_ce": 0.0,
        "greedy_acceptance_length": 0.0,
        "selector_loss": 0.0,
        "selector_candidate_recall": 0.0,
        "selector_teacher_accuracy": 0.0,
        "selector_path_acceptance_length": 0.0,
    }
    samples = 0
    anchors_total = 0
    skipped = 0
    for row in holdout_rows(args.data, args.holdout_shards):
        prepared = tokenize_last_answer(
            row, tokenizer, args.max_length, draft.block_size
        )
        if prepared is None:
            skipped += 1
            continue
        input_ids, valid_anchors = prepared
        anchors = rng.sample(
            valid_anchors, min(args.anchors_per_sample, len(valid_anchors))
        )
        input_ids = input_ids.to("cuda")
        features = target_features(target, input_ids, draft.target_layer_ids)
        output = compute_loss(
            target=target,
            draft=draft,
            input_ids=input_ids,
            features=features,
            anchors=anchors,
            ngram_beta=architecture["ngram_beta"],
        )
        weight = len(anchors)
        position_ce += output.position_ce.double().cpu() * weight
        position_accuracy += output.position_accuracy.double().cpu() * weight
        for key in totals:
            totals[key] += getattr(output, key).item() * weight
        samples += 1
        anchors_total += weight
        if samples % 100 == 0:
            print(
                json.dumps(
                    {
                        "experiment": args.experiment,
                        "samples": samples,
                        "anchors": anchors_total,
                        "plain_ce": totals["plain_ce"] / anchors_total,
                        "greedy_acceptance_length": totals["greedy_acceptance_length"]
                        / anchors_total,
                    }
                ),
                flush=True,
            )
        if samples == args.max_samples:
            break

    if samples != args.max_samples:
        raise RuntimeError(
            f"holdout supplied only {samples} usable samples, requested {args.max_samples}"
        )
    summary = {
        **architecture,
        "samples": samples,
        "anchors": anchors_total,
        "supervised_labels": anchors_total * position_count,
        "skipped_rows": skipped,
        "seed": args.seed,
        "holdout_shards": args.holdout_shards,
        "weights": args.weights.name,
        **{key: value / anchors_total for key, value in totals.items()},
        "position_ce": (position_ce / anchors_total).tolist(),
        "position_accuracy": (position_accuracy / anchors_total).tolist(),
        "token_accuracy": (position_accuracy / anchors_total).mean().item(),
    }
    atomic_json(args.output_dir / "completed.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
