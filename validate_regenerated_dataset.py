"""Validate a completed Qwen3 regeneration dataset as a streaming scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoConfig, AutoTokenizer

from regenerate_qwen3_4b import source_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def metadata(path: Path) -> dict[str, str]:
    raw = pq.ParquetFile(path).schema_arrow.metadata
    if raw is None:
        raise ValueError(f"missing schema metadata: {path}")
    return {key.decode(): value.decode() for key, value in raw.items()}


def validate_metadata(
    path: Path,
    current: dict[str, str],
    expected: dict[str, str] | None,
    args: argparse.Namespace,
) -> dict[str, str]:
    if current.get("generator_model") != args.model.name:
        raise ValueError(f"generator model mismatch: {path}")
    if current.get("enable_thinking") != "false":
        raise ValueError(f"thinking must be disabled: {path}")
    if current.get("max_new_tokens") != str(args.max_new_tokens):
        raise ValueError(f"max_new_tokens mismatch: {path}")
    if expected is not None and current != expected:
        raise ValueError(f"inconsistent schema metadata: {path}")
    return current


def validate_row(
    row: dict[str, Any],
    expected_id: int,
    tokenizer: Any,
    target_vocab_size: int,
    max_new_tokens: int,
    row_index: int,
) -> tuple[int, int, int]:
    if row.get("id") != expected_id:
        raise ValueError(
            f"source/output id mismatch at row {row_index}: "
            f"{expected_id} != {row.get('id')}"
        )
    token_ids = row.get("output_token_ids")
    if not token_ids:
        raise ValueError(f"missing output_token_ids at row {row_index}")
    if len(token_ids) > max_new_tokens:
        raise ValueError(f"too many output tokens at row {row_index}")
    if any(token_id < 0 or token_id >= target_vocab_size for token_id in token_ids):
        raise ValueError(f"output token out of vocabulary at row {row_index}")

    status = row.get("status")
    if status == "success":
        if token_ids[-1] != tokenizer.eos_token_id:
            raise ValueError(f"successful row has no terminal EOS: {row_index}")
        success, max_tokens = 1, 0
    elif status == "max_tokens":
        if len(token_ids) != max_new_tokens:
            raise ValueError(f"max_tokens row has unexpected length: {row_index}")
        success, max_tokens = 0, 1
    else:
        raise ValueError(f"unknown status at row {row_index}: {status!r}")

    if row_index < 1024 or row_index % 1000 == 0:
        conversations = row.get("conversations")
        if not conversations or conversations[-1].get("role") != "assistant":
            raise ValueError(f"missing assistant response at row {row_index}")
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        if (
            status == "success"
            and decoded != conversations[-1].get("content")
        ):
            raise ValueError(f"text/token mismatch at row {row_index}")
    return len(token_ids), success, max_tokens


def main() -> None:
    args = parse_args()
    paths = sorted(args.data.glob("train-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no output shards under {args.data}")
    temporary_paths = sorted(args.data.glob(".train-*.parquet.tmp"))
    if temporary_paths:
        raise ValueError(f"incomplete output shard: {temporary_paths[0]}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    target_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    expected_prompts = source_prompts(
        args.source, tokenizer, args.max_input_tokens
    )

    rows = 0
    output_tokens = 0
    success = 0
    max_tokens = 0
    expected_metadata = None
    output_bytes = 0
    for shard_index, path in enumerate(paths):
        expected_name = f"train-{shard_index:06d}.parquet"
        if path.name != expected_name:
            raise ValueError(f"non-contiguous shard: {path.name} != {expected_name}")
        expected_metadata = validate_metadata(
            path, metadata(path), expected_metadata, args
        )
        output_bytes += path.stat().st_size
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                prompt = next(expected_prompts, None)
                if prompt is None:
                    raise ValueError(f"output has extra row at index {rows}")
                tokens, row_success, row_max_tokens = validate_row(
                    row,
                    int(prompt["id"]),
                    tokenizer,
                    target_config.vocab_size,
                    args.max_new_tokens,
                    rows,
                )
                rows += 1
                output_tokens += tokens
                success += row_success
                max_tokens += row_max_tokens

    remaining_prompt = next(expected_prompts, None)
    if remaining_prompt is not None and not args.allow_incomplete:
        raise ValueError(
            f"output is incomplete after {rows} rows; "
            f"next source id is {remaining_prompt['id']}"
        )
    print(
        json.dumps(
            {
                "valid": True,
                "complete": remaining_prompt is None,
                "shards": len(paths),
                "rows": rows,
                "success": success,
                "max_tokens": max_tokens,
                "output_tokens": output_tokens,
                "output_bytes": output_bytes,
            }
        )
    )


if __name__ == "__main__":
    main()
