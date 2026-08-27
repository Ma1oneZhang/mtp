"""Regenerate target responses through a vLLM OpenAI-compatible server."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--served-model", default="qwen3-4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("perfectblend-qwen3-8b-regen/data"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--rows-per-shard", type=int, default=4096)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_dir():
        raise FileNotFoundError(f"model directory not found: {args.model}")
    if not args.source.is_dir():
        raise FileNotFoundError(f"source directory not found: {args.source}")
    if args.output.exists() and not args.resume:
        raise FileExistsError(
            f"refusing to reuse output without --resume: {args.output}"
        )
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if args.rows_per_shard < 1:
        raise ValueError("rows-per-shard must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("max-rows must be positive")
    if args.max_input_tokens < 1 or args.max_new_tokens < 1:
        raise ValueError("token limits must be positive")


def parquet_rows_once(data_dir: Path) -> Iterator[dict[str, Any]]:
    paths = sorted(data_dir.glob("train-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no train-*.parquet files under {data_dir}")
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024):
            yield from batch.to_pylist()


def source_prompts(
    source: Path,
    tokenizer: Any,
    max_input_tokens: int,
) -> Iterator[dict[str, Any]]:
    for row in parquet_rows_once(source):
        conversations = row.get("conversations")
        if (
            row.get("status") != "success"
            or not conversations
            or len(conversations) != 2
            or conversations[0].get("role") != "user"
            or conversations[1].get("role") != "assistant"
        ):
            continue
        prompt = conversations[0].get("content")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if (
            len(tokenizer.encode(prompt_text, add_special_tokens=False))
            > max_input_tokens
        ):
            continue
        yield {"id": row["id"], "prompt": prompt, "prompt_text": prompt_text}


def output_schema(args: argparse.Namespace) -> pa.Schema:
    metadata = {
        b"generator_model": args.model.name.encode(),
        b"served_model": args.served_model.encode(),
        b"enable_thinking": b"false",
        b"temperature": b"0.7",
        b"top_p": b"0.8",
        b"top_k": b"20",
        b"max_new_tokens": str(args.max_new_tokens).encode(),
        b"seed": str(args.seed).encode(),
    }
    return pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field(
                "conversations",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("role", pa.string()),
                            pa.field("content", pa.string()),
                            pa.field("thinking", pa.null()),
                        ]
                    )
                ),
            ),
            pa.field("output_token_ids", pa.list_(pa.int32())),
            pa.field("status", pa.string()),
        ],
        metadata=metadata,
    )


def completed_output(args: argparse.Namespace) -> tuple[int, int]:
    paths = sorted(args.output.glob("train-*.parquet"))
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    return len(paths), rows


async def generate_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    item: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    payload = {
        "model": args.served_model,
        "prompt": item["prompt_text"],
        "max_tokens": args.max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "seed": args.seed + int(item["id"]),
        "return_token_ids": True,
    }
    for attempt in range(3):
        try:
            async with semaphore:
                response = await client.post("/completions", json=payload)
                response.raise_for_status()
            break
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
            if attempt == 2:
                raise
            await asyncio.sleep(2**attempt)
    body = response.json()
    choice = body["choices"][0]
    finish_reason = choice["finish_reason"]
    if finish_reason == "stop":
        status = "success"
    elif finish_reason == "length":
        status = "max_tokens"
    else:
        raise RuntimeError(f"unexpected finish reason: {finish_reason!r}")
    output_token_ids = choice["token_ids"]
    if output_token_ids is None:
        raise RuntimeError("vLLM response omitted token_ids")
    record = {
        "id": item["id"],
        "conversations": [
            {"role": "user", "content": item["prompt"], "thinking": None},
            {
                "role": "assistant",
                "content": choice["text"],
                "thinking": None,
            },
        ],
        "output_token_ids": output_token_ids,
        "status": status,
    }
    return record, len(output_token_ids)


async def generate_shard(
    client: httpx.AsyncClient,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], int]:
    semaphore = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(
        *(generate_one(client, semaphore, item, args) for item in items)
    )
    return [record for record, _ in results], sum(tokens for _, tokens in results)


async def run(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    shard_index, already_written = completed_output(args)
    prompts = source_prompts(args.source, tokenizer, args.max_input_tokens)
    for _ in islice(prompts, already_written):
        pass

    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    timeout = httpx.Timeout(connect=30.0, read=3600.0, write=30.0, pool=3600.0)
    generated_tokens = 0
    written = already_written
    started = time.monotonic()
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"), limits=limits, timeout=timeout
    ) as client:
        health = await client.get("/models")
        health.raise_for_status()
        while args.max_rows is None or written < args.max_rows:
            count = args.rows_per_shard
            if args.max_rows is not None:
                count = min(count, args.max_rows - written)
            items = list(islice(prompts, count))
            if not items:
                break
            records, shard_tokens = await generate_shard(client, items, args)
            final_path = args.output / f"train-{shard_index:06d}.parquet"
            temporary_path = args.output / f".train-{shard_index:06d}.parquet.tmp"
            pq.write_table(
                pa.Table.from_pylist(records, schema=output_schema(args)),
                temporary_path,
                compression="zstd",
                compression_level=3,
            )
            temporary_path.replace(final_path)
            shard_index += 1
            written += len(records)
            generated_tokens += shard_tokens
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "rows": written,
                        "generated_tokens_this_run": generated_tokens,
                        "elapsed_seconds": round(elapsed, 1),
                        "tokens_per_second": round(generated_tokens / elapsed, 1),
                        "output": str(final_path),
                    }
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "complete": True,
                "rows": written,
                "shards": shard_index,
                "generated_tokens_this_run": generated_tokens,
                "elapsed_seconds": round(time.monotonic() - started, 1),
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
