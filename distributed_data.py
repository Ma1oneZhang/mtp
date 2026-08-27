from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


@dataclass(frozen=True)
class ParquetSlice:
    path: Path
    start: int
    stop: int

    @property
    def rows(self) -> int:
        return self.stop - self.start


class RankRowStream:
    """Infinite, resumable stream over one disjoint contiguous dataset partition."""

    def __init__(
        self,
        data_dir: Path,
        rank: int,
        world_size: int,
        holdout_shards: int,
    ) -> None:
        paths = sorted(data_dir.glob("train-*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no train-*.parquet files under {data_dir}")
        if not 0 <= holdout_shards < len(paths):
            raise ValueError("holdout-shards must leave at least one training shard")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")

        training_paths = paths[:-holdout_shards] if holdout_shards else paths
        row_counts = [pq.ParquetFile(path).metadata.num_rows for path in training_paths]
        self.total_rows = sum(row_counts)
        partition_start = self.total_rows * rank // world_size
        partition_stop = self.total_rows * (rank + 1) // world_size
        self.rows_in_partition = partition_stop - partition_start
        self.slices = self._partition_slices(
            training_paths, row_counts, partition_start, partition_stop
        )
        if sum(part.rows for part in self.slices) != self.rows_in_partition:
            raise RuntimeError("dataset partition does not cover its assigned rows")

        self.cycle = 0
        self.slice_index = 0
        self.row_index = 0
        self._loaded_slice = -1
        self._rows: list[dict[str, Any]] = []

    @staticmethod
    def _partition_slices(
        paths: list[Path],
        row_counts: list[int],
        partition_start: int,
        partition_stop: int,
    ) -> list[ParquetSlice]:
        result = []
        global_start = 0
        for path, row_count in zip(paths, row_counts, strict=True):
            global_stop = global_start + row_count
            overlap_start = max(global_start, partition_start)
            overlap_stop = min(global_stop, partition_stop)
            if overlap_start < overlap_stop:
                result.append(
                    ParquetSlice(
                        path,
                        overlap_start - global_start,
                        overlap_stop - global_start,
                    )
                )
            global_start = global_stop
        return result

    def _load_current_slice(self) -> None:
        if self._loaded_slice == self.slice_index:
            return
        part = self.slices[self.slice_index]
        table = pq.read_table(part.path).slice(part.start, part.rows)
        self._rows = table.to_pylist()
        if len(self._rows) != part.rows:
            raise RuntimeError(f"short parquet read: {part.path.name}")
        self._loaded_slice = self.slice_index

    def __iter__(self) -> RankRowStream:
        return self

    def __next__(self) -> dict[str, Any]:
        self._load_current_slice()
        row = self._rows[self.row_index]
        self.row_index += 1
        if self.row_index == len(self._rows):
            self.row_index = 0
            self.slice_index += 1
            self._loaded_slice = -1
            self._rows = []
            if self.slice_index == len(self.slices):
                self.slice_index = 0
                self.cycle += 1
        return row

    def state_dict(self) -> dict[str, int]:
        return {
            "cycle": self.cycle,
            "slice_index": self.slice_index,
            "row_index": self.row_index,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        cycle = state["cycle"]
        slice_index = state["slice_index"]
        row_index = state["row_index"]
        if cycle < 0:
            raise ValueError("dataset cycle cannot be negative")
        if not 0 <= slice_index < len(self.slices):
            raise ValueError("dataset slice index is out of range")
        if not 0 <= row_index < self.slices[slice_index].rows:
            raise ValueError("dataset row index is out of range")
        self.cycle = cycle
        self.slice_index = slice_index
        self.row_index = row_index
        self._loaded_slice = -1
        self._rows = []
