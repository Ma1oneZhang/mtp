import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from distributed_data import RankRowStream


class RankRowStreamTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary.name)
        next_id = 0
        for file_index, rows in enumerate((5, 4, 7, 3)):
            ids = list(range(next_id, next_id + rows))
            next_id += rows
            pq.write_table(
                pa.table({"id": ids}),
                self.data_dir / f"train-{file_index:06d}.parquet",
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_partitions_are_disjoint_and_cover_training_rows(self) -> None:
        partitions = []
        for rank in range(3):
            stream = RankRowStream(self.data_dir, rank, 3, holdout_shards=1)
            partitions.append(
                [next(stream)["id"] for _ in range(stream.rows_in_partition)]
            )
        flattened = [value for partition in partitions for value in partition]
        self.assertEqual(sorted(flattened), list(range(16)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_resume_continues_at_exact_row(self) -> None:
        stream = RankRowStream(self.data_dir, rank=1, world_size=3, holdout_shards=1)
        consumed = [next(stream)["id"] for _ in range(3)]
        state = stream.state_dict()
        expected = [next(stream)["id"] for _ in range(5)]

        resumed = RankRowStream(self.data_dir, rank=1, world_size=3, holdout_shards=1)
        resumed.load_state_dict(state)
        actual = [next(resumed)["id"] for _ in range(5)]
        self.assertEqual(actual, expected)
        self.assertEqual(len(consumed), 3)


if __name__ == "__main__":
    unittest.main()
