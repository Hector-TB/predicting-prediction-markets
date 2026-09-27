"""
stream_parquet.py
=================
Batch-at-a-time parquet rewriting, so pipeline steps never hold the whole
snapshot dataset in memory. After the ADR-013 re-fetch the dataset (~8M rows)
no longer fits comfortably in RAM on an 8 GB machine.

    for df in iter_frames(path): ...                # read in batches
    with FrameWriter(out_path) as w: w.write(df)    # write in batches
    rewrite(src, dst, transform)                    # both, src may equal dst
"""

import logging
import os
from pathlib import Path
from typing import Callable, Iterator, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

BATCH_ROWS = 250_000


def iter_frames(path: Path, batch_rows: int = BATCH_ROWS,
                columns: Optional[list[str]] = None) -> Iterator[pd.DataFrame]:
    """Yield the parquet file at `path` as DataFrames of up to `batch_rows` rows."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
        yield batch.to_pandas()


def unique_values(path: Path, column: str) -> set:
    """Distinct values of one column, read batch by batch."""
    values: set = set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=[column]):
        values.update(batch.column(0).unique().to_pylist())
    return values


class FrameWriter:
    """
    Append DataFrames to one parquet file.

    Writes to a temporary sibling and moves it into place only on a clean
    exit, so `path` may be the file being read, and a crash never leaves a
    half-written dataset behind.

    The schema is fixed by the first batch. Columns that are entirely null in
    that batch (e.g. `category` before categorisation) would get Arrow's null
    type and reject later non-null batches, so they are promoted to string.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        self.writer: Optional[pq.ParquetWriter] = None
        self.schema: Optional[pa.Schema] = None
        self.rows = 0

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.schema = pa.schema([
                f.with_type(pa.string()) if pa.types.is_null(f.type) else f
                for f in table.schema
            ]).remove_metadata()
            self.writer = pq.ParquetWriter(self.tmp, self.schema)
        self.writer.write_table(table.select(self.schema.names).cast(self.schema))
        self.rows += len(df)

    def __enter__(self) -> "FrameWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.writer is not None:
            self.writer.close()
        if exc_type is None and self.writer is not None:
            os.replace(self.tmp, self.path)
        elif self.tmp.exists():
            self.tmp.unlink()


def rewrite(src: Path, dst: Path,
            transform: Callable[[pd.DataFrame], pd.DataFrame],
            batch_rows: int = BATCH_ROWS) -> tuple[int, int]:
    """Stream `src` through `transform` into `dst`. Returns (rows_in, rows_out)."""
    rows_in = 0
    with FrameWriter(dst) as w:
        for df in iter_frames(src, batch_rows):
            rows_in += len(df)
            w.write(transform(df))
    log.info("  %s → %s: %s rows in, %s rows out",
             Path(src).name, Path(dst).name, f"{rows_in:,}", f"{w.rows:,}")
    return rows_in, w.rows
