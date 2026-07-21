"""DuckDB catalog over the parquet lake.

No database server, no ETL into tables -- DuckDB queries the parquet files
where they sit. Each dataset becomes a view that unions all seasons.

    from ffdata.db import connect
    con = connect()
    con.sql("select * from weekly where season = 2024 limit 5").show()
"""

from __future__ import annotations

from pathlib import Path

import duckdb

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def connect(db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_path)
    if not RAW.exists():
        raise FileNotFoundError(
            f"No data lake at {RAW}. Run `python -m ffdata.cli` to ingest first."
        )
    for dataset_dir in sorted(RAW.iterdir()):
        if not dataset_dir.is_dir() or not any(dataset_dir.glob("*.parquet")):
            continue
        name = dataset_dir.name
        con.sql(
            f"create or replace view {name} as "
            f"select * from read_parquet('{dataset_dir}/*.parquet', union_by_name=true)"
        )
    return con
