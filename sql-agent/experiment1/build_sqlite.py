#!/usr/bin/env python3
"""Load BiomedSQL parquet tables into one SQLite file.

Usage:
  python3 build_sqlite.py \\
      --parquet-dir ../data/db_data \\
      --out ../data/biomedsql.sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pyarrow.parquet as pq


# Tables used by the 546-question eval set.
CORE_TABLES = {
    "AlzheimerDisease_CombinedGeneData_UUID",
    "DrugGeneTargets_ComprehensiveAnnotations_updated",
    "DrugTargets_IndicationsAndTherapeuticUses",
    "NeurodegenerativeDiseases_SMR_Genes_Full",
    "ParkinsonDisease_CompleteGeneData_No23andMe",
}


def arrow_to_sqlite_type(type_str: str) -> str:
    t = type_str.lower()
    if "int" in t:
        return "INTEGER"
    if "float" in t or "double" in t or "decimal" in t:
        return "REAL"
    if "bool" in t:
        return "INTEGER"
    return "TEXT"


def load_table(connection: sqlite3.Connection, parquet_path: Path, table: str) -> int:
    print(f"loading {table} from {parquet_path.name} ...", flush=True)
    parquet_file = pq.ParquetFile(parquet_path)
    schema = parquet_file.schema_arrow
    columns = schema.names
    types = [arrow_to_sqlite_type(str(field.type)) for field in schema]
    col_defs = ", ".join(f'"{c}" {t}' for c, t in zip(columns, types))
    connection.execute(f'DROP TABLE IF EXISTS "{table}"')
    connection.execute(f'CREATE TABLE "{table}" ({col_defs})')

    placeholders = ", ".join("?" for _ in columns)
    insert_sql = f'INSERT INTO "{table}" VALUES ({placeholders})'
    batch_size = 50_000
    n = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        data = batch.to_pydict()
        keys = list(data.keys())
        nrows = len(next(iter(data.values()))) if data else 0
        rows = []
        for i in range(nrows):
            row = []
            for k in keys:
                v = data[k][i]
                if isinstance(v, bool):
                    v = int(v)
                row.append(v)
            rows.append(tuple(row))
        connection.executemany(insert_sql, rows)
        n += len(rows)
        if n % 200_000 == 0:
            print(f"  {table}: {n:,} rows", flush=True)
            connection.commit()
    connection.commit()
    print(f"  {table}: done ({n:,} rows)", flush=True)
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only-core", action="store_true", help="load only the 5 eval tables")
    args = parser.parse_args()

    files = sorted(args.parquet_dir.glob("*.parquet"))
    if not files:
        print(f"no parquet files in {args.parquet_dir}", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    connection = sqlite3.connect(args.out)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")

    total = 0
    for path in files:
        table = path.stem
        if args.only_core and table not in CORE_TABLES:
            print(f"skip {table} (not core)", flush=True)
            continue
        total += load_table(connection, path, table)

    index_specs = [
        ("ParkinsonDisease_CompleteGeneData_No23andMe", "SNP"),
        ("AlzheimerDisease_CombinedGeneData_UUID", "SNP"),
        ("ParkinsonDisease_CompleteGeneData_No23andMe", "nearestGene"),
        ("AlzheimerDisease_CombinedGeneData_UUID", "nearestGene"),
        ("NeurodegenerativeDiseases_SMR_Genes_Full", "Gene"),
        ("DrugGeneTargets_ComprehensiveAnnotations_updated", "drugName"),
    ]
    for table, col in index_specs:
        try:
            connection.execute(
                f'CREATE INDEX IF NOT EXISTS "idx_{table}_{col}" ON "{table}" ("{col}")'
            )
        except sqlite3.Error as exc:
            print(f"index skip {table}.{col}: {exc}", flush=True)
    connection.commit()
    connection.close()
    size_gb = args.out.stat().st_size / 1e9
    print(f"wrote {args.out} ({size_gb:.2f} GB, {total:,} rows total)", flush=True)


if __name__ == "__main__":
    main()
