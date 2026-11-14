"""Read and write lakehouse tables the same way locally and in Microsoft Fabric.

In a Fabric notebook attached to a lakehouse, tables are saved as managed Delta
tables (so Power BI can read them in Direct Lake mode) and raw files live under
``Files/``. Locally, the same calls write Parquet folders under ``./lakehouse``
so the whole pipeline can run and be tested on a laptop or in CI.
"""

from __future__ import annotations

import os
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession


def in_fabric() -> bool:
    """True when running inside a Fabric notebook."""
    try:
        import notebookutils  # noqa: F401  (only exists in Fabric runtimes)

        return True
    except ImportError:
        return os.environ.get("LEADSCORE_ENV") == "fabric"


def get_spark(app_name: str = "leadscore") -> SparkSession:
    if in_fabric():
        return SparkSession.builder.getOrCreate()
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def _local_root() -> Path:
    return Path(os.environ.get("LEADSCORE_LAKEHOUSE", "lakehouse")).resolve()


def raw_file_path(file_name: str) -> str:
    """Location of a raw landing file."""
    if in_fabric():
        return f"Files/raw/{file_name}"
    return str(Path(os.environ.get("LEADSCORE_RAW", "data/raw")).resolve() / file_name)


def write_table(df: DataFrame, name: str) -> None:
    """Overwrite a lakehouse table."""
    if in_fabric():
        (df.write.format("delta").mode("overwrite")
           .option("overwriteSchema", "true").saveAsTable(name))
    else:
        df.write.mode("overwrite").parquet(str(_local_root() / "Tables" / name))


def read_table(spark: SparkSession, name: str) -> DataFrame:
    if in_fabric():
        return spark.read.table(name)
    return spark.read.parquet(str(_local_root() / "Tables" / name))
