"""Data quality checks that stop the pipeline when a layer looks wrong."""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


class DataQualityError(RuntimeError):
    pass


def _check(name: str, passed: bool, detail: str) -> CheckResult:
    return CheckResult(name, bool(passed), detail)


def check_silver(silver: DataFrame, bronze_rows: int) -> list[CheckResult]:
    rows = silver.count()
    stats = silver.agg(
        F.min("call_month_start").alias("first_month"),
        F.max("call_month_start").alias("last_month"),
        F.sum(F.col("converted").isNull().cast("int")).alias("null_labels"),
        F.avg("converted").alias("conversion_rate"),
        F.countDistinct("lead_id").alias("distinct_leads"),
        F.sum(F.col("call_month_num").isNull().cast("int")).alias("bad_months"),
        F.min("age").alias("min_age"),
    ).first()
    removed = bronze_rows - rows
    return [
        _check("duplicates removed are a small share", 0 <= removed <= bronze_rows * 0.001,
               f"{removed} exact duplicate rows removed"),
        _check("lead_id is unique", stats.distinct_leads == rows, f"{stats.distinct_leads} ids / {rows} rows"),
        _check("no missing labels", stats.null_labels == 0, f"{stats.null_labels} null labels"),
        _check("every month parsed", stats.bad_months == 0, f"{stats.bad_months} unparsed months"),
        _check("reconstructed dates match documented range (May 2008 to Nov 2010)",
               str(stats.first_month) == "2008-05-01" and str(stats.last_month) == "2010-11-01",
               f"{stats.first_month} to {stats.last_month}"),
        _check("conversion rate is plausible", 0.05 < stats.conversion_rate < 0.25,
               f"{stats.conversion_rate:.4f}"),
        _check("ages are adults", stats.min_age >= 17, f"min age {stats.min_age}"),
    ]


def check_gold_features(gold: DataFrame, silver_rows: int) -> list[CheckResult]:
    leaked = [c for c in ("call_duration_sec", "y") if c in gold.columns]
    return [
        _check("one feature row per silver lead", gold.count() == silver_rows, f"{gold.count()} rows"),
        _check("no post-call or raw label columns", not leaked, f"leaked columns: {leaked}"),
    ]


def enforce(results: list[CheckResult], layer: str) -> None:
    for r in results:
        print(f"[{layer}] {'PASS' if r.passed else 'FAIL'}  {r.name}: {r.detail}")
    failed = [r for r in results if not r.passed]
    if failed:
        raise DataQualityError(f"{len(failed)} {layer} check(s) failed: {[r.name for r in failed]}")
