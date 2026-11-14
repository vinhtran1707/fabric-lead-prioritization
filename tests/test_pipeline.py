"""Unit tests for the silver and gold transforms on small hand-built inputs."""
from pyspark.sql import functions as F

from leadscore import pipeline as pl

HEADER = ["age", "job", "marital", "education", "default", "housing", "loan", "contact", "month",
          "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome", "emp.var.rate",
          "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"]


def _row(month, pdays="999", y="no", age="40", duration="100"):
    return [age, "admin.", "married", "university.degree", "no", "yes", "no", "cellular", month,
            "mon", duration, "1", pdays, "0", "nonexistent", "1.1", "93.9", "-36.4", "4.8", "5191", y]


def _bronze(spark, rows):
    data = [(*r, i + 1, "test.csv") for i, r in enumerate(rows)]
    cols = HEADER + ["source_row_number", "source_file"]
    return spark.createDataFrame(data, cols).withColumn("ingested_at", F.current_timestamp())


def test_year_rolls_over_when_month_goes_backwards(spark):
    rows = [_row("nov"), _row("dec"), _row("mar", age="41"), _row("may", age="42"), _row("apr", age="43")]
    out = {r.source_row_number: str(r.call_month_start)
           for r in pl.build_silver(_bronze(spark, rows)).collect()}
    assert out == {1: "2008-11-01", 2: "2008-12-01", 3: "2009-03-01", 4: "2009-05-01", 5: "2010-04-01"}


def test_pdays_sentinel_becomes_null_with_flag(spark):
    silver = pl.build_silver(_bronze(spark, [_row("may"), _row("may", pdays="6", age="50")]))
    got = {r.age: (r.days_since_previous_contact, r.was_previously_contacted) for r in silver.collect()}
    assert got == {40: (None, 0), 50: (6, 1)}


def test_exact_duplicates_are_dropped_keeping_first(spark):
    silver = pl.build_silver(_bronze(spark, [_row("may"), _row("may"), _row("may", age="55")]))
    assert sorted(r.source_row_number for r in silver.collect()) == [1, 3]


def test_label_and_gold_excludes_post_call_duration(spark):
    silver = pl.build_silver(_bronze(spark, [_row("may", y="yes"), _row("may", y="no", age="30")]))
    gold = pl.build_gold_features(silver)
    assert "call_duration_sec" not in gold.columns
    assert sorted(r.converted for r in gold.select("converted").collect()) == [0, 1]
    assert set(pl.CATEGORICAL_FEATURES + pl.NUMERIC_FEATURES) <= set(gold.columns)
