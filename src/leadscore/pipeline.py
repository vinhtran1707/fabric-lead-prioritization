"""Bronze -> silver -> gold transforms for the lead conversion dataset.

Source: UCI Bank Marketing (bank-additional-full.csv), 41,188 outbound sales
calls made by a Portuguese bank, May 2008 to November 2010, ordered by date.
Each row is one lead's most recent call in a campaign and whether the lead
converted (subscribed to a term deposit).
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

SOURCE_FILE = "bank-additional-full.csv"
FIRST_YEAR = 2008  # documented start of the data (May 2008)

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

# Known only after the call ends, so it cannot be used to decide whom to call.
POST_CALL_COLUMNS = ["call_duration_sec"]

CATEGORICAL_FEATURES = [
    "job", "marital", "education", "credit_default", "housing_loan", "personal_loan",
    "contact_channel", "call_month", "call_day_of_week", "previous_outcome",
]
NUMERIC_FEATURES = [
    "age", "contacts_this_campaign", "days_since_previous_contact", "previous_contacts",
    "was_previously_contacted", "emp_var_rate", "cons_price_idx", "cons_conf_idx",
    "euribor_3m", "nr_employed",
]


# --------------------------------------------------------------------------- bronze
def build_bronze(spark: SparkSession, path: str) -> DataFrame:
    """Land the raw file as-is (all strings) with lineage columns.

    The file is chronological but has no date or id, so the original row order
    is captured here with zipWithIndex before anything can reorder it.
    """
    raw = spark.read.option("header", True).option("sep", ";").option("inferSchema", False).csv(path)
    indexed = raw.rdd.zipWithIndex().map(lambda r: (*r[0], r[1] + 1))
    schema = T.StructType(raw.schema.fields + [T.StructField("source_row_number", T.LongType(), False)])
    return (
        spark.createDataFrame(indexed, schema)
        .withColumn("source_file", F.lit(SOURCE_FILE))
        .withColumn("ingested_at", F.current_timestamp())
    )


# --------------------------------------------------------------------------- silver
_RENAMES = {
    "default": "credit_default", "housing": "housing_loan", "loan": "personal_loan",
    "contact": "contact_channel", "month": "call_month", "day_of_week": "call_day_of_week",
    "duration": "call_duration_sec", "campaign": "contacts_this_campaign",
    "pdays": "days_since_previous_contact", "previous": "previous_contacts",
    "poutcome": "previous_outcome", "emp.var.rate": "emp_var_rate",
    "cons.price.idx": "cons_price_idx", "cons.conf.idx": "cons_conf_idx",
    "euribor3m": "euribor_3m", "nr.employed": "nr_employed",
}
_INTS = ["age", "call_duration_sec", "contacts_this_campaign", "days_since_previous_contact",
         "previous_contacts"]
_DOUBLES = ["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor_3m", "nr_employed"]


def build_silver(bronze: DataFrame) -> DataFrame:
    """Typed, renamed, de-duplicated calls with a reconstructed call month."""
    df = bronze
    for old, new in _RENAMES.items():
        df = df.withColumnRenamed(old, new)
    for c in _INTS:
        df = df.withColumn(c, F.col(c).cast("int"))
    for c in _DOUBLES:
        df = df.withColumn(c, F.col(c).cast("double"))

    # pdays = 999 is a sentinel for "never contacted before", not a real number of days.
    df = (
        df.withColumn("was_previously_contacted", (F.col("days_since_previous_contact") != 999).cast("int"))
          .withColumn("days_since_previous_contact",
                      F.when(F.col("days_since_previous_contact") == 999, None)
                       .otherwise(F.col("days_since_previous_contact")))
          .withColumn("converted", (F.col("y") == "yes").cast("int"))
          .drop("y")
    )

    # Rows are in date order but carry only the month. Each time the month number
    # goes backwards, a new calendar year has started.
    month_map = F.create_map(*[F.lit(x) for kv in MONTHS.items() for x in kv])
    order = Window.orderBy("source_row_number")
    df = (
        df.withColumn("call_month_num", month_map[F.col("call_month")])
          .withColumn("_prev_month", F.lag("call_month_num").over(order))
          .withColumn("_new_year", F.when(F.col("call_month_num") < F.col("_prev_month"), 1).otherwise(0))
          .withColumn("call_year", F.lit(FIRST_YEAR) + F.sum("_new_year").over(
              order.rowsBetween(Window.unboundedPreceding, Window.currentRow)))
          .withColumn("call_month_start", F.make_date("call_year", "call_month_num", F.lit(1)))
          .drop("_prev_month", "_new_year")
    )

    # 12 rows are exact copies of another row on every field. Keep the first.
    business_cols = [c for c in df.columns
                     if c not in ("source_row_number", "source_file", "ingested_at")]
    first = Window.partitionBy(*business_cols).orderBy("source_row_number")
    df = (df.withColumn("_dup_rank", F.row_number().over(first))
            .filter("_dup_rank = 1").drop("_dup_rank"))

    return df.withColumn("lead_id", F.format_string("L%06d", F.col("source_row_number")))


# --------------------------------------------------------------------------- gold
def build_gold_features(silver: DataFrame) -> DataFrame:
    """Model-ready table: pre-call features only, plus label and time keys.

    call_duration_sec is excluded because it is only known after the call ends
    (and a 0-second call is always a non-conversion), so it would leak the outcome.
    """
    return silver.select(
        "lead_id", "source_row_number", "call_year", "call_month_start",
        *CATEGORICAL_FEATURES, *NUMERIC_FEATURES, "converted",
    )


def build_gold_monthly(silver: DataFrame) -> DataFrame:
    """Monthly campaign summary for reporting and drift analysis."""
    return (
        silver.groupBy("call_month_start")
        .agg(
            F.count("*").alias("calls"),
            F.sum("converted").alias("conversions"),
            F.round(F.avg("converted"), 4).alias("conversion_rate"),
            F.round(F.avg("contacts_this_campaign"), 2).alias("avg_contacts_per_lead"),
            F.round(F.avg("was_previously_contacted"), 4).alias("share_previously_contacted"),
            F.round(F.avg("euribor_3m"), 3).alias("avg_euribor_3m"),
            F.round(F.avg("nr_employed"), 1).alias("avg_nr_employed"),
        )
        .orderBy("call_month_start")
    )
