"""Run bronze -> silver -> gold end to end (locally or in Fabric)."""
from leadscore import lakehouse as lh
from leadscore import pipeline as pl
from leadscore import quality as dq

spark = lh.get_spark()
spark.sparkContext.setLogLevel("ERROR")

bronze = pl.build_bronze(spark, lh.raw_file_path(pl.SOURCE_FILE))
lh.write_table(bronze, "bronze_bank_calls")
bronze = lh.read_table(spark, "bronze_bank_calls")
bronze_rows = bronze.count()
print(f"bronze_bank_calls: {bronze_rows} rows")

silver = pl.build_silver(bronze)
dq.enforce(dq.check_silver(silver, bronze_rows), "silver")
lh.write_table(silver, "silver_calls")
silver = lh.read_table(spark, "silver_calls")

gold = pl.build_gold_features(silver)
dq.enforce(dq.check_gold_features(gold, silver.count()), "gold")
lh.write_table(gold, "gold_lead_features")
lh.write_table(pl.build_gold_monthly(silver), "gold_campaign_monthly")
print("done")
