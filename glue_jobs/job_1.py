import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# -----------------------------
# Args
# -----------------------------
args = getResolvedOptions(sys.argv, ["JOB_NAME", "SRC_CUSTOMERS", "SRC_ORDERS", "TGT_PATH"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

src_customers = args["SRC_CUSTOMERS"]  # e.g. s3://my-bucket/lineage/customers/
src_orders    = args["SRC_ORDERS"]     # e.g. s3://my-bucket/lineage/orders/
tgt_path      = args["TGT_PATH"]       # e.g. s3://my-bucket/lineage/output/customer_order_mart/

# (Optional) Make lineage output deterministic for tests
spark.conf.set("spark.sql.session.timeZone", "UTC")

# -----------------------------
# Source 1: Customers
# -----------------------------
df_customers = (
    spark.read.format("parquet")
    .load(src_customers)
    .select(
        F.col("customer_id").cast("string").alias("customer_id"),
        F.col("full_name").cast("string").alias("customer_name"),
        F.col("email").cast("string").alias("customer_email"),
        F.col("country").cast("string").alias("customer_country"),
        F.col("created_at").cast("timestamp").alias("customer_created_at")
    )
)

# Transformations on customers (clean + standardize)
df_customers_t = (
    df_customers
    .withColumn("customer_email_norm", F.lower(F.trim(F.col("customer_email"))))
    .withColumn("customer_country_norm", F.upper(F.trim(F.col("customer_country"))))
    .withColumn("customer_domain", F.regexp_extract(F.col("customer_email_norm"), r"@(.+)$", 1))
    .withColumn("customer_is_corporate", F.col("customer_domain").isin("example.com", "corp.com"))
    .filter(F.col("customer_id").isNotNull())
)

# Dedup customers (keep latest record per customer_id)
w_cust = Window.partitionBy("customer_id").orderBy(F.col("customer_created_at").desc_nulls_last())
df_customers_dedup = (
    df_customers_t
    .withColumn("cust_rn", F.row_number().over(w_cust))
    .filter(F.col("cust_rn") == 1)
    .drop("cust_rn")
)

# -----------------------------
# Source 2: Orders
# -----------------------------
df_orders = (
    spark.read.format("parquet")
    .load(src_orders)
    .select(
        F.col("order_id").cast("string").alias("order_id"),
        F.col("customer_id").cast("string").alias("customer_id"),
        F.col("order_ts").cast("timestamp").alias("order_ts"),
        F.col("order_status").cast("string").alias("order_status"),
        F.col("amount").cast("double").alias("amount"),
        F.col("currency").cast("string").alias("currency")
    )
)

# Transformations on orders (filter + derived + normalization)
df_orders_t = (
    df_orders
    .withColumn("order_status_norm", F.upper(F.trim(F.col("order_status"))))
    .withColumn("currency_norm", F.upper(F.trim(F.col("currency"))))
    .withColumn("order_date", F.to_date(F.col("order_ts")))
    .withColumn("amount_usd", F.when(F.col("currency_norm") == "USD", F.col("amount"))
                           .when(F.col("currency_norm") == "EUR", F.col("amount") * F.lit(1.10))  # dummy FX
                           .otherwise(F.col("amount") * F.lit(1.00)))
    .withColumn("is_successful_order", F.col("order_status_norm").isin("PAID", "SHIPPED", "DELIVERED"))
    .filter(F.col("order_id").isNotNull())
    .filter(F.col("customer_id").isNotNull())
)

# Optional: keep only last 90 days (good lineage filter step)
df_orders_recent = df_orders_t.filter(F.col("order_ts") >= F.date_sub(F.current_timestamp(), 90))

# -----------------------------
# Join: Customers + Orders
# -----------------------------
df_joined = (
    df_orders_recent.alias("o")
    .join(df_customers_dedup.alias("c"), on="customer_id", how="left")
)

# -----------------------------
# Aggregations (customer mart)
# -----------------------------
df_customer_metrics = (
    df_joined
    .groupBy(
        F.col("customer_id"),
        F.col("customer_name"),
        F.col("customer_email_norm"),
        F.col("customer_country_norm"),
        F.col("customer_is_corporate")
    )
    .agg(
        F.countDistinct("order_id").alias("orders_cnt_90d"),
        F.sum(F.when(F.col("is_successful_order"), F.col("amount_usd")).otherwise(F.lit(0.0))).alias("revenue_usd_90d"),
        F.max("order_ts").alias("last_order_ts"),
        F.min("order_ts").alias("first_order_ts")
    )
    .withColumn("avg_order_value_usd_90d",
                F.when(F.col("orders_cnt_90d") > 0, F.col("revenue_usd_90d") / F.col("orders_cnt_90d"))
                 .otherwise(F.lit(None).cast("double")))
)

# -----------------------------
# Window: customer segment label
# -----------------------------
seg_w = Window.orderBy(F.col("revenue_usd_90d").desc_nulls_last())
df_out = (
    df_customer_metrics
    .withColumn("revenue_rank", F.dense_rank().over(seg_w))
    .withColumn(
        "customer_segment",
        F.when(F.col("revenue_usd_90d") >= 5000, F.lit("PLATINUM"))
         .when(F.col("revenue_usd_90d") >= 1000, F.lit("GOLD"))
         .when(F.col("revenue_usd_90d") >= 100,  F.lit("SILVER"))
         .otherwise(F.lit("BRONZE"))
    )
    .withColumn("lineage_job_name", F.lit(args["JOB_NAME"]))
    .withColumn("lineage_run_ts", F.current_timestamp())
)

# -----------------------------
# Target write (S3 Parquet)
# -----------------------------
(
    df_out
    .repartition(1)  # for test determinism; remove for real scale
    .write
    .mode("overwrite")
    .format("parquet")
    .option("compression", "snappy")
    .save(tgt_path)
)

job.commit()