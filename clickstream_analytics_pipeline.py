# Databricks notebook source
import os
from pyspark.sql import functions as F

os.environ['RAW_DATA_PATH'] = 'dbfs:/FileStore/juanc_parra/clickstream/raw'
os.environ['DATA_LAKE_PATH'] = 'dbfs:/FileStore/juanc_parra/clickstream'
os.environ['AUTOLOADER_CHECKPOINT_PATH'] = 'dbfs:/tmp/juanc_parra/clickstream_checkpoint'
os.environ['MODEL_NAME'] = 'clickstream_predictor'

# COMMAND ----------

RAW_DATA_PATH = os.environ.get('RAW_DATA_PATH')
DATA_LAKE_PATH = os.environ.get('DATA_LAKE_PATH')
AUTOLOADER_CHECKPOINT_PATH = os.environ.get('AUTOLOADER_CHECKPOINT_PATH')
MODEL_NAME = os.environ.get('MODEL_NAME')

# COMMAND ----------

(spark.readStream  # Begins the structured streaming read process
  .format("cloudFiles")  # Specifies the file source format
  .option("cloudFiles.format", "csv")  # Specifies that the files are in CSV format
  .option("cloudFiles.schemaLocation", AUTOLOADER_CHECKPOINT_PATH)  # Set the schema location of cloud files
  .load(RAW_DATA_PATH)  # Loads the data from the defined constant path
  .select("*", 
          F.col("_metadata.file_path").alias("source_file"), 
          F.current_timestamp().alias("processing_time"))  # Selects all columns, adds a column for the source file, and a timestamp column for processing time
  .writeStream  # Begins the structured streaming write process
  .option("mergeSchema", "true")  # Merge schema if any changes occur in incoming data
  .option("checkpointLocation", AUTOLOADER_CHECKPOINT_PATH)  # Sets the checkpoint location
  .trigger(availableNow=True)  # Controls the rate of data processing (process as soon as data available)
  .start(f"{DATA_LAKE_PATH}/staging/events"))  # Starts the stream to a specific destination

# COMMAND ----------

# Creating a DataFrame 'bronze_events' by reading the "events" dataset located in the staging area of the data lake.
# The Delta Lake format of Apache Spark is used to load the data. The specified path corresponds to the location in the DATA_LAKE_PATH with '/staging/events'.
# A filter is applied to keep only those events that have occurred before February 1, 2021.
# Also selecting specific fields from the dataset - event_time, event_type, product_id, category_id, category_code, brand, price, user_id, and user_session.
bronze_events = (
    spark
    .read
    .format("delta")
    .load(f"{DATA_LAKE_PATH}/staging/events")
    .filter(F.col("event_time") < '2021-02-01')
    .select(
        "event_time",
        "event_type",
        "product_id", 
        "category_id", 
        "category_code", 
        "brand", 
        "price",
        "user_id",
        "user_session"
        )
)

# Saving the bronze_events DataFrame to a new location in the data lake (/bronze/events), overwriting any existing data at this location.
# Allowing for schema evolution by setting 'overwriteSchema' to 'true'.
# Registering this DataFrame as a temporary table named 'bronze_events' using saveAsTable() method.
_ = (
    bronze_events
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("path", f"{DATA_LAKE_PATH}/bronze/events")
    .saveAsTable("bronze_events")
)

# COMMAND ----------

from pyspark.sql.types import TimestampType, StringType, FloatType

# Creating a DataFrame 'silver_events' from the 'bronze_events'. 
# Casting different fields to their respective types using Spark's withColumn operation.
# The event_time is cast as a Timestamp, event_type, product_id, category_id, category_code, brand, user_id, and user_session are treated as String type, and price is considered as Float type.
# Thereafter, any records missing at least one value in the 'event_time', 'event_type', 'product_id', or 'user_session' columns are dropped from the DataFrame using the dropna function.
# Then, duplicate rows based on the combination of 'event_time', 'event_type', 'category_id', and 'user_session' are removed via the dropDuplicates function.
silver_events = (
    bronze_events
    .withColumn("event_time", F.col("event_time").cast(TimestampType()))
    .withColumn("event_type", F.col("event_type").cast(StringType()))
    .withColumn("product_id", F.col("product_id").cast(StringType()))
    .withColumn("category_id", F.col("category_id").cast(StringType()))
    .withColumn("category_code", F.col("category_code").cast(StringType()))
    .withColumn("brand", F.col("brand").cast(StringType()))
    .withColumn("price", F.col("price").cast(FloatType()))
    .withColumn("user_id", F.col("user_id").cast(StringType()))
    .withColumn("user_session", F.col("user_session").cast(StringType()))
    .dropna(how='any', subset=['event_time', 'event_type', 'product_id', 'user_session'])
    .dropDuplicates(['event_time', 'event_type', 'category_id', 'user_session'])
)

# Saving the silver_events DataFrame to a new location in the data lake (/silver/events), overwriting any existing data at this location and allowing schema evolution by setting 'overwriteSchema' to 'true'.
# Then registering this DataFrame as a temporary table named 'silver_events'.
_ = (
    silver_events
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("path", f"{DATA_LAKE_PATH}/silver/events")
    .saveAsTable("silver_events")
)

# COMMAND ----------

# Creating session_metrics DataFrame by selecting specific fields from silver_events DataFrame.
# Calculating multiple metrics based on user session activities, such as total events, view events, cart events, and purchase events.
# Then, calculating the ratios of these activities relative to each other.
session_metrics = (
    silver_events
    .select('user_id','user_session','event_time','event_type')
    .withColumn('events', F.expr("COUNT(*) OVER(PARTITION BY user_id, user_session ORDER BY event_time)"))
    .withColumn('views', F.expr(f"COUNT_IF(event_type='view') OVER(PARTITION BY user_id, user_session ORDER BY event_time)"))
    .withColumn('carts', F.expr(f"COUNT_IF(event_type='cart') OVER(PARTITION BY user_id, user_session ORDER BY event_time)"))
    .withColumn('purchases', F.expr(f"COUNT_IF(event_type='purchase') OVER(PARTITION BY user_id, user_session ORDER BY event_time)"))
    .drop('event_type')
    .withColumn('view_to_events', F.expr("views/events"))
    .withColumn('carts_to_events', F.expr("carts/events"))
    .withColumn('purchases_to_events', F.expr("purchases/events"))
    .withColumn('carts_to_views', F.expr("carts/views"))
    .withColumn('purchases_to_views', F.expr("purchases/views"))
    .withColumn('purchases_to_carts', F.expr("purchases/carts"))
  )

# Writing the session_metrics DataFrame to a new location in the gold area of the data lake, replacing any existing data at this location.
# Allowing schema evolution by setting 'overwriteSchema' to 'true'.
# Lastly, registering this DataFrame as a temporary table named 'session_metrics' using saveAsTable() method.
_ = (
    session_metrics
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("path", f"{DATA_LAKE_PATH}/gold/session_metrics")
    .saveAsTable("session_metrics")
)

# COMMAND ----------

# Creating product_session_metrics DataFrame by selecting specific fields from silver_events DataFrame.
# Calculating various metrics based on user session activities for each product, like total events, view events, cart events, and purchase events.
# Then, calculating the ratios of these activities relative to each other.
product_session_metrics = (
    silver_events
    .select('user_id', 'user_session', 'product_id', 'event_time', 'event_type')
    .withColumn("events", F.expr("COUNT(*) OVER (PARTITION BY user_id, user_session, product_id ORDER BY event_time)"))
    .withColumn("views", F.expr("COUNT_IF(event_type='view') OVER (PARTITION BY user_id, user_session, product_id ORDER BY event_time)"))
    .withColumn("carts", F.expr("COUNT_IF(event_type='cart') OVER (PARTITION BY user_id, user_session, product_id ORDER BY event_time)"))
    .withColumn("purchases", F.expr("COUNT_IF(event_type='purchase') OVER (PARTITION BY user_id, user_session, product_id ORDER BY event_time)"))
    .drop('event_type')
    .withColumn('view_to_events', F.expr("views/events"))
    .withColumn('carts_to_events', F.expr("carts/events"))
    .withColumn('purchases_to_events', F.expr("purchases/events"))
    .withColumn('carts_to_views', F.expr("carts/views"))
    .withColumn('purchases_to_views', F.expr("purchases/views"))
    .withColumn('purchases_to_carts', F.expr("purchases/carts"))
  )

# Writing the product_session_metrics DataFrame to a new location in the gold zone of the data lake, replacing any existing data at this location.
# Allowing schema evolution by setting 'overwriteSchema' to 'true'.
# Finally, registering this DataFrame as a temporary table named 'product_session_metrics' using saveAsTable() method.
_ = (
    product_session_metrics
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("path", f"{DATA_LAKE_PATH}/gold/product_session_metrics")
    .saveAsTable("product_session_metrics")
)

# COMMAND ----------

# Creating user_metrics DataFrame by groupBy operations on 'user_id' and 'batch_time' fields from silver_events DataFrame.
# Then, calculating different metrics based on user activities like total events, view events, cart events, and purchase events.
# Also, calculating the cumulative sum of these activities per user over time.
# Then, calculating the ratios of these activities relative to each other.
user_metrics = (
    silver_events
    .withColumn("batch_time", F.expr("DATE_TRUNC('day', DATE_ADD(event_time, 1))"))
    .groupBy("user_id", "batch_time")
    .agg(
        F.count('*').alias('events'),
        F.sum(F.when(F.col('event_type') == 'view', 1).otherwise(0)).alias('views'),
        F.sum(F.when(F.col('event_type') == 'cart', 1).otherwise(0)).alias('carts'),
        F.sum(F.when(F.col('event_type') == 'purchase', 1).otherwise(0)).alias('purchases')
    )
    .withColumn("events", F.expr("SUM(events) OVER (PARTITION BY user_id ORDER BY batch_time)"))
    .withColumn("views", F.expr("SUM(views) OVER (PARTITION BY user_id ORDER BY batch_time)"))
    .withColumn("carts", F.expr("SUM(carts) OVER (PARTITION BY user_id ORDER BY batch_time)"))
    .withColumn("purchases", F.expr("SUM(purchases) OVER (PARTITION BY user_id ORDER BY batch_time)"))
    .withColumn('view_to_events', F.expr("views/events"))
    .withColumn('carts_to_events', F.expr("carts/events"))
    .withColumn('purchases_to_events', F.expr("purchases/events"))
    .withColumn('carts_to_views', F.expr("carts/views"))
    .withColumn('purchases_to_views', F.expr("purchases/views"))
    .withColumn('purchases_to_carts', F.expr("purchases/carts"))
)

# Writing the user_metrics DataFrame to a new location in the gold zone of the data lake, replacing any existing data at this location.
# Allowing schema evolution by setting 'overwriteSchema' to 'true'.
# Finally, registering this DataFrame as a temporary table named 'user_metrics' using saveAsTable() method.
_ = (
    user_metrics
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("path", f"{DATA_LAKE_PATH}/gold/user_metrics")
    .saveAsTable("user_metrics")
)

# COMMAND ----------



# COMMAND ----------

# Creating product_metrics DataFrame by groupBy operations on 'product_id' and 'batch_time' fields from silver_events DataFrame.
# Then, calculating different metrics based on product activities like total events, view events, cart events, and purchase events.
# Also, accumulating these activities per product over time.
# Then, calculating ratios of these activities relative to each other.
product_metrics = (
    silver_events
    .withColumn("batch_time", F.expr("DATE_TRUNC('day', DATE_ADD(event_time, 1))"))
    .groupBy("product_id", "batch_time")
    .agg(
        F.count('*').alias('events'),
        F.sum(F.when(F.col('event_type') == 'view', 1).otherwise(0)).alias('views'),
        F.sum(F.when(F.col('event_type') == 'cart', 1).otherwise(0)).alias('carts'),
        F.sum(F.when(F.col('event_type') == 'purchase', 1).otherwise(0)).alias('purchases')
    )
    .withColumn("events", F.expr("SUM(events) OVER (PARTITION BY product_id ORDER BY batch_time)"))
    .withColumn("views", F.expr("SUM(views) OVER (PARTITION BY product_id ORDER BY batch_time)"))
    .withColumn("carts", F.expr("SUM(carts) OVER (PARTITION BY product_id ORDER BY batch_time)"))
    .withColumn("purchases", F.expr("SUM(purchases) OVER (PARTITION BY product_id ORDER BY batch_time)"))
    .withColumn('view_to_events', F.expr("views/events"))
    .withColumn('carts_to_events', F.expr("carts/events"))
    .withColumn('purchases_to_events', F.expr("purchases/events"))
    .withColumn('carts_to_views', F.expr("carts/views"))
    .withColumn('purchases_to_views', F.expr("purchases/views"))
    .withColumn('purchases_to_carts', F.expr("purchases/carts"))
)

# Writing the product_metrics DataFrame to a new location in the gold zone of the data lake, replacing any existing data at this location.
# Allowing schema evolution by setting 'overwriteSchema' to 'true'.
# Finally, registering this DataFrame as a temporary table named 'product_metrics' using saveAsTable() method.
_ = (
    product_metrics
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("path", f"{DATA_LAKE_PATH}/gold/product_metrics")
    .saveAsTable("product_metrics")
)

# COMMAND ----------

user_product_metrics = (
    silver_events
    .withColumn("batch_time", F.expr("DATE_TRUNC('day', DATE_ADD(event_time, 1))"))
    .groupBy("user_id", "product_id", "batch_time")
    .agg(
        F.count('*').alias('events'),
        F.sum(F.when(F.col('event_type') == 'view', 1).otherwise(0)).alias('views'),
        F.sum(F.when(F.col('event_type') == 'cart', 1).otherwise(0)).alias('carts'),
        F.sum(F.when(F.col('event_type') == 'purchase', 1).otherwise(0)).alias('purchases')
    )
    .withColumn("events", F.expr("SUM(events) OVER (PARTITION BY user_id, product_id ORDER BY batch_time)"))
    .withColumn("views", F.expr("SUM(views) OVER (PARTITION BY user_id, product_id  ORDER BY batch_time)"))
    .withColumn("carts", F.expr("SUM(carts) OVER (PARTITION BY user_id, product_id  ORDER BY batch_time)"))
    .withColumn("purchases", F.expr("SUM(purchases) OVER (PARTITION BY user_id, product_id  ORDER BY batch_time)"))
    .withColumn('view_to_events', F.expr("views/events"))
    .withColumn('carts_to_events', F.expr("carts/events"))
    .withColumn('purchases_to_events', F.expr("purchases/events"))
    .withColumn('carts_to_views', F.expr("carts/views"))
    .withColumn('purchases_to_views', F.expr("purchases/views"))
    .withColumn('purchases_to_carts', F.expr("purchases/carts"))
)

_ = (
    user_product_metrics
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("path", f"{DATA_LAKE_PATH}/gold/user_product_metrics")
    .saveAsTable("user_product_metrics")
)

# COMMAND ----------

# Initiating a DataFrame raw_features that identifies events to which we wish to assign feature metrics.
# Including 'event_type' for validation and troubleshooting. 'Batch_time' is calculated as it would be at the event time.
raw_features = (
    silver_events
    .select('user_id','user_session','product_id','event_time','event_type') 
    .withColumn('batch_time', F.expr("DATE_TRUNC('day', event_time)")) 
)

# Defining a dictionary 'metrics_map' that specifies metrics tables and their corresponding key fields for joining with event data.
metrics_map = { 
  'session_metrics':['user_id','user_session','event_time'],
  'product_session_metrics':['user_id','user_session','product_id','event_time'],
  'user_metrics':['user_id','batch_time'],
  'product_metrics':['product_id','batch_time'],
  'user_product_metrics':['user_id','product_id','batch_time']
}

# Looping through each table in 'metrics_map' to attach respective metrics to each event record.
for table_name, key_fields in metrics_map.items():
  
  # Retrieving metrics from the current table.
  temp_features = spark.table(table_name)
  
  # Naming convention for fields from metric table when added in feature set: "<table_name>__<field_name>".
  prefix = table_name
  renamed_fields = [f"{c} as {prefix}__{c}" for c in temp_features.columns if c not in key_fields]
  
  # Renaming metric fields per the above defined naming convention.
  temp_features = temp_features.selectExpr(key_fields + renamed_fields)
 
  # Joining metrics to event instances.
  raw_features = (
    raw_features
      .join(
        temp_features,
        on=key_fields,
        how='left'
        )
  )

# Writing the raw_features DataFrame to a new location in the silver zone of the data lake, replacing any existing data at this location.
# Allowing schema evolution by setting 'overwriteSchema' to 'true'.
# Registering this DataFrame as a temporary table named 'raw_features' using saveAsTable() method.
_ = (
    raw_features
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("path", f"{DATA_LAKE_PATH}/silver/raw_features")
    .saveAsTable("raw_features")
)

# COMMAND ----------

# Creating a new DataFrame 'session_product_outcomes' by grouping 'silver_events'
# based on 'user_id', 'user_session', and 'product_id'.
# Using agg() function to perform group-wise calculations. Here it checks if 'event_type' for a group is 'purchase',
# If true, returns 1 else 0, then takes maximum of these values which effectively 
# tells us if there was at least one purchase event in the group.
session_product_outcomes = (
    silver_events
    .groupBy('user_id','user_session','product_id')
      .agg(
            F.max(F.expr("CASE WHEN event_type='purchase' THEN 1 ELSE 0 END")).alias('purchased')
        )
)

# Writing the session_product_outcomes DataFrame to a new location in the gold zone of the data lake, replacing any existing data.
# Allowing schema evolution by setting 'overwriteSchema' to 'true'.
# Registering this DataFrame as a temporary table named 'session_product_outcomes' using saveAsTable() method.
_ = (
    session_product_outcomes
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("path", f"{DATA_LAKE_PATH}/gold/session_product_outcomes")
    .saveAsTable("session_product_outcomes")
)

# COMMAND ----------

# Creating a new DataFrame 'raw_features_and_labels' by joining 'raw_features' with 'session_product_outcomes'
# on the fields 'user_id', 'user_session', and 'product_id'.
raw_features_and_labels = (
    raw_features
    .join(
        session_product_outcomes,
        on=['user_id', 'user_session', 'product_id']
    )
)

# Writing the raw_features_and_labels DataFrame to a new location in the silver zone of the data lake, replacing any existing data.
# Allowing schema evolution by setting 'overwriteSchema' to 'true'.
# Registering this DataFrame as a temporary table named 'raw_features_and_labels' using saveAsTable() method.
_ = (
    raw_features_and_labels
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("path", f"{DATA_LAKE_PATH}/silver/raw_features_and_labels")
    .saveAsTable("raw_features_and_labels")
)

# COMMAND ----------

# Creating a new DataFrame 'features_and_labels' by filtering 'raw_features_and_labels'
# for records where 'event_type' is not equal to 'purchase'.
# Dropping columns "event_type", "product_id", "event_time", "batch_time" from the DataFrame.
features_and_labels = (
    raw_features_and_labels
    .filter(F.col('event_type') != 'purchase')
    .drop("event_type", "product_id", "event_time", "batch_time")
)

# Writing the features_and_labels DataFrame to a new location in the silver zone of the data lake, replacing any existing data.
# Allowing schema evolution by setting 'overwriteSchema' to 'true'.
# Registering this DataFrame as a temporary table named 'features_and_labels' using saveAsTable() method.
_ = (
    features_and_labels
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("path", f"{DATA_LAKE_PATH}/silver/features_and_labels")
    .saveAsTable("features_and_labels")
)
