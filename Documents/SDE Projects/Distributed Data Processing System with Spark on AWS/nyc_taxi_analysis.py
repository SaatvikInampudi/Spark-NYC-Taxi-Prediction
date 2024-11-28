from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("NYCTaxiAnalysis") \
    .getOrCreate()

# Define schema
schema = StructType([
    StructField("VendorID", IntegerType(), True),
    StructField("tpep_pickup_datetime", TimestampType(), True),
    StructField("tpep_dropoff_datetime", TimestampType(), True),
    StructField("passenger_count", IntegerType(), True),
    StructField("trip_distance", DoubleType(), True),
    StructField("pickup_longitude", DoubleType(), True),
    StructField("pickup_latitude", DoubleType(), True),
    StructField("RatecodeID", IntegerType(), True),
    StructField("store_and_fwd_flag", StringType(), True),
    StructField("dropoff_longitude", DoubleType(), True),
    StructField("dropoff_latitude", DoubleType(), True),
    StructField("payment_type", IntegerType(), True),
    StructField("fare_amount", DoubleType(), True),
    StructField("extra", DoubleType(), True),
    StructField("mta_tax", DoubleType(), True),
    StructField("tip_amount", DoubleType(), True),
    StructField("tolls_amount", DoubleType(), True),
    StructField("improvement_surcharge", DoubleType(), True),
    StructField("total_amount", DoubleType(), True)
])

# Read data from S3
df = spark.read.csv("s3a://nyc-taxi-data-7vik/yellow_tripdata_2016-01.csv",
                    header=True, schema=schema)

# Filter out invalid data
df = df.filter(df.trip_distance > 0)
df = df.filter(df.fare_amount > 0)
df = df.filter(df.passenger_count > 0)

# Handle missing values
df = df.na.drop()

# Feature Engineering
# Add new columns for analysis
df = df.withColumn('pickup_hour', hour(df.tpep_pickup_datetime))
df = df.withColumn('pickup_day', date_format(df.tpep_pickup_datetime, 'E'))

# Analysis 1: Trips per Hour
trips_per_hour = df.groupBy('pickup_hour').count().orderBy('pickup_hour')
trips_per_hour.write.mode("overwrite").csv('s3a://nyc-taxi-data-7vik/output/trips_per_hour', header=True)

# Analysis 2: Top Pickup Locations
pickup_locations = df.groupBy('pickup_longitude', 'pickup_latitude').count()
top_pickup_locations = pickup_locations.orderBy(desc('count')).limit(10)
top_pickup_locations.write.mode("overwrite").csv('s3a://nyc-taxi-data-7vik/output/top_pickup_locations', header=True)

# Analysis 3: Average Fare per Hour
avg_fare_per_hour = df.groupBy('pickup_hour').avg('fare_amount').orderBy('pickup_hour')
avg_fare_per_hour.write.mode("overwrite").csv("s3a://nyc-taxi-data-7vik/output/avg_fare_per_hour", mode="overwrite")
    
# Analysis 4: Clustering Pickup Locations
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

# Prepare data for clustering
assembler = VectorAssembler(inputCols=['pickup_longitude', 'pickup_latitude'], outputCol='features')
cluster_data = assembler.transform(df)

# Remove rows with null or zero coordinates
cluster_data = cluster_data.filter((col('pickup_longitude') != 0) & (col('pickup_latitude') != 0))

# Apply KMeans clustering
kmeans = KMeans(k=5, seed=1)
model = kmeans.fit(cluster_data.select('features'))
clusters = model.transform(cluster_data)

# Save cluster centers and data
cluster_centers = model.clusterCenters()
clusters.select('pickup_longitude', 'pickup_latitude', 'prediction').write.csv(
    's3a://nyc-taxi-data-7vik/output/geo_clusters', header=True
)

# Analysis 5: Fare Prediction Model
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

# Prepare data for regression
assembler = VectorAssembler(
    inputCols=['trip_distance', 'passenger_count', 'pickup_hour'],
    outputCol='features'
)
regression_data = assembler.transform(df)
regression_data = regression_data.select('features', 'fare_amount')

# Split data into training and test sets
train_data, test_data = regression_data.randomSplit([0.7, 0.3], seed=42)

# Train Linear Regression model
lr = LinearRegression(featuresCol='features', labelCol='fare_amount')
lr_model = lr.fit(train_data)

# Evaluate the model
test_results = lr_model.evaluate(test_data)
print(f"RMSE: {test_results.rootMeanSquaredError}")
print(f"R^2: {test_results.r2}")

# Save the model
lr_model.save('s3a://nyc-taxi-data-7vik/output/fare_prediction_model')


# Stop Spark Session 
spark.stop()
