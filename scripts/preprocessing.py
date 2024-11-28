# scripts/preprocessing.py

import argparse
from utils import get_spark_session, get_schema
from logging_setup import setup_logging
from error_handling import handle_exception
from pyspark.sql.functions import hour, date_format, col

def main(input_path, output_path):
    # Setup logging
    logger = setup_logging('/home/hadoop/output/logs/preprocessing.log')
    
    try:
        logger.info("Starting Data Preprocessing...")
        spark = get_spark_session()
        schema = get_schema()
        
        # Read data from S3
        logger.info(f"Reading data from {input_path}...")
        df = spark.read.csv(input_path, header=True, schema=schema)
        
        # Filter out invalid data
        logger.info("Filtering invalid data...")
        df = df.filter(col('trip_distance') > 0)
        df = df.filter(col('fare_amount') > 0)
        df = df.filter(col('passenger_count') > 0)
        
        # Handle missing values
        logger.info("Handling missing values...")
        df = df.na.drop()
        
        # Feature Engineering
        logger.info("Adding new features: pickup_hour and pickup_day...")
        df = df.withColumn('pickup_hour', hour(col('tpep_pickup_datetime')))
        df = df.withColumn('pickup_day', date_format(col('tpep_pickup_datetime'), 'E'))
        
        # Save preprocessed data to S3
        logger.info(f"Saving preprocessed data to {output_path}...")
        df.write.mode("overwrite").parquet(output_path)
        
        spark.stop()
        logger.info("Data Preprocessing completed successfully.")
        
    except Exception as e:
        handle_exception(logger, e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Preprocessing for NYCTaxiAnalysis')
    parser.add_argument('--input', required=True, help='S3 path to input CSV data')
    parser.add_argument('--output', required=True, help='S3 path to save preprocessed data')
    args = parser.parse_args()
    
    main(args.input, args.output)