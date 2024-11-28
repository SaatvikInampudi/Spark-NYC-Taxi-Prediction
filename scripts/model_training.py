# scripts/model_training.py

import argparse
from utils import get_spark_session
from logging_setup import setup_logging
from error_handling import handle_exception
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

def main(input_path, model_output_path):
    # Setup logging
    logger = setup_logging('/home/hadoop/output/logs/model_training.log')
    
    try:
        logger.info("Starting Model Training...")
        spark = get_spark_session()
        
        # Read preprocessed data
        logger.info(f"Reading preprocessed data from {input_path}...")
        df = spark.read.parquet(input_path)
        
        # Prepare data for regression
        logger.info("Assembling features for regression...")
        assembler = VectorAssembler(
            inputCols=['trip_distance', 'passenger_count', 'pickup_hour'],
            outputCol='features'
        )
        regression_data = assembler.transform(df).select('features', 'fare_amount')
        
        # Split data into training and test sets
        logger.info("Splitting data into training and test sets...")
        train_data, test_data = regression_data.randomSplit([0.7, 0.3], seed=42)
        
        # Train Linear Regression model
        logger.info("Training Linear Regression model...")
        lr = LinearRegression(featuresCol='features', labelCol='fare_amount')
        lr_model = lr.fit(train_data)
        
        # Evaluate the model
        logger.info("Evaluating the model on test data...")
        test_results = lr_model.evaluate(test_data)
        logger.info(f"RMSE: {test_results.rootMeanSquaredError}")
        logger.info(f"R^2: {test_results.r2}")
        
        # Save the model
        logger.info(f"Saving the trained model to {model_output_path}...")
        lr_model.write().overwrite().save(model_output_path)
        
        spark.stop()
        logger.info("Model Training completed successfully.")
        
    except Exception as e:
        handle_exception(logger, e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training for NYCTaxiAnalysis')
    parser.add_argument('--input', required=True, help='S3 path to preprocessed data')
    parser.add_argument('--output', required=True, help='S3 path to save the trained model')
    args = parser.parse_args()
    
    main(args.input, args.output)