#!/bin/bash

# pipeline.sh

# Variables
KEY_PAIR="7vikTestKeyPair.pem"
EC2_DNS="ec2-54-152-154-212.compute-1.amazonaws.com"
S3_SCRIPTS_BUCKET="s3://nyc-taxi-data-7vik/scripts/"
LOCAL_SCRIPTS_DIR="/home/hadoop/scripts/"
LOGS_DIR="/home/hadoop/output/logs/"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r scripts/requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install Python dependencies."
    exit 1
fi
echo "Python dependencies installed successfully."

# Step 1: Upload all scripts to S3
echo "Uploading scripts to S3..."
aws s3 cp scripts/ $S3_SCRIPTS_BUCKET --recursive
if [ $? -ne 0 ]; then
    echo "Failed to upload scripts to S3."
    exit 1
fi
echo "Scripts uploaded successfully."

# Step 2: Set permissions for SSH
echo "Setting permissions for SSH key..."
chmod 400 $KEY_PAIR
if [ $? -ne 0 ]; then
    echo "Failed to set permissions for SSH key."
    exit 1
fi
echo "Permissions set successfully."

# Step 3: SSH into EMR cluster and execute scripts
echo "Connecting to EMR cluster and executing scripts..."

ssh -i "$KEY_PAIR" hadoop@$EC2_DNS << EOF
    set -e  # Exit immediately if a command exits with a non-zero status
    
    # Navigate to home directory
    cd /home/hadoop/
    
    # Create scripts directory if it doesn't exist
    mkdir -p scripts
    
    # Copy scripts from S3
    echo "Downloading scripts from S3..."
    aws s3 cp $S3_SCRIPTS_BUCKET scripts/ --recursive
    if [ $? -ne 0 ]; then
        echo "Failed to download scripts from S3."
        exit 1
    fi
    echo "Scripts downloaded successfully."
    
    # Create logs directory
    mkdir -p $LOGS_DIR
    
    # Execute Preprocessing script with logging
    echo "Running Preprocessing script..."
    spark-submit scripts/preprocessing.py --input "s3a://nyc-taxi-data-7vik/yellow_tripdata_2016-01.csv" --output "s3a://nyc-taxi-data-7vik/output/preprocessed_data/" > $LOGS_DIR/preprocessing.log 2>&1
    echo "Preprocessing completed."
    
    # Execute Model Training script with logging
    echo "Running Model Training script..."
    spark-submit scripts/model_training.py --input "s3a://nyc-taxi-data-7vik/output/preprocessed_data/" --output "s3a://nyc-taxi-data-7vik/output/model_output/fare_prediction_model" > $LOGS_DIR/model_training.log 2>&1
    echo "Model Training completed."
    
    echo "All scripts executed successfully."
EOF

if [ $? -ne 0 ]; then
    echo "Failed to execute scripts on EMR cluster."
    exit 1
fi

echo "Pipeline execution completed successfully."