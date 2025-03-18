import os
import pandas as pd
import logging
import argparse
from commons.utils.logger import setup_logger
from commons.mlflow_utils.mlflow_manager import log_params, log_artifact

# Set up logging
logger = setup_logger(__name__)

def concatenate_columns(df, output_file):  # Changed to accept DataFrame directly
    """
    Concatenates all columns of a DataFrame into a single text column and saves the result.

    Args:
        df (pd.DataFrame): The DataFrame to concatenate.
        output_file (str): Path to save the concatenated data.
    """
    try:
        logger.info("Concatenating columns...")

        # Concatenate all columns into a single text column
        df['concat_text'] = df.apply(lambda x: ' '.join(x.astype(str)).lower(),axis = 1)  # More descriptive column name

        # Save the concatenated data
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)

        logger.info(f"Concatenated data saved to {output_file}")

        # Log with MLflow
        log_params({"output_file": output_file})  # Use log_params from mlflow_manager
        log_artifact(output_file)  # Use log_artifact from mlflow_manager

    except Exception as e:
        logger.error(f"Error during concatenation: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save concatenated data')

    args = parser.parse_args()

    # Load the dataset using ingest_data
    from commons.utils.file_io import ingest_data
    df = ingest_data(args.file_path)

    if df is not None:
        concatenate_columns(df, args.output_file)