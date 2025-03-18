import os
import pandas as pd
import logging
from commons.utils.logger import setup_logger

# Set up logging
logger = setup_logger(__name__)

def preprocess_data(df, preprocessed_data_path):
    """
    Preprocesses data by handling missing values:
    - Numerical columns: Fill missing values with the mean.
    - Categorical columns: Drop rows with missing values.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        preprocessed_data_path (str): Path to save the preprocessed data.

    Returns:
        pd.DataFrame or None: Preprocessed DataFrame if successful, None otherwise.
    """
    try:
        logger.info("Starting data preprocessing...")

        # Identify numerical and categorical columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(exclude=["number"]).columns

        logger.info(f"Numeric columns: {list(numeric_cols)}")
        logger.info(f"Categorical columns: {list(categorical_cols)}")

        # Handle missing values
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())  # Impute numerical columns
        df = df.dropna(subset=categorical_cols)  # Drop rows with missing categorical values

        # Log data shape after preprocessing
        logger.info(f"Data shape after preprocessing: {df.shape}")

        # Ensure the directory exists before saving the file
        os.makedirs(os.path.dirname(preprocessed_data_path), exist_ok=True)

        # Save the preprocessed data
        df.to_csv(preprocessed_data_path, index=False)
        logger.info(f"Preprocessed data saved to: {preprocessed_data_path}")

        return df

    except Exception as e:
        logger.exception(f"An unexpected error occurred during preprocessing: {e}")
        return None
