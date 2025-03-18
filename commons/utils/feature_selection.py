import pandas as pd
import logging
import os
from commons.utils.logger import setup_logger

# Set up logging
logger = setup_logger(__name__)

def select_features(df, selected_columns, selected_features_path):  # Accepts DataFrame
    """Selects specified features from a DataFrame."""
    try:
        logger.info("Starting feature selection...")

        # Check if selected columns exist in the DataFrame
        missing_columns = [col for col in selected_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns: {', '.join(missing_columns)}")
            return None

        # Select specified columns
        selected_data = df[selected_columns]
        logger.info(f"Features selected: {selected_columns}")
        os.makedirs(os.path.dirname(selected_features_path), exist_ok=True)
        selected_data.to_csv(selected_features_path, index=False)
        logger.info(f"Selected features saved to: {selected_features_path}")
        return selected_data

    except Exception as e:
        logger.error(f"An error occurred during feature selection: {e}")
        return None