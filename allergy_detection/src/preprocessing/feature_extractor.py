import argparse
import pandas as pd
from commons.utils.logger import setup_logger
from commons.utils.feature_selection import select_features
from commons.utils.file_io import ingest_data  # Importing ingest_data

logger = setup_logger(__name__)

def select_data_features(df, selected_columns, selected_features_path):
    """
    Selects features from the data using the common select_features utility.

    Args:
        df (pd.DataFrame): The DataFrame to select features from.
        selected_columns (list): List of columns to select.
        selected_features_path (str): Path to save the selected features.

    Returns:
        pd.DataFrame: The DataFrame with selected features or None if an error occurs.
    """
    try:
        if df is None or df.empty:
            logger.error("DataFrame is empty or None. Cannot proceed with feature selection.")
            return None

        logger.info(f"Starting feature selection. Initial shape: {df.shape}")

        # Ensure selected columns exist in DataFrame
        missing_columns = [col for col in selected_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}. Feature selection aborted.")
            return None

        df_selected = select_features(df, selected_columns, selected_features_path)

        if df_selected is not None:
            logger.info(f"Feature selection completed. Final shape: {df_selected.shape}")
        else:
            logger.error("Feature selection returned None.")

        return df_selected

    except Exception as e:
        logger.error(f"An error occurred during feature selection: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--selected_columns', type=str, required=True, help='Comma-separated list of columns to select')
    parser.add_argument('--selected_features_path', type=str, required=True, help='Path to save the selected features')
    args = parser.parse_args()

    selected_columns = [col.strip() for col in args.selected_columns.split(',')]

    df = ingest_data(args.file_path)  # Use ingest_data to load data
    if df is not None:
        select_data_features(df, selected_columns, args.selected_features_path)
    else:
        logger.error("Data ingestion failed. Feature selection process aborted.")
