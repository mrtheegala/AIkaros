from commons.utils.logger import setup_logger
from commons.utils.data_preprocessing import preprocess_data

logger = setup_logger(__name__)

def clean_data(df, preprocessed_data_path):
    """
    Cleans the data using the common preprocess_data utility.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        preprocessed_data_path (str): Path to save the cleaned data.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    try:
        df = preprocess_data(df, preprocessed_data_path)
        return df
    except Exception as e:
        logger.error(f"An error occurred during cleaning: {e}")
        return None

if __name__ == "__main__":
    import argparse
    from commons.utils.file_io import ingest_data

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--cleaned_data_path', type=str, required=True, help='Path to save cleaned data')
    args = parser.parse_args()

    df = ingest_data(args.file_path)
    if df is not None:
        clean_data(df, args.cleaned_data_path)