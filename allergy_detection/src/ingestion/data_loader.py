from commons.utils.logger import setup_logger
from commons.utils.file_io import ingest_data

logger = setup_logger(__name__)

# This function now simply calls the common ingest_data function
def load_data(file_path):
    """
    Loads data using the common ingest_data utility.

    Args:
        file_path (str): Path to the data file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = ingest_data(file_path)
        if df is not None:
            logger.info(f"Data loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True, help='Path to the data file')
    args = parser.parse_args()

    load_data(args.file_path)