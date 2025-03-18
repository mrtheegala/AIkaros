import os
import pandas as pd
import logging
from commons.utils.logger import setup_logger  # Import the setup_logger function

# Set up the logger
logger = setup_logger(__name__)

def ingest_data(file_path):
    """
    Ingests data from a file, automatically detecting the file type.

    Args:
        file_path (str): The path to the data file.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the ingested data.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file does not exist.
        pd.errors.ParserError: If there is an issue parsing the data file.
    """
    logger.info(f"Starting data ingestion from: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"File not found at: {file_path}")
        raise FileNotFoundError(f"File not found at {file_path}")

    try:
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ('.xls', '.xlsx'):
            df = pd.read_excel(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        logger.info(f"Data ingested successfully with shape: {df.shape}")
        return df

    except ValueError as e:
        logger.error(e)
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing data file: {e}")
        raise
