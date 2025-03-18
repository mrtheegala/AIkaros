import argparse
import pandas as pd
from commons.utils.file_io import ingest_data
from commons.utils.mongodb_manager import EmbeddingStore  # Import the updated MongoDB manager
from commons.utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

def save_to_mongodb(file_path, mongo_uri, database, collection):
    """
    Loads embeddings from a CSV file and saves them to MongoDB.

    Args:
        file_path (str): Path to the CSV file containing embeddings.
        mongo_uri (str): MongoDB connection URI.
        database (str): MongoDB database name.
        collection (str): MongoDB collection name.
    """
    try:
        logger.info(f"Loading embeddings from {file_path}")
        df = ingest_data(file_path)
        if df is None or "embedding" not in df.columns:
            logger.error("Failed to load embeddings data or missing 'embedding' column.")
            raise ValueError("Failed to load embeddings data or missing 'embedding' column.")

        # Convert 'embedding' column from string to list if necessary
        if isinstance(df["embedding"].iloc[0], str):
            df["embedding"] = df["embedding"].apply(eval)  # Convert string representation of lists to actual lists

        # Prepare documents for MongoDB
        documents = []
        for index, row in df.iterrows():
            doc = {
                "_id": str(index),  # Use row index as document ID (modify if needed)
                "embedding": row["embedding"],
                "metadata": row.drop("embedding").to_dict()  # Store other columns as metadata
            }
            documents.append(doc)

        # Connect to MongoDB and save embeddings
        logger.info(f"Connecting to MongoDB at {mongo_uri}, Database: {database}, Collection: {collection}")
        embedding_store = EmbeddingStore(mongo_uri, database)
        embedding_store.save_embeddings(collection, documents)
        
        logger.info(f"Successfully saved {len(documents)} embeddings to MongoDB collection '{collection}' in database '{database}'.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file with embeddings")
    parser.add_argument("--mongo_uri", type=str, required=True, help="MongoDB connection URI")
    parser.add_argument("--database", type=str, required=True, help="MongoDB database name")
    parser.add_argument("--collection", type=str, required=True, help="MongoDB collection name")

    args = parser.parse_args()
    
    logger.info("Starting the process to save embeddings to MongoDB.")
    try:
        save_to_mongodb(args.file_path, args.mongo_uri, args.database, args.collection)
        logger.info("Embedding storage process completed successfully.")
    except Exception as e:
        logger.error(f"Script terminated with an error: {e}")
