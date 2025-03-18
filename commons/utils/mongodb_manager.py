import pymongo
from pymongo import MongoClient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingStore:
    def __init__(self, mongo_uri: str, db_name: str):
        """Initialize connection to MongoDB."""
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        logger.info(f"Connected to MongoDB database: {db_name}")

    def save_embeddings(self, collection_name: str, documents: list):
        """
        Save embeddings to the database.
        - If the collection is empty, insert all embeddings using insert_many.
        - Otherwise, update existing documents or insert new ones individually.

        Args:
            collection_name (str): The name of the MongoDB collection.
            documents (list): List of documents (each must contain '_id', 'embedding', and metadata).
        """
        collection = self.db[collection_name]

        if not documents:
            logger.warning("No embeddings provided to save.")
            return

        # Check if the collection is empty
        if collection.count_documents({}) == 0:
            collection.insert_many(documents)
            logger.info(f"Inserted {len(documents)} embeddings using insert_many.")
        else:
            for doc in documents:
                doc_id = doc["_id"]
                existing_doc = collection.find_one({"_id": doc_id})  # Check if document exists
                
                if existing_doc:
                    collection.update_one(
                        {"_id": doc_id}, 
                        {"$set": doc}  # Update all fields in the document
                    )
                    logger.info(f"Updated embedding for document ID: {doc_id}")
                else:
                    collection.insert_one(doc)
                    logger.info(f"Inserted new embedding for document ID: {doc_id}")

    def delete_embedding(self, collection_name: str, doc_id: str):
        """Delete an embedding from the database."""
        collection = self.db[collection_name]
        result = collection.delete_one({"_id": doc_id})
        if result.deleted_count:
            logger.info(f"Deleted embedding for document ID: {doc_id}")
        else:
            logger.warning(f"Document ID {doc_id} not found in collection {collection_name}")
