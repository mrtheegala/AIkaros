import argparse
import pandas as pd
import mlflow
import mlflow.sentence_transformers
from sentence_transformers import SentenceTransformer
from commons.utils.file_io import ingest_data
from commons.mlflow_utils.mlflow_manager import log_params
from commons.utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

def generate_embeddings(file_path, output_path, model_name="allergy_detection"):
    """
    Loads preprocessed text, generates embeddings, and logs them to MLflow.

    Args:
        file_path (str): Path to the input CSV file.
        output_path (str): Path to save the embeddings CSV file.
        model_name (str): SentenceTransformer model name.
    """
    logger.info("Starting MLflow run for generating embeddings.")
    with mlflow.start_run():  # Start an MLflow run

        # Load data
        logger.info(f"Loading data from {file_path}")
        df = ingest_data(file_path)
        if df is None:
            logger.error("Failed to load input data.")
            raise ValueError("Failed to load input data.")
        logger.info(f"Data loaded successfully with {len(df)} records.")

        # Load SentenceTransformer model
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info("Model loaded successfully.")

        # Generate embeddings
        logger.info("Generating embeddings.")
        df["embedding"] = df["concat_text"].apply(lambda text: model.encode(text).tolist())
        logger.info(f"Embeddings generated successfully for {len(df)} records.")

        # Save embeddings to the specified output path
        logger.info(f"Saving embeddings to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info("Embeddings saved successfully.")
        
        mlflow.log_artifact(output_path)  # Log to MLflow
        logger.info("Embeddings logged to MLflow.")

        # Log the SentenceTransformer model to MLflow
        logger.info("Logging SentenceTransformer model to MLflow.")
        mlflow.sentence_transformers.log_model(
            model, artifact_path="sentence_transformer_model"
        )
        logger.info("Model logged to MLflow successfully.")

        # Log run parameters
        log_params({"file_path": file_path, "output_path": output_path, "model_name": model_name})
        logger.info("Run parameters logged to MLflow.")
        
        logger.info("MLflow run completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the embeddings CSV file")
    parser.add_argument("--model_name", type=str, required=True, help="SentenceTransformer model name")

    args = parser.parse_args()
    
    logger.info("Starting the embedding generation script.")
    try:
        generate_embeddings(args.file_path, args.output_path, args.model_name)
        logger.info("Embedding generation script completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
