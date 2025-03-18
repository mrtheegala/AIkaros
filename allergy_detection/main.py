import argparse
import sys
import os
from commons.utils.logger import setup_logger
from commons.mlflow_utils.mlflow_manager import start_mlflow_run,run_mlflow_experiment,end_mlflow_run

# Initialize logger
logger = setup_logger(__name__)

def main(args):
    try:
        logger.info("Starting MLflow pipeline execution...")
        experiment_name = "allergy_detection"

        # Start a single parent MLflow run
        parent_run_id = start_mlflow_run(experiment_name)
        logger.info(f"Parent Run ID: {parent_run_id}")

        steps = [
            ("ingest", {"file_path": args.file_path}),

            ("select_features", {
                "file_path": args.file_path,
                "selected_columns": args.selected_columns,
                "selected_features_path": args.selected_features_path

            }),

            ("preprocess", {
                "file_path": args.selected_features_path,
                "cleaned_data_path": args.cleaned_data_path
            }),


            ("concatenate", {
                "file_path": args.cleaned_data_path,
                "output_file": args.concatenated_data_path
            }),

            
            ("transform", {
                "file_path": args.concatenated_data_path,
                "output_path": args.embeddings_output_path,
                "model_name": args.model_name
              }),

            ("embeddings", {
                "file_path": args.embeddings_output_path,
                "mongo_uri": args.mongo_uri, 
                "database": args.database,
                "collection": args.collection, 

            })
        ]

        for step_name, params in steps:
            logger.info(f"Starting '{step_name}' step...")
            try:
                run_mlflow_experiment(experiment_name, step_name, params)
                logger.info(f"Completed '{step_name}' step successfully.")
            except Exception as e:
                logger.error(f"Error in '{step_name}' step: {str(e)}", exc_info=True)
                sys.exit(1)

        # End the parent run
        end_mlflow_run()
        logger.info("MLflow pipeline execution completed successfully!")

    except Exception as e:
        logger.critical(f"Fatal Error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow pipeline for allergy detection")

    # ingest
    parser.add_argument('--file_path', type=str, required=True)

    # preprocess
    parser.add_argument('--cleaned_data_path', type=str, required=True)

    # feature selection
    parser.add_argument('--selected_columns', type=str, required=True)
    parser.add_argument('--selected_features_path', type=str, required=True)
    
    # concatenate
    parser.add_argument('--concatenated_data_path', type=str, required=True)

    #transform
    parser.add_argument('--embeddings_output_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)

    #embeddings
    parser.add_argument('--mongo_uri', type=str, required=True)
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--collection', type=str, required=True)

    args = parser.parse_args()
    main(args)