import mlflow
from commons.utils.logger import setup_logger

logger = setup_logger(__name__)

# Set the tracking URI
mlflow.set_tracking_uri("azureml://centralindia.api.azureml.ms/mlflow/v1.0/subscriptions/984f0ec9-e17c-43ae-a685-91c37732d7c4/resourceGroups/plevenn-plm-dev-reg/providers/Microsoft.MachineLearningServices/workspaces/ple-ml-dev-workspace")

def start_mlflow_run(experiment_name):
    """
    Starts an MLflow parent run if not already active.

    Args:
        experiment_name (str): The name of the MLflow experiment.

    Returns:
        str: The parent run ID.
    """
    try:
        mlflow.set_experiment(experiment_name)

        if not mlflow.active_run():
            run = mlflow.start_run(run_name="Main_Run")
            logger.info(f"Started MLflow parent run: {run.info.run_id}")
            return run.info.run_id
        else:
            return mlflow.active_run().info.run_id
    except Exception as e:
        logger.error(f"Error starting MLflow run: {e}", exc_info=True)
        return None

def ensure_active_run():
    """Ensures that an MLflow run is active."""
    if not mlflow.active_run():
        mlflow.start_run()

def end_mlflow_run():
    """
    Ends the currently active MLflow run if one exists.
    """
    try:
        if mlflow.active_run():
            mlflow.end_run()
            logger.info("MLflow parent run ended successfully.")
    except Exception as e:
        logger.error(f"Error ending MLflow run: {e}", exc_info=True)

def run_mlflow_experiment(experiment_name, entry_point, parameters):
    """
    Runs an MLflow experiment with the specified entry point and parameters.
    
    Args:
        experiment_name (str): The name of the MLflow experiment.
        entry_point (str): The name of the entry point to run.
        parameters (dict): A dictionary of parameters to pass to the entry point.
    """
    try:
        parent_run_id = start_mlflow_run(experiment_name)

        with mlflow.start_run(run_name=entry_point, nested=True):
            mlflow.log_param("entry_point", entry_point)
            for key, value in parameters.items():
                mlflow.log_param(key, value)

            mlflow.run(".", entry_point=entry_point, parameters=parameters)

    except Exception as e:
        logger.error(f"An error occurred while running the MLflow experiment: {e}", exc_info=True)

def log_params(params):
    """Logs model parameters to MLflow."""
    try:
        ensure_active_run()
        for key, value in params.items():
            mlflow.log_param(key, value)
    except Exception as e:
        logger.error(f"An error occurred while logging parameters: {e}")

def log_metrics(metrics):
    """Logs model metrics to MLflow."""
    try:
        ensure_active_run()
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
    except Exception as e:
        logger.error(f"An error occurred while logging metrics: {e}")

def log_model(model, model_name):
    """Logs the model to MLflow."""
    try:
        ensure_active_run()
        mlflow.sklearn.log_model(model, model_name)
    except Exception as e:
        logger.error(f"An error occurred while logging the model: {e}")

def log_artifact(artifact_path, artifact_folder=None):
    """Logs an artifact to MLflow."""
    try:
        ensure_active_run()
        mlflow.log_artifact(artifact_path, artifact_folder)
    except Exception as e:
        logger.error(f"An error occurred while logging the artifact: {e}")

def register_model(model_registry_name, model_name, model_uri):
    """Registers the model in MLflow."""
    try:
        current_run_id = get_current_run_id()
        if current_run_id is None:
            logger.error("No active MLflow run found. Cannot register the model.")
            return
        
        model_uri = f"runs:/{current_run_id}/model"  # Ensure model_uri is set properly
        registered_model = mlflow.register_model(model_uri, model_registry_name)
        logger.info(f"Model registered in the registry: {model_registry_name}/{model_name}")
    except Exception as e:
        logger.error(f"An error occurred while registering the model: {e}")

def get_current_run_id():
    """Returns the ID of the currently active MLflow run."""
    try:
        active_run = mlflow.active_run()
        run_id = active_run.info.run_id if active_run else None
        logger.info(f"Current active run ID: {run_id}")
        return run_id
    except Exception as e:
        logger.error(f"An error occurred while retrieving the current run ID: {e}")
        return None
