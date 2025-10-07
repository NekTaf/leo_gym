# Third-party
import mlflow


class MLflowLogger:
    """
    Wrapper around mlflow.log_metric
    """

    def __init__(self, run_name: str = None):
        if run_name is not None:
            mlflow.set_experiment(run_name)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        mlflow.log_metric(key=key, value=value, step=step)

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_text(self, key: str, text: str) -> None:
        mlflow.log_text(text, artifact_file=key + ".txt")
            