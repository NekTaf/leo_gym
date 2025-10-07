# Standard library
import os
import sys
from typing import Any, Dict, Tuple, Union

# Third-party
import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


loggers = Logger(
    folder=None,
    output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
)



# Custom policy save frequency 
class CustomCheckpoint(BaseCallback):
    def __init__(self, save_path: str, 
                 name_prefix: str = "policy",
                 save_freq: int = 100000):
        super().__init__()
        
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_freq = save_freq

        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            fname = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.num_timesteps:08d}_steps"
            )
            self.model.save(fname)
        return True
