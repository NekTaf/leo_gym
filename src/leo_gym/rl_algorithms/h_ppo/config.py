# Standard library
from typing import Any, Dict, List, Union, Literal, Callable
from dataclasses import dataclass

# Third-party
from gymnasium.spaces import Space
from pydantic import BaseModel, ConfigDict, Field

# Local
from leo_gym.rl_algorithms.h_ppo.actor_critic_nets import (
    PolicyNetwork,
    ValueNetwork,
    ObservationEncoder,  # make sure this is exported in the module
)

from leo_gym.rl_algorithms.utils.utils import (
    SquashedNormal,
    Normal,
)
from typing import Any, List, Union, Callable, Literal, Optional

class PPOConfig(BaseModel):
    env_obs: Any
    env_actions: Any

    gamma: float
    gae_lambda: float
    policy_clip: float
    target_kl: float
    lr: float
    lr_decay_coef: float
    init_entropy_coef: float

    batch_size: int
    epochs: int
    n_envs: int

    init_std: Union[float, List[float]]

    log_to_mlflow: bool
    normalize_advantage: bool

    default_num_envs: int
    steps_per_env: int
    max_training_timesteps: int
    save_nets_period: int
    trained_algorithm_config_path: Optional[str] = None

    policy_wrapper: Any = PolicyNetwork
    critic_wrapper: Any = ValueNetwork
    observation_encoder: Any = ObservationEncoder
    encoder_hidden_size: int = 256

    device: Literal["cpu", "cuda"] = "cuda"
    activation_fun: Any = Normal

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
