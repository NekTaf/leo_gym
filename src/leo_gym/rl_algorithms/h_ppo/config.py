# Standard library
from typing import Any, Dict, List, Union, Literal, Callable
from dataclasses import dataclass

# Third-party
from gymnasium.spaces import Space
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ImportString

# Local
from leo_gym.rl_algorithms.h_ppo.actor_critic_nets import (
    PolicyNetwork,
    ValueNetwork,
    ObservationEncoder,   
)

from leo_gym.rl_algorithms.utils.utils import (
    SquashedNormal,
    Normal,
)
from typing import Any, List, Union, Callable, Literal, Optional
import importlib
import torch.nn as nn
import json
import json, re, importlib
from pathlib import Path

class PPOConfig(BaseModel):

    @classmethod
    def load_cfg_params_from_file(cls, cfg_file_path: str | Path) -> "PPOConfig":
        text = Path(cfg_file_path).read_text(encoding="utf-8")
        text = re.sub(r"""\"<class '([^']+)'>\"""", r'"\1"', text)
        cfg_kwargs = json.loads(text)

        for k in ("policy_wrapper", "critic_wrapper", "observation_encoder", "continuous_dist_cls"):
            v = cfg_kwargs.get(k)
            if isinstance(v, str) and "." in v:
                mod, name = v.rsplit(".", 1)
                cfg_kwargs[k] = getattr(importlib.import_module(mod), name)

        net_arch = cfg_kwargs.get("net_arch")
        if isinstance(net_arch, list) and net_arch and isinstance(net_arch[-1], str):
            if "Tanh" in net_arch[-1]:
                net_arch[-1] = nn.Tanh
            elif "ReLU" in net_arch[-1]:
                net_arch[-1] = nn.ReLU
            else:
                raise TypeError(net_arch[-1])

        return cls.model_validate(cfg_kwargs)

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

    steps_per_env: int
    max_training_timesteps: int
    save_nets_period: int #after how many updates to save (batch_size x save_nets_period)
    trained_algorithm_config_path: Optional[str] = None

    policy_wrapper: ImportString = PolicyNetwork
    critic_wrapper: ImportString = ValueNetwork
    observation_encoder: ImportString = ObservationEncoder
    hidden_layer_size: int = 256

    device: Literal["cpu", "cuda"] = "cuda"
    continuous_dist_cls: ImportString = Normal
    
    net_arch: List[ImportString]
    
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    

class TrainingConfig(BaseModel):
    tracking_uri: Optional[str] = ""
    experiment_name: Optional[str] = ""
    run_name: Optional[str] = ""
    seed: Optional[int] = 0
    model_config = ConfigDict(frozen=True)
