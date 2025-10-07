# Standard library
from typing import Any, Dict, List, Union, Literal

# Third-party
from gymnasium.spaces import Space
from pydantic import BaseModel, ConfigDict, Field

# Local
from leo_gym.rl_algorithms.h_ppo.actor_critic_nets import (
    PolicyNetwork,
    ValueNetwork,
)


class PPOConfig(BaseModel):
    env_obs: Any = Field(..., description="gymnasium environment space")
    
    env_actions: Any = Field(...,  description="gymnasium environment space")
    
    gamma: float = Field(..., gt=0, description="Discount factor")
    
    gae_lambda: float = Field(..., ge=0, le=1, description="GAE λ")
    
    policy_clip: float = Field(..., gt=0, description="Clipping ε")
    
    target_kl: float = Field(..., ge=0, description="Target KL divergence")
    
    lr: float = Field(..., gt=0, description="Learning rate")
    
    lr_decay_coef: float = Field(..., ge=0, description="LR decay coefficient")
    
    init_entropy_coef: float = Field(..., ge=0,  description="Initial entropy coef")
    
    batch_size: int = Field(..., gt=0, description="Batch size")
    
    epochs: int = Field(..., gt=1, description="Epochs per update")
    
    n_envs: int = Field(..., gt=0, description="Number of envs")
    
    use_squashed_gaussian: bool = Field(...,  description="Squash Gaussian output")
    
    init_std: Union[float, List[float]] = Field(...,  description="Initial std dev")
    
    log_to_mlflow: bool = Field(..., description="Enable MLflow logging")
    
    normalize_advantage: bool = Field(..., description="Normalize advantage")
    
    model_config = ConfigDict(arbitrary_types_allowed=True,
                              frozen=True)
    std_type: int = Field(0, description="0: Action‐noise exploration,\
                                        1: State-dependent Gaussian")
    
    policy_wrapper: Any = Field(PolicyNetwork,
                                description="Policy network class, with modified architectures")
    
    critic_wrapper: Any = Field(ValueNetwork,
                                description="Critic network class, with modified architectures")
    
    device: Literal["cpu", "cuda"] = Field(default="cpu",
                                           description="Device to run the model on (cpu or cuda)")


class LagrangianConfig(BaseModel):
    # Constraint / Lagrange settings
    constr_limit: float = Field(..., gt=0, description="Constraint limit")
    lag_lr: float = Field(..., gt=0, description="Lagrange LR")
    lag_max: float = Field(..., gt=0, description="Max Lagrange multiplier")

    # Flags to include cost penalty per action branch
    flag_lag_update_cont: bool = Field(..., description="Update continuous lag")
    flag_lag_update_dis: bool = Field(..., description="Update discrete lag")

    model_config = ConfigDict(frozen=True)
