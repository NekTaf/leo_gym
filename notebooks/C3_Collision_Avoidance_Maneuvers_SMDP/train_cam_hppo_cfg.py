
"""
Script used to train cam policy:

+ Uses async parallel environments for faster training
"""

# Standard library
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from typing import List, Optional

# Third-party
import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
import torch as T
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

# Local
from leo_gym.rl_algorithms.h_ppo.config import PPOConfig
from leo_gym.rl_algorithms.h_ppo.h_ppo_agent import Agent
from leo_gym.gyms.cam_gym import CamEnv, CamEnvConfig
from leo_gym.orbit.dynamics.dynamics import DynamicsConfig
from leo_gym.satellite.sat_debris_cluster import SatDebrisClusterConfig
from leo_gym.utils.utils import seed_all


# ===== Environment configurations =====
env_cfg = CamEnvConfig(
    high_action=[55, 41],
    low_action=[9, 1],
    max_time_index=1300,
    debris_observation_feature_size=7,
    max_observable_debris=1,
    satellite_observation_feature_size=6,
    continuous_actions_size=2,
    discrete_actions_size=3,
    p_max_limit=1e-3,
    adl_req=400,
    ade_norm_req=100,
    debris_cluster_config=SatDebrisClusterConfig(
        params_dyn=DynamicsConfig(
            flag_rtn_thrust=True,
            flag_mass_loss=False,
            flag_pert_moon=False,
            flag_pert_sun=False,
            flag_pert_srp=False,
            flag_pert_drag=False,
            flag_pert_irr_grav=False,
            eph_time_0=6.338304141847866e+08,
            m=150,
            f_max=18e-3,
            Isp=860,
            Ad=1.3,
            Cd=2.2,
            Cr=1.3,
            As=1.3,
            mf=136
        ),
        params_dyn_ideal=DynamicsConfig(
            flag_rtn_thrust=True,
            flag_mass_loss=False,
            flag_pert_moon=False,
            flag_pert_sun=False,
            flag_pert_srp=False,
            flag_pert_drag=False,
            flag_pert_irr_grav=False,
            eph_time_0=6.338304141847866e+08,
            m=150,
            f_max=18e-3,
            Isp=860,
            Ad=1.3,
            Cd=2.2,
            Cr=1.3,
            As=1.3,
            mf=136
        ),
        days=1,
        dt=60,
        max_debris=1,
        min_debris=1,
        conjunction_time_window_index=[550, 800]
    ),
)

# ===== PPO configurations =====
ppo_cfg = PPOConfig(
    env_obs={},  
    env_actions={},
    gamma=0.99,
    policy_clip=0.2,
    gae_lambda=0.95,
    lr=3e-4,
    init_entropy_coef=0.001,
    batch_size=1000,
    target_kl=0.05,
    lr_decay_coef=0,
    epochs=5,
    n_envs=100,
    use_sde=False,
    use_squashed_gaussian=True,
    normalize_advantage=True,
    init_std=[0.3, 0.3],
    log_to_mlflow=True,
    device="cuda"
)


class TrainingConfig(BaseModel):
    tracking_uri: Optional[str] = None
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    max_training_timesteps: int = None
    save_nets_period: int = None
    steps_per_env: int = None
    seed: Optional[int] = None
    model_config = ConfigDict(frozen=True)
    actor_file_path: Optional[str] = None,
    critic_file_path: Optional[str] = None
    trained_algorithm_config_path:Optional[str] = None
    default_num_envs: int = None

training_cfg = TrainingConfig(
    tracking_uri=None,
    experiment_name=None,
    max_training_timesteps=int(15e6),
    save_nets_period=int(1e5),
    steps_per_env=20,
    run_name=None,
    actor_file_path = None,
    critic_file_path = None,
    trained_algorithm_config_path = None,
    default_num_envs=100    
)
