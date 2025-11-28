"""
Config script to train the reconfiguration (R/T/N) policy with HPPO.
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
from leo_gym.gyms.recfg_gym import RecfgEnv, RecfgEnvConfig
from leo_gym.orbit.dynamics.dynamics import DynamicsConfig
from leo_gym.satellite.satellite_roe import SatelliteROEConfig
from leo_gym.utils.utils import seed_all, gen_rv0
from leo_gym.rl_algorithms.utils.utils import SquashedNormal, Normal
from leo_gym.rl_algorithms.h_ppo.actor_critic_nets import (
    PolicyNetwork,
    ValueNetwork,
    ObservationEncoder,
)


# ===== Environment configuration =====
dyn_kwargs = dict(
    flag_rtn_thrust=True,
    flag_mass_loss=False,
    flag_pert_moon=False,
    flag_pert_sun=False,
    flag_pert_srp=False,
    flag_pert_drag=False,
    flag_pert_irr_grav=False,
    eph_time_0=6.338304141847866e8,
    m=150.0,
    f_max=0.018,
    Isp=860.0,
    Ad=1.3,
    Cd=2.2,
    Cr=1.3,
    As=1.3,
    mf=136.0,
)

sat_cfg = SatelliteROEConfig(
    dt=60.0,
    days=0.5,
    ideal_traj_params=DynamicsConfig(**dyn_kwargs),
    pert_traj_params=DynamicsConfig(**dyn_kwargs),
    rv0=gen_rv0(sma=7000e3),  # simple circular LEO start to avoid None
)

env_cfg = RecfgEnvConfig(
    low_action=[9, 0],
    high_action=[55, 41],
    satellite_config=sat_cfg,
    satellite_observation_feature_size=6,
    continuous_actions_size=2,
    discrete_actions_size=7,  # R/T/N with +/- and coast
    f_max=sat_cfg.pert_traj_params.f_max,
    max_time_index=int(24 * sat_cfg.days * 60 * 60 / sat_cfg.dt),
    target_roe=None,
    target_tolerance=25.0,
    reward_distance_scale=1e4,
    success_reward=10.0,
    fuel_penalty_weight=1e-2,
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
    n_envs=16,
    normalize_advantage=True,
    init_std=[0.4, 0.4],
    log_to_mlflow=True,
    device="cpu",
    activation_fun=SquashedNormal,
    encoder_hidden_size=256,
    policy_wrapper=PolicyNetwork,
    critic_wrapper=ValueNetwork,
    observation_encoder=ObservationEncoder,
    default_num_envs=16, # Number of parallel environments (default = 100)
    steps_per_env=20,
    save_nets_period=int(1e5),
    max_training_timesteps=int(1e6),
    trained_algorithm_config_path=None,
)


class TrainingConfig(BaseModel):
    tracking_uri: Optional[str] = None
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    seed: Optional[int] = None
    model_config = ConfigDict(frozen=True)


training_cfg = TrainingConfig(
    tracking_uri="mlruns",
    experiment_name="recfg_hppo",
    run_name=None,
)
