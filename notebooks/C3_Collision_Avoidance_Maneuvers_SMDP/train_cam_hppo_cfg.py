
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

from leo_gym.rl_algorithms.utils.utils import (
    SquashedNormal,
    Normal,
)

from leo_gym.rl_algorithms.h_ppo.actor_critic_nets import (
    PolicyNetwork,
    ValueNetwork,
    ObservationEncoder,  # make sure this is exported in the module
)



# ===== Environment configurations =====
env_cfg = CamEnvConfig(
    high_action=[55, 41],
    low_action=[9, 1],
    max_time_index=1300,
    p_max_limit=1e-3,
    adl_req=400,
    ade_norm_req=75,
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
        conjunction_time_window_index=[650, 800],
        Droe_ranges=[
            [0,0], #ada
            # [-1500,+1500], #adl
            # [-300,+300], #adex
            # [-300,+300], #adey
            [0,0], #adl
            [0,0], #adex
            [0,0], #adey
            [0,0], #adix
            [0,0] #adiy
        ],
        C_rtn_s_ranges = [
            [100,150],
            [150,200],
            [100,150]
        ],
        
        C_rtn_p_ranges = [
            [10,50],
            [75,100],
            [10,50]
        ],

        radius_combined_ranges=[10,100],
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
    normalize_advantage=True,
    init_std=[0.4, 0.4],
    log_to_mlflow=True,
    device="cuda",
    activation_fun = SquashedNormal,
    encoder_hidden_size=256,
    policy_wrapper=PolicyNetwork,
    critic_wrapper=ValueNetwork,
    observation_encoder=ObservationEncoder,
    default_num_envs=100, 
    steps_per_env=70,
    save_nets_period=int(1e5),
    max_training_timesteps=int(7e6),
    trained_algorithm_config_path = None

)


class TrainingConfig(BaseModel):
    tracking_uri: Optional[str] = None
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    seed: Optional[int] = None
    model_config = ConfigDict(frozen=True)

training_cfg = TrainingConfig(
    tracking_uri="/home/nektaf/mlruns",
    experiment_name="cam_shap_journal_2",
    run_name= None,
)
