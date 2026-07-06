
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

from functools import partial


import logging

# logging.basicConfig(level=logging.DEBUG) 

# logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)

# ===== Environment configurations =====
env_cfg = CamEnvConfig(
    high_action=[55, 41],
    low_action=[9, 1],
    
    reduced_obs = True,
    
    # Change action space as to not allign with opposite sides of the orbit
    # high_action=[20, 15],
    # low_action=[9, 1],
    
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
        conjunction_time_window_index=(300, 900),
        Droe_ranges=[
            [0,0], #ada
            
            [-500,+500], #adl
            [-300,+300], #adex
            [-300,+300], #adey
            
            # [0,0], #adl
            # [0,0], #adex
            # [0,0], #adey
            
            [0,0], #adix
            [0,0] #adiy
        ],
        C_rtn_s_ranges = [
            [100,300],
            [200,400],
            [100,300]
        ],
        
        C_rtn_p_ranges = [
            [50,200],
            [100,300],
            [50,200]
        ],

        radius_combined_ranges=(25,75),
        
        beta_sampling_values=(0.8,0.8),
        Dlon_range=(-3e3,3e3),
        # Dlon_range=(-0,0),

        Decc_range=(0,0),

    ),
)

# ===== PPO configurations =====
ppo_cfg = PPOConfig(
    env_obs={},  
    env_actions={},
    gamma=0.99,
    policy_clip=0.2,
    gae_lambda=0.95,
    lr=3e-3,
    init_entropy_coef=0.001,
    batch_size=4000,
    target_kl=0.05,
    lr_decay_coef=0,
    epochs=10,
    n_envs=400,
    normalize_advantage=True,
    init_std=[0.4, 0.4],
    log_to_mlflow=True,
    device="cuda",
    continuous_dist_cls = SquashedNormal,
    net_arch=[64,64,T.nn.Tanh],
    policy_wrapper=PolicyNetwork,
    critic_wrapper=ValueNetwork,
    observation_encoder=ObservationEncoder,
    steps_per_env=20,
    save_nets_period=int(13),
    max_training_timesteps=int(3e6),
    trained_algorithm_config_path = None
)
