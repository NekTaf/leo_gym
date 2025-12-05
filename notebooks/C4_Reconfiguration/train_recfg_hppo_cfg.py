"""
Config script to train the reconfiguration (R/T/N) policy with HPPO.
"""

# Standard library
from typing import Optional

# Third-party

import numpy as np
from pydantic import BaseModel, ConfigDict

# Local
from leo_gym.rl_algorithms.h_ppo.config import PPOConfig
from leo_gym.rl_algorithms.h_ppo.h_ppo_agent import Agent
from leo_gym.gyms.recfg_gym import RecfgEnvConfig
from leo_gym.satellite.satellite_roe import SatelliteROEConfig
from leo_gym.rl_algorithms.utils.utils import SquashedNormal
from leo_gym.rl_algorithms.h_ppo.actor_critic_nets import (
    PolicyNetwork,
    ValueNetwork,
    ObservationEncoder,
)


# ===== Satellite configuration =====
satellite_params = {
    "dt":   60,
    "days": 1,
    "ideal_traj_params": {
        "flag_rtn_thrust": True,
        "flag_mass_loss": True,
        "flag_pert_moon": False,
        "flag_pert_sun": False,
        "flag_pert_srp": False,
        "flag_pert_drag": False,
        "flag_pert_irr_grav": False,
        "eph_time_0": 6.338304141847866e08,
        "m": 150,
        "f_max": 18e-3,
        "Isp": 860,
        "Ad": 1.3,
        "Cd": 2.2,
        "Cr": 1.3,
        "As": 1.3,
        "mf": 136,
    },
    "pert_traj_params": {
        "flag_rtn_thrust": True,
        "flag_mass_loss": True,
        "flag_pert_moon": False,
        "flag_pert_sun": False,
        "flag_pert_srp": False,
        "flag_pert_drag": False,
        "flag_pert_irr_grav": False,
        "eph_time_0": 6.338304141847866e08,
        "m": 150,
        "f_max": 18e-3,
        "Isp": 860,
        "Ad": 1.3,
        "Cd": 2.2,
        "Cr": 1.3,
        "As": 1.3,
        "mf": 136,
    },
    "rv0":np.array([-475449.609559833, 277131.557120858, 7553438.90796208,
                    -7238.56711050838, -16.6455796498367, -454.213727291452])}

sat_cfg = SatelliteROEConfig(**satellite_params)

# ===== Environment configuration =====
env_cfg = RecfgEnvConfig(
    low_action=[9, 1],
    high_action=[55, 41],
    satellite_config=sat_cfg,
    satellite_observation_feature_size=6,
    continuous_actions_size=2,
    discrete_actions_size=7,  # R/T/N with +/- and coast
    f_max=sat_cfg.pert_traj_params.f_max,
    max_time_index=int(12 * sat_cfg.days * 60 * 60 / sat_cfg.dt), # 12 hour episodes
    Droe_ranges=[[0,0],
                [0,0],
                [0.0,0.0],
                [0.0,0.0], 
                [-800, +800],
                [-800,+800]],
    target_roe=None,
    target_tolerance=25.0,
    success_reward=10.0,
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
    batch_size=200,
    target_kl=0.05,
    lr_decay_coef=0,
    epochs=5,
    n_envs=10,
    normalize_advantage=True,
    init_std=[0.4, 0.4],
    log_to_mlflow=True,
    device="cpu",
    activation_fun=SquashedNormal,
    encoder_hidden_size=256,
    policy_wrapper=PolicyNetwork,
    critic_wrapper=ValueNetwork,
    observation_encoder=ObservationEncoder,
    default_num_envs=10, # Number of parallel environments (default = 100)
    steps_per_env=20,
    save_nets_period=int(100_000),
    max_training_timesteps=int(50_000),
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
    run_name="test",
)
