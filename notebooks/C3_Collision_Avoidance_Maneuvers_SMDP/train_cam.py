"""
Script used to train cam policy:

+ Uses async parallel environments for faster training
"""

# Standard library
import json
import os
import random
import sys
from typing import List

# Third-party
import gymnasium as gym
import mlflow
import numpy as np
import torch as T
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from pydantic import Field
from tqdm import tqdm
from dataclasses import replace

# Local
from leo_gym.rl_algorithms.h_ppo.config import PPOConfig
from leo_gym.rl_algorithms.h_ppo.h_ppo_agent import Agent
from leo_gym.gyms.cam_gym import CamEnv, CamEnvConfig
from leo_gym.utils.utils import seed_all
from libs.leo_gym.notebooks.C3_Collision_Avoidance_Maneuvers_SMDP.train_cam_hppo_cfg import training_cfg, env_cfg, ppo_cfg

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Seed run 
SEED = 10
seed_all(seed = SEED)

enam = "cam_squashed_gaussian"
enam = "/home/montezuma/Desktop/my_experiment"


training_cfg = training_cfg.model_copy(
    update={
        "experiment_name": enam,
        "seed": SEED,
        "tracking_uri": "/home/nektaf/mlruns"
    }
)

# Make environment 
def make_env(env_cfg: CamEnvConfig, 
             seeds: List[int], 
             idx: int
             )->CamEnv:
    def _init():
        seed = seeds[idx]
        return CamEnv(cfg=env_cfg, seed=seed)
    return _init



if __name__ == "__main__":
    # Prepare vectorized environments
    SEEDS = [random.randint(0, 2**32 - 1) for _ in range(training_cfg.default_num_envs)]
    print("Seeds: ",SEEDS)
    
    env = AsyncVectorEnv([make_env(env_cfg, SEEDS, i) for i in range(training_cfg.default_num_envs)])
    
    if training_cfg.trained_algorithm_config_path is not None:
        with open(training_cfg.trained_algorithm_config_path, "r") as f:
            cfg_kwargs = json.load(f)

        ppo = Agent(
            env_obs=env.single_observation_space,
            env_actions=env.single_action_space,
            cfg=PPOConfig(**cfg_kwargs),
        )
        
    else:
        
        # Update PPO config with environment spaces
        ppo_cfg = ppo_cfg.model_copy(update={
            "env_obs":    env.single_observation_space,
            "env_actions": env.single_action_space,
        })

        ppo = Agent(
            env_obs=env.single_observation_space,
            env_actions=env.single_action_space,
            cfg=ppo_cfg,
        )
        
        ppo.train(env, training_cfg, env_cfg, ppo_cfg)


        # # ===== Save final models =====
        # directory_save = os.path.join(
        #     training_cfg.tracking_uri,
        #     experiment_id,
        #     run_id,
        #     "models",
        #     "final"
        # )
        
        # os.makedirs(directory_save, exist_ok=True)
        # ppo.save_models(directory_save=directory_save)
