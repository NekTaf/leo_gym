"""
Script used to train recfguration agent using H-PPO algorithm.

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
from leo_gym.gyms.recfg_gym import RecfgEnv, RecfgEnvConfig
from leo_gym.utils.utils import seed_all
from train_recfg_hppo_cfg import training_cfg, env_cfg, ppo_cfg
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)


# Make environment 
def make_env(env_cfg: RecfgEnvConfig, 
             seeds: List[int], 
             idx: int
             )->RecfgEnv:
    def _init():
        seed = seeds[idx]
        return RecfgEnv(cfg=env_cfg, seed=seed)
    return _init



if __name__ == "__main__":
    
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=False, default=None)
    p.add_argument("--seed", required=False, default=0)

    args = p.parse_args()
    
    # Seed run 
    SEED = int(args.seed)
    seed_all(seed = SEED)

    
    training_cfg = training_cfg.model_copy(
    update={
        "seed": SEED,
        "run_name":args.run_name
    }
)


    # Prepare vectorized environments
    SEEDS = [random.randint(0, 2**32 - 1) for _ in range(ppo_cfg.default_num_envs)]
    print("Seeds: ",SEEDS)
    
    env = AsyncVectorEnv([make_env(env_cfg, SEEDS, i) for i in range(ppo_cfg.default_num_envs)])
    
    if ppo_cfg.trained_algorithm_config_path is not None:
        with open(ppo_cfg.trained_algorithm_config_path, "r") as f:
            cfg_kwargs = json.load(f)

        ppo = Agent(
            env_obs=env.single_observation_space,
            env_actions=env.single_action_space,
            ppo_cfg=PPOConfig(**cfg_kwargs),
            env_cfg=env_cfg,
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
            ppo_cfg=ppo_cfg,
            env_cfg=env_cfg,
        )
        
        ppo.train(env, training_cfg)


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
    env.close()