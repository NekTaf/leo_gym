""" ===== Monte Carlo CAM Data Collection Script =====
Monte Carlo data collections using multiprocessing to speed up simulations
Objectives:
    - Record Collision Avoidance Maneuver (CAM) success ratio
    - Collect data for SHAP validation (explaining model outputs based on inputs)
"""
# Standard library
import os
import random
import time

# Third-party
import gymnasium as gym
import numpy as np
import pandas as pd
import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

# Local
from leo_gym.rl_algorithms.h_ppo.h_ppo_agent import Agent
from leo_gym.utils.utils import create_dir, seed_all
from leo_gym.gyms.cam_gym import CamEnv, CamEnvConfig
from pathlib import Path
from leo_gym.rl_algorithms.h_ppo.config import PPOConfig
import json, importlib, re
import argparse
from pathlib import Path
from train_cam_hppo_cfg import env_cfg

# Set global seed for Monte-Carlo Runs
seed_all(seed=10)

parser = argparse.ArgumentParser()
parser.add_argument("--exp_path", required=True, help="Path to the MLflow run folder")
parser.add_argument("--policy_timestep", required=True, help="Path to the MLflow run folder")

args = parser.parse_args()

experiment_path = Path(args.exp_path)
policy_timestep = Path(args.policy_timestep)
policy_file_path = experiment_path / "artifacts/models" / policy_timestep / "policynet.pth"
critic_file_path = experiment_path / "artifacts/models" / policy_timestep / "valuenet.pth"    
    
ppo_cfg = experiment_path / "artifacts/ppo_cfg.json"

#Make Environment 
def make_env():
    seed = np.random.randint(0, 2**32)
    return CamEnv(env_cfg, seed)

# Prepare Agent 
env = make_env()

ppo = Agent(
    env_obs=env.observation_space,
    env_actions=env.action_space,
    ppo_cfg=ppo_cfg,
    train=False,
    policy_file_path=policy_file_path,
    critic_file_path=critic_file_path,
    device='cpu'
)

# Distributed Process Initializer 
def init_process(rank, size, fn, data_dir, seed, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.2'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, data_dir, seed)

# Monte Carlo Simulation Runner 
def run(rank, size, data_dir, seed):
    print(f"Process {rank} out of {size} started........")
    env = CamEnv(env_cfg, seed=seed)

    stats = []
    observations=[]
    runs = 10

    for i in range(runs):
        print(f"Rank:{rank}, completed: {(i/runs)*100:.1f}%")
        obs, _ = env.reset()
        os.chdir(data_dir)

        while True:
            action_dis, action_cont, *_ = ppo.choose_action(state=obs, deterministic=True)
            action = {"discrete": action_dis, "continuous": action_cont}
            
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)

            if terminated or truncated:
                radial = [c[0] for c in env.DebrisSwarm_1.controls_RTN]
                P_max = [x for x in env.DebrisSwarm_1.p_max_predictions if x != 0]
                stats.append({
                    'p_max_start': P_max[0],
                    'p_max_end': P_max[-1],
                    "covariance": env.inv_sqrt_det ,
                    "collision_radius": env.DebrisSwarm_1.radius_combined ,
                    'reward': reward,
                    'terminated':terminated,
                    'truncated': truncated,
                    "num_debris":env.DebrisSwarm_1.num_debris,
                    "Delta_v":  (sum(abs(a) for a in radial) * env.DebrisSwarm_1.cfg.dt) / env.DebrisSwarm_1.dynamics_ideal.m,
                })
                break
            
    df = pd.DataFrame(stats)
    df.to_csv('mc_cam_results.csv', index=False, mode='a', header=False)
    os.chdir(data_dir)

    with open('background_data_states.csv', 'ab') as f:
        np.savetxt(f, np.array(observations), delimiter=',')

    print(f"Process {rank} finished")
    
    return 0


if __name__ == '__main__':
    start_time = time.time()

    usages = psutil.cpu_percent(percpu=True, interval=1)
    free_cpus = sum(1 for u in usages)

    size = 100
    seeds = [random.randint(0, 2**32 - 1) for _ in range(size)]
    
    try:
        os.mkdir(experiment_path / "artifacts/models" / policy_timestep)
    except FileExistsError:
        pass
    
    os.chdir(experiment_path / "artifacts/models" / policy_timestep)
    data_dir = create_dir("monte_carlo_cam")
    data_dir = os.path.abspath(data_dir)
    os.chdir(data_dir)

            
    # Data file headers
    headers = ['p_max_start', 
               'p_max_end', 
               'covariance',
               'collision_radius', 
               'reward',
               'terminated',
               'truncated',
               'num_debris',
               'Delta_v']
    
    pd.DataFrame([headers]).to_csv('mc_cam_results.csv', index=False, header=False)
    mp.set_start_method("spawn", force=True)
    processes = []

    #  Launch Process 
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run, data_dir, seeds[rank]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")
