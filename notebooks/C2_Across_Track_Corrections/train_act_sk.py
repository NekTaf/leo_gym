import numpy as np
from leo_gym.rl_algorithms.ppo_sb3_smdp.ppo_sb3_smdp import *
from leo_gym.gyms.roe_sk_gym import RoeGym, RoeGymConfig, SatelliteROEConfig
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from leo_gym.rl_algorithms.ppo_sb3_smdp.ppo_sb3_smdp import SMDP_PPO
import random
import mlflow

from libs.leo_gym.notebooks.C2_Across_Track_Corrections.train_act_sk_cfg import env_cfg, sat_cfg, ppo_cfg, ppo_smdp_cfg, NUM_ENV, SEED
from leo_gym.utils.utils import seed_all
from dataclasses import asdict, dataclass, replace
from leo_gym.utils.sb3_mlflow_loggers import CustomCheckpoint, loggers
from urllib.parse import urlparse


def make_env(config:RoeGymConfig, 
             seeds: List[int],
             idx:int
             )->RoeGym:
    def _init():
        seed = seeds[idx]
        env = RoeGym(cfg=config, seed=seed)
        return env
    return _init

mlflow.set_experiment("PPO ACT SK")

if __name__ == '__main__':
    
    with mlflow.start_run():
        
        seed_all(seed=SEED)
        SEEDS = [random.randint(0, 2**32 - 1) for _ in range(NUM_ENV)]
        print(SEEDS)
        
        mlflow.log_params({"env_number":NUM_ENV,
                           "global_seed":SEED})

        env_fns = [make_env(env_cfg,SEEDS,i) for i in range(NUM_ENV)]
        vec_env = SubprocVecEnv(env_fns) 
        vec_env = VecMonitor(vec_env) 

        model = PPO(
            "MlpPolicy",
            vec_env, **asdict(ppo_cfg))
        
        # SAC Algorithm alternative 
        
        # model = SAC(policy="MlpPolicy",
        #             env=vec_env,
        #             verbose=1,
        #             learning_rate=0.003,
        #             batch_size=1024,
        #             buffer_size=1_000_000)
        
        
        model.set_logger(loggers)
        ckpt_dir = urlparse(mlflow.get_artifact_uri("models")).path

        save_policy_callback = CustomCheckpoint(ckpt_dir,
                                                name_prefix="ppo_act_sk", 
                                                save_freq=100_000)
        
        try:
            model.learn(total_timesteps=5_000_000, 
                        progress_bar=True,
                        callback=save_policy_callback)
            
        except KeyboardInterrupt:
            model.save("ppo_mdp_act_sk")
            mlflow.log_artifact("ppo_mdp_act_sk.zip", artifact_path="models")
