# Standard library
from typing import Any, Optional, Tuple
from dataclasses import asdict

# Third-party
import gymnasium as gym
import numpy as np
import torch as T
import torch.nn as nn
import mlflow
import os
from numpy.typing import NDArray

from tqdm import tqdm
# Local
from leo_gym.rl_algorithms.h_ppo.actor_critic_nets import (
    PolicyNetwork,
    ValueNetwork,
)
from leo_gym.rl_algorithms.h_ppo.buffer import TrajectoryBuffer
from leo_gym.rl_algorithms.h_ppo.config import PPOConfig, TrainingConfig
from leo_gym.rl_algorithms.h_ppo.logger import MLflowLogger
from leo_gym.rl_algorithms.h_ppo.losses import (
    ActorLoss,
    CriticLoss,
)
from textwrap import dedent
from pydantic import BaseModel
from pathlib import Path

class Agent:

    def __init__(
        self, 
        env_obs, 
        env_actions, 
        #  env_cfg,
        ppo_cfg: PPOConfig | str | Path,
        policy_file_path:str|Path|None=None,
        critic_file_path:str|Path|None=None,
        train:bool=False,
        device:str|None = None,
        ):
        
        self.load_cfg(ppo_cfg)        
        
        if device is not None:
            self.device = device
        else:
            self.device = T.device(self.ppo_cfg.device)

        # Whether to normalize advantage 
        self.normalize_advantage = getattr(self.ppo_cfg, "normalize_advantage", False)

        # Entropy coefficient (learnable) 
        init_log_ent = T.log(T.tensor(self.ppo_cfg.init_entropy_coef))

        # Logger 
        if self.ppo_cfg.log_to_mlflow:
            self.logger = MLflowLogger()
        else:
            self.logger = None

        action_type = []
        # default continuous action size if no continuous actions are present
        self.cont_act_size = 0
        for name, space in env_actions.spaces.items():
            if isinstance(space, gym.spaces.Box):
                action_type.append(["continuous", space.shape[0]])
                self.cont_act_size = space.shape[0]
                # Store continuous bounds
                self.ha = T.tensor(space.high, dtype=T.float32)
                self.la = T.tensor(space.low, dtype=T.float32)
            elif isinstance(space, gym.spaces.Discrete):
                action_type.append(["discrete", space.n])
                        
        # move entropy coef to device
        self.log_ent_coef = nn.Parameter(init_log_ent.to(self.device))
    
        self.policy_net = self.ppo_cfg.policy_wrapper(
            state_dim=env_obs.shape[0],
            lr=3e-4,
            action_type=action_type,
            device=self.device,
            continuous_dist_cls=self.ppo_cfg.continuous_dist_cls,
            std_0=0.5,
            net_arch=self.ppo_cfg.net_arch,
            observation_encoder=self.ppo_cfg.observation_encoder,
        )

        self.value_net = self.ppo_cfg.critic_wrapper(
            state_dim=env_obs.shape[0],
            device=self.device,
            lr=self.ppo_cfg.lr,
            net_arch=self.ppo_cfg.net_arch,
            observation_encoder=self.ppo_cfg.observation_encoder
        )

        for g in self.policy_net.optimizer.param_groups:
            g["lr"] = self.ppo_cfg.lr
        self.policy_net.optimizer.add_param_group(
            {
                "params": [self.log_ent_coef],
                "lr": self.ppo_cfg.lr,
            }
        )
        
        self.load_trained_networks(
            train=train,
            policy_file_path=policy_file_path,
            critic_file_path=critic_file_path
        )

        # Rollout Buffer
        self.buffer = TrajectoryBuffer(batch_size=self.ppo_cfg.batch_size)

        # Loss Components
        self.actor_loss_fn = ActorLoss(
            policy_net=self.policy_net,
            policy_clip=self.ppo_cfg.policy_clip,
            log_ent_coef=self.log_ent_coef,
            target_kl=self.ppo_cfg.target_kl
        )
        
        self.value_loss_fn = CriticLoss(
            value_net=self.value_net,policy_clip=self.ppo_cfg.policy_clip)

        self.loss_components = [self.actor_loss_fn, self.value_loss_fn]


    def choose_action(self, 
                      state: Any,
                      deterministic:Optional[bool]=False
                      )-> Tuple[NDArray,NDArray,NDArray,NDArray,NDArray]:
        """
        Get discrete and continuous action with probabilities and value function output 
        
        :param state: policy observation
        :param deterministic: wether the policy should sample actions or not
        
        :return: (action_dis, action_cont, logp_dis, logp_cont, value)
        """
        dist_dis, dist_cont = self.policy_net(state)
        value = T.squeeze(self.value_net(state)).detach().cpu().numpy()

        # Discrete actions
        if deterministic:
            action_dis = T.argmax(dist_dis.probs, dim=-1)
        else:
            action_dis = dist_dis.sample()
        logp_dis = T.squeeze(dist_dis.log_prob(action_dis)).detach().cpu().numpy()
        action_dis = T.squeeze(action_dis).detach().cpu().numpy()

        # Continuous actions        
        if deterministic:
            action_cont = dist_cont.mean
        else:
            action_cont = dist_cont.sample()
        logp_cont = dist_cont.log_prob(action_cont).sum(dim=-1)
        logp_cont = T.squeeze(logp_cont).detach().cpu().numpy()

        action_cont = T.squeeze(action_cont).detach().cpu().numpy()

        return action_dis, action_cont, logp_dis, logp_cont, value


    def rollout_buffer(
        self,
        state,
        action_dis,
        action_cont,
        logp_dis,
        logp_cont,
        value,
        reward,
        done,
        sojourn_t,
    ):
        
        step_data = {
            "obs": state,
            "act_dis": action_dis,
            "act_cont": action_cont,
            "old_logp_dis": logp_dis,
            "old_logp_cont": logp_cont,
            "values": value,
            "rewards": reward,
            "dones": done,
            "sojourn_t": sojourn_t,
        }
        self.buffer.push(step_data)


    def GAE_fun(self,
                reward_arr,
                vals_arr,
                dones_arr,
                t_soujourn_arr,        
                gae_lambda=None,
                gamma=None
                )->T.Tensor:

        # Custom gamma-unused
        if gae_lambda is None:
            gae_lambda = self.ppo_cfg.gae_lambda
        if gamma is None:
            gamma = self.ppo_cfg.gamma
        
        gamma_arr = gamma ** t_soujourn_arr

        T_steps = len(reward_arr)
        n_envs = self.ppo_cfg.n_envs
        advantage = np.zeros((T_steps, n_envs), dtype=np.float32)

        # standard reversed GAE computation, handle per-env dones
        last_adv = np.zeros(n_envs, dtype=np.float32)
        for t in reversed(range(T_steps)):
            mask = 1.0 - dones_arr[t].astype(np.float32)

            # next value: use 0 for bootstrap at final step (if episode done)
            if t + 1 < T_steps:
                next_values = vals_arr[t + 1]
            else:
                next_values = np.zeros(n_envs, dtype=np.float32)

            delta = (
                reward_arr[t]
                + gamma_arr[t] * next_values * mask
                - vals_arr[t]
            )

            last_adv = delta + gamma_arr[t] * gae_lambda * mask * last_adv
            advantage[t] = last_adv

        return T.tensor(advantage).to(self.device)


    def update(self, 
              timesteps_so_far:int, 
              total_timesteps:int
              )->None:
        """
        Single PPO update: generate batches, compute losses, update networks.
        """
        
        plot_policy_loss_dis, plot_policy_loss_cont, plot_value_loss, plot_policy_loss_total = [], [], [], []
        plot_entr_cont, plot_kl_cont, = [], []
        plot_entr_dis, plot_kl_dis, plot_clipped_probs_dis = [], [], []


        frac = timesteps_so_far / total_timesteps if total_timesteps is not None else 0
        new_lr = self.lr * self.ppo_cfg.lr_decay_coef * (1 - frac)
        new_lr = max(new_lr, 3e-4)
        self.lr = new_lr

        for g in self.policy_net.optimizer.param_groups:
            g["lr"] = self.lr
        for g in self.value_net.optimizer.param_groups:
            g["lr"] = self.lr

        if self.logger:
            self.logger.log_scalar("lr", self.lr, timesteps_so_far)


        for _ in range(self.ppo_cfg.epochs):

            all_data, batches = self.buffer.generate_batches()
            
            obs_arr = all_data["obs"]   
            act_dis_arr = all_data["act_dis"] 
            act_cont_arr = all_data["act_cont"]
            old_logp_dis_arr = all_data["old_logp_dis"]
            old_logp_cont_arr = all_data["old_logp_cont"]
            values_arr = all_data["values"]      
            rewards_arr = all_data["rewards"]      
            dones_arr = all_data["dones"]       
            sojourn_t_arr = all_data["sojourn_t"]   

            advantage = self.GAE_fun(rewards_arr, values_arr, dones_arr, sojourn_t_arr)
            values = T.tensor(values_arr).to(self.device)

            for batch_idxs in batches:
                
                states_batch = T.tensor(obs_arr[batch_idxs], dtype=T.float32).to(self.device)

                batch_data = {
                    "obs":states_batch,
                    "act_dis": None,
                    "act_cont": None,
                    "old_logp_dis": None,
                    "old_logp_cont": None,
                    "adv_rew": None,
                    "values_old": None,
                    "returns": None,
                }                
                                
                # Advantage Normalization                
                adv = advantage[batch_idxs]
                if self.normalize_advantage: 
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                action_advantage = adv

                batch_data["adv_rew"] = action_advantage

                # Discrete actions
                if act_dis_arr is not None:
                    ad = T.tensor(act_dis_arr[batch_idxs], dtype=T.long).to(self.device)
                    oldpd = T.tensor(old_logp_dis_arr[batch_idxs], dtype=T.float32).to(self.device)

                    batch_data["act_dis"] = ad
                    batch_data["old_logp_dis"] = oldpd

                # Continuous actions
                if act_cont_arr is not None:
                    ac = T.tensor(act_cont_arr[batch_idxs], dtype=T.float32).to(self.device)
                    oldpc = T.tensor(old_logp_cont_arr[batch_idxs], dtype=T.float32).to(self.device)

                    batch_data["act_cont"] = ac
                    batch_data["old_logp_cont"] = oldpc

                # Returns
                returns = advantage[batch_idxs] + values[batch_idxs]
                
                batch_data["values_old"] = values[batch_idxs]
                batch_data["returns"] = returns

                # Compute total_loss
                total_loss = T.tensor(0.0, dtype=T.float32, device=self.device)                
                actor_loss, actor_plot_data = self.actor_loss_fn.compute_loss(batch_data)
                value_loss, value_plot_data = self.value_loss_fn.compute_loss(batch_data)    
                total_loss = actor_loss + value_loss 

                # Zero optimizers
                self.policy_net.optimizer.zero_grad()
                self.value_net.optimizer.zero_grad()
          
                total_loss.backward()

                # Gradient clipping
                T.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
                T.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)

                # Optimizer step
                self.policy_net.optimizer.step()
                self.value_net.optimizer.step()                
                
                
                # Plot data 
                plot_policy_loss_dis.append(actor_plot_data['actor_loss_dis'])
                plot_entr_dis.append(actor_plot_data['entropy_bonus_dis'])
                plot_kl_dis.append(actor_plot_data['approx_kl_dis'])
                plot_policy_loss_cont.append(actor_plot_data['actor_loss_cont'])
                plot_entr_cont.append(actor_plot_data['entropy_bonus_cont'])
                plot_kl_cont.append(actor_plot_data['approx_kl_cont'])
                plot_policy_loss_total.append(actor_plot_data['total_actor_loss'])
                plot_value_loss.append(value_plot_data['critic_loss'])

        if self.logger:
            self.logger.log_scalar("train/actor_loss_dis", float(np.mean(plot_policy_loss_dis)), timesteps_so_far)
            self.logger.log_scalar("train/actor_loss_cont", float(np.mean(plot_policy_loss_cont)), timesteps_so_far)
            self.logger.log_scalar("train/total_actor_loss", float(np.mean(plot_policy_loss_total)), timesteps_so_far)
            self.logger.log_scalar("train/critic_loss", float(np.mean(plot_value_loss)), timesteps_so_far)
            self.logger.log_scalar("train/entropy_loss x entropy_coef dis", float(np.mean(plot_entr_dis)), timesteps_so_far)
            self.logger.log_scalar("train/approx_kl dis", float(np.mean(plot_kl_dis)), timesteps_so_far)
            self.logger.log_scalar("train/entropy_loss x entropy_coef cont", float(np.mean(plot_entr_cont)), timesteps_so_far)
            self.logger.log_scalar("train/approx_kl cont", float(np.mean(plot_kl_cont)), timesteps_so_far)
            self.logger.log_scalar("train/entropy_coef", float(T.exp(self.log_ent_coef).item()), timesteps_so_far)



        self.buffer.clear()

        return
    
        
    def save_models(self, directory_save):
        self.policy_net.save_checkpoint(directory_save)
        self.value_net.save_checkpoint(directory_save)


    def load_trained_networks(
        self, 
        train:bool=False,
        policy_file_path:str|Path|None=None,
        critic_file_path:str|Path|None=None,
        )->None:
        

        if policy_file_path is not None:
            self.policy_net.load_checkpoint(policy_file_path,train,self.device)

        elif critic_file_path is not None:
            self.value_net.load_checkpoint(critic_file_path,train,self.device)
        
        return
        
    def load_cfg(
        self, 
        ppo_cfg: PPOConfig|Path|str, 
        )->None:
        
        try:
            self.lr = ppo_cfg.lr
            # self.env_cfg = env_cfg
            self.ppo_cfg = ppo_cfg
        except AttributeError:
            self.ppo_cfg = PPOConfig.load_cfg_params_from_file(cfg_file_path=ppo_cfg)

        return 
        
                    
        
    def train(self,env:gym.Env, training_cfg:TrainingConfig, env_cfg:BaseModel)->None:
        
        try:
            # Experiment and run setup
            os.makedirs(training_cfg.tracking_uri, exist_ok=True)
            mlflow.set_tracking_uri(training_cfg.tracking_uri)
            mlflow.set_experiment(training_cfg.experiment_name)

            
            with mlflow.start_run(run_name=training_cfg.run_name) as run:
                
                mlflow.log_dict(
                    env_cfg.model_dump(), 
                    "env_cfg.json"
                    )
                
                mlflow.log_dict(
                    self.ppo_cfg.model_dump(mode="python"), 
                    "ppo_cfg.json"
                    )

                mlflow.log_dict(
                    training_cfg.model_dump(), 
                    "training_cfg.json"
                    )
                
                experiment_id = run.info.experiment_id
                run_id = run.info.run_id
                
                print(f"Experiment ID: {experiment_id}, Run ID: {run_id}")

                env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=self.ppo_cfg.steps_per_env)

                self.timesteps_so_far = 0
                state, _ = env.reset(seed=None)
                self.global_episode_count = 0

                    
                with tqdm(total=self.ppo_cfg.max_training_timesteps, desc="Training Progress") as pbar:
                    while self.timesteps_so_far <= self.ppo_cfg.max_training_timesteps:

                        action_dis, action_cont, prob_dis, prob_cont, val = self.choose_action(state)
                        
                        action = {"discrete": action_dis,
                                    "continuous": action_cont}

                        # Environment step
                        next_state, reward, terminated, truncated, info = env.step(action)
                        sojourn_t = info["sojourn_t"]

                        # Process done/truncated flags and Save to buffer
                        self.rollout_buffer(
                            state,
                            action_dis,
                            action_cont,
                            prob_dis,
                            prob_cont,
                            val,
                            reward,
                            terminated,
                            sojourn_t
                        )

                        self.timesteps_so_far += self.ppo_cfg.n_envs
                        pbar.update(self.ppo_cfg.n_envs)
                        state = next_state

                        # Update the original environment variable "training_steps" through all vectorized envs (for curriculum training)
                        env.set_attr("training_steps", self.timesteps_so_far)
                        
                        # Call training
                        if self.timesteps_so_far % (self.ppo_cfg.steps_per_env * self.ppo_cfg.n_envs) == 0:
                            self.update(self.timesteps_so_far, total_timesteps=self.ppo_cfg.max_training_timesteps)
                            print(f"""
                                ===========================================
                                Timesteps so far : {self.timesteps_so_far}
                                Average cumulative episodic reward: {np.mean(env.return_queue)}
                                Average episode length: {np.mean(env.length_queue)}
                                =========================================== """)


                        mlflow.log_metric(
                            "rollout/sum_rewards",
                            float(np.mean(env.return_queue)),
                            step=self.timesteps_so_far
                        )
                        mlflow.log_metric(
                            "rollout/episode_timesteps",
                            float(np.mean(env.length_queue)),
                            step=self.timesteps_so_far
                        )

                        if self.timesteps_so_far % (self.ppo_cfg.save_nets_period*self.ppo_cfg.batch_size) == 0:
                            directory_save = os.path.join(
                                training_cfg.tracking_uri,
                                experiment_id,
                                run_id,
                                "artifacts/models",
                                f"{self.timesteps_so_far}"
                            )
                            
                            os.makedirs(directory_save, exist_ok=True)
                            self.save_models(directory_save=directory_save)
                            
                            
        except KeyboardInterrupt:
            # allow graceful interrupt, don't swallow other exceptions
            pass
        
        # Save final models 
        if 'experiment_id' in locals() and 'run_id' in locals():
            directory_save = os.path.join(
                training_cfg.tracking_uri,
                experiment_id,
                run_id,
                "artifacts/models",
                "final"
            )
        else:
            directory_save = os.path.join(
                training_cfg.tracking_uri,
                "artifacts/models",
                "final"
            )
        
        os.makedirs(directory_save, exist_ok=True)
        self.save_models(directory_save=directory_save)


