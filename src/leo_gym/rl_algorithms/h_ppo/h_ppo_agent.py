# Standard library
from typing import Any, Optional

# Third-party
import gymnasium as gym
import numpy as np
import torch as T
import torch.nn as nn
import mlflow
import os

from tqdm import tqdm
# Local
from leo_gym.rl_algorithms.h_ppo.actor_critic_nets import (
    PolicyNetwork,
    ValueNetwork,
)
from leo_gym.rl_algorithms.h_ppo.buffer import TrajectoryBuffer
from leo_gym.rl_algorithms.h_ppo.config import LagrangianConfig, PPOConfig
from leo_gym.rl_algorithms.h_ppo.logger import MLflowLogger
from leo_gym.rl_algorithms.h_ppo.losses import (
    ActorLagrangianLoss,
    ActorLoss,
    CostCriticLoss,
    CriticLoss,
    LagrangianLoss,
)
from textwrap import dedent


class Agent:

    def __init__(self, 
                 env_obs, 
                 env_actions, 
                 cfg: PPOConfig):
        
        self.cfg = cfg
        
        self.lr = cfg.lr

        # Whether to normalize advantage 
        self.normalize_advantage = getattr(cfg, "normalize_advantage", False)

        # Entropy coefficient (learnable) 
        init_log_ent = T.log(T.tensor(cfg.init_entropy_coef))
        self.log_ent_coef = nn.Parameter(init_log_ent)

        # Logger 
        if cfg.log_to_mlflow:
            self.logger = MLflowLogger()
        else:
            self.logger = None

        action_type = []
        for name, space in env_actions.spaces.items():
            if isinstance(space, gym.spaces.Box):
                action_type.append(["continuous", space.shape[0]])
                self.cont_act_size = space.shape[0]
                # Store continuous bounds
                self.ha = T.tensor(space.high, dtype=T.float32)
                self.la = T.tensor(space.low, dtype=T.float32)
            elif isinstance(space, gym.spaces.Discrete):
                action_type.append(["discrete", space.n])

        if isinstance(env_obs, gym.spaces.Dict):
            state_dim = []
            for key, subspace in env_obs.spaces.items():
                state_dim.append((key, subspace.shape))

        self.device = T.device(self.cfg.device)
        
        
        policy_net_wrapper = cfg.policy_wrapper
        value_net_wrapper = cfg.critic_wrapper

        
        self.policy_net = policy_net_wrapper(
            state_dim=state_dim,
            action_type=action_type,
            lr=self.cfg.lr,
            device=self.device,
            std_type=cfg.std_type,
            use_squashed_gaussian=cfg.use_squashed_gaussian,
            cont_act_size=self.cont_act_size,
            std_0=cfg.init_std,
        )

        self.value_net = value_net_wrapper(
            state_dim=state_dim,
            device=self.device,
            lr=self.cfg.lr,
        )

        # Add entropy coefficient to policy optimizer
        for g in self.policy_net.optimizer.param_groups:
            g["lr"] = self.cfg.lr
        self.policy_net.optimizer.add_param_group(
            {
                "params": [self.log_ent_coef],
                "lr": self.cfg.lr,
            }
        )

        # Rollout Buffer
        self.buffer = TrajectoryBuffer(batch_size=self.cfg.batch_size)

        # Loss Components
        self.actor_loss_fn = ActorLoss(
            policy_net=self.policy_net,
            policy_clip=self.cfg.policy_clip,
            log_ent_coef=self.log_ent_coef,
            target_kl=self.cfg.target_kl
        )
        
        self.value_loss_fn = CriticLoss(
            value_net=self.value_net,policy_clip=self.cfg.policy_clip)

        self.loss_components = [self.actor_loss_fn, self.value_loss_fn]


    def _observations_wrapper(self, state: dict):
        ds_np = state["ds"]
        nds_np = state["nds"]
        return ds_np, nds_np


    def choose_action(self, 
                      state: dict,
                      deterministic:Optional[bool]=False):
        """
        Given a state dict, return:
          (action_dis, action_cont, action_cont_scaled, logp_dis, logp_cont, value)
        """
        ds_np, nds_np = self._observations_wrapper(state)
        
        ds_tensor = T.tensor(ds_np, dtype=T.float32).to(self.device)
        nds_tensor = T.tensor(nds_np, dtype=T.float32).to(self.device)

        dist_dis, dist_cont = self.policy_net([ds_tensor, nds_tensor])
        value = T.squeeze(self.value_net([ds_tensor, nds_tensor])).detach().cpu().numpy()

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

        # Normalize and scale action for environment
        act_scale = ((self.ha - self.la) / 2).to(self.device)
        act_bias = ((self.ha + self.la) / 2).to(self.device)
        action_cont_scaled = T.clip(action_cont, -1, 1) * act_scale + act_bias
        action_cont_scaled = T.squeeze(action_cont_scaled).detach().cpu().numpy()
        action_cont = T.squeeze(action_cont).detach().cpu().numpy()

        return action_dis, action_cont, action_cont_scaled, logp_dis, logp_cont, value


    def rollout_buffer(
        self,
        state: dict,
        action_dis,
        action_cont,
        action_cont_scaled,
        logp_dis,
        logp_cont,
        value,
        reward,
        done,
        sojourn_t,
    ):
        ds_np, nds_np = self._observations_wrapper(state)

        step_data = {
            "state_ds": ds_np,
            "state_nds": nds_np,
            "action_dis": action_dis,
            "action_cont": action_cont,
            "old_logp_dis": logp_dis,
            "old_logp_cont": logp_cont,
            "value": value,
            "reward": reward,
            "done": done,
            "sojourn_t": sojourn_t,
        }
        self.buffer.push(step_data)


    def GAE_fun(self,
                reward_arr,
                vals_arr,
                dones_arr,
                t_soujourn_arr,        
                gae_lambda=None,
                gamma=None):

        # Custom gamma-unused
        if gae_lambda is None:
            gae_lambda = self.cfg.gae_lambda
        if gamma is None:
            gamma = self.cfg.gamma
        
        gamma_arr = gamma**t_soujourn_arr
        advantage = np.zeros((len(reward_arr), self.cfg.n_envs), dtype=np.float32)

        for t in range(len(reward_arr) - 1):
            discount = 1.0
            a_t = 0.0
            for k in range(t, len(reward_arr) - 1):
                mask = (1 - dones_arr[k].astype(np.int32))  

                delta = (
                    reward_arr[k]
                    + gamma_arr[k] * vals_arr[k + 1] * mask
                    - vals_arr[k]
                )
                a_t += discount * delta

                discount *= gamma_arr[k] * gae_lambda * mask

            advantage[t] = a_t
                    
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
        new_lr = self.lr * self.cfg.lr_decay_coef * (1 - frac)
        new_lr = max(new_lr, 3e-4)
        self.lr = new_lr

        for g in self.policy_net.optimizer.param_groups:
            g["lr"] = self.lr
        for g in self.value_net.optimizer.param_groups:
            g["lr"] = self.lr

        if self.logger:
            self.logger.log_scalar("lr", self.lr, timesteps_so_far)


        for _ in range(self.cfg.epochs):

            all_data, batches = self.buffer.generate_batches()
            
            state_ds_arr = all_data["state_ds"]   
            state_nds_arr = all_data["state_nds"]   
            action_dis_arr = all_data["action_dis"] 
            action_cont_arr = all_data["action_cont"]
            old_logp_dis_arr = all_data["old_logp_dis"]
            old_logp_cont_arr = all_data["old_logp_cont"]
            vals_arr = all_data["value"]      
            reward_arr = all_data["reward"]      
            done_arr = all_data["done"]       
            sojourn_t_arr = all_data["sojourn_t"]   

            advantage = self.GAE_fun(reward_arr, vals_arr, done_arr, sojourn_t_arr)
            values = T.tensor(vals_arr).to(self.device)

            for batch_idxs in batches:
                
                ds_batch   = T.tensor(state_ds_arr[batch_idxs], dtype=T.float32).to(self.device)
                nds_batch  = T.tensor(state_nds_arr[batch_idxs], dtype=T.float32).to(self.device)

                batch_data = {
                    "states":        (ds_batch, nds_batch),
                    "actions_dis": None,
                    "actions_cont": None,
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
                if action_dis_arr is not None:
                    ad    = T.tensor(action_dis_arr[batch_idxs], dtype=T.long).to(self.device)
                    oldpd = T.tensor(old_logp_dis_arr[batch_idxs], dtype=T.float32).to(self.device)

                    batch_data["actions_dis"] = ad
                    batch_data["old_logp_dis"] = oldpd

                # Continuous actions
                if action_cont_arr is not None:
                    ac    = T.tensor(action_cont_arr[batch_idxs], dtype=T.float32).to(self.device)
                    oldpc = T.tensor(old_logp_cont_arr[batch_idxs], dtype=T.float32).to(self.device)

                    batch_data["actions_cont"] = ac
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
                plot_policy_loss_dis.append(actor_plot_data['actor_loss_dis'].item())
                plot_entr_dis.append(actor_plot_data['entropy_bonus_dis'])
                plot_kl_dis.append(actor_plot_data['approx_kl_dis'])
                plot_policy_loss_cont.append(actor_plot_data['actor_loss_cont'].item())
                plot_entr_cont.append(actor_plot_data['entropy_bonus_cont'])
                plot_kl_cont.append(actor_plot_data['approx_kl_cont'])
                plot_policy_loss_total.append(actor_plot_data['total_actor_loss'].item())
                plot_value_loss.append(value_plot_data['critic_loss'].item())

        self.logger.log_scalar("train/actor_loss_dis", np.mean(plot_policy_loss_dis), timesteps_so_far)
        self.logger.log_scalar("train/actor_loss_cont", np.mean(plot_policy_loss_cont), timesteps_so_far)
        self.logger.log_scalar("train/total_actor_loss", np.mean(plot_policy_loss_total), timesteps_so_far)
        self.logger.log_scalar("train/critic_loss", np.mean(plot_value_loss), timesteps_so_far)
        self.logger.log_scalar("train/entropy_loss x entropy_coef dis", np.mean(plot_entr_dis), timesteps_so_far)
        self.logger.log_scalar("train/approx_kl dis", np.mean(plot_kl_dis), timesteps_so_far)
        self.logger.log_scalar("train/entropy_loss x entropy_coef cont", np.mean(plot_entr_cont), timesteps_so_far)
        self.logger.log_scalar("train/approx_kl cont", np.mean(plot_kl_cont), timesteps_so_far)
        self.logger.log_scalar("train/entropy_coef",T.exp(self.log_ent_coef).item(),timesteps_so_far)



        self.buffer.clear()

        return
    
        
    def save_models(self, directory_save):
        self.policy_net.save_checkpoint(directory_save)
        self.value_net.save_checkpoint(directory_save)


    def load_trained_networks(self, 
                              train:bool,
                              device:T.device,
                              file_name_policy:str,
                              file_name_critic:str,
                              )->None:
        
        self.policy_net.load_checkpoint(file_name_policy,train,device)
        self.value_net.load_checkpoint(file_name_critic,train,device)
        
                    
        
    def train(self,env, training_cfg, env_cfg, ppo_cfg):
        
        try:
            # Experiment and run setup
            os.makedirs(training_cfg.tracking_uri, exist_ok=True)
            mlflow.set_tracking_uri(training_cfg.tracking_uri)
            mlflow.set_experiment(training_cfg.experiment_name)

            
            with mlflow.start_run(run_name=training_cfg.run_name) as run:

                mlflow.log_artifact("/home/nektaf/optacom/src/leo_gym")
                
                mlflow.log_dict(env_cfg.model_dump(), 
                                "env_cfg.json")
                
                mlflow.log_dict(ppo_cfg.model_dump(),
                                "ppo_cfg.json")
                
                mlflow.log_dict(training_cfg.model_dump(), 
                                "training_cfg.json")
                
                mlflow.log_param("max_training_timesteps", 
                                 training_cfg.max_training_timesteps)
                
                mlflow.log_param("update_policy_period", 
                                 training_cfg.steps_per_env)

                experiment_id = run.info.experiment_id
                run_id = run.info.run_id
                
                print(f"Experiment ID: {experiment_id}, Run ID: {run_id}")

                env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=training_cfg.steps_per_env)

                episode_costs = np.zeros(training_cfg.default_num_envs)
                self.timesteps_so_far = 0
                state, _ = env.reset(seed=None)
                self.global_episode_cost_sum = 0.0
                self.global_episode_count = 0

                with tqdm(total=training_cfg.max_training_timesteps, desc="Training Progress") as pbar:
                    while self.timesteps_so_far <= training_cfg.max_training_timesteps:

                        action_dis, action_cont, action_cont_sq_scaled, \
                            prob_dis, prob_cont, val = self.choose_action(state)
                        
                        action = {"discrete": action_dis,
                                    "continuous": action_cont_sq_scaled}

                        # Environment step
                        next_state, reward, terminated, truncated, info = env.step(action)
                        cost = info["cost"]
                        episode_costs += np.array(cost)
                        sojourn_t = info["sojourn_t"]

                        # Save to buffer
                        self.rollout_buffer(
                            state,
                            action_dis,
                            action_cont,
                            action_cont_sq_scaled,
                            prob_dis,
                            prob_cont,
                            val,
                            reward,
                            terminated,
                            sojourn_t
                        )

                        self.timesteps_so_far += training_cfg.default_num_envs
                        pbar.update(training_cfg.default_num_envs)
                        state = next_state
                        
                        # Call training
                        if self.timesteps_so_far % (training_cfg.steps_per_env * training_cfg.default_num_envs) == 0:
                            self.update(self.timesteps_so_far, total_timesteps=training_cfg.max_training_timesteps)
                            print(f"""
                                ===========================================
                                Timesteps so far : {self.timesteps_so_far}
                                Average cumulative episodic reward: {np.mean(env.return_queue)}
                                Average episode length: {np.mean(env.length_queue)}
                                =========================================== """)


                        mlflow.log_metric(
                            "rollout/sum_rewards",
                            np.mean(env.return_queue),
                            step=self.timesteps_so_far
                        )
                        mlflow.log_metric(
                            "rollout/episode_timesteps",
                            np.mean(env.length_queue),
                            step=self.timesteps_so_far
                        )

                        # Environment associated cost if needed for CMDP formulation 
                        for i in range(training_cfg.default_num_envs):
                            if terminated[i] or truncated[i]:
                                self.global_episode_cost_sum += episode_costs[i]
                                self.global_episode_count += 1
                                episode_costs[i] = 0.0

                        if self.global_episode_count > 0:
                            mean_episode_cost = self.global_episode_cost_sum / self.global_episode_count
                            mlflow.log_metric(
                                "rollout/mean_cost_episode", mean_episode_cost, step=self.timesteps_so_far
                            )

                        if self.timesteps_so_far % training_cfg.save_nets_period == 0:
                            directory_save = os.path.join(
                                training_cfg.tracking_uri,
                                experiment_id,
                                run_id,
                                "models",
                                f"{self.timesteps_so_far}"
                            )
                            
                            os.makedirs(directory_save, exist_ok=True)
                            self.save_models(directory_save=directory_save)
                            
                            
        except (KeyboardInterrupt, AttributeError):
            pass
        
        # Save final models 
        directory_save = os.path.join(
            training_cfg.tracking_uri,
            experiment_id,
            run_id,
            "models",
            "final"
        )
        
        os.makedirs(directory_save, exist_ok=True)
        self.save_models(directory_save=directory_save)






class AgentLag(Agent):

    def __init__(self, env_obs, env_actions, ppo_cfg: PPOConfig, lag_cfg: LagrangianConfig):
        super().__init__(env_obs, env_actions, ppo_cfg)

        # Attach Lagrangian config
        self.constr_limit = lag_cfg.constr_limit
        self.lag_lr = lag_cfg.lag_lr
        self.lag_max = lag_cfg.lag_max
        self.flag_lag_update_cont = lag_cfg.flag_lag_update_cont
        self.flag_lag_update_dis = lag_cfg.flag_lag_update_dis

        # Cost‐value network (critic for costs)
        self.cost_value_net = ppo_cfg.critic_wrapper(
            state_dim=[(k, v.shape) for k, v in env_obs.spaces.items()],
            device=self.device,
            lr=self.lr,
        )

        # Lagrange multiplier 
        self.lag_parameter = nn.Parameter(T.ones(1, dtype=T.float32), requires_grad=True)
        self.lag_param_optimizer = T.optim.Adam([self.lag_parameter], lr=self.lag_lr)

        # Replace buffer to expect cost fields too
        self.buffer = TrajectoryBuffer(batch_size=self.cfg.batch_size)

        # Lagrangian loss
        self.lagrange_loss_fn = LagrangianLoss(
            lag_parameter=self.lag_parameter,
            constr_limit=self.constr_limit,
        )

        # Actor loss
        self.actor_loss_fn = ActorLagrangianLoss(
            policy_net=self.policy_net,
            policy_clip=self.cfg.policy_clip,
            log_ent_coef=self.log_ent_coef,
            lag_parameter=self.lag_parameter,
            flag_lag_update_dis=self.flag_lag_update_dis,
            flag_lag_update_cont=self.flag_lag_update_cont,
            target_kl=self.cfg.target_kl
        )

        # critic loss
        self.value_loss_fn = CriticLoss(
            value_net=self.value_net,
            policy_clip=self.cfg.policy_clip)

        # Cost critic loss
        self.cost_loss_fn = CostCriticLoss(
            cost_value_net=self.cost_value_net,
        )

    def choose_action(self, state, deterministic=False):
        base_out = super().choose_action(state, deterministic)
        ds, nds = self._observations_wrapper(state)
        cost_val = T.squeeze(self.cost_value_net([ds, nds])).detach().cpu().numpy()
        return (*base_out, cost_val)


    def rollout_buffer(self, *step_args):
        super().rollout_buffer(*step_args)
        state_ds, state_nds, ad, ac, acs, lp_d, lp_c, val, rew, done, sojourn_t, cost, cost_val = step_args
        self.buffer.push({
          "cost": cost,
          "cost_value": cost_val,
        })


    def compute_episode_costs(self, cost_arr, done_arr):
        """
        Convert a (T, n_envs) array of incremental costs and a corresponding
        (T, n_envs) boolean done mask into a 1D tensor of per‐environment
        average episode cost.
        """
        T_steps, n_envs = done_arr.shape
        episode_costs = [[] for _ in range(n_envs)]
        current_costs = np.zeros(n_envs, dtype=np.float32)

        for t in range(T_steps):
            current_costs += cost_arr[t]
            for env_i in range(n_envs):
                if done_arr[t, env_i]:
                    episode_costs[env_i].append(current_costs[env_i])
                    current_costs[env_i] = 0.0

        # If some env never sent done at last step, record partial cost
        for env_i in range(n_envs):
            if current_costs[env_i] != 0.0:
                episode_costs[env_i].append(current_costs[env_i])

        mean_episode_costs = [
            np.mean(env_cost_list) if env_cost_list else 0.0
            for env_cost_list in episode_costs
        ]
        return T.tensor(mean_episode_costs, dtype=T.float32, device=self.device)


    def update(self, timesteps_so_far=None, total_timesteps=None):
        
        plot_policy_loss_dis, plot_policy_loss_cont, plot_value_loss, plot_policy_loss_total = [], [], [], []
        plot_entr_cont, plot_kl_cont, plot_lag_param = [], [], []
        plot_entr_dis, plot_kl_dis, plot_clipped_probs_dis = [], [], []
        plot_cost_loss = []

        
        frac = timesteps_so_far / total_timesteps if total_timesteps is not None else 0
        new_lr = self.lr * self.cfg.lr_decay_coef * (1 - frac)
        new_lr = max(new_lr, 3e-4)
        self.lr = new_lr

        for g in self.policy_net.optimizer.param_groups:
            g["lr"] = self.lr
        for g in self.value_net.optimizer.param_groups:
            g["lr"] = self.lr
        for g in self.cost_value_net.optimizer.param_groups:
            g["lr"] = self.lr

        if self.logger:
            self.logger.log_scalar("lr", self.lr, timesteps_so_far)


        for _ in range(self.cfg.epochs):
            all_data, batches = self.buffer.generate_batches()

            state_ds_arr = all_data["state_ds"]     
            state_nds_arr = all_data["state_nds"]     
            action_dis_arr = all_data["action_dis"]    
            action_cont_arr = all_data["action_cont"]   
            old_logp_dis_arr = all_data["old_logp_dis"]  
            old_logp_cont_arr = all_data["old_logp_cont"]
            vals_arr = all_data["value"]         
            reward_arr = all_data["reward"]        
            done_arr = all_data["done"]          
            sojourn_t_arr = all_data["sojourn_t"]          

            cost_arr = all_data["cost"]          
            cost_val_arr = all_data["cost_value"]   

            # Update lagrangian param
            episode_costs = self.compute_episode_costs(cost_arr, done_arr)
            lag_loss, _ = self.lagrange_loss_fn.compute_loss({"episode_costs": episode_costs})
            self.lag_param_optimizer.zero_grad()
            lag_loss.backward()
            self.lag_param_optimizer.step()
            
            with T.no_grad():
                self.lag_parameter.data.clamp_(0.0, self.lag_max)

            adv_reward_tensor = self.GAE_fun(reward_arr, vals_arr, done_arr, sojourn_t_arr)                
            values_tensor = T.tensor(vals_arr, 
                                     dtype=T.float32, device=self.device)
            
            adv_cost_tensor = self.GAE_fun(cost_arr, cost_val_arr, done_arr, sojourn_t_arr)
            costval_tensor = T.tensor(cost_val_arr, 
                                      dtype=T.float32, device=self.device)

            for batch_idxs in batches:
                ds_batch  = T.tensor(state_ds_arr[batch_idxs], 
                                     dtype=T.float32).to(self.device)
                nds_batch = T.tensor(state_nds_arr[batch_idxs], 
                                     dtype=T.float32).to(self.device)

                batch_data = {
                    "states": (ds_batch, nds_batch),
                    "actions_dis": None,
                    "actions_cont": None,
                    "old_logp_dis": None,
                    "old_logp_cont": None,
                    "adv_rew": None,
                    "adv_cost": None,
                    "values_old": None,
                    "returns": None,
                    "cost_returns": None,
                    "cost_values_old": None,
                    "episode_costs": episode_costs,
                }

                # Advantage Normalization (per‐batch)
                adv_r = adv_reward_tensor[batch_idxs]
                if self.normalize_advantage:
                    adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
                adv_c = adv_cost_tensor[batch_idxs]
                if self.normalize_advantage:
                    adv_c = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-8)

                batch_data["adv_rew"]  = adv_r
                batch_data["adv_cost"] = adv_c

                # Discrete actions
                if action_dis_arr is not None:
                    ad    = T.tensor(action_dis_arr[batch_idxs], dtype=T.long).to(self.device)
                    oldpd = T.tensor(old_logp_dis_arr[batch_idxs], dtype=T.float32).to(self.device)
                    batch_data["actions_dis"]  = ad
                    batch_data["old_logp_dis"] = oldpd

                # Continuous actions
                if action_cont_arr is not None:
                    ac    = T.tensor(action_cont_arr[batch_idxs], dtype=T.float32).to(self.device)
                    if ac.ndim == 1:
                        ac = ac.unsqueeze(-1)
                        
                    oldpc = T.tensor(old_logp_cont_arr[batch_idxs], dtype=T.float32).to(self.device)
                    batch_data["actions_cont"]  = ac
                    batch_data["old_logp_cont"] = oldpc

                # Returns & Old Values
                batch_data["values_old"] = T.tensor(vals_arr[batch_idxs], dtype=T.float32).to(self.device)
                returns = adv_reward_tensor[batch_idxs] + values_tensor[batch_idxs]
                batch_data["returns"] = returns

                batch_data["cost_values_old"] = T.tensor(cost_val_arr[batch_idxs], dtype=T.float32).to(self.device)
                cost_returns = adv_cost_tensor[batch_idxs] + costval_tensor[batch_idxs]
                batch_data["cost_returns"] = cost_returns

                # Compute total_loss
                total_loss = T.tensor(0.0, dtype=T.float32, device=self.device)                
                actor_loss, actor_plot_data = self.actor_loss_fn.compute_loss(batch_data)
                value_loss, value_plot_data = self.value_loss_fn.compute_loss(batch_data)    
                cost_loss, cost_plot_data = self.cost_loss_fn.compute_loss(batch_data)    
                total_loss = actor_loss + value_loss + cost_loss

                # Zero optimizers
                self.policy_net.optimizer.zero_grad()
                self.value_net.optimizer.zero_grad()
                self.cost_value_net.optimizer.zero_grad()
          
                total_loss.backward()

                # Gradient clipping
                T.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
                T.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
                T.nn.utils.clip_grad_norm_(self.cost_value_net.parameters(), max_norm=0.5)

                # Optimizer step
                self.policy_net.optimizer.step()
                self.value_net.optimizer.step()
                self.cost_value_net.optimizer.step()
                
                # Plot data 
                plot_policy_loss_dis.append(actor_plot_data['actor_loss_dis'].item())
                plot_entr_dis.append(actor_plot_data['entropy_bonus_dis'])
                plot_kl_dis.append(actor_plot_data['approx_kl_dis'])
                plot_policy_loss_cont.append(actor_plot_data['actor_loss_cont'].item())
                plot_entr_cont.append(actor_plot_data['entropy_bonus_cont'])
                plot_kl_cont.append(actor_plot_data['approx_kl_cont'])
                plot_policy_loss_total.append(actor_plot_data['total_actor_loss'].item())
                plot_value_loss.append(value_plot_data['critic_loss'].item())
                plot_cost_loss.append(cost_plot_data['cost_loss'].item())

        self.logger.log_scalar("train/actor_loss_dis", np.mean(plot_policy_loss_dis), timesteps_so_far)
        self.logger.log_scalar("train/actor_loss_cont", np.mean(plot_policy_loss_cont), timesteps_so_far)
        self.logger.log_scalar("train/total_actor_loss", np.mean(plot_policy_loss_total), timesteps_so_far)
        self.logger.log_scalar("train/critic_loss", np.mean(plot_value_loss), timesteps_so_far)
        self.logger.log_scalar("train/cost_loss", np.mean(plot_cost_loss), timesteps_so_far)
        self.logger.log_scalar("train/entropy_loss x entropy_coef dis", np.mean(plot_entr_dis), timesteps_so_far)
        self.logger.log_scalar("train/approx_kl dis", np.mean(plot_kl_dis), timesteps_so_far)
        self.logger.log_scalar("train/entropy_loss x entropy_coef cont", np.mean(plot_entr_cont), timesteps_so_far)
        self.logger.log_scalar("train/approx_kl cont", np.mean(plot_kl_cont), timesteps_so_far)
        self.logger.log_scalar("lag_parameter", self.lag_parameter.item(), timesteps_so_far)
        self.logger.log_scalar("train/entropy_coef",T.exp(self.log_ent_coef).item(),timesteps_so_far)

        self.buffer.clear()

        return 

    
    def save_models(self, 
                    directory_save:str
                    )->None:
        self.policy_net.save_checkpoint(directory_save)
        self.value_net.save_checkpoint(directory_save)


    def load_trained_networks(self, 
                              train:bool,
                              device:T.device,
                              file_name_policy:str,
                              file_name_critic:str,
                              )->None:
        
        self.policy_net.load_checkpoint(file_name_policy,train,device)
        self.value_net.load_checkpoint(file_name_critic,train,device)
        
        
