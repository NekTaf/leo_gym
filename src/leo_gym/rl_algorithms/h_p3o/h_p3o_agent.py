from typing import Any, Optional, Tuple
import numpy as np
import torch as T
import gymnasium as gym 

from leo_gym.rl_algorithms.h_ppo.h_ppo_agent import Agent  
from leo_gym.rl_algorithms.h_p3o.losses import CostP3OLoss, CostCriticLoss


class HP3OAgent(Agent):
    """
    HPPO Agent with multi-cost P3O (penalized PPO) extension.

    - Uses the base HPPO Agent reward actor/critic.
    - For each environment cost component:
        * Adds a separate cost value network (critic)
        * Adds a P3O-style cost penalty term to the actor loss.
    """

    def __init__(self,
                 env_obs,
                 env_actions,
                 env_cfg,
                 ppo_cfg):
        # Initialize standard HPPO Agent
        super().__init__(env_obs, env_actions, env_cfg, ppo_cfg)

        # Multi-cost configuration  
        # Expect a list of cost limits in env_cfg
        self.cost_limits = getattr(self.env_cfg, "cost_limits", [])
        self.num_costs = len(self.cost_limits)

        # Separate gamma / lambda for costs (fallback to reward ones)
        self.cost_gamma = getattr(self.ppo_cfg, "cost_gamma", self.ppo_cfg.gamma)
        self.cost_gae_lambda = getattr(self.ppo_cfg, "cost_gae_lambda", self.ppo_cfg.gae_lambda)

        # Penalty coefficient (shared across costs by default)
        self.cost_kappa = getattr(self.ppo_cfg, "cost_kappa", 1.0)

        # Cost value networks + loss objects
        self.cost_value_nets = []
        self.cost_critic_losses = []
        self.cost_actor_losses = []

        if self.num_costs > 0:
            for i in range(self.num_costs):
                # Cost critic network
                cost_v_net = self.ppo_cfg.critic_wrapper(
                    state_dim=env_obs.shape[0],
                    device=self.device,
                    lr=self.ppo_cfg.lr,
                    observation_encoder=self.ppo_cfg.observation_encoder,
                )
                self.cost_value_nets.append(cost_v_net)

                # Cost critic loss
                self.cost_critic_losses.append(
                    CostCriticLoss(
                        value_net=cost_v_net,
                        policy_clip=self.ppo_cfg.policy_clip,
                    )
                )

                # Cost actor loss (P3O)
                self.cost_actor_losses.append(
                    CostP3OLoss(
                        policy_net=self.policy_net,
                        kappa=self.cost_kappa,
                        cost_limit=self.cost_limits[i],
                    )
                )

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
        costs=None,
    ):
        """
        Same as base Agent.rollout_buffer, but also stores 'costs'
        (a vector per environment step) if provided.
        """
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

        if costs is not None:
            step_data["costs"] = costs

        self.buffer.push(step_data)

    def update(self,
               timesteps_so_far: int,
               total_timesteps: int) -> None:
        """
        Single PPO/P3O update: generate batches, compute losses, update nets.

        - Reward actor & critic: from base HPPO.
        - For each cost dimension:
            * Compute cost GAE, returns.
            * Add P3O actor penalty + cost critic loss.
        """
        plot_policy_loss_dis, plot_policy_loss_cont, plot_value_loss, plot_policy_loss_total = [], [], [], []
        plot_entr_cont, plot_kl_cont = [], []
        plot_entr_dis, plot_kl_dis = [], []

        # Cost logging
        plot_cost_actor_loss = [[] for _ in range(self.num_costs)]
        plot_cost_critic_loss = [[] for _ in range(self.num_costs)]
        plot_cost_Jc = [[] for _ in range(self.num_costs)]
        plot_cost_surr_cadv = [[] for _ in range(self.num_costs)]

        # LR scheduling
        frac = timesteps_so_far / total_timesteps if total_timesteps is not None else 0
        new_lr = self.lr * self.ppo_cfg.lr_decay_coef * (1 - frac)
        new_lr = max(new_lr, 3e-4)
        self.lr = new_lr

        for g in self.policy_net.optimizer.param_groups:
            g["lr"] = self.lr
        for g in self.value_net.optimizer.param_groups:
            g["lr"] = self.lr
        for v_net in self.cost_value_nets:
            for g in v_net.optimizer.param_groups:
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

            # Reward GAE / returns
            advantage = self.GAE_fun(
                reward_arr=rewards_arr,
                vals_arr=values_arr,
                dones_arr=dones_arr,
                t_soujourn_arr=sojourn_t_arr,
                gae_lambda=None,
                gamma=None,
            )
            values = T.tensor(values_arr).to(self.device)

            # Cost GAEs / returns
            has_costs = ("costs" in all_data) and (self.num_costs > 0)

            if has_costs:
                costs_arr = all_data["costs"]  # [T, n_envs, num_costs] or [T, n_envs]

                # Expand to 3D if single cost
                if costs_arr.ndim == 2:
                    costs_arr = costs_arr[..., None]

                assert costs_arr.shape[-1] == self.num_costs, \
                    f"num_costs mismatch: env has {costs_arr.shape[-1]}, agent has {self.num_costs}"

                # Precompute cost value baselines and GAEs for all states
                states_all = T.tensor(obs_arr, dtype=T.float32).to(self.device)

                cost_advantages = []
                cost_values_all = []
                cost_returns_all = []
                ep_costs = []

                for i in range(self.num_costs):
                    # Old cost values for all states
                    with T.no_grad():
                        v_i = T.squeeze(self.cost_value_nets[i](states_all)).cpu().numpy()

                    cost_values_all.append(v_i)

                    # GAE for cost
                    adv_cost_i = self.GAE_fun(
                        reward_arr=costs_arr[..., i],
                        vals_arr=v_i,
                        dones_arr=dones_arr,
                        t_soujourn_arr=sojourn_t_arr,
                        gae_lambda=self.cost_gae_lambda,
                        gamma=self.cost_gamma,
                    )
                    cost_advantages.append(adv_cost_i)

                    # Cost returns
                    v_i_t = T.tensor(v_i).to(self.device)
                    ret_cost_i = adv_cost_i + v_i_t
                    cost_returns_all.append(ret_cost_i)

                    # Episodic mean cost over buffer (for constraint)
                    ep_cost_i = float(costs_arr[..., i].mean())
                    ep_costs.append(ep_cost_i)

            # Mini-batch loop
            for batch_idxs in batches:
                
                states_batch = T.tensor(obs_arr[batch_idxs], dtype=T.float32).to(self.device)

                batch_data = {
                    "obs": states_batch,
                    "act_dis": None,
                    "act_cont": None,
                    "old_logp_dis": None,
                    "old_logp_cont": None,
                    "adv_rew": None,
                    "values_old": None,
                    "returns": None,
                }                
                                
                # Reward advantage
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

                # Reward returns
                returns = advantage[batch_idxs] + values[batch_idxs]
                batch_data["values_old"] = values[batch_idxs]
                batch_data["returns"] = returns

                # Reward actor/critic losses
                actor_loss, actor_plot_data = self.actor_loss_fn.compute_loss(batch_data)
                value_loss, value_plot_data = self.value_loss_fn.compute_loss(batch_data)
                total_loss = actor_loss + value_loss

                # Cost losses (actor + critic)
                if has_costs:
                    for i in range(self.num_costs):
                        idx_str = str(i)

                        adv_cost_batch = cost_advantages[i][batch_idxs]
                        values_cost_batch = T.tensor(
                            cost_values_all[i][batch_idxs],
                            dtype=T.float32,
                        ).to(self.device)
                        returns_cost_batch = cost_returns_all[i][batch_idxs]

                        if self.normalize_advantage:
                            adv_cost_batch = (adv_cost_batch - adv_cost_batch.mean()) / (
                                adv_cost_batch.std() + 1e-8
                            )

                        cost_batch_data = dict(batch_data)
                        cost_batch_data["adv_cost_" + idx_str] = adv_cost_batch
                        cost_batch_data["ep_cost_" + idx_str] = ep_costs[i]
                        cost_batch_data["values_cost_" + idx_str] = values_cost_batch
                        cost_batch_data["returns"] = returns_cost_batch

                        cost_actor_loss, cost_actor_plot = self.cost_actor_losses[i].compute_loss(
                            cost_batch_data, cost_batch_key_index=idx_str
                        )
                        total_loss = total_loss + cost_actor_loss

                        cost_critic_loss, cost_critic_plot = self.cost_critic_losses[i].compute_loss(
                            cost_batch_data, cost_batch_key_index=idx_str
                        )
                        total_loss = total_loss + cost_critic_loss

                        # Logging
                        plot_cost_actor_loss[i].append(cost_actor_plot["p3o_cost_loss"])
                        plot_cost_critic_loss[i].append(cost_critic_plot["values_cost_" + idx_str])
                        plot_cost_Jc[i].append(cost_actor_plot["p3o_Jc"])
                        plot_cost_surr_cadv[i].append(cost_actor_plot["p3o_surr_cadv"])

                # Backward / step 
                self.policy_net.optimizer.zero_grad()
                self.value_net.optimizer.zero_grad()
                for v_net in self.cost_value_nets:
                    v_net.optimizer.zero_grad()
          
                total_loss.backward()

                T.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
                T.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
                for v_net in self.cost_value_nets:
                    T.nn.utils.clip_grad_norm_(v_net.parameters(), max_norm=0.5)

                self.policy_net.optimizer.step()
                self.value_net.optimizer.step()
                for v_net in self.cost_value_nets:
                    v_net.optimizer.step()                
                
                # Reward logging arrays
                plot_policy_loss_dis.append(actor_plot_data['actor_loss_dis'])
                plot_entr_dis.append(actor_plot_data['entropy_bonus_dis'])
                plot_kl_dis.append(actor_plot_data['approx_kl_dis'])
                plot_policy_loss_cont.append(actor_plot_data['actor_loss_cont'])
                plot_entr_cont.append(actor_plot_data['entropy_bonus_cont'])
                plot_kl_cont.append(actor_plot_data['approx_kl_cont'])
                plot_policy_loss_total.append(actor_plot_data['total_actor_loss'])
                plot_value_loss.append(value_plot_data['critic_loss'])

        # Scalar logging
        if self.logger:
            self.logger.log_scalar("train/actor_loss_dis", np.mean(plot_policy_loss_dis), timesteps_so_far)
            self.logger.log_scalar("train/actor_loss_cont", np.mean(plot_policy_loss_cont), timesteps_so_far)
            self.logger.log_scalar("train/total_actor_loss", np.mean(plot_policy_loss_total), timesteps_so_far)
            self.logger.log_scalar("train/critic_loss", np.mean(plot_value_loss), timesteps_so_far)
            self.logger.log_scalar("train/entropy_loss x entropy_coef dis", np.mean(plot_entr_dis), timesteps_so_far)
            self.logger.log_scalar("train/approx_kl dis", np.mean(plot_kl_dis), timesteps_so_far)
            self.logger.log_scalar("train/entropy_loss x entropy_coef cont", np.mean(plot_entr_cont), timesteps_so_far)
            self.logger.log_scalar("train/approx_kl cont", np.mean(plot_kl_cont), timesteps_so_far)
            self.logger.log_scalar("train/entropy_coef", T.exp(self.log_ent_coef).item(), timesteps_so_far)

            for i in range(self.num_costs):
                idx_str = str(i)
                if len(plot_cost_actor_loss[i]) > 0:
                    self.logger.log_scalar(f"train/p3o_cost_actor_loss_{idx_str}",
                                           np.mean(plot_cost_actor_loss[i]), timesteps_so_far)
                    self.logger.log_scalar(f"train/p3o_cost_critic_loss_{idx_str}",
                                           np.mean(plot_cost_critic_loss[i]), timesteps_so_far)
                    self.logger.log_scalar(f"train/p3o_Jc_{idx_str}",
                                           np.mean(plot_cost_Jc[i]), timesteps_so_far)
                    self.logger.log_scalar(f"train/p3o_surr_cadv_{idx_str]",
                                           np.mean(plot_cost_surr_cadv[i]), timesteps_so_far)

        self.buffer.clear()

    def train(self, env, training_cfg):
        """
        Same as base Agent.train, but expects env.step(info) to include
        a 'costs' entry: list/np.array of shape (num_costs,) per env.
        """
        import os
        import mlflow
        from tqdm import tqdm
        import numpy as np

        try:
            os.makedirs(training_cfg.tracking_uri, exist_ok=True)
            mlflow.set_tracking_uri(training_cfg.tracking_uri)
            mlflow.set_experiment(training_cfg.experiment_name)

            with mlflow.start_run(run_name=training_cfg.run_name) as run:
                
                mlflow.log_dict(self.env_cfg.model_dump(), 
                                "env_cfg.json")
                
                mlflow.log_dict(self.ppo_cfg.model_dump(mode="python"), 
                                "ppo_cfg.json")

                mlflow.log_dict(training_cfg.model_dump(), 
                                "training_cfg.json")
                
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

                        # Step env
                        next_state, reward, terminated, truncated, info = env.step(action)
                        sojourn_t = info["sojourn_t"]
                        costs = info.get("costs", None)

                        # Buffer
                        self.rollout_buffer(
                            state,
                            action_dis,
                            action_cont,
                            prob_dis,
                            prob_cont,
                            val,
                            reward,
                            terminated,
                            sojourn_t,
                            costs=costs,
                        )

                        self.timesteps_so_far += self.ppo_cfg.default_num_envs
                        pbar.update(self.ppo_cfg.default_num_envs)
                        state = next_state
                        
                        # Update
                        if self.timesteps_so_far % (self.ppo_cfg.steps_per_env * self.ppo_cfg.default_num_envs) == 0:
                            self.update(self.timesteps_so_far, total_timesteps=self.ppo_cfg.max_training_timesteps)
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

                        if self.timesteps_so_far % self.ppo_cfg.save_nets_period == 0:
                            directory_save = os.path.join(
                                training_cfg.tracking_uri,
                                experiment_id,
                                run_id,
                                "artifacts/models",
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
            "artifacts/models",
            "final"
        )
        
        os.makedirs(directory_save, exist_ok=True)
        self.save_models(directory_save=directory_save)
