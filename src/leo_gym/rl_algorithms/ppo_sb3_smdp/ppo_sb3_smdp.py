import numpy as np
import torch as th
import math
from typing import Optional, Tuple, List, Dict, Any

from stable_baselines3.ppo import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import GymObs, RolloutReturn
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
import math
import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor
import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces

class SMDPRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer with time-aware (Semi-MDP) discounting.
    """
    
    def __init__(
        self,
        *args,
        **kwargs
    ):
        
        super().__init__(*args, **kwargs)
        self.sojourn_t_buff = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # exp(-beta*1) = gamma


    def add(self, obs, action, reward, episode_start, value, log_prob, sojourn_t):
        self.sojourn_t_buff[self.pos] = sojourn_t
        super().add(obs, action, reward, episode_start, value, log_prob)


    def _gamma_t(self,
                 sojourn_t_arr: np.ndarray
                 ) -> np.ndarray:
        return np.power(self.gamma, sojourn_t_arr).astype(np.float32)


    def _lambda_t(self, 
                  sojourn_t_arr: np.ndarray
                  ) -> np.ndarray:

        return np.power(self.gae_lambda, sojourn_t_arr).astype(np.float32)



    def compute_returns_and_advantage(self, 
                                      last_values:th.Tensor,
                                      dones:np.ndarray
                                      )->None:
        # shape: (n_envs,)
        last_values = last_values.clone().cpu().numpy().flatten()
        dones = np.array(dones).astype(bool, copy=False).reshape(-1)        
        
        last_gae_lam = np.zeros(self.n_envs, dtype=np.float32)

        for step in reversed(range(self.buffer_size)):
            # (n_envs,)
            dt = self.sojourn_t_buff[step]                    
            gamma_t  = self._gamma_t(dt)           
            lambda_t = self._lambda_t(dt)         

            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]   

            delta = self.rewards[step] + next_non_terminal * gamma_t * next_values - self.values[step]
            last_gae_lam = delta + next_non_terminal * gamma_t * lambda_t * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values
        
        
        
class SMDP_PPO(PPO):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        self.rollout_buffer = SMDPRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 \
                and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, \
                        self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            sojourn_t_arr = np.full((env.num_envs,), 1.0, dtype=np.float32)
            if isinstance(infos, (list, tuple)):
                for i, info in enumerate(infos):
                    if info:
                        sojourn_t_arr[i] = float(info.get("sojourn_t", 1.0))

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]\
                                                ["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]

                    gamma_t = float(rollout_buffer._gamma_t(\
                        np.asarray([sojourn_t_arr[idx]], 
                        dtype=np.float32))[0])
                    
                    rewards[idx] += gamma_t * terminal_value

            try:
                rollout_buffer.add(
                    self._last_obs,  # type: ignore[arg-type]
                    actions,
                    rewards,
                    self._last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs,
                    sojourn_t=sojourn_t_arr,
                )
            except TypeError:
                # Fallback for non-SMDP buffers
                rollout_buffer.add(
                    self._last_obs,  # type: ignore[arg-type]
                    actions,
                    rewards,
                    self._last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs,
                )

            self._last_obs = new_obs   
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True
