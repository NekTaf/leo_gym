"""Reconfiguration gym environment for a single satellite.

The agent picks:
1) two continuous actions  -> [burn start delay, burn duration] in *environment time-steps*
2) seven discrete logits   -> [fire_r, -fire_r, fire_t, -fire_t, fire_n, -fire_n, dont_fire]

The discrete vector is interpreted as a probability distribution (softmax).
A single thruster direction is sampled so only one axis fires at a time.

`Satellite.apply_manplan` expects manplan = [thrust (N), delay (steps), duration (steps)]
and a flag describing the thrust axis (rad/alt/act).
"""

# Standard library
import random
from typing import Any, Dict, Optional, Tuple

# Third-party
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

# Local
from leo_gym.satellite.satellite_roe import Satellite, SatelliteROEConfig
from leo_gym.utils.utils import seed_all


class RecfgEnvConfig(BaseModel):
    """Static configuration for the reconfiguration environment."""

    low_action: Any = Field(..., description="Lower bounds for [delay, duration] (in env steps)")
    high_action: Any = Field(..., description="Upper bounds for [delay, duration] (in env steps)")
    satellite_config: SatelliteROEConfig = Field(..., description="Satellite dynamics used for propagation")

    satellite_observation_feature_size: int = Field(
        6, ge=1, description="Number of ROE values exposed in the observation"
    )
    continuous_actions_size: int = Field(2, ge=1, description="Dimensionality of continuous actions")
    discrete_actions_size: int = Field(7, ge=1, description="Number of discrete thrust choices")
    f_max: float = Field(1.0, gt=0, description="Maximum thrust magnitude (N)")
    max_time_index: int = Field(1440, ge=1, description="Episode length in simulator steps")

    Droe_ranges: list # Delta([ada, adl, adex, adey, adix, adiy]) Initial ROE deviation ranges for reset
    target_roe: Optional[Any] = Field(
        None, description="Desired ROE vector. Defaults to zeros (match the reference orbit)."
    )
    target_tolerance: float = Field(
        25, ge=0, description="Distance to target (in meters of ROE norm) that counts as success"
    )

    success_reward: float = Field(100.0, description="Bonus when the satellite reaches the target band")

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class RecfgEnv(gym.Env):
    """Minimal reconfiguration environment aligned with existing CAM/ROE gyms."""

    def __init__(self, cfg: RecfgEnvConfig, seed: int = 0):
        super().__init__()
        seed_all(seed)
        self.cfg = cfg
        
        self.satellite = Satellite(cfg=self.cfg.satellite_config) # type: ignore
        self.rewards_plot_list = []

        # Build initial satellite state so the first call to step has data
        self.reset()
        self._observation_states()

        # Direction map: (flag for `apply_manplan`, sign for thrust)
        self._direction_map: Dict[int, Tuple[str, int]] = {
            0: ("R", +1),   # + radial
            1: ("R", -1),   # - radial
            2: ("T", +1),   # + tangential (along-track)
            3: ("T", -1),   # - tangential
            4: ("N", +1),   # + normal (across-track)
            5: ("N", -1),   # - normal
            6: ("N", 0),    # no firing, coast only
        }

        # Action: Dict for hierarchical PPO (matches cam_gym style)
        self.action_space = spaces.Dict(
            {
                "discrete": spaces.Discrete(self.cfg.discrete_actions_size),
                "continuous": spaces.Box(
                    low=np.array(self.cfg.low_action, dtype=np.float64),
                    high=np.array(self.cfg.high_action, dtype=np.float64),
                    shape=(self.cfg.continuous_actions_size,),
                    dtype=np.float64,
                ),
            }
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.satellite_observation_feature_size,),
            dtype=np.float64,
        )



    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _observation_states(self) -> NDArray[np.float64]:
        """Assemble flat observation vector."""
        if self.satellite is None or len(self.satellite.roe) == 0:
            return np.zeros(self.cfg.satellite_observation_feature_size, dtype=np.float64)
        else:
            nds = np.array(self.satellite.roe[-1], dtype=np.float64).reshape(-1)
            return nds[: self.cfg.satellite_observation_feature_size]
        
    
    def _select_discrete_action(self, discrete_action: Any) -> int:
        try:
            return int(np.clip(int(np.array(discrete_action).item()), 0, self.cfg.discrete_actions_size - 1))
        except Exception:
            return 0
        

    def _target_vector(self) -> NDArray[np.float64]:
        """Return the desired ROE vector, defaulting to zeros if none is provided."""
        if self.cfg.target_roe is None:
            return np.zeros(self.cfg.satellite_observation_feature_size, dtype=np.float64)

        target = np.array(self.cfg.target_roe, dtype=np.float64).reshape(-1)
        return target[: self.cfg.satellite_observation_feature_size]


    def _compute_reward(self) -> Tuple[float, bool]:
        """
        Shaping reward:
        - negative distance to the target ROE (scaled to keep numbers small)
        - optional bonus when within `target_tolerance`
        - penalty for using thrust (encourages minimal fuel)
        """
        terminated = False
        reward = 0
        
        current = self._observation_states()
        target = self._target_vector()

        distance = float(np.linalg.norm(current - target))
        reward = -np.log(distance + 1)

        if distance <= self.cfg.target_tolerance:
            reward += self.cfg.success_reward
            terminated = True

        return float(reward), terminated
    
    def process_action_cont(self
                          )->None:
        """It is good practice to always scale the network output to [-1,+1]
        This is then min-max rescaled to the appropriate bounds 
        
        Note different algorithms handle action bounds differently, eg.:
        + PPO relies on environment level clipping \b
        + SAC relies on tanh() squashing on algo-network level \b
    
        """
        
        self.action_cont = np.clip(self.action_cont,-1,+1)

        a_min = np.array(self.cfg.low_action)
        a_max = np.array(self.cfg.high_action)
        
        self.action_cont = ((self.action_cont + 1) / 2) * (a_max - a_min) + a_min

            
        return

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def step(self, action: Dict[str, Any]
             ) -> Tuple[NDArray[np.float64], float, bool, bool, Dict[str, Any]]:
        """Apply burn plan, propagate the satellite, and return gym-style outputs."""
        if self.satellite is None:
            raise RuntimeError("Environment used before calling reset().")

        terminated = False
        truncated = False
        reward = 0
        
        self.action_cont = action["continuous"]
        self.process_action_cont()       
        sojourn_t = int(np.sum(self.action_cont[-2:]))
        
        delay_steps = int(np.round(self.action_cont[0]))
        duration_steps = int(np.round(self.action_cont[1]))

        # Map discrete choice to an RTN axis and thrust sign
        discrete_idx = self._select_discrete_action(action["discrete"])
        axis_flag, sign = self._direction_map.get(discrete_idx, ("N", 0))

        thrust_mag = float(sign * self.cfg.f_max)
        manplan = np.array([thrust_mag, delay_steps, duration_steps], dtype=np.float64)

        self.satellite.apply_manplan(
            manplan=manplan,
            flag_man_type=axis_flag)
        
        self.current_episode_timestep += 1

        reward, terminated = self._compute_reward()
        
        # Truncation check
        time_up = self.satellite.discrete_time_index_simulation >= self.cfg.max_time_index
        if time_up:
            truncated = True

        info = {
            "cost": abs(thrust_mag) * duration_steps,
            "sojourn_t": sojourn_t/60,  # hours
            "chosen_discrete": discrete_idx,
            "manplan": manplan.tolist(),
            "axis_flag": axis_flag,
            "n": getattr(self.satellite, "discrete_time_index_simulation", 0),
        }

        obs = self._observation_states()
        self.rewards_plot_list.append(reward)

        return obs, reward, terminated, truncated, info


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None): # type: ignore
        """Reset the satellite to its reference trajectory with a small ROE offset."""
        super().reset(seed=seed)
        # Gym may pass seed=None; fall back to a random integer to avoid torch seeding errors.
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        self.satellite = Satellite(cfg=self.cfg.satellite_config) # type: ignore
        # Keep the reference trajectory and only reset the perturbed track
        self.satellite.reset_sat_states(keep_ref_trajectory=True)
        
        self.rewards_plot_list = []
        
        self.current_episode_timestep = 0

        Droe_ranges = self.cfg.Droe_ranges
        Droe = np.zeros(6, dtype=np.float64)
        
        Droe[0] = np.random.uniform(Droe_ranges[0][0], Droe_ranges[0][1])  # ada
        Droe[1] = np.random.uniform(Droe_ranges[1][0], Droe_ranges[1][1])  # adl
        Droe[2] = np.random.uniform(Droe_ranges[2][0], Droe_ranges[2][1])  # adex  
        Droe[3] = np.random.uniform(Droe_ranges[3][0], Droe_ranges[3][1])  # adey
        Droe[4] = np.random.uniform(Droe_ranges[4][0], Droe_ranges[4][1])  # adix
        Droe[5] = np.random.uniform(Droe_ranges[5][0], Droe_ranges[5][1])  # adiy
        
        
        # Small default deviation keeps observations informative at t=0
        self.satellite.set_initial_deviation(Droe) # type: ignore


        obs = self._observation_states()
        info = {"cost": 0.0, "sojourn_t": 0.0, "n": getattr(self.satellite, "discrete_time_index_simulation", 0)}
        return obs, info

