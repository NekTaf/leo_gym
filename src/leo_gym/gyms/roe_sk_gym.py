""" Gymnasium Environment for training RL policy to conduct Along Track (ALT)
and Across Track (ACT) maneuvers. \b


ACT:
+ controls relative inclination $a\delta \mathbf{i}$ \b
+ uses Normal maneuvers \b

ALT:
+ controls relative SMA ($a\delta a$) \b
+ relative eccentricity vector $a\delta\mathbf{e}$ \b
+ relative argument of longitude $a\delta\lambda$ \b

"""

# Standard library
import os
from typing import Any, Tuple, List

# Third-party
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from pydantic import Field
from numpy.typing import NDArray

# Local
from leo_gym.satellite.satellite_roe import SatelliteROEConfig, Satellite
from leo_gym.utils.matplot_style_cfg import *
from leo_gym.utils.utils import random_vector_know_norm, seed_all
from dataclasses import dataclass

@dataclass
class RoeGymConfig():
    high_action: list = None
    low_action: list = None
    Droe_ranges: list = None #Delta([ada, adl, adex, adey, adix, adiy])
    no_timesteps: int = None
    satellite_params: SatelliteROEConfig = None
    dt: int = None
    adi_norm_cutoff: float = None
    adi_norm_target: float = None
    flag_man_type: str = None
    Delta_ada_targ: float = None
    ada_target: float = None

class RoeGym(gym.Env):

    def __init__(self,
                 cfg:RoeGymConfig, 
                 seed:int):
        super(RoeGym, self).__init__()
        
        seed_all(seed=seed)
    
        self.cfg = cfg
        self.sat1 = Satellite(cfg=self.cfg.satellite_params)
        
        self.reset()
        self._observation_states()

        self.action_space = spaces.Box(low=np.array([-1.0,-1.0,-1.0]), 
                                       high=np.array([+1.0,+1.0,+1.0]), 
                                       shape=(3,), 
                                       dtype=np.float64)
        
        self.observation_space = spaces.Box(low=-np.inf, 
                                            high=np.inf, 
                                            shape=(self.obs_shape,), 
                                            dtype=np.float64)
        
            
    def _observation_states(self
                            )->NDArray:
        """Returns Observations based on maneuver type"""
        
        if self.cfg.flag_man_type == "alt":
            
            obs = np.array([self.sat1.oe_ns[-1][1],
                            self.sat1.roe[-1][1]/1e3,
                            self.sat1.roe[-1][0],
                            self.sat1.roe[-1][2]/1e2,
                            self.sat1.roe[-1][3]/1e2])
            
            self.obs_shape = len(obs)
            
            return obs
            
        if self.cfg.flag_man_type == "act":
            
            obs = np.array([self.sat1.oe_ns[-1][1],
                                    self.sat1.roe[-1][4]/1e3, 
                                    self.sat1.roe[-1][5]/1e3])
            
            self.obs_shape = len(obs)
            
            return obs
                    
    def _init_noise(self,
                    D_kep_ranges:List
                    )->NDArray:
        
        Delta_keps = np.zeros(6,)
        
        for D_kep_range, Delta_kep in zip(D_kep_ranges,Delta_keps):
            Delta_kep = np.random.uniform(D_kep_range[0], D_kep_range[1]) 
            operator = np.random.choice([-1,1])
            Delta_kep = Delta_kep*operator
            
        return Delta_keps


    def _reward_fun_act(self
                        )->tuple[float,bool]:
        """Reward function for ACT
        Returns reward and termination: \b
        + Success: target boundary reached \b
        + Failure: target boundary violated \b
        """
        terminated = False
        reward = 0
        
        # Inclination vectors
        adix = self.sat1.roe[-1][4]
        adiy = self.sat1.roe[-1][5]
        r_inc = np.sqrt(adix**2 + adiy**2)
                
        reward += -np.log(r_inc+1)
        
        if self.termination_check_act():
            reward = -100
            terminated = True
            
        elif r_inc<=self.cfg.adi_norm_target:
            reward += 100
            terminated = True
            
            
        return reward, terminated

            
    def process_action(self
                          )->None:
        """It is good practice to always scale the network output to [-1,+1]
        This is then min-max rescaled to the appropriate bounds 
        
        Note different algorithms handle action bounds differently, eg.:
        + PPO relies on environment level clipping \b
        + SAC relies on tanh() squashing on algo-network level \b
    
        """
        
        self.action = np.clip(self.action,-1,+1)

        a_min = np.array(self.cfg.low_action)
        a_max = np.array(self.cfg.high_action)
        
        self.action = ((self.action + 1) / 2) * (a_max - a_min) + a_min
        
        if self.action[0] >= 0:
            self.action[0] = self.cfg.satellite_params.ideal_traj_params.f_max
        elif self.action[0] < 0:
            self.action[0] = -self.cfg.satellite_params.ideal_traj_params.f_max
            
        return
    
    
    def termination_check_act(self,
                          )-> bool:
        
        """Check if inclination magnitude surpasses cut-off magnitude.
        
        Useful in speeding up training by not allowing satellite violate
        """
        
        if np.sqrt(self.sat1.roe[-1][4]**2 + self.sat1.roe[-1][5]**2) >= self.cfg.adi_norm_cutoff:
            return True
        else:
            return False
        
        
    def truncation_check(self
                         )-> bool:
        
        """Limits episode length based on number of days worth of orbit parameter
        """
        if (self.sat1.discrete_time_index_simulation*self.cfg.dt)/(60*60*24)\
                                    >= self.cfg.satellite_params.days:
            return True
        else:
            return False


        
    def step(self, action
             )->Tuple[NDArray,float,bool,bool,Any]:
        
        terminated = False
        truncated = False
        reward = 0
        
        self.action = action
        self.process_action()
        sojourn_t = int(np.sum(self.action[-2:]))
        
        self.sat1.apply_manplan(
            manplan=self.action,
            flag_man_type=self.cfg.flag_man_type)    
        
        self.current_episode_timestep += 1  
        
        if self.cfg.flag_man_type == "alt":  
            raise NotImplementedError
            
        elif self.cfg.flag_man_type == "act":
            reward, terminated = self._reward_fun_act()
                                
        truncated = self.truncation_check()
                
        info ={"sojourn_t": sojourn_t/60 #hours
            }
        
        self.rewards_plot_list.append(reward)
        
        return self._observation_states(), reward, terminated, truncated, info
    
    
    
    
    def plot_rewards(self,
                     save_path:str
                     )->None:
        
        
        plt.figure(figsize=(4, 3))
        plt.plot(self.rewards_plot_list) 
        plt.xlabel("Environment timestep")
        plt.ylabel("Reward")
        plt.show()
        
        if save_path is not None:
            plt.savefig(os.path.join(save_path,"rewards_plot.png"))
            # plt.savefig(os.path.join(save_path,"rewards_plot.pdf"))
        
        return 
    
    
    def reset(self, 
              seed=None, 
              options=None
              )->Tuple[NDArray, Any]:
        super().reset(seed=seed)
        
        self.sat1.reset_sat_states(keep_ref_trajectory=True)
                
        self.rewards_plot_list = []
        
        self.current_episode_timestep = 0
        
        Droe_ranges = self.cfg.Droe_ranges
        Droe = np.zeros(6)
        
        # Adjust RAAN and Inc
        if self.cfg.flag_man_type == "act":
            Droe[4] = np.random.uniform(Droe_ranges[4][0],Droe_ranges[4][1])
            Droe[5] = np.random.uniform(Droe_ranges[5][0],Droe_ranges[5][1])
            
        self.sat1.set_initial_deviation(Droe = Droe)
        
        return self._observation_states(), None

