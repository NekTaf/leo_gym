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
    Delta_kep_ranges: list = None
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
    
    # def plot_states_interactive(self)->None:
        

    #     fig = make_subplots(
    #         rows=6, cols=2,
    #         shared_xaxes=True,
    #         vertical_spacing=0.03,    
    #         column_widths=[0.7, 0.3],    

    #         specs=[
    #             [{"type": "xy"}, {"type": "scene", "rowspan": 6}],
    #             [{"type": "xy"}, None],
    #             [{"type": "xy"}, None],
    #             [{"type": "xy"}, None],
    #             [{"type": "xy"}, None],
    #             [{"type": "xy"}, None],
    #         ]
    #     )
        
    #     doe = np.array(DebrisSwarm_1.doe)
    #     ada1 = doe[:,0]
    #     adl1 = doe[:,1]
    #     ade1 = np.linalg.norm(doe[:, [2, 3]], axis=1)
    #     simulation_times = np.array(simulation_times)
    #     rewards = np.array(rewards)
    #     controls = np.array(DebrisSwarm_1.controls_RTN)
    #     R, T, N = controls[:,0], controls[:,1], controls[:,2]

    #     fig.add_trace(go.Scatter(y=ada1, mode='lines'), row=1, col=1)
    #     fig.update_yaxes(title_text=r"$a\delta a\ (m)$", row=1, col=1)

    #     fig.add_trace(go.Scatter(y=adl1, mode='lines'), row=2, col=1)
    #     fig.update_yaxes(title_text=r"$a\delta\lambda\ (m)$", row=2, col=1)

    #     fig.add_trace(go.Scatter(y=ade1, mode='lines'), row=3, col=1)
    #     fig.update_yaxes(title_text=r"$\|\!a\,\delta\mathbf e\|\ (m)$", row=3, col=1)

    #     fig.add_trace(go.Scatter(x=simulation_times, y=rewards, mode='markers'), row=4, col=1)
    #     fig.update_yaxes(title_text="Reward", row=4, col=1)

    #     fig.add_trace(go.Scatter(y=R, mode='markers', name='Radial'), row=5, col=1)
    #     fig.update_yaxes(title_text="$f_r$(mN)", row=5, col=1)

    #     P_max = DebrisSwarm_1.p_max_predictions
    #     fig.add_trace(go.Scatter(x=simulation_times,y=np.log(P_max), mode='markers', name='P^{max}_c'), row=6, col=1)
    #     fig.update_yaxes(title_text=r"$\log(P^\text{c}_\text{max})$", row=6, col=1)

    #     for t in simulation_times:
    #         for r in range(1, 7):
    #             fig.add_vline(
    #                 x=t,
    #                 row=r, col=1,          
    #                 line_width=1,
    #                 line_dash="dash",
    #                 line_color="gray",
    #                 opacity=0.5
    #             )

    #     fig.update_xaxes(title_text="Discrete Time", row=6, col=1)

    #     for space_object in DebrisSwarm_1.primary_sat_and_debris_rvm:
    #         coords = np.array(space_object)
    #         fig.add_trace(
    #             go.Scatter3d(
    #                 x=coords[:,0],
    #                 y=coords[:,1],
    #                 z=coords[:,2],
    #                 mode='markers',
    #                 marker=dict(size=5)
    #             ),
    #             row=1, col=2
    #         )

    #     fig.update_layout(
    #         scene=dict(
    #             xaxis_title='X (km)',
    #             yaxis_title='Y (km)',
    #             zaxis_title='Z (km)',
    #             aspectmode='auto'
    #         ),
    #         height=900,
    #         width=1200,       
    #         showlegend=False,
    #         title_text=""
    #     )

    #     fig.show()

        
        
    #     return

    

    def reset(self, 
              seed=None, 
              options=None
              )->Tuple[NDArray, Any]:
        super().reset(seed=seed)
        
        self.sat1.reset_sat_states(keep_ref_trajectory=True)
                
        self.rewards_plot_list = []
        
        self.current_episode_timestep = 0
        
        Delta_kep = self._init_noise(D_kep_ranges=self.cfg.Delta_kep_ranges)
        
        # Adjust RAAN and Inc
        if self.cfg.flag_man_type == "act":
            d_inc = self.cfg.Delta_kep_ranges[2]
            rand_inc_r = np.random.uniform(d_inc[0],d_inc[1])
            
            ix_iy = random_vector_know_norm(2,rand_inc_r)
            Delta_kep[2] = ix_iy[0]
            Delta_kep[3] = ix_iy[1]
            
        self.sat1.set_initial_deviation(d_kep = Delta_kep)
        
        return self._observation_states(), None

