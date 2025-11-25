# Standard library
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

# Third-party
import gymnasium as gym
import numpy as np
import plotly.graph_objects as go
import torch
from gymnasium import spaces
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from pydantic import BaseModel, ConfigDict, Field
from scipy.integrate import quad

# Local
from leo_gym.satellite.sat_debris_cluster import (
    SatDebrisCluster,
    SatDebrisClusterConfig,
)
from leo_gym.utils.matplot_style_cfg import *
from leo_gym.utils.utils import seed_all

gym.logger.set_level(40)


class CamEnvConfig(BaseModel):
    """
    Config for CAM environment
    """
    high_action: Any = Field(..., description="Upper bounds for each action dimension")
    low_action: Any  = Field(..., description="Lower bounds for each action dimension")
    debris_cluster_config: SatDebrisClusterConfig = Field(
        ..., description="Configuration for debris cluster dynamics object"
    )

    p_max_limit: float = Field(..., gt=0, description="Max thrust limit")
    adl_req: float = Field(..., ge=0, description="ADL requirement")
    ade_norm_req: float = Field(..., ge=0, description="ADE normalization")
    max_time_index: int = Field(..., ge=1, description="Max real time per episode")
    
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    


class CamEnv(gym.Env):
    
    """
    + Subscript _p denotes primary satellite
    + Subscript _s denotes secondary debris objects
    + TCA stands for time of closest approach where collision happens 
    + For data, interact with SatDebrisCluster object containing satellite and debris data and propagation function
    + Config function for object defining satellite and debris parameters is 
    required, they are assumed to be identical in this example

    """

    def __init__(self, 
                 cfg:CamEnvConfig,
                 seed:int):
        
        super(CamEnv, self).__init__()
        
        self.cfg = cfg
        seed_all(seed)        
        self.reset()
        
        
        self.action_space = spaces.Dict({
            "discrete": spaces.Discrete(3),
            "continuous": spaces.Box(low=np.array(self.cfg.low_action), 
                                       high=np.array(self.cfg.high_action), 
                                       shape=(2,), 
                                       dtype=np.float64)
        })
            
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float64
        )

                        
    
    def _observation_states(self
                            )->NDArray[np.float64]:
        
        # Relative orbital elements between satellite and unperturbated trajectory
        roe = np.array(self.DebrisSwarm_1.roe)
        
        # Primary satellite non_singular_oe
        non_singular_oe_p = np.array(self.DebrisSwarm_1.non_singular_oe[0:1])[:,-1, :].reshape(6,) 
        
        # Secondary debvris object values non_singular_oe
        non_singular_oe_s = np.array(self.DebrisSwarm_1.non_singular_oe[1:])[:,-1, :] 
 
        adl = roe[-1][1]/1e3 # convert to km to normalize // range: 0 - -/+5
        adex = roe[-1][2]/1e2 # convert to km to normalize // range: 0 - -/+ 9
        adey = roe[-1][3]/1e2 # convert to km to normalize // range 0 - -/+ 9
        
        l_p = non_singular_oe_p[1] # range 0 - 6 no need to normalize
        inc_p = non_singular_oe_p[4] # range 0 - 6 no need to normalize
        raan_p = non_singular_oe_p[5] #range 0 - 6 no need to normalize

        # Observations of primary satellite 
        obs_satellite = np.array((
            l_p,
            inc_p,
            raan_p,
            adl,
            adex, 
            adey))
        
        obs_satellite_minimal = np.array([
            l_p,
            adl,
            adex,
            adey])        

        # Debris related observations
        obs_debris = np.zeros((self.DebrisSwarm_1.no_debris,7))
        obs_debris_minimal = np.zeros((self.DebrisSwarm_1.no_debris,2))
        
        P_max_propagated = np.array(self.DebrisSwarm_1.metrics_at_tca)

        for i, (nsoe) in enumerate(non_singular_oe_s):
        
            #discrete conjuction time (discretized dt=60s)
            tca_true = self.DebrisSwarm_1.conjuction_points_time[i]
            tca_till = np.array(tca_true - self.DebrisSwarm_1.n).reshape(1,)

            if np.sign(tca_till) == -1: 
                ## no observations if tca is past
                pass
            
            else:             
                ## log to normalize
                p_max_at_tca = np.array([np.log((P_max_propagated[i,1]))]) 
                
                ## Min-Max scale to -/+2.5
                C_min = np.array([
                    [self.cfg.debris_cluster_config.C_rtn_s_ranges[0][0]**2 + \
                        self.cfg.debris_cluster_config.C_rtn_p_ranges[0][0]**2, 0, 0],
                    [0, self.cfg.debris_cluster_config.C_rtn_s_ranges[1][0]**2 +\
                        self.cfg.debris_cluster_config.C_rtn_p_ranges[1][0]**2, 0],
                    [0, 0, self.cfg.debris_cluster_config.C_rtn_s_ranges[2][0]**2 +\
                        self.cfg.debris_cluster_config.C_rtn_p_ranges[2][0]**2]
                ])
                C_max = np.array([
                    [self.cfg.debris_cluster_config.C_rtn_s_ranges[0][1]**2 + \
                        self.cfg.debris_cluster_config.C_rtn_p_ranges[0][1]**2, 0, 0],
                    [0, self.cfg.debris_cluster_config.C_rtn_s_ranges[1][1]**2 +\
                        self.cfg.debris_cluster_config.C_rtn_p_ranges[1][1]**2, 0],
                    [0, 0, self.cfg.debris_cluster_config.C_rtn_s_ranges[2][1]**2 +\
                        self.cfg.debris_cluster_config.C_rtn_p_ranges[2][1]**2]
                ])


                # Min-Max scale to into [–5,+5]
                # precompute their inv-sqrt‐dets
                _, logdet_min = np.linalg.slogdet(C_min)
                _, logdet_max = np.linalg.slogdet(C_max)
                
                norm_min = np.exp(-0.5 * logdet_min)
                norm_max = np.exp(-0.5 * logdet_max)
                
                C = self.DebrisSwarm_1.C_eci_combined[i]   
                _, logdet = np.linalg.slogdet(C)
                
                inv_sqrt_det = np.array([np.exp(-0.5 * logdet)])
                det_cov_scaled = 10 * (inv_sqrt_det - norm_min) / (norm_max - norm_min) - 5

                # Same for collision radius
                combined_radius = self.DebrisSwarm_1.radius_combined[i]
                combined_radius = 10 * (combined_radius - self.cfg.debris_cluster_config.radius_combined_ranges[0])\
                    / (self.cfg.debris_cluster_config.radius_combined_ranges[1] - self.cfg.debris_cluster_config.radius_combined_ranges[0]) - 5
                combined_radius = np.array([combined_radius]).reshape(1,)
                
                # Convert from minutes to hours
                tca_till = tca_till/60 

                l_s = np.array([nsoe[1]]) # range 0 - 6 no need to normalize
                inc_s = np.array([nsoe[4]]) # range 0 - 6 no need to normalize
                raan_s = np.array([nsoe[5]]) # range 0 - 6 no need to normalize

                obs_debris[i,:] = np.concatenate((
                    l_s,
                    inc_s,
                    raan_s,
                    tca_till,
                    det_cov_scaled,
                    combined_radius,
                    p_max_at_tca
                    ), axis=0)
                
                obs_debris_minimal[i,:] = np.concatenate((
                    tca_till,
                    p_max_at_tca
                    ), axis=0)

        obs_debris = np.array(obs_debris)
                        
        obs = np.concatenate([obs_satellite.ravel(), obs_debris.ravel()], axis=0)
        
        return obs
    

    def _reward_fun(self
                    )->Tuple[float,bool]:

        reward = 0
        terminated = False
        self.cost = 0
        
        roe = np.array(self.DebrisSwarm_1.roe)
        adl = roe[-1][1] # reference AOL
        ade_vector = np.array([roe[-1][2],roe[-1][3]]) # reference ecc vector
        ade = np.linalg.norm(ade_vector) # reference ecc norm             
        
        P_max_propagated = np.array(self.DebrisSwarm_1.metrics_at_tca)
        P_max_product = abs(1 - np.prod(1 - P_max_propagated[:,1]))        

        # Recovery phase rewards
        if P_max_product == 0: #No more debris, P_max list empty
            if  ade<=self.cfg.ade_norm_req and abs(adl)<=self.cfg.adl_req:
                reward = +1e3
                terminated = True
            else:           
                adl_based_weight = (abs(adl)/self.cfg.adl_req)
                ade_based_weight = (ade/self.cfg.ade_norm_req)

                reward = (-adl_based_weight -ade_based_weight)/3
        # CAM phase rewards
        else:
            reward = -np.log(P_max_product) -9 
        
        # Check if collision occured 
        for i in range(self.DebrisSwarm_1.no_debris):
            # Check at TCA if a collision had a P_max greater than the limit
            # If it fails, except index error, then it means collision hasn't occurred yet
            try:
                conjuction_time = self.DebrisSwarm_1.conjuction_points_time[i]
                p_max = self.DebrisSwarm_1.primary_sat_and_debris_mahala[1+i][conjuction_time][1] 

                if p_max>=self.cfg.p_max_limit: # Propagation index
                    reward = -100
                    terminated = True
                    break
            except IndexError:
                # collision hasn't occurred yet
                pass
            
            if self.DebrisSwarm_1.p_max_predictions[-1]<self.cfg.p_max_limit and P_max_product!=0:
                # reward = -self.delta_t0_and_man[1]*abs(self.f_direction)/2
                pass
        
        reward = reward 
        
        return reward, terminated
                
                
    def _discretize_action(self, 
                           action:int)->list:
        

        dis_actions_list = [[0,0],
                            [0,+1], 
                            [0,-1]]

        return dis_actions_list[action]
    
    def _clip_and_rescale(self, x, low, high):
        x = np.asarray(x, dtype=np.float32)
        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)

        x = np.clip(x, -1.0, 1.0)
        y = low + (x + 1.0) * (high - low) / 2.0

        return np.clip(y, low, high)


    def step(self, action):
        
        delay_duration_thrust:np.ndarray = action["continuous"]
        delay_duration_thrust = self._clip_and_rescale(delay_duration_thrust, self.cfg.low_action, self.cfg.high_action)


        self.f_direction = self._discretize_action(int(action["discrete"]))
        manplan = self.f_direction + delay_duration_thrust.tolist()
        self.delta_t0_and_man = delay_duration_thrust 
                
        # Propagate and apply manplan 
        self.DebrisSwarm_1.apply_manplan(manplan=manplan)
                
        # Calculate reward
        reward, terminated = self._reward_fun()        
        truncated = False 
        
        # Truncate if episode time over
        if (self.DebrisSwarm_1.n) >= self.cfg.max_time_index:
            truncated = True
            
        info ={"cost": self.cost,
               "sojourn_t": sum(self.delta_t0_and_man)/60, #Convert to hours
               "n": self.DebrisSwarm_1.n} # Discrete time in simulation (real time = dt*index)
        
        self.rewards_plot_list.append(reward)
        self.n_plot_list.append(self.DebrisSwarm_1.n)
        
        return self._observation_states(), reward, terminated, truncated, info



    def reset(self, seed=None, options=None):
        super().reset(seed=None)
                                
        self.DebrisSwarm_1 = SatDebrisCluster(self.cfg.debris_cluster_config)
        self.rewards_plot_list = []
        self.n_plot_list = []
        
        info ={"sat": self.DebrisSwarm_1,
               "reward": 0,
               "cost":0,
               "sojourn_t": 0,
               "n":0}


        return self._observation_states(), info

    def close(self):
        pass
    
    
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

    
    def plot_states_interactive(self)->None:
        
        # If you’re in a Jupyter notebook and need MathJax for LaTeX:
        from IPython.display import (HTML, display,)

        display(HTML(
        '<script type="text/javascript" async '
        'src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js'
        '?config=TeX-MML-AM_SVG"></script>'
        ))

        fig = make_subplots(
            rows=6, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.03,    
            column_widths=[0.7, 0.3],    

            specs=[
                [{"type": "xy"}, {"type": "scene", "rowspan": 6}],
                [{"type": "xy"}, None],
                [{"type": "xy"}, None],
                [{"type": "xy"}, None],
                [{"type": "xy"}, None],
                [{"type": "xy"}, None],
            ]
        )
        
        simulation_times=self.n_plot_list
        
        rewards = self.rewards_plot_list
        
        roe = np.array(self.DebrisSwarm_1.roe)
        ada1 = roe[:,0]
        adl1 = roe[:,1]
        ade1 = np.linalg.norm(roe[:, [2, 3]], axis=1)
        simulation_times = np.array(simulation_times)
        rewards = np.array(rewards)
        controls = np.array(self.DebrisSwarm_1.controls_RTN)
        R, T, N = controls[:,0], controls[:,1], controls[:,2]

        fig.add_trace(go.Scatter(y=ada1, mode='lines'), row=1, col=1)
        fig.update_yaxes(title_text=r"$a\delta a\ (m)$", row=1, col=1)

        fig.add_trace(go.Scatter(y=adl1, mode='lines'), row=2, col=1)
        fig.update_yaxes(title_text=r"$a\delta\lambda\ (m)$", row=2, col=1)

        fig.add_trace(go.Scatter(y=ade1, mode='lines'), row=3, col=1)
        fig.update_yaxes(title_text=r"$\|\!a\,\delta\mathbf e\|\ (m)$", row=3, col=1)

        fig.add_trace(go.Scatter(x=simulation_times, y=rewards, mode='markers'), row=4, col=1)
        fig.update_yaxes(title_text="Reward", row=4, col=1)

        fig.add_trace(go.Scatter(y=R, mode='markers', name='Radial'), row=5, col=1)
        fig.update_yaxes(title_text="$f_r$(mN)", row=5, col=1)

        P_max = self.DebrisSwarm_1.p_max_predictions
        fig.add_trace(go.Scatter(x=simulation_times,y=np.log10(P_max), mode='markers', name='P^{max}_c'), row=6, col=1)
        fig.update_yaxes(title_text=r"$\log(P^\text{c}_\text{max})$", row=6, col=1)

        for t in simulation_times:
            for r in range(1, 7):
                fig.add_vline(
                    x=t,
                    row=r, col=1,          
                    line_width=1,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5
                )

        fig.update_xaxes(title_text="Discrete Time", row=6, col=1)

        for space_object in self.DebrisSwarm_1.primary_sat_and_debris_rvm:
            coords = np.array(space_object)
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:,0],
                    y=coords[:,1],
                    z=coords[:,2],
                    mode='markers',
                    marker=dict(size=5)
                ),
                row=1, col=2
            )

        fig.update_layout(
            scene=dict(
                xaxis_title='X (km)',
                yaxis_title='Y (km)',
                zaxis_title='Z (km)',
                aspectmode='auto'
            ),
            height=900,
            width=1200,       
            showlegend=False,
            title_text=""
        )

        fig.show()

        
        
        return


