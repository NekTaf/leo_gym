# Standard library
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple
import logging

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
import seaborn as sns

# Local
from leo_gym.satellite.sat_debris_cluster import (
    SatDebrisCluster,
    SatDebrisClusterConfig,
)
from leo_gym.utils.matplot_style_cfg import *
from leo_gym.utils.utils import *
from pathlib import Path
import json 

gym.logger.set_level(40)


class CamEnvConfig(BaseModel):
    """
    Config for CAM environment
    """
    high_action: Any = Field(..., description="Upper bounds for each action dimension")
    low_action: Any  = Field(..., description="Lower bounds for each action dimension")
    debris_cluster_config: SatDebrisClusterConfig = Field(..., description="Configuration for debris cluster dynamics object")
    p_max_limit: float = Field(..., gt=0, description="Max thrust limit")
    adl_req: float = Field(..., ge=0, description="ADL requirement")
    ade_norm_req: float = Field(..., ge=0, description="ADE normalization")
    max_time_index: int = Field(..., ge=1, description="Max real time per episode")
    reduced_obs: bool = Field(False, description="Reduce observation size based on SHAP analysis")
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    


class CamEnv(gym.Env):
    """
    Satellite-Debris gym environemnt for training LEO collision acoidance policies.

    Notes:
        - Subscript `_p` denotes the primary satellite.
        - Subscript `_s` denotes secondary debris objects.
        - TCA (Time of Closest Approach) is the instant at which a collision
        would occur.
        - Data are managed through the `SatDebrisCluster` object, which stores
        satellite and debris data and provides propagation functionality.
        - A configuration function is required to define the satellite and
        debris parameters. In this example, the parameters are assumed to be
        identical.
    """

    def __init__(
        self, 
        cfg:CamEnvConfig|str|Path,
        seed:int):
        
        super(CamEnv, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Intialzing CAM environment")

        self.load_cfg(cfg)
        seed_all(seed)        
        self.reset()
        
        
        self.action_space = spaces.Dict(
            {
            "discrete": spaces.Discrete(3),
            "continuous": spaces.Box(
                low=np.array(self.cfg.low_action), 
                high=np.array(self.cfg.high_action), 
                shape=(2,), 
                dtype=np.float64)
            }
        )
                    
        if self.cfg.reduced_obs:
            self.logger.info("Using Reduced Observations")
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4 + self.cfg.debris_cluster_config.max_debris * 4,),
                dtype=np.float64
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(int(6+self.cfg.debris_cluster_config.max_debris*9),),
                dtype=np.float64
            )
        
    
    def _observation_states(self)->NDArray[np.float64]:
        # Relative orbital elements between satellite and unperturbated trajectory
        roe = np.array(self.DebrisSwarm_1.roe)
        
        # Primary satellite non_singular_oe
        non_singular_oe_p = np.array(self.DebrisSwarm_1.nsoe_sat)[-1].reshape(6,) 
        
        # Secondary debvris object values non_singular_oe
        non_singular_oe_s = np.array([nsoe[-1] for nsoe in self.DebrisSwarm_1.nsoe_debris])
        
        # convert to km to normalize // range: 0 - -/+5
        adl = roe[-1][1]/1e3 
        
        # convert to km to normalize // range: 0 - -/+ 9
        adex = roe[-1][2]/1e2 
        
        # convert to km to normalize // range 0 - -/+ 9
        adey = roe[-1][3]/1e2 
        
        # range 0 - 6 no need to normalize
        u_p = non_singular_oe_p[1] 
        
        # range 0 - 6 no need to normalize
        inc_p = non_singular_oe_p[4]
        
        #range 0 - 6 no need to normalize
        raan_p = non_singular_oe_p[5] 

        if not self.cfg.reduced_obs:
            obs_debris = np.zeros((self.DebrisSwarm_1.cfg.max_debris,9))

            # Observations of primary satellite 
            obs_satellite = np.array(
                (
                    u_p,
                    inc_p,
                    raan_p,
                    adl,
                    adex, 
                    adey
                )
            )
        else:
            obs_debris = np.zeros((self.DebrisSwarm_1.cfg.max_debris, 4))

            obs_satellite = np.array(
                [
                    u_p,
                    adl,
                    adex,
                    adey
                ]
            )        
        
        latest_metrics = self.DebrisSwarm_1.metrics_at_tca[-1]

        for i, nsoe in enumerate(non_singular_oe_s):

            tca_true = self.DebrisSwarm_1.conjuction_points_time[i]
            tca_till = np.array([tca_true - self.DebrisSwarm_1.dt_steps])

            if tca_till[0] < 0:
                continue

            metric = latest_metrics[i]

            p_max_at_tca = np.array([np.log(max(metric[1], 1e-12))])

            # Min-Max scale covariance determinant
            c_min = np.array(
                [
                    [self.cfg.debris_cluster_config.C_rtn_s_ranges[0][0]**2 +
                    self.cfg.debris_cluster_config.C_rtn_p_ranges[0][0]**2, 0, 0],
                    [0, self.cfg.debris_cluster_config.C_rtn_s_ranges[1][0]**2 +
                    self.cfg.debris_cluster_config.C_rtn_p_ranges[1][0]**2, 0],
                    [0, 0, self.cfg.debris_cluster_config.C_rtn_s_ranges[2][0]**2 +
                    self.cfg.debris_cluster_config.C_rtn_p_ranges[2][0]**2]
                ]
            )

            c_max = np.array(
                [
                    [self.cfg.debris_cluster_config.C_rtn_s_ranges[0][1]**2 +
                    self.cfg.debris_cluster_config.C_rtn_p_ranges[0][1]**2, 0, 0],
                    [0, self.cfg.debris_cluster_config.C_rtn_s_ranges[1][1]**2 +
                    self.cfg.debris_cluster_config.C_rtn_p_ranges[1][1]**2, 0],
                    [0, 0, self.cfg.debris_cluster_config.C_rtn_s_ranges[2][1]**2 +
                    self.cfg.debris_cluster_config.C_rtn_p_ranges[2][1]**2]
                ]
            )

            _, logdet_min = np.linalg.slogdet(c_min)
            _, logdet_max = np.linalg.slogdet(c_max)

            norm_min = np.exp(-0.5 * logdet_min)
            norm_max = np.exp(-0.5 * logdet_max)

            c_eci_comb = self.DebrisSwarm_1.c_eci_combined[i]
            _, logdet = np.linalg.slogdet(c_eci_comb)

            inv_sqrt_det = np.array([np.exp(-0.5 * logdet)])
            det_cov_scaled = 10 * (inv_sqrt_det - norm_min) / (norm_max - norm_min) - 5

            combined_radius = self.DebrisSwarm_1.radius_combined[i]
            combined_radius = 10 * (
                combined_radius - self.cfg.debris_cluster_config.radius_combined_ranges[0]
            ) / (
                self.cfg.debris_cluster_config.radius_combined_ranges[1]
                - self.cfg.debris_cluster_config.radius_combined_ranges[0]
            ) - 5
            combined_radius = np.array([combined_radius])

            tca_till = tca_till / 60

            u_s = np.array([nsoe[1]])
            inc_s = np.array([nsoe[4]])
            raan_s = np.array([nsoe[5]])

            delta_r_b = np.asarray(metric[3:5]) / 5e2

            assert_no_nan(u_s, "u_s")
            assert_no_nan(inc_s, "inc_s")
            assert_no_nan(raan_s, "raan_s")
            assert_no_nan(tca_till, "tca_till")
            assert_no_nan(det_cov_scaled, "det_cov_scaled")
            assert_no_nan(combined_radius, "combined_radius")
            assert_no_nan(p_max_at_tca, "p_max_at_tca")
            assert_no_nan(delta_r_b, "delta_r_b")

            if not self.cfg.reduced_obs:
                obs_debris[i, :] = np.concatenate(
                    (
                        u_s,
                        inc_s,
                        raan_s,
                        tca_till,
                        det_cov_scaled,
                        combined_radius,
                        p_max_at_tca,
                        delta_r_b
                    )
                )
            else:
                obs_debris[i, :] = np.concatenate(
                    (
                        tca_till,
                        p_max_at_tca,
                        delta_r_b
                    )
                )
        obs_debris = np.array(obs_debris)
        
        obs = np.concatenate([obs_satellite.ravel(), obs_debris.ravel()], axis=0)
        
        return obs
    

    def _get_reward(self)->Tuple[float,bool]:
        reward = 0
        terminated = False
        self.cost = 0
        
        roe = np.array(self.DebrisSwarm_1.roe)
        adl = roe[-1][1] # reference AOL
        ade_vector = np.array([roe[-1][2],roe[-1][3]]) # reference ecc vector
        ade = np.linalg.norm(ade_vector) # reference ecc norm             
        
        P_max_propagated = np.array(self.DebrisSwarm_1.metrics_at_tca[-1])
        P_max_propagated = P_max_propagated[[x is not None for x in P_max_propagated[:, 1]]]

        P_max_product = abs(1 - np.prod(1 - P_max_propagated[:, 1]))     
        
        # Collsiion avoidance phase rewards
        # Check if Debris exist
        if self.DebrisSwarm_1.dt_steps<np.max(self.DebrisSwarm_1.conjuction_points_time):
            # No collision risk
            if P_max_product<=self.cfg.p_max_limit:
                self.logger.debug("No Collision Risk")
                self.logger.debug("reward: %s", reward)
                reward += +10 -abs(self.delta_t0_and_man[1]*np.sign(self.f_direction[1]))
            
            # Collision risk
            else: 
                reward += np.minimum(10,-np.log(P_max_product)-15)

        else: 
            # Check if collisions have occured
            for i in range(self.DebrisSwarm_1.num_debris):
                conjunction_time = self.DebrisSwarm_1.conjuction_points_time[i]
                p_max = self.DebrisSwarm_1.metrics_at_tca[-1][i][1]
                
                if conjunction_time<=self.DebrisSwarm_1.dt_steps:
                    
                    idx = next(
                        y for y in range(len(self.DebrisSwarm_1.metrics_at_tca))
                        if self.DebrisSwarm_1.metrics_at_tca[y][i][5] == conjunction_time
                    )

                    p_max_tca = self.DebrisSwarm_1.metrics_at_tca[idx][i][1] 

                    if p_max_tca >= self.cfg.p_max_limit:
                        reward = -100
                        terminated = True

                        self.logger.debug("Collision occurred")
                        self.logger.debug("P_max = %s", p_max)
                        self.logger.debug("reward: %s", reward)
                        
                        return reward, terminated
        
            if ade <= self.cfg.ade_norm_req and abs(adl) <= self.cfg.adl_req:
                reward = +1e3
                terminated = True
                
                return reward, terminated
            
            else:
                adl_based_weight = abs(adl) / self.cfg.adl_req
                ade_based_weight = ade / self.cfg.ade_norm_req

                reward = (-adl_based_weight - ade_based_weight) / 3

            self.logger.debug("Recovery")
            self.logger.debug("limit = %s", self.cfg.p_max_limit)
            self.logger.debug("P_max_product = %s", P_max_product)
            self.logger.debug("reward: %s", reward)

        self.logger.debug("CAM")
        self.logger.debug("limit = %s", self.cfg.p_max_limit)
        self.logger.debug("P_max_product = %s", P_max_product)
        self.logger.debug("reward: %s", reward)

        return reward, terminated
    
    def _discretize_action(self, action:int)->list:
        dis_actions_list = [
            [0,0],
            [0,+1], 
            [0,-1]
        ]

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
        
        P_max_propagated = np.array(self.DebrisSwarm_1.metrics_at_tca[-1])
        P_max_product = abs(1 - np.prod(1 - P_max_propagated[:,1]))      
                
        manplan = self.f_direction + delay_duration_thrust.tolist()
        self.delta_t0_and_man = delay_duration_thrust 
                
        # Propagate and apply manplan 
        self.DebrisSwarm_1.apply_manplan(manplan=manplan)
                
        # Calculate reward
        reward, terminated = self._get_reward()        
        truncated = False 
        
        # Truncate if episode time over
        if (self.DebrisSwarm_1.dt_steps) >= self.cfg.max_time_index:
            truncated = True
            
        info ={
            "cost": self.cost,
            "sojourn_t": sum(self.delta_t0_and_man)/60, #Convert to hours
            "n": self.DebrisSwarm_1.dt_steps
        } # Discrete time in simulation (real time = dt*index)
        
        self.rewards_plot_list.append(reward)
        self.n_plot_list.append(self.DebrisSwarm_1.dt_steps)
        
        self.logger.debug("\n====================")            
        return self._observation_states(), reward, terminated, truncated, info



    def reset(self, seed=None, options=None):
        super().reset(seed=None)
        
        self.DebrisSwarm_1 = SatDebrisCluster(self.cfg.debris_cluster_config)
        self.rewards_plot_list = []
        self.n_plot_list = []
        
        info = {
            "sat": self.DebrisSwarm_1,
            "reward": 0,
            "cost":0,
            "sojourn_t": 0,
            "n":0
        }

        return self._observation_states(), info

    def close(self):
        pass
    
    
    def plot_rewards(self, save_path:str)->None:
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
        controls = np.array(self.DebrisSwarm_1.forces_rtn)
        R, T, N = controls[:,0], controls[:,1], controls[:,2]

        fig.add_trace(go.Scatter(y=ada1, mode='lines'), row=1, col=1)
        fig.update_yaxes(title_text=r"$a\delta a\ [m]$", row=1, col=1)

        fig.add_trace(go.Scatter(y=adl1, mode='lines'), row=2, col=1)
        fig.update_yaxes(title_text=r"$a\delta\lambda\ [m]$", row=2, col=1)

        fig.add_trace(go.Scatter(y=ade1, mode='lines'), row=3, col=1)
        fig.update_yaxes(title_text=r"$\|\!a\,\delta\mathbf e\|\ [m]$", row=3, col=1)

        fig.add_trace(go.Scatter(x=simulation_times, y=rewards, mode='markers'), row=4, col=1)
        fig.update_yaxes(title_text="Reward", row=4, col=1)

        fig.add_trace(go.Scatter(y=R, mode='markers', name='Radial'), row=5, col=1)
        fig.update_yaxes(title_text="$f_r$(mN)", row=5, col=1)

        P_max = np.array([
            m[1] for m in self.DebrisSwarm_1.metrics_at_tca
        ])

        times = np.array([
            m[-1] for m in self.DebrisSwarm_1.metrics_at_tca
        ])
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=np.log10(np.maximum(P_max, 1e-12)),
                mode="markers"
            ),
            row=6,
            col=1,
        )
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

        for space_object in self.DebrisSwarm_1.sat_debris_rvm:
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
    
    
    def plot_interactive_collision_close_up(self):

        fig = go.Figure()

        for i in range(self.DebrisSwarm_1.num_debris):
            states = np.array(self.DebrisSwarm_1.sat_debris_rvm)

            x = states[0][self.DebrisSwarm_1.conjuction_points_time[i]-10:self.DebrisSwarm_1.conjuction_points_time[i]+10, 0]
            y = states[0][self.DebrisSwarm_1.conjuction_points_time[i]-10:self.DebrisSwarm_1.conjuction_points_time[i]+10, 1]
            z = states[0][self.DebrisSwarm_1.conjuction_points_time[i]-10:self.DebrisSwarm_1.conjuction_points_time[i]+10, 2]
            
            
            x_deb = states[i+1][self.DebrisSwarm_1.conjuction_points_time[i]-10:self.DebrisSwarm_1.conjuction_points_time[i]+10, 0]
            y_deb = states[i+1][self.DebrisSwarm_1.conjuction_points_time[i]-10:self.DebrisSwarm_1.conjuction_points_time[i]+10, 1]
            z_deb = states[i+1][self.DebrisSwarm_1.conjuction_points_time[i]-10:self.DebrisSwarm_1.conjuction_points_time[i]+10, 2]

            
            if i != 0:
                color = "red"
            else:
                color = None

            fig.add_trace(go.Scatter3d(x=x, y=y, z=z,  marker=dict(color="blue")))
            fig.add_trace(go.Scatter3d(x=x_deb, y=y_deb, z=z_deb,  marker=dict(color="red")))

            radius = 2.5e5

            conjuction_point=self.DebrisSwarm_1.sat_debris_rvm[0][self.DebrisSwarm_1.conjuction_points_time[i]]

            x = conjuction_point[0]
            y = conjuction_point[1]
            z = conjuction_point[2]

            phi, theta = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]

            xx = radius*np.cos(theta)*np.sin(phi) + x
            yy = radius*np.sin(theta)*np.sin(phi) + y
            zz = radius*np.cos(phi) + z

            fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=0.4, showscale=False, colorscale=[[0, 'gray'], [1, 'gray']]))

        fig.update_layout(scene=dict(
                            xaxis_title='X',
                            yaxis_title='Y',
                            zaxis_title='Z',
                            aspectmode='data'),
                            width=800*1.2, height=800)

        fig.update_layout(scene_camera=dict(eye=dict(x=-1.25, y=-1.25, z=0.7)))

        fig.show()
        
    
    
    def publication_ready_plots(self, save_path:str|Path|None)->None:
        
        roe = np.array(self.DebrisSwarm_1.roe)

        # relative longitude and eccentricity plots
        adl = roe[:,1]
        ade = np.sqrt(roe[:,2]**2 + roe[:,3]**2)

        fig, axs = plt.subplots(2, 1, sharex=True,figsize=(8, 4))

        axs[0].plot(adl)
        axs[0].axhline(self.cfg.adl_req,label=r"$|a \delta \lambda|_\mathrm{req}$ [m]",color="red")
        axs[0].axhline(-self.cfg.adl_req, color="red")
        axs[0].set_ylabel(r"$a \delta \lambda$ [m]")


        axs[1].plot(ade)
        axs[1].axhline(self.cfg.ade_norm_req, label=r"$\|a \delta \mathbf{e}\|_\mathrm{req}$ [m]",color="red")
        axs[1].set_ylabel(r"$\|a \delta \mathbf{e}\|$ [m]")

        axs[0].legend()
        axs[1].legend()
        
        axs[1].set_xlabel("Minutes")

        
        
        # Radial thrust plot
        fig4, ax = plt.subplots(figsize=(8, 2))

        rtn = np.asarray(self.DebrisSwarm_1.forces_rtn)
        ax.plot(rtn[:, 0])
        
        ax.set_xlabel("Minutes")
        ax.set_ylabel("$f_r$ [N]")

        plt.show()
        
        
        
        fig3, ax = plt.subplots(figsize=(8, 2))

        times = np.array([
            m[0][5] for m in self.DebrisSwarm_1.metrics_at_tca
        ]) * self.cfg.debris_cluster_config.dt / 60
                
        for debris_idx in range(self.DebrisSwarm_1.num_debris):

            pmax = np.array([
                m[debris_idx][1]
                for m in self.DebrisSwarm_1.metrics_at_tca
            ])

            ax.plot(
                times,
                np.log10(np.maximum(pmax, 1e-6)),
                label=f"Debris {debris_idx}",
            )

        pmax_product = np.array([
            abs(1 - np.prod(1 - np.array([metric[1] for metric in m])))
            for m in self.DebrisSwarm_1.metrics_at_tca
        ])

        ax.plot(
            times,
            np.log10(np.maximum(pmax_product, 1e-6)),
            color="black",
            linewidth=2,
            label=r"$P_c^{\max,\mathrm{cum}}$",
        )

        ax.axhline(
            np.log10(self.cfg.p_max_limit),
            color="red",
            linestyle="--",
            label=r"$\log_{10}(P^\mathrm{max}_{c,\mathrm{req}})$",
        )

        ax.set_xlabel("Minutes")
        ax.set_ylabel(r"$\log_{10}(P_c^{\max})$")
        ax.legend()                
        
        
        
        # 3D Plot
        fig2 = plt.figure(figsize=(4, 4), constrained_layout=True)
        ax = fig2.add_subplot(111, projection="3d")

        fig2.patch.set_facecolor("white")
        ax.xaxis.set_pane_color((0.92, 0.92, 0.92, 1))
        ax.yaxis.set_pane_color((0.92, 0.92, 0.92, 1))
        ax.zaxis.set_pane_color((0.92, 0.92, 0.92, 1))
        ax.patch.set_facecolor("none")

        prim = np.asarray(self.DebrisSwarm_1.rvm_sat)

        ax.scatter(
            prim[:, 0], prim[:, 1], prim[:, 2],
            s=5,
            label="Satellite"
        )

        for i, sec in enumerate(self.DebrisSwarm_1.cached_rvm_debris):

            sec = np.asarray(sec)

            # Last cached point is the TCA
            x_cp, y_cp, z_cp = sec[-1, :3]

            if i == 0:
                ax.scatter(
                    x_cp, y_cp, z_cp,
                    s=3,
                    label="Conjunction Point", color="black"
                )

                ax.scatter(
                    x_cp, y_cp, z_cp,
                    s=80,
                    color="black",
                )
                ax.scatter(
                    sec[:, 0], sec[:, 1], sec[:, 2],
                    s=3,
                    color="red",
                    label="Debris"
                )
            else:
                ax.scatter(x_cp, y_cp, z_cp, s=80, color="black")
                ax.scatter(sec[:, 0], sec[:, 1], sec[:, 2], s=3, color="red")

        ax.set_xlabel("$X$ [m]")
        ax.set_ylabel("$Y$ [m]")
        ax.set_zlabel("$Z$ [m]")

        ax.set_box_aspect(None, zoom=0.80)
        ax.grid(False)
        ax.legend(markerscale=3)                
                

        return

    
    
    def load_cfg(self,cfg:str|CamEnvConfig|Path):
        
        if isinstance(cfg, (str, os.PathLike, Path)):
            with open(cfg, "r") as f:
                data = json.load(f)
            self.cfg = CamEnvConfig(**data)
        else:
            self.cfg = cfg
    


