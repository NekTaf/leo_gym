"""
Creates satellite and n number of debris for conjuction event at time t randomly 

The data for the relative orbital elements for each of the space objects
with each other is saved in a dictionary containing lists 

The propagation ends at the choosen TCA conjuction point
The probability of collision between the debris and the sat is calculated 
This will be fed back to the environment for training 
    
For simplicity, it is assumed that the satellites and debris have all the same parameters,
Weight, area and size.

However, the uncertainty/variance of different obstacles will be variable, and will need to be taken into account by the RL.
This will vary between a few meters to a few Km. 

To investigate:
    + How much time to before TCA to apply correction manuevers 


"""

# Standard library
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from zendir.maths.astro import (
    classical_to_non_singular_elements,
    vector_to_classical_elements,
    vector_to_relative_elements_mean,
)

# Local
from leo_gym.orbit.dynamics.collision import *
from leo_gym.orbit.dynamics.conversions import *
from leo_gym.orbit.dynamics.dynamics import Dynamics, DynamicsConfig
from leo_gym.orbit.dynamics.init_weather_mjd import load_spice_kernels
from leo_gym.satellite.satellite_base import Satellite
from leo_gym.utils.utils import *


class SatDebrisClusterConfig(BaseModel):
    """
    Pydantic model for configuring a debris cluster simulation.
    """
    max_debris: int = Field(..., ge=1, description="Maximum number of debris")
    min_debris: int = Field(..., ge=0, description="Minimum number of debris; must be <= max_debris")
    params_dyn: DynamicsConfig = Field(..., description="cfg for real dynamics")
    params_dyn_ideal: DynamicsConfig = Field(..., description="cfg for ideal dynamics")
    dt: float = Field(..., gt=0, description="simulation discretization in seconds")
    days: float = Field(..., gt=0, description="simulation duration in days")
    conjunction_time_window_index: List[int] = Field(...,
                                                    min_items=2,max_items=2,
                                                    description="[min, max] time window for conjunction checks",)
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)



class SatDebrisCluster():
    def __init__(self,
                 cfg:SatDebrisClusterConfig):
        
        self.cfg = cfg
                
        self.reset_states()
        load_spice_kernels()
        
        
    def DCM_eci2rtn(self,
                    r:NDArray,
                    v:NDArray)->NDArray:
        
        r = r.reshape(3,)
        v = v.reshape(3,)
        
        return self.dynamics1._DCM_eci2rtn(r,v)

    def init_debris_objects(self
                            )->None:
        """Initialize debris objects
        """
        
        # Initialize lists for parameter storage of each debris
        self.conjuction_points_time:list[int] = []
        self.C_rtn_s:list[np.ndarray] = []
        self.C_eci_combined:list[np.ndarray] = []
        self.radius_combined:list[float] = []
        self.no_debris:int = 1
        self.dt_col:int = 60

        if self.cfg.max_debris != 1:
            self.no_debris = np.random.randint(low=self.cfg.min_debris, high=self.cfg.max_debris+1, dtype=int)

        
        # Initialize primary satellite covariance
        self.C_rtn_p:np.ndarray = np.array([[np.random.uniform(10,50)**2, 0, 0], 
                                [0, np.random.uniform(75,100)**2, 0],
                                [0, 0, np.random.uniform(10,50)**2]])

        
        for i in range(self.no_debris):
            
            self.conjuction_time:int = int(np.random.randint(low=self.cfg.conjunction_time_window_index[0],
                                                         high=self.cfg.conjunction_time_window_index[1],
                                                         dtype=int)*60/self.cfg.dt)
            
            self.conjuction_points_time.append(self.conjuction_time)
            pvm_col0 = collision_generator(rv0=np.array(self.all_object_rvm0[0][:6]),
                                    dt = self.dt_col, 
                                    days=self.cfg.days,
                                    relative_t_tca=int(self.conjuction_time*self.cfg.dt/self.dt_col),
                                    params_dyn=self.cfg.params_dyn,
                                    )
            
            self.all_object_rvm0.append(pvm_col0.tolist())

            # Initialize secondary debris object covariance
            self.C_rtn_s.append(np.array([[np.random.uniform(150,250)**2, 0, 0], 
                                        [0, np.random.uniform(250,450)**2, 0],
                                        [0, 0, np.random.uniform(150,250)**2]]))
            
            rv_p:np.ndarray = np.array(self.all_object_rvm0[0])
            rv_s:np.ndarray = np.array(self.all_object_rvm0[-1])
            
            _, C_eci_combined_individual = covariance_converisions(rv_p=rv_p,
                                                                    rv_s=rv_s,
                                                                    C_rtn_p=self.C_rtn_p,
                                                                    C_rtn_s=self.C_rtn_s[i])
            
            # Append combined covariance and combined radius
            self.C_eci_combined.append(C_eci_combined_individual)
            
            # self.radius_combined.append(100)
            self.radius_combined.append(np.random.uniform(85, 145))
            
        return

    def init_ideal_traj(self
                        )->None:
        """Calculate relative orbital elements for station keeping
        """
        self.ideal_traj_rvm.append(self.all_object_rvm0[0]) 
        doe, oe, _ = rv_2_roe_and_non_singular_oe(np.array(self.all_object_rvm0[0][:6]), 
                                            np.array(self.ideal_traj_rvm[-1][:6]))
        self.doe.append(doe)
        self.oe_ns.append(oe)
        
        return

    def calculate_non_singular_oe(self, 
                                  rv:NDArray
                                  )-> NDArray:
        """Calculates non singular orbital elements for circular orbits

        Args:
            rv (arr): size (6,), rv posiiton velocity in ECI 

        Returns:
            arr: (6,) oe_ns (a, AOL, e_x, e_y, i, RAAN), Non-singular orbital elements for non-defined eccentricity in circular orbits
        """
                
        rv = np.array(rv)
        
        oe_kep = vector_to_classical_elements(r_bn_n=rv[:3], 
                                            v_bn_n=rv[-3:], 
                                            planet = 'earth')
        
        oe_ns = classical_to_non_singular_elements(semi_major_axis=oe_kep[0],
                                                    eccentricity=oe_kep[1],
                                                    inclination=oe_kep[2],
                                                    right_ascension=oe_kep[3],
                                                    argument_of_periapsis=oe_kep[4],
                                                    true_anomaly=oe_kep[5])
        
        oe_ns = np.array([oe_ns[0],
                        oe_ns[5],
                        oe_ns[1],
                        oe_ns[2],
                        oe_ns[3],
                        oe_ns[4]]).reshape(6,)

        return oe_ns
   
    def get_projected_collision_metrics(self
                        )->list[NDArray]:
        """Propagate to TCA and return P_c max for each debris
        """
        primary_rvm:list[NDArray] = []        # Dummy variable, do not append to main 
        object_collision_metrics:list[NDArray] = []      # Store mahala outputs 
        u = np.zeros(3)

        for i, (object_rvm) in enumerate((self.primary_sat_and_debris_rvm)):
            if i == 0:
                primary_rvm.append(self.primary_sat_and_debris_rvm[0][-1])
                max_time_till_tca = np.max(np.array(self.conjuction_points_time)-self.n) + 1
                
                for _ in range(max_time_till_tca):
                    primary_rvm.append(self.dynamics1.propagate(x=primary_rvm[-1],
                                        u=u,t=self.t[self.n],
                                        dt=self.cfg.dt))
                          
            if i!=0:
                conjuction_time = self.conjuction_points_time[i-1]
                time_till_conjuction_time = conjuction_time - self.n + 1

                rvm = object_rvm[-1]

                if time_till_conjuction_time < 0:
                    object_collision_metrics.append(np.zeros((5,)))
                    continue
                
                for _ in range(time_till_conjuction_time):    
                    rvm = self.dynamics1.propagate(x=np.array(rvm),
                                                    u=u,
                                                    t=self.t[self.n],
                                                    dt=self.cfg.dt)
                    
                primary_rvm_at_tca = primary_rvm[time_till_conjuction_time][:6]
                
                rv_p = np.array(primary_rvm_at_tca).reshape(-1)
                rv_s = np.array(rvm[:6]).reshape(-1)
                        
                object_bplane = (delta_r_eci_2_rb(np.array(primary_rvm_at_tca).reshape(-1)
                                                            ,np.array(rvm[:6]).reshape(-1)))
                
                delta_r = np.array(object_bplane[:3])
                                    
                # Get combined covariance
                C_b_combined, C_eci_combined_individual = covariance_converisions(rv_p=rv_p,
                                                    rv_s=rv_s,
                                                    C_rtn_p=self.C_rtn_p,
                                                    C_rtn_s=self.C_rtn_s[i-1])
                
                object_collision_metrics.append(collision_metrics(delta_r_b=delta_r, 
                                                                l=self.radius_combined[i-1], 
                                                                cov_b=C_b_combined))
                
        return object_collision_metrics
        
    def propagate_sat_debris(self,
                             u:NDArray
                             )->None:
                
        self.C_eci_combined = []
        
        for i, (object_rvm,object_mahala,object_bplane,nsoe) in enumerate(zip(self.primary_sat_and_debris_rvm,
                                             self.primary_sat_and_debris_mahala,
                                             self.primary_sat_and_debris_b_plane,
                                             self.non_singular_oe)):
            
            # Apply control only to primary satellite
            if i != 0:  
                u = np.zeros(3) 
            
            # Objects rvm in ECI
            object_rvm.append(self.dynamics1.propagate(x=np.array(object_rvm[-1]),
                                                       u=u,t=self.t[self.n],dt=self.cfg.dt))  
            rv_p = np.array(self.primary_sat_and_debris_rvm[0][-1][:6]).reshape(-1) #primary satellite rvm
            rv_s = np.array(object_rvm[-1][:6]).reshape(-1) #secondary object rvm

            # Object bplane params
            object_bplane.append(delta_r_eci_2_rb(np.array(self.primary_sat_and_debris_rvm[0][-1][:6]).reshape(-1)
                                                    ,np.array(object_rvm[-1][:6]).reshape(-1)))
            
            # Secondary debris object non-singular params
            oe_non_singular = self.calculate_non_singular_oe(rv=rv_s)
            nsoe.append(oe_non_singular)
            
            # Position difference in B-plane
            delta_r_b = np.array(object_bplane[-1][:3])
            
            if i != 0:
                # Get combined covariance
                C_b_combined, C_eci_combined_individual = covariance_converisions(rv_p=rv_p,
                                                    rv_s=rv_s,
                                                    C_rtn_p=self.C_rtn_p,
                                                    C_rtn_s=self.C_rtn_s[i-1])
                
                self.C_eci_combined.append(C_eci_combined_individual)
                
                if np.isnan(delta_r_b).any():
                    object_mahala.append(np.zeros(5))
                else:
                    object_mahala.append(collision_metrics(delta_r_b=delta_r_b, 
                                                                    l=self.radius_combined[i-1], 
                                                                    cov_b=C_b_combined))

            # Save only satellite thrust commands 
            if i == 0:
                self.controls_RTN.append(self.dynamics1.u_thruster_rtn)


        self.ideal_traj_rvm.append(self.dynamics_ideal.propagate(x=np.array(self.ideal_traj_rvm[-1]),
                                                                 u=np.zeros(3),
                                                                 t=self.t[self.n],
                                                                 dt=self.cfg.dt))             

        doe, oe, oe_ref = rv_2_roe_and_non_singular_oe(np.array(self.primary_sat_and_debris_rvm[0][-1][:6]), 
                                            np.array(self.ideal_traj_rvm[-1][:6]))
        self.doe.append(doe)
        self.oe_ns.append(oe)
        
        # Update index
        self.n += int(np.sign(self.cfg.dt)*1)
        
        return

    def propagate_2_tca(self
                          )->None:
        """Propagate after maneuvers to evaluate TCA collision probability
        """

        self.metrics_at_tca = self.get_projected_collision_metrics()
        metrics_at_tca = np.array(self.metrics_at_tca)
        
        try: 
            P_max_product = 1 - np.prod(1 - metrics_at_tca[:,1])

            if P_max_product < 0:
                P_max_product = 0
                
            self.p_max_predictions.append(P_max_product)
            self.delta_r_b_plane.append(np.array([metrics_at_tca[:,3],
                                                  metrics_at_tca[:,4]]))
            
        except IndexError: # No more debris objects left
            self.p_max_predictions.append(0)
            self.delta_r_b_plane.append(np.zeros(2))
        
        return
                                           
    def apply_manplan(self, 
                      manplan:list)->None:
        """Applies manuever plan and updates satellite and debris object positions 

        Args:
            manplan (list): size(4,) thrust axis, thrust magnitude,
            maneuver start time, maneuver duration time \b
            
            manplan[0] = "rad", "tan", "norm" \b
            
        Returns:
            int: the new discrete time index after the manuever finishes executing 
        """
                
        self.manplans.append(manplan + [self.n*self.cfg.dt+
                                        self.cfg.params_dyn.eph_time_0])
                    
        thrust_magn = manplan[1]
        thrust_axis = manplan[0]
                
        firing_direction_vector = np.zeros(3)
        firing_direction_vector[thrust_axis] = thrust_magn        
        
        t_delay = int(manplan[2])
        t_duration = int(manplan[3])
        t_coast = 0
        
        self.manaplan_call_relative_time.append(self.n*self.cfg.dt)
        
        manplan_padded = np.zeros((t_delay+t_duration+t_coast,3)) 
        manplan_padded[t_delay:t_delay+t_duration,:] = firing_direction_vector[thrust_axis]
        self.manplan_control_inputs[self.n+t_delay:
            t_delay+t_duration+self.n,thrust_axis] = thrust_magn

        for _ in range(int(t_delay+t_duration+t_coast)):
            if self.n+1 >= self.traj_len:
                print("Ran out of data points. Set higher number of days!")
                return self.n
            self.propagate_sat_debris(u=self.manplan_control_inputs[self.n,:])
        
        self.propagate_2_tca()
        
    
    def plot_projected_position_bplane(self,
                            save_path:str=None)->None:
    
        data = np.array(self.delta_r_b_plane).squeeze()

        # Separate columns
        x = data[:, 0]
        y = data[:, 1]
        indices = np.arange(len(data))

        # Filter out 0,0
        mask = ~((x == 0) & (y == 0))
        x = x[mask]
        y = y[mask]
        indices = indices[mask]

        plt.figure(figsize=(6,6))
        scatter = plt.scatter(
            x, y,
            c=indices,
            cmap='viridis',
            marker='o'
        )

        lim = np.max(np.abs(np.concatenate([x, y])))
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.gca().set_aspect('equal', 'box')

        plt.xlabel(r'$\xi \mathrm{ (m)}$ ')
        plt.ylabel(r'$\zeta \mathrm{ (m)}$')
        plt.title('Projected Position B-Plane')
        plt.colorbar(scatter, label='Episode step index')
        plt.show()
    
        return

    def plot_doe_states(self,
                    save_path:str)->None:
        
        
        
        
        
        return
    
    def reset_states(self
                     )->None:
        
        self.n:int = 0 #discrete time index 
        self.traj_len:int = int(self.cfg.days*24*60*60/(np.sign(self.cfg.dt)*self.cfg.dt)) # discrete trajectory time length
        self.t:np.ndarray = np.linspace(0, self.traj_len * np.sign(self.cfg.dt)*self.cfg.dt, self.traj_len + 1) # time array for whole simulation

        self.all_object_rvm0:list = []
        self.ideal_traj_rvm:list = []
        self.doe:list = []
        self.oe_ns:list = []
        self.p_max_predictions:list = []
        self.delta_r_b_plane:list = []
        self.controls_RTN:list = []
        self.manplans:list = []
        self.manaplan_call_relative_time:list = []
        # For evaluating P_c max at TCA
        self.metrics_at_tca:list = []

        # init starting position and velocity vector
        self.all_object_rvm0.append(gen_rv0(sma=7573459).tolist()+[self.cfg.params_dyn.m])

        # init debris space objects
        self.init_debris_objects()
        self.init_ideal_traj()
        
        # initialize satellite and derbis lists  
        self.primary_sat_and_debris_rvm = [ [] for _ in range(self.no_debris+1)]
        self.primary_sat_and_debris_mahala = [ [] for _ in range(self.no_debris+1)]
        self.primary_sat_and_debris_b_plane = [ [] for _ in range(self.no_debris+1)]
        self.non_singular_oe = [ [] for _ in range(self.no_debris+1)]

        for i, (object_rvm,object_mahala,object_bplane,rvm0,nsoe) in enumerate(zip(self.primary_sat_and_debris_rvm,
                                        self.primary_sat_and_debris_mahala,
                                        self.primary_sat_and_debris_b_plane,
                                        self.all_object_rvm0, 
                                        self.non_singular_oe)):
            
            object_rvm.append(np.array(rvm0)) 
            object_bplane.append(np.zeros(3))
            object_mahala.append(np.zeros(3))
            
            # Object non-singular params
            oe_non_singular = self.calculate_non_singular_oe(np.array(rvm0)[:6])
            nsoe.append(oe_non_singular)

        # Initialize only one dynamical model as all debris and satellites use same params
        self.dynamics1:Dynamics = \
            Dynamics(self.cfg.params_dyn)
        
        self.dynamics_ideal:Dynamics =\
            Dynamics(self.cfg.params_dyn_ideal)

        # Predefined array to store all controls throughout simulation 
        self.manplan_control_inputs = np.zeros((self.traj_len,3))
        self.n += 1
        
        self.propagate_2_tca()
        
        return

    def save_states(self,
                    path:str
                    )->None:
        
        """collect all important data of object and save in csv files
        """
        
        
        
        
        
        rvm_cols    = ['$r_x$','$r_y$', '$r_z$', 
                       '$v_x$','$v_y$', '$v_z$', '$m$']
        
        thrust_cols = ['f_r', 'f_t', 'f_n']
        
        relative_cart_cols  = ['$\delta r_x$','$\delta r_y$', '\delta $r_z$', 
                       '$ \delta v_x$','$\delta v_y$', '$\delta v_z$']
        
        doe_cols    = ['$a\delta a$','$a \delta \lambda$', 
                       '$a\delta e_x$', '$a\delta e_y$',
                       '$a\delta i_x$', '$a\delta i_y$']
                
        oe_ns_cols = ['$a$','$u$', 
                       '$e_x$', '$e_y$',
                       '$i$', '$\Omega$']
            

        thrust_rtn = np.array(self.controls_RTN)
        p_max_predictions = np.array(self.p_max_predictions)
        delta_r_b_plane = np.array(self.delta_r_b_plane)
        doe = np.array(self.doe)
        primary_sat_and_debris_rvm = np.array(self.primary_sat_and_debris_rvm)
        oe_ns = np.array(self.oe_ns)

        pd.DataFrame(thrust_rtn).to_csv(os.path.join(path,'thrust_rtn.csv'), index=False, header=thrust_cols)
        pd.DataFrame(p_max_predictions).to_csv(os.path.join(path,'p_max_predictions.csv'), index=False, header=False)
        pd.DataFrame(delta_r_b_plane.squeeze()).to_csv(os.path.join(path,'delta_r_b_plane.csv'), index=False, header=False)
        pd.DataFrame(doe).to_csv(os.path.join(path,'doe.csv'), index=False, header=doe_cols)
        pd.DataFrame(self.conjuction_points_time).to_csv(os.path.join(path,'conjuction_points_time.csv'), index=False, header=False)
        pd.DataFrame(doe).to_csv(os.path.join(path,'doe.csv'), index=False, header=doe_cols)
        pd.DataFrame(oe_ns).to_csv(os.path.join(path,'oe_ns.csv'), index=False, header=oe_ns_cols)
        pd.DataFrame(self.manaplan_call_relative_time).to_csv(os.path.join(path,'manaplan_call_relative_time.csv'), index=False, header=False)

        for i in range(len(primary_sat_and_debris_rvm)):
            if i == 0:
                object_name = "primary"
            else:
                object_name = f"secondary_{i}"
                
            pd.DataFrame(primary_sat_and_debris_rvm[i]).to_csv(os.path.join(path,f'rvm_{object_name}.csv'), index=False, header=rvm_cols)

        
        return 