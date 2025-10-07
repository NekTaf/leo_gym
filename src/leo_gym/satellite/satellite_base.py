# Standard library
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Third-party
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

# Local
from leo_gym.orbit.dynamics.conversions import *
from leo_gym.orbit.dynamics.dynamics import Dynamics, DynamicsConfig
from leo_gym.orbit.dynamics.init_weather_mjd import load_spice_kernels
from leo_gym.utils.utils import gen_rv0, random_vector_know_norm


class SatelliteConfig(BaseModel):
    """
    Configuration for generating nominal trajectories for station keeping.
    """
    delta_r_norm: float = Field(default=0.0, description="Normalized position offset")
    delta_v_norm: float = Field(default=0.0, description="Normalized velocity offset")
    rv0: Optional[NDArray[np.float64]] = Field(default=None, description="Initial [r, v] state vector")
    sma: Optional[float] = Field(default=None, description="SMA")
    params_dyn: DynamicsConfig = Field(..., description="Dynamics parameters config")
    dt: float = Field(60, description="Time step in seconds")
    days: float = Field(365, gt=0, description="Propagation duration in days")

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class Satellite():
    """
    Use this to generate nominal trajectories for station keeping
    """
    def __init__(self, 
                 cfg: SatelliteConfig):

        self.sma= cfg.sma
        self.delta_r_norm = cfg.delta_r_norm
        self.delta_v_norm = cfg.delta_v_norm
        self.rv0 = cfg.rv0
        self.dt = cfg.dt
        self.params_dyn = cfg.params_dyn
        self.days = cfg.days

        self.reset_sat_states()
        self._set_initial_pvm(sma=self.sma,
                              r_norm=self.delta_r_norm,
                              v_norm=self.delta_v_norm,
                              rv0=self.rv0)
                
        load_spice_kernels()
                            
    def _propagate_one_step(self, 
                            x:NDArray,
                            u:NDArray
                            )->None:
        
        self.rvm_eci_states.append(self.dynamics1.propagate(x=x,u=u,t=self.t[self.i],dt=self.dt))   
        self.kep_states.append(rv_2_kepler_oe_pyorb(x[:6]))
        lat,lon = self.dynamics1.get_lat_lon(r=x[:3])
        
        self.lat.append(lat)
        self.lon.append(lon)

        self.i += int(np.sign(self.dt)*1)
        self.f_rtn.append(u)
        
        return
        
    def _set_initial_pvm(self,
                        r_norm:float,
                        v_norm:float,
                        rv0: Optional[NDArray]=None,
                        sma: Optional[float]=None,
                        )->None:
        
        if sma is not None:
            x = gen_rv0(sma=sma)
        else:
            x = rv0.copy()
        
        r_dev = random_vector_know_norm(k=3,norm=r_norm)
        v_dev = random_vector_know_norm(k=3,norm=v_norm)

        x[:3] += r_dev
        x[3:6] += v_dev

        x = np.concatenate((x, [self.params_dyn.m])).reshape(7,)
        
        self.rvm_eci_states.append(x)
        self.kep_states.append(rv_2_kepler_oe_pyorb(x[:6]))
        
        # # Hasnt been initalized yet 
        # self.dynamics1.t=0.0
        # lat,lon = self.dynamics1.get_lat_lon(r=x[:3])
        # self.lat.append(lat)
        # self.lon.append(lon)
        
        self.i += int(np.sign(self.dt)*1)
        
        return

    def reset_sat_states(self
                         )->None:
        self.dynamics1 = Dynamics(self.params_dyn)
        self.rvm_eci_states = []
        self.kep_states = []
        self.rv_relative = []
        self.f_rtn = []
        self.lat = []
        self.lon = []
        self.f_rtn.append(np.zeros(3))

        self.i=0
        self.traj_len=int(self.days*24*60*60/(np.sign(self.dt)*self.dt))
        self.t = np.linspace(0, self.traj_len * np.sign(self.dt)*self.dt, self.traj_len + 1)
        
        return

    def sat_propagate(self,
                      u:NDArray[np.float64]
                      )->None:
        
        x=self.rvm_eci_states[-1]
        self._propagate_one_step(x,u)
        
        return
    
    def save_states(self,
                    path:str
                    )->None:
        
        
        return
    
    def plot_states(self,
                    save_path:str=None
                    )->None:
        
        
        return
    
    
    