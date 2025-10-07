""" Satellite class used for Relative Orbital Element (ROE) control.

The Relative Orbital Elements (ROE) are nonlinear combinations of classical orbital elements.
They are constructed from differences in semi-major axis (a), inclination (i), 
and trigonometric components of eccentricity (e), argument of perigee (ω), and RAAN (☊). \b

The relative motion is expressed in a linear, nonsingular, and geometrically intuitive form. \b    

Nonsingular elements remain valid and continuous for all orbit shapes 
and orientations, avoiding undefined angles at circular or equatorial conditions

oe_ns = ["SMA (a) m", 
        "AOML (u) rad", 
        "EX (e_x)-", 
        "EY (e_y)-", 
        "INC (i) rad", 
        "RAAN (☊) rad"]\b

roe = ["a \delta a m", 
        "a \delta \lambda m", 
        "a \delta ex m", 
        "a \delta ey m", 
        "a \delta ix m",
        "a \delta iy m"]\b
"""


# Standard library
import os
from typing import Optional, Tuple

# Third-party
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Local
from leo_gym.orbit.dynamics.conversions import *
from leo_gym.orbit.dynamics.dynamics import Dynamics, DynamicsConfig
from leo_gym.orbit.dynamics.init_weather_mjd import load_spice_kernels
from leo_gym.satellite.satellite_base import SatelliteConfig, Satellite as sat_ideal
from leo_gym.utils.matplot_style_cfg import *
from leo_gym.utils.utils import list_2_txt


class SatelliteROEConfig(BaseModel):
    dt:   float = Field(..., gt=0, description="Simulation discretization timestep in seconds")
    days: float = Field(..., gt=0, description="Simulation length in days")
    ideal_traj_params: DynamicsConfig = Field(
        ..., description="Nominal/original orbit dynamics parameters"
    )
    pert_traj_params: DynamicsConfig = Field(
        ..., description="Perturbed orbit dynamics parameters"
    )

    rv0: Optional[NDArray]

    model_config = ConfigDict(
      arbitrary_types_allowed=True,
      frozen=True,
      json_encoders={ NDArray: lambda v: v.tolist() }
    )


class Satellite():
    
    def __init__(self,
                 cfg:SatelliteConfig):

        self.cfg = cfg
        load_spice_kernels()
        self.reset_sat_states(keep_ref_trajectory=False)

    def set_ideal_orbit(self)->None:
        self.traj_len=len(self.rvm_eci_ref)
        self.t = np.linspace(0, self.traj_len * self.cfg.dt, self.traj_len + 1)
        
        return

    def _propagate_one_step(self, 
                            x:NDArray,
                            u:NDArray
                            )->None:
        
        self.rvm_eci.append(self.dynamics1.propagate(x=x,u=u,t=self.t[self.discrete_time_index_simulation],dt=self.cfg.dt))        
        self.discrete_time_index_simulation += 1
        
        self.thrust_rtn.append(self.dynamics1.u_thruster_rtn)
        self.controls_ECI.append(self.dynamics1.u_thruster_eci)

        self.delta_rv_eci.append(self.rvm_eci[-1][:6] - self.rvm_eci_ref[self.discrete_time_index_simulation][:6])
        roe = self.calc_roe(self.rvm_eci[-1][:6],self.rvm_eci_ref[self.discrete_time_index_simulation][:6])
        
        self.roe.append(roe[0])
        self.oe_ns.append(roe[1])
        
        return
                        
    def set_initial_pvm(self,
                        x:NDArray
                        )->None:
        
        self.rvm_eci.append(x)
        self.delta_rv_eci.append(self.rvm_eci[-1][:6] - self.rvm_eci_ref[self.discrete_time_index_simulation, :6])
        self.controls_ECI.append(np.zeros(3))
        self.thrust_rtn.append(np.zeros(3))
        
        roe_oe = self.calc_roe(self.rvm_eci[-1][:6],self.rvm_eci_ref[self.discrete_time_index_simulation, :6])
        self.roe.append(roe_oe[0])
        self.oe_ns.append(roe_oe[1])
        
        return
   
    def sat_propagate(self,
                      u:NDArray
                      )->None:
        self._propagate_one_step(self.rvm_eci[-1], u)
        
        return 

    def calc_roe(self, 
                 pv_sat:NDArray,
                 pv_ref:NDArray
                 )->Tuple[NDArray, NDArray, NDArray]:
        """Convert Cartesian state vectors into relative orbital elements (ROE)
        and non-singular orbital elements for both deputy and chief.
        
        Args:
            pv_sat (NDArray): _description_
            pv_ref (NDArray): _description_

        Returns:
            Tuple[NDArray, NDArray, NDArray]: _description_
        """
        return rv_2_roe_and_non_singular_oe(pv_sat, pv_ref)
     
    def set_initial_deviation(self, 
                              d_kep:NDArray
                              )->None:  
        """
        d_kep = [0, 0, 0, 0, 1.2, 0] 
        [d_sma, d_ecc, d_inc, d_ran, d_aop, d_tan]
        """
        
        rv_ref = self.rvm_eci_ref[self.discrete_time_index_simulation,:6]
        
        rv_ref = rv_ref.reshape(6,1)
        oe = rv_2_kepler_oe(rv_ref)

        base = oe[[0,1,2,3,4,6],:].squeeze()  
        d_kep_rad = np.deg2rad(np.array(d_kep))                
        oe_mod_1d = base + d_kep_rad
        oe_mod = oe_mod_1d.reshape(6,1)         

        pv = kepler_oe_2_rv(oe_mod)
        
        # pv = roe_error_init_state(roe=d_kep.reshape(6,),
        #                              rv_r=rv_ref.reshape(6,))
                
        pv = np.append(pv, self.rvm_eci_ref[self.discrete_time_index_simulation,6])
        pv = pv.reshape(7,)
                
        self.set_initial_pvm(pv)

        return
                             
    def apply_manplan(self, 
                      manplan:NDArray, 
                      flag_man_type:str
                      )->None:
        
        """Apply manuever plan
        
        Maneuver includes: \b
        + Thrust direction -- RTN \b
        + Thrust delay -- t_delay \b
        + Thrust duration -- t_duration \b
        + Coasting time after applied thrust -- t_coast \b
        
        Args:
            manplan (NDArray): maneuver plan 
            flag_man_type (str): Along (tangential) or Across (normal) track corrections 

        Returns:
            int: elapsed time in from the start of maneuver to end 
            
        """
        self.manplans.append(manplan.tolist() + \
            [flag_man_type,self.discrete_time_index_simulation*self.cfg.dt+self.cfg.pert_traj_params.eph_time_0])
        
        dt_delay:int = int(manplan[1])
        dt_duration:int = int(manplan[2])
        dt_coast = 0                     

        manplan_total = np.zeros((dt_delay+dt_duration+dt_coast,3))

        if flag_man_type == "alt":
            manplan_total[dt_delay:dt_delay+dt_duration,:] = np.array([0,manplan[0],0])
        elif flag_man_type == "act":
            manplan_total[dt_delay:dt_delay+dt_duration,:] = np.array([0,0,manplan[0]])
            
        self.manplan_control_inputs[self.discrete_time_index_simulation+dt_delay:dt_delay+dt_duration+self.discrete_time_index_simulation,2] = manplan[0]
            
        for i in range(int(dt_delay+dt_duration+dt_coast)):
            if self.discrete_time_index_simulation+1 >= self.traj_len:
                # print("WARNING: Ran out of trajectory data. Increase reference state number!")
                return i
            self.sat_propagate(u=self.manplan_control_inputs[self.discrete_time_index_simulation,:])
                                
        return 
     
    def reset_sat_states(self, 
                         keep_ref_trajectory:bool=True
                         )->None:
        
        self.rvm_eci = []
        self.controls_ECI = []
        self.thrust_rtn = []
        self.delta_rv_eci = []
        self.roe = []
        self.oe_ns = []
        self.manplans = []


        if not keep_ref_trajectory:
            self.rvm_eci_ref = []

            config_ideal_sat = SatelliteConfig(delta_r_norm=0, 
                                            delta_v_norm=0,
                                            dt=self.cfg.dt, 
                                            days=self.cfg.days,
                                            params_dyn=self.cfg.ideal_traj_params, 
                                            rv0=self.cfg.rv0)
            
            sat_nom = sat_ideal(config_ideal_sat)

            for _ in range(int(24*self.cfg.days*60*60/self.cfg.dt)):
                sat_nom.sat_propagate(np.zeros(3))   

            self.rvm_eci_ref = np.array(sat_nom.rvm_eci_states)
        
        self.dynamics1 = Dynamics(self.cfg.pert_traj_params)
 
        self.set_ideal_orbit()
        self.discrete_time_index_simulation = 0
        
        self.manplan_control_inputs = np.zeros((self.traj_len,3))
        
        return 
    
    def plot_states_act(self)->None:

        x = np.arange(len(self.roe))/(60*24)

        adix = np.array([arr[4] for arr in self.roe])
        adiy = np.array([arr[5] for arr in self.roe])
        idi  = np.sqrt(adix**2 + adiy**2)

        N_thrust = np.array([arr[2] for arr in self.thrust_rtn])
        mass = np.array([arr[6] for arr in self.rvm_eci])
        massloss = mass

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(x, adix, label=r'$a \delta i_x$')
        axes[0].plot(x, adiy, label=r'$a \delta i_y$')
        axes[0].plot(x, idi,  label=r'$\|a \delta \mathbf{i}\|$')
        axes[0].legend()
        axes[0].set_ylabel('Relative Inclination (m)')

        axes[1].plot(x, N_thrust)
        axes[1].set_ylabel('Normal Thrust (N)')

        axes[2].plot(x, massloss)
        axes[2].set_ylabel('Mass (Kg)')
        axes[2].set_xlabel('Days')


        # plt.savefig(os.path.join(path,'act_sk_results.pdf'))
        plt.show()

        return
    
    def plot_states_alt(self)->None:
        
        x = np.arange(len(self.roe))/(60*24)

        ada = np.array([arr[0] for arr in self.roe])
        adlambda = np.array([arr[1] for arr in self.roe])
        adex = np.array([arr[2] for arr in self.roe])
        adey = np.array([arr[3] for arr in self.roe])
        ide_norm  = np.sqrt(adex**2 + adey**2)

        T_thrust = np.array([arr[1] for arr in self.thrust_rtn])
        mass = np.array([arr[6] for arr in self.rvm_eci])
        massloss = mass

        fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
        
        
        axes[0].plot(x, ada, label=r'$a \delta a$ (m)')
        axes[0].legend()

        axes[1].plot(x, adlambda, label=r'$a \delta \lambda$ (m)')
        axes[1].legend()

        axes[2].plot(x, adex, label=r'$a \delta e_x$ (m)')
        axes[2].plot(x, adey, label=r'$a \delta e_y$ (m)')
        axes[2].plot(x, ide_norm,  label=r'$\|a \delta e\|$ (m)')
        axes[2].legend()
        axes[2].set_ylabel('Eccentricity')

        axes[3].plot(x, T_thrust)
        axes[3].set_ylabel('Tang. Thrust - (N)')

        axes[4].plot(x, massloss)
        axes[4].set_ylabel('Mass (Kg)')
        axes[4].set_xlabel('Days')

        plt.show()

        return
    
    def plot_states_interactive(self)->None:
        

        x = np.arange(len(self.roe)) * (self.cfg.dt / 86400.0)

        signals = [
            ([arr[1] for arr in self.oe_ns], '$u$, rad'),
            ([arr[0] for arr in self.roe], '$a \delta a$, m'),
            ([arr[1] for arr in self.roe], '$a \delta \lambda$, m'),
            ([arr[2] for arr in self.roe], '$a \delta e_x$, m'),
            ([arr[3] for arr in self.roe], '$a \delta e_y$, m'),
            (np.sqrt(np.array([arr[2] for arr in self.roe])**2 + \
                np.array([arr[3] for arr in self.roe])**2), '$\|a \delta \mathbf{e}\|$, m'),
            ([arr[4] for arr in self.roe], '$a \delta i_x$, m'),
            ([arr[5] for arr in self.roe], '$a \delta i_y$, m'),
            (np.sqrt(np.array([arr[4] for arr in self.roe])**2 + \
                np.array([arr[5] for arr in self.roe])**2), '$\|a \delta \mathbf{i}\|$, m'),
            ([arr[6] for arr in self.rvm_eci], 'Mass, kg'),
            ([arr[0] for arr in self.thrust_rtn], 'R Thrust, N'),
            ([arr[1] for arr in self.thrust_rtn], 'T Thrust, N'),
            ([arr[2] for arr in self.thrust_rtn], 'N Thrust, N')
        ]
        num_plots = len(signals)

        fig = sp.make_subplots(rows=num_plots, cols=1)
        fontsize_axes = 12

        for idx, (data_list, label) in enumerate(signals, start=1):
            fig.add_trace(
                go.Scatter(x=x, y=data_list, mode='lines', name=label),
                row=idx, col=1
            )
            fig.update_xaxes(title_text='Sample Index', row=idx, col=1, title_font=dict(size=fontsize_axes))
            fig.update_yaxes(title_text=label, row=idx, col=1, title_font=dict(size=fontsize_axes))

        fig.update_layout(
            height=200 * num_plots,
            width=1000,
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20)
        )

        fig.show()
            
            
        
        return 

    def save_states(self,
                    path:str
                    )->None:
        
        """collect all important data of object and save in csv files
        
        """
        
        rvm_cols    = ['$r_x$','$r_y$', '$r_z$', '$v_x$','$v_y$', '$v_z$', '$m$']
        
        thrust_cols = ['f_r', 'f_t', 'f_n']
        
        delta_cols  = ['$\delta r_x$','$\delta r_y$', '\delta $r_z$', 
                       '$ \delta v_x$','$\delta v_y$', '$\delta v_z$']
        
        roe_cols    = ['$a\delta a$','$a \delta \lambda$', 
                       '$a\delta e_x$', '$a\delta e_y$',
                       '$a\delta i_x$', '$a\delta i_y$']
        
        # manplans_cols     = ["f", "\Delta t_0", "\Delta t_\text{man}"]
        
        oe_ns_cols = ['$a$','$u$', 
                       '$e_x$', '$e_y$',
                       '$i$', '$\Omega$']

        rvm_eci = np.array(self.rvm_eci)
        rvm_eci_ref = np.array(self.rvm_eci_ref)
        thrust_rtn = np.array(self.thrust_rtn)
        delta_rv_eci = np.array(self.delta_rv_eci)
        roe = np.array(self.roe)
        # manplans = np.array(self.manplans)
        oe_ns = np.array(self.oe_ns)
        
        
        non_zero_manplans = [x for x in self.manplans if x[0] != 0]

        pd.DataFrame(rvm_eci).to_csv(os.path.join(path,'rvm_eci.csv'), header=rvm_cols, index=False)
        pd.DataFrame(thrust_rtn).to_csv(os.path.join(path,'thrust_rtn.csv'),  header=thrust_cols, index=False)
        pd.DataFrame(delta_rv_eci).to_csv(os.path.join(path,'delta_rv_eci.csv'),  header=delta_cols, index=False)
        pd.DataFrame(roe).to_csv(os.path.join(path,'roe.csv'),  header=roe_cols, index=False)
        pd.DataFrame(oe_ns).to_csv(os.path.join(path,'oe_ns.csv'),  header=oe_ns_cols, index=False)
        pd.DataFrame(rvm_eci_ref).to_csv(os.path.join(path,'rvm_eci_ref.csv'),  header=rvm_cols, index=False)
        list_2_txt(list=non_zero_manplans, filename=os.path.join(path,'manplans'))
        
        
        return 