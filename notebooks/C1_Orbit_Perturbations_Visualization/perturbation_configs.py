from leo_gym.satellite.satellite_base import Satellite, SatelliteConfig
from leo_gym.orbit.dynamics.dynamics import DynamicsConfig
import numpy as np

# Sampling time in seconds and days of simulation
DT_SIMULATION = 60
DAYS_SIMULATION = 1

# Starting position-velocity vector for circular orbit
RV0 = np.array([-6648164.33933423,
                -134028.719688348,
                -3653085.498202,
                3487.94012840057, 
                -232.615833196755,
                -6347.41684371367],
               dtype=np.float64)


# Ideal satellite config
ideal_cfg = SatelliteConfig(days=DAYS_SIMULATION,
                            delta_r_norm=0,
                            delta_v_norm=0, 
                            rv0=RV0,
                            params_dyn=DynamicsConfig(
                                        flag_rtn_thrust=True,
                                        flag_mass_loss=False,
                                        flag_pert_moon=False,
                                        flag_pert_sun=False,
                                        flag_pert_srp=False,
                                        flag_pert_drag=False,
                                        flag_pert_irr_grav=False,
                                        eph_time_0=6.338304141847866e+08,
                                        m=150,
                                        f_max=18e-3,
                                        Isp=860,
                                        Ad=1.3,
                                        Cd=2.2,
                                        Cr=1.3,
                                        As=1.3,
                                        mf=136
                                    ),
                            dt=DT_SIMULATION
                            )


# Real or perturbed satellite config
real_cfg = SatelliteConfig(days=DAYS_SIMULATION,
                            delta_r_norm=0,
                            delta_v_norm=0, 
                            rv0=RV0,
                            params_dyn=DynamicsConfig(
                                        flag_rtn_thrust=True,
                                        flag_mass_loss=False,
                                        flag_pert_moon=True,
                                        flag_pert_sun=True,
                                        flag_pert_srp=True,
                                        flag_pert_drag=True,
                                        flag_pert_irr_grav=True,
                                        eph_time_0=6.338304141847866e+08,
                                        m=150,
                                        f_max=18e-3,
                                        Isp=860,
                                        Ad=1.3,
                                        Cd=2.2,
                                        Cr=1.3,
                                        As=1.3,
                                        mf=136
                                    ),
                            dt=DT_SIMULATION
                            )
