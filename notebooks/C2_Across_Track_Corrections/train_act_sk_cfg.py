import numpy as np
from leo_gym.gyms.roe_sk_gym import RoeGymConfig, SatelliteROEConfig
from dataclasses import dataclass
from typing import Dict, Any
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional


satellite_params = {
    "dt":   60,
    "days": 2,
    "ideal_traj_params": {
        "flag_rtn_thrust": True,
        "flag_mass_loss": True,
        "flag_pert_moon": False,
        "flag_pert_sun": False,
        "flag_pert_srp": False,
        "flag_pert_drag": False,
        "flag_pert_irr_grav": False,
        "eph_time_0": 6.338304141847866e08,
        "m": 150,
        "f_max": 18e-3,
        "Isp": 860,
        "Ad": 1.3,
        "Cd": 2.2,
        "Cr": 1.3,
        "As": 1.3,
        "mf": 136,
    },
    "pert_traj_params": {
        "flag_rtn_thrust": True,
        "flag_mass_loss": True,
        "flag_pert_moon": False,
        "flag_pert_sun": False,
        "flag_pert_srp": False,
        "flag_pert_drag": False,
        "flag_pert_irr_grav": False,
        "eph_time_0": 6.338304141847866e08,
        "m": 150,
        "f_max": 18e-3,
        "Isp": 860,
        "Ad": 1.3,
        "Cd": 2.2,
        "Cr": 1.3,
        "As": 1.3,
        "mf": 136,
    },
    "rv0":np.array([-475449.609559833, 277131.557120858, 7553438.90796208,
                    -7238.56711050838, -16.6455796498367, -454.213727291452])}

sat_cfg = SatelliteROEConfig(**satellite_params)


env_cfg = RoeGymConfig(high_action=[15.0,110,41],
                         low_action=[-15.0,10,1],
                         no_timesteps=20,
                         dt=60,
                         flag_man_type="act",
                         Delta_kep_ranges=[[0,400],[0,0],[0.005,0.008],
                                           [0.005,0.008], [0.5, 0.5],[0,0]],
                         ada_target=50,
                         adi_norm_cutoff=1500,
                         adi_norm_target=100,
                         Delta_ada_targ=50,
                         satellite_params=sat_cfg)

NUM_ENV = 1

SEED=57
@dataclass
class PpoSmdpCfg():
    gamma:float=0.99
    gae_lambda:float=0.95
    n_steps:int=20
    batch_size:int = field(init=False)
    verbose:int=2
    policy_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"log_std_init": -1} 
    )
    learning_rate:float=0.0003
    
    def __post_init__(self):
        self.batch_size = int(NUM_ENV * self.n_steps)

    
@dataclass
class PpoCfg():
    gamma:float=0.99
    gae_lambda:float=0.95
    n_steps:int=20
    batch_size:int=field(init=False)
    verbose:int=2
    learning_rate:float=0.0003
    policy_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"log_std_init": -1} 
    )
    def __post_init__(self):
        self.batch_size = int(NUM_ENV * self.n_steps)


ppo_cfg = PpoCfg()
ppo_smdp_cfg = PpoSmdpCfg()