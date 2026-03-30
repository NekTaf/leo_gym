"""
Generate collision at TCA (time of closest approach). 
"""

# Standard library
import copy
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np

# Local
from leo_gym.satellite.satellite_base import SatelliteConfig, Satellite
from leo_gym.utils.utils import generate_random_perpendicular_normalized_vector


def collision_generator(
                rv0:np.ndarray,
                relative_t_tca:int,
                dt:int,
                params_dyn:Any,
                days:float
                )->np.ndarray:
    
    """Generates starting coordinates for debris collision 
    
    :params pvm0: starting position velocity and mass of debris object
    :params relative_t_tca: discrete relative time to collision from starting simulation ephemeris time
    :params dt: sampling time for simulation
    :params days: number of days for simulation (in this case anything above relative_t_tca is fine)
    :params params_dyn: parameters for debris object using satellite_base class, all perturbations are included
        
    :returns: (7,) pvm of obstacle 
    """

    satellite_config = SatelliteConfig(
        rv0=rv0, 
        params_dyn=params_dyn, 
        dt=dt,
        days=days)

    object_secondary = Satellite(satellite_config)
    
    for _ in range(relative_t_tca):
        object_secondary.sat_propagate(np.zeros(3))   
        
    p_vector = object_secondary.rvm_eci_states[-1][:3]
    
    # v_norm  = np.linalg.norm(object_secondary.rvm_eci_states[-1][3:6])
    # v_vector = generate_random_perpendicular_normalized_vector(p_vector)*v_norm
    # v_vector = v_vector.reshape(3,)
    
    # generate random velocity vector but keep position same
    old_v   = object_secondary.rvm_eci_states[-1][3:6]
    v_norm  = np.linalg.norm(old_v)

    while True:
        v_vector = generate_random_perpendicular_normalized_vector(p_vector) * v_norm
        if abs(np.dot(v_vector, old_v)) < 0.95 * v_norm**2:   # reject if too similar
            break

    
    
    object_secondary.rvm_eci_states[-1] = np.concatenate((p_vector,v_vector,object_secondary.rvm_eci_states[-1][6:7]),axis=0)

    # do back propagation to get starting relative position and velocity vectors for debris
    object_secondary.dt = -abs(object_secondary.dt)
    
    for _ in range(relative_t_tca):
        object_secondary.sat_propagate(np.zeros(3))   

    return object_secondary.rvm_eci_states[-1]

    
def collision_metrics(
    delta_r_b: np.ndarray,
    cov_b: np.ndarray,
    l: float,
    eps: float = 1e-8) -> Tuple[float,float,float,np.ndarray]:
    """ 
    :params delta_r: (3,) [xi,eta,zeta] -b-plane meters \b
    :params covariance_matrix: (3,3) Their covariance matrix \b
    :params l: (int or float) The collision sphere radius in meters \b 
        
    :returns: Tuple
    + Pc: Collision probability \b 
    + Pc_max (float): Max collision probability \b
    + mahalanobis_dist (float): Mahalanobis distance \b
    + delta_r_b (np.ndarray)(2,1): 2D projected position on B-plane
    """
    # Project to (xi, zeta)
    delta_3d = np.asarray(delta_r_b).reshape(3,)
    delta_2d = np.array([delta_3d[0], delta_3d[2]]).reshape(2, 1)

    cov_2d = cov_b[np.ix_([0, 2], [0, 2])]
    inv_cov_2d = np.linalg.inv(cov_2d)

    d2 = (delta_2d.T @ inv_cov_2d @ delta_2d).item()
    d = np.sqrt(max(d2, 0.0))
    d_safe = max(d, eps)

    det_cov = float(np.linalg.det(cov_2d))
    det_sqrt = np.sqrt(max(det_cov, eps))

    Pc = (l**2 / (2 * det_sqrt)) * np.exp(-0.5 * d_safe**2)
    Pc_max = (l**2 / (d_safe**2 * det_sqrt * np.e))

    Pc = float(np.clip(Pc, 0.0, 1.0))
    Pc_max = float(np.clip(Pc_max, 0.0, 1.0))

    return np.array([Pc, Pc_max, d, delta_2d[0, 0], delta_2d[1, 0]])
