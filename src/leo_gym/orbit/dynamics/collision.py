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
    
    Inputs: 
        pvm0 (list): starting position velocity and mass of debris object
        relative_t_tca (int): discrete relative time to collision from starting simulation ephemeris time
        dt (int): sampling time for simulation
        days (float): number of days for simulation (in this case anything above relative_t_tca is fine)
        params_dyn (any): parameters for debris object using satellite_base class, all perturbations are included
        
    Returns:
        array (7,): pvm of obstacle 
    """
    
    
    satellite_config = SatelliteConfig(rv0=rv0, 
                                       params_dyn=params_dyn, 
                                       dt=dt,
                                       days=days)

    object_secondary = Satellite(satellite_config)
    
        
    for _ in range(relative_t_tca):
        object_secondary.sat_propagate(np.zeros(3))   
        
    p_vector = object_secondary.rvm_eci_states[-1][:3]
    
    # generate random velocity vector but keep position same
    v_norm  = np.linalg.norm(object_secondary.rvm_eci_states[-1][3:6])
    v_vector = generate_random_perpendicular_normalized_vector(p_vector)*v_norm
    v_vector = v_vector.reshape(3,)
    
    object_secondary.rvm_eci_states[-1] = np.concatenate((p_vector,v_vector,object_secondary.rvm_eci_states[-1][6:7]),axis=0)

    # Do back propagation to get starting relative position and velocity vectors for debris
    object_secondary.dt = -abs(object_secondary.dt)
    
    for _ in range(relative_t_tca):
        object_secondary.sat_propagate(np.zeros(3))   

    return object_secondary.rvm_eci_states[-1]


def collision_metrics(
                    delta_r_b:np.ndarray,
                    cov_b:np.ndarray,
                    l:float
                    )-> np.ndarray:
    
    """ 
    Args:
        delta_r (arr, size (3,)): [xi,eta,zeta] -b-plane meters \b
        covariance_matrix (arr, size(3,3)): Their covariance matrix \b
        l (int or float): The collision sphere radius in meters \b 
        
    Returns:
        Tuple: 
            Pc (float): Collision probability \b 
            Pc_max (float): Max collision probability \b
            mahalanobis_dist (float): Mahalanobis distance \b
            delta_r_b (np.ndarray)(2,1): 2D projected position on B-plane
    """
    
    # eta axis relative potion is 0 at TCA, both objects present on [xi,zeta] 
    
    delta_r_b = np.array([delta_r_b[0],delta_r_b[2]])
    delta_r_b = delta_r_b.reshape(2,1)
    
    cov_b = cov_b[[0, 2], :][:, [0, 2]]
    
    mahalanobis_dist = np.sqrt(delta_r_b.T@np.linalg.inv(cov_b)@delta_r_b)
    mahalanobis_dist = np.max(mahalanobis_dist,1) #Limit for numerical stability of Pc_max 
    
    Pc = l**2/(2*np.linalg.det(cov_b)**(0.5))*np.e**(-0.5*(mahalanobis_dist**2))
    Pc_max = l**2/((mahalanobis_dist**2)*(np.linalg.det(cov_b)**(0.5))*np.e)
    
    Pc = np.clip(Pc,0,1)
    Pc_max = np.clip(Pc_max,0,1)

    projected_metrics = np.array([Pc, Pc_max, mahalanobis_dist, delta_r_b[0], delta_r_b[1]]).reshape(5,)

    return projected_metrics
