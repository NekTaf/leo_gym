# Standard library
import time

# Third-party
import cvxpy as cp
import numpy as np
from scipy.signal import cont2discrete


def _DCM_eci2rtn(r, v):

    r_norm = r / np.linalg.norm(r)
    v_norm = v / np.linalg.norm(v)

    h_norm = (np.cross(r_norm, v_norm)
            / np.linalg.norm(np.cross(r_norm, v_norm)))

    t_norm = np.cross(h_norm, r_norm)
    
    return np.vstack((r_norm, t_norm, h_norm))


def cw_state_matrix(n):
    """
    Returns the state matrix A for the Clohessy-Wiltshire equations in the RTN frame.
    
    The state vector is defined as:
        x = [x, y, z, x_dot, y_dot, z_dot]^T,
    and the dynamics are given by:
        x_dot = A * x.
    
    The matrix A is:
        [  0      0      0      1      0      0  ]
        [  0      0      0      0      1      0  ]
        [  0      0      0      0      0      1  ]
        [ 3n^2    0      0      0     2n      0  ]
        [  0      0      0    -2n      0      0  ]
        [  0      0   -n^2      0      0      0  ]
    
    Parameters:
        n (float): Mean motion of the reference orbit.
    
    Returns:
        np.ndarray: 6x6 state-space matrix for the CW equations.
    """
    A = np.array([
        [0,      0,     0,    1,    0,    0],
        [0,      0,     0,    0,    1,    0],
        [0,      0,     0,    0,    0,    1],
        [3*n**2, 0,     0,    0,   2*n,   0],
        [0,      0,     0,  -2*n,   0,     0],
        [0,      0,  -n**2,   0,    0,     0]
    ])
    return A

def cw_input_matrix():
    B = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    return B



class MPC:
    """
    A class to solve the finite-horizon MPC problem as a QP.
    
    Dynamics:
         x[k+1] = A_d x[k] + B_d u[k],
    Cost function:
         J = x[N]^T Qf x[N] + sum_{k=0}^{N-1} (x[k]^T Q x[k] + u[k]^T R u[k])
    Subject to:
         -u_max <= u[k] <= u_max for all k.
    
    The class converts the continuous-time dynamics to discrete time using a zero-order hold
    over the time step dt. It also transforms the error state from ECI to RTN using the nominal
    state.
    
    Parameters:
        n               : Parameter to build the CW state transition matrix.
        mass            : Mass value used for the CW input matrix.
        dt              : Time step for discretization.
        error_state_eci : The error state vector expressed in the ECI frame.
        rv_nom          : Nominal state vector (assumed to be 6-dimensional: [pos; vel]).
        N_c             : Prediction horizon (number of discrete time steps).
        Q               : State cost matrix.
        Qf              : Terminal state cost matrix.
        R               : Control cost matrix.
        u_max           : Maximum absolute value allowed for each control input element.
        x0              : (Optional) Initial state vector. If not provided, it is computed from the error state.
    """
    def __init__(self, n, mass, dt, N_c, Q, Qf, R, u_max,dynamics):
        self.n = n
        self.mass = mass
        self.dt = dt
        self.N_c = N_c
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.u_max = u_max, 
        self.dynamics = dynamics

    def solve(self, rv_sat, rv_nom, elapsed_time):
        
        
        error_state_eci = rv_sat - rv_nom
        
        A = cw_state_matrix(self.n)
        B = cw_input_matrix()

        system = (A, B, np.eye(A.shape[0]), 0)
        A_d, B_d, _, _, _ = cont2discrete(system, self.dt)

        pos_nom = rv_nom[:3]
        vel_nom = rv_nom[3:6]
        dcm = _DCM_eci2rtn(pos_nom, vel_nom)

        error_pos_rtn = dcm @ error_state_eci[:3]
        error_vel_rtn = dcm @ error_state_eci[3:6]
        error_state_rtn = np.concatenate((error_pos_rtn, error_vel_rtn))

        n_states = A_d.shape[0]
        n_inputs = B_d.shape[1]

        x = cp.Variable((n_states, self.N_c + 1))
        u = cp.Variable((n_inputs, self.N_c))
        
        cost = 0
        constraints = []

        constraints += [x[:, 0] == error_state_rtn]

        for k in range(self.N_c):
            _,_,pert_sun,pert_moon,pert_srp,pert_drag = self.dynamics.get_current_pertubration_forces(x=rv_sat,t=elapsed_time+k*self.dt) 

            cost += cp.quad_form(x[:, k], self.Q) + cp.quad_form(u[:, k], self.R)
            constraints += [x[:, k+1] == A_d @ x[:, k] + B_d/self.mass @ u[:, k] + B_d @ (pert_srp+pert_drag+pert_sun+pert_moon)]
            constraints += [cp.abs(u[:, k]) <= self.u_max] 

        cost += cp.quad_form(x[:, self.N_c], self.Qf)
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        start = time.perf_counter()
        prob.solve(solver=cp.CLARABEL)
        computation_time = time.perf_counter() - start
        computation_time = prob.solver_stats.solve_time

        if prob.status in ["infeasible", "unbounded"]:
            # raise ValueError("The MPC QP is infeasible or unbounded.")
            u_opt = np.zeros((3,3))
            x_opt = None
        
        u_opt = u.value
        x_opt = x.value
        
        return u_opt, x_opt, computation_time
