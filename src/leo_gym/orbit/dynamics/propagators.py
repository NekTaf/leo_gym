# Third-party
import numpy as np
from numpy.typing import NDArray


class PropagatorModels:
    def __init__(self):
        pass

    @staticmethod
    def rk4(x:NDArray[np.float64],
            u:NDArray[np.float64],
            dx_dt_dynamics_fun: any,
            dt: float,
            t: float
            )->NDArray[np.float64]:
        
        """RK4 dynamics propagator

        Args:
            x (NDArray[np.float64]): current state 
            u (NDArray[np.float64]): control input
            dx_dt_dynamics_fun (any): callback dynamics function
            dt (float): sampling time
            t (float): current time used in model propagation

        Returns:
            NDArray[np.float64]: next timestep state
        """
        k1 = dt * dx_dt_dynamics_fun(x, u, t)
        k2 = dt * dx_dt_dynamics_fun(x + 0.5 * k1, u, t)
        k3 = dt * dx_dt_dynamics_fun(x + 0.5 * k2, u, t)
        k4 = dt * dx_dt_dynamics_fun(x + k3, u, t)

        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


    @staticmethod
    def rk8mod(x:NDArray[np.float64],
            u:NDArray[np.float64],
            dx_dt_dynamics_fun: any,
            dt: float,
            t: float
            )->NDArray[np.float64]:
        
        """RK8 dynamics propagator -- Higher accuracy

        Args:
            x (NDArray[np.float64]): current state 
            u (NDArray[np.float64]): control input
            dx_dt_dynamics_fun (any): callback dynamics function
            dt (float): sampling time
            t (float): current time used in model propagation

        Returns:
            NDArray[np.float64]: next timestep state
        """
        
        sqrt21 = 4.58257569495584000680

        a21 = 1.0 / 2.0
        a31 = 1.0 / 4.0
        a32 = 1.0 / 4.0
        a41 = 1.0 / 7.0
        a42 = -(7.0 + 3.0 * sqrt21) / 98.0
        a43 = (21.0 + 5.0 * sqrt21) / 49.0
        a51 = (11.0 + sqrt21) / 84.0
        a53 = (18.0 + 4.0 * sqrt21) / 63.0
        a54 = (21.0 - sqrt21) / 252.0
        a61 = (5.0 + sqrt21) / 48.0
        a63 = (9.0 + sqrt21) / 36.0
        a64 = (-231.0 + 14.0 * sqrt21) / 360.0
        a65 = (63.0 - 7.0 * sqrt21) / 80.0
        a71 = (10.0 - sqrt21) / 42.0
        a73 = (-432.0 + 92.0 * sqrt21) / 315.0
        a74 = (633.0 - 145.0 * sqrt21) / 90.0
        a75 = (-504.0 + 115.0 * sqrt21) / 70.0
        a76 = (63.0 - 13.0 * sqrt21) / 35.0
        a81 = 1.0 / 14.0
        a85 = (14.0 - 3.0 * sqrt21) / 126.0
        a86 = (13.0 - 3.0 * sqrt21) / 63.0
        a87 = 1.0 / 9.0
        a91 = 1.0 / 32.0
        a95 = (91.0 - 21.0 * sqrt21) / 576.0
        a96 = 11.0 / 72.0
        a97 = -(385.0 + 75.0 * sqrt21) / 1152.0
        a98 = (63.0 + 13.0 * sqrt21) / 128.0
        a10_1 = 1.0 / 14.0
        a10_5 = 1.0 / 9.0
        a10_6 = -(733.0 + 147.0 * sqrt21) / 2205.0
        a10_7 = (515.0 + 111.0 * sqrt21) / 504.0
        a10_8 = -(51.0 + 11.0 * sqrt21) / 56.0
        a10_9 = (132.0 + 28.0 * sqrt21) / 245.0
        a11_5 = (-42.0 + 7.0 * sqrt21) / 18.0
        a11_6 = (-18.0 + 28.0 * sqrt21) / 45.0
        a11_7 = -(273.0 + 53.0 * sqrt21) / 72.0
        a11_8 = (301.0 + 53.0 * sqrt21) / 72.0
        a11_9 = (28.0 - 28.0 * sqrt21) / 45.0
        a11_10 = (49.0 - 7.0 * sqrt21) / 18.0

        b1 = 9.0 / 180.0
        b8 = 49.0 / 180.0
        b9 = 64.0 / 180.0

        k_1 = dx_dt_dynamics_fun(x, u, t)
        k_2 = dx_dt_dynamics_fun(x + dt * (a21 * k_1), u, t)
        k_3 = dx_dt_dynamics_fun( x + dt * (a31 * k_1 + a32 * k_2), u, t)
        k_4 = dx_dt_dynamics_fun(x + dt * (a41 * k_1 + a42 * k_2 + a43 * k_3), u, t)
        k_5 = dx_dt_dynamics_fun(x + dt * (a51 * k_1 + a53 * k_3 + a54 * k_4), u, t)
        k_6 = dx_dt_dynamics_fun(x + dt * (a61 * k_1 + a63 * k_3 + a64 * k_4 + a65 * k_5), u, t)
        k_7 = dx_dt_dynamics_fun(x + dt * (a71 * k_1 + a73 * k_3 + a74 * k_4 + a75 * k_5 + a76 * k_6), u, t)
        k_8 = dx_dt_dynamics_fun( x + dt * (a81 * k_1 + a85 * k_5 + a86 * k_6 + a87 * k_7), u, t)
        k_9 = dx_dt_dynamics_fun(x + dt * (a91 * k_1 + a95 * k_5 + a96 * k_6 + a97 * k_7 + a98 * k_8), u, t)
        k_10 = dx_dt_dynamics_fun(x + dt * (
                    a10_1 * k_1 + a10_5 * k_5 + a10_6 * k_6 + a10_7 * k_7 + a10_8 * k_8 + a10_9 * k_9), u, t)
        k_11 = dx_dt_dynamics_fun(x + dt * (
                    a11_5 * k_5 + a11_6 * k_6 + a11_7 * k_7 + a11_8 * k_8 + a11_9 * k_9 + a11_10 * k_10), u, t)

        new_state = x + dt * (b1 * k_1 + b8 * k_8 + b9 * k_9 + b8 * k_10 + b1 * k_11)
        return new_state
