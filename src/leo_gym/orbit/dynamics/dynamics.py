# Standard library
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Third-party
import numpy as np
import spiceypy as spice
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

# Local
from leo_gym.orbit.dynamics.propagators import PropagatorModels

np.seterr(invalid='ignore')

class DynamicsConfig(BaseModel):
    """
    Combined configuration for debris dynamics parameters and physical constants.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    flag_rtn_thrust: bool = Field(..., description="Use RTN unit vectors")
    flag_mass_loss: bool = Field(..., description="Enable mass loss modelling")
    flag_pert_moon: bool = Field(..., description="Include lunar perturbations")
    flag_pert_sun: bool = Field(..., description="Include solar perturbations")
    flag_pert_srp: bool = Field(..., description="Include solar radiation pressure")
    flag_pert_drag: bool = Field(..., description="Include atmospheric drag")
    flag_pert_irr_grav: bool = Field(..., description="Include Earth's J2 perturbation")
    eph_time_0: float = Field(..., description="Epoch time (MJD) for initial state")

    f_max: float = Field(..., gt=0, description="Maximum thrust [N]")
    m: float = Field(..., gt=0, description="Mass [kg]")

    Isp: float = Field(..., gt=0, description="Specific impulse [s]")
    Ad: float = Field(..., gt=0, description="Drag area [m²]")
    Cd: float = Field(..., gt=0, description="Drag coefficient")
    Cr: float = Field(..., gt=0, description="Reflectivity coefficient")
    As: float = Field(..., gt=0, description="Sunlit area [m²]")
    mf: float = Field(..., gt=0, description="Fuel mass [kg]")

    Cnm: NDArray = Field(
        default_factory=lambda: np.empty((361, 361)),
        description="Gravity field cosine coefficients"
    )
    Snm: NDArray = Field(
        default_factory=lambda: np.empty((361, 361)),
        description="Gravity field sine coefficients"
    )
    
    GM_Earth: float = Field(default=3.986004415e14, description="Earth GM [m³/s²]")
    GM_Sun: float = Field(default=1.327124400412794e20, description="Sun GM [m³/s²]")
    GM_Moon: float = Field(default=4.902800192171394e12, description="Moon GM [m³/s²]")
    g0: float = Field(default=9.807, description="Standard gravity [m/s²]")

    omega_Earth: float = Field(default=7.292115146706388e-05, description="Earth rotation rate [rad/s]")
    R_Sun: float = Field(default=696000000, description="Solar radius [m]")
    R_Moon: float = Field(default=1738000, description="Lunar radius [m]")
    f_Earth: float = Field(default=1/298.257223563, description="Earth flattening factor")
    R_Earth: float = Field(default=6378136.3, description="Earth radius [m]")
    P: float = Field(default=4.51e-6, description="Solar radiation pressure at 1 AU [N/m²]")

    AU: float = Field(default=149597870699.999988, description="Astronomical unit [m]")
    c_light: float = Field(default=299792457.999999984, description="Speed of light [m/s]")
    j2: float = Field(default=0.00108263, description="Earth’s J2 coefficient")


class Dynamics():
    
    def __init__(self, 
                 cfg:DynamicsConfig):
        
        
        self.cfg = cfg
        self.m = self.cfg.m

        for key, value in cfg.model_dump().items():
            setattr(self, key, value)
        
        if self.cfg.flag_pert_irr_grav:
            original_cwd = os.getcwd()
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            os.chdir("..")
            self.read_space_weather(file="spice_kernels/GGM03C.txt")
            os.chdir(original_cwd)
        else:
            pass        

    # Initalise ephemeris time and weather data 

    # Weather data_SPICE
    def read_space_weather(self, 
                           file:str
                           )->None:
        
        with open(file, 'r') as fid:
            for n in range(361):
                for m in range(n + 1):
                    temp = np.array([float(x) for x in fid.readline().split()])
                    self.cfg.Cnm[n, m] = temp[2]
                    self.cfg.Snm[n, m] = temp[3]      
        return
                            
    # Reference Frame TF
    def _DCM_eci2rtn(self,
                     r: NDArray, 
                     v: NDArray
                     )->NDArray:
        #Hill frame Transformation DCM from ECI to RTN
        r_norm = r / np.linalg.norm(r)
        v_norm = v / np.linalg.norm(v)
        # Orbit normal
        h_norm = (np.cross(r_norm, v_norm)
                  / np.linalg.norm(np.cross(r_norm, v_norm)))
        # Orbit tangential
        t_norm = np.cross(h_norm, r_norm)
        return np.vstack((r_norm, t_norm, h_norm))
    
    def _DCM_eci2ecef(self
                      )->NDArray:
        DCM = (spice.sxform('J2000', 'ITRF93', self._currentET()))
        DCM = DCM[:3, :3]
        return DCM
    
    # For plotting
    def action_rtn_to_eci(self
                          )->NDArray:
        u_thruster_rtn = self.u
        DCM_eci2rtn = self._DCM_eci2rtn(self.r, self.v)
        u_thruster_eci = DCM_eci2rtn.T @ u_thruster_rtn
        return u_thruster_eci

    def action_eci_to_rtn(self
                          )->NDArray:
        u_thruster_eci = self.u
        DCM_eci2rtn = self._DCM_eci2rtn(self.r, self.v)
        u_thruster_rtn = DCM_eci2rtn @ u_thruster_eci
        u_thruster_rtn = np.clip(u_thruster_rtn, -self.cfg.f_max, self.cfg.f_max)
        
        return u_thruster_rtn

    def _currentET(self):
        return self.cfg.eph_time_0 + self.t

    # Celestial Bodies Positions
    def _r_moon_eci(self
                    )->NDArray:
        r_moon_eci, _ = spice.spkpos('MOON', self._currentET(), 'J2000',
                                     'NONE', 'Earth')
        return r_moon_eci*10**3
    
    def _r_sun_eci(self
                   )->NDArray:
        r_sun_eci, _ = spice.spkpos('SUN', self._currentET(), 'J2000',
                                    'NONE', 'Earth')
        return r_sun_eci*10**3
    
    
    def _irregularGrav(self, 
                       r: NDArray, 
                       n_max:int,
                       m_max:int
                       )->NDArray:
        """Irregular Earth Gravity Calculation
        """
        # Earth-fixed position
        r_bf = np.dot(self._DCM_eci2ecef(), r)

        # Auxiliary quantities
        d = np.linalg.norm(r_bf)  # distance
        latgc = np.arcsin(r_bf[2] / d)
        lon = np.arctan2(r_bf[1], r_bf[0])

        pnm, dpnm = self._legrende(n_max, m_max, latgc)
        dUdr = 0
        dUdlatgc = 0
        dUdlon = 0
        q3 = 0
        q2 = q3
        q1 = q2

        for n in range(n_max+1):
            b1 = (-self.cfg.GM_Earth / d ** 2) * (self.cfg.R_Earth / d) ** n * (n + 1)
            b2 = (self.cfg.GM_Earth / d) * (self.cfg.R_Earth / d) ** n
            b3 = (self.cfg.GM_Earth / d) * (self.cfg.R_Earth / d) ** n
            for m in range(m_max+1):
                q1 += pnm[n][m] * (self.cfg.Cnm[n][m] * np.cos(m * lon) + self.cfg.Snm[n][m] * np.sin(m * lon))
                q2 += dpnm[n][m] * (self.cfg.Cnm[n][m] * np.cos(m * lon) + self.cfg.Snm[n][m] * np.sin(m * lon))
                q3 += m * pnm[n][m] * (self.cfg.Snm[n][m] * np.cos(m * lon) - self.cfg.Cnm[n][m] * np.sin(m * lon))
            dUdr += q1 * b1
            dUdlatgc += q2 * b2
            dUdlon += q3 * b3
            q3 = 0
            q2 = q3
            q1 = q2

        # Body-fixed acceleration
        r2xy = r_bf[0] ** 2 + r_bf[1] ** 2

        ax = (1 / d * dUdr - r_bf[2] / (d ** 2 * np.sqrt(r2xy)) * dUdlatgc) * r_bf[0] - (1 / r2xy * dUdlon) * r_bf[1]
        ay = (1 / d * dUdr - r_bf[2] / (d ** 2 * np.sqrt(r2xy)) * dUdlatgc) * r_bf[1] + (1 / r2xy * dUdlon) * r_bf[0]
        az = 1 / d * dUdr * r_bf[2] + np.sqrt(r2xy) / (d ** 2) * dUdlatgc

        a_bf = np.array([ax, ay, az])

        # Inertial acceleration
        a = np.dot(self._DCM_eci2ecef().T, a_bf)

        return a
    
    def _legrende(self,
                  n_max:int,
                  m_max:int,
                  latgc:float
                  )->NDArray:
        
        pnm = np.zeros((n_max + 1, m_max + 1))
        dpnm = np.zeros((n_max + 1, m_max + 1))

        pnm[0, 0] = 1
        dpnm[0, 0] = 0

        fi = latgc

        pnm[1, 1] = np.sqrt(3) * np.cos(fi)
        dpnm[1, 1] = -np.sqrt(3) * np.sin(fi)

        for i in range(2, n_max + 1):
            pnm[i, i] = np.sqrt((2 * i + 1) / (2 * i)) * np.cos(fi) * pnm[i - 1, i - 1]
            dpnm[i, i] = np.sqrt((2 * i + 1) / (2 * i)) * (
                        (np.cos(fi) * dpnm[i - 1, i - 1]) - (np.sin(fi) * pnm[i - 1, i - 1]))

        for i in range(1, n_max + 1):
            pnm[i, i - 1] = np.sqrt(2 * i + 1) * np.sin(fi) * pnm[i - 1, i - 1]
            dpnm[i, i - 1] = (np.sqrt(2 * i + 1) *
                              ((np.cos(fi) * pnm[i - 1, i - 1]) + (np.sin(fi) * dpnm[i - 1, i - 1])))

        j = 0
        k = 2

        while True:
            for i in range(k, n_max + 1):
                pnm[i, j] = np.sqrt((2 * i + 1) / ((i - j) * (i + j))) * (
                            (np.sqrt(2 * i - 1) * np.sin(fi) * pnm[i - 1, j]) - (
                                np.sqrt(((i + j - 1) * (i - j - 1)) / (2 * i - 3)) * pnm[i - 2, j]))
                dpnm[i, j] = np.sqrt((2 * i + 1) / ((i - j) * (i + j))) * (
                            (np.sqrt(2 * i - 1) * np.sin(fi) * dpnm[i - 1, j]) + (
                                np.sqrt(2 * i - 1) * np.cos(fi) * pnm[i - 1, j]) - (
                                        np.sqrt(((i + j - 1) * (i - j - 1)) / (2 * i - 3)) * dpnm[i - 2, j]))
            j += 1
            k += 1

            if j > m_max:
                break

        return pnm, dpnm

    def _j2_acc(self
                )->NDArray:
        r_norm = np.linalg.norm(self.r)
        d1 = - (3 / 2) * self.cfg.j2 * self.cfg.R_Earth ** 2 * self.cfg.GM_Earth / (r_norm ** 5)
        d2 = 1 - 5 * self.r[2] ** 2 / (r_norm ** 2)
        aj2_x = self.r[0] * d1 * d2
        aj2_y = self.r[1] * d1 * d2
        aj2_z = self.r[2] * d1 * (d2 + 2)
        # print(np.array([aj2_x, aj2_y, aj2_z]))
        return np.array([aj2_x, aj2_y, aj2_z])
    
    def _a_thruster(self
                    )->NDArray:
        
        if self.cfg.flag_rtn_thrust: 
            if self.m <= 0 or self.u is None:
                a_thruster_rtn = np.array([0,0,0])
                
                self.u_thruster_eci = np.array([0,0,0])
                self.u_thruster_rtn = np.array([0,0,0])

            else:
                a_thruster_rtn = self.u/self.m
                DCM_eci2rtn = self._DCM_eci2rtn(self.r, self.v)
                a_thruster_eci = DCM_eci2rtn.T @ a_thruster_rtn.reshape(3, 1)
                
                self.u_thruster_eci = a_thruster_eci*self.m
                self.u_thruster_rtn = a_thruster_rtn*self.m
                
                return a_thruster_eci
        
        else:
            if self.m <= 0 or self.u is None:
                a_thruster_eci = np.array([0,0,0])
                
                self.u_thruster_eci = np.array([0,0,0])
                self.u_thruster_rtn = np.array([0,0,0])

            else:
                a_thruster_eci = self.u/self.m

                DCM_eci2rtn = self._DCM_eci2rtn(self.r, self.v)
                a_thruster_rtn = DCM_eci2rtn @ a_thruster_eci.reshape(3, 1)
                a_thruster_rtn = np.clip(a_thruster_rtn, -self.cfg.f_max/self.m, self.cfg.f_max/self.m)
                a_thruster_eci = DCM_eci2rtn.T @ a_thruster_rtn.reshape(3, 1)

                self.u_thruster_eci = a_thruster_eci*self.m
                self.u_thruster_rtn = a_thruster_rtn*self.m

                return a_thruster_eci
            
    def _a_grav_moon(self
                     )->NDArray:
        return self.cfg.GM_Moon * (((self._r_moon_eci()-self.r) /
                             np.linalg.norm(self._r_moon_eci() - self.r)**3) -
                            (self._r_moon_eci() / np.linalg.norm(self._r_moon_eci())**3))
    
    def _a_grav_sun(self
                    )->NDArray:
        return self.cfg.GM_Sun * (((self._r_sun_eci()-self.r) /
                            np.linalg.norm(self._r_sun_eci() -self.r)**3) -
                           (self._r_sun_eci() / np.linalg.norm(self._r_sun_eci())**3))
    
    def _a_drag(self
                )->NDArray:
        self.rho = self._pedm()
        v_rel = self.v - np.cross(np.array([0, 0, self.cfg.omega_Earth]), self.r)
        return - 1 / 2 * ((self.cfg.Ad * self.cfg.Cd) /
                          self.m) * self.rho * np.linalg.norm(v_rel)**2 * (v_rel / np.linalg.norm(v_rel))
      
    def _pedm(self
              )->float:
        """Piece-wise exponential density model
        """
        
        h = np.linalg.norm(self.r) - self.cfg.R_Earth
        # Convert h to kilometers
        h = h / 1000
        # Determine the appropriate parameters based on the value of h
        if 0 <= h < 25:
            h0 = 0
            rho0 = 1.225
            H = 7.249
        elif 25 <= h < 30:
            h0 = 25
            rho0 = 3.899e-2
            H = 6.349
        elif 30 <= h < 40:
            h0 = 30
            rho0 = 1.774e-2
            H = 6.682
        elif 40 <= h < 50:
            h0 = 40
            rho0 = 3.972e-3
            H = 7.554
        elif 50 <= h < 60:
            h0 = 50
            rho0 = 1.057e-3
            H = 8.382
        elif 60 <= h < 70:
            h0 = 60
            rho0 = 3.206e-4
            H = 7.714
        elif 70 <= h < 80:
            h0 = 70
            rho0 = 8.770e-5
            H = 6.549
        elif 80 <= h < 90:
            h0 = 80
            rho0 = 1.905e-5
            H = 5.799
        elif 90 <= h < 100:
            h0 = 90
            rho0 = 3.396e-6
            H = 5.382
        elif 100 <= h < 110:
            h0 = 100
            rho0 = 5.297e-7
            H = 5.877
        elif 110 <= h < 120:
            h0 = 110
            rho0 = 9.661e-8
            H = 7.263
        elif 120 <= h < 130:
            h0 = 120
            rho0 = 2.438e-8
            H = 9.473
        elif 130 <= h < 140:
            h0 = 130
            rho0 = 8.484e-9
            H = 12.636
        elif 140 <= h < 150:
            h0 = 140
            rho0 = 3.845e-9
            H = 16.149
        elif 150 <= h < 180:
            h0 = 150
            rho0 = 2.070e-9
            H = 22.523
        elif 180 <= h < 200:
            h0 = 180
            rho0 = 5.464e-10
            H = 29.740
        elif 200 <= h < 250:
            h0 = 200
            rho0 = 2.789e-10
            H = 37.105
        elif 250 <= h < 300:
            h0 = 250
            rho0 = 7.248e-11
            H = 45.546
        elif 300 <= h < 350:
            h0 = 300
            rho0 = 2.418e-11
            H = 53.628
        elif 350 <= h < 400:
            h0 = 350
            rho0 = 9.518e-12
            H = 53.298
        elif 400 <= h < 450:
            h0 = 400
            rho0 = 3.725e-12
            H = 58.515
        elif 450 <= h < 500:
            h0 = 450
            rho0 = 1.585e-12
            H = 60.828
        elif 500 <= h < 600:
            h0 = 500
            rho0 = 6.967e-13
            H = 63.822
        elif 600 <= h < 700:
            h0 = 600
            rho0 = 1.454e-13
            H = 71.835
        elif 700 <= h < 800:
            h0 = 700
            rho0 = 3.614e-14
            H = 88.667
        elif 800 <= h < 900:
            h0 = 800
            rho0 = 1.170e-14
            H = 124.64
        elif 900 <= h < 1000:
            h0 = 900
            rho0 = 5.245e-15
            H = 181.05
        elif 1000 <= h:
            h0 = 1000
            rho0 = 3.019e-15
            H = 268.00
        else:
            h0 = 1000
            rho0 = 3.019e-15
            H = 268.00

        # Calculate and return the density
        rho = rho0 * np.exp(-(h - h0) / H)
        return rho
   
    def _a_srp(self
               )->NDArray:
        # solar radiation pressure
        # dual conical shadow model (Montenbrück & Gill 2000)
        s = self.r
        a = np.arcsin(self.cfg.R_Sun / np.linalg.norm(self._r_sun_eci() - self.r))
        b = np.arcsin(self.cfg.R_Earth/ np.linalg.norm(s))
        c = np.arccos(np.dot(-s, (self._r_sun_eci() - self.r)) /
                      (np.linalg.norm(s) * np.linalg.norm(self._r_sun_eci() - self.r)))
        X = (c ** 2 + a ** 2 - b ** 2) / (2 * c)
        Y = np.sqrt(a ** 2 - X ** 2)
        A = a ** 2 * np.arccos(X / a) + b ** 2 * np.arccos((c - X) / b) - c * Y

        if c >= a + b:  # no eclipse
            nu = 1
        elif np.linalg.norm(a - b) < c and c < a + b:  # penumbra
            nu = 1 - A / (np.pi * a ** 2)
        elif c < b - a and b > a:  # umbra
            nu = 0
        elif c < a - b and a > b:  # penumbra maximum
            nu = 1 - (b ** 2 / a ** 2)
        else:
            nu = 0
        return (nu * self.cfg.P * (self.cfg.AU ** 2) * self.cfg.Cr * self.cfg.As / self.m * (self.r - self._r_sun_eci())
                / (np.linalg.norm(self.r - self._r_sun_eci()) ** 3))

    def _a_grav(self, 
                flag:int=0
                )->NDArray:
        if flag == 1:
            return self._irregularGrav(self.r,20,20)
        elif flag == 2:
            return self._irregularGrav(self.r,2,2)
        elif flag == 0:
            return (-self.cfg.GM_Earth / (np.linalg.norm(self.r) ** 3)) * self.r
        
    def _total_acc(self
                   )->NDArray:
        
        pert_sun = np.zeros(3)
        pert_moon = np.zeros(3)
        pert_srp = np.zeros(3)
        pert_drag = np.zeros(3)
        
        if self.cfg.flag_pert_moon:
            pert_moon  = self._a_grav_moon()
            
        if self.cfg.flag_pert_sun:
            pert_sun  = self._a_grav_sun()
            
        if self.cfg.flag_pert_srp:
            pert_srp  = self._a_srp()
            
        if self.cfg.flag_pert_drag:
            pert_drag = self._a_drag()
            
        if self.cfg.flag_pert_irr_grav:
            gravity = self._a_grav(2)
        else:
            gravity = self._a_grav(0)

        thrust = np.squeeze(self._a_thruster())
            
        return (gravity
                + thrust
                + pert_moon
                + pert_sun
                + pert_srp
                + pert_drag
                )

    def _massflow(self
                  )->float:
        if self.cfg.flag_mass_loss:
            return -np.linalg.norm(self._a_thruster() * self.m)/(self.cfg.g0 * self.cfg.Isp)
        else:
            return 0
    
    
    def get_current_pertubration_forces(self,
                                        x:NDArray,
                                        t:float
                                        )->Tuple[NDArray,
                                                 NDArray,
                                                 NDArray,
                                                 NDArray,
                                                 NDArray]: 
        """Get current perturbation forces

        Args:
            x (arr 6,): space object position and velocity 
            t (int): relative elapsed time from starting ephemeris time

        Returns:
            tuple: gravity, irregular gravity, moon, sun, srp, air drag
        """
        
        self.r = x[:3]
        self.v = x[3:6]
        self.t = t
        
        return  self._a_grav(2),\
                self._a_grav_moon(),\
                self._a_grav_sun(),\
                self._a_srp(),\
                self._a_drag()
                

    def get_lat_lon(self, 
                    r:NDArray
                    )->Tuple[float,float]:
        """get Sat Lat Lon
        """
        
        E = self._DCM_eci2ecef()
        
        # Earth-fixed position
        r_bf = np.dot(E, r)

        # Auxiliary quantities
        d = np.linalg.norm(r_bf)  # distance
        lat = np.arcsin(r_bf[2] / d)
        lon = np.arctan2(r_bf[1], r_bf[0])
        
        return lat,lon

    
    def dxdt(self,x: NDArray,
             u:NDArray,
             t:float
             )-> NDArray: 
        
        self.r = x[:3]
        self.v = x[3:6]
        self.u = u
        self.t = t
        
        return np.concatenate((self.v, self._total_acc(), [self._massflow()]),axis=0)

    def propagate(self,x:NDArray,
                  u:NDArray,
                  t:float,
                  dt:float, 
                  rk4_flag:bool=True
                  )->NDArray:
        
        u = np.clip(u,-self.cfg.f_max,self.cfg.f_max)
                
        if rk4_flag:
            return PropagatorModels.rk4(x, u, self.dxdt, dt, t)
        else:
            return PropagatorModels.rk8mod(x, u, self.dxdt, dt, t)

