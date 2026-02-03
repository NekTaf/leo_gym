# Standard library
from typing import Tuple

# Third-party
import numpy as np
import pyorb
from numpy.typing import NDArray
from zendir.maths.astro import (
    classical_to_non_singular_elements,
    vector_to_classical_elements,
    vector_to_relative_elements_mean,
    classical_to_vector_elements,
    classical_to_relative_elements_mean
)



def minpipi(x):
    """
    Wrap angle(s) to the range (-π, π].

    :param x: Angle or array of angles in radians.
    :return: Wrapped angle(s) in radians.
    """
    x = np.mod(x + np.pi, 2 * np.pi) - np.pi
    return x

def zero2twopi(x):
    """
    Wrap angle(s) to the range [0, 2π).

    :param x: Angle or array of angles in radians.
    :return: Wrapped angle(s) in radians.
    """
    return np.mod(x, 2 * np.pi)


def rv_to_roe_and_nsoe(
    rv_d: NDArray,
    rv_c: NDArray
    ) -> Tuple[NDArray, 
               NDArray, 
               NDArray]:
    
    """
    Convert Cartesian state vectors into relative orbital elements (ROE)
    and non-singular orbital elements for both deputy and chief.

    :param rv_d: Deputy satellite state vector in ECI — position [m] and velocity [m/s] (arr of shape (6,))
    :param rv_c: Chief satellite state vector in ECI — position [m] and velocity [m/s] (arr of shape (6,))
    :return:
        - Relative orbital elements (arr of shape (6,))
        - Deputy non-singular orbital elements (arr of shape (6,))
        - Chief non-singular orbital elements (arr of shape (6,))
    """

    oe_kep = vector_to_classical_elements(
        r_bn_n=rv_d[:3],
        v_bn_n=rv_d[-3:],
        planet="earth")
    
    oe_kep = np.array(oe_kep).reshape(6,)

    
    oe_ns = classical_to_non_singular_elements(
        semi_major_axis=oe_kep[0],
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
    
    oe_kep_r = vector_to_classical_elements(
        r_bn_n=rv_c[:3],
        v_bn_n=rv_c[-3:],
        planet="earth")
    
    oe_ns_r = classical_to_non_singular_elements(
        semi_major_axis=oe_kep_r[0],
        eccentricity=oe_kep_r[1],
        inclination=oe_kep_r[2],
        right_ascension=oe_kep_r[3],
        argument_of_periapsis=oe_kep_r[4],
        true_anomaly=oe_kep_r[5])
    
    oe_ns_r = np.array(
        [oe_ns_r[0],
        oe_ns_r[5],
        oe_ns_r[1],
        oe_ns_r[2],
        oe_ns_r[3],
        oe_ns_r[4]]
        ).reshape(6,)
    
    oe_kep_r = np.array(oe_kep_r).reshape(6,)
    
    oe_kep_r[1] = zero2twopi(oe_kep_r[1])
    oe_kep_r[2] = zero2twopi(oe_kep_r[2])
    oe_kep_r[3] = zero2twopi(oe_kep_r[3])
    oe_kep_r[4] = zero2twopi(oe_kep_r[4])
    oe_kep_r[5] = zero2twopi(oe_kep_r[5])

    oe_kep[1] = zero2twopi(oe_kep[1])
    oe_kep[2] = zero2twopi(oe_kep[2])
    oe_kep[3] = zero2twopi(oe_kep[3])
    oe_kep[4] = zero2twopi(oe_kep[4])
    oe_kep[5] = zero2twopi(oe_kep[5])
    
    roe = classical_to_relative_elements_mean(
        semi_major_axis_leader=oe_kep_r[0],
        eccentricity_leader=oe_kep_r[1],
        inclination_leader=oe_kep_r[2],
        right_ascension_leader=oe_kep_r[3],
        argument_of_periapsis_leader=oe_kep_r[4],
        true_anomaly_leader=oe_kep_r[5],
        
        semi_major_axis_follower=oe_kep[0],
        eccentricity_follower=oe_kep[1],
        inclination_follower=oe_kep[2],
        right_ascension_follower=oe_kep[3],
        argument_of_periapsis_follower=oe_kep[4],
        true_anomaly_follower=oe_kep[5],
        )
        
    roe = np.array(roe).reshape(6,)
    roe = np.array(([roe[0],roe[5],roe[1],roe[4],roe[2],roe[3]])).reshape(6,)
                
    return roe, oe_ns, oe_ns_r
    
def _normc(m):
    norm = np.linalg.norm(m, axis=0)
    return m / norm if np.all(norm != 0) else m

def rv_2_non_singular_oe(
    rv:NDArray, 
    mu:float=3.986004415000000e+14
    )-> NDArray:
    
    
    rv = rv.reshape(6, 1)
    n = rv.shape[1]
    
    rvec = rv[0:3, :]
    vvec = rv[3:6, :]
    
    r = np.linalg.norm(rvec, axis=0)
    v = np.linalg.norm(vvec, axis=0)
    
    hvec = np.cross(rvec.T, vvec.T).T
    evec = (1/mu) * ((v**2 - (mu / r)) * rvec - np.sum(rvec * vvec, axis=0) * vvec)
    
    Esp = (v**2 / 2) - mu / r
    a = -mu / (2 * Esp)
    Ivecp = np.array([-hvec[1, :], hvec[0, :], np.zeros(n)])
    
    assert not np.any((hvec[0, :] == 0) & (hvec[1, :] == 0)), 'This set of orbital elements cannot deal with an inclination of zero.'
    
    xlon = _normc(Ivecp)
    zlon = _normc(hvec)
    ylon = np.cross(zlon.T, xlon.T).T
    dcmlon = np.stack([xlon, ylon, zlon], axis=0)  # Shape (3, 3, n)
    
    evec = evec.reshape(3, n)
    evecPeri = np.einsum('ijk,jk->ik', dcmlon[:, :, :2], evec)  # Shape (2, n)
    evecPeri = evecPeri[:2]

    # Properly calculate the dot products
    la = np.arctan2(np.dot(dcmlon[1, :, :].T, rvec).squeeze(), np.dot(dcmlon[0, :, :].T, rvec).squeeze())
    RAAN = np.arctan2(Ivecp[1, :], Ivecp[0, :])
    
    klon = dcmlon[2, :, :]
    i = np.arctan2(klon[1, :], klon[2, :])
    w = np.arctan2(evecPeri[1, :], evecPeri[0, :])
    v = la - w
    e = np.linalg.norm(evecPeri, axis=0)
    E = 2 * np.arctan(np.sqrt((1-e) / (1+e)) * np.tan(v/2))
    M = E - e * np.sin(E)
    u = np.mod(M + w, 2 * np.pi)
    
    oe_ns = np.vstack([a, evecPeri, -i, np.mod(np.vstack([RAAN, u, la]), 2 * np.pi)])
    
    return oe_ns

def calculate_ecc_anomaly(M:float, 
                          ecc:float,
                          tol:float=1e-10,
                          maxIter:int=20):
    M = np.mod(M, 2 * np.pi)
    E = np.where(M > np.pi, M - ecc, M + ecc)
    for iter_count in range(maxIter):
        dE = (M - E + ecc * np.sin(E)) / (1 - ecc * np.cos(E))
        E += dE
        if np.all(np.abs(dE) < tol):
            break
    else:
        raise RuntimeError(f'Not converged in {maxIter} iterations')
    return E

def calculate_true_anomaly(M: float, 
                           ecc: float, 
                           tol:float=1e-10,
                           maxIter:int=20
                           )-> float:
    
    E = calculate_ecc_anomaly(M, ecc, tol, maxIter)
    TA = np.mod(2 * np.arctan(np.tan(E / 2) * np.sqrt((1 + ecc) / (1 - ecc))), 2 * np.pi)
    return TA


def osc_2_mean_non_singular_oe_brouwer_lyddane_method(oe:NDArray, 
                                                      r_E:float=6378136.3,
                                                      j2:float=1082.19e-6
                                                      )->NDArray:
    """
    Convert osculating non-singular orbital elements to mean non-singular orbital elements 
    using the Brouwer–Lyddane analytical method.

    :param oe: Osculating non-singular orbital elements (a, e_x, e_y, i, \Omega, u, \lambda) (arr of shape (7, N))
    :param r_E: Equatorial radius of the central body [m] (float)
    :param j2: Second zonal harmonic coefficient of the gravitational potential (float)
    :return: Mean non-singular orbital elements (a, e_x, e_y, i, \Omega, u, \lambda) (arr of shape (7, N))
    """
    
    a = oe[0, :]
    ex = oe[1, :]
    ey = oe[2, :]
    i = oe[3, :]
    Om = oe[4, :]
    u = oe[5, :]
    la = oe[6, :]

    e = np.sqrt(ex**2 + ey**2)
    om = np.mod(np.arctan2(ey, ex), 2 * np.pi)
    
    M = np.mod(u - om, 2 * np.pi)
    f = np.mod(la - om, 2 * np.pi)
    
    ga2 = -j2 / 2 * (r_E / a)**2
    et = np.sqrt(1 - e**2)
    gap2 = ga2 / et**4
    ar = (1 + e * np.cos(f)) / et**2

    ap = a + a * ga2 * ((3 * np.cos(i)**2 - 1) * (ar**3 - 1 / et**3) +
                        3 * (1 - np.cos(i)**2) * ar**3 * np.cos(2 * om + 2 * f))

    de1 = gap2 / 8 * e * et**2 * (1 - 11 * np.cos(i)**2 - 40 * np.cos(i)**4 / (1 - 5 * np.cos(i)**2)) * np.cos(2 * om)
    de = de1 + et**2 / 2 * (ga2 * ((3 * np.cos(i)**2 - 1) / et**6 * (e * et + e / (1 + et) + 3 * np.cos(f) +
        3 * e * np.cos(f)**2 + e**2 * np.cos(f)**3) + 3 * (1 - np.cos(i)**2) / et**6 * (e + 3 * np.cos(f) +
        3 * e * np.cos(f)**2 + e**2 * np.cos(f)**3) * np.cos(2 * om + 2 * f)) -
        gap2 * (1 - np.cos(i)**2) * (3 * np.cos(2 * om + f) + np.cos(2 * om + 3 * f)))

    di = -e * de1 / (et**2 * np.tan(i)) + gap2 / 2 * np.cos(i) * np.sqrt(1 - np.cos(i)**2) * (
        3 * np.cos(2 * om + 2 * f) + 3 * e * np.cos(2 * om + f) + e * np.cos(2 * om + 3 * f))
    di[np.isnan(di)] = 0  # Handle NaNs if any

    # Extended calculations for MpompOmp, dOm, d1, d2, Mp, ep, d3, d4, Omp, ip, omp, fp, exp, eyp, up, lap
    MpompOmp = M + om + Om  + gap2/8.*et**3.*(1-11*np.cos(i)**2-40*np.cos(i)**4/(1-5*np.cos(i)**2))
    edM = (gap2 / 8 * e * et**3 * (1 - 11 * np.cos(i)**2 - 40 * np.cos(i)**4 / (1 - 5 * np.cos(i)**2))
       - gap2 / 4 * et**3 * (2 * (3 * np.cos(i)**2 - 1) * (ar * et**2 + ar + 1) * np.sin(f)
       + 3 * (1 - np.cos(i)**2) * ((-(ar * et)**2 - ar + 1) * np.sin(2 * om + f)
       + (ar * et**2 + ar + 1/3) * np.sin(2 * om + 3 * f))))

    dOm = (-gap2 / 8 * e**2 * np.cos(i) * 
        (11 + 80 * np.cos(i)**2 / (1 - 5 * np.cos(i)**2) + 
            200 * np.cos(i)**4 / (1 - 5 * np.cos(i)**2)**2)
        - gap2 / 2 * np.cos(i) * 
        (6 * (f - M + e * np.sin(f)) - 
            3 * np.sin(2 * om + 2 * f) - 
            3 * e * np.sin(2 * om + f) - 
            e * np.sin(2 * om + 3 * f)))

    d1 = (e+de)*np.sin(M) + edM*np.cos(M)
    d2 = (e+de)*np.cos(M) - edM*np.sin(M)
    
    Mp = np.arctan2(d1, d2)
    ep = np.sqrt(d1**2 + d2**2)

    d3 = (np.sin(i/2) + np.cos(i/2) * di/2) * np.sin(Om) + np.sin(i/2) * dOm * np.cos(Om)
    d4 = (np.sin(i/2) + np.cos(i/2) * di/2) * np.cos(Om) - np.sin(i/2) * dOm * np.sin(Om)
    
    Omp = np.arctan2(d3, d4)
    ip = 2 * np.arcsin(np.sqrt(d3**2 + d4**2))

    omp = MpompOmp - Mp - Omp
    exp = ep * np.cos(omp)
    eyp = ep * np.sin(omp)
    up = Mp + omp
    fp = calculate_true_anomaly(Mp,ep)
    lap = fp + omp

    oeMean = np.vstack([ap, exp, eyp, ip, np.mod(Omp, 2 * np.pi), np.mod(up, 2 * np.pi), np.mod(lap, 2 * np.pi)])
        
    return oeMean


def damico_relative_elements(
    oe_r:NDArray,
    oe:NDArray
    )->NDArray:

    doe = np.zeros_like(oe)
    doe[0, :] = (oe[0, :] - oe_r[0, :]) / (oe_r[0, :] + 1)
    doe[1, :] = minpipi(oe[1, :] - oe_r[1, :]) + minpipi(oe[5, :] - oe_r[5, :]) * np.cos(oe_r[4, :])
    doe[2:5, :] = oe[2:5, :] - oe_r[2:5, :]
    doe[5, :] = minpipi(oe[5, :] - oe_r[5, :]) * np.sin(oe_r[4, :])
    
    return doe
    
def nonsingular_to_classical_oe(
    semi_major_axis: float,
    e_x: float,
    e_y: float,
    inclination: float,
    right_ascension: float,
    mean_argument_of_latitude: float,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    :param semi_major_axis: Semi-major axis [m]
    :param e_x: x component of eccentricity (e*cos(ω))
    :param e_y: y component of eccentricity (e*sin(ω))
    :param inclination: Inclination i [rad]
    :param right_ascension: Right ascension of ascending node Ω [rad]
    :param mean_argument_of_latitude: Mean argument of latitude ū = ω + M [rad]
    :return: (a, e, i, Ω, ω, M, ν)
    """
    e = np.hypot(e_x, e_y)
    omega = np.arctan2(e_y, e_x)
    Omega = right_ascension
    M = (mean_argument_of_latitude - omega) % (2*np.pi)

    # Kepler solve for E, then ν
    E = M if np.isscalar(M) else M.copy()
    for _ in range(20):
        E -= (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
    nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2)) % (2*np.pi)

    return (
        semi_major_axis,
        e,
        inclination,
        Omega % (2*np.pi),
        omega % (2*np.pi),
        M % (2*np.pi),
        nu,
    )

       
def dcm_eci_2_pf(
    inc:float, 
    raan:float, 
    aop:float
    )->NDArray:
    
    """
    ECI to Perifocal frame 
    
    :param inc: Inclination (float)
    :param raan: Right Ascension of the Ascending Node (float)
    :param aop: Argument of Periapsis (float)
    
    :returns: Direction Cosine Matrix ECI->PF (arr (3,3))
    
    """

    dcm = np.zeros((3, 3))

    dcm[0, 0] = np.cos(raan) * np.cos(aop) - np.sin(raan) * np.sin(aop) * np.cos(inc)
    dcm[0, 1] = np.sin(raan) * np.cos(aop) + np.cos(raan) * np.sin(aop) * np.cos(inc)
    dcm[0, 2] = np.sin(aop) * np.sin(inc)

    dcm[1, 0] = -np.cos(raan) * np.sin(aop) - np.sin(raan) * np.cos(aop) * np.cos(inc)
    dcm[1, 1] = -np.sin(raan) * np.sin(aop) + np.cos(raan) * np.cos(aop) * np.cos(inc)
    dcm[1, 2] = np.cos(aop) * np.sin(inc)

    dcm[2, 0] = np.sin(raan) * np.sin(inc)
    dcm[2, 1] = -np.cos(raan) * np.sin(inc)
    dcm[2, 2] = np.cos(inc)

    return dcm

def rv_2_kepler_oe_pyorb(
    rv:NDArray,
    mu:float=3.986004415000000e+14, 
    degrees:bool=False
    )->NDArray:
    """Cartesian to orbital elelment wrapper
    
    :param rv: ECI position and velocity vector (6,)
    :param mu: Gravitational parameter, defaults to Earth's μ if not provided
    :param degrees: If True, output angles are in degrees; if False, in radians
    :return: Orbital elements (6,) → [a, e, i, \omega, \Omega, \nu]
    """
    
    kep = pyorb.cart_to_kep(rv, 
                            mu=mu, 
                            degrees=degrees,)
    return kep


def kepler_oe_2_rv_pyorb(oe_kep:NDArray,
                         mu:float=3.986004415000000e+14,
                         degrees:bool=False
                         )-> NDArray:
    
    r"""
    Orbital element to Cartesian wrapper.

    :param oe: Classical Keplerian orbital elements — (a, e, i, \omega, \Omega, \nu) (arr of shape (6,))
    :param mu: Gravitational parameter, defaults to Earth's \mu if not provided (float, optional)
    :param degrees: If True, input angles are in degrees; if False, in radians (bool)
    :return: Cartesian state vector in the ECI frame (arr of shape (6,))
    """
    
    rv = pyorb.kep_to_cart(oe_kep, 
                           mu=mu, 
                           degrees=degrees,)
    return rv


def delta_rv_rtn(rv_1:NDArray, 
                 rv_2:NDArray)->NDArray:
    r"""
    Orbital element to Cartesian wrapper.

    :param oe: Classical Keplerian orbital elements (a, e, i, \omega, \Omega, \nu) (arr of shape (6,))
    :param mu: Gravitational parameter, defaults to Earth's $\mu$ if not provided (float, optional)
    :param degrees: If True, input angles are in degrees; if False, in radians (bool)
    :return: Cartesian state vector in the ECI frame (arr of shape (6,))
    """
    A = dcm_eci_2_rtn(rv_1[:3],rv_1[3:6])
    seperation = A@(rv_1[:3]-rv_2[:3])
    
    return seperation



def dcm_eci_2_rtn(r:NDArray, 
                  v:NDArray)->NDArray:
    """
    Hill frame transformation DCM from ECI to RTN.

    :param r: ECI position vector (arr of shape (3,))
    :param v: ECI velocity vector (arr of shape (3,))
    :return: ECI→RTN direction cosine matrix (arr of shape (3,3))
    """
    r_norm = r / np.linalg.norm(r)
    v_norm = v / np.linalg.norm(v)
    # Orbit normal
    h_norm = (np.cross(r_norm, v_norm)
                / np.linalg.norm(np.cross(r_norm, v_norm)))
    # Orbit tangential
    t_norm = np.cross(h_norm, r_norm)
    return np.vstack((r_norm, t_norm, h_norm))


def dcm_eci_2_b_plane(rv_p:NDArray, 
                      rv_s:NDArray)->NDArray:
    """
    Matrix for frame conversion from ECI to B-plane.

    :param rv_p: Primary object Cartesian state vector (arr of shape (6,))
    :param rv_s: Secondary object Cartesian state vector (arr of shape (6,))
    :return: Direction cosine matrix for the conversion (arr of shape (3,3))
    """
        
    eta = (rv_p[3:6] - rv_s[3:6])/np.linalg.norm(rv_p[3:6] - rv_s[3:6])    
    xi = np.cross(rv_s[3:6],rv_p[3:6])/np.linalg.norm(np.cross(rv_s[3:6],rv_p[3:6]))
    zeta = np.cross(xi,eta)
    
    return np.vstack((xi, eta, zeta))


def delta_r_eci_2_rb(rv_p: NDArray, 
                     rv_s: NDArray)-> NDArray:
    """
    Compute the relative difference between a debris object and the primary satellite,
    transforming from ECI coordinates to the B-plane ({xi, eta, zeta}).

    :param sat_rv: Primary satellite ECI position and velocity (arr of shape (6,))
    :param deb_rv: Secondary debris ECI position and velocity (arr of shape (6,))
    :return: Relative separation in the B-plane frame (arr of shape (3,))
    """
    
    rv_p = np.array(rv_p[:6])
    rv_s = np.array(rv_s[:6])
    
    A = dcm_eci_2_b_plane(rv_p, rv_s)
    
    delta_r = A@(rv_p[:3]-rv_s[:3])

    return np.array(delta_r).reshape(3,)



def covariance_converisions(rv_p:NDArray, 
                            rv_s:NDArray,
                            C_rtn_p:NDArray, 
                            C_rtn_s:NDArray)->Tuple[NDArray,NDArray]:

    """
    Convert combined covariance between primary (satellite) and secondary (debris) objects.

    :param rv_p: Primary object Cartesian state vector (arr of shape (6,))
    :param rv_s: Secondary object Cartesian state vector (arr of shape (6,))
    :return: 
        - Combined covariance matrix in the B-plane frame (arr)
        - Combined covariance matrix in the ECI frame (arr)
    """
    R_eci2rtn_p = dcm_eci_2_rtn(r=np.array(rv_p).reshape(-1)[:3], 
                                v=np.array(rv_p).reshape(-1)[3:6])
    
    R_eci2rtn_s = dcm_eci_2_rtn(r=np.array(rv_s).reshape(-1)[:3], 
                            v=np.array(rv_s).reshape(-1)[3:6])

    C_eci_p = R_eci2rtn_p.T @ C_rtn_p @ R_eci2rtn_p
    C_eci_s = R_eci2rtn_s.T @ C_rtn_s @ R_eci2rtn_s

    C_eci_combined = C_eci_p + C_eci_s

    R_eci2b = dcm_eci_2_b_plane(rv_p=rv_p, rv_s=rv_s)
    
    C_b_combined = R_eci2b @ C_eci_combined @ R_eci2b.T
    
    return C_b_combined, C_eci_combined




def Delta_roe_to_rv(
    Droe:NDArray,
    rv_ref:NDArray
    )-> NDArray:

    """
    Returns real trajectory point based on displacement ROE vector
    and nominal trajectory  
    
    :param Droe: (arr (6,)) Displacement in ROE [da,dl,dex,dey,dix,diy]
    :param rv_ref: (arr (6,)) Postion velocity vector 
    """
    rv_ref = np.asarray(rv_ref).reshape(-1)   # shape (6,)

    oe_kep_ref = vector_to_classical_elements(
        r_bn_n=rv_ref[:3],
        v_bn_n=rv_ref[-3:],
        planet="earth")
    
    oe_ns_ref = classical_to_non_singular_elements(
        semi_major_axis=oe_kep_ref[0],
        eccentricity=oe_kep_ref[1],
        inclination=oe_kep_ref[2],
        right_ascension=oe_kep_ref[3],
        argument_of_periapsis=oe_kep_ref[4],
        true_anomaly=oe_kep_ref[5]
        )
    
    oe_ns_ref = np.array([
        oe_ns_ref[0],
        oe_ns_ref[5],
        oe_ns_ref[1],
        oe_ns_ref[2],
        oe_ns_ref[3],
        oe_ns_ref[4]]).reshape(6,)

    Droe[0] = Droe[0]*oe_ns_ref[0]

    oe_ns_real = oe_ns_ref - Droe/oe_ns_ref[0]
    
    oe_real = nonsingular_to_classical_oe(semi_major_axis=oe_ns_real[0],
                                          e_x=oe_ns_real[2],
                                          e_y=oe_ns_real[3],
                                          inclination=oe_ns_real[4],
                                          mean_argument_of_latitude=oe_ns_real[1],
                                          right_ascension=oe_ns_real[5])
    
    rv = classical_to_vector_elements(
        semi_major_axis=oe_real[0],
        eccentricity=oe_real[1],
        inclination=oe_real[2],
        right_ascension=oe_real[3],
        argument_of_periapsis=oe_real[4],
        true_anomaly=oe_real[6])
    
    rv = np.array(rv).reshape(6,)
        
    return rv

