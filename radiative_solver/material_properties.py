import numpy as np
from .constants import a_rad, kappa_0, rho_0

def temperature_from_energy(E):
    T = (np.maximum(E, 1e-20) / a_rad)**0.25
    return np.clip(T, 10.0, 2000.0)

def opacity_from_temperature(T):
    return kappa_0 * (T / 100.0)**2

def density_profile(r, r_min):
    return rho_0 * (r / r_min)**-1.5