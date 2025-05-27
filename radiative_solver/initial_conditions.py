import numpy as np

def initial_energy_density(r, r_min, r_max):
    return 1e-5 * np.exp(-((r - r_min) / (r_max - r_min))**2)