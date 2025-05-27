import numpy as np

def stellar_heating(r, L_star=3.828e33):
    # Isotropic heating: simplified radiative flux
    return L_star / (4 * np.pi * r**2)