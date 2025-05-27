import numpy as np

def create_grid(Nr=200, r_min=1.5e13*0.1, r_max=1.5e13*500):
    r = np.linspace(r_min, r_max, Nr)
    dr = r[1] - r[0]
    return r, dr