import numpy as np
from scipy.sparse import diags

def compute_diffusion_coefficient(c, kappa, rho):
    return c / (3 * kappa * rho)

def build_matrix(D, dr, dt, Nr):
    alpha = D / dr**2
    main_diag = (1.0 / dt + 2 * alpha)
    off_diag = -alpha[:-1]
    A = diags([main_diag, off_diag, off_diag], [0, -1, 1], format='csr')

    A = A.tolil()
    A[0, :] = 0
    A[0, 0] = 1
    A[-1, -2] = -1 / dr
    A[-1, -1] = 1 / dr
    return A.tocsr()