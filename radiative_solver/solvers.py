from scipy.sparse.linalg import bicgstab, gmres, spsolve

def solve_linear_system(A, b, x0, method, tol):
    if method == "bicgstab":
        return bicgstab(A, b, x0=x0, tol=tol)
    elif method == "gmres":
        return gmres(A, b, x0=x0, tol=tol)
    elif method == "implicit":
        return spsolve(A, b), 0
    else:
        raise ValueError(f"Unknown solver {method}")