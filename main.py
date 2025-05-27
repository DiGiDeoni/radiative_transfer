import numpy as np
import matplotlib.pyplot as plt
import argparse

from radiative_solver.constants import c, a_rad
from radiative_solver.grid import create_grid
from radiative_solver.initial_conditions import initial_energy_density
from radiative_solver.material_properties import (
    temperature_from_energy, opacity_from_temperature, density_profile
)
from radiative_solver.diffusion import compute_diffusion_coefficient, build_matrix
from radiative_solver.solvers import solve_linear_system
from radiative_solver.heating import stellar_heating

def main(solver_name="bicgstab", max_steps=2000, dt=1e4, tol=1e-8):
    r, dr = create_grid()
    Nr = len(r)
    r_min, r_max = r[0], r[-1]

    E = initial_energy_density(r, r_min, r_max)
    T = temperature_from_energy(E)
    kappa = opacity_from_temperature(T)
    rho = density_profile(r, r_min)
    D = compute_diffusion_coefficient(c, kappa, rho)
    A = build_matrix(D, dr, dt, Nr)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    line_E, = ax1.plot(r, E)
    line_T, = ax2.plot(r, T)
    ax1.set_ylabel("Energy Density")
    ax2.set_ylabel("Temperature (K)")
    ax2.set_xlabel("Radius (cm)")

    for step in range(max_steps):
        E_old = E.copy()

        Q_star = stellar_heating(r)
        b = (E_old + Q_star * dt) / dt
        b[0] = 1e-5
        b[-1] = 0

        E, info = solve_linear_system(A, b, x0=E_old, method=solver_name, tol=tol)

        if info != 0:
            print(f"Solver failed at step {step}, info = {info}")
            break

        T = temperature_from_energy(E)
        kappa = opacity_from_temperature(T)
        rho = density_profile(r, r_min)
        D = compute_diffusion_coefficient(c, kappa, rho)
        A = build_matrix(D, dr, dt, Nr)

        if step % 10 == 0:
            print(f"Step {step}, Max T: {np.max(T):.2f} K")
            line_E.set_ydata(E)
            line_T.set_ydata(T)
            ax1.relim(); ax1.autoscale_view()
            ax2.relim(); ax2.autoscale_view()
            plt.pause(0.01)

        if np.max(np.abs(E - E_old)) < tol:
            print(f"Converged at step {step}")
            break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="bicgstab")
    args = parser.parse_args()
    main(solver_name=args.solver)