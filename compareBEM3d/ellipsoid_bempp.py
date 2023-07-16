import bempp.api
import numpy as np
import time
import matplotlib.pyplot as plt
from bempp.api.linalg import lu
from platform import python_version
from bempp.api.operators.boundary.maxwell import multitrace_operator
from bempp.api.operators.boundary.sparse import multitrace_identity
import sys

# main code
if __name__ == "__main__":

    start0 = time.time()

    print("Python version: ", python_version())
    
    mode = sys.argv[1]
    print("polarization T"+mode)

    filename = 'mesh/BEM/ellipsoid_3_3_2_sub4.msh'
    grid = bempp.api.import_grid(filename)

    vacuum_permittivity = 8.854187817E-12
    vacuum_permeability = 4 * np.pi * 1E-7
    c0 = 299792458

    wavelength = 0.5
    frequency = c0 / wavelength

    ior = 1.55 + 0.1j

    eps_r = ior**2
    mu_r = 1.0

    k_ext = 2 * np.pi * frequency * np.sqrt(vacuum_permittivity * vacuum_permeability)
    k_int = k_ext * np.sqrt(eps_r * mu_r)

    theta = 0 # Incident wave travelling at a 0 degree angle
    direction = np.array([np.cos(theta), np.sin(theta), 0])
    if mode == 'M':
        polarization = np.array([0, 0, 1.0]) # TM
    else:
        polarization = np.array([0, -1.0, 0]) # TE

    def plane_wave(point):
        return polarization * np.exp(1j * k_ext * np.dot(point, direction))

    @bempp.api.complex_callable
    def tangential_trace(point, n, domain_index, result):
        value = polarization * np.exp(1j * k_ext * np.dot(point, direction))
        result[:] =  np.cross(value, n)

    @bempp.api.complex_callable
    def neumann_trace(point, n, domain_index, result):
        value = np.cross(direction, polarization) * 1j * k_ext * np.exp(1j * k_ext * np.dot(point, direction))
        result[:] =  1./ (1j * k_ext) * np.cross(value, n)

    start = time.time()
    A0_int = multitrace_operator(
        grid, k_int, epsilon_r=eps_r, mu_r=mu_r, space_type='all_rwg', assembler='dense', device_interface="numba")
    A0_ext = multitrace_operator(
        grid, k_ext, space_type='all_rwg', assembler='dense', device_interface="numba")
    A = bempp.api.GeneralizedBlockedOperator([[A0_int + A0_ext]])
    end = time.time()
    print("numba", time.time() - start)

    rhs = [bempp.api.GridFunction(space=A.range_spaces[0], dual_space=A.dual_to_range_spaces[0], fun=tangential_trace),
       bempp.api.GridFunction(space=A.range_spaces[1], dual_space=A.dual_to_range_spaces[1], fun=neumann_trace)]
    
    # solve
    print("Solving...")
    bempp.api.enable_console_logging()
    sol = bempp.api.linalg.lu(A, rhs)

    # far field
    print("Far field...")
    number_of_angles = 3600
    angles = 2*np.pi * np.linspace(0, 1, number_of_angles)
    unit_points = np.array([-np.cos(angles), -np.sin(angles), np.zeros(number_of_angles)])

    far_field = np.zeros((3, number_of_angles), dtype='complex128')

    electric_far = bempp.api.operators.far_field.maxwell.electric_field(sol[1].space, unit_points, k_ext)
    magnetic_far = bempp.api.operators.far_field.maxwell.magnetic_field(sol[0].space, unit_points, k_ext)    
    far_field += -electric_far * sol[1] - magnetic_far * sol[0]

    cross_section = np.sum(np.abs(far_field)**2, axis=0)

    np.save("output/ellipsoid_r3_r2_w0.5_TM_sub4_3600_T"+str(mode)+".npy", cross_section)

    end0 = time.time()
    print("time", end0 - start0)
