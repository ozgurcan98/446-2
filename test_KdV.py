
import numpy as np
import spectral
from scipy import sparse
import pytest
from equations import KdVEquation


KdV_IC1_errors = {16: 0.5, 32: 2e-4, 64: 1e-6}
@pytest.mark.parametrize('N', [16, 32, 64])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_KdV_IC1(N, dtype):
    x_basis = spectral.Fourier(N, interval=(0, 4*np.pi))
    x = x_basis.grid()
    u = spectral.Field([x_basis], dtype=dtype)
    u.require_grid_space()
    u.data = -2*np.cosh((x-2*np.pi))**(-2)

    KdV = KdVEquation(u)

    KdV.evolve(spectral.SBDF2, 1e-3, 10000)

    u.require_coeff_space()
    u.require_grid_space(scales=128//N)

    sol = np.loadtxt('KdV_IC1.dat')

    error = np.max(np.abs(sol - u.data))
    print(error)

    assert error < KdV_IC1_errors[N]


KdV_IC2_errors = {32: 1.5e-3, 64: 3e-8, 128: 4e-10}
@pytest.mark.parametrize('N', [32, 64, 128])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_KdV_IC2(N, dtype):
    x_basis = spectral.Fourier(N, interval=(0,4*np.pi))
    x = x_basis.grid()
    u = spectral.Field([x_basis], dtype=dtype)
    u.require_grid_space()
    u.data = -2*np.exp(-0.5*(x-2*np.pi)**2)

    KdV = KdVEquation(u)

    KdV.evolve(spectral.SBDF2, 1e-3, 10000)

    u.require_coeff_space()
    u.require_grid_space(scales=256//N)

    sol = np.loadtxt('KdV_IC2.dat')

    error = np.max(np.abs(sol - u.data))
    print(error)

    assert error < KdV_IC2_errors[N]


