
import numpy as np
import spectral
from scipy import sparse
import pytest
from equations import SHEquation

SHE_IC1_errors = {32: 6e-3, 64: 3e-7, 128: 2e-9}
@pytest.mark.parametrize('N', [32, 64, 128])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_SHE_IC1(N, dtype):
    x_basis = spectral.Fourier(N, interval=(0,12*np.pi))
    scale = 2
    x = x_basis.grid(scale=scale)
    u = spectral.Field([x_basis], dtype=dtype)
    u.require_grid_space(scales=scale)
    u.data = 0.646*np.cos(x/2)**2

    SHE = SHEquation(u)

    SHE.evolve(spectral.SBDF2, 1e-2, 10000)

    sol = np.loadtxt('SHE_IC1.dat')

    u.require_coeff_space()
    u.require_grid_space(scales=256//N)

    error = np.max(np.abs(u.data - sol))

    assert error < SHE_IC1_errors[N]

SHE_IC2_errors = {32: 1e-2, 64: 3e-7, 128: 5e-9}
@pytest.mark.parametrize('N', [32, 64, 128])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_SHE_IC2(N, dtype):
    x_basis = spectral.Fourier(N, interval=(0,12*np.pi))
    scale = 2
    x = x_basis.grid(scale=scale)
    u = spectral.Field([x_basis], dtype=dtype)
    u.require_grid_space(scales=scale)
    u.data = 2.036*np.cos(x/2)**2*np.exp(-0.1*(x-5*np.pi)**2)

    SHE = SHEquation(u)

    SHE.evolve(spectral.SBDF2, 1e-2, 10000)

    sol = np.loadtxt('SHE_IC2.dat')

    u.require_coeff_space()
    u.require_grid_space(scales=256//N)

    error = np.max(np.abs(u.data - sol))

    assert error < SHE_IC2_errors[N]


SHE_IC3_errors = {32: 1e-2, 64: 3e-7, 128: 6e-9}
@pytest.mark.parametrize('N', [32, 64, 128])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_SHE_IC3(N, dtype):
    x_basis = spectral.Fourier(N, interval=(0,12*np.pi))
    scale = 2
    x = x_basis.grid(scale=scale)
    u = spectral.Field([x_basis], dtype=dtype)
    u.require_grid_space(scales=scale)
    u.data = 1.01*np.cos(x/2)**2*np.exp(-0.01*(x-6*np.pi)**2)

    SHE = SHEquation(u)

    SHE.evolve(spectral.SBDF2, 1e-2, 10000)

    sol = np.loadtxt('SHE_IC3.dat')

    u.require_coeff_space()
    u.require_grid_space(scales=256//N)

    error = np.max(np.abs(u.data - sol))

    assert error < SHE_IC3_errors[N]

