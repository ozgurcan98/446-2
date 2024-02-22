
import numpy as np
import spectral
from scipy import sparse
import pytest
from equations import SoundWaves

waves_const_errors = {32: 0.2, 64: 5e-3, 128: 1e-8}
@pytest.mark.parametrize('N', [32, 64, 128])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_SoundWaves_const(N, dtype):
    x_basis = spectral.Chebyshev(N, interval=(0, 3))
    x = x_basis.grid()
    u = spectral.Field([x_basis], dtype=dtype)
    p = spectral.Field([x_basis], dtype=dtype)
    p0 = spectral.Field([x_basis], dtype=dtype)

    u.require_grid_space()
    u.data = np.exp(-(x-0.5)**2/0.01)

    p0.require_grid_space()
    p0.data = 1 + 0*x

    waves = SoundWaves(u, p, p0)

    # check sparsity of M and L matrices
    assert len(waves.problem.subproblems[0].M.data) < 5*N
    assert len(waves.problem.subproblems[0].L.data) < 5*N

    waves.evolve(spectral.SBDF2, 2e-3, 5000)

    p.require_coeff_space()
    p.require_grid_space(scales=256//N)

    sol = np.loadtxt('waves_const.dat')

    error = np.max(np.abs(sol - p.data))

    assert error < waves_const_errors[N]

waves_variable_errors = {32: 4e-2, 64: 3e-4, 128: 1e-12}
@pytest.mark.parametrize('N', [32, 64, 128])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_SoundWaves_variable(N, dtype):
    x_basis = spectral.Chebyshev(N, interval=(0, 3))
    x = x_basis.grid()
    u = spectral.Field([x_basis], dtype=dtype)
    p = spectral.Field([x_basis], dtype=dtype)
    p0 = spectral.Field([x_basis], dtype=dtype)
    
    u.require_grid_space()
    u.data = np.exp(-(x-0.5)**2/0.01)

    p0.require_grid_space()
    p0.data = 0.1 + x**2/9

    waves = SoundWaves(u, p, p0)

    # check sparsity of M and L matrices
    assert len(waves.problem.subproblems[0].M.data) < 5*N
    assert len(waves.problem.subproblems[0].L.data) < 5*N

    waves.evolve(spectral.SBDF2, 2e-3, 5000)

    p.require_coeff_space()
    p.require_grid_space(scales=256//N)

    sol = np.loadtxt('waves_variable.dat')

    error = np.max(np.abs(sol - p.data))

    assert error < waves_variable_errors[N]

