
import numpy as np
import spectral
from scipy import sparse
import pytest
from equations import CGLEquation

CGL_errors = {256: 0.01, 384: 1e-5, 512: 4e-7}
CGL_long_errors = {256: 1, 384: 1e-2, 512: 1e-4}
@pytest.mark.parametrize('N', [256, 384, 512])
@pytest.mark.parametrize('num_steps', [2000, 3000])
def test_CGL(N, num_steps):
    x_basis = spectral.Chebyshev(N, interval=(0, 100))
    x = x_basis.grid()
    u = spectral.Field([x_basis], dtype=np.complex128)

    u.require_grid_space()
    u.data = (1e-3 * np.sin(5 * np.pi * x / 100)
            + 1e-3 * np.sin(2 * np.pi * x / 100))

    CGL = CGLEquation(u)

    # check sparsity of M and L matrices
    assert len(CGL.problem.subproblems[0].M.data) < 3*N
    assert len(CGL.problem.subproblems[0].L.data) < 9*N

    CGL.evolve(spectral.SBDF2, 0.05, num_steps)

    u.require_coeff_space()
    u.require_grid_space(scales=1024/N)

    if num_steps == 2000:
        sol = np.loadtxt('CGL.dat')
    elif num_steps == 3000:
        sol = np.loadtxt('CGL_long.dat')

    error = np.max(np.abs(sol - np.abs(u.data)))

    if num_steps == 2000:
        assert error < CGL_errors[N]
    elif num_steps == 3000:
        assert error < CGL_long_errors[N]

