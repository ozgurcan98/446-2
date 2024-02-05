
import numpy as np
import spectral
from scipy import sparse

class KdVEquation:

    def __init__(self, u):
        dtype = u.dtype;
        self.u = u
        x_basis = u.bases[0]
        
        self.dudx = spectral.Field([x_basis], dtype = dtype)
        self.RHS = spectral.Field([x_basis], dtype = dtype)

        self.problem = spectral.InitialValueProblem([self.u], [self.RHS])

        p = self.problem.subproblems[0]

        N = x_basis.N
        self.kx = x_basis.wavenumbers(dtype)
        I = sparse.eye(N)
        p.M = I
        if dtype == np.complex128:
            diag = -1j*self.kx**3
            p.L = sparse.diags(diag)
        else:
            upper_diagonal = np.zeros(N-1)
            lower_diagonal = np.zeros(N-1)

            upper_diagonal[::2] = self.kx[1::2]**3
            lower_diagonal[::2] = -self.kx[::2]**3
            p.L = sparse.diags([upper_diagonal, lower_diagonal], offsets=[1, -1])         
        

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        dudx = self.dudx
        RHS = self.RHS
        kx = self.kx
        
        for i in range(num_steps):
            u.require_coeff_space()
            dudx.require_coeff_space()

            if self.u.dtype == np.complex128:      
                dudx.data = 1j*kx*u.data
            else:
                # Calculate dudx:
                upper_diagonal = np.zeros(len(u.data)-1)
                lower_diagonal = np.zeros(len(u.data)-1)

                upper_diagonal[::2] = -kx[1::2]
                lower_diagonal[::2] = kx[::2]

                D = sparse.diags([upper_diagonal, lower_diagonal], offsets=[1, -1])
                dudx.require_coeff_space()
                u.require_coeff_space()
                dudx.data = D * u.data
    
            u.require_grid_space(scales=3/2)
            dudx.require_grid_space(scales=3/2)
            RHS.require_grid_space(scales=3/2)
            RHS.data = 6 * u.data * dudx.data
    
            ts.step(dt)

class SHEquation:

    def __init__(self, u ):
        dtype = u.dtype
        self.u = u
        x_basis = u.bases[0]

        self.dudx = spectral.Field([x_basis], dtype = dtype)
        self.RHS = spectral.Field([x_basis], dtype = dtype)
        self.problem = spectral.InitialValueProblem([self.u], [self.RHS])

        p = self.problem.subproblems[0]
        r = -0.3
        
        N = x_basis.N
        self.kx = x_basis.wavenumbers(dtype)
        I = sparse.eye(N)
        p.M = I

        if dtype == np.complex128:
            diag = (1 - self.kx**2)**2 - r
            p.L = sparse.diags(diag)
        else:
            diagonal = np.zeros(len(u.data))

            diagonal = -self.kx**2
            D = sparse.diags(diagonal)
            p.L = (sparse.eye(N)+D) * (sparse.eye(N)+D) - r * sparse.eye(N)

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        dudx = self.dudx
        RHS = self.RHS
        kx = self.kx

        for i in range(num_steps):
            u.require_grid_space(scales=2)
            dudx.require_grid_space(scales=2)
            RHS.require_grid_space(scales=2)
            RHS.data = u.data * u.data * (1.8 - u.data)

            ts.step(dt)
            


