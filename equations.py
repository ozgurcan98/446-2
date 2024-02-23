
import spectral
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

class SoundWaves:

    def __init__(self, u, p, p0):
        self.u = u
        self.p = p
        self.x_basis = u.bases[0]
        self.dtype = dtype = u.dtype
        self.u_RHS = spectral.Field([self.x_basis], dtype = dtype)
        self.p_RHS = spectral.Field([self.x_basis], dtype = dtype)
        self.p0 = p0
        # Length of the physical domain:
        self.L = self.x_basis.interval
        self.L = self.L[1] - self.L[0]
        
        self.problem = spectral.InitialValueProblem([u, p], [self.u_RHS, self.p_RHS], num_BCs=2)

        sp = self.problem.subproblems[0]

        N = self.x_basis.N
        self.N = N
        
        # Derivative matrix: Note that we need to include Jacobian of the transformation
        # which is 2/L for mapping from physical to parameter space. Linear mapping is:
        # xi = -1 + 2/L * x where x is in the physical space and xi is in the parameter
        # space.
        
        diag = ( np.arange(N-1)+1 ) * (2/self.L)
        D = sparse.diags(diag, offsets=1) 
        self.D = D
        
        diag0 = np.ones(N)/2
        diag0[0] = 1
        diag2 = -np.ones(N-2)/2
        self.C = C = sparse.diags((diag0, diag2), offsets=(0,2))

        # M matrix = [ C 0 0 ; 0 C 0 ; 0 0 0 ] 

        M = sparse.csr_matrix((2*N+2,2*N+2))
        M[0:N,  0:N] = C
        M[N:2*N, N:2*N] = C
        sp.M = M

        # L matrix
        BC_rows = np.zeros((2, 2*N))
        i = np.arange(N)
        BC_rows[0, :N] = (-1)**i
        BC_rows[1, :N] = (+1)**i

        cols = np.zeros((2*N,2))
        cols[  N-1, 0] = 1
        cols[  2*N-1, 1] = 1
        corner = np.zeros((2,2))

        Z = np.zeros((N, N))
        L = sparse.bmat([[Z, D],
                         [D, Z]])
        L = sparse.bmat([[      L,   cols],
                         [BC_rows, corner]])

        sp.L = L
        self.t = 0

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        p = self.p
        p0 = self.p0
        p0.require_coeff_space()
        p0.require_grid_space( scales = 2 )
        p_RHS = self.p_RHS

        for i in range(num_steps):
            # take a timestep
            u.require_coeff_space()
            p.require_coeff_space()
            p_RHS.require_coeff_space()
            
            p_RHS.data = self.D @ u.data # U basis
            p_RHS.data = spla.spsolve(self.C, p_RHS.data) # T basis
            p_RHS.require_grid_space( scales = 2 ) # Grid space
            p_RHS.data = ( 1 - p0.data ) * p_RHS.data
            p_RHS.require_coeff_space() # T basis
            p_RHS.data = self.C @ p_RHS.data # U basis

            ts.step(dt, [0,0])
            self.t += dt


class CGLEquation:

    def __init__(self, u):
        self.u = u
        self.x_basis = u.bases[0]
        self.dtype = dtype = u.dtype
        self.u_RHS = spectral.Field([self.x_basis], dtype=dtype)
        self.ux = spectral.Field([self.x_basis], dtype=dtype)
        self.ux_RHS = spectral.Field([self.x_basis], dtype=dtype)
        self.L = self.x_basis.interval
        self.L = self.L[1] - self.L[0]
        
        self.problem = spectral.InitialValueProblem([self.u, self.ux], [self.u_RHS, self.ux_RHS], num_BCs=2)

        sp = self.problem.subproblems[0]

        self.N = N = N = self.x_basis.N

        # D matrix:
        diag = ( np.arange(N-1)+1 ) * (2/self.L)
        D = sparse.diags(diag, offsets=1) 
        self.D = D
        
        # C matrix:
        diag0 = np.ones(N)/2
        diag0[0] = 1
        diag2 = -np.ones(N-2)/2
        self.C = C = sparse.diags((diag0, diag2), offsets=(0,2))

        # M matrix:
        M = sparse.csr_matrix((2*N+2,2*N+2))
        M[N:2*N, :N] = C
        M.eliminate_zeros()
        sp.M = M

        # L matrix
        BC_rows = np.zeros((2, 2*N))
        i = np.arange(N)
        BC_rows[0, :N] = (-1)**i
        BC_rows[1, :N] = (+1)**i

        cols = np.zeros((2*N,2))
        cols[  N-1, 0] = 1
        cols[2*N-1, 1] = 1
        corner = np.zeros((2,2))

        Z = np.zeros((N, N))
        L = sparse.bmat([[D    , -C],
                         [-C   , -(1+0.5*1j)*D]])
        L = sparse.bmat([[      L,   cols],
                         [BC_rows, corner]])
        
        L.eliminate_zeros()
        sp.L = L
        self.t = 0

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        ux = self.ux
        ux_RHS = self.ux_RHS
        u_RHS = self.u_RHS

        for i in range(num_steps):  
            # take a timestep
            u.require_coeff_space()
            ux.require_coeff_space()
            ux_RHS.require_coeff_space()
            
            u.require_grid_space(scales=2)
            ux_RHS.require_grid_space(scales=2)
            
            #ux_RHS.data = -(1 - 1j * 1.76) * abs(u.data)**2 * u.data
            ux_RHS.data = -(1-1.76*1j) * np.conj(u.data) * u.data * u.data
            ux_RHS.require_coeff_space()
            
            ux_RHS.data = self.C @ ux_RHS.data
            
            ts.step(dt, [0,0])
            self.t += dt


class KdVEquation:
    
    def __init__(self, domain, u):
        dtype = u.dtype
        self.dealias = 3/2
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u], [self.u_RHS], dtype=dtype)
        
        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        D = x_basis.derivative_matrix(dtype)
        p.L = D@D@D
        
    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        dudx = self.dudx
        u_RHS = self.u_RHS
        for i in range(num_steps):
            dudx.require_coeff_space()
            u.require_coeff_space()
            dudx.data = u.differentiate(0)
            u.require_grid_space(scales=self.dealias)
            dudx.require_grid_space(scales=self.dealias)
            u_RHS.require_grid_space(scales=self.dealias)
            u_RHS.data = 6*u.data*dudx.data
            ts.step(dt)


class SHEquation:

    def __init__(self, domain, u):
        dtype = u.dtype
        self.dealias = 2
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u], [self.u_RHS], dtype=dtype)

        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        D = x_basis.derivative_matrix(dtype)
        op = I + D@D
        p.L = op @ op + 0.3*I

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        u_RHS = self.u_RHS
        for i in range(num_steps):
            u.require_coeff_space()
            u.require_grid_space(scales=self.dealias)
            u_RHS.require_grid_space(scales=self.dealias)
            u_RHS.data = 1.8*u.data**2 - u.data**3
            ts.step(dt)



