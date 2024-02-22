import numpy as np
import scipy.fft
import scipy.sparse.linalg as spla
from scipy import sparse
from collections import deque

# These functions are by Keaton Burns
def axindex(axis, index):
    """Index array along specified axis."""
    if axis < 0:
        raise ValueError("`axis` must be positive")
    # Add empty slices for leading axes
    return (slice(None),)*axis + (index,)

def axslice(axis, start, stop, step=None):
    """Slice array along a specified axis."""
    return axindex(axis, slice(start, stop, step))

def reshape_vector(data, dim=2, axis=-1):
    """Reshape 1-dim array as a multidimensional vector."""
    # Build multidimensional shape
    shape = [1] * dim
    shape[axis] = data.size
    return data.reshape(shape)


class Basis:

    def __init__(self, N, interval):
        self.N = N
        self.interval = interval


class Chebyshev(Basis):

    def __init__(self, N, interval=(0, 2*np.pi)):
        super().__init__(N, interval)

    def grid(self, scale=1):
        N_grid = int(np.ceil(self.N*scale))
        i = np.arange(N_grid)
        interval = self.interval
        return (interval[1]+interval[0])/2 + ((interval[1]-interval[0])/2) * np.cos((2*i+1)/(2*N_grid)*np.pi)

    def transform_to_grid(self, data, axis, dtype, scale=1):
        N_grid = int(np.ceil(self.N*scale))
        shape = list(data.shape)
        shape[axis] = N_grid
        coeff_data = np.zeros(shape, dtype=dtype)
        zero = axslice(axis, 0, 1)
        coeff_data[zero] = data[zero]
        nonzero = axslice(axis, 1, self.N)
        coeff_data[nonzero] = data[nonzero]/2
        return scipy.fft.dct(coeff_data, type=3, axis=axis)

    def transform_to_coeff(self, data, axis, dtype):
        coeff_data = scipy.fft.dct(data, type=2, axis=axis)
        coeff_data = coeff_data[axslice(axis, 0, self.N)]
        N_grid = data.shape[axis]
        coeff_data[axslice(axis, 0, 1)] /= 2*N_grid
        coeff_data[axslice(axis, 1, self.N)] /= N_grid
        return coeff_data


class Fourier(Basis):

    def __init__(self, N, interval=(0, 2*np.pi)):
        super().__init__(N, interval)

    def grid(self, scale=1):
        N_grid = int(np.ceil(self.N*scale))
        return np.linspace(self.interval[0], self.interval[1], num=N_grid, endpoint=False)

    def wavenumbers(self, dtype=np.float64):
        if dtype == np.float64:
            k_half = np.arange(self.N//2, dtype=np.float64)
            k = np.zeros(self.N, dtype=np.float64)
            k[::2] = k_half
            k[1::2] = k_half
        elif dtype == np.complex128:
            k = np.arange(self.N, dtype=np.float64)
            k[-self.N//2:] -= self.N
        k *= 2*np.pi/(self.interval[1] - self.interval[0])
        return k

    def unique_wavenumbers(self, dtype=np.float64):
        if dtype == np.float64:
            return self.wavenumbers(dtype=dtype)[::2]
        else:
            return self.wavenumbers(dtype=dtype)

    def slice(self, wavenumber, dtype=np.float64):
        i = np.argwhere(self.unique_wavenumbers(dtype) == wavenumber)[0,0]
        if dtype == np.float64:
            index = slice(2*i, 2*i+2, None)
        elif dtype == np.complex128:
            index = slice(i, i+1, None)
        return index

    def transform_to_grid(self, data, axis, dtype, scale=1):
        if dtype == np.complex128:
            return self._transform_to_grid_complex(data, axis, scale)
        elif dtype == np.float64:
            return self._transform_to_grid_real(data, axis, scale)
        else:
            raise NotImplementedError("Can only perform transforms for float64 or complex128")

    def transform_to_coeff(self, data, axis, dtype):
        if dtype == np.complex128:
            return self._transform_to_coeff_complex(data, axis)
        elif dtype == np.float64:
            return self._transform_to_coeff_real(data, axis)
        else:
            raise NotImplementedError("Can only perform transforms for float64 or complex128")

    def _resize_rescale_complex(self, data_in, data_out, axis, Kmax, rescale):
        # array indices for padding
        posfreq = axslice(axis, 0, Kmax+1)
        badfreq = axslice(axis, Kmax+1, -Kmax)
        negfreq = axslice(axis, -Kmax, None)
        # rescale
        np.multiply(data_in[posfreq], rescale, data_out[posfreq])
        data_out[badfreq] = 0
        np.multiply(data_in[negfreq], rescale, data_out[negfreq])

    def _transform_to_grid_complex(self, data, axis, scale):
        N_grid = int(np.ceil(self.N*scale))
        shape = list(data.shape)
        shape[axis] = N_grid
        grid_data = np.zeros(shape, dtype=np.complex128)
        Kmax = (self.N - 1) // 2
        self._resize_rescale_complex(data, grid_data, axis, Kmax, N_grid)
        grid_data = scipy.fft.ifft(grid_data, axis=axis)
        return grid_data

    def _transform_to_coeff_complex(self, data, axis):
        shape = list(data.shape)
        N_grid = shape[axis]
        shape[axis] = self.N
        coeff_data = np.zeros(shape, dtype=np.complex128)
        data = scipy.fft.fft(data, axis=axis)
        Kmax = (self.N - 1) // 2
        self._resize_rescale_complex(data, coeff_data, axis, Kmax, 1/N_grid)
        return coeff_data

    def _pack_rescale_real(self, data_in, data_out, axis, Kmax, rescale):
        # pack real data_in into complex data_out for irfft
        meancos = axslice(axis, 0, 1)
        data_out[meancos] = data_in[meancos] * rescale
        posfreq = axslice(axis, 1, Kmax+1)
        badfreq = axslice(axis, Kmax+1, None)
        posfreq_cos = axslice(axis, 2, (Kmax+1)*2, 2)
        posfreq_msin = axslice(axis, 3, (Kmax+1)*2, 2)
        np.multiply(data_in[posfreq_cos], rescale/2, data_out[posfreq].real)
        np.multiply(data_in[posfreq_msin], rescale/2, data_out[posfreq].imag)
        data_out[badfreq] = 0.

    def _transform_to_grid_real(self, data, axis, scale):
        N_grid = int(np.ceil(self.N*scale))
        shape = list(data.shape)
        shape[axis] = (N_grid + 1)//2 # divide by 2 for complex
        grid_data = np.zeros(shape, dtype=np.complex128)
        Kmax = (self.N - 1) // 2
        self._pack_rescale_real(data, grid_data, axis, Kmax, N_grid)
        grid_data = scipy.fft.irfft(grid_data, axis=axis, n=N_grid) # note we need the n=N_grid!!
        return grid_data

    def _unpack_scale_real(self, data_in, data_out, axis, Kmax, rescale):
        # unpack complex data_in from rfft into real data_out
        meancos = axslice(axis, 0, 1)
        meansin = axslice(axis, 1, 2)
        data_out[meancos] = data_in[meancos].real * rescale
        data_out[meansin] = 0
        posfreq = axslice(axis, 1, Kmax+1)
        badfreq = axslice(axis, 2*(Kmax+1), None)
        posfreq_cos = axslice(axis, 2, (Kmax+1)*2, 2)
        posfreq_msin = axslice(axis, 3, (Kmax+1)*2, 2)
        np.multiply(data_in[posfreq].real, 2*rescale, data_out[posfreq_cos])
        np.multiply(data_in[posfreq].imag, 2*rescale, data_out[posfreq_msin])
        data_out[badfreq] = 0.

    def _transform_to_coeff_real(self, data, axis):
        shape = list(data.shape)
        N_grid = shape[axis]
        shape[axis] = self.N
        coeff_data = np.zeros(shape, dtype=np.float64)
        data = scipy.fft.rfft(data, axis=axis)
        Kmax = (self.N - 1) // 2
        self._unpack_scale_real(data, coeff_data, axis, Kmax, 1/N_grid)
        return coeff_data


class Field:

    def __init__(self, bases, dtype=np.float64):
        self.bases = bases
        self.dim = len(bases)
        self.dtype = dtype
        self.data = np.zeros([basis.N for basis in self.bases], dtype=dtype)
        self.coeff = np.array([True]*self.dim)

    def _remedy_scales(self, scales):
        if scales is None:
            scales = 1
        if not hasattr(scales, "__len__"):
            scales = [scales] * self.dim
        return scales

    def subproblem_length(self):
        N = self.bases[-1].N
        if self.dtype == np.float64:
            N *= 2**(len(self.bases)-1)
        return N

    def towards_coeff_space(self):
        if self.coeff.all():
            # already in full coeff space
            return
        axis = np.where(self.coeff == False)[0][0]
        self.data = self.bases[axis].transform_to_coeff(self.data, axis, self.dtype)
        self.coeff[axis] = True

    def require_coeff_space(self):
        if self.coeff.all():
            # already in full coeff space
            return
        else:
            self.towards_coeff_space()
            self.require_coeff_space()

    def towards_grid_space(self, scales=None):
        if not self.coeff.any():
            # already in full grid space
            return
        axis = np.where(self.coeff == True)[0][-1]
        scales = self._remedy_scales(scales)
        self.data = self.bases[axis].transform_to_grid(self.data, axis, self.dtype, scale=scales[axis])
        self.coeff[axis] = False

    def require_grid_space(self, scales=None):
        if not self.coeff.any(): 
            # already in full grid space
            return
        else:
            self.towards_grid_space(scales)
            self.require_grid_space(scales)


class Problem:

    def __init__(self, variables, num_BCs=0):
        self.variables = variables
        self.num_BCs = num_BCs
        # assume all variables have the same dtype and bases
        self.dtype = variables[0].dtype
        bases = variables[0].bases
        self.subproblems = []
        if len(bases) > 1:
            shape = [len(basis.unique_wavenumbers(dtype)) for basis in domain.bases[:-1]]
            for i in range(np.prod(shape)):
                multiindex = np.unravel_index(i, shape)
                wavenumbers = [basis.unique_wavenumbers(dtype)[j] for (basis, j) in zip(domain.bases[:-1], multiindex)]
                slices = [basis.slice(wavenumber, dtype) for (basis, wavenumber) in zip(domain.bases[:-1], wavenumbers)]
                self.subproblems.append(Subproblem(slices, wavenumbers))
        else:
            slices = [slice(None, None, None)]
            self.subproblems.append(Subproblem(slices, 1))
        self.X = StateVector(variables, self, num_BCs)


class InitialValueProblem(Problem):

    def __init__(self, variables, RHS_variables, num_BCs=0):
        super().__init__(variables, num_BCs)
        self.F = StateVector(RHS_variables, self, num_BCs)


class Timestepper:

    def __init__(self, problem):
        self.problem = problem
        self.iteration = 0
        self.time = 0
        self.a0_old = None
        self.b0_old = None
        self.dt = deque([0]*self.amax)
        for sp in problem.subproblems:
            shape = problem.X.vector.shape
            sp.LX = deque()
            sp.MX = deque()
            for a in range(self.amax):
                sp.MX.append(np.zeros(shape, problem.dtype))
            sp.LX = deque()
            for b in range(self.bmax):
                sp.LX.append(np.zeros(shape, problem.dtype))
            sp.F = deque()
            for c in range(self.cmax):
                sp.F.append(np.zeros(shape, problem.dtype))
            sp.RHS = np.zeros(shape, problem.dtype)

            if problem.num_BCs > 0:
                sp.taus = np.zeros(problem.num_BCs, dtype=problem.dtype)

    def step(self, dt, BCs=None):
        problem = self.problem
        X = problem.X
        F = problem.F
        self.dt.rotate()
        self.dt[0] = dt
        a, b, c = self.coefficients(self.dt, self.iteration)
        for sp in problem.subproblems:
            F.gather(sp, BCs)
            sp.F.rotate()
            np.copyto(sp.F[0], F.vector)

            X.gather(sp)
            if self.amax > 0:
                sp.MX.rotate()
                sp.MX[0] = sp.M @ X.vector
            if self.bmax > 0:
                sp.LX.rotate()
                sp.LX[0] = sp.L @ X.vector

            sp.RHS = c[0]*sp.F[0]
            for i in range(1, len(c)):
                sp.RHS += c[i]*sp.F[i]
            for i in range(1, len(b)):
                sp.RHS -= b[i]*sp.LX[i-1]
            for i in range(1, len(a)):
                sp.RHS -= a[i]*sp.MX[i-1]

            if self.a0_old != a[0] or self.b0_old != b[0]:
                LHS = a[0]*sp.M + b[0]*sp.L
                LHS = LHS.astype(problem.dtype)
                sp.LU = spla.splu(LHS.T)
            X.vector = sp.LU.solve(sp.RHS, trans='T')
            if problem.num_BCs > 0:
                X.scatter(sp, sp.taus)
            else:
                X.scatter(sp)

        self.a0_old = a[0]
        self.b0_old = b[0]

        self.time += dt
        self.iteration += 1


class SBDF1(Timestepper):

    amax = 1
    bmax = 0
    cmax = 1

    @classmethod
    def coefficients(self, dt, iteration):
        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax)
        dt = dt[0]
        a[0] = 1/dt
        a[1] = -1/dt
        b[0] = 1
        c[0] = 1

        return a, b, c


class SBDF2(Timestepper):

    amax = 2
    bmax = 0
    cmax = 2

    @classmethod
    def coefficients(self, dt, iteration):
        if iteration == 0:
            return SBDF1.coefficients(dt, iteration)

        h, k = dt[0], dt[1]
        w = h/k

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax)
        a[0] = (1 + 2*w) / (1 + w) / h
        a[1] = -(1 + w) / h
        a[2] = w**2 / (1 + w) / h
        b[0] = 1
        c[0] = 1 + w
        c[1] = -w

        return a, b, c


class StateVector:

    def __init__(self, fields, problem, num_BCs):
        self.dtype = problem.dtype
        self.fields = fields
        self.num_BCs = num_BCs
        self.subproblems = problem.subproblems
        data_size = len(fields)*fields[0].subproblem_length() + num_BCs
        self.vector = np.zeros(data_size, dtype=self.dtype)

    def gather(self, sp, BCs=None):
        for field in self.fields:
            field.require_coeff_space()
        sp.gather(self.fields, self.vector)
        if not (BCs is None):
            self.vector[-self.num_BCs:] = BCs

    def scatter(self, sp, taus=None):
        for field in self.fields:
            field.require_coeff_space()
        sp.scatter(self.fields, self.vector)
        if not (taus is None):
            np.copyto(taus, self.vector[-self.num_BCs:])


class Subproblem:

    def __init__(self, slices, wavenumbers):
        self.slices = tuple(slices)
        self.wavenumbers = wavenumbers

    def gather(self, fields, vector):
        for i, field in enumerate(fields):
            N = field.subproblem_length()
            vector[i*N:(i+1)*N] = field.data[self.slices].ravel()

    def scatter(self, fields, vector):
        for i, field in enumerate(fields):
            N = field.subproblem_length()
            shape = field.data[self.slices].shape
            field.data[self.slices] = vector[i*N:(i+1)*N].reshape(shape)

