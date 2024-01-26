
import numpy as np
import scipy.fft

class Basis:

    def __init__(self, N, interval):
        self.N = N
        self.interval = interval


class Fourier(Basis):

    def __init__(self, N, interval=(0, 2*np.pi)):
        super().__init__(N, interval)

    def grid(self, scale=1):
        N_grid = int(np.ceil(self.N*scale))
        return np.linspace(self.interval[0], self.interval[1], num=N_grid, endpoint=False)

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

    def _transform_to_grid_complex(self, data, axis, scale): 
        C_scaled = np.zeros([int(scale * self.N)]) + np.zeros([int(scale * self.N)]) * 1j
        C_scaled[0:int(self.N/2)] = data[0:int(self.N/2)]
        C_scaled[-int(self.N/2)+1:] = data[-int(self.N/2)+1:]
        return scipy.fft.ifft(C_scaled * self.N * scale)

    def _transform_to_coeff_complex(self, data, axis):
        C_scaled = scipy.fft.fft(data) / len(data)
        C_orig = np.zeros([int(self.N)]) + np.zeros([int(self.N)]) * 1j
        C_orig[0:int(self.N/2)] = C_scaled[0:int(self.N/2)]
        C_orig[-int(self.N/2)+1:] = C_scaled[-int(self.N/2)+1:]
        return C_orig

    def _transform_to_grid_real(self, data, axis, scale):
         # Fourier coefficients of the scaled version:
        Coeff_rfft_scaled = np.zeros([int(scale * self.N)])
        Coeff_rfft_scaled[:self.N] = data
        ## Get grid points by using irfft:
        A0_rfft = Coeff_rfft_scaled[0]
        An_rfft = Coeff_rfft_scaled[2:int(self.N*scale-1):2] 
        Bn_rfft = Coeff_rfft_scaled[3:int(self.N*scale):2]

        C0_real = A0_rfft
        Cn_real = An_rfft / 2
        Cn_imag = Bn_rfft / 2

        Cn_real = np.hstack((C0_real,Cn_real))
        Cn_real = np.hstack((Cn_real,0))

        Cn_imag = np.hstack((0,Cn_imag))
        Cn_imag = np.hstack((Cn_imag,0))

        C_grid_scaled = ( Cn_real + Cn_imag * 1j ) * self.N * scale

        Grid_irfft_scaled = scipy.fft.irfft(C_grid_scaled)
        
        return Grid_irfft_scaled
        
    def _transform_to_coeff_real(self, data, axis):
        C = scipy.fft.rfft(data) / len(data)
        
        A0_rfft = C[0].real
        An_rfft = 2 * C[1:-1].real
        Bn_rfft = 2 * C[1:-1].imag

        Coeff_rfft = np.zeros([self.N])
        # Insert A0 and B0
        Coeff_rfft[0] = A0_rfft
        Coeff_rfft[1] = 0
        # Insert An values
        Coeff_rfft[2:self.N-1:2] = An_rfft[:int(self.N/2-1)]
        # Insert Bn  values
        Coeff_rfft[3:self.N:2] = Bn_rfft[:int(self.N/2-1)]
        return Coeff_rfft


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
