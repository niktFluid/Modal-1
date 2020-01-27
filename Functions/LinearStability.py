import numpy as np
# import scipy as sp
from scipy.sparse import linalg

from Functions.FieldData import FieldData


class LSMode(FieldData):
    def __init__(self, mesh, operator, n_val=5, k=10, sigma=None, which='LM'):
        self.operator = operator
        self._eig_options = {'k': k, 'sigma':sigma, 'which': which}
        self._n_values = n_val

        self.eigs = np.empty(k)

        data_list_base = ['rho', 'u', 'v', 'w', 'p']
        data_list = []
        for i_mode in range(k):
            data_list += ['mode{:0=4}_'.format(i_mode) + x for x in data_list_base]
        super(LSMode, self).__init__(mesh, n_val=n_val * k, data_list=data_list)

    def _init_field(self, *args, **kwargs):
        k = self._eig_options['k']
        self.data = np.empty((self.n_cell, self._n_values * k), dtype=np.float64)

    def solve(self):
        eigs, vecs = linalg.eigs(self.operator, **self._eig_options)
        # eigs, vecs = linalg.eigs(self.operator, k=self.k)
        self.eigs = eigs
        self._set_vec_data(vecs)

        w_eigs = np.empty((len(eigs), 2), dtype=np.float64)
        w_eigs[:, 0] = np.real(eigs * 1j)
        w_eigs[:, 1] = np.imag(eigs * 1j)

        # noinspection PyTypeChecker
        np.savetxt('eigs.txt', w_eigs)

    def _set_vec_data(self, vecs):
        for i_mode, vec in enumerate(vecs.T):
            i_start = self._n_values * i_mode
            i_end = self._n_values * (i_mode + 1)

            w_vec = vec.reshape((self.n_cell, self._n_values), order='F')
            self.data[:, i_start:i_end] = np.real(w_vec)
