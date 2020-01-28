import numpy as np
import pickle
# import scipy as sp
from scipy.sparse import linalg
from scipy import sparse

from Functions.FieldData import FieldData


class ModalData(FieldData):
    def __init__(self, mesh, operator_name='matL.npz', n_val=5, k=10, **kwargs):  # kwargs for set up the operator.
        self._k = k
        self._n_q = n_val

        super(ModalData, self).__init__(mesh, n_val=self._data_num(), data_list=self._data_name_list())

        operator = sparse.load_npz(operator_name)
        self.operator = self._set_operator(operator, **kwargs)
        self._vec_data = None

    def _init_field(self, *args, **kwargs):
        self.data = np.empty((self.n_cell, self._data_num()), dtype=np.float64)

    def _data_num(self):
        raise NotImplementedError

    def _data_name_list(self):
        raise NotImplementedError

    def _set_operator(self, operator, **kwargs):
        raise NotImplementedError

    def solve(self, **kwargs):  # kwargs for ARPACK.
        result = self._calculate(**kwargs)
        self._set_data(result)  # Set self.data and self._vec_data for the visualization.

    def save_data(self, filename='modalData.pickle'):
        with open(filename, 'wb') as file_obj:
            pickle.dump(self._vec_data, file_obj)

    def load_data(self, filename='modalData.pickle'):
        self._vec_data = pickle.load(filename)

    def _calculate(self, **kwargs):
        raise NotImplementedError

    def _set_data(self, data):
        raise NotImplementedError


class LinearStabilityMode(ModalData):
    def __init__(self, mesh, operator, n_val=5, k=10, **kwargs):
        super(LinearStabilityMode, self).__init__(mesh, operator, n_val, k, **kwargs)

    def _data_num(self):
        return self._n_q * self._k

    def _data_name_list(self):
        data_list_base = ['rho', 'u', 'v', 'w', 'p']
        data_list = []
        for i_mode in range(self._k):
            data_list.append(['mode{:0=4}_'.format(i_mode) + x for x in data_list_base])
        return data_list

    def _set_operator(self, operator, **kwargs):
        return operator

    def _calculate(self, **kwargs):
        return linalg.eigs(self.operator, k=self._k, **kwargs)

    def _set_data(self, data):
        eigs, vecs = data
        self._vec_data = [eigs, vecs]

        for i_mode, vec in enumerate(vecs.T):
            i_start = self._n_q * i_mode
            i_end = self._n_q * (i_mode + 1)

            w_vec = vec.reshape((self.n_cell, self._n_q), order='F')
            self.data[:, i_start:i_end] = np.real(w_vec)


class ResolventMode(ModalData):
    def __init__(self, mesh, operator, omega, n_val=5, k=6, **kwargs):
        self.omega = omega

        super(ResolventMode, self).__init__(mesh, operator, n_val, k, **kwargs)

    def _data_num(self):
        return self._n_q * self._k * 2

    def _data_name_list(self):
        data_list_base = ['rho', 'u', 'v', 'w', 'p']
        data_list = []
        for i_mode in range(self._k):
            data_list.append(['forcing{:0=4}_'.format(i_mode) + x for x in data_list_base])
            data_list.append(['response{:0=4}_'.format(i_mode) + x for x in data_list_base])
        return data_list

    def _set_operator(self, operator, **kwargs):
        n_mat = operator.shape[0]

        qi, qo = self._get_norm_quadrature()
        eye = sparse.eye(n_mat, dtype=np.float64, format='csr')

        return qo @ (-1j * self.omega * eye - operator) @ qi

    def _calculate(self, **kwargs):
        return linalg.svds(self.operator, k=self._k, which='SM', **kwargs)

    def _set_data(self, data):
        r_vecs, svs, f_vecs = data
        self._vec_data = [self.omega, 1.0 / svs, r_vecs, f_vecs]  # Freq, gain, response, forcing

        for i_mode, (f_vec, r_vec) in enumerate(zip(f_vecs.T, r_vecs.T)):
            fw_vec = f_vec.reshape((self.n_cell, self._n_q), order='F')
            rw_vec = r_vec.reshape((self.n_cell, self._n_q), order='F')

            i_start = 2 * self._n_q * i_mode
            i_end = 2 * self._n_q * (i_mode + 1)
            self.data[:, i_start:i_end] = np.real(fw_vec)

            i_start = 2 * self._n_q * i_mode + self._n_q
            i_end = 2 * self._n_q * (i_mode + 1) + self._n_q
            self.data[:, i_start:i_end] = np.real(rw_vec)

    def _get_norm_quadrature(self):
        diags = np.tile(self.mesh.volumes, self._n_q)

        qi = sparse.diags(1.0 / np.square(diags), format='csr')
        qo = sparse.diags(np.square(diags), format='csr')

        return qi, qo
