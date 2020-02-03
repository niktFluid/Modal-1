import os
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

        self.operator = self._set_operator(sparse.load_npz(operator_name), **kwargs)
        self._vec_data = None

        self._arpack_options = {
            'k': self._k,
            'sigma': None,
            'which': 'LM',
            'tol': 1.0e-8
        }

    def _init_field(self, *args, **kwargs):
        self.data = np.empty((self.n_cell, self._data_num()), dtype=np.float64)

    def _data_num(self):
        raise NotImplementedError

    def _data_name_list(self):
        raise NotImplementedError

    def _set_operator(self, operator, **kwargs):
        raise NotImplementedError

    def solve(self):  # kwargs for ARPACK.
        self._vec_data = self._calculate()
        self._set_data(self._vec_data)  # Set self.data and self._vec_data for the visualization.

    def save_data(self, filename='modalData.pickle'):
        with open(filename, 'wb') as file_obj:
            pickle.dump(self._vec_data, file_obj)

    def load_data(self, filename='modalData.pickle'):
        self._vec_data = pickle.load(filename)
        self._set_data(self._vec_data)

    def _calculate(self):
        raise NotImplementedError

    def _set_data(self, data):
        raise NotImplementedError


class LinearStabilityMode(ModalData):
    def __init__(self, mesh, operator, n_val=5, k=10, **kwargs):
        super(LinearStabilityMode, self).__init__(mesh, operator, n_val, k, **kwargs)
        self._arpack_options.update(**kwargs)

    def _data_num(self):
        return self._n_q * self._k

    def _data_name_list(self):
        data_list_base = ['rho', 'u', 'v', 'w', 'T']
        data_list = []
        for i_mode in range(self._k):
            data_list += ['mode{:0=4}_'.format(i_mode) + x for x in data_list_base]
        return data_list

    def _set_operator(self, operator, **kwargs):
        return operator

    def _calculate(self):
        return linalg.eigs(self.operator, **self._arpack_options)

    def _set_data(self, data):
        _, vecs = data

        for i_mode, vec in enumerate(vecs.T):
            i_start = self._n_q * i_mode
            i_end = self._n_q * (i_mode + 1)

            w_vec = vec.reshape((self.n_cell, self._n_q), order='F')
            self.data[:, i_start:i_end] = np.real(w_vec)

    def save_data(self, filename='modalData.pickle'):
        save_name, _ = os.path.splitext(filename)

        eigs, _ = self._vec_data
        # noinspection PyTypeChecker
        np.savetxt(save_name + '_eigs.txt', np.vstack((np.real(eigs), np.imag(eigs))).T)

        with open(filename, 'wb') as file_obj:
            pickle.dump(self._vec_data, file_obj)


class ResolventMode(ModalData):
    def __init__(self, mesh, ave_field, operator, omega, alpha=0.0, n_val=5, k=6, mode=None, **kwargs):
        self._ave_field = ave_field
        self.omega = omega
        self.alpha = alpha

        self._mode = mode  # 'F' for the forcing mode or 'R' for the response mode. 'None' will get both.
        self._mode_f = mode == 'Both' or mode == 'Forcing'
        self._mode_r = mode == 'Both' or mode == 'Response'

        super(ResolventMode, self).__init__(mesh, operator, n_val, k, **kwargs)
        self._arpack_options.update(sigma=0.0, which='LM', **kwargs)

    def _data_num(self):
        if self._mode == 'Both':
            return self._n_q * self._k * 2
        else:
            return self._n_q * self._k

    def _data_name_list(self):
        data_list_base = ['rho', 'u', 'v', 'w', 'T']
        data_list = []
        for i_mode in range(self._k):
            if self._mode_f:
                data_list += ['forcing{:0=4}_'.format(i_mode) + x for x in data_list_base]
            if self._mode_r:
                data_list += ['response{:0=4}_'.format(i_mode) + x for x in data_list_base]
        return data_list

    def _set_operator(self, operator, **kwargs):
        qi, qo = self._get_norm_quadrature()
        eye = sparse.eye(operator.shape[0], dtype=np.complex128, format='csc')
        omegaI = 1.0j * (self.omega + 1.0j * self.alpha) * eye

        return qo * (-omegaI - operator) * qi

    def _calculate(self):
        svs = None
        if self._mode_f:
            matF = self.operator * self.operator.H
            svs, mode_f = linalg.eigsh(matF, **self._arpack_options)
            print('Eigenvalues for forcing: ', svs)
        else:
            mode_f = None

        if self._mode_r:
            matR = self.operator.H * self.operator
            svs, mode_r = linalg.eigsh(matR, **self._arpack_options)
            print('Eigenvalues for response: ', svs)
        else:
            mode_r = None

        print('Singular values: ', np.sqrt(svs))
        print('Gains: ', 1.0 / np.sqrt(svs))
        return self.omega, 1.0 / np.sqrt(svs), mode_r, mode_f

    def _set_data(self, data):
        _, _, r_vecs, f_vecs = data

        coef_ind_1 = 1 + int(self._mode == 'Both')
        coef_ind_2 = self._n_q * int(self._mode == 'Both')
        for i_mode in range(self._k):
            if self._mode_f:
                f_vec = f_vecs[:, i_mode]
                fw_vec = f_vec.reshape((self.n_cell, self._n_q), order='F')
                i_start = coef_ind_1 * self._n_q * i_mode
                i_end = i_start + self._n_q
                self.data[:, i_start:i_end] = np.real(fw_vec)

            if self._mode_r:
                r_vec = r_vecs[:, i_mode]
                rw_vec = r_vec.reshape((self.n_cell, self._n_q), order='F')
                i_start = coef_ind_1 * self._n_q * i_mode + coef_ind_2
                i_end = i_start + self._n_q
                self.data[:, i_start:i_end] = np.real(rw_vec)

    def _get_norm_quadrature(self):
        ave_data = self._ave_field.data
        rho_data = ave_data[:, 0]
        t_data = ave_data[:, 6]

        gamma = 1.4  # heat ratio
        r_gas = 1.0 / 1.4  # Non-dimensionalized gas constant.

        # Chu's energy norm.
        vols = self.mesh.volumes  # / np.linalg.norm(self.mesh.volumes)
        diag_rho = vols * r_gas * t_data / rho_data
        diag_u = vols * rho_data
        diag_t = vols * r_gas * rho_data / ((gamma - 1) * t_data)
        diags = np.hstack((diag_rho, diag_u, diag_u, diag_u, diag_t))

        qi = sparse.diags(1.0 / np.square(diags), format='csc')
        qo = sparse.diags(np.square(diags), format='csc')

        return qi, qo
