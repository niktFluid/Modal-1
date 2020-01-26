import numpy as np
# import scipy as sp
from scipy.sparse import linalg

from Functions.FieldData import FieldData


class LSMode(FieldData):
    def __init__(self, mesh, operator, k=10, n_val=5):
        self.operator = operator
        self.k = k

        self.eigs = np.empty(k)

        super(LSMode, self).__init__(mesh, n_val=n_val)

    def _init_field(self, *args, **kwargs):
        self.data = np.empty((self.n_cell, self.n_val * self.k), dtype=np.float64)

    def solve(self):
        eigs, vecs = linalg.eigs(self.operator, self.k)
        self.eigs = eigs

        self._set_vec_data(vecs)

    def _set_vec_data(self, vecs):
        for i_mode, vec in enumerate(vecs):
            i_start = self.n_val * (i_mode - 1)
            i_end = self.n_val * i_mode
            self.data[:, i_start:i_end] = vec.reshape(self.n_cell, self.n_val)
