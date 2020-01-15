import itertools
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from Variables import Variables
from Functions.Identity import Identity


class MatMaker:
    def __init__(self, n_cell, n_val, target):
        self.n_cell = n_cell
        self.n_val = n_val
        self.n_size = self.n_cell * self.n_val

        # self.mesh = mesh
        # self.flow_data = flow_data

        self.variables = target
        # self._check_target()

        self.operator = self._set_matrix()

        # Arrays for compiling the sparse matrix, PlaceHolder, ph
        self._ph_data = np.array([1.0], dtype=np.float64)
        self._ph_indices = np.zeros(1, dtype=np.int32)
        self._ph_indptr = np.ones(self.n_cell+1, dtype=np.int32)

    def _check_target(self):
        if not issubclass(self.variables, Variables):
            raise TypeError

    def _set_matrix(self):
        return lil_matrix((self.n_size, self.n_size), dtype=np.float64)

    def get_mat(self):
        # for id_cell in range(5):
        for id_cell in range(self.n_cell):
            self._set_mat_for_cell(id_cell)

        return csr_matrix(self.operator)

    def _set_mat_for_cell(self, id_cell):
        val_list = self._calc_sub(id_cell)
        # print(val_list)

        for id_val, ref_cell, ref_val, val in val_list:
            i_row = self._serializer(id_cell, id_val)
            i_col = self._serializer(ref_cell, ref_val)
            self.operator[i_row, i_col] = val

    def _serializer(self, id_cell, id_val):
        return id_cell * self.n_val + id_val

    def _calc_sub(self, id_cell):
        variables = self.variables
        ref_cells = variables.get_leaves(id_cell)

        def iterator(id_val, ref_cell, ref_val):
            ph = self._set_place_holder(ref_cell, ref_val)
            val = variables.formula(ph, id_cell, id_val)
            # print(id_cell, id_val, ref_cell, ref_val, self._ph[id_cell, id_val], val)
            return val

        return [(id_val, ref_cell, ref_val, iterator(id_val, ref_cell, ref_val))
                for id_val, ref_cell, ref_val
                in itertools.product(range(self.n_val), ref_cells, range(self.n_val))]

    def _set_place_holder(self, id_cell, i_val):
        n_cell = self.n_cell
        n_val = self.n_val

        data = self._ph_data
        indices = self._ph_indices
        indptr = self._ph_indptr

        indices[0] = i_val
        indptr[0:id_cell+1] = 0
        indptr[id_cell+1:] = 1

        return csr_matrix((data, indices, indptr), shape=(n_cell, n_val))


class TargetEq(Variables):
    def __init__(self, mesh, flow_data):
        sub_list = [Identity(mesh, flow_data)]
        super(TargetEq, self).__init__(mesh, flow_data, sub_list=sub_list)

    def return_ref_cells(self, id_cell):
        idx = self._sub_list[0]
        return idx.return_ref_cells(id_cell)

    def formula(self, ph, id_cell, id_val, **kwargs):
        idx = self._sub_list[0]
        return idx.formula(ph, id_cell, id_val)
