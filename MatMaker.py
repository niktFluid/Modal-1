import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from Variables import Variables
from Variables import Identity


class MatMaker:
    def __init__(self, mesh, flow_data, target):
        self.n_cell = self.mesh.n_cell
        self.n_val = self.flow_data.n_val
        self.n_size = self.n_cell * self.n_val

        self.mesh = mesh
        self.flow_data = flow_data

        self.target = target
        self._check_target()

        self.operator = self._set_matrix()

    def _check_target(self):
        if not issubclass(self.target, Variables):
            raise TypeError

    def _set_matrix(self):
        return lil_matrix((self.n_size, self.n_size), dtype=np.float64)

    def get_mat(self):
        for id_cell in range(self.n_cell):
            self._set_mat_for_cell(id_cell)

        return csr_matrix(self.operator)

    def _set_mat_for_cell(self, id_cell):
        target = self.target

        val_list = target.calc(id_cell)

        for id_val, ref_cell, ref_val, val in val_list:
            i_row = self._serializer(id_cell, id_val)
            i_col = self._serializer(ref_cell, ref_val)

            self.operator[i_row, i_col] = val

    def _serializer(self, id_cell, id_val):
        return id_cell * self.n_val + id_val


class TargetEq(Variables):
    def __init__(self, mesh, flow_data):
        sub_list = [Identity(mesh, flow_data)]
        super(TargetEq, self).__init__(mesh, flow_data, sub_list=sub_list)

    def return_ref_cells(self, id_cell):
        idx = self._sub_list[0]
        return idx.return_ref_cells(id_cell)

    def equation(self, id_cell, i_val):
        idx = self._sub_list[0]
        return idx.equation(id_cell, i_val)
