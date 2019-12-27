import itertools
import numpy as np
from scipy.sparse import csr_matrix


class Variables:
    def __init__(self, placeholder, mesh, flow_data, sub_list=None):
        self.ph = placeholder
        self.mesh = mesh
        self.data = flow_data
        self.n_val = flow_data.n_val

        self._sub_list = sub_list
        self._leave_list = []

        self._check_placeholder()
        if self._sub_list is not None:
            self._check_sub_list()

    def _check_placeholder(self):
        n_cell = self.mesh.n_cell
        n_val = self.n_val

        if self.ph.shape != (n_cell, n_val):
            raise Exception

    def _check_sub_list(self):
        for sub_variable in self._sub_list:
            if not issubclass(sub_variable, Variables):
                raise TypeError

    def return_ref_cells(self, id_cell):
        raise NotImplementedError

    def _set_leaves(self, id_cell):
        self._leave_list = []
        ref_cell_list = []

        my_ref_cells = self.return_ref_cells(id_cell)

        if self._sub_list is not None:
            for sub_variable, ref_cell in itertools.product(self._sub_list, my_ref_cells):
                ref_cell_list += sub_variable.return_ref_cells(ref_cell)

        self._leave_list = list(set(my_ref_cells + ref_cell_list))

    def calc(self, id_cell):
        self._set_leaves(id_cell)

        def iterator(id_val, ref_cell, ref_val):
            self._set_place_holder(ref_cell, ref_val)
            x = self.equation(id_cell, id_val)
            self._set_place_holder(ref_cell, ref_val, init=True)

            return x

        return [(id_val, ref_cell, ref_val, iterator(id_val, ref_cell, ref_val))
                for id_val, ref_cell, ref_val
                in itertools.product(range(self.n_val), self._leave_list, range(self.n_val))]

    def _set_place_holder(self, id_cell, i_val, init=False):
        if init:
            x = 0.0
        else:
            x = 1.0

        self.ph[id_cell, i_val] = x

    def equation(self, id_cell, i_val):
        raise NotImplementedError

    def _boundary_basic(self):
        pass


class Identity(Variables):
    def __init__(self, placeholder, mesh, flow_data):
        super(Identity, self).__init__(placeholder, mesh, flow_data, sub_list=None)

    def return_ref_cells(self, id_cell):
        return [id_cell]

    def equation(self, id_cell, i_val):
        return self.ph(id_cell, i_val)


class Gradient(Variables):
    def __init__(self, placeholder, mesh, flow_data):
        super(Gradient, self).__init__(placeholder, mesh, flow_data, sub_list=None)

        self._nb_cells = None
        self._faces = None

    def return_ref_cells(self, id_cell):
        self._nb_cells = [id_cell] + self.mesh.cell_neighbours(id_cell)
        ref_cells = [i_cell for i_cell in self._nb_cells if i_cell >= 0]

        return list(set(ref_cells))

    def equation(self, id_cell, i_val):
        self._faces = self.mesh.cell_faces[id_cell]

        grad = 0.0

        return grad

    def _set_mat(self, id_cell):
        pass

    def _set_rhs(self):
        pass

    def _get_val_diff(self):
        pass

    def _get_pos_diff(self):
        pass

    def _bd_val_diff(self):
        pass

    def _bd_pos_diff(self):
        pass
