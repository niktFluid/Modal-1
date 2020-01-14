import numpy as np
from Variables import Variables


class Gradient(Variables):
    def __init__(self, mesh, flow_data, sub_list=None):
        super(Gradient, self).__init__(mesh, flow_data, sub_list)

        self._nb_cells = None
        self._faces = None

    def return_ref_cells(self, id_cell):
        self._nb_cells = [id_cell] + self.mesh.cell_neighbours(id_cell)
        ref_cells = [i_cell for i_cell in self._nb_cells if i_cell >= 0]

        return list(set(ref_cells))

    def formula(self, ph, id_cell, id_val, axis=None):
        self._faces = self.mesh.cell_faces[id_cell]

        grad = np.empty(3, dtype=np.float64)

        if axis is not None:
            return grad[axis]
        else:
            return grad

    def _set_left_mat(self):
        n_cell = self.mesh.n_cell
        n_face = self.mesh.n_face

        matA = np.zeros((n_cell, 3, 3), dtype=np.float64)


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
