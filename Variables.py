import numpy as np
# from scipy.sparse import csr_matrix


class Variables:
    def __init__(self, placeholder, mesh, flow_data, leaves=None):
        self.ph = placeholder
        self.mesh = mesh
        self.data = flow_data
        self.n_val = flow_data.n_val

        self.leave_list = None
        self.leaves = leaves
        # self._check_leaves()

    def calculate(self, id_cell):
        raise NotImplementedError

    def _boundary(self):
        pass

    def _check_leaves(self):
        if self.leave_list is None:
            return True
        else:
            if not isinstance(self.leaves, dict):
                raise TypeError

            if set(self.leaves.keys()) == set(self.leave_list):
                return True
            else:
                return False


class Gradient(Variables):
    def __init__(self, placeholder, mesh, flow_data):
        super(Gradient, self).__init__(placeholder, mesh, flow_data, leaves=None)

        self.leave_list = None
        if not self._check_leaves():
            raise TypeError

        self.nb_cells = None
        self.faces = None

    def calculate(self, id_cell):
        self.nb_cells = self.mesh.cell_neighbours(id_cell)
        self.faces = self.mesh.cell_faces[id_cell]

        grad = np.zeros(self.n_val)

        return np.zeros(self.n_val)

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
