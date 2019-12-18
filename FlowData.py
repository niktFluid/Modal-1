import numpy as np
import Ofpp


class FlowField:
    def __init__(self):
        dummy = np.empty(0)

        self.state_id = 0
        self.n_val = 0
        self.n_size = 0
        self.data = dummy

        self._init_field()

    def update_state_id(self):
        self.state_id += 1

    def _init_field(self, *args, **kwargs):
        raise NotImplementedError

    def update_from_file(self, *args, **kwargs):
        raise NotImplementedError

    def get_data(self):
        return self.data

    def get_interface_data(self, bd_cells):
        return [self.data[ind] for ind in bd_cells]

    def get_state_id(self):
        return self.state_id


class OfData(FlowField):
    def __init__(self, path_dir, path_u, path_p, path_rho):
        self.path_dir = path_dir
        self.path_u = path_u
        self.path_p = path_p
        self.path_rho = path_rho

        super(OfData, self).__init__()

    def _init_field(self):
        self.update_from_file()

    def update_from_file(self, path_u=None, path_p=None, path_rho=None):
        if path_u is None:
            path_u = self.path_u

        if path_p is None:
            path_p = self.path_p

        if path_rho is None:
            path_rho = self.path_rho

        u_data = Ofpp.parse_internal_field(self.path_dir + path_u)
        p_data = Ofpp.parse_internal_field(self.path_dir + path_p)
        rho_data = Ofpp.parse_internal_field(self.path_dir + path_rho)

        # For confirmation
        # _mesh = Ofpp.FoamMesh(self.path_dir)
        # _mesh.read_cell_centres(self.path_dir + '0.5/C')
        # u_data = _mesh.cell_centres

        self.n_val = u_data.shape[1] + 1
        self.n_size = u_data.shape[0]

        self.data = np.hstack((rho_data[:, np.newaxis], u_data, p_data[:, np.newaxis]))
        self.update_state_id()
