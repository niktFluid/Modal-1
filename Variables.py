import numpy as np
# from scipy.sparse import csr_matrix


class Variables:
    def __init__(self, placeholder, mesh, flow_data, leaves=None):
        self.ph = placeholder
        self.mesh = mesh
        self.data = flow_data
        self.leaves = leaves

        self._check_leaves()

    def calculate(self, id_cell):
        raise NotImplementedError

    def _boundary(self):
        pass

    def _check_leaves(self):
        if isinstance(self.leaves, dict):
            return True
        else:
            return False
