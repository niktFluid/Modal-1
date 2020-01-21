from itertools import product
import numpy as np

from Variables import Variables
from BoundaryCondition import BoundaryCondition as BDcond
from Functions.Gradient import Gradient


class LNS(Variables):  # Linearized Navier-Stokes equations
    def __init__(self, mesh, ave_field):
        self.mesh = mesh
        self.bd_cond = BDcond(mesh)

        self._grad = Gradient(mesh)
        sub_list = [self._grad]

        super(LNS, self).__init__(mesh, sub_list)

        self._ave_field = ave_field
        self._grad_ave = self._grad_ave_field()

    def return_ref_cells(self, id_cell):
        cell_list = [id_cell] + self.mesh.cell_neighbours(id_cell)
        ref_cells = [i_cell for i_cell in cell_list if i_cell >= 0]

        return list(set(ref_cells))

    def formula(self, data, id_cell, **kwargs):
        pass

    def _grad_ave_field(self):
        grad = self._grad
        data = self._ave_field.data
        n_cell = self._ave_field.n_cell
        n_val = self._ave_field.n_val

        grad_ave = np.empty((n_cell, n_val, 3), dtype=np.float64)
        for i_cell, i_val in product(range(n_cell), range(n_val)):
            grad_ave[i_cell, i_val, :] = grad.formula(data, i_cell, i_val)

        return self._ave_field

    def _bd_for_grad(self):
        pass
