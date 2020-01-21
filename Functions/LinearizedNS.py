from itertools import product
import numpy as np

from Variables import Variables
from BoundaryCondition import BoundaryCondition as BDcond
from Functions.Gradient import Gradient


class LNS(Variables):  # Linearized Navier-Stokes equations
    def __init__(self, mesh, ave_field, mu, kappa, pr):
        self.gamma = 1.4
        self.gamma_1 = 1.0 / (1.4 - 1.0)
        self.mu = mu
        self.kappa = kappa
        self.pr = pr
        self.pr_inv = 1.0 / pr

        self.mesh = mesh
        self.bd_cond = BDcond(mesh)

        self._grad = Gradient(mesh)
        sub_list = [self._grad]

        super(LNS, self).__init__(mesh, n_return=5, sub_list=sub_list)

        self._ave_field = ave_field
        self._grad_ave = self._grad_ave_field()

    def return_ref_cells(self, id_cell):
        cell_list = [id_cell] + self.mesh.cell_neighbours(id_cell)
        ref_cells = [i_cell for i_cell in cell_list if i_cell >= 0]

        return list(set(ref_cells))

    def formula(self, data, id_cell, **kwargs):
        nb_cells = self.mesh.cell_neighbours(id_cell)
        faces = self.mesh.cell_faces[id_cell]

        rhs_vec = np.zeros(5, dtype=np.float64)
        for id_cell, id_face in zip(nb_cells, faces):
            rhs_vec += self._calc_inviscid_flux(id_cell, id_face)
            rhs_vec += self._calc_viscous_flux(id_cell, id_face)

        return rhs_vec

    def _grad_ave_field(self):
        grad = self._grad
        data = self._ave_field.data
        n_cell = self._ave_field.n_cell
        n_val = self._ave_field.n_val

        grad_ave = np.empty((n_cell, n_val, 3), dtype=np.float64)
        for i_cell, i_val in product(range(n_cell), range(n_val)):
            grad_ave[i_cell, i_val] = grad.formula(data, i_cell, i_val)

        return self._ave_field

    def _calc_inviscid_flux(self, id_cell, id_face):
        ave_data = self._ave_field.data
        return np.zeros(5, dtype=np.float64)

    def _calc_viscous_flux(self, id_cell, id_face):
        ave_data = self._ave_field.data
        return np.zeros(5, dtype=np.float64)

    def _bd_for_grad(self):
        pass
