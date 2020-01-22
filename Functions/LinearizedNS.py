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
        for nb_cell, nb_face in zip(nb_cells, faces):
            area = self.mesh.face_area[nb_face]
            flip = -1.0 + 2.0 * float(nb_cell - id_cell or nb_cell < 0)

            rhs_vec += self._calc_inviscid_flux(data, id_cell, nb_cell, nb_face) * area * flip
            rhs_vec += self._calc_viscous_flux(data, id_cell, nb_cell, nb_face) * area * flip

        return self._conv2prime(rhs_vec, id_cell)

    def _grad_ave_field(self):
        grad = self._grad
        data = self._ave_field.data
        n_cell = self._ave_field.n_cell
        n_val = self._ave_field.n_val

        grad_ave = np.empty((n_cell, n_val, 3), dtype=np.float64)
        for i_cell, i_val in product(range(n_cell), range(n_val)):
            grad_ave[i_cell, i_val] = grad.formula(data, i_cell, i_val)

        return self._ave_field

    def _calc_inviscid_flux(self, data, id_cell, nb_cell, nb_face):
        vec_a, vec_b = self._get_cell_vals(data, id_cell, nb_cell, nb_face)
        vec_f = self.mesh.conv_vel(0.5 * (vec_a + vec_b), nb_face, 'G2L')
        rho = vec_f[0]
        u = vec_f[1]
        v = vec_f[2]
        w = vec_f[3]
        p = vec_f[4]
        e = vec_f[5]
        # t = vec_f[6]

        ave_data = self._ave_field.data
        ave_a, ave_b = self._get_cell_vals(ave_data, id_cell, nb_cell, nb_face)
        ave_f = self.mesh.conv_vel(0.5 * (ave_a + ave_b), nb_face, 'G2L')
        rho_ave = ave_f[0]
        u_ave = ave_f[1]
        v_ave = ave_f[2]
        w_ave = ave_f[3]
        p_ave = ave_f[4]
        e_ave = ave_f[5]
        # t_ave = ave_f[6]

        f = np.empty(5, dtype=np.float64)
        f[1] = rho * u_ave + rho_ave * u
        f[2] = 2.0 * rho_ave * u_ave * u + rho * u_ave * u_ave + p
        f[3] = rho_ave * u_ave * v + rho_ave * u * v_ave + rho * u_ave * v_ave
        f[4] = rho_ave * u_ave * w + rho_ave * u * w_ave + rho * u_ave * w_ave
        f[5] = rho_ave * (e_ave + p_ave) * u + rho_ave * (e + p) * u_ave + rho * (e_ave + p_ave) * u_ave

        return self.mesh.conv_vel(f, nb_face, 'L2G')

    def _calc_viscous_flux(self, data, id_cell, nb_cell, nb_face):
        ave_data = self._ave_field.data
        return np.zeros(5, dtype=np.float64)

    def _conv2prime(self, vec_conv, id_cell):
        rho = vec_conv[0]
        ru = vec_conv[1]
        rv = vec_conv[2]
        rw = vec_conv[3]
        e = vec_conv[4]

        ave_data = self._ave_field.data
        rho_ave = ave_data[id_cell, 0]
        u_ave = ave_data[id_cell, 1]
        v_ave = ave_data[id_cell, 2]
        w_ave = ave_data[id_cell, 3]
        p_ave = ave_data[id_cell, 4]
        ra_inv = 1.0 / rho_ave

        vec_pr = np.empty(5, dtype=np.float64)
        vec_pr[0] = rho
        vec_pr[1] = ra_inv * ru - u_ave * ra_inv * rho
        vec_pr[2] = ra_inv * rv - v_ave * ra_inv * rho
        vec_pr[3] = ra_inv * rw - w_ave * ra_inv * rho

        u = vec_pr[1]
        v = vec_pr[2]
        w = vec_pr[3]
        vec_pr[4] = 0.4 * e
        vec_pr[4] -= - 0.4 * 0.5 * rho * (u_ave * u_ave + v_ave * v_ave + w_ave * w_ave)
        vec_pr[4] -= - 0.4 * rho_ave * (u * u_ave + v * v_ave + w * w_ave)

        return vec_pr

    def _get_cell_vals(self, data, id_cell, nb_cell, nb_face):
        n_val = data.n_val

        def get_vals(i_cell):
            val_vec = np.empty(n_val, dtype=np.float64)
            for i_val in range(n_val):
                val_vec[i_val] = data[i_cell, i_val]
            return val_vec

        val_vec_0 = get_vals(id_cell)
        if nb_cell >= 0:  # For inner cells
            val_vec_nb = get_vals(nb_cell)
        else:  # For boundary cells
            val_vec_nb = self.bd_cond.get_bd_val(val_vec_0, nb_face, nb_cell)

        return val_vec_0, val_vec_nb

    def _get_face_grad(self, data, id_cell, nb_cell, nb_face):
        grad = self._grad
        return 0.0

    def _bd_for_grad(self):
        pass
