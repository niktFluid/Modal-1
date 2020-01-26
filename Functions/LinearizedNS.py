from itertools import product
import numpy as np

from Variables import Variables
from BoundaryCondition import BoundaryCondition as BDcond
from Functions.Gradient import Gradient


class LNS(Variables):  # Linearized Navier-Stokes equations
    def __init__(self, mesh, ave_field, mu, pr, is2d=False):
        self.gamma = 1.4
        self.gamma_1 = 1.0 / (1.4 - 1.0)
        self.mu = mu
        self.pr = pr
        self.coef_heat_flux = mu / ((self.gamma - 1) * pr)

        self.mesh = mesh
        self.bd_cond = BDcond(mesh)

        self._grad = Gradient(mesh, is2d=is2d)
        sub_list = [self._grad]

        super(LNS, self).__init__(mesh, n_return=5, sub_list=sub_list)

        self._ave_field = ave_field
        self._grad_ave = self._grad_ave_field()

    def _return_ref_cells(self, id_cell):
        cell_list = [id_cell] + self.mesh.cell_neighbours(id_cell)
        ref_cells = [i_cell for i_cell in cell_list if i_cell >= 0]

        return list(set(ref_cells))

    def formula(self, data, id_cell, **kwargs):
        nb_cells = self.mesh.cell_neighbours(id_cell)
        faces = self.mesh.cell_faces[id_cell]

        rhs_vec = np.zeros(5, dtype=np.float64)
        for nb_cell, nb_face in zip(nb_cells, faces):
            area = self.mesh.face_area[nb_face]
            flip = -1.0 + 2.0 * float(nb_cell - id_cell > 0 or nb_cell < 0)

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

        return grad_ave

    def _calc_inviscid_flux(self, data, id_cell, nb_cell, nb_face):
        vec_a, vec_b = self._get_cell_vals(data, id_cell, nb_cell, nb_face)
        vec_f = self.mesh.conv_vel(0.5 * (vec_a + vec_b), nb_face)
        rho = vec_f[0]
        u = vec_f[1]
        v = vec_f[2]
        w = vec_f[3]
        p = vec_f[4]
        e = vec_f[5]
        # t = vec_f[6]

        ave_data = self._ave_field.data
        ave_a, ave_b = self._get_cell_vals(ave_data, id_cell, nb_cell, nb_face)
        ave_f = self.mesh.conv_vel(0.5 * (ave_a + ave_b), nb_face)
        rho_ave = ave_f[0]
        u_ave = ave_f[1]
        v_ave = ave_f[2]
        w_ave = ave_f[3]
        p_ave = ave_f[4]
        e_ave = ave_f[5]
        # t_ave = ave_f[6]

        f = np.empty(5, dtype=np.float64)
        f[0] = rho * u_ave + rho_ave * u
        f[1] = 2.0 * rho_ave * u_ave * u + rho * u_ave * u_ave + p
        f[2] = rho_ave * u_ave * v + rho_ave * u * v_ave + rho * u_ave * v_ave
        f[3] = rho_ave * u_ave * w + rho_ave * u * w_ave + rho * u_ave * w_ave
        f[4] = rho_ave * (e_ave + p_ave) * u + rho_ave * (e + p) * u_ave + rho * (e_ave + p_ave) * u_ave

        return self.mesh.conv_vel(f, nb_face, inverse=True)

    def _calc_viscous_flux(self, data, id_cell, nb_cell, nb_face):
        flux = np.zeros(5, dtype=np.float64)
        face_normal_vec = self.mesh.face_mat[nb_face, 0]

        vec_a, vec_b = self._get_cell_vals(data, id_cell, nb_cell, nb_face)
        vec_f = 0.5 * (vec_a + vec_b)
        u_vel = vec_f[1:4]

        g_face = self._get_face_grad(id_cell, nb_cell, data)
        tau = self._get_stress_tensor(g_face)

        ave_data = self._ave_field.data
        ave_a, ave_b = self._get_cell_vals(ave_data, id_cell, nb_cell, nb_face)
        ave_f = 0.5 * (ave_a + ave_b)
        u_ave = ave_f[1:4]

        g_face_ave = self._get_face_grad(id_cell, nb_cell, ave_value=True)
        tau_ave = self._get_stress_tensor(g_face_ave)

        flux[1:4] = tau @ face_normal_vec
        energy_flux = tau @ u_ave + tau_ave @ u_vel + self.coef_heat_flux * g_face[6, :]
        flux[4] = energy_flux @ face_normal_vec
        return flux

    def _conv2prime(self, vec_conv, id_cell):
        # Convert the conservative variables to the prime variables.
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
        # p_ave = ave_data[id_cell, 4]
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
        vec_pr[4] += - 0.4 * 0.5 * rho * (u_ave * u_ave + v_ave * v_ave + w_ave * w_ave)
        vec_pr[4] += - 0.4 * rho_ave * (u * u_ave + v * v_ave + w * w_ave)

        return vec_pr

    def _get_cell_vals(self, data, id_cell, nb_cell, nb_face):
        n_val = data.shape[1]

        def get_vals(i_cell):
            val_vec = np.empty(n_val, dtype=np.float64)
            for i_val in range(n_val):
                val_vec[i_val] = data[i_cell, i_val]
            return val_vec

        val_vec_0 = get_vals(id_cell)
        if nb_cell >= 0:  # For inner cells
            val_vec_nb = get_vals(nb_cell)
        else:  # For boundary cells
            val_vec_nb = self.bd_cond.get_bd_val(val_vec_0, nb_face)

        return val_vec_0, val_vec_nb

    def _get_face_grad(self, id_cell, nb_cell, data=None, val_list=None, ave_value=False):
        if val_list is None:
            val_list = [1, 2, 3, 6]  # Only consider the velocities and temperature.

        def get_grad_vec(i_cell):
            if ave_value:
                grad_vec = self._grad_ave[id_cell]
            else:
                grad_vec = np.zeros((data.n_val, 3), dtype=np.float64)
                for i_val in val_list:
                    grad_vec[i_val] = self._grad.formula(data, i_cell)
            return grad_vec
        grad_id = get_grad_vec(id_cell)
        vol_ib = self.mesh.volumes[id_cell]

        if nb_cell > 0:  # For inner faces
            grad_nb = get_grad_vec(nb_cell)
            vol_nb = self.mesh.volumes[nb_cell]
        else:  # For boundary faces.
            grad_nb = get_grad_vec(id_cell)
            vol_nb = self.mesh.volumes[id_cell]

        vol_inv = 1.0 / (vol_ib + vol_nb)
        grad_face = (grad_id * vol_ib + grad_nb * vol_nb) * vol_inv

        return grad_face

    def _get_stress_tensor(self, grad):
        tensor = np.empty((3, 3), dtype=np.float64)
        mu = self.mu

        div_u = (grad[1, 0] + grad[2, 1] + grad[3, 2]) / 3.0
        tensor[0, 0] = 2.0 * mu * (grad[1, 0] - div_u)
        tensor[1, 1] = 2.0 * mu * (grad[2, 1] - div_u)
        tensor[2, 2] = 2.0 * mu * (grad[3, 2] - div_u)

        tensor[0, 1] = mu * (grad[1, 1] + grad[2, 0])
        tensor[0, 2] = mu * (grad[1, 2] + grad[3, 0])
        tensor[1, 2] = mu * (grad[2, 2] + grad[3, 1])

        tensor[1, 0] = tensor[0, 1]
        tensor[2, 0] = tensor[0, 2]
        tensor[2, 1] = tensor[1, 2]

        return tensor
