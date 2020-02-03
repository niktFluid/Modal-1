from itertools import product
import numpy as np

from Functions.Variables import Variables
from Functions.BoundaryCondition import BoundaryCondition as BDcond
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
        # self._vol_weight = mesh.volumes / np.sum(mesh.volumes)

        self._grad = Gradient(mesh, is2d=is2d)
        sub_list = [self._grad]

        super(LNS, self).__init__(mesh, n_return=5, sub_list=sub_list)

        self._ave_field = ave_field
        self._grad_ave = self._grad_ave_field()

        self._ref_cells = [0]
        self._grad_refs = [np.empty(0)]

    def _return_ref_cells(self, id_cell):
        cell_list = [id_cell] + self.mesh.cell_neighbours(id_cell)
        ref_cells = [i_cell for i_cell in cell_list if i_cell >= 0]
        return list(set(ref_cells))

    def formula(self, data, id_cell, **kwargs):
        self._ref_cells = self._return_ref_cells(id_cell)
        self._grad_data(data)

        nb_cells = self.mesh.cell_neighbours(id_cell)
        faces = self.mesh.cell_faces[id_cell]

        rhs_vec = np.zeros(5, dtype=np.float64)
        for nb_cell, nb_face in zip(nb_cells, faces):
            area = self.mesh.face_area[nb_face]
            flip = -1.0 + 2.0 * float(nb_cell - id_cell > 0 or nb_cell < 0)

            rhs_vec -= self._calc_inviscid_flux(data, id_cell, nb_cell, nb_face) * area * flip
            rhs_vec += self._calc_viscous_flux(data, id_cell, nb_cell, nb_face) * area * flip
        return self._conv2prime(rhs_vec, id_cell) / self.mesh.volumes[id_cell]

    def _grad_data(self, data):
        def grad(id_cell):
            grad_data = np.empty((data.shape[1], 3), dtype=np.float64)
            for i_val in range(data.n_val):
                grad_data[i_val] = self._grad.formula(data, id_cell, i_val)
            return grad_data
        self._grad_refs = [grad(i_cell) for i_cell in self._ref_cells]

    def _grad_ave_field(self):
        data = self._ave_field.data
        n_cell = self._ave_field.n_cell
        n_val = self._ave_field.n_val

        grad_ave = np.empty((n_cell, n_val, 3), dtype=np.float64)
        for i_cell, i_val in product(range(n_cell), range(n_val)):
            grad_ave[i_cell, i_val] = self._grad.formula(data, i_cell, i_val)
        return grad_ave

    def _calc_inviscid_flux(self, data, id_cell, nb_cell, nb_face):
        def flux(face_val, face_ave):
            rho = face_val[0]
            u = face_val[1]
            v = face_val[2]
            w = face_val[3]
            p = face_val[4]
            e = face_val[5]
            # t = vec_f[6]

            rho_ave = face_ave[0]
            u_ave = face_ave[1]
            v_ave = face_ave[2]
            w_ave = face_ave[3]
            p_ave = face_ave[4]
            e_ave = face_ave[5]
            # t_ave = ave_f[6]

            f = np.empty(5, dtype=np.float64)
            f[0] = rho * u_ave + rho_ave * u
            f[1] = 2.0 * rho_ave * u_ave * u + rho * u_ave * u_ave + p
            f[2] = rho_ave * u_ave * v + rho_ave * u * v_ave + rho * u_ave * v_ave
            f[3] = rho_ave * u_ave * w + rho_ave * u * w_ave + rho * u_ave * w_ave
            f[4] = rho_ave * (e_ave + p_ave) * u + rho_ave * (e + p) * u_ave + rho * (e_ave + p_ave) * u_ave
            return f

        # vec_a, vec_b = self._get_face_vals(data, id_cell, nb_cell, nb_face)
        # ave_a, ave_b = self._get_face_vals(self._ave_field.data, id_cell, nb_cell, nb_face, ave_value=True)
        #
        # vec_fa = self.mesh.conv_vel(vec_a, nb_face)
        # ave_fa = self.mesh.conv_vel(ave_a, nb_face)
        # fa = flux(vec_fa, ave_fa)
        #
        # vec_fb = self.mesh.conv_vel(vec_b, nb_face)
        # ave_fb = self.mesh.conv_vel(ave_b, nb_face)
        # fb = flux(vec_fb, ave_fb)

        vec_id, vec_nb = self._get_cell_vals(data, id_cell, nb_cell, nb_face)
        vec_f = 0.5 * (vec_id + vec_nb)

        ave_id, ave_nb = self._get_cell_vals(self._ave_field.data, id_cell, nb_cell, nb_face)
        ave_f = 0.5 * (ave_id + ave_nb)

        return self.mesh.conv_vel(flux(vec_f, ave_f), nb_face, inverse=True)

    def _calc_viscous_flux(self, data, id_cell, nb_cell, nb_face):
        flux = np.zeros(5, dtype=np.float64)
        face_normal_vec = self.mesh.face_mat[nb_face, 0]

        vec_a, vec_b = self._get_cell_vals(data, id_cell, nb_cell, nb_face)
        vec_f = 0.5 * (vec_a + vec_b)
        u_vel = vec_f[1:4]

        g_face = self._get_face_grad(data, id_cell, nb_cell, nb_face)
        tau = self._get_stress_tensor(g_face)

        ave_data = self._ave_field.data
        ave_a, ave_b = self._get_cell_vals(ave_data, id_cell, nb_cell, nb_face)
        ave_f = 0.5 * (ave_a + ave_b)
        u_ave = ave_f[1:4]

        g_face_ave = self._get_face_grad(data, id_cell, nb_cell, nb_face, ave_value=True)
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

        # Convert to pressure
        # u = vec_pr[1]
        # v = vec_pr[2]
        # w = vec_pr[3]
        # vec_pr[4] = 0.4 * e
        # vec_pr[4] += - 0.4 * 0.5 * rho * (u_ave * u_ave + v_ave * v_ave + w_ave * w_ave)
        # vec_pr[4] += - 0.4 * rho_ave * (u * u_ave + v * v_ave + w * w_ave)

        # Concert to temperature
        u = vec_pr[1]
        v = vec_pr[2]
        w = vec_pr[3]
        vec_pr[4] = 1.4 * 0.4 * (e * ra_inv - rho * ra_inv * ra_inv)
        vec_pr[4] += - 1.4 * 0.4 * rho_ave * (u * u_ave + v * v_ave + w * w_ave)

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

    def _get_grad_vec(self, i_cell, ave_value):
        if not ave_value:
            return self._grad_refs[self._ref_cells.index(i_cell)]
        else:
            return self._grad_ave[i_cell]

    def _get_face_vals(self, data, id_cell, nb_cell, nb_face, ave_value=False):
        val_0, val_nb = self._get_cell_vals(data, id_cell, nb_cell, nb_face)

        def reconstruct(val_vec, ind):
            grad = self._get_grad_vec(ind, ave_value)
            r_vec = self.mesh.face_centers[nb_face] - self.mesh.centers[ind]
            return val_vec + grad @ r_vec

        if nb_cell >= 0:
            val_vec_0 = reconstruct(val_0, id_cell)
            val_vec_nb = reconstruct(val_nb, nb_cell)
        else:
            val_vec_0, val_vec_nb = val_0, val_nb

        return val_vec_0, val_vec_nb

    def _get_face_grad(self, data, id_cell, nb_cell, nb_face, ave_value=False):
        grad_id = self._get_grad_vec(id_cell, ave_value)
        vol_id = self.mesh.volumes[id_cell]
        if nb_cell >= 0:  # For inner faces
            grad_nb = self._get_grad_vec(nb_cell, ave_value)
            vol_nb = self.mesh.volumes[nb_cell]
        else:  # For boundary faces.
            grad_nb = grad_id
            vol_nb = vol_id

        grad_face = (grad_id * vol_id + grad_nb * vol_nb) / (vol_id + vol_nb)
        # return (grad_id * vol_id + grad_nb * vol_nb) / (vol_id + vol_nb)

        # For prevent even-odd instability.
        vec_lr = self._get_pos_diff(id_cell, nb_cell, nb_face)
        inv_lr = 1.0 / (vec_lr[0]*vec_lr[0] + vec_lr[1]*vec_lr[1] + vec_lr[2]*vec_lr[2])
        vec_a, vec_b = self._get_cell_vals(data, id_cell, nb_cell, nb_face)
        coef = (grad_face @ vec_lr - (vec_b - vec_a)) * inv_lr

        return grad_face - coef.reshape(7, 1) @ vec_lr.reshape(1, 3) * inv_lr

    def _get_pos_diff(self, id_cell, nb_cell, nb_face):
        flip = -1.0 + 2.0 * float(nb_cell - id_cell > 0 or nb_cell < 0)
        return self.mesh.vec_lr[nb_face] * flip

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
