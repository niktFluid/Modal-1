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

        self.n_cell = mesh.n_cell
        self.n_val = 7

        self.mesh = mesh
        self.bd_cond = BDcond(mesh)
        # self._vol_weight = mesh.volumes / np.sum(mesh.volumes)

        self._grad = Gradient(mesh, is2d=is2d)
        sub_list = [self._grad]

        super(LNS, self).__init__(mesh, n_return=5, sub_list=sub_list)

        self._data = None
        self.ave_data = ave_field.data
        self._grad_ave = self._grad_ave_field()

        self._ref_cells = [0]
        self._grad_refs = {}

    def _return_ref_cells(self, id_cell):
        cell_list = [id_cell] + self.mesh.cell_neighbours(id_cell)
        ref_cells = [i_cell for i_cell in cell_list if i_cell >= 0]
        return list(set(ref_cells))

    @staticmethod
    def _get_face_direction(id_cell, nb_cell):
        return -1.0 + 2.0 * float(nb_cell > id_cell >= 0)

    def formula(self, data, id_cell, **kwargs):
        self._data = data
        self._ref_cells = self._return_ref_cells(id_cell)
        self._grad_data()

        nb_cells = self.mesh.cell_neighbours(id_cell)
        faces = self.mesh.cell_faces[id_cell]
        rhs_vec = np.zeros(5, dtype=np.float64)

        for nb_cell, nb_face in zip(nb_cells, faces):
            area = self.mesh.face_area[nb_face]
            flip = self._get_face_direction(id_cell, nb_cell)

            rhs_vec -= self._calc_inviscid_flux(id_cell, nb_cell, nb_face) * area * flip
            rhs_vec += self._calc_viscous_flux(id_cell, nb_cell, nb_face) * area * flip
        return self._conv2prime(rhs_vec, id_cell) / self.mesh.volumes[id_cell]

    def _grad_data(self):
        def grad(id_cell):
            grad_data = np.empty((self.n_val, 3), dtype=np.float64)
            for i_val in range(self.n_val):
                grad_data[i_val] = self._grad.formula(self._data, id_cell, i_val)
            return grad_data
        self._grad_refs = {i_cell: grad(i_cell) for i_cell in self._ref_cells}

    def _grad_ave_field(self):
        grad_ave = np.empty((self.n_cell, self.n_val, 3), dtype=np.float64)
        for i_cell, i_val in product(range(self.n_cell), range(self.n_val)):
            grad_ave[i_cell, i_val] = self._grad.formula(self.ave_data, i_cell, i_val)
        return grad_ave

    def _calc_inviscid_flux(self, id_cell, nb_cell, nb_face):
        def flux(vec_face, ave_face):
            rho = vec_face[0]
            u = vec_face[1]
            v = vec_face[2]
            w = vec_face[3]
            # t = vec_f[4]
            p = vec_face[5]
            e = vec_face[6]

            rho_ave = ave_face[0]
            u_ave = ave_face[1]
            v_ave = ave_face[2]
            w_ave = ave_face[3]
            # t_ave = ave_f[4]
            p_ave = ave_face[5]
            e_ave = ave_face[6]

            f = np.empty(5, dtype=np.float64)
            f[0] = rho * u_ave + rho_ave * u
            f[1] = 2.0 * rho_ave * u_ave * u + rho * u_ave * u_ave + p
            f[2] = rho_ave * u_ave * v + rho_ave * u * v_ave + rho * u_ave * v_ave
            f[3] = rho_ave * u_ave * w + rho_ave * u * w_ave + rho * u_ave * w_ave
            f[4] = (e_ave + p_ave) * u + (e + p) * u_ave
            return f

        # vec_a, vec_b = self._get_face_vals(data, id_cell, nb_cell, nb_face)
        # ave_a, ave_b = self._get_face_vals(self._ave_field.data, id_cell, nb_cell, nb_face, ave_value=True)

        # vec_fa = self.mesh.conv_vel(vec_a, nb_face)
        # ave_fa = self.mesh.conv_vel(ave_a, nb_face)
        # fa = flux(vec_fa, ave_fa)

        # vec_fb = self.mesh.conv_vel(vec_b, nb_face)
        # ave_fb = self.mesh.conv_vel(ave_b, nb_face)
        # fb = flux(vec_fb, ave_fb)

        vec_0, vec_nb = self._get_cell_vals(self._data, id_cell, nb_cell, nb_face)
        vec_f = self.mesh.conv_vel(0.5 * (vec_0 + vec_nb), nb_face)

        ave_0, ave_nb = self._get_cell_vals(self.ave_data, id_cell, nb_cell, nb_face)
        ave_f = self.mesh.conv_vel(0.5 * (ave_0 + ave_nb), nb_face)

        # return self.mesh.conv_vel(0.5 * (fa + fb), nb_face, inverse=True)
        return self.mesh.conv_vel(flux(vec_f, ave_f), nb_face, inverse=True)

    def _calc_viscous_flux(self, id_cell, nb_cell, nb_face):
        flux = np.zeros(5, dtype=np.float64)
        face_normal_vec = self.mesh.face_mat[nb_face, 0]  # * self._get_face_direction(id_cell, nb_cell)

        vec_a, vec_b = self._get_cell_vals(self._data, id_cell, nb_cell, nb_face)
        vec_f = 0.5 * (vec_a + vec_b)
        u_vel = vec_f[1:4]

        g_face = self._get_face_grad(self._data, self._grad_refs, id_cell, nb_cell, nb_face)
        tau = self._get_stress_tensor(g_face)

        ave_a, ave_b = self._get_cell_vals(self.ave_data, id_cell, nb_cell, nb_face)
        ave_f = 0.5 * (ave_a + ave_b)
        u_ave = ave_f[1:4]

        g_face_ave = self._get_face_grad(self.ave_data, self._grad_ave, id_cell, nb_cell, nb_face)
        tau_ave = self._get_stress_tensor(g_face_ave)

        flux[1:4] = tau @ face_normal_vec
        energy_flux = tau @ u_ave + tau_ave @ u_vel + self.coef_heat_flux * g_face[4, :]
        flux[4] = energy_flux @ face_normal_vec
        return flux

    def _conv2prime(self, vec_conv, id_cell):
        # Convert the conservative variables to the prime variables.
        # [rho, rho-u, rho-v, rho-w, e] -> [rho, u, v, w, T]
        rho = vec_conv[0]
        ru = vec_conv[1]
        rv = vec_conv[2]
        rw = vec_conv[3]
        e = vec_conv[4]

        rho_ave = self.ave_data[id_cell, 0]
        u_ave = self.ave_data[id_cell, 1]
        v_ave = self.ave_data[id_cell, 2]
        w_ave = self.ave_data[id_cell, 3]
        # t_ave = ave_data[id_cell, 4]
        e_ave = self.ave_data[id_cell, 6]
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
        vec_pr[4] = 1.4 * 0.4 * (e * ra_inv - e_ave * rho * ra_inv * ra_inv)
        vec_pr[4] += - 1.4 * 0.4 * (u * u_ave + v * v_ave + w * w_ave)

        return vec_pr

    def _get_cell_vals(self, data, id_cell, nb_cell, nb_face):
        def get_vals(i_cell):
            val_vec = np.empty(self.n_val, dtype=np.float64)
            for i_val in range(self.n_val):
                val_vec[i_val] = data[i_cell, i_val]
            return val_vec

        val_vec_0 = get_vals(id_cell)
        if nb_cell >= 0:  # For inner cells
            val_vec_nb = get_vals(nb_cell)
        else:  # For boundary cells
            val_vec_nb = self.bd_cond.get_bd_val(val_vec_0, nb_face)
        return val_vec_0, val_vec_nb

    def _get_face_vals(self, data, grad_data, id_cell, nb_cell, nb_face):
        val_0, val_nb = self._get_cell_vals(data, id_cell, nb_cell, nb_face)

        def reconstruct(val_vec, ind):
            grad = grad_data[ind]
            r_vec = self.mesh.face_centers[nb_face] - self.mesh.centers[ind]
            return val_vec + grad @ r_vec

        if nb_cell >= 0:
            val_vec_0 = reconstruct(val_0, id_cell)
            val_vec_nb = reconstruct(val_nb, nb_cell)
        else:
            val_vec_0, val_vec_nb = val_0, val_nb

        return val_vec_0, val_vec_nb

    def _get_face_grad(self, data, grad_data, id_cell, nb_cell, nb_face):
        grad_id = grad_data[id_cell]
        vol_id = self.mesh.volumes[id_cell]
        if nb_cell >= 0:  # For inner faces
            grad_nb = grad_data[nb_cell]
            vol_nb = self.mesh.volumes[nb_cell]
        else:  # For boundary faces.
            grad_nb = grad_data[id_cell]
            vol_nb = self.mesh.volumes[id_cell]

        grad_face = (grad_id * vol_nb + grad_nb * vol_id) / (vol_id + vol_nb)
        # grad_face = 0.5 * (grad_id + grad_nb)

        # For prevent even-odd instability.
        flip = self._get_face_direction(id_cell, nb_cell)
        vec_lr = self.mesh.vec_lr[nb_face]
        inv_lr = 1.0 / (vec_lr[0]*vec_lr[0] + vec_lr[1]*vec_lr[1] + vec_lr[2]*vec_lr[2])
        vec_a, vec_b = self._get_cell_vals(data, id_cell, nb_cell, nb_face)
        coef = (grad_face @ vec_lr - (vec_b - vec_a) * flip) * inv_lr

        return grad_face - coef.reshape(7, 1) @ vec_lr.reshape(1, 3) * inv_lr

    def _get_stress_tensor(self, grad):
        tensor = np.empty((3, 3), dtype=np.float64)
        mu = self.mu

        dudx = grad[1, 0]
        dudy = grad[1, 1]
        dudz = grad[1, 2]
        dvdx = grad[2, 0]
        dvdy = grad[2, 1]
        dvdz = grad[2, 2]
        dwdx = grad[3, 0]
        dwdy = grad[3, 1]
        dwdz = grad[3, 2]

        div_u = (dudx + dvdy + dwdz) / 3.0
        tensor[0, 0] = 2.0 * mu * (dudx - div_u)
        tensor[1, 1] = 2.0 * mu * (dvdy - div_u)
        tensor[2, 2] = 2.0 * mu * (dwdz - div_u)

        tensor[0, 1] = mu * (dudy + dvdx)
        tensor[0, 2] = mu * (dudz + dwdx)
        tensor[1, 2] = mu * (dwdy + dvdz)

        tensor[1, 0] = tensor[0, 1]
        tensor[2, 0] = tensor[0, 2]
        tensor[2, 1] = tensor[1, 2]

        return tensor


class LNS2(LNS):
    def __init__(self, mesh, ave_field, mu, pr, is2d=False):
        super(LNS2, self).__init__(mesh, ave_field, mu, pr, is2d)

    def formula(self, data, id_cell, **kwargs):
        eps = 1.0e-6  # Knoll and Keyes, 2004

        self._data = self.ave_data

    def _conv2prime(self, vec_conv, id_cell):
        pass

    def _calc_inviscid_flux(self, id_cell, nb_cell, nb_face):
        pass

    def _calc_viscous_flux(self, id_cell, nb_cell, nb_face):
        pass
