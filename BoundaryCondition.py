import numpy as np


class BoundaryCondition:
    def __init__(self, mesh):
        self.mesh = mesh

        self._id_wall = -11
        self._id_moving_wall = -10
        self._id_empty = -12

    def get_bd_val(self, val_vec, id_face, id_bd):
        vec_loc = self._conv_vec(val_vec, id_face, conv_type='G2L')

        bd_func = self._select_bd_func(id_bd)
        bd_val = bd_func(vec_loc, id_face)

        # noinspection PyTypeChecker
        return self._conv_vec(bd_val, id_face, conv_type='L2G')

    def _select_bd_func(self, id_bd):
        if id_bd == self._id_wall:
            return self._bd_wall
        if id_bd == self._id_moving_wall:
            return self._bd_wall
        if id_bd == self._id_empty:
            return self._bd_symmetry

    def _conv_vec(self, val_vec, id_face, conv_type):
        if conv_type == 'G2L':
            vec_n = self.mesh.face_vec_n[id_face]
            vec_t1 = self.mesh.face_vec_t1[id_face]
            vec_t2 = self.mesh.face_vec_t2[id_face]
        elif conv_type == 'L2G':
            vec_n = self.mesh.face_vec_ni[id_face]
            vec_t1 = self.mesh.face_vec_t1i[id_face]
            vec_t2 = self.mesh.face_vec_t2i[id_face]
        else:
            raise TypeError

        mat_conv = np.vstack((vec_n, vec_t1, vec_t2))

        rho, u_vel, pressure = self._split_vec(val_vec)
        u_conv = mat_conv @ u_vel

        return np.hstack((rho, u_conv, pressure))

    @staticmethod
    def _split_vec(val_vec):
        # print(val_vec)
        return val_vec[0], val_vec[1:4], val_vec[4]

    def _bd_symmetry(self, val_vec, _):
        rho, u_vel, pressure = self._split_vec(val_vec)

        ref_1 = np.array([-1, 1, 1], dtype=np.float64)
        u_bd = u_vel * ref_1

        return np.hstack((rho, u_bd, pressure))

    def _bd_wall(self, val_vec, id_face):
        rho, u_vel, pressure = self._split_vec(val_vec)

        bc_list = self.mesh.boundary
        bd_id = self.mesh.get_bd_id(id_face)

        vel_wall_g = bc_list[bd_id].u_val
        if vel_wall_g is None:
            vel_wall_g = np.array([0, 0, 0], dtype=np.float64)

        vec_wall = self._conv_vec(np.hstack((rho, vel_wall_g, pressure)), id_face, conv_type='G2L')
        _, vel_wall, _ = self._split_vec(vec_wall)

        u_bd = 2.0 * vel_wall - u_vel

        return np.hstack((rho, u_bd, pressure))


class BoundaryData:
    def __init__(self, mesh, field):
        self.state_id = 0

        self.mesh = mesh
        self.field = field

        self.bd_data = None

    def get_bd(self):
        if self.state_id != self.field.get_state_id():
            self.set_bd()
        return self.bd_data

    def set_bd(self):
        bd_cells = self.mesh.bd_cells
        bc_list = self.mesh.boundary
        int_data = self.field.get_interface_data(bd_cells)

        # Global -> local coordinate
        u_loc = [self._g2l(i_bd, int_data[i_bd][:, 0:3]) for i_bd in range(len(bc_list))]
        p = [int_data[i_bd][:, -1] for i_bd in range(len(bc_list))]

        self.bd_data = [self._select_bd(bc.type)(i_bd, u_loc[i_bd], p[i_bd])
                        for i_bd, bc in enumerate(bc_list)]
        self.state_id = self.field.get_state_id()

    def _select_bd(self, bd_type):
        if bd_type == 'wall':
            return self._bd_wall
        if bd_type == 'empty':
            return self._bd_symmetry

    def _g2l(self, i_bd, u_global):
        _, vec_n, vec_t1, vec_t2 = self.mesh.get_bd_geom(i_bd)

        return np.vstack((
            np.sum(u_global * vec_n, axis=1),
            np.sum(u_global * vec_t1, axis=1),
            np.sum(u_global * vec_t2, axis=1)
        )).T

    def _l2g(self, i_bd, u_local):
        _, vec_ni, vec_t1i, vec_t2i = self.mesh.get_bd_geom_inv(i_bd)

        return np.vstack((
            np.sum(u_local * vec_ni, axis=1),
            np.sum(u_local * vec_t1i, axis=1),
            np.sum(u_local * vec_t2i, axis=1)
        )).T

    def _bd_wall(self, i_bd, u_loc, p):
        bc_list = self.mesh.boundary

        u_wall = bc_list[i_bd].u_val
        if u_wall is None:
            u_wall = np.array([0, 0, 0], dtype=np.float64)

        uw_array = np.tile(u_wall, (u_loc.shape[0], 1))
        u_bd = 2.0 * self._g2l(i_bd, uw_array) - u_loc
        p_bd = p

        return np.hstack((self._l2g(i_bd, u_bd), p_bd[:, np.newaxis]))

    def _bd_symmetry(self, i_bd, u_loc, p):
        ref_1 = np.array([-1, 1, 1], dtype=np.float64).reshape((1, 3))
        u_bd = u_loc * ref_1
        p_bd = p

        return np.hstack((self._l2g(i_bd, u_bd), p_bd[:, np.newaxis]))
