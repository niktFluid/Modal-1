import numpy as np


class BoundaryCondition:
    def __init__(self, mesh):
        self.mesh = mesh

        self._id_wall = -11
        self._id_moving_wall = -10
        self._id_empty = -12

        self._bd_vel = np.empty(5, dtype=np.float64)

    def get_bd_val(self, val_vec, id_face, id_bd):
        rho, vel_vec, pressure = self._split_vec(val_vec)

        vel_loc = self._conv_vel(vel_vec, id_face, conv_type='G2L')

        bd_func = self._select_bd_func(id_bd)
        bd_rho, bd_vel, bd_pres = bd_func(rho, vel_loc, pressure, id_face)

        # noinspection PyTypeChecker
        bd_vel_g = self._conv_vel(bd_vel, id_face, conv_type='L2G')

        self._bd_vel[0] = bd_rho
        self._bd_vel[1:4] = bd_vel_g
        self._bd_vel[4] = bd_pres

        return self._bd_vel

    def _select_bd_func(self, id_bd):
        if id_bd == self._id_wall:
            return self._bd_wall
        if id_bd == self._id_moving_wall:
            return self._bd_wall
        if id_bd == self._id_empty:
            return self._bd_symmetry

    def _conv_vel(self, u_vel, id_face, conv_type):
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

        u_loc = vec_n @ u_vel
        v_loc = vec_t1 @ u_vel
        w_loc = vec_t2 @ u_vel

        return np.array((u_loc, v_loc, w_loc))

    @staticmethod
    def _split_vec(val_vec):
        # print(val_vec)
        return val_vec[0], val_vec[1:4], val_vec[4]

    @staticmethod
    def _bd_symmetry(rho, u_vel, pressure, _):
        # ref_1 = np.array([-1, 1, 1], dtype=np.float64)
        # u_bd = u_vel * ref_1
        u_vel[0] = u_vel[0] * -1.0

        return rho, u_vel, pressure

    def _bd_wall(self, rho, u_vel, pressure, id_face):
        bc_list = self.mesh.boundary
        bd_id = self.mesh.get_bd_id(id_face)

        vel_wall_g = bc_list[bd_id].u_val
        if vel_wall_g is None:
            vel_wall = 0.0
        else:
            vel_wall = self._conv_vel(vel_wall_g, id_face, conv_type='G2L')
        # _, vel_wall, _ = self._split_vec(vec_wall)

        u_bd = 2.0 * vel_wall - u_vel

        return rho, u_bd, pressure
