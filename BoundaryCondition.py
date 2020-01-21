import numpy as np


class BoundaryCondition:
    def __init__(self, mesh):
        self.mesh = mesh

        self._id_wall = -11
        self._id_moving_wall = -10
        self._id_empty = -12

    def get_bd_val(self, val_vec, id_face, id_bd):
        # n_val = len(val_vec)
        # rho, vel_vec, pressure = self._split_vec(val_vec)
        vec_loc = self._conv_vel(val_vec, id_face, conv_type='G2L')

        bd_func = self._select_bd_func(id_bd)
        bd_vec_loc = bd_func(vec_loc, id_face)

        # noinspection PyTypeChecker
        bd_vec = self._conv_vel(bd_vec_loc, id_face, conv_type='L2G')
        return bd_vec

    def _select_bd_func(self, id_bd):
        if id_bd == self._id_wall:
            return self._bd_wall
        elif id_bd == self._id_moving_wall:
            return self._bd_wall
        elif id_bd == self._id_empty:
            return self._bd_symmetry
        else:
            raise Exception

    def _conv_vel(self, val_vec, id_face, conv_type):
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

        u_vel = val_vec[1:4]
        val_vec[1] = vec_n @ u_vel
        val_vec[2] = vec_t1 @ u_vel
        val_vec[3] = vec_t2 @ u_vel

        return val_vec

    @staticmethod
    def _bd_symmetry(val_vec, _):
        val_vec[1] *= -1.0
        return val_vec

    def _bd_wall(self, val_vec, id_face):
        bc_list = self.mesh.boundary
        bd_id = self.mesh.get_bd_id(id_face)

        vel_wall_g = bc_list[bd_id].u_val
        if vel_wall_g is None:
            vel_wall = 0.0
        else:
            vel_wall = self._conv_vel(vel_wall_g, id_face, conv_type='G2L')

        val_vec[1:4] = 2.0 * vel_wall - val_vec[1:4]
        return val_vec
