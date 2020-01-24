import numpy as np


class BoundaryCondition:
    def __init__(self, mesh):
        self.mesh = mesh

        self.id_wall = -11
        self.id_moving_wall = -10
        self.id_empty = -12

    def get_bd_val(self, val_vec, id_face, id_bd):
        # n_val = len(val_vec)
        # rho, vel_vec, pressure = self._split_vec(val_vec)
        vec_loc = self.mesh.conv_vel(val_vec, id_face)

        bd_func = self._select_bd_func(id_bd)
        bd_vec_loc = bd_func(vec_loc, id_face)

        # noinspection PyTypeChecker
        return self.mesh.conv_vel(bd_vec_loc, id_face, inverse=True)

    def _select_bd_func(self, id_bd):
        if id_bd == self.id_wall:
            return self._bd_wall
        elif id_bd == self.id_moving_wall:
            return self._bd_wall
        elif id_bd == self.id_empty:
            return self._bd_symmetry
        else:
            raise Exception

    @staticmethod
    def _bd_symmetry(val_vec, _):
        val_vec[1] *= -1.0
        return val_vec

    def _bd_wall(self, val_vec, id_face):
        bc_list = self.mesh.boundary
        bd_id = self.mesh.get_bd_id(id_face)

        vel_wall_g = bc_list[bd_id].u_val
        if vel_wall_g is None or not np.any(vel_wall_g):
            vel_wall = 0.0
        else:
            vel = np.zeros(5, dtype=np.float64)
            vel[1:4] = vel_wall_g
            vel_loc = self.mesh.conv_vel(vel, id_face)
            vel_wall = vel_loc[1:4]

        val_vec[1:4] = 2.0 * vel_wall - val_vec[1:4]
        return val_vec
