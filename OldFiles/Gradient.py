import numpy as np
from scipy import linalg
# from itertools import combinations_with_replacement
from itertools import product


class GradientData:
    def __init__(self, mesh, field, bd_field):
        self.state_id = 0

        self.mesh = mesh
        self.field = field
        self.bd_field = bd_field

        # LS mat
        n_cell = mesh.n_cell
        self.matLU1 = np.zeros((n_cell, 3, 3), dtype=np.float64)
        self.matLU2 = np.zeros((n_cell, 3), dtype=np.float64)
        self._set_left_mat()

        # GradState
        n_var = self.field.n_val
        self.grad_state = np.empty((n_cell, n_var, 3), dtype=np.float64)
        self.limiter = np.full((n_cell, n_var), 2.0, dtype=np.float64)

    def get_grad(self):
        # This program estimate the gradient of variables by weighted least square method.
        if self.state_id != self.field.get_state_id():
            self.set_grad()
        return self.grad_state

    def _set_left_mat(self):
        n_cell = self.mesh.n_cell
        n_face = self.mesh.n_face

        matA = np.zeros((n_cell, 3, 3), dtype=np.float64)

        def add_func(l_ind, r_ind, cell_ind, x):
            matA[cell_ind, l_ind, r_ind] += x
        addA = np.frompyfunc(add_func, 4, 0)

        # Inner faces
        owner = self.mesh.owner
        neighbour = self.mesh.neighbour

        area = self.mesh.face_area[:n_face]
        dis_o = self.mesh.dis_fc_o
        dis_n = self.mesh.dis_fc_n
        dis_inv = self.mesh.dis_fc_inv

        vec_lr = self.mesh.vec_lr
        lr_inv = self.mesh.lr_inv
        # -----

        w_ls_o = (2.0 * dis_o * dis_inv)**2 * area * lr_inv
        w_ls_n = (2.0 * dis_n * dis_inv)**2 * area * lr_inv

        for i_ind, j_ind in product(range(3), range(3)):
            addA(i_ind, j_ind, owner, w_ls_o * vec_lr[:, i_ind] * vec_lr[:, j_ind])
            addA(i_ind, j_ind, neighbour, w_ls_n * vec_lr[:, i_ind] * vec_lr[:, j_ind])

        # Boundary faces
        bd_cells = self.mesh.bd_cells

        bc_info = self.mesh.boundary
        area_bd = [self.mesh.get_bd_geom(i_bd)[0] for i_bd in range(len(bc_info))]

        dis_bd = self.mesh.dis_fc_bd
        dis_inv_bd = self.mesh.dis_fc_bd_inv

        lr_bd = self.mesh.vec_lr_bd
        lr_inv_bd = self.mesh.lr_inv_bd
        # -----

        w_ls_o = [(2.0 * dist * inv)**2 * area * lr_inv
                  for dist, inv, area, lr_inv in zip(dis_bd, dis_inv_bd, area_bd, lr_inv_bd)]

        for i_bd in range(len(bc_info)):
            for i_ind, j_ind in product(range(3), range(3)):
                addA(i_ind, j_ind, bd_cells[i_bd],
                     w_ls_o[i_bd] * lr_bd[i_bd][:, i_ind] * lr_bd[i_bd][:, j_ind])

        # Calculate LU factor for computing gradient
        for i_cell in range(n_cell):
            self.matLU1[i_cell], self.matLU2[i_cell] = linalg.lu_factor(matA[i_cell])

    def set_grad(self):
        n_cell = self.mesh.n_cell
        n_face = self.mesh.n_face

        # Assume the shape of data matrix is set to be (n_cell, n_var)
        n_var = self.field.n_val
        data = self.field.get_data()
        data_bd = self.bd_field.get_bd()

        matB = np.zeros((n_cell, n_var, 3), dtype=np.float64)

        # Inner faces
        owner = self.mesh.owner
        neighbour = self.mesh.neighbour

        area = self.mesh.face_area[:n_face]
        dis_o = self.mesh.dis_fc_o
        dis_n = self.mesh.dis_fc_n
        dis_inv = self.mesh.dis_fc_inv

        vec_lr = self.mesh.vec_lr
        lr_inv = self.mesh.lr_inv
        # -----

        w_ls_o = (2.0 * dis_o * dis_inv)**2 * area * lr_inv
        w_ls_n = (2.0 * dis_n * dis_inv)**2 * area * lr_inv

        def add_func(var_ind, co_ind, o_ind, n_ind, x):
            matB[o_ind, var_ind, co_ind] += (data[n_ind, var_ind] - data[o_ind, var_ind]) * x
        addB = np.frompyfunc(add_func, 5, 0)

        for v_ind, c_ind in product(range(n_var), range(3)):
            addB(v_ind, c_ind, owner, neighbour, w_ls_o * vec_lr[:, c_ind])
            addB(v_ind, c_ind, owner, neighbour, w_ls_n * vec_lr[:, c_ind])

        # Boundary faces
        bd_nums = self.mesh.bd_nums
        bd_cells = self.mesh.bd_cells

        bc_info = self.mesh.boundary
        area_bd = [self.mesh.get_bd_geom(i_bd)[0] for i_bd in range(len(bc_info))]

        dis_bd = self.mesh.dis_fc_bd
        dis_inv_bd = self.mesh.dis_fc_bd_inv

        lr_bd = self.mesh.vec_lr_bd
        lr_inv_bd = self.mesh.lr_inv_bd
        # -----

        def add_func(id_bd, var_ind, co_ind, o_ind, n_ind, x):
            matB[o_ind, var_ind, co_ind] += (data_bd[id_bd][n_ind, var_ind] - data[o_ind, var_ind]) * x
        addB = np.frompyfunc(add_func, 6, 0)

        w_ls_o = [(2.0 * dist * inv)**2 * area * lr_inv
                  for dist, inv, area, lr_inv in zip(dis_bd, dis_inv_bd, area_bd, lr_inv_bd)]

        for i_bd in range(len(bc_info)):
            for v_ind, c_ind in product(range(n_var), range(3)):
                addB(i_bd, v_ind, c_ind, bd_cells[i_bd], np.array(range(bd_nums[i_bd])),
                     w_ls_o[i_bd] * lr_bd[i_bd][:, c_ind])

        # Should this "for loop" be vectorized (use np.frompyfunc) to calculate faster?
        for i_cell, i_var in product(range(n_cell), range(n_var)):
            self.grad_state[i_cell, i_var] = linalg.lu_solve((self.matLU1[i_cell], self.matLU2[i_cell]),
                                                             matB[i_cell][i_var])

        self.set_limiter()
        self.state_id = self.field.get_state_id()

    def set_limiter(self):
        # Hishida limiter (van Albada like)
        epsilon = 1.0e-9
        # Inner faces
        owner = self.mesh.owner
        neighbour = self.mesh.neighbour

        # area = self.mesh.face_area[:n_face]
        dis_o = self.mesh.dis_fc_o
        dis_n = self.mesh.dis_fc_n
        dis_inv = self.mesh.dis_fc_inv

        vec_lr = self.mesh.vec_lr
        lr_inv = self.mesh.lr_inv
        # -----
