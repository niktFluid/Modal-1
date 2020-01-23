import time

import itertools
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from Variables import Variables
# from Functions.Identity import Identity


class MatMaker:
    def __init__(self, target, n_cell, n_val, ave_field=None):
        self.n_cell = n_cell
        self.n_val = n_val
        self.n_size_in = n_cell * n_val

        # self.mesh = mesh
        # self.flow_data = flow_data

        self._variables = target
        # self._check_target()
        self.n_return = self._variables.n_return
        self.n_size_out = n_cell * self.n_return

        self.operator = lil_matrix((self.n_size_out, self.n_size_in), dtype=np.float64)

        if ave_field is None:
            n_val_ph = 5  # rho, u, v, w, pressure
        else:
            n_val_ph = 7  # add energy and temperature
        self._ph = PlaceHolder(n_cell, n_val_ph, ave_field)

    def _check_target(self):
        if not issubclass(self._variables, Variables):
            raise TypeError

    def get_mat(self):
        t_start = time.time()

        # for id_cell in range(5):
        for id_cell in range(self.n_cell):
            self._set_mat_for_cell(id_cell)
            self._print_progress(id_cell, t_start)

        t_end = time.time() - t_start
        print('Done. Elapsed time: {:.0f}'.format(t_end))
        return csr_matrix(self.operator)

    def _print_progress(self, id_cell, t_start):
        interval = 10

        prog_0 = int(100.0 * (id_cell - 1) / self.n_cell)
        prog_1 = int(100.0 * id_cell / self.n_cell)

        if prog_0 % interval == interval - 1 and prog_1 % interval == 0:
            t_elapse = time.time() - t_start
            print(str(prog_1) + ' %, Elapsed time: {:.0f}'.format(t_elapse) + ' [sec]')

    def _set_mat_for_cell(self, id_cell):
        val_list = self._calc_sub(id_cell)
        # print(val_list)

        for id_val, ref_cell, ref_val, val in val_list:
            i_row = id_cell * self.n_return + id_val
            i_col = ref_cell * self.n_val + ref_val
            self.operator[i_row, i_col] = val

    def _calc_sub(self, id_cell):
        variables = self._variables
        n_func_val = variables.n_return
        ref_cells = variables.get_leaves(id_cell)
        # print(ref_cells)

        val_list = []
        for ref_cell, ref_val in itertools.product(ref_cells, range(self.n_val)):
            self._ph.set_ph(ref_cell, ref_val)
            func_val = variables.formula(self._ph, id_cell)
            for id_val in range(n_func_val):
                val_list.append((id_val, ref_cell, ref_val, func_val[id_val]))
        return val_list


class PlaceHolder:
    def __init__(self, n_cell, n_val, ave_field=None):
        self._gamma = 1.4
        self._gamma_2 = 1.0 / (1.4 - 1.0)

        self.n_cell = n_cell
        self.n_val = n_val
        self.shape = (self.n_cell, self.n_val)

        self.i_cell = -1
        self.i_val = -1

        self._ave_field = ave_field

        if n_val > 5 and ave_field is None:
            # For calculating energy and temperature
            raise Exception

    def set_ph(self, i_cell, i_val):
        self.i_cell = i_cell
        self.i_val = i_val

    def __getitem__(self, x):
        i_cell = x[0]
        i_val = x[1]

        if 0 <= i_val < 5:
            # for Rho, u vel, v vel, w vel, pressure
            return float(i_cell == self.i_cell and i_val == self.i_val)
        elif i_val == 5:
            # for energy
            return self._calc_energy(i_cell)
        elif i_val == 6:
            # for temperature
            return self._calc_temperature(i_cell)
        else:
            raise Exception

    def _calc_energy(self, i_cell):
        # for energy variation
        g2 = self._gamma_2

        rho = self[i_cell, 0]
        u = self[i_cell, 1]
        v = self[i_cell, 2]
        w = self[i_cell, 3]
        p = self[i_cell, 4]

        ave = self._ave_field.data
        rho_ave = ave[i_cell, 0]
        u_ave = ave[i_cell, 1]
        v_ave = ave[i_cell, 2]
        w_ave = ave[i_cell, 3]

        term_1 = g2 * p
        term_2 = 0.5 * rho * (u_ave * u_ave + v_ave * v_ave + w_ave * w_ave)
        term_3 = rho_ave * (u * u_ave + v * v_ave + w * w_ave)

        return term_1 + term_2 + term_3

    def _calc_temperature(self, i_cell):
        # for temperature variation
        g1 = self._gamma

        rho = self[i_cell, 0]
        p = self[i_cell, 4]

        ave = self._ave_field.data
        rho_ave = ave[i_cell, 0]
        p_ave = ave[i_cell, 4]

        term_1 = g1 * p / rho_ave
        term_2 = g1 * p_ave * rho / (rho_ave * rho_ave)

        return term_1 - term_2
