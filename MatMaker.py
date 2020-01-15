import time

import itertools
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from Variables import Variables
# from Functions.Identity import Identity


class MatMaker:
    def __init__(self, n_cell, n_val, target):
        self.n_cell = n_cell
        self.n_val = n_val
        self.n_size = self.n_cell * self.n_val

        # self.mesh = mesh
        # self.flow_data = flow_data

        self.variables = target
        # self._check_target()

        self.operator = self._set_matrix()

        self._ph = PlaceHolder(n_cell, n_val)

    def _check_target(self):
        if not issubclass(self.variables, Variables):
            raise TypeError

    def _set_matrix(self):
        return lil_matrix((self.n_size, self.n_size), dtype=np.float64)

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
            i_row = self._serializer(id_cell, id_val)
            i_col = self._serializer(ref_cell, ref_val)
            self.operator[i_row, i_col] = val

    def _serializer(self, id_cell, id_val):
        return id_cell * self.n_val + id_val

    def _calc_sub(self, id_cell):
        variables = self.variables
        ref_cells = variables.get_leaves(id_cell)
        # print(ref_cells)

        def iterator(i_val, r_cell, r_val):
            # ph = self._set_place_holder(r_cell, r_val)
            self._ph.set_ph(r_cell, r_val)
            val = variables.formula(self._ph, id_cell, i_val)
            return val

        val_list = []
        for id_val, ref_cell, ref_val in itertools.product(range(self.n_val), ref_cells, range(self.n_val)):
            val_list.append((id_val, ref_cell, ref_val, iterator(id_val, ref_cell, ref_val)))

        return val_list


class PlaceHolder:
    def __init__(self, n_cell, n_val):
        self.n_cell = n_cell
        self.n_val = n_val

        self.i_cell = 0
        self.i_val = 0

    def shape(self):
        return self.n_cell, self.n_val

    def set_ph(self, i_cell, i_val):
        self.i_cell = i_cell
        self.i_val = i_val

    def __getitem__(self, x):
        if len(x) != 2:
            raise TypeError

        # if isinstance(x, tuple):
        i_cell = x[0]
        i_val = x[1]

        if i_cell == self.i_cell and i_val == self.i_val:
            return 1.0
        else:
            return 0.0
        # elif isinstance(x, slice):
        #     pass
