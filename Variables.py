import itertools


class Variables:
    def __init__(self, mesh, flow_data, sub_list=None):
        self.mesh = mesh
        self.flow_data = flow_data

        self.leave_list = None

        self._sub_list = sub_list

        # if self._sub_list is not None:
        #     self._check_sub_list()

    # def _check_sub_list(self):
    #     for sub_variable in self._sub_list:
    #         if not issubclass(sub_variable, Variables):
    #             raise TypeError

    def get_leaves(self, id_cell):
        my_ref_cells = self.return_ref_cells(id_cell)

        sub_ref_cells = []
        if self._sub_list is not None:
            for sub_eqs, ref_cell in itertools.product(self._sub_list, my_ref_cells):
                sub_ref_cells += sub_eqs.get_leaves(ref_cell)

        self.leave_list = list(set(my_ref_cells + sub_ref_cells))

        return self.leave_list

    def formula(self, ph, id_cell, id_val):
        raise NotImplementedError

    def return_ref_cells(self, id_cell):
        raise NotImplementedError

    def _boundary_basic(self):
        pass


class Identity(Variables):
    def __init__(self, mesh, flow_data, sub_list=None):
        super(Identity, self).__init__(mesh, flow_data, sub_list)

    def return_ref_cells(self, id_cell):
        return [id_cell]

    def formula(self, ph, id_cell, id_val):
        return ph[id_cell, id_val]


class Gradient(Variables):
    def __init__(self, mesh, flow_data, sub_list=None):
        super(Gradient, self).__init__(mesh, flow_data, sub_list)

        self._nb_cells = None
        self._faces = None

    def return_ref_cells(self, id_cell):
        self._nb_cells = [id_cell] + self.mesh.cell_neighbours(id_cell)
        ref_cells = [i_cell for i_cell in self._nb_cells if i_cell >= 0]

        return list(set(ref_cells))

    def formula(self, ph, id_cell, id_val):
        self._faces = self.mesh.cell_faces[id_cell]

        grad = 0.0

        return grad

    def _set_mat(self, id_cell):
        pass

    def _set_rhs(self):
        pass

    def _get_val_diff(self):
        pass

    def _get_pos_diff(self):
        pass

    def _bd_val_diff(self):
        pass

    def _bd_pos_diff(self):
        pass
