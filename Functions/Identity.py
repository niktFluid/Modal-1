from Variables import Variables


class Identity(Variables):
    # Identity.py function
    def __init__(self, mesh, flow_data, sub_list=None):
        super(Identity, self).__init__(mesh, flow_data, sub_list)

    def return_ref_cells(self, id_cell):
        return [id_cell]

    def formula(self, ph, id_cell, id_val, **kwargs):
        return ph[id_cell, id_val]
