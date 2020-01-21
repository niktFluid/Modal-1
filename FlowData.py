import numpy as np
import Ofpp


class FlowField:
    def __init__(self, add_e=False, add_T=False):
        dummy = np.empty(0)

        self.n_val = 5  # density, velocity, pressure
        self.n_cell = 0
        self.data = dummy
        self.sol_time = 0.0

        self._init_field()
        if add_e:
            self._add_energy()
        if add_T:
            self._add_temperature()

    def _init_field(self, *args, **kwargs):
        raise NotImplementedError

    def update_from_file(self, *args, **kwargs):
        raise NotImplementedError

    def get(self, id_cell, id_val):
        return self.data[id_cell, id_val]

    def get_interface_data(self, bd_cells):
        return [self.data[ind] for ind in bd_cells]

    def _add_energy(self):
        self.n_val += 1

        data = self.data
        rho = data[:, 0]
        u = data[:, 1]
        v = data[:, 2]
        w = data[:, 3]
        p = data[:, 4]

        g2 = 1.0 / (1.4 - 1.0)
        e = g2 * p + 0.5 * rho * (u * u + v * v + w * w)

        np.vstack((self.data, e))

    def _add_temperature(self):
        self.n_val += 1

        data = self.data
        rho = data[:, 0]
        p = data[:, 4]

        temp = 1.4 * p / rho

        np.vstack((self.data, temp))

    # noinspection PyTypeChecker
    def vis_tecplot(self, mesh, file_name='Default.dat'):
        header = self._make_tec_header(mesh)
        nodes = mesh.nodes
        data = self.data

        with open(file_name, mode='w') as file_obj:
            file_obj.write(header)

            def conv_str(x):
                return str(x) + ' '

            def write_data(w_data):
                for w_list in self._make_write_list(w_data):
                    file_obj.writelines(list(map(conv_str, w_list)) + ['\n'])

            for ind in range(3):
                write_data(nodes[:, ind])

            for ind in range(5):
                write_data(data[:, ind])

            for connectivity in self._make_connectivity(mesh):
                c_list = list(map(conv_str, connectivity))
                file_obj.writelines(c_list + ['\n'])

    def _make_tec_header(self, mesh):
        header = 'Title = "Data" \n'
        header += 'Variables = "x", "y", "z", "density", "u_vel", "v_vel", "w_vel", "pressure" \n'
        header += 'Zone T = "Fluid area" \n'
        # header += 'StrandID=1, SolutionTime=' + str(self.sol_time) + ' \n'
        header += 'Nodes=' + str(mesh.n_node) + ' \n'
        header += 'Elements=' + str(mesh.n_cell) + ' \n'
        header += 'DATAPACKING=BLOCK \nZONETYPE=FEBRICK \n'
        header += 'VARLOCATION=([4-' + str(self.n_val + 3) + ']=CELLCENTERED) \n'
        return header

    @staticmethod
    def _make_write_list(data, n_line=8):
        n_data = data.size
        w_data = data.reshape(n_data)

        n_row = int(n_data / n_line)
        for i_row in range(n_row):
            i_start = n_line * i_row
            i_end = n_line * (i_row + 1)
            row_data = w_data[i_start:i_end]
            yield row_data

        # noinspection PyUnboundLocalVariable
        last_data = w_data[i_end:]
        if not len(last_data) == 0:
            yield last_data

    def _make_connectivity(self, mesh):
        raise NotImplementedError


class OfData(FlowField):
    def __init__(self, path_dir, path_u, path_p, path_rho):
        self.path_dir = path_dir
        self.path_u = path_u
        self.path_p = path_p
        self.path_rho = path_rho

        super(OfData, self).__init__()

    def _init_field(self):
        self.update_from_file()

    def update_from_file(self, path_u=None, path_p=None, path_rho=None):
        if path_u is None:
            path_u = self.path_u

        if path_p is None:
            path_p = self.path_p

        if path_rho is None:
            path_rho = self.path_rho

        u_data = Ofpp.parse_internal_field(self.path_dir + path_u)
        p_data = Ofpp.parse_internal_field(self.path_dir + path_p)
        rho_data = Ofpp.parse_internal_field(self.path_dir + path_rho)

        self.n_cell = u_data.shape[0]

        self.data = np.hstack((rho_data[:, np.newaxis], u_data, p_data[:, np.newaxis]))

    def _make_connectivity(self, mesh):
        # We should arrange this subroutine for clarify.
        for i_cell in range(self.n_cell):
            faces = mesh.cell_faces[i_cell]
            nodes = set(sum([mesh.face_nodes[x] for x in faces], []))

            id_0 = faces[0]
            fn_0 = set(mesh.face_nodes[id_0])
            fn_o = nodes - fn_0

            def find_opposite(face_list, fn_set):
                for i_face, i_fn in enumerate(face_list):
                    i_fn_set = set(mesh.face_nodes[i_fn])
                    if fn_set == i_fn_set:
                        return i_fn
            id_o = find_opposite(faces, fn_o)

            def find_connection(id_base, id_opposite):
                nodes_0 = mesh.face_nodes[id_base]
                side_0 = set(nodes_0[0:2])

                i_nb = -1
                for i_face in set(faces) - {id_base, id_opposite}:
                    node_set = set(mesh.face_nodes[i_face])
                    if len(side_0 & node_set) == 2:
                        i_nb = i_face
                nodes_nb = mesh.face_nodes[i_nb]

                nb_0 = nodes_nb.index(nodes_0[0])
                i_corner = nodes_nb[nb_0 - 2]
                last_node = set(nodes_nb) - side_0 - {i_corner}
                i_next_0 = last_node.pop()
                side_opposite = [i_next_0, i_corner]

                nodes_op = mesh.face_nodes[id_opposite]
                for i_rotation in range(3):
                    for i_shift in range(len(nodes_op)):
                        shifted_list = nodes_op[-i_shift:] + nodes_op[:-i_shift]
                        if shifted_list[0:2] == side_opposite:
                            return shifted_list
                    nodes_op.reverse()

            face_nodes_0 = mesh.face_nodes[id_0]
            face_nodes_op = find_connection(id_0, id_o)
            # face_nodes_op = mesh.face_nodes[id_0]

            def add_1(x):
                return x + 1
            yield list(map(add_1, face_nodes_0)) + list(map(add_1, face_nodes_op))
