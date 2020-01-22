import numpy as np
import Ofpp
from collections import namedtuple


class Mesh:
    def __init__(self):
        dummy = np.empty(0)
        # Grid sizes
        self.n_node = 0
        self.n_face = 0
        self.n_bdface = 0
        self.n_cell = 0

        # Basic geometry
        self.nodes = dummy
        self.face_nodes = dummy
        self.cell_faces = dummy
        self.cell_neighbour = dummy
        self.owner = dummy
        self.neighbour = dummy

        # Cell geometry
        self.centers = dummy
        self.volumes = dummy

        # Face geometry
        self.face_area = dummy
        self.face_centers = dummy

        # Vectors for global -> local coordinate transformation
        self.face_vec_n = dummy
        self.face_vec_t1 = dummy
        self.face_vec_t2 = dummy

        # Vectors for local -> global coordinate transformation
        self.face_vec_ni = dummy
        self.face_vec_t1i = dummy
        self.face_vec_t2i = dummy

        # Vectors: Cell center -> Face center
        self.fc_o = dummy  # Owner side
        self.fc_n = dummy  # Neighbour side
        self.dis_fc_o = dummy
        self.dis_fc_n = dummy
        self.dis_fc_inv = dummy

        self.fc_bd = dummy
        self.dis_fc_bd = dummy
        self.dis_fc_bd_inv = dummy

        # Vectors: Owner cell -> neighbour cell
        self.vec_lr = dummy
        self.lr_inv = dummy

        self.vec_lr_bd = dummy
        self.lr_inv_bd = dummy

        # BC information
        self._bd_tuple = namedtuple('Boundary', ['name', 'type', 'id', 'i_start', 'num', 'u_val', 'p_val'])
        self.boundary = [self._bd_tuple(None, None, None, None, None, None, None)]
        self.bd_nums = [0]
        self.bd_faces = [dummy]
        self.bd_cells = [dummy]

        # Initialization
        self._init_mesh()

    def _init_mesh(self):
        raise NotImplementedError

    def cell_neighbours(self, id_cell):
        return [self.owner[x] + self.neighbour[x] - id_cell for x in self.cell_faces[id_cell]]

    def get_bd_cond(self):
        bd = self.boundary
        bd_types = [bd[i].type for i in range(len(bd))]
        bd_u = [bd[i].u_val for i in range(len(bd))]
        bd_p = [bd[i].p_val for i in range(len(bd))]

        return bd_types, bd_u, bd_p

    def get_bd_geom(self, i_bd, inv=False):
        area = self.face_area[self.bd_faces[i_bd]]

        if not inv:
            vec_n = self.face_vec_n[self.bd_faces[i_bd]]
            vec_t1 = self.face_vec_t1[self.bd_faces[i_bd]]
            vec_t2 = self.face_vec_t2[self.bd_faces[i_bd]]
        else:
            vec_n = self.face_vec_ni[self.bd_faces[i_bd]]
            vec_t1 = self.face_vec_t1i[self.bd_faces[i_bd]]
            vec_t2 = self.face_vec_t2i[self.bd_faces[i_bd]]

        return area, vec_n, vec_t1, vec_t2

    def get_bdfc_vec(self):
        return [self.fc_o[ind] for ind in self.bd_faces]

    def get_bd_id(self, id_face):
        id_bd = None

        for i_bd, bd_data in enumerate(self.boundary):
            i_start = bd_data.i_start
            num = bd_data.num

            if i_start <= id_face < i_start + num:
                id_bd = i_bd
                break

        return id_bd

    def conv_vel(self, val_vec, id_face, conv_type):
        if conv_type == 'G2L':
            vec_n = self.face_vec_n[id_face]
            vec_t1 = self.face_vec_t1[id_face]
            vec_t2 = self.face_vec_t2[id_face]
        elif conv_type == 'L2G':
            vec_n = self.face_vec_ni[id_face]
            vec_t1 = self.face_vec_t1i[id_face]
            vec_t2 = self.face_vec_t2i[id_face]
        else:
            raise TypeError

        u_vel = val_vec[1:4]
        val_vec[1] = vec_n @ u_vel
        val_vec[2] = vec_t1 @ u_vel
        val_vec[3] = vec_t2 @ u_vel

        return val_vec


class OfMesh(Mesh):
    def __init__(self, path_dir, path_centres, path_vols, path_bd_u, path_bd_p):
        self.path_dir = path_dir
        self.path_centres = path_centres
        self.path_vols = path_vols
        self.path_bd_u = path_bd_u
        self.path_bd_p = path_bd_p

        super(OfMesh, self).__init__()

    def _init_mesh(self):
        mesh = Ofpp.FoamMesh(self.path_dir)
        mesh.read_cell_centres(self.path_dir + self.path_centres)
        mesh.read_cell_volumes(self.path_dir + self.path_vols)

        self.nodes = mesh.points
        self.face_nodes = mesh.faces
        self.cell_faces = mesh.cell_faces

        self.n_node = len(self.nodes)
        self.n_face = mesh.num_inner_face
        self.n_bdface = mesh.num_face - mesh.num_inner_face
        self.n_cell = len(self.cell_faces)

        self.owner = mesh.owner
        self.neighbour = mesh.neighbour

        self.centers = mesh.cell_centres
        self.volumes = mesh.cell_volumes

        self._set_boundary(mesh)
        self._calc_face_vec()
        self._calc_face_centers()
        self._calc_vec_lr()

    def _calc_face_vec(self):
        face_nodes = np.array(self.face_nodes)
        ind_a = np.vstack((face_nodes[:, 2], face_nodes[:, 0]))
        ind_b = np.vstack((face_nodes[:, 3], face_nodes[:, 1]))

        vec_a = np.squeeze(np.diff(self.nodes[ind_a], axis=0))
        vec_b = np.squeeze(np.diff(self.nodes[ind_b], axis=0))
        vec_c = np.cross(vec_a, vec_b)
        self.face_area = np.linalg.norm(vec_c, axis=1)

        def normalize(vec):
            l2 = np.linalg.norm(vec, axis=1, keepdims=True)
            return vec/l2
        self.face_vec_n = normalize(vec_c)
        self.face_vec_t1 = normalize(vec_a)
        self.face_vec_t2 = np.cross(self.face_vec_n, self.face_vec_t1)

        def vec_trans(v_ind):
            return np.vstack((self.face_vec_n[:, v_ind],
                              self.face_vec_t1[:, v_ind],
                              self.face_vec_t2[:, v_ind])).T
        self.face_vec_ni = normalize(vec_trans(0))
        self.face_vec_t1i = normalize(vec_trans(1))
        self.face_vec_t2i = normalize(vec_trans(2))

    def _calc_face_centers(self):
        points = self.nodes[self.face_nodes]
        self.face_centers = np.mean(points, axis=1)

        # Inner faces
        self.fc_o = self.face_centers - self.centers[self.owner]
        self.fc_n = self.face_centers[:self.n_face] - self.centers[self.neighbour[:self.n_face]]

        self.dis_fc_o = np.linalg.norm(self.fc_o, axis=1)
        self.dis_fc_n = np.linalg.norm(self.fc_n, axis=1)

        self.dis_fc_inv = 1.0 / (self.dis_fc_o[:self.n_face] + self.dis_fc_n)

        # Boundary faces
        self.fc_bd = [self.face_centers[ind_f] - self.centers[ind_c]
                      for ind_f, ind_c in zip(self.bd_faces, self.bd_cells)]
        self.dis_fc_bd = [np.linalg.norm(vector, axis=1) for vector in self.fc_bd]
        self.dis_fc_bd_inv = [1.0/(2.0 * dist) for dist in self.dis_fc_bd]

    def _calc_vec_lr(self):
        self.vec_lr = self.centers[self.neighbour[:self.n_face]] - self.centers[self.owner[:self.n_face]]
        self.lr_inv = 1.0 / np.linalg.norm(self.vec_lr, axis=1)

        vec_n_bd = [self.get_bd_geom(i_bd)[1] for i_bd in range(len(self.boundary))]
        self.vec_lr_bd = [2.0 * vec_n * np.tile(dist, (3, 1)).T
                          for vec_n, dist in zip(vec_n_bd, self.dis_fc_bd)]
        self.lr_inv_bd = [1.0 / np.linalg.norm(vec, axis=1) for vec in self.vec_lr_bd]

    def _set_boundary(self, mesh):
        bd_u = Ofpp.parse_boundary_field(self.path_dir + self.path_bd_u)
        bd_p = Ofpp.parse_boundary_field(self.path_dir + self.path_bd_p)
        # mesh = Ofpp.FoamMesh(self.path_dir)

        def make_bd_tuple(bd_key):
            source_dic = mesh.boundary[bd_key]

            if b'value' in bd_u[bd_key]:
                u_val = bd_u[bd_key][b'value']
            else:
                u_val = None

            if b'value' in bd_p[bd_key]:
                p_val = bd_p[bd_key][b'value']
            else:
                p_val = None

            bd_tuple = self._bd_tuple(bd_key.decode(),
                                      source_dic.type.decode(),
                                      source_dic.id,
                                      source_dic.start,
                                      source_dic.num,
                                      u_val, p_val)
            return bd_tuple

        bd = [make_bd_tuple(key) for key in mesh.boundary.keys()]
        owner = np.array(mesh.owner)

        self.boundary = bd
        self.bd_nums = [bd[i].num for i in range(len(bd))]
        self.bd_faces = [np.arange(bd[i].i_start, bd[i].i_start + bd[i].num) for i in range(len(bd))]
        self.bd_cells = [owner[self.bd_faces[i]] for i in range(len(bd))]
