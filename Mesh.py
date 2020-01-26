import numpy as np
# import copy
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

        # Matrix for global -> local coordinate transformation
        self.face_mat = dummy

        # Matrix for local -> global coordinate transformation
        # self.face_mat_i = dummy

        # Vectors: Owner cell -> neighbour cell
        self.vec_lr = dummy

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

    def get_bd_tuple(self, id_face):
        for bd_data in self.boundary:
            if bd_data.i_start <= id_face < bd_data.i_start + bd_data.num:
                return bd_data
        return None

    def conv_vel(self, val_vec, id_face, inverse=False):
        face_mat = self.face_mat[id_face]
        if inverse:
            face_mat = face_mat.T

        val_vec[1:4] = face_mat @ val_vec[1:4]
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
        # self.n_face = mesh.num_inner_face
        self.n_face = mesh.num_face
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
        nodes = self.nodes

        self.face_area = np.empty(self.n_face, dtype=np.float64)
        self.face_mat = np.empty((self.n_face, 3, 3), dtype=np.float64)

        def normalize(vec):
            return vec/np.linalg.norm(vec)

        for i_face in range(self.n_face):
            face_nodes = np.array(self.face_nodes[i_face])

            vec_a = nodes[face_nodes[2]] - nodes[face_nodes[0]]
            vec_b = nodes[face_nodes[3]] - nodes[face_nodes[1]]
            vec_c = np.cross(vec_a, vec_b)
            self.face_area[i_face] = np.linalg.norm(vec_c)

            vec_n = normalize(vec_c)  # Surface normal vector
            vec_t1 = normalize(vec_a)  # Surface tangential vector 1
            vec_t2 = np.cross(vec_n, vec_t1)

            self.face_mat[i_face, 0] = vec_n
            self.face_mat[i_face, 1] = vec_t1
            self.face_mat[i_face, 2] = vec_t2

        # face_nodes = np.array(self.face_nodes)
        # ind_a = np.vstack((face_nodes[:, 2], face_nodes[:, 0]))
        # ind_b = np.vstack((face_nodes[:, 3], face_nodes[:, 1]))
        #
        # print(ind_a)
        # vec_a = np.squeeze(np.diff(self.nodes[ind_a], axis=0))
        # vec_b = np.squeeze(np.diff(self.nodes[ind_b], axis=0))
        # vec_c = np.cross(vec_a, vec_b)
        # self.face_area = np.linalg.norm(vec_c, axis=1)
        #
        # def normalize(vec):
        #     l2 = np.linalg.norm(vec, axis=1, keepdims=True)
        #     return vec/l2
        # face_vec_n = normalize(vec_c)
        # face_vec_t1 = normalize(vec_a)
        # face_vec_t2 = np.cross(face_vec_n, face_vec_t1)
        #
        # def vec_trans(v_ind):
        #     return np.vstack((self.face_vec_n[:, v_ind],
        #                       self.face_vec_t1[:, v_ind],
        #                       self.face_vec_t2[:, v_ind])).T
        # self.face_vec_ni = normalize(vec_trans(0))
        # self.face_vec_t1i = normalize(vec_trans(1))
        # self.face_vec_t2i = normalize(vec_trans(2))

    def _calc_face_centers(self):
        points = self.nodes[self.face_nodes]
        self.face_centers = np.mean(points, axis=1)

    def _calc_vec_lr(self):
        self.vec_lr = np.zeros((self.n_face, 3), dtype=np.float64)
        centers = self.centers

        for i_face in range(self.n_face):
            id_o = self.owner[i_face]
            id_n = self.neighbour[i_face]

            if id_n >= 0:  # For inner faces
                self.vec_lr[i_face] = centers[id_n] - centers[id_o]
            else:  # For boundary faces
                face_vec_n = self.face_mat[i_face, 0]
                face_centers = self.face_centers

                dist_fc = np.linalg.norm(face_centers[i_face] - centers[id_o])
                self.vec_lr[i_face] = 2.0 * dist_fc * face_vec_n

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
