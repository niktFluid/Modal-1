from scipy import sparse

from Mesh import OfMesh
from FlowData import OfData

from MatMaker import MatMaker
# from MatMaker import TargetEq

# from BoundaryCondition import BoundaryCondition as BDcond
# from Functions.Gradient import Gradient
from Functions.LinearizedNS import LNS


def main():
    data_dir = 'Data/cavity_test/'

    mesh = OfMesh(data_dir, '1/C', '1/V', '1/U', '1/p')
    ave_field = OfData(data_dir, '1/UMean', '1/pMean', '1/rhoMean', add_e=True, add_temp=True)
    linear_ns = LNS(mesh, ave_field, 1.84e-5, 0.7)  # viscosity and Prandtl number

    mat_maker = MatMaker(linear_ns, mesh.n_cell, ave_field.n_val, ave_field)
    operator = mat_maker.get_mat()
    sparse.save_npz('matO.npz', operator)


if __name__ == '__main__':
    main()
