from scipy import sparse

from Mesh import OfMesh
from FlowData import OfData

# from MatMaker import MatMaker
# from MatMaker import TargetEq

# from BoundaryCondition import BoundaryCondition as BDcond
# from Functions.Gradient import Gradient


def main():
    data_dir = 'Data/cavity/'

    mesh = OfMesh(data_dir, '0.5/C', '0.5/V', '0.5/U', '0.5/p')
    field = OfData(data_dir, '0.5/U', '0.5/p', '0.5/p')
    operator = sparse.load_npz('matO.npz')

    n_cell = field.n_cell
    n_val = field.n_val
    data = field.data.reshape(n_cell * n_val)

    grad_data = operator @ data
    w_data = grad_data.reshape(n_cell, n_val)
    field.data = w_data

    field.vis_tecplot(mesh, 'Grad_test.dat')


if __name__ == '__main__':
    main()
