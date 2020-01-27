from mpi4py import MPI

from scipy import sparse

from Functions.Mesh import OfMesh
from Functions.FieldData import OfData

from Functions.MatMaker import MatMaker
# from MatMaker import TargetEq

# from BoundaryCondition import BoundaryCondition as BDcond
# from Functions.Gradient import Gradient
from Functions.LinearizedNS import LNS


def main():
    comm = MPI.COMM_WORLD
    # size = comm.Get_size()
    rank = comm.Get_rank()

    case_dir = '/mnt/data/OpenFOAM/CylinderNoise/'
    data_dir = '499.992868672869065/'
    # case_dir = 'Data/cavity_test/'
    # data_dir = '1/'
    mesh = OfMesh(case_dir, data_dir + 'C', data_dir + 'V', data_dir + 'U', data_dir + 'p')
    ave_field = OfData(mesh, case_dir + data_dir, 'UMean', 'pMean', 'rhoMean', add_e=True, add_temp=True)

    linear_ns = LNS(mesh, ave_field, mu=1.333333e-3, pr=0.7, is2d=True)  # viscosity and Prandtl number

    mat_maker = MatMaker(linear_ns, mesh.n_cell, ave_field=ave_field, mpi_comm=comm)
    operator = mat_maker.get_mat()

    if rank == 0:
        sparse.save_npz('matL_Cylinder-0.npz', operator)


if __name__ == '__main__':
    main()
