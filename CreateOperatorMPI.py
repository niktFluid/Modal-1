from mpi4py import MPI

from scipy import sparse

from Mesh import OfMesh
from FlowData import OfData

from MatMaker import MatMaker
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
    mesh = OfMesh(case_dir, data_dir + 'C', data_dir + 'V', data_dir + 'U', data_dir + 'p')
    ave_field = OfData(case_dir + data_dir, 'UMean', 'pMean', 'rhoMean', add_e=True, add_temp=True)

    linear_ns = LNS(mesh, ave_field, 1.84e-5, 0.7, is2d=True)  # viscosity and Prandtl number

    mat_maker = MatMaker(linear_ns, mesh.n_cell, ave_field.n_val, ave_field, mpi_comm=comm)
    operator = mat_maker.get_mat()

    if rank == 0:
        sparse.save_npz('matO.npz', operator)


if __name__ == '__main__':
    main()
