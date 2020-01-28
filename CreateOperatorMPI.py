import argparse

from mpi4py import MPI

from scipy import sparse

from Functions.Mesh import OfMesh
from Functions.FieldData import OfData

from Functions.MatMaker import MatMaker
from Functions.LinearizedNS import LNS


def main(case_dir, time, filename, mu, pr):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # case_dir = '/mnt/data/OpenFOAM/CylinderNoise/'
    # data_dir = '499.992868672869065/'
    # case_dir = 'Data/cavity_test/'
    # data_dir = '1/'
    mesh = OfMesh(case_dir, time + 'C', time + 'V', time + 'U', time + 'p')
    ave_field = OfData(mesh, case_dir + time, 'UMean', 'pMean', 'rhoMean', add_e=True, add_temp=True)

    # mu = 1.333333e-3
    # pr = 0.7
    linear_ns = LNS(mesh, ave_field, mu=mu, pr=pr, is2d=True)  # viscosity and Prandtl number

    mat_maker = MatMaker(linear_ns, mesh.n_cell, ave_field=ave_field, mpi_comm=comm)
    operator = mat_maker.get_mat()

    if rank == 0:
        sparse.save_npz(filename, operator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make a liner operator from OpenFOAM results. MPI version.')

    parser.add_argument('source_dir', help='Case directory for the simulation.')
    parser.add_argument('time', help='Time step of the results.')
    parser.add_argument('-f', '--filename', default='matL.npz', help='Save name for the operator.')
    parser.add_argument('--mu', type=float, default=1.0e-5, help='Viscosity.')
    parser.add_argument('--pr', type=float, default=0.7, help='Prandtl number.')

    args = parser.parse_args()

    main(args.source_dir, args.time, args.filename, args.mu, args.pr)
