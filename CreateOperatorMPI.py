import argparse

from mpi4py import MPI

from Functions.Mesh import OfMesh
from Functions.FieldData import OfData

from Functions.MatMaker import MatMaker
from Functions.LinearizedNS import LNS


def MakeOperator(case_dir, time, filename, mu, pr):
    comm = MPI.COMM_WORLD

    # case_dir = '/mnt/data/OpenFOAM/CylinderNoise/'
    # data_dir = '499.992868672869065/'
    # case_dir = 'Data/cavity_test/'
    # data_dir = '1/'

    # print('Initializing...')
    mesh = OfMesh(case_dir, time + 'C', time + 'V', time + 'U', time + 'p')
    ave_field = OfData(mesh, case_dir + time, 'UMean', 'pMean', 'rhoMean', add_e=True, add_temp=True)

    # mu = 1.333333e-3
    # pr = 0.7
    linear_ns = LNS(mesh, ave_field, mu=mu, pr=pr, is2d=True)  # viscosity and Prandtl number

    mat_maker = MatMaker(linear_ns, mesh.n_cell, ave_field=ave_field, mpi_comm=comm)
    mat_maker.make_mat()
    mat_maker.save_mat(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make a liner operator from OpenFOAM results. MPI version.')

    parser.add_argument('source_dir', help='Case directory for the simulation.')
    parser.add_argument('time', help='Time step of the results.')
    parser.add_argument('-f', '--filename', default='matL.npz', help='Save name for the operator.')
    parser.add_argument('--mu', type=float, default=1.0e-5, help='Viscosity.')  # Should be obtained from the case dir.
    parser.add_argument('--pr', type=float, default=0.7, help='Prandtl number.')

    args = parser.parse_args()

    MakeOperator(args.source_dir, args.time, args.filename, args.mu, args.pr)
