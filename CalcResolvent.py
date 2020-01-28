import argparse

from Functions.Mesh import OfMesh
from Functions.ModalAnalysis import ResolventMode as Resolvent


def CalcResolvent(case_dir, time, operator_name, omega, k):
    # case_dir = '/mnt/data/OpenFOAM/CylinderNoise/'
    # data_dir = '499.992868672869065/'
    # omega = 0.0
    # k = 6

    mesh = OfMesh(case_dir, time + 'C', time + 'V', time + 'U', time + 'p')
    resolvent_mode = Resolvent(mesh, operator_name, omega=omega, k=k)
    resolvent_mode.solve()
    resolvent_mode.save_data('resolvent_modes_0.pickle')
    resolvent_mode.vis_tecplot('resolvent_modes_0.dat')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make a liner operator from OpenFOAM results. MPI version.')

    parser.add_argument('source_dir', help='Case directory for the simulation.')
    parser.add_argument('time', help='Time step of the results.')
    parser.add_argument('-f', '--filename', default='matL.npz', help='File name for the operator.')
    parser.add_argument('-o', '--omega', type=float, default=0.0, help='Angular frequency.')
    parser.add_argument('-k', '--K', type=int, default=6, help='Number of modes.')

    args = parser.parse_args()

    CalcResolvent(args.source_dir, args.time, args.filename, args.omega, args.K)
