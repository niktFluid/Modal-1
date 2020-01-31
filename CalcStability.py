import argparse

from Functions.Mesh import OfMesh
from Functions.ModalAnalysis import LinearStabilityMode as LSMode


def CalcStability(case_dir, time, operator_name, save_name, k, sigma=None):
    case_dir = case_dir
    data_dir = time

    mesh = OfMesh(case_dir, data_dir + 'C', data_dir + 'V', data_dir + 'U', data_dir + 'p')

    ls_mode = LSMode(mesh, operator_name, k=k)
    # ls_mode.solve()
    ls_mode.solve(sigma=sigma, which='LM')
    ls_mode.save_data(save_name + '.pickle')
    ls_mode.vis_tecplot(save_name + '.dat')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='.')

    parser.add_argument('source_dir', help='Case directory for the simulation.')
    parser.add_argument('time', help='Time step of the results.')
    parser.add_argument('-f', '--filename', default='matL.npz', help='File name for the operator.')
    parser.add_argument('-s', '--savename', default='modes', help='Name for save data.')
    parser.add_argument('-S', '--Sigma', type=complex, default=None, help='Sigma value.')
    parser.add_argument('-k', '--K', type=int, default=6, help='Number of modes.')
    args = parser.parse_args()

    CalcStability(args.source_dir, args.time, args.filename, args.savename, args.K, args.Sigma)
