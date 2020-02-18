import os
import errno

import argparse
import configparser

from itertools import product
import numpy as np

from Functions.Mesh import OfMesh
from Functions.FieldData import OfData
from Functions.ModalAnalysis import ResolventMode as Resolvent
from Functions.ModalAnalysis import LinearStabilityMode as LSMode
from Functions.ModalAnalysis import RandomizedResolventMode as RandomizedResolvent


def main(param_file='Parameter.dat', profile='Default'):
    if not os.path.exists(param_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), param_file)

    profiles = configparser.ConfigParser()
    profiles.read(param_file, encoding='utf-8')
    params = profiles[profile]

    print('Calculation start.\nParameters:')
    for key, value in params.items():
        print(key + ' = ' + value)

    # Set up
    mode = params['Mode']
    case_dir = params['CaseDir']
    time_dir = params['TimeDir']
    operator = params['Operator']
    save_name = params['SaveName']
    k = int(params['ModeNum'])

    if mode == 'Stability':
        which = params['Which']
        sigma = params.get('Sigma')
        if sigma is not None:
            sigma = complex(''.join(sigma.split()))

        CalcStability(case_dir, time_dir, operator, save_name, k=k, sigma=sigma, which=which)

    elif mode == 'Resolvent':
        omega = float(params['Omega'])
        alpha = float(params['Alpha'])
        r_mode = params['ResolventMode']

        CalcResolvent(case_dir, time_dir, operator, save_name, k=k, omega=omega, alpha=alpha, mode=r_mode)

    elif mode == 'RandomizedResolvent':
        r_mode = params['ResolventMode']

        s_omega = float(params['OmegaStart'])
        e_omega = float(params['OmegaEnd'])
        n_omega = int(params['OmegaNum'])
        omega = (s_omega, e_omega, n_omega)

        s_alpha = float(params['AlphaStart'])
        e_alpha = float(params['AlphaEnd'])
        n_alpha = int(params['AlphaNum'])
        alpha = (s_alpha, e_alpha, n_alpha)

        CalcRandomizedResolvent(case_dir, time_dir, operator, save_name, k=k, omega=omega, alpha=alpha, mode=r_mode)

    print('Done.')


def CalcStability(case_dir, time, operator_name, save_name, k=3, sigma=None, which='LM'):
    mesh = OfMesh(case_dir, time + 'C', time + 'V', time + 'U', time + 'p')

    ls_mode = LSMode(mesh, operator_name, k=k, sigma=sigma, which=which)
    ls_mode.solve()
    ls_mode.save_data(save_name + '.pickle')
    ls_mode.vis_tecplot(save_name + '.dat')


def CalcResolvent(case_dir, time, operator_name, save_name, k=3, omega=0.0, alpha=0.0, mode='Both'):
    mesh = OfMesh(case_dir, time + 'C', time + 'V', time + 'U', time + 'p')
    ave_field = OfData(mesh, case_dir + time, 'UMean', 'pMean', 'rhoMean')

    grid_list = [(omega, alpha)]
    resolvent_mode = Resolvent(mesh, ave_field, operator_name, k=k, omega=omega, alpha=alpha, mode=mode)
    resolvent_mode.solve(grid_list, save_name)


def CalcRandomizedResolvent(case_dir, time, operator_name, save_name, k=3, omega=None, alpha=None, mode='Both'):
    mesh = OfMesh(case_dir, time + 'C', time + 'V', time + 'U', time + 'p')
    ave_field = OfData(mesh, case_dir + time, 'UMean', 'pMean', 'rhoMean')

    if isinstance(omega, tuple):
        omega_array = np.linspace(*omega)
    else:
        omega_array = np.array([omega])

    if isinstance(alpha, tuple):
        alpha_array = np.linspace(*alpha)
    else:
        alpha_array = np.array([alpha])

    grid_list = [(o, a) for o, a in product(omega_array, alpha_array)]
    resolvent_mode = RandomizedResolvent(mesh, ave_field, operator_name, k=k, mode=mode)
    resolvent_mode.solve(grid_list, save_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modal analysis.')

    # parser.add_argument('mode', help='Calculation mode. "Stability" or "Resolvent"')
    parser.add_argument('-f', '--filename', default='Parameter.dat', help='Parameter file for the calculation.')
    parser.add_argument('-p', '--profile', default='Default', help='Profile for the parameters.')
    args = parser.parse_args()

    main(param_file=args.filename, profile=args.profile)
