import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main(source, n_mode, l, u):
    data_array = np.loadtxt(source)

    st = data_array[:, 1] * l / (2.0 * np.pi * u)
    gains = data_array[:, 3:]

    figure, axis = make_figure()

    axis.set_xlim(np.amin(st), np.amax(st))
    axis.set_ylim(1.0e4, 1.0e8)

    for i_data in range(n_mode):
        # axis.plot(st, gains[:, i_data], color='k', linewidth=0.5)
        axis.plot(st, gains[:, i_data], linewidth=1.5)

    figure.savefig(source + '.pdf', bbox_inches='tight')


def make_figure():
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern
    plt.rcParams['font.family'] = 'cmr10'  # Computer Modern
    plt.rcParams['font.size'] = 14
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    figure_1 = plt.figure(figsize=(5, 3.5))
    ax1 = figure_1.add_subplot(111)

    ax1.set_yscale('log')

    ax1.set_xlabel(r'$St = \omega D / 2 \pi U_\infty$')
    ax1.set_ylabel(r'Gain: $\sigma$')

    return figure_1, ax1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot gains.')

    parser.add_argument('source', help='Source file name.')
    parser.add_argument('-k', '--modenum', default=3, help='Number of mode', type=int)
    parser.add_argument('-L', '--length', default=1.0, help='Characteristic length', type=float)
    parser.add_argument('-U', '--velocity', default=1.0, help='Velocity.', type=float)
    args = parser.parse_args()

    main(args.source, args.modenum, args.length, args.velocity)
