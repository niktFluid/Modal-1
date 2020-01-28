from scipy import sparse

from Functions.Mesh import OfMesh
from Functions.ModalAnalysis import ResolventMode as Resolvent


def main():
    case_dir = '/mnt/data/OpenFOAM/CylinderNoise/'
    data_dir = '499.992868672869065/'

    mesh = OfMesh(case_dir, data_dir + 'C', data_dir + 'V', data_dir + 'U', data_dir + 'p')
    operator = sparse.load_npz('matL_Cylinder-0.npz')

    resolvent_mode = Resolvent(mesh, operator, omega=0.25, k=6)
    resolvent_mode.solve()
    resolvent_mode.save_data('resolvent_modes_0.pickle')
    resolvent_mode.vis_tecplot('resolvent_modes_0.dat')


if __name__ == '__main__':
    main()
