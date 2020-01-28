from scipy import sparse

from Functions.Mesh import OfMesh
from Functions.LinearStability import LSMode


def main():
    case_dir = '/mnt/data/OpenFOAM/CylinderNoise/'
    data_dir = '499.992868672869065/'

    mesh = OfMesh(case_dir, data_dir + 'C', data_dir + 'V', data_dir + 'U', data_dir + 'p')
    operator = sparse.load_npz('matL_Cylinder-0.npz')

    ls_mode = LSMode(mesh, operator, k=250, which='LR')
    ls_mode.solve()
    ls_mode.vis_tecplot('ls_mode_0.dat')


if __name__ == '__main__':
    main()
