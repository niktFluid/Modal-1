from Functions.Mesh import OfMesh
from Functions.FieldData import OfData
from Functions.ModalAnalysis import RHS


def main(case_dir, time, mu, pr):
    mesh = OfMesh(case_dir, time + 'C', time + 'V', time + 'U', time + 'p')
    field = OfData(mesh, case_dir + time, 'U', 'p', 'rho')

    rhs = RHS(mesh, field, mu, pr, is2d=True)
    rhs.vis_tecplot('rhs_data.dat')


if __name__ == '__main__':
    main('/mnt/data/OpenFOAM/CylinderNoise/', '50000/', 1.33333e-3, 0.7)
    # main('/mnt/data/OpenFOAM/CylinderLowRe/', '10000/', 4.444444e-3, 0.7)
