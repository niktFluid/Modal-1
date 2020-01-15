from Mesh import OfMesh
from FlowData import OfData

from MatMaker import MatMaker
# from MatMaker import TargetEq

from BoundaryCondition import BoundaryCondition as BDcond
from Functions.Gradient import Gradient


data_dir = 'Data/cavity/'

mesh = OfMesh(data_dir, '0.5/C', '0.5/V', '0.5/U', '0.5/p')
field = OfData(data_dir, '0.5/U', '0.5/p', '0.5/p')

bd_cond = BDcond(mesh)
target_eq = Gradient(mesh, field, bd_cond, axis=0)
mat_maker = MatMaker(mesh.n_cell, field.n_val, target_eq)

operator_test = mat_maker.get_mat()
