from Mesh import OfMesh
from FlowData import OfData
from MatMaker import MatMaker
from MatMaker import TargetEq

data_dir = 'Data/cavity/'

mesh = OfMesh(data_dir, '0.5/C', '0.5/V', '0.5/U', '0.5/p')
field = OfData(data_dir, '0.5/U', '0.5/p', '0.5/p')

target_eq = TargetEq(mesh, field)
mat_maker = MatMaker(mesh, field, target_eq)

operator_test = mat_maker.get_mat()
