import numpy as np
import Ofpp


data_dir = 'Data/cavity/'

vel = Ofpp.parse_internal_field(data_dir + '0.5/U')
vel_b = Ofpp.parse_boundary_field(data_dir + '0.5/U')

mesh = Ofpp.FoamMesh(data_dir)
