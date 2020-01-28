import numpy as np
import pickle
# import scipy as sp
# from scipy.sparse import linalg

from Functions.FieldData import FieldData


class ModalData(FieldData):
    def __init__(self, mesh, operator, n_val=5, k=10, **kwargs):
        self._k = k
        self._n_values = n_val
        self.operator = self._set_operator(operator, **kwargs)
        self._vec_data = None
        super(ModalData, self).__init__(mesh, n_val=self._data_num(), data_list=self._data_name_list())

    def _init_field(self, *args, **kwargs):
        self.data = np.empty((self.n_cell, self._data_num()), dtype=np.float64)

    def _data_num(self):
        raise NotImplementedError

    def _data_name_list(self):
        raise NotImplementedError

    def solve(self, **kwargs):
        result = self._calculate(**kwargs)
        self._set_data(result)  # Set self.data and self._vec_data for the visualization.

    def save_data(self, filename='modalData.pickle'):
        with open(filename, 'wb') as file_obj:
            pickle.dump(self._vec_data, file_obj)

    def load_data(self, filename='modalData.pickle'):
        self._vec_data = pickle.load(filename)

    def _calculate(self, **kwargs):
        raise NotImplementedError

    def _set_operator(self, operator, **kwargs):
        raise NotImplementedError

    def _set_data(self, data):
        raise NotImplementedError
