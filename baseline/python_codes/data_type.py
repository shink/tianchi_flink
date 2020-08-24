from abc import abstractmethod
from enum import Enum

from pyflink.table import DataTypes
from pyproxima2 import *


class DataType(object):
    @abstractmethod
    def to_proxima_type(self):
        pass

    @abstractmethod
    def to_flink_type(self):
        pass

    @abstractmethod
    def to_numpy_type(self):
        pass


class FloatDataType(DataType):
    def to_proxima_type(self):
        return IndexMeta.FT_FP32

    def to_flink_type(self):
        return DataTypes.FLOAT()

    def to_numpy_type(self):
        return 'f'


class DoubleDataType(DataType):
    def to_proxima_type(self):
        return IndexMeta.FT_FP64

    def to_flink_type(self):
        return DataTypes.DOUBLE()

    def to_numpy_type(self):
        return 'f'

