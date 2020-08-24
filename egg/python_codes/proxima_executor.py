import shutil
from typing import List

import numpy as np
from flink_ai_flow.pyflink import FlinkFunctionContext
from flink_ai_flow.pyflink.user_define_executor import Executor
from pyflink.table import Table, ScalarFunction, DataTypes
from pyflink.table.descriptors import FileSystem, OldCsv, Schema
from pyflink.table.udf import TableFunction
from pyflink.table.udf import udf, udtf
from pyproxima2 import *

from data_type import DataType


class SearchUDTF1(TableFunction):
    def __init__(self, index_path: str, element_type: DataType):
        self.path = index_path
        self.topk = 20
        self.element_type = element_type
        self.ctx = None

    def open(self, function_context):
        container = IndexContainer(name='MMapFileContainer', params={})
        container.load(self.path)
        searcher = IndexSearcher("ClusteringSearcher")
        self.ctx = searcher.load(container).create_context(topk=self.topk)

    def eval(self, vec):
        if self.ctx is None:
            raise RuntimeError()
        if len(vec) != 0 and not vec.isspace():
            vec = np.array([float(v) for v in vec.split(' ')]).astype(self.element_type.to_numpy_type())
            results = self.ctx.search(query=vec)
            for i in results[0]:
                yield str(i.key())
        return None


class SearchUDTF2(ScalarFunction):
    def __init__(self, index_path: str, element_type: DataType):
        self.path = index_path
        self.topk = 20
        self.element_type = element_type
        self.ctx = None
        self.map = {0: []}
        self.may_be_person_num = 0

    def open(self, function_context):
        container = IndexContainer(name='MMapFileContainer', params={})
        container.load(self.path)
        searcher = IndexSearcher("ClusteringSearcher")
        self.ctx = searcher.load(container).create_context(topk=self.topk)

    def eval(self, vec):
        if self.ctx is None:
            raise RuntimeError()
        if len(vec) != 0 and not vec.isspace():
            vec = np.array([float(v) for v in vec.split(' ')]).astype(self.element_type.to_numpy_type())
            results = self.ctx.search(query=vec)
            near_key = results[0][0].key
            for k, v in self.map.items():
                if near_key not in v:
                    self.map[self.may_be_person_num] = []
                    self.map[self.may_be_person_num].append(near_key)
                    self.may_be_person_num += 1
                    return self.may_be_person_num - 1
                else:
                    self.map[k].append(near_key)
                    return k
        return None


class SearchExecutor(Executor):
    def __init__(self, index_path: str, element_type: DataType, dimension: int):
        super().__init__()
        self.path = index_path
        self.element_type = element_type
        self.dimension = dimension

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        t_env = function_context.get_table_env()
        table = input_list[0]
        t_env.register_function("search", udtf(SearchUDTF1(self.path, self.element_type),
                                               DataTypes.STRING(), DataTypes.STRING()))
        return [table.join_lateral("search(feature_data) as near_id").select("face_id, near_id")]


class SearchExecutor3(Executor):
    def __init__(self, index_path: str, element_type: DataType, dimension: int):
        super().__init__()
        self.path = index_path
        self.element_type = element_type
        self.dimension = dimension

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        t_env = function_context.get_table_env()
        table = input_list[0]
        t_env.register_function("search", udf(SearchUDTF2(self.path, self.element_type),
                                              DataTypes.STRING(), DataTypes.INT()))
        return [table.select("face_id, device_id, search(feature_data) as near_id")]


class BuildIndexUDF(ScalarFunction):
    def __init__(self, index_path: str, element_type: DataType, dimension: int):
        self.element_type = element_type
        self.dimension = dimension
        self.path = index_path
        self._docs = 100000
        self.holder = None
        self.builder = None

    def open(self, function_context):
        self.holder = IndexHolder(type=self.element_type.to_proxima_type(), dimension=self.dimension)
        self.builder = IndexBuilder(
            name="ClusteringBuilder",
            meta=IndexMeta(type=self.element_type.to_proxima_type(), dimension=self.dimension),
            params={'proxima.hc.builder.max_document_count': self._docs})

    def eval(self, key, vec):
        if len(vec) != 0 and not vec.isspace():
            vector = [float(v) for v in vec.split(' ')]
            self.holder.emplace(int(key), np.array(vector).astype(self.element_type.to_numpy_type()))
            return key
        return None

    def close(self):
        self.builder.train_and_build(self.holder)
        dumper = IndexDumper(path=self.path)
        self.builder.dump(dumper)
        dumper.close()


class BuildIndexExecutor(Executor):
    def __init__(self, index_path: str, element_type: DataType, dimension: int):
        self.element_type = element_type
        self.dimension = dimension
        self.path = index_path
        self._docs = 100000

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        t_env = function_context.get_table_env()
        statement_set = function_context.get_statement_set()
        table = input_list[0]
        t_env.register_function("build_index", udf(BuildIndexUDF(self.path, self.element_type, self.dimension),
                                                   [DataTypes.STRING(), DataTypes.STRING()], DataTypes.STRING()))
        dummy_output_path = '/tmp/indexed_key'
        if os.path.exists(dummy_output_path):
            if os.path.isdir(dummy_output_path):
                shutil.rmtree(dummy_output_path)
            else:
                os.remove(dummy_output_path)
        t_env.connect(FileSystem().path(dummy_output_path)) \
            .with_format(OldCsv()
                         .field('key', DataTypes.STRING())) \
            .with_schema(Schema()
                         .field('key', DataTypes.STRING())) \
            .create_temporary_table('train_sink')
        statement_set.add_insert("train_sink", table.select("build_index(uuid, feature_data)"))
        return []
