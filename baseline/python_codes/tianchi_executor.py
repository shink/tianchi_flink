import json
import os
import shutil
from typing import List

import numpy
from ai_flow import ExampleMeta, update_notification
from flink_ai_flow.pyflink.user_define_executor import TableEnvCreator, SourceExecutor, FlinkFunctionContext, \
    SinkExecutor, Executor
from pyflink.dataset import ExecutionEnvironment
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import Table, DataTypes, ScalarFunction
from pyflink.table import TableEnvironment, StreamTableEnvironment, EnvironmentSettings, BatchTableEnvironment, \
    CsvTableSink
from pyflink.table.udf import udf, FunctionContext
from zoo.serving.client import InputQueue


class StreamTableEnvCreatorBuildIndex(TableEnvCreator):

    def create_table_env(self):
        stream_env = StreamExecutionEnvironment.get_execution_environment()
        stream_env.set_parallelism(20) #100
        t_env = StreamTableEnvironment.create(
            stream_env,
            environment_settings=EnvironmentSettings.new_instance()
                .in_streaming_mode().use_blink_planner().build())
        statement_set = t_env.create_statement_set()
        t_env.get_config().set_python_executable('/opt/module/anaconda/envs/ai/bin/python')
        t_env.get_config().get_configuration().set_boolean("python.fn-execution.memory.managed", True)
        return stream_env, t_env, statement_set


class StreamTableEnvCreator(TableEnvCreator):

    def create_table_env(self):
        stream_env = StreamExecutionEnvironment.get_execution_environment()
        stream_env.set_parallelism(1)
        t_env = StreamTableEnvironment.create(
            stream_env,
            environment_settings=EnvironmentSettings.new_instance()
                .in_streaming_mode().use_blink_planner().build())
        statement_set = t_env.create_statement_set()
        t_env.get_config().set_python_executable('/opt/module/anaconda/envs/ai/bin/python')
        t_env.get_config().get_configuration().set_boolean("python.fn-execution.memory.managed", True)
        return stream_env, t_env, statement_set


class BatchTableEnvCreator(TableEnvCreator):

    def create_table_env(self):
        exec_env = ExecutionEnvironment.get_execution_environment()
        t_env = BatchTableEnvironment.create(
            environment_settings=EnvironmentSettings.new_instance().in_batch_mode().use_blink_planner().build())
        t_env._j_tenv.getPlanner().getExecEnv().setParallelism(1)
        statement_set = t_env.create_statement_set()
        t_env.get_config().set_python_executable('/opt/module/anaconda/envs/ai/bin/python')
        t_env.get_config().get_configuration().set_boolean("python.fn-execution.memory.managed", True)
        return exec_env, t_env, statement_set


class ReadTrainExample(SourceExecutor):
    def execute(self, function_context: FlinkFunctionContext) -> Table:
        table_env: TableEnvironment = function_context.get_table_env()
        path = function_context.get_example_meta().batch_uri
        ddl = """create table training_table(
                                uuid varchar,
                                face_id varchar,
                                device_id varchar,
                                feature_data varchar
                    ) with (
                        'connector.type' = 'filesystem',
                        'format.type' = 'csv',
                        'connector.path' = '{}',
                        'format.ignore-first-line' = 'false',
                        'format.field-delimiter' = ';'
                    )""".format(path)
        table_env.execute_sql(ddl)
        return table_env.from_path('training_table')


class FindHistory(Executor):
    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        t_env = function_context.get_table_env()
        table_0 = input_list[0]
        t_env.create_temporary_view('near_table', table_0)
        join_query = """select
        near_table.face_id, training_table.face_id
        from training_table
        inner join near_table
        on training_table.uuid=near_table.near_id"""
        return [t_env.sql_query(join_query)]


class ReadPredictExample(SourceExecutor):

    def execute(self, function_context: FlinkFunctionContext) -> Table:
        table_env: TableEnvironment = function_context.get_table_env()
        ddl = """create table test_table (
                face_id varchar,
                feature_data varchar
                )with (
                        'connector.type' = 'filesystem',
                        'format.type' = 'csv',
                        'connector.path' = '{}',
                        'format.field-delimiter' = ';'
                    )""".format(function_context.get_example_meta().batch_uri)
        table_env.execute_sql(ddl)
        return table_env.from_path('test_table')


class ReadOnlinePredictExample(SourceExecutor):

    def execute(self, function_context: FlinkFunctionContext) -> Table:
        table_env: TableEnvironment = function_context.get_table_env()
        table_env.execute_sql("""
            create table online_example (
                face_id varchar,
                device_id varchar,
                feature_data varchar
            ) with (
                'connector' = 'kafka',
                'topic' = 'tianchi_read_example',
                'properties.bootstrap.servers' = 'localhost:9092',
                'properties.group.id' = 'read_example',
                'format' = 'csv',
                'scan.startup.mode' = 'earliest-offset'
            )
        """)
        table = table_env.from_path('online_example')
        # Notification AIFlow to send online example messages.
        update_notification('source', function_context.node_spec.instance_id)
        return table


class TransformTrainExample(Executor):

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        input_table = input_list[0]
        return [input_table]


class PredictAutoEncoderWithTrain(Executor):

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        class Predict(ScalarFunction):

            def __init__(self):
                super().__init__()
                self._input_api = None

            def open(self, function_context: FunctionContext):
                self._input_api = InputQueue(host="localhost", port="6379", sync=True,
                                             frontend_url="http://127.0.0.1:10020")

            def eval(self, feature_data):
                try:
                    feature_samples = []
                    for feature_element in feature_data.split(' '):
                        feature_samples.append(float(feature_element))
                    request_instances = {'instances': [{'ids': numpy.array(feature_samples).tolist()}]}
                    # Cluster serving predict sync API
                    response = self._input_api.predict(json.dumps(request_instances))
                    prediction = ' '.join(response.replace('[', '').replace(']', '').split(','))
                    return prediction
                except Exception:
                    return ''

        function_context.t_env.register_function("predict1", udf(f=Predict(), input_types=[DataTypes.STRING()],
                                                                 result_type=DataTypes.STRING()))
        return [input_list[0].select('uuid, face_id, predict1(feature_data) as feature_data')]


class PredictAutoEncoder(Executor):

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        class Predict(ScalarFunction):

            def __init__(self):
                super().__init__()
                self._input_api = None

            def open(self, function_context: FunctionContext):
                self._input_api = InputQueue(host="localhost", port="6379", sync=True,
                                             frontend_url="http://127.0.0.1:10020")

            def eval(self, feature_data):
                try:
                    feature_samples = []
                    for feature_element in feature_data.split(' '):
                        feature_samples.append(float(feature_element))
                    request_instances = {'instances': [{'ids': numpy.array(feature_samples).tolist()}]}
                    # Cluster serving predict sync API
                    response = self._input_api.predict(json.dumps(request_instances))
                    prediction = ' '.join(response.replace('[', '').replace(']', '').split(','))
                    return prediction
                except Exception:
                    return ''

        function_context.t_env.register_function("predict1", udf(f=Predict(), input_types=[DataTypes.STRING()],
                                                                 result_type=DataTypes.STRING()))
        return [input_list[0].select('face_id, predict1(feature_data) as feature_data')]


class OnlinePredictAutoEncoder(Executor):

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        class Predict(ScalarFunction):

            def __init__(self):
                super().__init__()
                self._input_api = None

            def open(self, function_context: FunctionContext):
                self._input_api = InputQueue(host="localhost", port="6379", sync=True,
                                             frontend_url="http://127.0.0.1:10020")

            def eval(self, feature_data):
                try:
                    feature_samples = []
                    for feature_element in feature_data.split(' '):
                        feature_samples.append(float(feature_element))
                    request_instances = {'instances': [{'ids': numpy.array(feature_samples).tolist()}]}
                    # Cluster serving predict sync API
                    response = self._input_api.predict(json.dumps(request_instances))
                    prediction = ' '.join(response.replace('[', '').replace(']', '').split(','))
                    return prediction
                except Exception:
                    return ''

        function_context.t_env.register_function("predict2", udf(f=Predict(), input_types=[DataTypes.STRING()],
                                                                 result_type=DataTypes.STRING()))
        return [input_list[0].select('face_id, device_id, predict2(feature_data) as feature_data')]


class SearchSink(SinkExecutor):

    def execute(self, function_context: FlinkFunctionContext, input_table: Table) -> None:
        example_meta: ExampleMeta = function_context.get_example_meta()
        output_file = example_meta.batch_uri
        if os.path.exists(output_file):
            if os.path.isdir(output_file):
                shutil.rmtree(output_file)
            else:
                os.remove(output_file)
        t_env = function_context.get_table_env()
        statement_set = function_context.get_statement_set()
        sink = CsvTableSink(['a', 'b'],
                            [DataTypes.STRING(), DataTypes.STRING()],
                            output_file,
                            ';')

        t_env.register_table_sink('mySink', sink)
        statement_set.add_insert('mySink', input_table)


class WriteSecondResult(SinkExecutor):

    def execute(self, function_context: FlinkFunctionContext, input_table: Table) -> None:
        table_env: TableEnvironment = function_context.get_table_env()
        statement_set = function_context.get_statement_set()
        table_env.execute_sql("""
               create table write_example (
                    face_id varchar,
                    device_id varchar,
                    near_id int
                ) with (
                    'connector' = 'kafka',
                    'topic' = 'tianchi_write_example',
                    'properties.bootstrap.servers' = 'localhost:9092',
                    'properties.group.id' = 'write_example',
                    'format' = 'csv',
                    'scan.startup.mode' = 'earliest-offset',
                    'csv.disable-quote-character' = 'true'
                )
                """)
        statement_set.add_insert('write_example', input_table)
