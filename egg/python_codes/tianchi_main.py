import os
import sys

import ai_flow as af
from ai_flow import ExampleSupportType, ModelType, ExampleMeta, ModelMeta, PythonObjectExecutor, BaseJobConfig
from flink_ai_flow import LocalFlinkJobConfig, FlinkPythonExecutor

from data_type import FloatDataType
from proxima_executor import BuildIndexExecutor, SearchExecutor, SearchExecutor3
from python_job_executor import TrainAutoEncoder, ReadCsvExample, MergePredictResult
from tianchi_executor import ReadTrainExample, StreamTableEnvCreator, ReadPredictExample, PredictAutoEncoder, \
    SearchSink, WriteSecondResult, ReadOnlinePredictExample, FindHistory, OnlinePredictAutoEncoder, \
    StreamTableEnvCreatorBuildIndex, PredictAutoEncoderWithTrain, WritePredictResult, ReadMergeExample


def get_project_path():
    """
    Get the current project path.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect_data_file():
    """
    Collect the example data file.
    """
    # Example data sets are under the following data set path.
    data_set = '{}/data_set/'.format(os.environ['ENV_HOME'])
    # First output result file is under the following output path.
    output = '{}/codes/{}/output/'.format(os.environ['ENV_HOME'], os.environ['TASK_ID'])
    predict_result_directory = '{}/codes/{}/'.format(os.environ['ENV_HOME'], os.environ['TASK_ID']) + 'predict_result'
    merge_predict_result_path = '{}/codes/{}/'.format(os.environ['ENV_HOME'], os.environ['TASK_ID']) + 'merge_data.csv'
    train_data_file = data_set + 'train_data.csv'
    label_data_file = data_set + 'label_file.csv'
    first_test_file = data_set + 'first_test_data.csv'
    first_result_data_file = output + 'first_result.csv'
    return train_data_file, predict_result_directory, merge_predict_result_path, first_test_file, first_result_data_file


def prepare_workflow(train_data_file: str, predict_result_directory: str,
                     merge_predict_result_path: str, first_test_data_file: str,
                     first_result_data_file: str):
    """
    Prepare workflow: Example & Model Metadata registration.
    """
    train_example_meta: ExampleMeta = af.register_example(name='train_data',
                                                          support_type=ExampleSupportType.EXAMPLE_BATCH,
                                                          data_type='pandas',
                                                          data_format='csv',
                                                          batch_uri=train_data_file)
    predict_result_meta: ExampleMeta = af.register_example(name='predict_result',
                                                           support_type=ExampleSupportType.EXAMPLE_BATCH,
                                                           batch_uri=predict_result_directory)
    merge_data_meta: ExampleMeta = af.register_example(name='merge_data',
                                                       support_type=ExampleSupportType.EXAMPLE_BATCH,
                                                       batch_uri=merge_predict_result_path)
    first_test_example_meta: ExampleMeta = af.register_example(name='first_test_data',
                                                               support_type=ExampleSupportType.EXAMPLE_BATCH,
                                                               data_type='pandas',
                                                               data_format='csv',
                                                               batch_uri=first_test_data_file)
    second_test_example_data: ExampleMeta = af.register_example(name='second_test_data',
                                                                support_type=ExampleSupportType.EXAMPLE_STREAM,
                                                                data_type='kafka',
                                                                data_format='csv',
                                                                stream_uri='localhost:9092')
    first_result_example_meta: ExampleMeta = af.register_example(name='first_result_111',
                                                                 support_type=ExampleSupportType.EXAMPLE_BATCH,
                                                                 data_type='pandas',
                                                                 data_format='csv',
                                                                 batch_uri=first_result_data_file)
    second_result_example_meta: ExampleMeta = af.register_example(name='second_result_111',
                                                                  support_type=ExampleSupportType.EXAMPLE_STREAM,
                                                                  data_type='kafka',
                                                                  data_format='csv',
                                                                  stream_uri='localhost:9092')
    train_model_meta: ModelMeta = af.register_model(model_name='auto_encoder',
                                                    model_type=ModelType.SAVED_MODEL)
    return train_example_meta, predict_result_meta, merge_data_meta, \
           first_test_example_meta, second_test_example_data, \
           first_result_example_meta, second_result_example_meta, train_model_meta


def run_workflow():
    """
    Run the user-defined workflow definition.
    """
    train_data_file, predict_result_directory, merge_predict_result_path, \
    first_test_data_file, first_result_data_file = collect_data_file()
    # Prepare workflow: Example & Model Metadata registration.
    train_example_meta, predict_result_meta, merge_data_meta, first_test_example_meta, second_test_example_meta, \
    first_result_example_meta, second_result_example_meta, train_model_meta = \
        prepare_workflow(train_data_file=train_data_file,
                         predict_result_directory=predict_result_directory,
                         merge_predict_result_path=merge_predict_result_path,
                         first_test_data_file=first_test_data_file,
                         first_result_data_file=first_result_data_file)

    # Save proxima indexes under the following index path.
    index_path = '{}/codes/{}/'.format(os.environ['ENV_HOME'], os.environ['TASK_ID']) + 'test.index'

    # Set Python job config to train model.
    python_job_config_0 = BaseJobConfig(platform='local', engine='python', job_name='train')

    python_job_config_1 = BaseJobConfig(platform='local', engine='python', job_name='start_cluster_serving')

    python_job_config_2 = BaseJobConfig(platform='local', engine='python', job_name='merge_predict_result')

    # Set Flink job config to predict with cluster serving
    global_job_config_1 = LocalFlinkJobConfig()
    global_job_config_1.local_mode = 'cluster'
    global_job_config_1.flink_home = os.environ['FLINK_HOME']
    global_job_config_1.job_name = 'cluster_serving'
    global_job_config_1.set_table_env_create_func(StreamTableEnvCreatorBuildIndex())

    # Set Flink job config to build index.
    global_job_config_2 = LocalFlinkJobConfig()
    global_job_config_2.local_mode = 'cluster'
    global_job_config_2.flink_home = os.environ['FLINK_HOME']
    global_job_config_2.job_name = 'build_index'
    global_job_config_2.set_table_env_create_func(StreamTableEnvCreator())

    # Set Flink job config to fink sick.
    global_job_config_3 = LocalFlinkJobConfig()
    global_job_config_3.local_mode = 'cluster'
    global_job_config_3.flink_home = os.environ['FLINK_HOME']
    global_job_config_3.job_name = 'find_sick'
    global_job_config_3.set_table_env_create_func(StreamTableEnvCreator())

    # Set Flink job config to online cluster.
    global_job_config_4 = LocalFlinkJobConfig()
    global_job_config_4.local_mode = 'cluster'
    global_job_config_4.flink_home = os.environ['FLINK_HOME']
    global_job_config_4.job_name = 'online_cluster'
    global_job_config_4.set_table_env_create_func(StreamTableEnvCreator())

    with af.config(python_job_config_0):
        # Under first job config, we construct the first job, the job is going to train an auto_encoder model.
        python_job_0_read_train_example = af.read_example(example_info=train_example_meta,
                                                          executor=PythonObjectExecutor(python_object=ReadCsvExample()))
        python_job_0_train_model = af.train(input_data_list=[python_job_0_read_train_example],
                                            executor=PythonObjectExecutor(python_object=TrainAutoEncoder()),
                                            model_info=train_model_meta,
                                            name='trainer_0')

    with af.config(python_job_config_1):
        python_job_1_cluster_serving_channel = af.cluster_serving(model_info=train_model_meta, parallelism=16)

    with af.config(global_job_config_1):
        flink_job_0_read_train_example = af.read_example(example_info=train_example_meta,
                                                         executor=FlinkPythonExecutor(python_object=ReadTrainExample()))
        flink_job_0_predict_model = af.predict(input_data_list=[flink_job_0_read_train_example],
                                               model_info=train_model_meta,
                                               executor=FlinkPythonExecutor(
                                                   python_object=PredictAutoEncoderWithTrain()))
        flink_job_0_write_predict_data = af.write_example(input_data=flink_job_0_predict_model,
                                                          example_info=predict_result_meta,
                                                          executor=FlinkPythonExecutor(
                                                              python_object=WritePredictResult()))

    with af.config(python_job_config_2):
        python_job_2_merge_train_data_file = af.user_define_operation(executor=PythonObjectExecutor(
            python_object=MergePredictResult()))

    with af.config(global_job_config_2):
        flink_job_1_read_train_example = af.read_example(example_info=merge_data_meta,
                                                         executor=FlinkPythonExecutor(python_object=ReadMergeExample()))
        flink_job_1_build_index_channel = af.transform([flink_job_1_read_train_example],
                                                       executor=FlinkPythonExecutor(
                                                           python_object=BuildIndexExecutor(index_path, FloatDataType(),
                                                                                            128)))

    with af.config(global_job_config_3):
        flink_job_2_read_history_example = af.read_example(example_info=first_test_example_meta,
                                                           executor=FlinkPythonExecutor(
                                                               python_object=ReadPredictExample()))
        flink_job_2_predict_model = af.predict(input_data_list=[flink_job_2_read_history_example],
                                               model_info=train_model_meta,
                                               executor=FlinkPythonExecutor(python_object=PredictAutoEncoder()))
        flink_job_2_transformed_data = af.transform([flink_job_2_predict_model],
                                                    executor=FlinkPythonExecutor(
                                                        python_object=SearchExecutor(index_path, FloatDataType(), 2)))
        flink_job_2_read_train_example = af.read_example(example_info=train_example_meta,
                                                         executor=FlinkPythonExecutor(python_object=ReadTrainExample()))
        flink_job_2_join_channel = af.transform(
            input_data_list=[flink_job_2_transformed_data, flink_job_2_read_train_example],
            executor=FlinkPythonExecutor(python_object=FindHistory()))
        flink_job_2_write_result = af.write_example(input_data=flink_job_2_join_channel,
                                                    example_info=first_result_example_meta,
                                                    executor=FlinkPythonExecutor(python_object=SearchSink()))

    with af.config(global_job_config_4):
        flink_job_3_read_online_example = af.read_example(example_info=second_test_example_meta,
                                                    executor=FlinkPythonExecutor(
                                                        python_object=ReadOnlinePredictExample()))
        flink_job_3_predict_model = af.predict(input_data_list=[flink_job_3_read_online_example],
                                         model_info=train_model_meta,
                                         executor=FlinkPythonExecutor(python_object=OnlinePredictAutoEncoder()))
        flink_job_3_transformed_data = af.transform([flink_job_3_predict_model],
                                              executor=FlinkPythonExecutor(
                                                  python_object=SearchExecutor3(index_path, FloatDataType(), 2)))
        af.write_example(input_data=flink_job_3_transformed_data,
                         example_info=second_result_example_meta,
                         executor=FlinkPythonExecutor(python_object=WriteSecondResult()))

    af.stop_before_control_dependency(python_job_1_cluster_serving_channel, python_job_0_train_model)
    af.stop_before_control_dependency(flink_job_0_read_train_example, python_job_1_cluster_serving_channel)
    af.stop_before_control_dependency(python_job_2_merge_train_data_file, flink_job_0_read_train_example)
    af.stop_before_control_dependency(flink_job_1_build_index_channel, python_job_2_merge_train_data_file)
    af.stop_before_control_dependency(flink_job_2_read_history_example, flink_job_1_build_index_channel)
    af.stop_before_control_dependency(flink_job_3_read_online_example, flink_job_2_write_result)
    workflow_id = af.run(get_project_path() + '/python_codes')
    res = af.wait_workflow_execution_finished(workflow_id)
    sys.exit(res)


if __name__ == '__main__':
    af.set_project_config_file(get_project_path() + '/project.yaml')
    run_workflow()
