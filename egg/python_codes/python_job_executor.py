import os
import shutil

import ai_flow as af
import numpy as np
import pandas as pd
import tensorflow as tf
from ai_flow import FunctionContext, List, ExampleMeta, register_model_version, ModelMeta
from python_ai_flow.user_define_funcs import Executor
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class ReadCsvExample(Executor):
    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        example_meta: ExampleMeta = function_context.node_spec.example_meta
        data = pd.read_csv(example_meta.batch_uri, sep=';', header=None, usecols=[3])
        n = data.values.tolist()
        rows = len(n)
        xx = []
        for i in range(rows):
            yy = n[i][0].split(' ')
            x = []
            for y in yy:
                x.append(float(y))
            xx.append(np.array(x))
        xx = np.array(xx)
        return [xx]


class TrainAutoEncoder(Executor):
    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        x_train = input_list[0]
        input_dim = 512
        encoding_dim = 128
        x_test = np.random.rand(30, input_dim)
        model_input = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim)(model_input)
        decoder = Dense(input_dim)(encoder)
        model = Model(model_input, decoder)
        model.compile(loss='binary_crossentropy', optimizer=Adam())
        model.fit(x_train, x_train, validation_data=(x_test, x_test), epochs=1)
        encoder = Model(model_input, encoder)
        model_path = os.path.dirname(os.path.abspath(__file__)) + '/model'
        print('Save trained model to {}'.format(model_path))
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        tf.saved_model.simple_save(
            tf.keras.backend.get_session(),
            model_path,
            inputs={"aaa_input": encoder.input},
            outputs={"bbb": encoder.output}
        )
        model_meta: ModelMeta = function_context.node_spec.output_model
        # Register model version to notify that cluster serving is ready to start loading the registered model version.
        register_model_version(model=model_meta, model_path=model_path)
        return []


class MergePredictResult(Executor):
    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        num_of_files = 100
        path = af.get_example_by_name('predict_result').batch_uri
        filenames = []
        for i in range(1, num_of_files + 1):
            filenames.append(str(path + '/' + str(i)))

        outfile_path = af.get_example_by_name('merge_data').batch_uri
        with open(outfile_path, 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
        return []
