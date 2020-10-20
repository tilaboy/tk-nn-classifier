import os
import re
import tensorflow as tf
import numpy as np
import pickle
import functools
from tensorflow.python.keras.preprocessing import sequence

from ..data_loader import WordVector, download_tk_embedding
from ..data_loader import TFDataReader, tokenize
from .. import LOGGER
from .utils import TrainHelper, FileHelper


class KerasClassifier:
    def __init__(self, config):
        self.config = config
        self.type = config['model_type']
        self.max_sequence_length = config['max_sequence_length']
        self.data_sets = {}
        self.embedding = None
        self.data_reader = TFDataReader(self.config)

    def build_and_train(self):
        self.load_embedding()
        self.build_graph()
        if 'all_data' in self.config['datasets']:
            if 'train' in self.config['datasets'] or \
                    'eval' in self.config['datasets']:
                raise ValueError("config conflict: all_data <=> train/eval")
            else:
                # split the data
                LOGGER.info('split all_data into train and test')
                train_source, eval_source = self.data_reader.get_split_data()
                self.config['datasets']['train'] = train_source
                self.config['datasets']['eval'] = eval_source

        self.train()
        #self.load_saved_model('best_model.26-0.45.h5')
        if 'test' in self.config['datasets']:
            self.evaluate_on_tests()

    def evaluate_on_tests(self):
        self.load_saved_model()
        for test_set_name in self.config['datasets']['test']:
            LOGGER.info('evaluate {}'.format(test_set_name))
            x_test, y_test, seqlen_test = self.load_data_set(self.config['datasets']['test'][test_set_name])
            #dev_loss, dev_acc = self.classifier.evaluate(x_test, y_test)
            #LOGGER.info("Devel: loss %s\tacc %s", str(dev_loss), str(dev_acc))
            predictions = self.classifier.predict_on_batch(x_test)
            result = [
                int(score + 0.5)
                for score in predictions.flatten()
            ]
            TrainHelper.print_test_result(result, y_test)

    def predict_batch(self, data_path):
        predicted_classes = [
                predict['classes']
                for predict in
                self.classifier.predict(
                        input_fn=functools.partial(self.input_fn, data_path)
                )
        ]
        return predicted_classes

    def load_embedding(self):
        target_file = self.config['embedding']['filepath']
        download_tk_embedding(self.config['language'], target_file)
        if self.embedding is None:
            self.embedding = WordVector(target_file)

    def _pad_vectors(self, datain, padding='post'):
        length = len(datain)
        x_shape = [length, self.max_sequence_length, self.embedding.vector_size]
        x_vector = np.zeros(x_shape, dtype=np.float32)
        for i, datain_i in enumerate(datain):
            seqlen = min(len(datain_i), self.max_sequence_length)
            if padding == 'post':
                x_vector[i][:seqlen] = datain_i[:seqlen]
            else:
                x_vector[i][-seqlen:] = datain_i[-seqlen:]
        return x_vector

    def load_data_set(self, data_path):
        if data_path not in self.data_sets:
            texts, labels = self.data_reader.get_data(data_path)

            data_vecs = [[
                    self.embedding.get_vector(token)
                    for token in tokenize(text)]
                    for text in texts]
            data_length = np.array([
                min(len(data_vec), self.max_sequence_length)
                for data_vec in data_vecs
            ])

            data = self._pad_vectors(data_vecs)
            self.data_sets[data_path] = (data, np.array(labels), data_length)
        return self.data_sets[data_path]


    def build_graph(self):
        """
        A hard coded training graph

        params:
            - input_dimension: the dimension of the input data
            - l_rate: the learning rate

        output: neural netword model
        """

        inputs = tf.keras.Input((self.max_sequence_length, self.embedding.vector_size,))
        inputs_encoder = tf.keras.layers.Dropout(0.3)(inputs)

        for i in range(self.config['cnn']['nr_layers']):
            conv = tf.keras.layers.Conv1D(self.config['cnn']['filter_size'],
                                          (self.config['cnn']['kernel_size']),
                                          padding='same',
                                          activation='relu')(inputs_encoder)
            pool = tf.keras.layers.MaxPool1D(pool_size=2)(conv)
            inputs_encoder = tf.keras.layers.Dropout(0.5)(pool)


        flat = tf.keras.layers.Flatten()(inputs_encoder)
        densed = tf.keras.layers.Dense(8, activation=tf.nn.sigmoid)(flat)
        preds = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(densed)

        model = tf.keras.models.Model(
            inputs=inputs,
            outputs=preds
        )
        print(model.summary())

        model.compile(loss=tf.keras.losses.binary_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(self.config['learning_rate']),
                      metrics=['accuracy'])

        self.classifier = model

    def train(self):
        """Training process"""
        LOGGER.info('Reading: %s', self.config['datasets']['train'])
        x_train, y_train, seqlen_train = self.load_data_set(self.config['datasets']['train'])

        LOGGER.info('Reading: %s', self.config['datasets']['eval'])
        x_eval, y_eval, seqlen_eval = self.load_data_set(self.config['datasets']['eval'])

        callbacks_list = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config['model_path'],
                                      'best_model.{epoch:02d}-{val_loss:.2f}.h5'),
                monitor='val_loss', mode='auto', verbose=1, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             mode='min',
                                             verbose=1,
                                             patience=self.config['patience_epochs'])
        ]

        self.classifier.fit(x_train,
                  y_train,
                  epochs=self.config['num_epochs'],
                  batch_size=self.config['batch_size'],
                  validation_data=(x_eval, y_eval),
                  callbacks=callbacks_list
        )

        # # TODO
        # after training, clean up

        self.classifier.save(os.path.join(self.config['model_path'], self.config['model_file']))

        #test_loss, test_acc = self.classifier.evaluate()
        #LOGGER.info("Test: loss %s\tacc %s", str(test_loss), str(test_acc))


    def predict_on_text(self, text):
        return self.classifier.predict(
            input_fn=functools.partial(self._prepare_single_input, text))


    @staticmethod
    def _get_file_with_largest_epoch(model_path):
        largest_epoch = 0
        best_model_file = ''

        for filename in os.listdir(model_path):
            if filename.startswith('best_model'):
                matched = re.match(r'best_model\.(\d+)\-', filename)
                epoch = int(matched.group(1))
                if epoch > largest_epoch:
                    largest_epoch = epoch
                    best_model_file = os.path.join(model_path, filename)
        return best_model_file

    def load_saved_model(self, model_path=None):
        if self.embedding == None:
            self.load_embedding()
        if model_path is None:
            model_path = self._get_file_with_largest_epoch(self.config['model_path'])
        LOGGER.info("loading model from %s", model_path)
        self.classifier = tf.keras.models.load_model(model_path)


    def process_with_saved_model(self, input):
        data = self._input_text_to_pad_vec(input)
        result = self.classifier.predict_on_batch(data)
        probability = result.flatten()[0]
        return [1.0 - probability, probability]


    # tf.keras
    def evaluation(self, test_file):
        """Evaluate on the data set"""

        text_lines, x_eval, y_eval = self.data_reader.read_file(test_file)
        results = self.classifier.predict(x_eval)
        for text, i in enumerate(text_lines):
            LOGGER.info("predicted={:0.2f}\tclass={}\ttext={}".format(
                results[i][0], y_eval[i], text))


    # todo:
    # the padding is probably not needed in the predicting mode
    # if needed, should use the text length as max_sequence_length
    def _input_text_to_pad_vec(self, text):

        data_vecs = [[
                self.embedding.get_vector(token)
                for token in tokenize(text)]]

        data = self._pad_vectors(data_vecs)

        return data
