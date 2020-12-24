import os
import re
import tensorflow as tf
import numpy as np
import pickle
import functools
from tensorflow.python.keras.preprocessing import sequence

from ..word_embedding import WordVector, download_tk_embedding
from ..model_input import TFDataReader, tokenize
from .. import LOGGER
from .base_classifier import BaseClassifier

from tqdm import tqdm

class KerasClassifier(BaseClassifier):
    def __init__(self, config):
        super().__init__(config)
        self.max_sequence_length = self.config['max_sequence_length']
        self.embedding = None
        self.data_reader = TFDataReader(self.config)

    def prepare_train_eval_data(self):
        LOGGER.info('Reading: %s', self.config['datasets']['train'])
        x_train, y_train, seqlen_train = self.load_data_set(self.config['datasets']['train'])

        LOGGER.info('Reading: %s', self.config['datasets']['eval'])
        x_eval, y_eval, seqlen_eval = self.load_data_set(self.config['datasets']['eval'])
        return ([x_train, y_train, seqlen_train],
                [x_eval, y_eval, seqlen_eval])

    def build_and_train(self):
        self.load_embedding()
        train_data, eval_data = self.prepare_train_eval_data()
        self.build_graph()
        self.train(train_data, eval_data)
        if 'test' in self.config['datasets']:
            self.evaluate_on_tests()

    def load_embedding(self):
        target_file = self.config['embedding']['filepath']
        if not self.config['embedding']['use_local']:
            download_tk_embedding(self.config['language'], target_file)
        if self.embedding is None:
            self.embedding = WordVector(target_file)

    def load_data_set(self, data_path):
        if data_path not in self.data_sets:
            texts, labels = self.data_reader.get_data(data_path)

            data_vecs = [[
                    self.embedding.get_vector(token)
                    for token in tokenize(text)]
                    for text in tqdm(texts)]
            data_length = np.array([
                min(len(data_vec), self.max_sequence_length)
                for data_vec in data_vecs
            ])

            data = self._pad_vectors(data_vecs)
            self.data_sets[data_path] = (data, np.array(labels), data_length)
        return self.data_sets[data_path]


    def build_graph(self):
        graph_selector = GraphSelector(self.config, self.embedding)
        self.classifier = graph_selector.keras_graph()

    def train(self, train_data, eval_data):
        """Training process"""

        x_train, y_train, _ = train_data
        x_eval, y_eval, _ = eval_data

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

        # since best model is already stored, seems this is not necessary
        # self.classifier.save(os.path.join(self.config['model_path'], self.config['model_file']))

        #test_loss, test_acc = self.classifier.evaluate()
        #LOGGER.info("Test: loss %s\tacc %s", str(test_loss), str(test_acc))


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

    def evaluate_on_tests(self):
        self.load_saved_model()
        for test_set_name in self.config['datasets']['test']:
            LOGGER.info('evaluate {}'.format(test_set_name))
            x_test, y_test, seqlen_test = self.load_data_set(self.config['datasets']['test'][test_set_name])
            predictions = self.classifier.predict_on_batch(x_test)
            result = [
                int(score + 0.5)
                for score in predictions.flatten()
            ]

    def process_with_saved_model(self, input):
        data = self._input_text_to_pad_vec(input)
        result = self.classifier.predict_on_batch(data)
        probability = result.flatten()[0]
        return [1.0 - probability, probability]

    # tf.keras
    def evaluate(self, test_file):
        """Evaluate on the data set"""
        x_test, y_test, seqlen_test = self.load_data_set(test_file)
        likelihoods = self.classifier.predict_on_batch(x_test)
        return likelihoods, y_test

    def predict_on_text(self, input):
        print('TO IMPLEMENT')

    # todo:
    # the padding is probably not needed in the predicting mode
    # if needed, should use the text length as max_sequence_length
    def _input_text_to_pad_vec(self, text):

        data_vecs = [[
                self.embedding.get_vector(token)
                for token in tokenize(text)]]

        data = self._pad_vectors(data_vecs)

        return data
