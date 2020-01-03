import os
import tensorflow as tf
import numpy as np
import pickle
import functools
from tensorflow.python.keras.preprocessing import sequence

from ..data_loader import WordVector, TFDataReader, tokenize
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
        self.train()
        if 'test' in self.config['datasets']:
            self.evaluate_on_tests()

    def evaluate_on_tests(self):
        for test_set_name in self.config['datasets']['test']:

            x_test, y_test, seqlen_test = self.load_data_set(self.config['datasets']['test'][test_set_name])
            dev_loss, dev_acc = model.evaluate(x_devel, y_devel)
            LOGGER.info("Devel: loss %s\tacc %s", str(dev_loss), str(dev_acc))

            result = self.classifier.predict_on_batch(x_test)
            print(result)
            #TrainHelper.print_test_result(predicted_classes, labels)

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
        if self.embedding is None:
            self.embedding = WordVector(self.config['embedding']['file'])

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
                min(len(data_id), self.max_sequence_length)
                for data_id in data_ids
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
                                          activation='relu')(inputs_encoder)
            pool = tf.keras.layers.MaxPool1D(pool_size=2)(conv)
            inputs_encoder = tf.keras.layers.Dropout(0.5)(pool)


        flat = tf.keras.layers.Flatten()(input_encoder)
        densed = tf.keras.layers.Dense(100, activation=tf.nn.sigmoid)(flat)
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
        LOGGER.info('Reading: %s', model_cfg['datasets']['train'])
        x_train, y_train, seqlen_train = self.load_data_set(model_cfg['datasets']['train'])

        LOGGER.info('Reading: %s', model_cfg['datasets']['eval'])
        x_eval, y_eval, seqlen_eval = self.load_data_set(model_cfg['datasets']['eval'])

        callbacks_list = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
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
                  validation_data=(x_test, y_test),
                  callbacks=callbacks_list
        )

        # # TODO
        # after training, clean up

        self.classifier.save(model_cfg['model_file'])

        #test_loss, test_acc = self.classifier.evaluate()
        #LOGGER.info("Test: loss %s\tacc %s", str(test_loss), str(test_acc))


    def predict_on_text(self, text):
        return self.classifier.predict(
            input_fn=functools.partial(self._prepare_single_input, text))

    def load_saved_model(self, model_path=None):
        LOGGER.info("loading model from %s", model_path)
        self.model = tf.keras.models.load_model(model_path)


    def process_with_saved_model(self, input):
        data = self._input_text_to_pad_id(input)
        result = self.model(data)
        probabilities = result['probabilities'][0]
        return probabilities.tolist()

    # tf.keras
    def evaluation(test_file):
        """Evaluate on the data set"""

        text_lines, x_eval, y_eval = self.data_reader.read_file(test_file)
        results = self.classifier.predict(x_eval)
        for text, i in enumerate(text_lines):
            LOGGER.info("predicted={:0.2f}\tclass={}\ttext={}".format(
                results[i][0], y_eval[i], text))





    # todo:
    # the padding is probably not needed in the predicting mode
    # if needed, should use the text length as max_sequence_length
    def _input_text_to_pad_id(self, text):
        data_id = [
                self.vocab_to_ids[token]
                if token in self.vocab_to_ids else WordVector.UNK_ID
                for token in tokenize(text)
                ]
        data = sequence.pad_sequences([data_id],
                                      maxlen=self.max_sequence_length,
                                      truncating='post',
                                      padding='post',
                                      value=WordVector.PAD_ID)
        return {'input': data}
