import tensorflow as tf
from tensorflow import keras
import random
from pathlib import Path


class TFClassifier:
    def __init__(self, config):
        self.config = config
        self.type = config['model_type']

    def build_graph(self):
        """
        A hard coded training graph

        params:
            - input_dimension: the dimension of the input data
            - l_rate: the learning rate

        output: neural netword model
        """

        model = keras.Sequential()
        model.add(keras.layers.Dense(
            128,
            input_dim=self.config['input_dimension'],
            activation=tf.nn.sigmoid))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(64, activation=tf.nn.sigmoid))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

        model.compile(loss=tf.keras.losses.binary_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(self.config["l_rate"]),
                      metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, x_test, y_test):
        """Training process"""

        self.model.fit(
            x_train, y_train,
            epochs=self.config['num_epochs'],
            batch_size=self.config['batch_size'],
            validation_data=(x_test, y_test)
        )
