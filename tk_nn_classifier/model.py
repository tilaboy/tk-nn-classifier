'''
Model class:
    hub to integrate methods from both spaCy and Tensorflow frame
'''
import os
import tensorflow as tf
from shutil import copy
from . import LOGGER
from .classifiers import TFClassifier, SpacyClassifier, KerasClassifier


class Model:
    def __init__(self, config):

        self.config = config

        if self.config['model_type'] .startswith('tf'):
            LOGGER.info('use tensorflow %s' % self.config['model_type'] )
            self.classifier = TFClassifier(self.config)
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        elif self.config['model_type'] .startswith('keras'):
            LOGGER.info('use tensorflow %s' % self.config['model_type'] )
            self.classifier = KerasClassifier(self.config)
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        elif self.config['model_type'] .startswith('spacy'):
            LOGGER.info('use %s' % self.config['model_type'] )
            self.classifier = SpacyClassifier(self.config)

        else:
            raise ValueError("unknown classifier type [{}]".format(self.config['model_type'] ))

    def build_graph(self):
        self.classifier.build_graph()

    def prepare_input(self, data_gen, train_mode=True):
        return self.classifier.prepare_input(data_gen, train_mode)

    def train(self, train_data=None, eval_data=None):
        self.classifier.train(train_data, eval_data)

    def save(self, output_path):
        self.classifier.save(output_path)

    def load(self, model_path=None):
        self.classifier.load_saved_model(model_path)

    def evaluate_data_set(self, test_data_path):
        self.classifier.evaluate(test_data_path)

    def predict_likelihoods(self, inputs):
        return self.classifier.predict_likelihoods(inputs)

    def build_and_train(self):
        os.makedirs(self.config['model_path'], exist_ok=True)
        copy(self.config['config_file_path'], self.config['model_path'])
        self.classifier.build_and_train()
