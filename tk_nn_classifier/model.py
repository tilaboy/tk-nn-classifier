'''Model class: hub to integrate methods from both spaCy and Tensorflow frame'''
import sys
import tensorflow as tf
from pathlib import Path
import logging
from . import LOGGER
from .config import load_config, spacy_lang_model_consistency
from .classifiers import TFClassifier, SpaceClassifier, TFMultiFeatClassifier

class Model:
    def __init__(self, config):

        if 'model_type' in config:
            self.config = config
            self.type = config['model_type']
        else:
            self.config = load_config(config['config_file_path'], poc_defaults=True)

        # derived parameters
        self.config['dropout_keep_rate'] = 1 - self.config['dropout_rate']

        if self.type.startswith('tf_multi_feat'):
            LOGGER.info('use tensorflow with multi feature support %s' % self.type)
            self.config['classifier_frame'] = 'tensorflow_multi_feat'
            self.classifier = TFMultiFeatClassifier(self.config)
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        elif self.type.startswith('tf'):
            LOGGER.info('use tensorflow %s' % self.type)
            self.config['classifier_frame'] = 'tensorflow'
            self.classifier = TFClassifier(self.config)
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        elif self.type.startswith('spacy'):
            LOGGER.info('use spacy %s' % self.type)
            self.config['classifier_frame'] = 'spacy'
            spacy_lang_model_consistency(self.config)
            print(self.config)
            self.classifier = SpaceClassifier(self.config)

        else:
            raise ValueError("unknown classifier type [{}]".format(self.type))

    def build_graph(self):
        self.classifier.build_graph()

    def train(self, train_data=None, eval_data=None):
        if train_data:
            self.classifier.train(train_data, eval_data)
        else:
            self.classifier.train()

    def save(self, output_path):
        self.classifier.save(output_path)

    def build_and_train(self):
        self.classifier.build_and_train()

    def evaluate(self, test_data_path):
        self.classifier.evaluate(test_data_path)

    def load(self, model_path=None):
        self.classifier.load_saved_model(model_path)

    def process_with_saved_model(self, input):
        return self.classifier.process_with_saved_model(input)

    def predict_on_text(self, text):
        return self.classifier.predict_on_text(text)
