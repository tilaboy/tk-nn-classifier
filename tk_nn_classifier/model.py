'''
Model class:
    hub to integrate methods from both spaCy and Tensorflow frame
'''
import os
import tensorflow as tf
from shutil import copy
from . import LOGGER
from .config import load_config, spacy_lang_model_consistency
from .classifiers import TFClassifier, SpacyClassifier, TFMultiFeatClassifier, KerasClassifier


class Model:
    def __init__(self, config):

        self.config = config
        self.config['model_path'] = os.path.join(self.config['model_dir'],
                                                 self.config['model_version'])
        self.config['model_eval_path'] = os.path.join(self.config['model_path'],
                                                      'res')

        self.type = self.config['model_type']

        # derived parameters
        self.config['dropout_keep_rate'] = 1 - self.config['dropout_rate']

        if self.type.startswith('tf_multi_feat'):
            LOGGER.info('use tensorflow with multi feature %s' % self.type)
            self.config['classifier_frame'] = 'tensorflow_multi_feat'
            self.classifier = TFMultiFeatClassifier(self.config)
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        elif self.type.startswith('tf'):
            LOGGER.info('use tensorflow %s' % self.type)
            self.config['classifier_frame'] = 'tensorflow'
            self.classifier = TFClassifier(self.config)
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        elif self.type.startswith('keras'):
            LOGGER.info('use tensorflow %s' % self.type)
            self.config['classifier_frame'] = 'tf-keras'
            self.classifier = KerasClassifier(self.config)
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        elif self.type.startswith('spacy'):
            LOGGER.info('use spacy %s' % self.type)

            spacy_lang_model_consistency(self.config)
            self.classifier = SpacyClassifier(self.config)

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
        os.makedirs(self.config['model_path'], exist_ok=True)
        copy(self.config['config_file_path'], self.config['model_path'])
        self.classifier.build_and_train()

    def evaluate(self, test_data_path):
        self.classifier.evaluate(test_data_path)

    def load(self, model_path=None):
        self.classifier.load_saved_model(model_path)

    def process_with_saved_model(self, input):
        return self.classifier.process_with_saved_model(input)

    def predict_on_text(self, text):
        return self.classifier.predict_on_text(text)
