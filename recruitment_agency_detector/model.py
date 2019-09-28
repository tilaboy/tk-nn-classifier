import sys
import tensorflow as tf
from pathlib import Path
from . import LOGGER
import logging
from .classifiers import TFClassifier, SpaceClassifier

class Model:
    def __init__(self, config):
        self.config = config
        self.type = config['model_type']
        LOGGER.info("model type: %s", self.type)

        # derived parameters
        self.config['dropout_keep_rate'] = 1 - self.config['dropout_rate']

        if self.type.startswith('tf'):
            self.classifier = TFClassifier(config)
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
            # TODO: commented out due to log comflict
            #Path('logs').mkdir(exist_ok=True)
            #handlers = [
            #    logging.FileHandler('logs/main.log'),
            #    logging.StreamHandler(sys.stdout)
            #]
            #logging.getLogger('tensorflow').handlers = handlers
        elif self.type.startswith('spacy'):
            self.classifier = SpaceClassifier(config)
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
