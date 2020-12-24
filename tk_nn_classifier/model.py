'''
Model class:
    hub to integrate methods from both spaCy and Tensorflow frame
'''
import os
import csv
import tensorflow as tf
from shutil import copy
from . import LOGGER
from .classifiers import TFClassifier, SpacyClassifier, KerasClassifier
from .data_loader import load_data_set, split_data_set, analysis_field_names
from .classifiers.utils import eval_predictions


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

    def predict_likelihoods(self, inputs):
        return self.classifier.predict_likelihoods(inputs)

    def classify_batch(self, iputs):
        return self.classifier.classify_batch(inputs)

    def eval_test_set(self, test_file, analysis_output_file=None):
        test_data_set = list(load_data_set(self.config, test_file, train_mode=False))
        test_input = self.prepare_input(test_data_set, train_mode=False)
        features, labels = zip(*test_input)
        if isinstance(labels[0], dict):
            labels = [max(label, key=label.get) for label in labels]
        likelihoods = list(self.predict_likelihoods(features))
        predictions = [max(likelihood, key=likelihood.get) for likelihood in likelihoods]
        eval_predictions(predictions, labels)

        if analysis_output_file is not None:
            with open(analysis_output_file, 'w', newline='') as output_fh:
                field_names = analysis_field_names(self.config, test_file)
                csv_writer = csv.DictWriter(output_fh,
                                            fieldnames=field_names,
                                            extrasaction='ignore',
                                            delimiter="\t",
                                            quoting=csv.QUOTE_MINIMAL)
                csv_writer.writeheader()
                for doc, likelihood, prediction in zip(test_data_set, likelihoods, predictions):
                    doc['prediction'] = prediction
                    doc['likelihood'] = likelihood
                    csv_writer.writerow(doc)
