'''common classifer'''
import os
import csv
from .. import LOGGER
from ..data_loader import load_data_set, analysis_field_names
from .utils import eval_predictions

class BaseClassifier:
    def __init__(self, config):
        self.config = config
        self.data_reader = None
        self.data_sets = {}
        os.makedirs(self.config['model_path'], exist_ok=True)

    def _load_train_eval(self):
        train_data = self.data_reader.get_data(
            self.config['datasets']['train'],
            shuffle=False,
            train_mode=True
        )
        eval_data = self.data_reader.get_data(self.config['datasets']['eval'])
        return train_data, eval_data

    def prepare_train_eval_data(self):
        train_data, eval_data = self._load_train_eval()

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
        return predictions, labels
