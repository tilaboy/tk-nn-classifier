'''common classifer'''
import os

from .. import LOGGER

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
