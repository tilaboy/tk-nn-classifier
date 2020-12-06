'''common classifer'''
import os

from .. import LOGGER

class BaseClassifier:
    def __init__(self, config):
        self.config = config
        self.data_reader = None
        self.data_sets = {}
        os.makedirs(self.config['model_path'], exist_ok=True)

    def split_data(self):
        if 'train' in self.config['datasets'] or \
                'eval' in self.config['datasets']:
            raise ValueError("config conflict: all_data <=> train/eval")
        else:
            # split the data
            LOGGER.info('split all_data into train and eval')
            train_source, eval_source = self.data_reader.get_split_data()
            self.config['datasets']['train'] = train_source
            self.config['datasets']['eval'] = eval_source

    def _load_train_eval(self):
        train_data = self.data_reader.get_data(
            self.config['datasets']['train'],
            shuffle=False,
            train_mode=True
        )
        eval_data = self.data_reader.get_data(self.config['datasets']['eval'])
        return train_data, eval_data

    def prepare_train_eval_data(self):
        if 'all_data' in self.config['datasets']:
            self.split_data()
        train_data, eval_data = self._load_train_eval()
