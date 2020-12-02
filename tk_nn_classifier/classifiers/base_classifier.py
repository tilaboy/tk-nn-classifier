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
