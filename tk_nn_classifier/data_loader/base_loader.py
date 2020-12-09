'''
Basic class to read files also get the field names relevant to the training
'''
from collections import Iterable

class BaseLoader:
    '''
    to_do:

    # load_set(path, field): support csv and folder
    #
    '''
    def __init__(self, field_config):
        self.field_config = field_config

    def _train_data(self, data_path):
        raise NotImplementedError('_train_data needs to be implemented')

    def _detail_data(self, data_path):
        raise NotImplementedError('_detail_data needs to be implemented')

    def _train_fields(self):
        fields = [self.field_config['features'],
                  self.field_config['class']]
        return fields

    def _detail_fields(self):
        fields = [self.field_config['features'],
                  self.field_config['class'],
                  self.field_config['doc_id']]
        if 'extra' in self.field_config:
            fields += self.field_config['extra']
        return fields
