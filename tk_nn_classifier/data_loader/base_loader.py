'''
Basic class to read files also get the field names relevant to the training
'''
from typing import List, Generator

class BaseLoader:
    '''
    Shared functions or properties for data loaders:
        - field names for train, and field names for detailed analysis
        - train_process:
    '''
    def __init__(self, field_config: List):
        self.field_config = field_config

    def _load_selected_data(self, fields: List, data_path: str) -> None:
        raise NotImplementedError('_load_selected_data needs to be implemented')

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

    def load_train_data(self, data_path:str) -> Generator:
        '''load data for training: features, and category'''
        fields = self._train_fields()
        return self._load_selected_data(fields, data_path)

    def load_detail_data(self, data_path:str) -> Generator:
        '''load data for eval and analysis: docid, features and category, and extra'''
        fields = self._detail_fields()
        return self._load_selected_data(fields, data_path)
