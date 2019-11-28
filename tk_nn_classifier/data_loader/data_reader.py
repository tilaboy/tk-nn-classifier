import os
from .. import LOGGER

from .label_class_mapper import LabelClassMapper
from .trxml_loader import TRXMLLoader
from .csv_loader import CSVLoader


class DataReader():
    def __init__ (self, config):
        self.config = config
        if 'label_mapper' not in self.config['datasets']:
            self.config['datasets']['label_mapper'] = os.path.join(
                    self.config['model_path'],
                    'label_mapper.json'
            )

        if os.path.isfile(self.config['datasets']['label_mapper']):
            self.label_mapper = \
            LabelClassMapper.from_file(
                    self.config['datasets']['label_mapper'])
        else:
            self.label_mapper = None

    def _data_reader_by_input_type(self, data_path):
        if os.path.isdir(data_path):
            data_reader = TRXMLLoader(self.config)
        elif os.path.isfile(data_path):
            if data_path.endswith('.csv'):
                data_reader = CSVLoader(self.config)
            elif data_path.endswith('.tsv'):
                data_reader = CSVLoader(self.config)
            else:
                raise ValueError(f'{data_path} is not supported type')
        else:
            raise FileNotFoundError(f'{data_path} not found')
        return data_reader

    def _detail_fields(self, data_path):
        data_reader =  self._data_reader_by_input_type(data_path)
        return data_reader._detail_fields()

    def _train_fields(self, data_path):
        data_reader =  self._data_reader_by_input_type(data_path)
        return data_reader._train_fields()

    def get_data_set(self, data_path):
        data_reader =  self._data_reader_by_input_type(data_path)
        return list(data_reader.get_train_data(data_path))

    def get_data_set_with_detail(self, data_path):
        data_reader =  self._data_reader_by_input_type(data_path)
        return list(data_reader.get_details(data_path))

    def get_split_data(self):
        data_path = self.config['datasets']['all_data']
        data_reader =  self._data_reader_by_input_type(data_path)
        return data_reader.split_data(data_path,
                                      self.config['split_ratio'],
                                      self.config['model_path']
                                     )

    def _build_label_mapper(self, labels):
        if self.label_mapper is None:
            self.label_mapper = LabelClassMapper.from_labels(
                    labels,
                    self.config['datasets']['label_mapper']
            )
            self.label_mapper.write()
