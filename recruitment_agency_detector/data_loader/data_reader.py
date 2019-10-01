import random
import re
import os
import csv
from xml_miner.miner import TRXMLMiner
from .label_class_mapper import LabelClassMapper

class CommonDataReader:
    def __init__ (self, config):
        self.max_lines = config['max_lines']
        self.config = config

    def _prepare_input_text(self, text):
        lines = text.split("\n")
        return "\n".join(lines[:self.max_lines])

    def get_train_data(self, data_path):
        raise NotImplementedError('get_train_data needs to be implemented')

    def get_details(self, data_path):
        raise NotImplementedError('get_details needs to be implemented')


class TRXMLDataReader(CommonDataReader):

    def get_train_data(self, data_path):
        fields = self._train_fields()
        return self._get_values_from_trxml(fields, data_path)

    def get_details(self, data_path):
        fields = self._detail_fields()
        return self._get_values_from_trxml(fields, data_path)

    def _get_values_from_trxml(self, fields, data_path):
        # the first element in the fields is the input text to the data models
        trxml_miner = TRXMLMiner(','.join(fields))
        for trxml in trxml_miner.mine(data_path):
            trxml['values'][fields[0]] = self._prepare_input_text(
                    trxml['values'][fields[0]])
            yield [trxml['values'][field] for field in fields]

    def _train_fields(self):
        fields = [self.config['trxml_fields']['features'],
                  self.config['trxml_fields']['class']]
        return fields

    def _detail_fields(self):
        fields = [self.config['trxml_fields']['features'],
                  self.config['trxml_fields']['class'],
                  self.config['trxml_fields']['doc_id']]
        fields += self.config['trxml_fields']['extra']
        return fields


class CSVDataReader(CommonDataReader):
    def get_train_data(self, data_path):
        fields = self._train_fields()
        return self._get_values_from_csv(fields, data_path)

    def get_details(self, data_path):
        fields = self._detail_fields()
        return self._get_values_from_csv(fields, data_path)

    def _get_values_from_csv(self, fields, data_path):
        with open(data_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row[fields[0]] = self._prepare_input_text(row[fields[0]])
                yield [row[field] for field in fields]

    def _train_fields(self):
        fields = [self.config['csv_fields']['features'],
                  self.config['csv_fields']['class']]
        return fields

    def _detail_fields(self):
        fields = [self.config['csv_fields']['features'],
                  self.config['csv_fields']['class'],
                  self.config['csv_fields']['doc_id']]
        fields += self.config['csv_fields']['extra']
        return fields


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
            data_reader = TRXMLDataReader(self.config)
        elif os.path.isfile(data_path):
            if data_path.endswith('.csv'):
                data_reader = CSVDataReader(self.config)
            elif data_path.endswith('.tsv'):
                data_reader = CSVDataReader(self.config)
            else:
                raise ValueError(f'{data_path} is not supported type')
        else:
            raise FileNotFoundError(f'{data_path} not found')
        return data_reader

    def _get_data_set(self, data_path):
        data_reader =  self._data_reader_by_input_type(data_path)
        return data_reader.get_train_data(data_path)

    def get_data_set_with_detail(self, data_path):
        data_reader =  self._data_reader_by_input_type(data_path)
        return data_reader.get_details(data_path)

    def _build_label_mapper(self, labels):
        if self.label_mapper is None:
            self.label_mapper = LabelClassMapper.from_labels(
                    labels,
                    self.config['datasets']['label_mapper']
            )
            self.label_mapper.write()
