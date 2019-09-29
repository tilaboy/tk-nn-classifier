import random
import re
import os
import csv
from xml_miner.miner import TRXMLMiner
from tk_preprocessing.common_processor import char_normalization


class CommonDataReader:
    def __init__ (self, config):
        self.max_lines = config['max_lines']
        self.config = config

    def _prepare_input_text(self, text):
        lines = text.split("\n")
        return "\n".join(lines[:self.max_lines])

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


class TRXMLDataReader(CommonDataReader):

    def get_data_from_trxml(self, data_dir):
        fields = self._train_fields()
        return self._get_values_from_trxml(fields, data_dir)

    def get_trxml_details(self, data_path):
        fields = self._detail_fields()
        return self._get_values_from_trxml(fields, data_path)

    def _get_values_from_trxml(self, fields, data_path):
        # the first element in the fields is the input text to the data models
        trxml_miner = TRXMLMiner(','.join(fields))
        for trxml in trxml_miner.mine(data_path):
            trxml['values'][fields[0]] = self._prepare_input_text(
                    trxml['values'][fields[0]])
            yield [trxml['values'][field] for field in fields]


class CSVDataReader(CommonDataReader):
    def get_data_from_csv(data_path, config):
        fields = self._train_fields()
        return self._get_values_from_csv(fields, data_dir)

    def get_csv_details(data_path):
        fields = self._detail_fields()
        return self._get_values_from_trxml(fields, data_path)

    def _get_values_from_csv(self, fields, data_path):
        with open(data_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row[fields[0]] = self._prepare_input_text(row[fields[0]])
                yield [row[field] for field in fields]

class DataReader(CommonDataReader):
    def __init__(self, config):
        if os.path.isdir(data_path):
            self.data_reader = TRXMLDataReader(config)
        elif os.path.isfile(data_path) and data_path.endswith('.csv'):
            self.data_reader = CSVDataReader(config)
        else:
            raise FileNotFoundError(f'{data_path} not found')

    def get_data(self, data_path):
        pass

    def get_data_with_detail(self, dagta_path):
        pass





def get_spacy_data(data_path, shuffle=False, train_mode=False):
    data_set = _get_data_set(data_path)
    if shuffle:
        random.shuffle(data_set)
    texts, labels = zip(*data_set)
    cats = [{"yes": label == "yes", "no": label == "no"} for label in labels]
    if train_mode:
        cats = [{"cats": cat} for cat in cats]
    return list(zip(texts, cats))


def get_tf_data(data_path, config):
    data_set = _get_data_set(data_path, config)
    texts, labels = zip(*data_set)
    cats = [1 if label=="yes" else 0 for label in labels]
    return list(zip(texts, cats))


def get_data_with_details(data_path, config):
    if os.path.isdir(data_path):
        return _get_trxml_details(data_path, config)
    elif os.path.isfile(data_path) and data_path.endswith('.csv'):
        return _get_csv_details(data_path, config)


def _get_data_set(data_path, config):
    if os.path.isdir(data_path):
        data_set = list(_get_data_from_trxml(data_path, config))
    elif os.path.isfile(data_path) and data_path.endswith('.csv'):
        data_set = list(_get_data_from_csv(data_path, config))
    else:
        raise FileNotFoundError(f'{data_path} not found')
    return data_set
