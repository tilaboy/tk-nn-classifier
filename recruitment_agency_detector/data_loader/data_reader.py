import random
import re
import os
import csv
from xml_miner.miner import TRXMLMiner
from tk_preprocessing.common_processor import char_normalization

HAS_TOKEN_REGEXP = re.compile(r'\w')
TOKEN_REGEXP = re.compile(r'\w+|[^\w\s]+')
MAX_LINES = 50

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
        return _get_trxml_details(data_path)
    elif os.path.isfile(data_path) and data_path.endswith('.csv'):
        return _get_csv_details(data_path)

def _get_data_set(data_path, config):
    if os.path.isdir(data_path):
        data_set = list(_get_data_from_trxml(data_path, config))
    elif os.path.isfile(data_path) and data_path.endswith('.csv'):
        data_set = list(_get_data_from_csv(data_path, config))
    else:
        raise FileNotFoundError(f'{data_path} not found')
    return data_set

def _get_csv_details(data_path, config):
    fields = [config['csv_fields']['doc_id']] + \
        config['csv_fields']['extra']
    with open(data_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fields_values = [
                _prepare_input_text(row[config['csv_fields']['features']]),
                _transfer_source_type(row[config['csv_fields']['class']]),
            ]
            details = [ row[field] for field in fields ]
            yield fields_values + details

def _get_trxml_details(date_path, config):
    fields = [config['trxml_fields']['features'],
              config['trxml_fields']['class'],
              config['trxml_fields']['doc_id']]
    fields += config['trxml_fields']['extra']
    return _get_values_from_trxml(fields, data_dir)


def _get_values_from_trxml(fields, data_dir):
    # the first element in the fields is the input text to the data models
    trxml_miner = TRXMLMiner(','.join(fields))
    for trxml in trxml_miner.mine(data_dir):
        trxml['values'][fields[0]] = _prepare_input_text(
                trxml['values'][fields[0]])
        yield [trxml['values'][field] for field in fields]


def _prepare_input_text(text):
    lines = text.split("\n")
    return "\n".join(lines[:MAX_LINES])


def _get_data_from_trxml(data_dir, config):
    fields = [config['trxml_fields']['features'],
              config['trxml_fields']['class']]
    return _get_values_from_trxml(fields, data_dir)


def tokenize(string):
    string = char_normalization(string)
    tokens = []
    if re.search(HAS_TOKEN_REGEXP, string):
        tokens = [
            match.group().upper()
            for match in TOKEN_REGEXP.finditer(string)
            if re.search(HAS_TOKEN_REGEXP, match.group())
        ]
    return tokens

def _get_data_from_csv(data_path, config):
    with open(data_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            yield [
                _prepare_input_text(row[config['csv_fields']['features']]),
                _transfer_source_type(row[config['csv_fields']['class']])
            ]


def _transfer_source_type(source_type):
    if source_type == '2':
        source_type = 'yes'
    elif source_type == '4':
        source_type = 'no'
    else:
        raise ValueError('Unknown source type')
    return source_type
