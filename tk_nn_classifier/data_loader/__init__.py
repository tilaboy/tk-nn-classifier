import os
import sys
from ..exceptions import FileTypeError
from .csv_loader import CSVLoader, split_csv_file
from .trxml_loader import TRXMLLoader, split_trxml_set
from .data_utils import file_ext

def _data_type(data_path):
    data_type = None
    if os.path.isdir(data_path):
        # TODO: check at least on trxml exist
        data_type = 'TRXML'
    elif os.path.isfile(data_path):
        file_extension = file_ext(data_path)
        if file_extension in ['csv', 'tsv']:
            data_type = 'CSV'
        else:
            raise FileTypeError(file_ext)
    else:
        raise FileNotFoundError(f'{data_path} not found')
    return data_type


def _select_data_reader(field_config, data_path):
    data_type = _data_type(data_path)
    loader_class = getattr(sys.modules[__name__], data_type + 'Loader')
    return loader_class(field_config)


def load_data_set(field_config, data_path, train_mode=True):
    data_reader = _select_data_reader(field_config, data_path)
    if train_mode:
        return data_reader.load_train_data(data_path)
    else:
        return data_reader.load_detail_data(data_path)

def split_data_set(data_path, ratio, des, rand_seed):
    data_type = _data_type(data_path)
    if data_type == 'TRXML':
        train_set, eval_set = split_trxml_set(data_path, ratio, des, rand_seed)
    elif data_type == 'CSV':
        train_set, eval_set = split_csv_file(data_path, ratio, des, rand_seed)
    return train_set, eval_set
