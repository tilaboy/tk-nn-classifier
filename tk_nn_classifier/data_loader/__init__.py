import os
from ..exceptions import FileTypeError
from .csv_loader import CSVLoader, split_csv_file
from .trxml_loader import TRXMLLoader, split_trxml_set

def _select_data_reader(field_config, data_path):
    if os.path.isdir(data_path):
        # TODO: check at least on trxml exist
        data_reader = TRXMLLoader(field_config)
    elif os.path.isfile(data_path):
        _, file_ext = os.path.splitext(data_path)
        if file_ext in ['.csv', '.tsv']:
            data_reader = CSVLoader(field_config)
        else:
            raise FileTypeError(file_ext)
    else:
        raise FileNotFoundError(f'{data_path} not found')
    return data_reader

def load_data_set(field_config, data_path, train_mode=True):
    data_reader = _select_data_reader(field_config, data_path)
    if train_mode:
        return data_reader.load_train_data(data_path)
    else:
        return data_reader.load_detail_data(data_path)
