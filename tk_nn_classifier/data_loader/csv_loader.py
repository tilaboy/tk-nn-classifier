''' CSV file reader: import data from csv files'''
from typing import Generator, Tuple
import random
import os
import csv
from .. import LOGGER
from .base_loader import BaseLoader
from .data_utils import file_ext, iter_flatten
from ..exceptions import FileTypeError

class CSVLoader(BaseLoader):
    '''
    methods:
        - load_train_data: load data with fields only needed for training
        - load_detail_data: load data with more detail fields, for evaluation
          and analysis
    '''

    @staticmethod
    def _delimit_type(data_path: str) -> str:
        file_extension = file_ext(data_path)
        csv_delimiter_mapper = { 'csv': ',', 'tsv': '\t'}
        if file_extension in csv_delimiter_mapper:
            delimiter = csv_delimiter_mapper[file_extension]
        else:
            raise FileTypeError(file_extension)
        return delimiter

    def _iter_csv(self, csv_file: str) -> Generator:
        '''row generator from csv file'''
        delimiter = self._delimit_type(csv_file)
        with open(csv_file, 'r', newline='', encoding='utf-8-sig') as csv_fh:
            reader = csv.DictReader(csv_fh, delimiter=delimiter)
            for row in reader:
                yield row

    def _load_selected_data(self, fields: str, data_path: str) -> Generator:
        for row in self._iter_csv(data_path):
            yield {field: row[field] for field in iter_flatten(fields)}

    def load_train_data(self, data_path:str) -> Generator:
        '''load data for training: features, and category'''
        fields = self._train_fields()
        return self._load_selected_data(fields, data_path)

    def load_detail_data(self, data_path:str) -> Generator:
        '''load data for eval and analysis: docid, features and category, and extra'''
        fields = self._detail_fields()
        return self._load_selected_data(fields, data_path)


def split_csv_file(data_path: str,
                   ratio: float=0.8,
                   des: str='models',
                   rand_seed: int=111) -> Tuple[str, str]:
    '''
    split the data into train and evel

    params:
        - data_path: input data path
        - ratio: a float number x in [0, 1], x will be train,
                 and 1 - x will be eval
        - des: output folder, generate des/train.csv, and des/eval.csv
        - rand_seed: random number seed

    output:
        - train_file_path
        - eval_file_path
    '''
    os.makedirs(des, exist_ok=True)
    train_file = os.path.join(des, 'train.csv')
    eval_file = os.path.join(des, 'eval.csv')
    LOGGER.info('split the data to train file %s and eval file %s',
                train_file, eval_file)
    train_fh = open(train_file, 'w', newline='', encoding='utf-8')
    eval_fh = open(eval_file, 'w', newline='', encoding='utf-8')

    random.seed(rand_seed)
    nr_train = nr_eval = 0

    with open(data_path, 'r', newline='', encoding='utf-8-sig') as orig_fh:
        delimiter = CSVLoader._delimit_type(data_path)
        orig_reader = csv.DictReader(orig_fh, delimiter=delimiter)
        fieldnames = orig_reader.fieldnames

        train_writer = csv.DictWriter(train_fh, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        eval_writer = csv.DictWriter(eval_fh, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        train_writer.writeheader()
        eval_writer.writeheader()

        for row in orig_reader:
            if random.random() <= ratio:
                train_writer.writerow(row)
                nr_train += 1
            else:
                eval_writer.writerow(row)
                nr_eval += 1

    train_fh.close()
    eval_fh.close()
    LOGGER.info('summary: %d split to %d train, and %d eval',
                nr_train + nr_eval,
                nr_train,
                nr_eval)
    return train_file, eval_file
