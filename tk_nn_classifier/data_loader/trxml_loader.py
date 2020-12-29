''' TRXML file reader: import data from trxml files'''
from typing import Generator, Tuple
import os
import random
from shutil import copy
import glob
from xml_miner.miner import TRXMLMiner

from .. import LOGGER
from .base_loader import BaseLoader


class TRXMLLoader(BaseLoader):
    def _load_selected_data(self, fields, data_path):
        trxml_miner = TRXMLMiner(','.join(fields))
        for trxml in trxml_miner.mine(data_path):
            yield {field: trxml['values'][field] for field in fields}

    @staticmethod
    def _split_docs_on_ratio(data_path, ratio, random_shuffle=False):
        files = os.listdir(data_path)
        if not files:
            raise ValueError('no file found in %s, please check config' %
                             data_path)

        if random_shuffle is True:
            random.shuffle(files)
        split_point = int(len(files) * ratio)
        train_files = files[:split_point]
        eval_files = files[split_point:]
        LOGGER.info('split %d docs into %d train and %d eval',
                    len(files),
                    len(train_files),
                    len(eval_files)
                    )
        return train_files, eval_files

def split_trxml_set(data_path: str,
                    ratio: float=0.8,
                    des: str='models',
                    rand_seed: int=111) -> Tuple[str, str]:
    '''
    split the data into train and evel

    params:
        - data_path: input data path
        - ratio: a float number x in [0, 1], x will be train,
                 and 1 - x will be eval
        - des: output folder, generate des/train/*.trxml, and des/eval/*.trxml
        - rand_seed: random number seed

    output:
        - train_trxml_folder
        - eval_trxml_folder
    '''

    train_folder = os.path.join(des, 'train')
    eval_folder = os.path.join(des, 'eval')
    LOGGER.info('copy the data to train: %s, and eval: %s',
                train_folder, eval_folder)

    random.seed(rand_seed)
    nr_train = nr_eval = 0

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(eval_folder, exist_ok=True)

    for file in glob.iglob(f'{data_path}/*.trxml'):
        if random.random() <= ratio:
            copy(file, train_folder)
            nr_train += 1
        else:
            copy(file, eval_folder)
            nr_eval += 1

    LOGGER.info('summary: %d split to %d train, and %d eval',
                nr_train + nr_eval,
                nr_train,
                nr_eval)

    return train_folder, eval_folder
