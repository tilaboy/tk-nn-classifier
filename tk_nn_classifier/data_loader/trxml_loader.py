''' TRXML file reader: import data from trxml files'''

import random
from shutil import copyfile
import os
from xml_miner.miner import TRXMLMiner
from .. import LOGGER
from .label_class_mapper import LabelClassMapper
from .common_loader import CommonLoader


class TRXMLLoader(CommonLoader):
    def _train_fields(self):
        return super()._get_train_fields('trxml_fields')

    def _detail_fields(self):
        return super()._get_detail_fields('trxml_fields')

    def get_train_data(self, data_path):
        return self._get_values_from_trxml(self._train_fields(), data_path)

    def get_details(self, data_path):
        return self._get_values_from_trxml(self._detail_fields(), data_path)

    def _get_values_from_trxml(self, fields, data_path):
        # the first element in the fields is the input text to the data models
        trxml_miner = TRXMLMiner(','.join(list(self._iter_flatten(fields))))
        for trxml in trxml_miner.mine(data_path):
            yield [
                trxml['values'][field] if isinstance(field, str) else
                [
                    self._prepare_input_text(trxml['values'][sub_field], index==0)
                    for sub_field in field
                ]
                for index, field in enumerate(fields)
            ]

    @staticmethod
    def _split_docs_on_ratio(data_path, ratio, random_shuffle=False):
        files = os.listdir(data_path)
        if not files:
            raise ValueError('no file found in %s, please check config' % data_path)


        if random_shuffle == True:
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


    def split_data(self, data_path, ratio=0.8, des='models'):
        '''split the data into train and evel'''
        train_files, eval_files = self._split_docs_on_ratio(data_path, ratio, random_shuffle=True)

        if des:
            os.makedirs(des, exist_ok=True)
            train_folder = os.path.join(des, 'train')
            LOGGER.info('copy the train data to train folder %s' % train_folder)
            os.makedirs(train_folder, exist_ok=True)

            eval_folder = os.path.join(des, 'eval')
            LOGGER.info('copy the eval data to eval folder %s' % eval_folder)
            os.makedirs(eval_folder, exist_ok=True)
        else:
            raise ValueError('train/eval destination needs to be specified')

        for file in train_files:
            copyfile(os.path.join(data_path, file),
                     os.path.join(train_folder, file))

        for file in eval_files:
            copyfile(os.path.join(data_path, file),
                     os.path.join(eval_folder, file))
        return train_folder, eval_folder
